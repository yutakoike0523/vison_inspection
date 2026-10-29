"""Replace masked tag regions with a translation-matched normal-surface patch.

The source patch is selected by SSD on a ring *outside* the tag.  Only pixels
inside the supplied mask are written, so the original surface outside the tag
is never altered.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def sample_points(y: np.ndarray, x: np.ndarray, maximum: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly subsample points so large masks remain practical."""
    if len(y) <= maximum:
        return y, x
    selected = rng.choice(len(y), size=maximum, replace=False)
    return y[selected], x[selected]


def outside_ring(component: np.ndarray, forbidden: np.ndarray, width: int) -> np.ndarray:
    """A comparison ring outside the target. Other masked areas are excluded."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    ring = cv2.dilate(component.astype(np.uint8), kernel).astype(bool) & ~component
    return ring & ~forbidden


def valid_translation(
    dy: int, dx: int, source_y: np.ndarray, source_x: np.ndarray, shape: tuple[int, int], forbidden: np.ndarray
) -> bool:
    h, w = shape
    moved_y, moved_x = source_y + dy, source_x + dx
    if moved_y.min() < 0 or moved_y.max() >= h or moved_x.min() < 0 or moved_x.max() >= w:
        return False
    return not bool(forbidden[moved_y, moved_x].any())


def ssd_for_translation(
    image: np.ndarray, ring_y: np.ndarray, ring_x: np.ndarray, dy: int, dx: int
) -> float:
    difference = image[ring_y, ring_x].astype(np.float32) - image[ring_y + dy, ring_x + dx].astype(np.float32)
    return float(np.mean(difference * difference))


def find_best_translation(
    image: np.ndarray,
    component: np.ndarray,
    all_mask: np.ndarray,
    max_shift: int,
    coarse_step: int,
    ring_width: int,
    rng: np.random.Generator,
) -> tuple[int, int, float]:
    """Search coarse offsets, then refine the best candidates at 1-pixel precision."""
    target_y, target_x = np.nonzero(component)
    ring = outside_ring(component, all_mask, ring_width)
    ring_y, ring_x = np.nonzero(ring)
    if not len(ring_y):
        raise ValueError("No unmasked pixels exist around this tag to score patch candidates.")

    # Samples are used for the broad search. Full target coverage is checked for finalists.
    source_y, source_x = sample_points(target_y, target_x, 8_000, rng)
    score_y, score_x = sample_points(ring_y, ring_x, 8_000, rng)
    coarse: list[tuple[float, int, int]] = []
    for dy in range(-max_shift, max_shift + 1, coarse_step):
        for dx in range(-max_shift, max_shift + 1, coarse_step):
            if dx == 0 and dy == 0:
                continue
            if not valid_translation(dy, dx, source_y, source_x, image.shape, all_mask):
                continue
            if not valid_translation(dy, dx, score_y, score_x, image.shape, all_mask):
                continue
            coarse.append((ssd_for_translation(image, score_y, score_x, dy, dx), dy, dx))
    if not coarse:
        raise RuntimeError("No source patch was found. Increase --max-shift or provide a smaller mask.")

    # Refine the 12 best coarse candidates. Full-mask validation avoids copying any tag pixel.
    finalists: list[tuple[float, int, int]] = []
    for _, coarse_dy, coarse_dx in sorted(coarse)[:12]:
        for dy in range(coarse_dy - coarse_step, coarse_dy + coarse_step + 1):
            for dx in range(coarse_dx - coarse_step, coarse_dx + coarse_step + 1):
                if dx == 0 and dy == 0:
                    continue
                if not valid_translation(dy, dx, target_y, target_x, image.shape, all_mask):
                    continue
                if not valid_translation(dy, dx, ring_y, ring_x, image.shape, all_mask):
                    continue
                finalists.append((ssd_for_translation(image, ring_y, ring_x, dy, dx), dy, dx))
    if not finalists:
        raise RuntimeError("No fully unmasked source patch was found after refinement.")
    score, dy, dx = min(finalists)
    return dy, dx, score


def replace_component(image: np.ndarray, result: np.ndarray, component: np.ndarray, dy: int, dx: int, blend_width: int) -> None:
    """Copy the translated patch inside only; a narrow inner blend suppresses seams."""
    y, x = np.nonzero(component)
    source = image[y + dy, x + dx].astype(np.float32)
    if blend_width <= 0:
        result[y, x] = source.astype(result.dtype)
        return
    distance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 3)[y, x]
    alpha = np.clip(distance / blend_width, 0.0, 1.0)
    blended = alpha * source + (1.0 - alpha) * image[y, x].astype(np.float32)
    result[y, x] = np.rint(blended).astype(result.dtype)


def repair(image: np.ndarray, mask: np.ndarray, max_shift: int, coarse_step: int, ring_width: int, blend_width: int) -> tuple[np.ndarray, list[dict]]:
    if image.ndim != 2:
        raise ValueError("Use a grayscale image. Convert colour images before calling repair().")
    all_mask = mask.astype(bool)
    count, labels = cv2.connectedComponents(all_mask.astype(np.uint8), connectivity=8)
    result = image.copy()
    records: list[dict] = []
    rng = np.random.default_rng(20261029)
    for component_id in range(1, count):
        component = labels == component_id
        dy, dx, score = find_best_translation(image, component, all_mask, max_shift, coarse_step, ring_width, rng)
        replace_component(image, result, component, dy, dx, blend_width)
        y, x = np.nonzero(component)
        records.append({"component": component_id, "pixels": int(len(y)), "source_shift_dx": dx, "source_shift_dy": dy,
                        "boundary_ssd": score, "target_bbox_xyxy": [int(x.min()), int(y.min()), int(x.max()), int(y.max())]})
    return result, records


def save_debug(image: np.ndarray, mask: np.ndarray, repaired: np.ndarray, records: list[dict], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / "tag_mask.png"), (mask.astype(np.uint8) * 255))
    cv2.imwrite(str(directory / "repaired.png"), repaired)
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for item in records:
        x0, y0, x1, y1 = item["target_bbox_xyxy"]
        dx, dy = item["source_shift_dx"], item["source_shift_dy"]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.rectangle(overlay, (x0 + dx, y0 + dy), (x1 + dx, y1 + dy), (0, 255, 255), 2)
        cv2.arrowedLine(overlay, ((x0 + x1)//2, (y0 + y1)//2), ((x0 + x1)//2 + dx, (y0 + y1)//2 + dy), (0, 255, 255), 2)
    cv2.imwrite(str(directory / "source_patch_selection.png"), overlay)
    (directory / "selection.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace a tag mask with a translation-matched normal-surface patch.")
    parser.add_argument("image", type=Path, help="Input grayscale image")
    parser.add_argument("mask", type=Path, help="Binary tag mask: non-zero pixels are replaced")
    parser.add_argument("output", type=Path, help="Output repaired image")
    parser.add_argument("--debug-dir", type=Path, default=Path("tag_patch_debug"))
    parser.add_argument("--max-shift", type=int, default=768, help="Maximum translation search in pixels")
    parser.add_argument("--coarse-step", type=int, default=8)
    parser.add_argument("--ring-width", type=int, default=12)
    parser.add_argument("--blend-width", type=int, default=3, help="Blend only this many pixels inside the tag edge; 0 disables blending")
    args = parser.parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    raw_mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if image is None or raw_mask is None:
        raise FileNotFoundError("Could not read image or mask.")
    if image.shape != raw_mask.shape:
        raise ValueError(f"Image shape {image.shape} and mask shape {raw_mask.shape} differ.")
    repaired, records = repair(image, raw_mask > 0, args.max_shift, args.coarse_step, args.ring_width, args.blend_width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), repaired)
    save_debug(image, raw_mask > 0, repaired, records, args.debug_dir)
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
