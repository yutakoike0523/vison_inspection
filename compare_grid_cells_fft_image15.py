"""Rectify full indexed grid cells and compare candidate/control FFTs for image15.

The cell geometry and checker phase are reused from the existing
grid_directional_autocorr.py report. The visible ROI is a candidate annotation,
not a ground-truth defect label.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = ROOT / "data20260903" / "data20260903" / "triming" / "image15_triming.png"
OUTPUT_DIR = ROOT / "outputs" / "⑭_格子セル単位FFT比較_image15"
ROI_XYXY = (705.0, 894.0, 775.0, 983.0)
DEFAULT_RECTIFIED_SIZE = 128
DEFAULT_CONTROL_COUNT = 8
LOW_BAND = (0.0, 0.025)
MID_BAND = (0.025, 0.08)
HIGH_BAND = (0.08, 0.5)


def find_indexed_report() -> Path:
    matches = []
    for image in (ROOT / "outputs").rglob("09_indexed_cells.png"):
        report = image.parent / "report.json"
        if not report.exists():
            continue
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("image_name") == "image15_triming.png":
            matches.append(report)
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one indexed-cell report for image15, found {len(matches)}: {matches}")
    return matches[0]


def poly_area(poly: np.ndarray) -> float:
    return abs(float(cv2.contourArea(np.asarray(poly, dtype=np.float32))))


def intersection_area(poly: np.ndarray, roi: np.ndarray) -> float:
    area, _ = cv2.intersectConvexConvex(np.asarray(poly, dtype=np.float32),
                                        np.asarray(roi, dtype=np.float32))
    return max(0.0, float(area))


def corners_for_cell(i: int, j: int, origin: np.ndarray, a: np.ndarray,
                     b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = origin + i * a + j * b
    local = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]], dtype=np.float64)
    corners = center + local[:, :1] * a + local[:, 1:] * b
    return center, corners


def enumerate_cells(width: int, height: int, origin: np.ndarray,
                    a: np.ndarray, b: np.ndarray, roi: np.ndarray) -> list[dict]:
    matrix = np.column_stack((a, b))
    image_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=float)
    ij = (image_corners - origin) @ np.linalg.inv(matrix).T
    low = np.floor(ij.min(axis=0)).astype(int) - 2
    high = np.ceil(ij.max(axis=0)).astype(int) + 2
    rows = []
    cell_area_expected = abs(float(np.linalg.det(matrix)))
    for j in range(low[1], high[1] + 1):
        for i in range(low[0], high[0] + 1):
            center, polygon = corners_for_cell(i, j, origin, a, b)
            cell_area = poly_area(polygon)
            if cell_area <= 1e-6:
                continue
            inter = intersection_area(polygon, roi)
            if inter <= 1e-4 and not np.all((polygon[:, 0] >= 0) & (polygon[:, 0] < width) &
                                             (polygon[:, 1] >= 0) & (polygon[:, 1] < height)):
                continue
            roi_area = poly_area(roi)
            rows.append({
                "i": int(i), "j": int(j), "checker_parity": int((i + j) % 2),
                "center_xy": center.tolist(), "vertices_xy": polygon.tolist(),
                "cell_area_px2": cell_area, "intersection_area_px2": inter,
                "roi_overlap_fraction": inter / max(roi_area, 1e-12),
                "cell_overlap_fraction": inter / max(cell_area, 1e-12),
                "roi_iou": inter / max(cell_area + roi_area - inter, 1e-12),
                "expected_parallelogram_area_px2": cell_area_expected,
            })
    rows.sort(key=lambda r: (r["j"], r["i"]))
    for index, row in enumerate(rows, start=1):
        row["cell_id"] = f"C{index:03d}"
    return rows


def rectified_cell(gray: np.ndarray, vertices: np.ndarray, size: int) -> np.ndarray:
    destination = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(np.asarray(vertices, dtype=np.float32), destination)
    return cv2.warpPerspective(gray, matrix, (size, size), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)


def fft_cell(cell: np.ndarray) -> dict:
    image = cell.astype(np.float64)
    centered = image - image.mean()
    hann = np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))
    coeff = np.fft.fftshift(np.fft.fft2(centered * hann))
    power = np.abs(coeff) ** 2
    h, w = image.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    fxm, fym = np.meshgrid(fx, fy)
    radius = np.hypot(fxm, fym)
    df = max(1 / h, 1 / w)
    indices = np.rint(radius.ravel() / df).astype(int)
    radial = np.bincount(indices, weights=power.ravel())
    freq = np.arange(len(radial), dtype=float) * df
    power_no_dc = power.copy()
    power_no_dc[h // 2, w // 2] = 0.0
    radial_no_dc = np.bincount(indices, weights=power_no_dc.ravel(), minlength=len(radial))
    total = float(power.sum())
    non_dc = float(power_no_dc.sum())
    curve = radial_no_dc / max(non_dc, 1e-12)
    valid = (freq > 0) & (freq <= .5)
    peak_index = int(np.argmax(curve[valid]))
    valid_freq, valid_curve = freq[valid], curve[valid]

    def band_fraction(low: float, high: float) -> float:
        selected = (freq >= low) & (freq < high)
        return float(curve[selected].sum())

    # Keep the 2D comparison DC-free and normalized so it describes spectral shape.
    normalized_2d = power_no_dc / max(non_dc, 1e-12)
    return {
        "fft_power": power, "normalized_2d_power": normalized_2d,
        "radial_frequency": freq, "radial_power_fraction": curve,
        "mean_intensity": float(image.mean()), "std_intensity": float(image.std()),
        "total_fft_power": total, "non_dc_fft_power": non_dc,
        "low_frequency_fraction": band_fraction(*LOW_BAND),
        "middle_frequency_fraction": band_fraction(*MID_BAND),
        "high_frequency_fraction": band_fraction(*HIGH_BAND),
        "dominant_radial_frequency_cycles_per_pixel": float(valid_freq[peak_index]),
        "dominant_radial_power_fraction": float(valid_curve[peak_index]),
    }


def draw_cell_positions(gray: np.ndarray, cells: list[dict], candidate: dict,
                        controls: list[dict], roi_xyxy: tuple[float, ...], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 10), constrained_layout=True)
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    control_keys = {(c["i"], c["j"]) for c in controls}
    cand_key = (candidate["i"], candidate["j"])
    for cell in cells:
        polygon = np.asarray(cell["vertices_xy"])
        color, alpha, lw = "white", .34, .45
        if (cell["i"], cell["j"]) == cand_key:
            color, alpha, lw = "red", 1.0, 2.8
        elif (cell["i"], cell["j"]) in control_keys:
            color, alpha, lw = "deepskyblue", 1.0, 1.8
        ax.add_patch(MplPolygon(polygon, closed=True, fill=False, edgecolor=color,
                                linewidth=lw, alpha=alpha))
        center = cell["center_xy"]
        if -20 <= center[0] <= gray.shape[1] + 20 and -20 <= center[1] <= gray.shape[0] + 20:
            if (cell["i"], cell["j"]) == cand_key or (cell["i"], cell["j"]) in control_keys:
                label_color = "red" if (cell["i"], cell["j"]) == cand_key else "deepskyblue"
                ax.text(center[0], center[1], f"{cell['cell_id']} ({cell['i']},{cell['j']})",
                        color="white", fontsize=7, ha="center", va="center",
                        bbox={"facecolor": label_color, "alpha": .8, "pad": 1.5})
            else:
                ax.text(center[0], center[1], f"{cell['i']},{cell['j']}", color="white",
                        fontsize=5, ha="center", va="center",
                        bbox={"facecolor": "black", "alpha": .36, "pad": .5})
    x1, y1, x2, y2 = roi_xyxy
    ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="#ffd400",
                           linewidth=2.5, label="Candidate ROI"))
    ax.plot([], [], color="red", lw=2.5, label=f"Candidate {candidate['cell_id']} ({candidate['i']},{candidate['j']})")
    ax.plot([], [], color="deepskyblue", lw=2, label="Same-parity controls")
    ax.set(xlim=(0, gray.shape[1]), ylim=(gray.shape[0], 0), xlabel="x [px]", ylabel="y [px]",
           title="image15: indexed cell boundaries, ROI, candidate, and controls")
    ax.legend(loc="upper right")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def draw_overlap_ranking(gray: np.ndarray, top: list[dict], roi_xyxy: tuple[float, ...], path: Path) -> None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1.4, 1]})
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    x1, y1, x2, y2 = roi_xyxy
    ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="yellow", linewidth=2.5))
    for k, cell in enumerate(top):
        color = "red" if k == 0 else "orange"
        poly = np.asarray(cell["vertices_xy"])
        ax.add_patch(MplPolygon(poly, closed=True, fill=False, edgecolor=color, linewidth=2.4-k*.2))
        ax.text(*cell["center_xy"], f"{k+1}: ({cell['i']},{cell['j']}) {cell['roi_overlap_fraction']:.1%}",
                color="white", fontsize=8, ha="center", bbox={"facecolor": color, "alpha": .75, "pad": 1.5})
    ax.set(xlim=(x1-260, x2+260), ylim=(y2+260, y1-260), title="Top overlapping cells around candidate ROI",
           xlabel="x [px]", ylabel="y [px]")
    labels = [f"({c['i']},{c['j']}) parity {c['checker_parity']}" for c in top[::-1]]
    vals = [c["roi_overlap_fraction"] * 100 for c in top[::-1]]
    bx.barh(labels, vals, color=["#e64b35" if i == len(vals)-1 else "#f0a050" for i in range(len(vals))])
    bx.set(xlabel="ROI area inside cell [%]", title="Intersection ranking")
    bx.grid(axis="x", alpha=.25)
    for y, value in enumerate(vals):
        bx.text(value + .5, y, f"{value:.1f}%", va="center", fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def masked_original_cell(gray: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    pts = np.asarray(polygon, dtype=np.float32)
    x0 = max(0, int(math.floor(pts[:, 0].min())))
    y0 = max(0, int(math.floor(pts[:, 1].min())))
    x1 = min(gray.shape[1], int(math.ceil(pts[:, 0].max())))
    y1 = min(gray.shape[0], int(math.ceil(pts[:, 1].max())))
    crop = gray[y0:y1, x0:x1].copy()
    local = np.round(pts - np.array([x0, y0], dtype=np.float32)).astype(np.int32)
    mask = np.zeros(crop.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, local, 255)
    return cv2.bitwise_and(crop, crop, mask=mask)


def montage(images: list[np.ndarray], titles: list[str], path: Path, title: str,
            cmap: str = "gray", limits: tuple[float, float] | None = None,
            overlay_text: list[str] | None = None) -> None:
    cols = 3
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5), squeeze=False, constrained_layout=True)
    for index, ax in enumerate(axes.ravel()):
        if index >= len(images):
            ax.axis("off")
            continue
        kwargs = {"cmap": cmap, "interpolation": "nearest"}
        if limits:
            kwargs.update(vmin=limits[0], vmax=limits[1])
        ax.imshow(images[index], **kwargs)
        ax.set_title(titles[index], fontsize=9)
        if overlay_text and index < len(overlay_text):
            ax.text(.02, .02, overlay_text[index], transform=ax.transAxes, color="yellow",
                    fontsize=8, bbox={"facecolor": "black", "alpha": .65, "pad": 2})
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def robust_z(value: float, controls: list[float]) -> float | None:
    median = float(np.median(controls))
    mad = float(np.median(np.abs(np.asarray(controls) - median)))
    return float((value - median) / (1.4826 * mad)) if mad > 1e-12 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--rectified-size", type=int, default=DEFAULT_RECTIFIED_SIZE)
    parser.add_argument("--control-count", type=int, default=DEFAULT_CONTROL_COUNT)
    args = parser.parse_args()
    if args.rectified_size < 32:
        parser.error("--rectified-size must be at least 32")
    image_path = args.image.resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    indexed_report_path = find_indexed_report()
    indexed = json.loads(indexed_report_path.read_text(encoding="utf-8"))
    if indexed.get("status") == "needs_review":
        print("WARNING: indexed grid report is marked needs_review; cell and overlap results are provisional.")
    a = np.asarray(indexed["new_basis_a"], dtype=np.float64)
    b = np.asarray(indexed["new_basis_b"], dtype=np.float64)
    phase = indexed["checker_phase_fit"]
    origin = np.asarray(phase["origin_xy"], dtype=np.float64)
    selected_parity = int(phase["checker_parity"])
    gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
    h, w = gray.shape
    roi = np.array([[ROI_XYXY[0], ROI_XYXY[1]], [ROI_XYXY[2], ROI_XYXY[1]],
                    [ROI_XYXY[2], ROI_XYXY[3]], [ROI_XYXY[0], ROI_XYXY[3]]], dtype=np.float32)
    cells = enumerate_cells(w, h, origin, a, b, roi)
    ranked = sorted((c for c in cells if c["intersection_area_px2"] > 1e-3),
                    key=lambda c: (-c["intersection_area_px2"], c["j"], c["i"]))
    if not ranked:
        raise RuntimeError("No indexed grid cell intersects the candidate ROI")
    candidate = ranked[0]
    overlap_context = ranked[1] if len(ranked) > 1 else None
    candidate_key = (candidate["i"], candidate["j"])
    eligible = []
    for cell in cells:
        polygon = np.asarray(cell["vertices_xy"])
        if (cell["i"], cell["j"]) == candidate_key or cell["checker_parity"] != candidate["checker_parity"]:
            continue
        if cell["intersection_area_px2"] > 1e-3:
            continue
        # Require the full cell inside the image to avoid reflected-border pixels.
        if not np.all((polygon[:, 0] >= 1) & (polygon[:, 0] < w-1) &
                      (polygon[:, 1] >= 1) & (polygon[:, 1] < h-1)):
            continue
        cell["distance_to_candidate_center_px"] = float(np.linalg.norm(
            np.asarray(cell["center_xy"]) - np.asarray(candidate["center_xy"])))
        eligible.append(cell)
    controls = sorted(eligible, key=lambda c: (c["distance_to_candidate_center_px"], c["j"], c["i"]))[:args.control_count]
    if len(controls) < 3:
        raise RuntimeError(f"Only {len(controls)} same-parity controls meet selection criteria")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = [candidate] + controls
    roles = ["candidate"] + ["control"] * len(controls)
    rectified, original_cells, features = [], [], []
    for cell, role in zip(selected, roles):
        polygon = np.asarray(cell["vertices_xy"], dtype=np.float32)
        original_cells.append(masked_original_cell(gray, polygon))
        rectified_cell_image = rectified_cell(gray, polygon, args.rectified_size)
        rectified.append(rectified_cell_image)
        fft = fft_cell(rectified_cell_image)
        rec = {
            "role": role,
            "cell_id": cell["cell_id"], "i": cell["i"], "j": cell["j"],
            "checker_parity": cell["checker_parity"],
            "vertices_image_xy": json.dumps(cell["vertices_xy"], separators=(",", ":")),
            "center_x": cell["center_xy"][0], "center_y": cell["center_xy"][1],
            "cell_area_px2": cell["cell_area_px2"], "roi_intersection_area_px2": cell["intersection_area_px2"],
            "roi_overlap_fraction": cell["roi_overlap_fraction"], "cell_overlap_fraction": cell["cell_overlap_fraction"],
            "roi_iou": cell["roi_iou"], "rectified_size_px": args.rectified_size,
            **{k: v for k, v in fft.items() if k not in {"fft_power", "normalized_2d_power", "radial_frequency", "radial_power_fraction"}},
        }
        features.append(rec)
        cell["normalized_cell"] = rectified_cell_image
        cell["fft"] = fft

    ctrl_cells = selected[1:]
    radial_curves = np.stack([c["fft"]["radial_power_fraction"] for c in ctrl_cells])
    median_radial = np.median(radial_curves, axis=0)
    normalized_maps = np.stack([c["fft"]["normalized_2d_power"] for c in ctrl_cells])
    median_map = np.median(normalized_maps, axis=0)
    candidate_radial = candidate["fft"]["radial_power_fraction"]
    candidate_map = candidate["fft"]["normalized_2d_power"]
    candidate_radial_l1 = float(np.abs(candidate_radial - median_radial).sum())
    candidate_2d_l1 = float(np.abs(candidate_map - median_map).sum())
    candidate_2d_l2 = float(np.sqrt(np.square(candidate_map - median_map).sum()))
    loo_radial, loo_2d_l1, loo_2d_l2 = [], [], []
    for index in range(len(ctrl_cells)):
        remain_radial = np.delete(radial_curves, index, axis=0)
        remain_maps = np.delete(normalized_maps, index, axis=0)
        radial_distance = float(np.abs(radial_curves[index] - np.median(remain_radial, axis=0)).sum())
        map_delta = normalized_maps[index] - np.median(remain_maps, axis=0)
        loo_radial.append(radial_distance)
        loo_2d_l1.append(float(np.abs(map_delta).sum()))
        loo_2d_l2.append(float(np.sqrt(np.square(map_delta).sum())))
    for index, rec in enumerate(features):
        if index == 0:
            rec.update({"radial_spectrum_L1_distance_to_control_median": candidate_radial_l1,
                        "normalized_2d_spectrum_L1_distance_to_control_median": candidate_2d_l1,
                        "normalized_2d_spectrum_L2_distance_to_control_median": candidate_2d_l2})
        else:
            cidx = index - 1
            rec.update({"radial_spectrum_L1_distance_to_control_median": loo_radial[cidx],
                        "normalized_2d_spectrum_L1_distance_to_control_median": loo_2d_l1[cidx],
                        "normalized_2d_spectrum_L2_distance_to_control_median": loo_2d_l2[cidx]})

    draw_cell_positions(gray, cells, candidate, controls, ROI_XYXY, args.output_dir / "01_cell_positions.png")
    draw_overlap_ranking(gray, ranked[:5], ROI_XYXY, args.output_dir / "02_candidate_overlap_ranking.png")
    visual_cells = [candidate] + ([overlap_context] if overlap_context else []) + controls
    visual_labels = [f"CANDIDATE {candidate['cell_id']} ({candidate['i']},{candidate['j']})\nROI={candidate['roi_overlap_fraction']:.1%}"]
    if overlap_context:
        visual_labels.append(f"ROI-OVERLAP CONTEXT {overlap_context['cell_id']} ({overlap_context['i']},{overlap_context['j']})\n"
                             f"parity {overlap_context['checker_parity']} (excluded from FFT controls), ROI={overlap_context['roi_overlap_fraction']:.1%}")
    visual_labels += [f"CONTROL {c['cell_id']} ({c['i']},{c['j']})\nROI={c['roi_overlap_fraction']:.1%}" for c in controls]
    visual_original = [masked_original_cell(gray, np.asarray(c["vertices_xy"])) for c in visual_cells]
    visual_rectified = [rectified_cell(gray, np.asarray(c["vertices_xy"], dtype=np.float32), args.rectified_size)
                        for c in visual_cells]
    montage(visual_original, visual_labels, args.output_dir / "03_original_cell_montage.png",
            "Original oblique cell regions (outside polygon masked)")
    montage(visual_rectified, visual_labels, args.output_dir / "04_rectified_cell_montage.png",
            f"Perspective-rectified cells ({args.rectified_size}x{args.rectified_size})")

    labels = [f"{role.upper()} {c['cell_id']} ({c['i']},{c['j']})\nROI={c['roi_overlap_fraction']:.1%}"
              for role, c in zip(roles, selected)]
    fft_logs = [np.log1p(c["fft"]["fft_power"]) for c in selected]
    all_non_dc = []
    for img in fft_logs:
        tmp = img.copy(); cy, cx = np.array(tmp.shape) // 2
        tmp[cy-1:cy+2, cx-1:cx+2] = np.nan
        all_non_dc.extend(tmp[np.isfinite(tmp)].tolist())
    vmax = float(np.percentile(all_non_dc, 99)) if all_non_dc else 1.0
    montage(fft_logs, labels, args.output_dir / "05_fft_2d_montage.png",
            "2D FFT log(1 + power), shared color scale; DC center remains visible", cmap="magma", limits=(0, vmax))

    freq = candidate["fft"]["radial_frequency"]
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for index, cell in enumerate(ctrl_cells):
        ax.plot(freq, cell["fft"]["radial_power_fraction"], color="#72b7c8", alpha=.58,
                lw=1.1, label="Control cells" if index == 0 else None)
    ax.plot(freq, median_radial, color="black", ls="--", lw=2.2, label="Control median")
    ax.plot(freq, candidate_radial, color="#d62728", lw=2.6, label="Candidate")
    ax.set(xlim=(0, .5), xlabel="Spatial frequency [cycles/rectified pixel]",
           ylabel="Normalized non-DC radial FFT power",
           title="Candidate cell vs same-checker-parity controls")
    ax.grid(alpha=.25); ax.legend()
    fig.savefig(args.output_dir / "06_radial_spectrum_comparison.png", dpi=180)
    plt.close(fig)

    difference = candidate_map - median_map
    limit = float(np.percentile(np.abs(difference), 99.5))
    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    image = ax.imshow(difference, cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest")
    center = args.rectified_size // 2
    ax.add_patch(Rectangle((center-1.5, center-1.5), 3, 3, fill=False, edgecolor="black", linewidth=1))
    ax.set(title="Candidate normalized 2D power − control median", xlabel="fx bin", ylabel="fy bin")
    fig.colorbar(image, ax=ax, label="Normalized power difference")
    fig.savefig(args.output_dir / "07_2d_spectrum_difference.png", dpi=180)
    plt.close(fig)

    csv_path = args.output_dir / "features.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(features[0].keys()))
        writer.writeheader(); writer.writerows(features)

    cells_csv_path = args.output_dir / "cells.csv"
    cell_fields = ["cell_id", "i", "j", "checker_parity", "center_xy", "vertices_xy", "cell_area_px2",
                   "intersection_area_px2", "roi_overlap_fraction", "cell_overlap_fraction", "roi_iou"]
    with cells_csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=cell_fields)
        writer.writeheader()
        for cell in cells:
            row = {key: cell[key] for key in cell_fields}
            row["center_xy"] = json.dumps(cell["center_xy"], separators=(",", ":"))
            row["vertices_xy"] = json.dumps(cell["vertices_xy"], separators=(",", ":"))
            writer.writerow(row)

    top = ranked[:5]
    controls_summary = [{k: c[k] for k in ("cell_id", "i", "j", "checker_parity", "center_xy", "cell_area_px2",
                                            "intersection_area_px2", "roi_overlap_fraction", "cell_overlap_fraction", "roi_iou")}
                        for c in controls]
    metric_comparison = {}
    candidate_feature = features[0]
    for name, dist, loo in [
        ("radial_spectrum_L1", candidate_radial_l1, loo_radial),
        ("normalized_2d_spectrum_L1", candidate_2d_l1, loo_2d_l1),
        ("normalized_2d_spectrum_L2", candidate_2d_l2, loo_2d_l2),
    ]:
        metric_comparison[name] = {"candidate_distance": dist, "control_leave_one_out_distances": loo,
                                   "control_loo_median": float(np.median(loo)),
                                   "control_loo_min": float(np.min(loo)), "control_loo_max": float(np.max(loo)),
                                   "candidate_robust_z_vs_control_loo": robust_z(dist, loo)}
    roi_area = poly_area(roi)
    intersections_sum = float(sum(c["intersection_area_px2"] for c in ranked))
    report = {
        "image": image_path.name, "image_size_wh": [w, h],
        "source_indexed_grid_report": str(indexed_report_path.relative_to(ROOT)),
        "grid_basis_a_xy_px": a.tolist(), "grid_basis_b_xy_px": b.tolist(),
        "grid_origin_xy_px": origin.tolist(), "selected_checker_parity_from_fit": selected_parity,
        "candidate_cell_parity": candidate["checker_parity"],
        "grid_fit_status": indexed.get("status"),
        "candidate_roi": {"xyxy": list(ROI_XYXY), "area_px2": roi_area,
                           "description": "Visible black C-shaped candidate annotation; not defect ground truth."},
        "candidate_cell": {k: candidate[k] for k in ("cell_id", "i", "j", "checker_parity", "center_xy", "vertices_xy",
                                                         "cell_area_px2", "intersection_area_px2", "roi_overlap_fraction",
                                                         "cell_overlap_fraction", "roi_iou")},
        "candidate_top5_by_roi_intersection": [{k: c[k] for k in ("cell_id", "i", "j", "checker_parity", "center_xy",
                                                                      "intersection_area_px2", "roi_overlap_fraction",
                                                                      "cell_overlap_fraction", "roi_iou")}
                                                for c in top],
        "overlap_context_cell_not_used_as_control": ({k: overlap_context[k] for k in ("cell_id", "i", "j", "checker_parity",
                                                                                       "center_xy", "intersection_area_px2",
                                                                                       "roi_overlap_fraction", "cell_overlap_fraction", "roi_iou")}
                                                       if overlap_context else None),
        "number_of_enumerated_image_cells": len(cells),
        "roi_coverage_check": {"sum_intersection_area_over_overlapping_cells_px2": intersections_sum,
                               "roi_area_px2": roi_area,
                               "coverage_ratio_sum_may_exceed_1_due_to_overlap": intersections_sum / roi_area},
        "controls": controls_summary,
        "control_selection": "Same parity as candidate; zero ROI intersection; all four vertices inside image; nearest cell centers first. Image15 trimmed frame contains no explicit material-label mask; controls still require visual review.",
        "rectified_size_px": [args.rectified_size, args.rectified_size],
        "homography_vertex_order": "origin + (-a-b)/2, origin + (a-b)/2, origin + (a+b)/2, origin + (-a+b)/2; mapped in that order to square corners",
        "preprocessing": "grayscale; subtract each rectified cell mean; separable 2D Hann; fft2; fftshift; power=abs(F)^2; radial curve excludes only DC sample and is normalized by total non-DC power",
        "frequency_bands_cycles_per_rectified_pixel": {"low": list(LOW_BAND), "middle": list(MID_BAND), "high": list(HIGH_BAND)},
        "metrics_candidate": {k: candidate_feature[k] for k in candidate_feature if k not in {"role", "cell_id", "i", "j", "checker_parity", "vertices_image_xy", "center_x", "center_y", "cell_area_px2", "roi_intersection_area_px2", "roi_overlap_fraction", "cell_overlap_fraction", "roi_iou", "rectified_size_px"}},
        "distance_comparison": metric_comparison,
        "interpretation": {
            "roi_fraction_in_candidate_cell": candidate["roi_overlap_fraction"],
            "candidate_cell_fraction_occupied_by_roi": candidate["cell_overlap_fraction"],
            "roi_cell_iou": candidate["roi_iou"],
            "roi_straddles_multiple_cells": len(ranked) > 1,
            "overlap_is_sufficient_to_interpret_fft_strongly": candidate["roi_overlap_fraction"] >= 0.8,
            "manual_visual_cell_check": (f"The black C-shaped mark is visible in context cell ({overlap_context['i']},{overlap_context['j']}) "
                                          f"rather than candidate cell ({candidate['i']},{candidate['j']}); "
                                          "the ROI bounding box slightly favors the wrong cell for the visible mark."
                                          if overlap_context else "No second overlapping cell to visually compare."),
            "conclusion": "The candidate ROI is not fully contained in one cell if overlap is below 80%; FFT differences must be treated as exploratory. The visible mark is not ground-truth labeled, and controls are unverified visual candidates.",
            "grid_fit_warning": "The reused indexed grid fit is marked needs_review; no local curvature correction is applied.",
        },
        "outputs": ["01_cell_positions.png", "02_candidate_overlap_ranking.png", "03_original_cell_montage.png",
                    "04_rectified_cell_montage.png", "05_fft_2d_montage.png", "06_radial_spectrum_comparison.png",
                    "07_2d_spectrum_difference.png", "features.csv", "cells.csv"],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = f"""# image15 格子セル単位FFT比較\n\n## 目的\n\n48×48 pxの格子中心パッチは、目視で囲った傷候補ROIに十分重ならないことが分かりました。今回は既存の indexed lattice を使い、格子線で囲まれたセル全体を解析単位にします。ROIは黒いC字状模様の目視範囲であり、傷のground truthではありません。\n\n## 手順\n\n`09_indexed_cells.png` を作った既存レポートから格子ベクトル・原点・(i,j)・checker parityを読み、各セルをベクトルの±1/2で囲む四辺形として復元しました。全セルとROIのpolygon intersectionを計算し、ROI面積を最も多く含むセルを候補に選択しています。同じchecker parity、ROI重なり0、セル全体が画像内という条件で、候補近傍の8セルをcontrolにしています。\n\n各セルをhomographyで **{args.rectified_size}×{args.rectified_size} px** に変換してから、同じ平均値除去・2D Hann窓・2D FFTを行いました。\n\n## 結果\n\n- Candidate: **{candidate['cell_id']}**, `(i,j)=({candidate['i']},{candidate['j']})`, parity **{candidate['checker_parity']}**\n- ROI面積のうちcandidateセル内: **{candidate['roi_overlap_fraction']:.1%}**\n- candidateセル面積のうちROI: **{candidate['cell_overlap_fraction']:.1%}**\n- ROI/cell IoU: **{candidate['roi_iou']:.3f}**\n- Control cells: {', '.join(f"{c['cell_id']}({c['i']},{c['j']})" for c in controls)}\n- Candidate radial-spectrum L1 distance: **{candidate_radial_l1:.4f}**; control leave-one-out median: **{np.median(loo_radial):.4f}** (range {np.min(loo_radial):.4f}–{np.max(loo_radial):.4f})\n- Candidate normalized 2D-spectrum L1 distance: **{candidate_2d_l1:.4f}**; control leave-one-out median: **{np.median(loo_2d_l1):.4f}**\n\nROIの重なり上位は report.json の `candidate_top5_by_roi_intersection` を確認してください。\n\n## 出力画像の確認ポイント\n\n- `01_cell_positions.png`: 全セル境界と(i,j)、候補（赤）、control（水色）、候補ROI（黄）。まずROIがどのセルにまたがるか確認します。\n- `02_candidate_overlap_ranking.png`: ROIとの交差面積が大きい上位セルと重なり率。\n- `03_original_cell_montage.png`: 射影変換前の四角形セル領域。\n- `04_rectified_cell_montage.png`: 同じ正方形サイズへ射影変換したセル。ここでC字模様が候補に入っているか確認します。\n- `05_fft_2d_montage.png`: 同じ色スケールの2D FFT log power。方向性を含めて比較します。\n- `06_radial_spectrum_comparison.png`: radial averageによる周波数対強度。これは補助比較で、単独で結論しません。\n- `07_2d_spectrum_difference.png`: 候補の正規化2D powerからcontrol中央値を引いた差分。\n\n## 解釈と制約\n\nセル候補は `(i,j)=({candidate['i']},{candidate['j']})` ですが、ROI面積の **{candidate['roi_overlap_fraction']:.1%}** しか含みません。上位セルが複数に分かれている場合、C字状模様がセル境界をまたいでいます。その場合、今回のセル単位比較でも傷形状全体を1セルに閉じ込められず、FFT差を強く解釈できません。\n\n格子推定レポートの状態は `{indexed.get('status')}` です。曲面による局所的なセル変形補正は行っていません。controlsは同じchecker parityの周辺候補ですが、正常確認済みラベルではありません。image15一枚の探索結果から傷判定性能を一般化できません。\n"""
    if overlap_context:
        readme += (f"\n## ROI crosses a cell boundary\n\nThe next-overlap cell is `({overlap_context['i']},{overlap_context['j']})` "
                   f"with {overlap_context['roi_overlap_fraction']:.1%} of the ROI inside it. "
                   "It has the opposite checker parity, so it is shown in the raw/rectified cell montages as context but is excluded from FFT controls. "
                   f"Visual inspection of the rectified montage shows the black C-shaped mark in this context cell, not in the selected candidate `({candidate['i']},{candidate['j']})`. "
                   "The rectangular ROI area therefore selects the wrong parity for measuring the mark itself. The FFT comparison is included for pipeline verification only and cannot assess that C-shaped mark.\n")
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote {args.output_dir}")
    print(f"Candidate cell ({candidate['i']},{candidate['j']}), parity={candidate['checker_parity']}, "
          f"ROI overlap={candidate['roi_overlap_fraction']:.1%}, cell fraction={candidate['cell_overlap_fraction']:.1%}, IoU={candidate['roi_iou']:.3f}")
    print("Top cells:", [(c["i"], c["j"], round(c["roi_overlap_fraction"], 3)) for c in ranked[:5]])
    print("Controls:", [(c["i"], c["j"]) for c in controls])
    print(f"Radial L1 candidate={candidate_radial_l1:.4f}; control LOO median={np.median(loo_radial):.4f} "
          f"range={np.min(loo_radial):.4f}..{np.max(loo_radial):.4f}")
    print(f"2D L1 candidate={candidate_2d_l1:.4f}; control LOO median={np.median(loo_2d_l1):.4f}")


if __name__ == "__main__":
    main()
