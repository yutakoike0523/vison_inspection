"""Render final 21x21/63x63 ring candidates as red squares on originals."""
from __future__ import annotations

from pathlib import Path
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fft_mask_ground_truth_evaluation as gt
from point_anomaly_size_expansion import DATA, TARGETS, POINT, evaluation_masks, ring_score

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs" / "point_anomaly_size_expansion"
OUT = SOURCE / "final_ring_21x21_63x63_candidates"
CONDITION = "A_center_and_ring_c21_o63_raw"


def selected_thresholds():
    with (SOURCE / "size_comparison_by_image.csv").open(encoding="utf-8", newline="") as f:
        return {r["image"]: float(r["selected_threshold"])
                for r in csv.DictReader(f) if r["condition"] == CONDITION}


def components(score, predicted):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(predicted.astype(np.uint8), 8)
    found = []
    for label in range(1, count):
        ys, xs = np.where(labels == label)
        if xs.size == 0:
            continue
        peak = np.argmax(score[ys, xs])
        x, y = int(xs[peak]), int(ys[peak])
        x0, y0, width, height, area = (int(v) for v in stats[label])
        found.append({"x": x, "y": y, "peak_D": float(score[y, x]),
                      "bbox_x": x0, "bbox_y": y0, "bbox_width": width,
                      "bbox_height": height, "component_pixels": area})
    return found


def candidate_squares(rgb, rows, square=12):
    """A cyan filled square marks the peak score in each connected candidate."""
    result = rgb.copy()
    half = square // 2
    for row in rows:
        x, y = row["x"], row["y"]
        cv2.rectangle(result, (max(0, x-half), max(0, y-half)),
                      (min(result.shape[1]-1, x+half), min(result.shape[0]-1, y+half)),
                      (0, 255, 255), thickness=-1)
    return result


def draw_damage_annotations(rgb, rec):
    """All annotated damage types are shown as red bounding boxes."""
    result = rgb.copy()
    damage_labels = set(gt.DAMAGE_MAP.values())
    for shape in rec["shapes"]:
        if shape["normalized_label"] not in damage_labels:
            continue
        x1, y1, x2, y2 = (int(v) for v in shape["box"])
        cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    thresholds = selected_thresholds(); all_rows = []; panels = []
    for name in TARGETS:
        rec = gt.load_record(DATA / f"{name}.json")
        _, point_mask, normal, evaluable, excluded = evaluation_masks(rec)
        score, _ = ring_score(rec["gray"], 21, 63, None)
        threshold = thresholds[name]
        # "evaluation" is the same domain used for the reported FPR.  "all"
        # intentionally keeps background/tag candidates, exposing the need for
        # upstream masking in a deployed inspection pipeline.
        for domain, mask in (("evaluation_area", evaluable), ("all_pixels", np.ones(score.shape, bool))):
            rows = components(score, (score > threshold) & mask)
            for row in rows:
                row.update({"image": name, "threshold": threshold, "domain": domain,
                            "is_point_overlap": bool(point_mask[row["y"], row["x"]]),
                            "is_normal_overlap": bool(normal[row["y"], row["x"]]),
                            "is_known_excluded_overlap": bool(excluded[row["y"], row["x"]])})
            all_rows.extend(rows)
            rendered = candidate_squares(rec["rgb"], rows)
            cv2.imwrite(str(OUT / f"{name}_{domain}_candidate_cyan_squares.png"), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
            if domain == "evaluation_area":
                annotated = draw_damage_annotations(rendered, rec)
                cv2.imwrite(str(OUT / f"{name}_{domain}_candidate_cyan_damage_red_boxes.png"), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                panels.append((name, annotated, threshold, len(rows)))

    fields = ["image", "domain", "threshold", "x", "y", "peak_D", "bbox_x", "bbox_y", "bbox_width", "bbox_height", "component_pixels", "is_point_overlap", "is_normal_overlap", "is_known_excluded_overlap"]
    with (OUT / "candidate_components.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(all_rows)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    for axis, (name, image, threshold, count) in zip(axes.flat, panels):
        axis.imshow(image); axis.set_title(f"{name}: 21/63, t={threshold:.0f}, candidates={count}")
        axis.axis("off")
    fig.suptitle("cyan squares: detector candidates / red boxes: annotated damage", fontsize=12)
    fig.savefig(OUT / "all_images_evaluation_area_candidate_cyan_damage_red_boxes.png", dpi=160); plt.close(fig)
    print(f"wrote {len(all_rows)} candidate rows to {OUT}")


if __name__ == "__main__":
    main()
