"""Reproducible one-parameter-at-a-time sensitivity review of the image15 experiment."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image

from phase_anomaly_detection import Config, ROOT, analyze, find_image, region_comparison
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output_phase_analysis/10_sensitivity")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = find_image()
    gray = np.asarray(Image.open(source).convert("L"), float)
    h, w = gray.shape
    cases = [(1.5, 100), (2, 60), (2, 100), (2, 160), (3, 100)]
    rows = []
    fig, axes = plt.subplots(2, len(cases), figsize=(17, 8), layout="constrained")
    for j, (band, smooth) in enumerate(cases):
        c = replace(Config(), band_sigma=float(band), phase_sigma=float(smooth), amplitude_sigma=float(smooth))
        result = analyze(gray, c)
        regions = region_comparison(result, source.name, [0, 0, w, h], gray.shape)
        if not regions:
            raise ValueError("Sensitivity review requires image15_triming.png with existing R171 coordinates")
        target = next(r for r in regions if r["region_id"] == 171)
        controls = [r for r in regions if r["region_id"] != 171]
        record = dict(band_sigma=band, smoothing_sigma=smooth, phase_valid_fraction=float(result["phase_valid"].mean()))
        for key in ("phase_score", "amplitude_score"):
            value = target[key]["median"]
            control_values = [r[key]["median"] for r in controls]
            record[f"{key}_target_median"] = value
            record[f"{key}_control_median_of_medians"] = float(np.median(control_values))
            record[f"{key}_target_rank_among_9"] = 1 + sum(v > value for v in control_values)
        rows.append(record)
        for i, key in enumerate(["normalized_phase_score", "normalized_amplitude_score"]):
            ax = axes[i, j]
            ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
            values = result[key]
            alpha = .7 * np.clip(np.nan_to_num(values) / 6, 0, 1)
            im = ax.imshow(np.ma.masked_invalid(values), cmap="inferno", vmin=0, vmax=6, alpha=alpha)
            x0, y0, x1, y1 = target["analysis_box_xyxy"]
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="cyan", linewidth=.8))
            ax.set_title(f"{'Phase' if i == 0 else 'Amplitude'}: band={band}, smooth={smooth}\nR171 rank {record[('phase_score' if i == 0 else 'amplitude_score')+'_target_rank_among_9']}/9")
            ax.set_axis_off()
        print(json.dumps(record), flush=True)
        del result
    fig.colorbar(im, ax=list(axes.ravel()), shrink=.65, label="positive robust z (common display scale)")
    fig.savefig(args.output / "sensitivity.png", dpi=140)
    plt.close(fig)
    with (args.output / "sensitivity.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "sensitivity.json").write_text(json.dumps(dict(image=str(source), cases=rows,
        note="One parameter changed at a time; rank 1 is highest median response. No labeled defect truth; no best-case selection."), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
