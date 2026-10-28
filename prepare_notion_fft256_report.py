"""Prepare 256-region figures while preserving the original 4x4 annotations."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from multiscale_fft_anomaly_analysis import analyze_scale, LOW_FREQ_EXCLUDE_RADIUS_RATIO

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "⑧_マルチスケールFFT解析"


def parent_region(record):
    return (int(record["row"])-1)//4*4 + (int(record["column"])-1)//4 + 1


def main():
    labels = json.loads((ROOT / "outputs" / "⑦_16分割FFT比較" / "scratch_regions.json").read_text(encoding="utf-8"))
    report = []
    for name, selected in labels.items():
        directory = OUTPUT / name / "16x16"
        with (directory / "features.csv").open(encoding="utf-8-sig", newline="") as stream:
            saved = list(csv.DictReader(stream))
        source = ROOT / "data20260903" / "data20260903" / "triming" / f"{name}_triming.png"
        gray = np.asarray(Image.open(source).convert("L"))
        analyzed = analyze_scale(gray, 16, LOW_FREQ_EXCLUDE_RADIUS_RATIO)
        records, radial, frequency = analyzed[0], analyzed[3], analyzed[10]
        if len(saved) != 256 or len(records) != 256:
            raise ValueError(f"{name}: expected 256 regions")
        for actual, previous in zip(records, saved):
            if actual["region_id"] != int(previous["region_id"]) or not np.isclose(
                    actual["combined_score"], float(previous["combined_score"]), rtol=1e-9, atol=1e-9):
                raise ValueError(f"{name}: recomputed spectra do not match the saved analysis")
        top = sorted(records, key=lambda record: record["combined_score"], reverse=True)[:3]
        top_ids = {record["region_id"] for record in top}
        h, w = gray.shape

        fig, ax = plt.subplots(figsize=(11, 10))
        fig.subplots_adjust(left=.06, right=.88, bottom=.07, top=.90)
        ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
        values = np.asarray([record["combined_score"] for record in records]).reshape(16, 16)
        overlay = ax.imshow(values, cmap="inferno", alpha=.40, interpolation="nearest",
                            extent=[0, w, h, 0], vmin=0, vmax=float(values.max()))
        for record in records:
            x, y = record["x1"], record["y1"]
            rw, rh = record["region_width_px"], record["region_height_px"]
            ax.add_patch(Rectangle((x, y), rw, rh, fill=False, edgecolor="white", linewidth=.35, alpha=.45))
            is_top = record["region_id"] in top_ids
            ax.text(x+rw/2, y+rh/2, str(record["region_id"]), ha="center", va="center",
                    fontsize=7, color="black" if is_top else "white", weight="bold" if is_top else "normal",
                    bbox=dict(facecolor="#ffe600" if is_top else "black", edgecolor="none",
                              alpha=.95 if is_top else .40, pad=1))
            if is_top:
                ax.add_patch(Rectangle((x, y), rw, rh, fill=False, edgecolor="#ffe600", linewidth=2.8, zorder=5))
        for region in selected:
            row, column = divmod(region-1, 4)
            x0, x1 = column*w//4, (column+1)*w//4
            y0, y1 = row*h//4, (row+1)*h//4
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="red", linewidth=3, zorder=6))
            ax.text(x0+2, y0+2, f"4x4 R{region:02d}", ha="left", va="top", color="white", fontsize=8,
                    weight="bold", bbox=dict(facecolor="#c90000", edgecolor="none", alpha=.95, pad=2), zorder=7)
        ax.set(xlim=(0, w), ylim=(h, 0), xlabel="x [pixel]", ylabel="y [pixel]")
        ax.set_title(f"{name}: 16x16 FFT score map\n"
                     "Red: user-marked 4x4 areas; yellow: top 3 FFT regions", fontsize=13)
        cax = fig.add_axes([.90, .17, .018, .63])
        fig.colorbar(overlay, cax=cax, label="Combined FFT difference score")
        map_path = directory / "06_256regions_user_areas_and_top3.png"
        fig.savefig(map_path, dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharey=False)
        fig.subplots_adjust(left=.09, right=.98, bottom=.08, top=.82, hspace=.34)
        indices = sorted(range(256), key=lambda index: parent_region(records[index]) in selected)
        for index in indices:
            inside = parent_region(records[index]) in selected
            style = dict(color="red" if inside else "#777777", alpha=.42 if inside else .20,
                         linewidth=.9 if inside else .6, zorder=3 if inside else 2)
            for ax in axes:
                ax.plot(frequency, radial[index], **style)
        axes[0].set_title("All 256 normalized radial profiles")
        axes[1].set_title("Low-frequency detail (same profiles)")
        axes[1].set_xlim(0, .20)
        for ax in axes:
            ax.set(xlabel="Spatial frequency [cycles/pixel]", ylabel="Normalized radial profile")
            ax.grid(alpha=.25)
        fig.suptitle(f"{name}: 16x16 FFT radial comparison\n"
                     "Red curves: tiles within user-marked 4x4 areas (not individual scratch labels)", fontsize=12)
        fig.legend(handles=[Line2D([0], [0], color="red", label="Within marked 4x4 areas"),
                            Line2D([0], [0], color="#777777", label="Other 16x16 tiles")],
                   loc="upper center", bbox_to_anchor=(.5, .90), ncol=2, fontsize=9)
        fft_path = directory / "07_256radial_user_areas_red.png"
        fig.savefig(fft_path, dpi=160)
        plt.close(fig)

        report.append(dict(image=name, parent_scratch_regions=selected,
                           map_path=str(map_path), fft_path=str(fft_path),
                           top_regions=[dict(region=record["region_id"], score=record["combined_score"],
                                             parent4x4=parent_region(record),
                                             within_marked_area=parent_region(record) in selected)
                                        for record in top]))
        print(f"Prepared {name}; saved scores verified", flush=True)
    (OUTPUT / "notion_256_report_data.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    matches = sum(row["top_regions"][0]["within_marked_area"] for row in report)
    print(f"Highest-score tiles within marked 4x4 areas: {matches}/{len(report)}", flush=True)


if __name__ == "__main__":
    main()
