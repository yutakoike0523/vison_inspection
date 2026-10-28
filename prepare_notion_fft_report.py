"""Prepare annotated region maps and metadata for the Notion FFT report."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "⑦_16分割FFT比較"


def main():
    labels = json.loads((OUTPUT / "scratch_regions.json").read_text(encoding="utf-8"))
    report = []
    for name, selected in labels.items():
        directory = OUTPUT / name
        with (directory / "features.csv").open(encoding="utf-8-sig", newline="") as stream:
            records = list(csv.DictReader(stream))
        source = ROOT / "data20260903" / "data20260903" / "triming" / records[0]["image"]
        gray = np.asarray(Image.open(source).convert("L"))
        fig, ax = plt.subplots(figsize=(9, 9))
        fig.subplots_adjust(left=.07, right=.98, bottom=.07, top=.90)
        ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
        for record in records:
            region = int(record["region"])
            x, y = int(record["x0"]), int(record["y0"])
            w, h = int(record["width_px"]), int(record["height_px"])
            is_scratch = region in selected
            ax.add_patch(Rectangle((x, y), w, h, fill=False,
                                   edgecolor="#ff0000" if is_scratch else "#00b8cc",
                                   linewidth=3.0 if is_scratch else .9,
                                   zorder=4 if is_scratch else 2))
            ax.text(x+w/2, y+h/2, str(region), ha="center", va="center",
                    color="white", fontsize=18, weight="bold",
                    bbox=dict(facecolor="#c90000" if is_scratch else "black",
                              edgecolor="none", alpha=.85 if is_scratch else .6, pad=4))
        h, w = gray.shape
        ax.set(xlim=(0, w), ylim=(h, 0), xlabel="x [pixel]", ylabel="y [pixel]")
        ax.set_title(f"{name}: 4x4 region map\n"
                     "User-marked scratch regions (red): " + ", ".join(map(str, selected)), fontsize=13)
        map_path = directory / "01_4x4_regions_scratch_red.png"
        fig.savefig(map_path, dpi=160)
        plt.close(fig)
        selected_records = [r for r in records if int(r["region"]) in selected]
        report.append(dict(image=name, regions=selected,
                           map_path=str(map_path),
                           fft_path=str(directory / "03_fft_16regions_1d_scratch_red.png"),
                           scratch_scores=[dict(region=int(r["region"]),
                                                score=float(r["fft_difference_score"]))
                                           for r in selected_records]))
    (OUTPUT / "notion_report_data.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {len(report)} maps / {sum(len(r['regions']) for r in report)} user-marked regions")


if __name__ == "__main__":
    main()
