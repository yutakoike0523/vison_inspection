#!/usr/bin/env python3
"""Regenerate image15 A/B/C reference-power comparison in normal Z units."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analysis_fft_normalization_128 as fft128
import fft_grid_notch_removal_analysis as grid
import window_size_comparison_unnormalized as labels


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_fft_normalization_128"
IMAGE_ID = 15
DISPLAY_LIMITS = (-2.0, 6.0)
OFFSETS = fft128.OFFSETS
plt.rcParams.update({"font.family": ["Yu Gothic", "Meiryo", "DejaVu Sans"],
                     "axes.unicode_minus": False})


def mean_reference_score(spec: np.ndarray) -> np.ndarray:
    step = fft128.SPACING // fft128.STRIDE
    target = spec[step:-step, step:-step]
    refs = np.stack([
        spec[step+dy*step:len(spec)-step+dy*step,
             step+dx*step:spec.shape[1]-step+dx*step]
        for dy, dx in OFFSETS
    ])
    return np.sqrt(np.mean((target-refs.mean(axis=0))**2, axis=-1)).astype(np.float32)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    image = grid.load_gray(IMAGE_ID).astype(np.float32)
    setup = fft128.radial_setup()
    result = fft128.spectra_for_image(image, setup, workers=4, batch=64)
    step = fft128.SPACING // fft128.STRIDE
    xs = result["xs"][step:-step]
    ys = result["ys"][step:-step]
    maps = {
        "A：8参照平均・パワー正規化なし": mean_reference_score(result["raw"]),
        "B：8参照中央値・パワー正規化なし": fft128.strict_scores(result["raw"]),
        "C：8参照中央値・パワー正規化あり": fft128.strict_scores(result["norm"]),
    }
    masks = labels.masks_for(IMAGE_ID, image.shape, xs, ys, grid.DATA)
    normal = masks["normal"]
    standardized = {}
    rows = []
    for name, score in maps.items():
        values = score[normal & np.isfinite(score)]
        median = float(np.median(values)); std = float(np.std(values))
        z = (score-median)/std
        standardized[name] = z
        rows.append({"condition": name, "normal_windows": len(values),
                     "normal_median_rms_db": median, "normal_std_rms_db": std,
                     "display_formula": "(score - normal_median) / normal_std",
                     "display_vmin": DISPLAY_LIMITS[0], "display_vmax": DISPLAY_LIMITS[1]})

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.2), constrained_layout=True)
    plotted = None
    extent = [xs[0]-fft128.STRIDE/2, xs[-1]+fft128.STRIDE/2,
              ys[-1]+fft128.STRIDE/2, ys[0]-fft128.STRIDE/2]
    for ax, (name, z) in zip(axes, standardized.items()):
        plotted = ax.imshow(z, extent=extent, cmap="inferno", interpolation="nearest",
                            vmin=DISPLAY_LIMITS[0], vmax=DISPLAY_LIMITS[1])
        if masks["defect_pixel"].any():
            ax.contour(masks["defect_pixel"].astype(np.uint8), [.5],
                       colors="#38ff5a", linewidths=.8)
        if masks["exclusion_pixel"].any():
            ax.contour(masks["exclusion_pixel"].astype(np.uint8), [.5],
                       colors="#00e5ff", linewidths=.7)
        ax.set(title=name, xlabel="x (pixel)", ylabel="y (pixel)",
               xlim=(0, image.shape[1]), ylim=(image.shape[0], 0))
    fig.colorbar(plotted, ax=axes, label="正常領域からの標準化距離 Z", shrink=.82)
    fig.suptitle("image15｜W=128｜参照距離128 px｜RMS帯域0.02–0.50 cycles/pixel\n"
                 "表示値 Z=(RMS−正常中央値)/正常標準偏差、全パネル共通 −2～6")
    output = OUT / "image15_reference_power_comparison_standardized.png"
    fig.savefig(output, dpi=150); plt.close(fig)

    with (OUT/"image15_reference_power_comparison_standardization.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(OUT/"image15_reference_power_comparison_standardized_maps.npz",
                        xs=xs, ys=ys, normal_mask=normal,
                        A_mean_raw=standardized["A：8参照平均・パワー正規化なし"],
                        B_median_raw=standardized["B：8参照中央値・パワー正規化なし"],
                        C_median_power_normalized=standardized["C：8参照中央値・パワー正規化あり"])
    print(output)


if __name__ == "__main__":
    main()
