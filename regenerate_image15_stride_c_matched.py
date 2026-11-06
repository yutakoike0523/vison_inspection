#!/usr/bin/env python3
"""Regenerate image15 stride maps using the exact previous-C signal pipeline."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft

import analysis_fft_normalization_128 as fft128
import fft_grid_notch_removal_analysis as grid
import window_size_comparison_unnormalized as labels


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output_stride_grid_overlap" / "image15" / "comparison"
IMAGE_ID = 15
WINDOW = 128
BASE_STRIDE = 4
SPACING = 128
STRIDES = (16, 8, 4)
DISPLAY_LIMITS = (-2.0, 6.0)
EPS = np.float32(1e-12)
OFFSETS = fft128.OFFSETS
plt.rcParams.update({"font.family": ["Yu Gothic", "Meiryo", "DejaVu Sans"],
                     "axes.unicode_minus": False})


def normalized_spectra_stride4(image: np.ndarray, workers: int = 4,
                               batch: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Previous C: total-power-normalised 0.02--0.50 radial dB spectra."""
    setup = fft128.radial_setup()
    h, w = image.shape; half = WINDOW//2
    xs = np.arange(half, w-half+1, BASE_STRIDE, dtype=np.int32)
    ys = np.arange(half, h-half+1, BASE_STRIDE, dtype=np.int32)
    spectra = np.empty((len(ys), len(xs), int(setup["valid"].sum())), np.float32)
    hann1 = np.hanning(WINDOW).astype(np.float32)
    hann = np.outer(hann1, hann1).astype(np.float32)
    ids = setup["ids"]; weights = setup["weight"]; mask = setup["mask"]
    counts = setup["counts"][setup["valid"]]
    for iy, y in enumerate(ys):
        row = np.lib.stride_tricks.sliding_window_view(
            image[y-half:y+half], WINDOW, axis=1)
        row = np.moveaxis(row, 1, 0)[::BASE_STRIDE]
        for start in range(0, len(xs), batch):
            values = np.asarray(row[start:start+batch], np.float32).copy()
            values -= values.mean(axis=(-2, -1), keepdims=True); values *= hann
            transformed = fft.rfft2(values, axes=(-2, -1), workers=workers)
            power = (np.abs(fft.fftshift(transformed, axes=(-2,)))**2).astype(np.float32)
            sums = np.stack([
                np.bincount(ids[mask], weights=q[mask]*weights[mask],
                            minlength=len(setup["counts"])) for q in power])
            mean_power = sums[:, setup["valid"]] / counts[None, :]
            total = np.maximum(power[:, mask].sum(axis=1), EPS)
            spectra[iy, start:start+len(values)] = 10*np.log10(
                mean_power/total[:, None]+EPS)
    return xs, ys, spectra


def median_reference_score(spec: np.ndarray, stride: int) -> np.ndarray:
    step = SPACING//stride
    target = spec[step:-step, step:-step]
    refs = np.stack([
        spec[step+dy*step:len(spec)-step+dy*step,
             step+dx*step:spec.shape[1]-step+dx*step]
        for dy, dx in OFFSETS
    ])
    reference = np.median(refs, axis=0)
    return np.sqrt(np.mean((target-reference)**2, axis=-1)).astype(np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    image = grid.load_gray(IMAGE_ID).astype(np.float32)
    base_xs, base_ys, base_spec = normalized_spectra_stride4(image)
    records = []
    maps = {}
    for stride in STRIDES:
        factor = stride//BASE_STRIDE
        spec = base_spec[::factor, ::factor]
        xs = base_xs[::factor]; ys = base_ys[::factor]
        score = median_reference_score(spec, stride)
        step = SPACING//stride
        target_xs = xs[step:-step]; target_ys = ys[step:-step]
        masks = labels.masks_for(IMAGE_ID, image.shape, target_xs, target_ys, grid.DATA)
        normal_values = score[masks["normal"] & np.isfinite(score)]
        normal_median = float(np.median(normal_values))
        normal_std = float(np.std(normal_values))
        z = (score-normal_median)/normal_std
        maps[stride] = {"score": score, "z": z, "xs": target_xs, "ys": target_ys,
                        "masks": masks}
        records.append({"image": IMAGE_ID, "window_px": WINDOW, "stride_px": stride,
                        "reference_distance_px": SPACING,
                        "frequency_band": "0.02-0.50", "reference_aggregation": "median of exactly 8",
                        "power_normalisation": "total power in 0.02-0.50",
                        "normal_windows": len(normal_values),
                        "normal_median_rms_db": normal_median,
                        "normal_std_rms_db": normal_std,
                        "display_formula": "(score-normal_median)/normal_std",
                        "display_vmin": DISPLAY_LIMITS[0], "display_vmax": DISPLAY_LIMITS[1]})

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.2), constrained_layout=True)
    plotted = None
    for ax, stride in zip(axes, STRIDES):
        item = maps[stride]
        extent = [item["xs"][0]-stride/2, item["xs"][-1]+stride/2,
                  item["ys"][-1]+stride/2, item["ys"][0]-stride/2]
        plotted = ax.imshow(item["z"], extent=extent, cmap="inferno",
                            interpolation="nearest", vmin=DISPLAY_LIMITS[0],
                            vmax=DISPLAY_LIMITS[1])
        masks = item["masks"]
        if masks["defect_pixel"].any():
            ax.contour(masks["defect_pixel"].astype(np.uint8), [.5],
                       colors="#38ff5a", linewidths=.8)
        if masks["exclusion_pixel"].any():
            ax.contour(masks["exclusion_pixel"].astype(np.uint8), [.5],
                       colors="#00e5ff", linewidths=.7)
        ax.set(title=f"stride {stride} px", xlabel="x (pixel)", ylabel="y (pixel)",
               xlim=(0, image.shape[1]), ylim=(image.shape[0], 0))
    fig.colorbar(plotted, ax=axes, label="正常領域からの標準化距離 Z", shrink=.82)
    fig.suptitle("image15｜前のCと同一処理：W=128、参照距離128 px、8参照中央値、\n"
                 "0.02–0.50総パワー正規化・同帯域RMS｜Z共通表示 −2～6")
    output = OUT / "stride_comparison_C_matched_standardized.png"
    fig.savefig(output, dpi=150); plt.close(fig)

    with (OUT/"stride_comparison_C_matched_standardization.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    np.savez_compressed(OUT/"stride_comparison_C_matched_maps.npz",
                        **{f"stride{stride}_score": maps[stride]["score"] for stride in STRIDES},
                        **{f"stride{stride}_z": maps[stride]["z"] for stride in STRIDES},
                        **{f"stride{stride}_xs": maps[stride]["xs"] for stride in STRIDES},
                        **{f"stride{stride}_ys": maps[stride]["ys"] for stride in STRIDES})
    print(output)


if __name__ == "__main__":
    main()
