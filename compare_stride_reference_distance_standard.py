#!/usr/bin/env python3
"""Compare stride and reference distance with the canonical image11 settings."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import fft

import fft_grid_notch_removal_analysis as grid
import image15_grid_removal_fft_comparison as comparison
import standard_local_fft_settings as standard
import window_size_comparison_unnormalized as labels


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_stride_reference_distance_standard"
IMAGE_IDS = (15, 16)
STRIDES = (16, 8, 4)
DISTANCES = (48, 128, 256)
BASE_STRIDE = 4
EPS = np.float32(1e-12)
OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1)
                for dx in (-1, 0, 1) if dy or dx)
plt.rcParams.update({
    "font.family": ["Yu Gothic", "Meiryo", "DejaVu Sans"],
    "axes.unicode_minus": False,
})


def spectra_stride4(image: np.ndarray, workers: int = 4,
                    batch: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute raw and normalised spectra exactly as the image11 generator."""
    setup = comparison.radial_setup()
    height, width = image.shape
    half = standard.WINDOW_PX // 2
    xs = np.arange(half, width-half+1, BASE_STRIDE, dtype=np.int32)
    ys = np.arange(half, height-half+1, BASE_STRIDE, dtype=np.int32)
    valid = setup["valid"]
    raw = np.empty((len(ys), len(xs), int(valid.sum())), np.float32)
    norm = np.empty_like(raw)
    hann1 = np.hanning(standard.WINDOW_PX).astype(np.float32)
    hann = np.outer(hann1, hann1).astype(np.float32)

    for iy, y in enumerate(ys):
        row = np.lib.stride_tricks.sliding_window_view(
            image[y-half:y+half], standard.WINDOW_PX, axis=1)
        row = np.moveaxis(row, 1, 0)[::BASE_STRIDE]
        for start in range(0, len(xs), batch):
            values = np.asarray(row[start:start+batch], np.float32).copy()
            values -= values.mean(axis=(-2, -1), keepdims=True)
            values *= hann
            transformed = fft.rfft2(values, axes=(-2, -1), workers=workers)
            power = np.abs(fft.fftshift(transformed, axes=(-2,)))**2
            power = power.astype(np.float32)
            sums = power.reshape(len(power), -1) @ setup["matrix"]
            mean_power = sums[:, valid] / setup["counts"][valid][None, :]
            total = np.maximum(power[:, setup["mask"]].sum(axis=1), EPS)
            raw[iy, start:start+len(values)] = 10*np.log10(mean_power + EPS)
            norm[iy, start:start+len(values)] = 10*np.log10(
                mean_power/total[:, None] + EPS)
    return xs, ys, raw, norm


def mean8_score(spec: np.ndarray, stride: int,
                distance: int) -> np.ndarray:
    step = distance // stride
    target = spec[step:-step, step:-step]
    refs = np.stack([
        spec[step+dy*step:len(spec)-step+dy*step,
             step+dx*step:spec.shape[1]-step+dx*step]
        for dy, dx in OFFSETS
    ])
    reference = refs.mean(axis=0)
    return np.sqrt(np.mean((target-reference)**2, axis=-1)).astype(np.float32)


def limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    lo, hi = np.percentile(finite, standard.DISPLAY_PERCENTILES)
    return float(lo), float(hi if hi > lo else lo+1e-6)


def overlay_annotations(ax, masks: dict) -> None:
    if masks["defect_pixel"].any():
        ax.contour(masks["defect_pixel"].astype(np.uint8), [.5],
                   colors="#38ff5a", linewidths=.8)
    if masks["exclusion_pixel"].any():
        ax.contour(masks["exclusion_pixel"].astype(np.uint8), [.5],
                   colors="#00e5ff", linewidths=.7)


def render(image_id: int, image: np.ndarray, kind: str, maps: dict,
           scales: tuple[float, float]) -> Path:
    display = "正規化あり" if kind == "norm" else "正規化なし"
    lo, hi = scales
    fig, axes = plt.subplots(3, 3, figsize=(18, 13), constrained_layout=True)
    plotted = None
    for row, distance in enumerate(DISTANCES):
        for col, stride in enumerate(STRIDES):
            item = maps[(distance, stride)]
            ax = axes[row, col]
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            extent = [item["xs"][0]-stride/2, item["xs"][-1]+stride/2,
                      item["ys"][-1]+stride/2, item["ys"][0]-stride/2]
            plotted = ax.imshow(item[kind], extent=extent, cmap="inferno",
                                interpolation="nearest", vmin=lo, vmax=hi,
                                alpha=.68)
            overlay_annotations(ax, item["masks"])
            ax.set(xlim=(0, image.shape[1]), ylim=(image.shape[0], 0),
                   title=f"d={distance} px｜stride={stride} px")
            ax.set_axis_off()
    fig.colorbar(plotted, ax=axes, label="スペクトル差RMS (dB)",
                 shrink=.78, pad=.012)
    fig.suptitle(
        f"image{image_id}｜{display}｜ストライド・参照距離比較\n"
        f"W=128、周囲8窓平均、0.01–0.50 cycles/pixel｜"
        f"基準(s16,d48)正常領域p05–p99を全パネルで共有：{lo:.3f}–{hi:.3f} dB",
        fontsize=15)
    output = OUT / f"image{image_id}_{kind}_stride_distance_comparison.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def contact_sheet(paths: list[Path]) -> None:
    thumbs = []
    for path in paths:
        with Image.open(path) as source:
            thumb = source.convert("RGB")
            thumb.thumbnail((850, 610), Image.Resampling.LANCZOS)
            thumbs.append((path.stem, thumb.copy()))
    sheet = Image.new("RGB", (1800, 1320), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (name, thumb) in enumerate(thumbs):
        col, row = index % 2, index // 2
        left = col*900+(900-thumb.width)//2
        top = row*660+35
        draw.text((col*900+15, row*660+10), name, fill="black", font=font)
        sheet.paste(thumb, (left, top))
    sheet.save(OUT / "image15_image16_all_comparisons.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    outputs: list[Path] = []
    for image_id in IMAGE_IDS:
        print(f"image{image_id}: spectra at stride 4", flush=True)
        image = grid.load_gray(image_id).astype(np.float32)
        base_xs, base_ys, base_raw, base_norm = spectra_stride4(image)
        maps = {}
        for distance in DISTANCES:
            for stride in STRIDES:
                factor = stride // BASE_STRIDE
                xs = base_xs[::factor]
                ys = base_ys[::factor]
                raw_spec = base_raw[::factor, ::factor]
                norm_spec = base_norm[::factor, ::factor]
                raw_score = mean8_score(raw_spec, stride, distance)
                norm_score = mean8_score(norm_spec, stride, distance)
                step = distance // stride
                target_xs = xs[step:-step]
                target_ys = ys[step:-step]
                masks = labels.masks_for(
                    image_id, image.shape, target_xs, target_ys, grid.DATA)
                maps[(distance, stride)] = {
                    "xs": target_xs, "ys": target_ys,
                    "raw": raw_score, "norm": norm_score, "masks": masks,
                }
                for kind, score in (("norm", norm_score), ("raw", raw_score)):
                    normal = score[masks["normal"] & np.isfinite(score)]
                    defect = score[masks["defect"] & np.isfinite(score)]
                    rows.append({
                        "image": image_id, "spectrum": kind,
                        "stride_px": stride, "reference_distance_px": distance,
                        "positions": int(np.isfinite(score).sum()),
                        "normal_positions": len(normal),
                        "defect_positions": len(defect),
                        "normal_median": float(np.median(normal)) if len(normal) else np.nan,
                        "normal_p99": float(np.percentile(normal, 99)) if len(normal) else np.nan,
                        "defect_median": float(np.median(defect)) if len(defect) else np.nan,
                        "defect_p99": float(np.percentile(defect, 99)) if len(defect) else np.nan,
                    })

        baseline = maps[(48, 16)]
        norm_scale = limits(baseline["norm"][baseline["masks"]["normal"]])
        raw_scale = limits(baseline["raw"][baseline["masks"]["normal"]])
        outputs.append(render(image_id, image, "norm", maps, norm_scale))
        outputs.append(render(image_id, image, "raw", maps, raw_scale))
        payload = {}
        for (distance, stride), item in maps.items():
            prefix = f"d{distance}_s{stride}"
            for key in ("xs", "ys", "raw", "norm"):
                payload[f"{prefix}_{key}"] = item[key]
        np.savez_compressed(OUT/f"image{image_id}_stride_distance_maps.npz", **payload)
        del base_raw, base_norm, maps

    with (OUT/"score_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    settings = {
        "canonical_source": str(ROOT/"analysis_all_images_grid_removal_fft_comparison"/"per_image"/"image11_normalised_unnormalised_shared_normal_scale.png"),
        "window_px": standard.WINDOW_PX,
        "baseline_stride_px": standard.STRIDE_PX,
        "baseline_reference_distance_px": standard.REFERENCE_DISTANCE_PX,
        "tested_stride_px": list(STRIDES),
        "tested_reference_distance_px": list(DISTANCES),
        "reference_count": standard.REFERENCE_COUNT,
        "reference_aggregation": standard.REFERENCE_AGGREGATION,
        "frequency_range_cycles_per_pixel": list(standard.FREQUENCY_RANGE_CYCLES_PER_PIXEL),
        "radial_bin_width": standard.RADIAL_BIN_WIDTH,
        "window_function": standard.WINDOW_FUNCTION,
        "remove_window_mean": standard.REMOVE_WINDOW_MEAN,
        "score": standard.SCORE,
        "normalised_spectrum": standard.POWER_NORMALISATION,
        "display": "baseline stride16/d48 original normal-region p05-p99, shared across all 9 panels",
        "images": {"main": 15, "low_visible_defect": 16},
    }
    (OUT/"settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    contact_sheet(outputs)
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
