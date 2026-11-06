#!/usr/bin/env python3
"""Create all-image grid-removal comparisons with the previous-C FFT settings."""
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

import analysis_fft_normalization_128 as fft128
import fft_grid_notch_removal_analysis as grid
import local_multiscale_grid_harmonic_analysis as local
import window_size_comparison_unnormalized as labels


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_all_images_grid_removal_fft_comparison_C"
PER_IMAGE = OUT / "per_image"
IMAGE_IDS = tuple(range(2, 19))
WINDOW = 128
STRIDE = 16
SPACING = 128
DISPLAY_LIMITS = (-2.0, 6.0)
EPS = np.float32(1e-12)
OFFSETS = fft128.OFFSETS
CONDITIONS = (
    ("No grid removal", "元画像（格子除去なし）", "none"),
    ("Local fundamentals", "基本周期を除去", "fundamentals"),
    ("Fundamental + confirmed harmonics",
     "基本周期＋確認済み高調波を除去", "harmonics"),
)
plt.rcParams.update({
    "font.family": ["Yu Gothic", "Meiryo", "DejaVu Sans"],
    "axes.unicode_minus": False,
})


def normalized_spectra(image: np.ndarray, workers: int = 4,
                       batch: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Previous C: total-power-normalised 0.02--0.50 radial dB spectra."""
    setup = fft128.radial_setup()
    height, width = image.shape
    half = WINDOW // 2
    xs = np.arange(half, width-half+1, STRIDE, dtype=np.int32)
    ys = np.arange(half, height-half+1, STRIDE, dtype=np.int32)
    spectra = np.empty(
        (len(ys), len(xs), int(setup["valid"].sum())), np.float32)
    hann1 = np.hanning(WINDOW).astype(np.float32)
    hann = np.outer(hann1, hann1).astype(np.float32)
    ids = setup["ids"]
    weights = setup["weight"]
    mask = setup["mask"]
    counts = setup["counts"][setup["valid"]]

    for iy, y in enumerate(ys):
        row = np.lib.stride_tricks.sliding_window_view(
            image[y-half:y+half], WINDOW, axis=1)
        row = np.moveaxis(row, 1, 0)[::STRIDE]
        for start in range(0, len(xs), batch):
            values = np.asarray(row[start:start+batch], np.float32).copy()
            values -= values.mean(axis=(-2, -1), keepdims=True)
            values *= hann
            transformed = fft.rfft2(values, axes=(-2, -1), workers=workers)
            power = (np.abs(fft.fftshift(transformed, axes=(-2,)))**2).astype(
                np.float32)
            sums = np.stack([
                np.bincount(ids[mask], weights=item[mask]*weights[mask],
                            minlength=len(setup["counts"]))
                for item in power
            ])
            mean_power = sums[:, setup["valid"]] / counts[None, :]
            total = np.maximum(power[:, mask].sum(axis=1), EPS)
            spectra[iy, start:start+len(values)] = 10*np.log10(
                mean_power/total[:, None] + EPS)
    return xs, ys, spectra


def median8_score(spec: np.ndarray) -> np.ndarray:
    step = SPACING // STRIDE
    target = spec[step:-step, step:-step]
    refs = np.stack([
        spec[step+dy*step:len(spec)-step+dy*step,
             step+dx*step:spec.shape[1]-step+dx*step]
        for dy, dx in OFFSETS
    ])
    reference = np.median(refs, axis=0)
    return np.sqrt(np.mean((target-reference)**2, axis=-1)).astype(np.float32)


def render(image_id: int, images: dict[str, np.ndarray], maps: dict[str, np.ndarray],
           xs: np.ndarray, ys: np.ndarray, masks: dict,
           normal_median: float, normal_std: float) -> Path:
    original = images["No grid removal"]
    extent = [xs[0]-STRIDE/2, xs[-1]+STRIDE/2,
              ys[-1]+STRIDE/2, ys[0]-STRIDE/2]
    fig, axes = plt.subplots(3, 3, figsize=(20, 15), constrained_layout=True)
    plotted_raw = None
    plotted_z = None
    for row, (key, label, _) in enumerate(CONDITIONS):
        axes[row, 0].imshow(images[key], cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title(f"{label}\nヒートマップ適用前")
        axes[row, 0].set_axis_off()

        plotted_raw = axes[row, 1].imshow(
            maps[key], extent=extent, cmap="inferno", interpolation="nearest",
            vmin=0.0, vmax=6.0)
        axes[row, 1].set(
            title=f"{label}\nC方式のRMS差そのもの（dB）",
            xlim=(0, original.shape[1]), ylim=(original.shape[0], 0))
        axes[row, 1].set_axis_off()

        z = (maps[key]-normal_median)/normal_std
        plotted_z = axes[row, 2].imshow(
            z, extent=extent, cmap="inferno", interpolation="nearest",
            vmin=DISPLAY_LIMITS[0], vmax=DISPLAY_LIMITS[1])
        if masks["defect_pixel"].any():
            axes[row, 2].contour(
                masks["defect_pixel"].astype(np.uint8), [.5],
                colors="#38ff5a", linewidths=.8)
        if masks["exclusion_pixel"].any():
            axes[row, 2].contour(
                masks["exclusion_pixel"].astype(np.uint8), [.5],
                colors="#00e5ff", linewidths=.7)
        axes[row, 2].set(
            title=f"{label}\n参考：正常領域基準のZ表示",
            xlim=(0, original.shape[1]), ylim=(original.shape[0], 0))
        axes[row, 2].set_axis_off()

    fig.colorbar(plotted_raw, ax=axes[:, 1], label="RMSスペクトル差 (dB)",
                 shrink=.78, pad=.015)
    fig.colorbar(plotted_z, ax=axes[:, 2], label="正常領域からの標準化距離 Z",
                 shrink=.78, pad=.015)
    fig.suptitle(
        f"image{image_id}｜C設定による格子除去前後の局所FFT比較\n"
        "W=128、stride=16、参照距離128 px、周囲8窓中央値、"
        "0.02–0.50総パワー正規化｜中央：元のCと同じRMS 0～6 dB｜右：参考Z表示 −2～6",
        fontsize=15)
    output = PER_IMAGE / f"image{image_id}_C_settings_raw_and_Z.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def contact_sheet(paths: list[Path]) -> None:
    thumbs = []
    for path in paths:
        with Image.open(path) as source:
            thumb = source.convert("RGB")
            thumb.thumbnail((420, 540), Image.Resampling.LANCZOS)
            thumbs.append((path.stem.split("_")[0], thumb.copy()))
    cols = 4
    rows = (len(thumbs)+cols-1)//cols
    cell_w, cell_h = 450, 590
    sheet = Image.new("RGB", (cols*cell_w, rows*cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (name, thumb) in enumerate(thumbs):
        col, row = index % cols, index // cols
        left = col*cell_w + (cell_w-thumb.width)//2
        top = row*cell_h + 32
        draw.text((col*cell_w+12, row*cell_h+10), name,
                  fill="black", font=font)
        sheet.paste(thumb, (left, top))
    sheet.save(OUT / "all_images_C_settings_contact_sheet.png")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    PER_IMAGE.mkdir(parents=True, exist_ok=True)
    top = grid.read_csv(local.PRIOR / "fft_peak_pairs_top20.csv")
    family = grid.read_csv(local.PRIOR / "grid_family_candidates.csv")
    records: list[dict] = []
    paths: list[Path] = []

    for image_id in IMAGE_IDS:
        print(f"image{image_id}: C settings", flush=True)
        original, windows, _ = local.analyze_image(
            image_id, top, family, detailed=(image_id == 15))
        fundamentals, _ = local.ola(original, windows, "local_base")
        harmonics, _ = local.ola(original, windows, "local_confirmed")
        images = {
            "No grid removal": original,
            "Local fundamentals": fundamentals,
            "Fundamental + confirmed harmonics": harmonics,
        }
        maps: dict[str, np.ndarray] = {}
        target_xs = target_ys = None
        for key, _, _ in CONDITIONS:
            xs, ys, spec = normalized_spectra(images[key])
            maps[key] = median8_score(spec)
            step = SPACING // STRIDE
            target_xs = xs[step:-step]
            target_ys = ys[step:-step]
            del spec

        assert target_xs is not None and target_ys is not None
        masks = labels.masks_for(
            image_id, original.shape, target_xs, target_ys, grid.DATA)
        base_normal = maps["No grid removal"][masks["normal"]]
        base_normal = base_normal[np.isfinite(base_normal)]
        normal_median = float(np.median(base_normal))
        normal_std = float(np.std(base_normal))
        if normal_std <= 0:
            normal_std = 1.0

        paths.append(render(
            image_id, images, maps, target_xs, target_ys, masks,
            normal_median, normal_std))
        for key, label, short in CONDITIONS:
            values = maps[key][np.isfinite(maps[key])]
            normal_values = maps[key][masks["normal"] & np.isfinite(maps[key])]
            records.append({
                "image": image_id,
                "condition": short,
                "condition_ja": label,
                "normalisation": "total power in 0.02-0.50",
                "reference": "median of exactly 8",
                "baseline_normal_median_rms_db": normal_median,
                "baseline_normal_std_rms_db": normal_std,
                "condition_normal_median_rms_db": float(np.median(normal_values)),
                "condition_normal_std_rms_db": float(np.std(normal_values)),
                "score_median_rms_db": float(np.median(values)),
                "score_p99_rms_db": float(np.percentile(values, 99)),
            })
        np.savez_compressed(
            OUT / f"image{image_id}_C_settings_maps.npz",
            xs=target_xs, ys=target_ys, normal_mask=masks["normal"],
            baseline_normal_median=normal_median,
            baseline_normal_std=normal_std,
            **{short: maps[key] for key, _, short in CONDITIONS})
        del original, fundamentals, harmonics, images, maps, windows

    write_csv(OUT / "all_images_C_settings_summary.csv", records)
    contact_sheet(paths)
    settings = {
        "images": list(IMAGE_IDS),
        "window_px": WINDOW,
        "stride_px": STRIDE,
        "reference_distance_px": SPACING,
        "reference_count": 8,
        "reference_aggregation": "per-frequency median",
        "frequency_range_cycles_per_pixel": [0.02, 0.50],
        "power_normalisation": "divide 2D power by total power in 0.02-0.50 before radial averaging",
        "display_standardisation": "use original/no-removal normal-region median and std for all three removal conditions",
        "display_formula": "(score - original_normal_median) / original_normal_std",
        "display_limits": list(DISPLAY_LIMITS),
    }
    (OUT / "settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2)+"\n",
        encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
