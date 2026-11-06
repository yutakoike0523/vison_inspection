#!/usr/bin/env python3
"""Build the normalised/unnormalised grid-removal comparison for image2--18."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import fft_grid_notch_removal_analysis as grid
import image15_grid_removal_fft_comparison as comparison
import local_multiscale_grid_harmonic_analysis as local
import window_size_comparison_unnormalized as labels


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_all_images_grid_removal_fft_comparison"
PER_IMAGE = OUT / "per_image"
IMAGE_IDS = tuple(range(2, 19))
plt.rcParams.update({
    "font.family": ["Yu Gothic", "Meiryo", "DejaVu Sans"],
    "axes.unicode_minus": False,
})
CONDITION_LABELS = {
    "No grid removal": "元画像（格子除去なし）",
    "Local fundamentals": "基本周期を除去",
    "Fundamental + confirmed harmonics": "基本周期＋確認済み高調波を除去",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(image_id: int, images: dict[str, np.ndarray], maps: dict,
           xs: np.ndarray, ys: np.ndarray, masks: dict,
           scales: dict[str, tuple[float, float]]) -> Path:
    original = images["No grid removal"]
    fig, axes = plt.subplots(3, 3, figsize=(20, 14), constrained_layout=True)
    columns = (("norm", "正規化あり"), ("raw", "正規化なし"))
    for row, (condition, background) in enumerate(images.items()):
        ax = axes[row, 0]
        ax.imshow(background, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{CONDITION_LABELS[condition]}\nヒートマップ重畳前")
        ax.set_axis_off()

    for map_col, (key, heading) in enumerate(columns, start=1):
        lo, hi = scales[key]
        plotted = []
        for row, (condition, background) in enumerate(images.items()):
            ax = axes[row, map_col]
            ax.imshow(background, cmap="gray", vmin=0, vmax=255)
            heat = ax.imshow(
                maps[condition][key], extent=comparison.extent(xs, ys),
                cmap="inferno", vmin=lo, vmax=hi, alpha=.68,
                interpolation="nearest")
            plotted.append(heat)
            if masks["defect_pixel"].any():
                ax.contour(masks["defect_pixel"].astype(np.uint8), [.5],
                           colors="#38ff5a", linewidths=.8)
            if masks["exclusion_pixel"].any():
                ax.contour(masks["exclusion_pixel"].astype(np.uint8), [.5],
                           colors="#00e5ff", linewidths=.7)
            ax.set(xlim=(0, original.shape[1]), ylim=(original.shape[0], 0))
            ax.set_title(
                f"{heading}：{CONDITION_LABELS[condition]}\n"
                f"共通色域（正常領域p05–p99）：{lo:.3f}–{hi:.3f} dB")
            ax.set_axis_off()
        fig.colorbar(plotted[-1], ax=axes[:, map_col], label="スペクトル差RMS (dB)",
                     shrink=.72, pad=.015)
    fig.suptitle(
        f"image{image_id}：格子除去前後の局所放射FFT比較\n"
        "窓128 px、ストライド16 px、48 px間隔の周囲8窓平均、0.01–0.50 cycles/pixel",
        fontsize=15)
    path = PER_IMAGE / f"image{image_id}_normalised_unnormalised_shared_normal_scale.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def contact_sheet(paths: list[Path]) -> None:
    thumbs = []
    for path in paths:
        with Image.open(path) as source:
            thumb = source.convert("RGB")
            thumb.thumbnail((450, 570), Image.Resampling.LANCZOS)
            thumbs.append((path.stem.split("_")[0], thumb.copy()))
    cols = 4
    rows = (len(thumbs)+cols-1)//cols
    cell_w, cell_h = 480, 620
    sheet = Image.new("RGB", (cols*cell_w, rows*cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (name, thumb) in enumerate(thumbs):
        col, row = index % cols, index // cols
        left = col*cell_w + (cell_w-thumb.width)//2
        top = row*cell_h + 32
        draw.text((col*cell_w+12, row*cell_h+10), name, fill="black", font=font)
        sheet.paste(thumb, (left, top))
    sheet.save(OUT / "all_images_contact_sheet.png")


def main() -> None:
    PER_IMAGE.mkdir(parents=True, exist_ok=True)
    top = grid.read_csv(local.PRIOR / "fft_peak_pairs_top20.csv")
    family = grid.read_csv(local.PRIOR / "grid_family_candidates.csv")
    summary: list[dict] = []
    output_paths: list[Path] = []

    for image_id in IMAGE_IDS:
        print(f"image{image_id}: local grid removal", flush=True)
        original, windows, _ = local.analyze_image(
            image_id, top, family, detailed=(image_id == 15))
        fundamentals, _ = local.ola(original, windows, "local_base")
        confirmed, _ = local.ola(original, windows, "local_all")
        images = {
            "No grid removal": original.astype(np.float32),
            "Local fundamentals": fundamentals.astype(np.float32),
            "Fundamental + confirmed harmonics": confirmed.astype(np.float32),
        }

        maps = {}
        target_xs = target_ys = None
        for condition, image in images.items():
            result = comparison.spectra(image)
            step = comparison.SPACING // comparison.STRIDE
            target_xs = result["xs"][step:-step]
            target_ys = result["ys"][step:-step]
            raw_score, _, _ = comparison.strict_mean8_score(result["raw"])
            norm_score, _, _ = comparison.strict_mean8_score(result["norm"])
            maps[condition] = {"raw": raw_score, "norm": norm_score}

        assert target_xs is not None and target_ys is not None
        masks = labels.masks_for(
            image_id, original.shape, target_xs, target_ys, grid.DATA)
        normal = masks["normal"]
        scales = {
            key: comparison.limits(maps["No grid removal"][key][normal])
            for key in ("norm", "raw")
        }
        output_paths.append(render(
            image_id, images, maps, target_xs, target_ys, masks, scales))

        for key, display in (("norm", "Normalised"), ("raw", "Unnormalised")):
            lo, hi = scales[key]
            baseline = float(np.median(
                maps["No grid removal"][key][np.isfinite(maps["No grid removal"][key])]))
            for condition in images:
                score = maps[condition][key]
                values = score[np.isfinite(score)]
                normal_values = score[normal & np.isfinite(score)]
                median = float(np.median(values))
                summary.append({
                    "image": image_id,
                    "spectrum": display,
                    "condition": condition,
                    "display_p05": lo,
                    "display_p99": hi,
                    "score_median": median,
                    "score_median_change_from_none_pct":
                        100*(median/baseline-1) if baseline else 0.0,
                    "score_p99": float(np.percentile(values, 99)),
                    "normal_median": float(np.median(normal_values)),
                    "normal_p99": float(np.percentile(normal_values, 99)),
                    "normal_positions": int(normal_values.size),
                })
        np.savez_compressed(
            OUT / f"image{image_id}_score_maps.npz",
            xs=target_xs, ys=target_ys, normal_mask=normal,
            **{f"{kind}_{key}": value[key]
               for kind, value in zip(
                   ("none", "fundamentals", "harmonics"), maps.values())
               for key in ("norm", "raw")})
        del images, maps, windows, original, fundamentals, confirmed

    write_csv(OUT / "all_images_score_summary.csv", summary)
    contact_sheet(output_paths)
    settings = {
        "images": list(IMAGE_IDS),
        "excluded_no_grid_images": [1, 19, 20, 21],
        "window_px": comparison.WINDOW,
        "stride_px": comparison.STRIDE,
        "reference_distance_px": comparison.SPACING,
        "reference_count": 8,
        "reference_aggregation": "per-frequency arithmetic mean",
        "frequency_range_cycles_per_pixel": list(comparison.FREQ),
        "display_scale": "per-image original auxiliary normal-region p05-p99; shared across three conditions within each spectrum column",
        "harmonic_confirmation": "image15 uses 512/256/128 multiscale confirmation; other images reproduce the existing experiment's 512-scale confirmation",
    }
    (OUT / "settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(OUT / "all_images_contact_sheet.png", flush=True)


if __name__ == "__main__":
    main()
