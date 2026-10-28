"""Compare local 2-D FFT spectra across a simple 4x4 image partition.

This experiment is independent of grid-period, lattice-vector, or phase
estimation. It processes every image*_triming.png file in the input folder.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data20260903" / "data20260903" / "triming"
DEFAULT_OUTPUT = ROOT / "outputs" / "⑦_16分割FFT比較"
FEATURES_FOR_SCORE = [
    "log_non_dc_power", "peak_to_median_ratio", "peak_frequency_x",
    "peak_frequency_y", "spectral_spread_cycles_per_pixel",
    "high_frequency_power_fraction", "non_dc_power_fraction",
]


def region_bounds(length: int, pieces: int = 4):
    """Integer boundaries that cover every pixel, including remainders."""
    return [(i * length // pieces, (i + 1) * length // pieces) for i in range(pieces)]


def fft_region(gray: np.ndarray, window: str):
    data = np.asarray(gray, dtype=np.float64)
    mean_brightness = float(data.mean())
    centered = data - mean_brightness
    h, w = data.shape
    if window == "hann":
        hann = np.outer(np.hanning(h), np.hanning(w))
        processed = centered * hann
    else:
        processed = centered

    spectrum = np.fft.fftshift(np.fft.fft2(processed))
    magnitude = np.abs(spectrum)
    power = magnitude ** 2
    log_magnitude = np.log1p(magnitude)
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    radius = np.hypot(fx_grid, fy_grid)
    dc_radius = 2.0 * max(1.0 / w, 1.0 / h)
    non_dc = radius > dc_radius
    non_dc_power = power[non_dc]
    non_dc_magnitude = magnitude[non_dc]
    if non_dc_power.size == 0:
        non_dc_power = power.ravel()
        non_dc_magnitude = magnitude.ravel()
        non_dc = np.ones_like(power, dtype=bool)

    total_power = float(power.sum())
    non_dc_total = float(non_dc_power.sum())
    peak_flat = int(np.argmax(np.where(non_dc, magnitude, -np.inf)))
    peak_y, peak_x = np.unravel_index(peak_flat, magnitude.shape)
    peak_magnitude = float(magnitude[peak_y, peak_x])
    background_median = float(np.median(non_dc_magnitude))
    peak_to_median = peak_magnitude / max(background_median, 1e-12)
    peak_radius = float(radius[peak_y, peak_x])

    weights = np.where(non_dc, power, 0.0)
    weight_sum = max(float(weights.sum()), 1e-12)
    radial_mean = float(np.sum(weights * radius) / weight_sum)
    radial_variance = float(np.sum(weights * (radius - radial_mean) ** 2) / weight_sum)
    high_frequency = radius >= .25  # cycles/pixel; Nyquist is 0.5 cycles/pixel.

    features = {
        "mean_brightness": mean_brightness,
        "std_brightness": float(data.std()),
        "fft_total_power": total_power,
        "fft_non_dc_power": non_dc_total,
        "log_non_dc_power": float(np.log1p(non_dc_total)),
        "non_dc_power_fraction": non_dc_total / max(total_power, 1e-12),
        "dominant_peak_magnitude": peak_magnitude,
        "peak_to_median_ratio": float(peak_to_median),
        "dominant_peak_power_fraction": float(power[peak_y, peak_x] / max(non_dc_total, 1e-12)),
        "peak_frequency_x": float(fx[peak_x]),
        "peak_frequency_y": float(fy[peak_y]),
        "peak_frequency_radius": peak_radius,
        "spectral_centroid_cycles_per_pixel": radial_mean,
        "spectral_spread_cycles_per_pixel": float(np.sqrt(max(radial_variance, 0.0))),
        "high_frequency_power_fraction": float(power[high_frequency].sum() / max(non_dc_total, 1e-12)),
        "region_width_px": w,
        "region_height_px": h,
        "dc_exclusion_radius_cycles_per_pixel": dc_radius,
    }
    radial_step = max(1.0 / w, 1.0 / h)
    radial_index = np.floor(radius / radial_step).astype(int)
    radial_count = int(radial_index.max()) + 1
    radial_pixels = np.bincount(radial_index.ravel(), minlength=radial_count)
    radial_magnitude_sum = np.bincount(radial_index.ravel(), weights=magnitude.ravel(), minlength=radial_count)
    radial_power_sum = np.bincount(radial_index.ravel(), weights=power.ravel(), minlength=radial_count)
    radial_mean_magnitude = radial_magnitude_sum / np.maximum(radial_pixels, 1)
    non_dc_bins = np.arange(radial_count) * radial_step > dc_radius
    radial_power_fraction = radial_power_sum / max(non_dc_total, 1e-12)
    radial_frequency = (np.arange(radial_count) + .5) * radial_step
    return features, log_magnitude, fx, fy, dc_radius, {
        "frequency_cycles_per_pixel": radial_frequency[non_dc_bins],
        "mean_magnitude": radial_mean_magnitude[non_dc_bins],
        "power_fraction": radial_power_fraction[non_dc_bins],
    }


def robust_scores(rows):
    score_components = {name: np.zeros(len(rows), dtype=float) for name in FEATURES_FOR_SCORE}
    for name in FEATURES_FOR_SCORE:
        values = np.asarray([row[name] for row in rows], dtype=float)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = 1.4826 * mad
        if scale <= 1e-12:
            scale = float(np.std(values))
        if scale <= 1e-12:
            z = np.zeros_like(values)
        else:
            z = (values - median) / scale
        score_components[name] = np.clip(np.abs(z), 0, 10)
    scores = np.mean(np.stack(list(score_components.values()), axis=1), axis=1)
    for index, row in enumerate(rows):
        row["fft_difference_score"] = float(scores[index])
        row["relative_rank_descending"] = int(np.count_nonzero(scores > scores[index]) + 1)
        for name, values in score_components.items():
            row[f"robust_abs_z_{name}"] = float(values[index])
    return scores


def save_region_map(gray, records, output):
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    h, w = gray.shape
    for record in records:
        x, y, rw, rh = record["x0"], record["y0"], record["width_px"], record["height_px"]
        ax.add_patch(Rectangle((x, y), rw, rh, fill=False, edgecolor="#00e5ff", linewidth=1.3))
        ax.text(x + rw / 2, y + rh / 2, str(record["region"]), ha="center", va="center",
                color="white", fontsize=17, weight="bold",
                bbox={"facecolor": "black", "alpha": .55, "pad": 3})
    ax.set(xlim=(0, w), ylim=(h, 0), title="4x4 equal partition; numbered left-to-right, top-to-bottom")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_fft_montage(spectra, records, output, window):
    # Shared limits make spectra comparable within this image.
    pooled = np.concatenate([v.ravel() for v in spectra])
    vmin = float(np.percentile(pooled, 5))
    vmax = float(np.percentile(pooled, 99.7))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    fig, axes = plt.subplots(4, 4, figsize=(15, 14), constrained_layout=True)
    image_handle = None
    for index, (ax, spectrum, row) in enumerate(zip(axes.ravel(), spectra, records)):
        h, w = row["height_px"], row["width_px"]
        fx = np.fft.fftshift(np.fft.fftfreq(w))
        fy = np.fft.fftshift(np.fft.fftfreq(h))
        displayed = spectrum.copy()
        dc_bins = 1
        displayed[max(0, h//2-dc_bins):h//2+dc_bins+1,
                  max(0, w//2-dc_bins):w//2+dc_bins+1] = vmin
        image_handle = ax.imshow(displayed, cmap="magma", origin="lower", vmin=vmin, vmax=vmax,
                                 extent=[fx[0], fx[-1], fy[0], fy[-1]], interpolation="nearest")
        ax.set_title(f"{row['region']:02d}  score={row['fft_difference_score']:.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"4x4 2-D FFT log magnitude; window={window}; shared color scale (DC neighborhood suppressed for display)", fontsize=13)
    if image_handle is not None:
        fig.colorbar(image_handle, ax=axes.ravel().tolist(), shrink=.72, label="log(1 + |FFT|)")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_radial_spectra(radial_spectra, records, output, radial_csv):
    scores = np.asarray([r["fft_difference_score"] for r in records], dtype=float)
    norm = plt.Normalize(float(scores.min()), float(scores.max() + 1e-9))
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    csv_rows = []
    for record, spectrum in zip(records, radial_spectra):
        color = cmap(norm(record["fft_difference_score"]))
        frequency = spectrum["frequency_cycles_per_pixel"]
        mean_magnitude = spectrum["mean_magnitude"]
        power_fraction = spectrum["power_fraction"]
        axes[0].plot(frequency, np.log1p(mean_magnitude), color=color, alpha=.82, linewidth=1.0)
        axes[1].plot(frequency, power_fraction, color=color, alpha=.82, linewidth=1.0,
                     label=f"R{record['region']:02d} (E={record['fft_difference_score']:.2f})")
        for f, mag, power in zip(frequency, mean_magnitude, power_fraction):
            csv_rows.append({"image": record["image"], "region": record["region"],
                             "frequency_cycles_per_pixel": float(f),
                             "mean_magnitude": float(mag), "log1p_mean_magnitude": float(np.log1p(mag)),
                             "radial_bin_power_fraction": float(power)})
    axes[0].set_ylabel("log(1 + mean |FFT|)")
    axes[0].set_title("Radial mean amplitude (1-D spectrum)")
    axes[1].set(xlabel="Spatial frequency [cycles/pixel]",
                ylabel="Fraction of non-DC power per radial bin",
                title="Radial power distribution (normalized per region)")
    for ax in axes:
        ax.grid(alpha=.25)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, .5), ncol=1,
               fontsize=7, title="Region / score")
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(scalar, ax=axes, label="FFT difference score", shrink=.82)
    fig.suptitle("16 regions: frequency vs strength (radial average; direction is collapsed)", fontsize=13)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    if csv_rows:
        with radial_csv.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader(); writer.writerows(csv_rows)


def save_anomaly_heatmap(gray, records, output):
    scores = np.asarray([r["fft_difference_score"] for r in records]).reshape(4, 4)
    h, w = gray.shape
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    overlay = ax.imshow(scores, cmap="inferno", alpha=.52, interpolation="nearest",
                        extent=[0, w, h, 0], vmin=float(scores.min()), vmax=float(scores.max() + 1e-9))
    for row in records:
        cx, cy = row["x0"] + row["width_px"] / 2, row["y0"] + row["height_px"] / 2
        ax.text(cx, cy, f"{row['region']}\n{row['fft_difference_score']:.2f}",
                ha="center", va="center", color="white", fontsize=10, weight="bold",
                bbox={"facecolor": "black", "alpha": .55, "pad": 2})
    ax.set(xlim=(0, w), ylim=(h, 0), title="Relative FFT difference score (not a defect decision)")
    fig.colorbar(overlay, ax=ax, label="FFT difference score")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_summary_panel(gray, spectra, radial_spectra, records, output, window):
    fig = plt.figure(figsize=(17, 13), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax0, ax1 = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    ax2, ax3 = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    h, w = gray.shape
    ax0.imshow(gray, cmap="gray", vmin=0, vmax=255)
    for row in records:
        x, y, rw, rh = row["x0"], row["y0"], row["width_px"], row["height_px"]
        ax0.add_patch(Rectangle((x, y), rw, rh, fill=False, edgecolor="cyan", linewidth=1))
        ax0.text(x + rw/2, y + rh/2, str(row["region"]), color="white", ha="center", va="center",
                 bbox={"facecolor": "black", "alpha": .55, "pad": 2})
    ax0.set(title="1. 4x4 region map", xlim=(0, w), ylim=(h, 0)); ax0.axis("off")

    score_min = min(r["fft_difference_score"] for r in records)
    score_max = max(r["fft_difference_score"] for r in records)
    line_cmap = plt.get_cmap("viridis")
    line_norm = plt.Normalize(score_min, score_max + 1e-9)
    for row, radial in zip(records, radial_spectra):
        ax1.plot(radial["frequency_cycles_per_pixel"], radial["power_fraction"],
                 color=line_cmap(line_norm(row["fft_difference_score"])), linewidth=.9, alpha=.8)
    ax1.set(title="2. Radial 1-D frequency spectra", xlabel="Frequency [cycles/pixel]",
            ylabel="Fraction of non-DC power")
    ax1.grid(alpha=.2)
    ax3.imshow(gray, cmap="gray", vmin=0, vmax=255)
    scores = np.asarray([r["fft_difference_score"] for r in records]).reshape(4, 4)
    hm = ax3.imshow(scores, cmap="inferno", alpha=.55, extent=[0, w, h, 0],
                    vmin=float(scores.min()), vmax=float(scores.max() + 1e-9))
    for row in records:
        ax3.text(row["x0"] + row["width_px"]/2, row["y0"] + row["height_px"]/2,
                 f"{row['region']}\n{row['fft_difference_score']:.2f}", color="white",
                 ha="center", va="center", fontsize=8,
                 bbox={"facecolor": "black", "alpha": .55, "pad": 1.5})
    ax3.set(title="3. FFT difference score", xlim=(0, w), ylim=(h, 0)); ax3.axis("off")
    sorted_rows = sorted(records, key=lambda r: r["fft_difference_score"], reverse=True)
    names = [str(r["region"]) for r in sorted_rows]
    vals = [r["fft_difference_score"] for r in sorted_rows]
    ax2.bar(np.arange(16), vals, color=["#d62728" if i < 3 else "#4c78a8" for i in range(16)])
    ax2.set_xticks(np.arange(16), names); ax2.set_xlabel("Region number, descending score")
    ax2.set_ylabel("FFT difference score"); ax2.set_title("4. Highest-scoring regions: " + ", ".join(names[:3]))
    ax2.grid(axis="y", alpha=.25)
    fig.suptitle(f"{Path(output).parent.name}: local 4x4 FFT comparison (Hann={window == 'hann'})", fontsize=15)
    fig.colorbar(hm, ax=ax3, fraction=.046, label="FFT difference score")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window", choices=("hann", "none"), default="hann",
                        help="Apply Hann window after mean subtraction (default: hann)")
    args = parser.parse_args()
    input_dir, output_root = args.input_dir.resolve(), args.output_dir.resolve()
    paths = sorted(input_dir.glob("image*_triming.png"),
                   key=lambda p: int(re.search(r"image(\d+)", p.stem).group(1)))
    if not paths:
        parser.error(f"No image*_triming.png files found in {input_dir}")
    all_rows = []
    global_summary = []

    for path in paths:
        key = re.search(r"image\d+", path.stem).group(0)
        out = output_root / key
        out.mkdir(parents=True, exist_ok=True)
        # Remove artifact names from the initial 2-D-only experiment if rerun in place.
        for legacy_name in ("02_fft_16regions.png", "03_anomaly_heatmap.png", "04_summary.png"):
            legacy_path = out / legacy_name
            if legacy_path.exists():
                legacy_path.unlink()
        gray = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        height, width = gray.shape
        y_bounds, x_bounds = region_bounds(height), region_bounds(width)
        records, spectra, radial_spectra = [], [], []
        number = 1
        for row in range(4):
            for col in range(4):
                y0, y1 = y_bounds[row]
                x0, x1 = x_bounds[col]
                crop = gray[y0:y1, x0:x1]
                features, log_mag, fx, fy, dc_radius, radial = fft_region(crop, args.window)
                record = {"image": path.name, "region": number, "row": row + 1, "column": col + 1,
                          "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                          "width_px": x1 - x0, "height_px": y1 - y0,
                          "hann_window": args.window == "hann", **features}
                records.append(record); spectra.append(log_mag); radial_spectra.append(radial)
                number += 1
        scores = robust_scores(records)
        sorted_rows = sorted(records, key=lambda r: r["fft_difference_score"], reverse=True)
        save_region_map(gray, records, out / "01_4x4_regions.png")
        save_fft_montage(spectra, records, out / "02_fft_16regions_2d.png", args.window)
        save_radial_spectra(radial_spectra, records, out / "03_fft_16regions_1d.png",
                            out / "radial_spectra.csv")
        save_anomaly_heatmap(gray, records, out / "04_anomaly_heatmap.png")
        save_summary_panel(gray, spectra, radial_spectra, records, out / "05_summary.png", args.window)
        feature_path = out / "features.csv"
        fieldnames = list(records[0])
        with feature_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader(); writer.writerows(records)
        local_summary = {
            "image": path.name, "image_size_wh": [width, height], "grid": "4x4",
            "window": args.window, "regions": 16,
            "top_regions_by_fft_difference_score": [
                {"region": r["region"], "score": r["fft_difference_score"],
                 "dominant_peak_frequency_xy": [r["peak_frequency_x"], r["peak_frequency_y"]]}
                for r in sorted_rows[:5]],
            "score_median": float(np.median(scores)),
            "score_max": float(np.max(scores)),
            "interpretation": "within-image relative spectral difference; not a confirmed defect label",
        }
        (out / "summary.json").write_text(json.dumps(local_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        all_rows.extend(records)
        global_summary.append(local_summary)
        print(f"{path.name}: top regions " + ", ".join(f"{r['region']} ({r['fft_difference_score']:.2f})" for r in sorted_rows[:3]), flush=True)

    global_csv = output_root / "全画像_16分割FFT_summary.csv"
    with global_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    (output_root / "全画像_summary.json").write_text(
        json.dumps(global_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = ["# 4×4領域ごとの局所FFT比較", "",
              f"入力フォルダ: `{input_dir}`", f"処理画像: {len(paths)}枚、計{len(all_rows)}領域", "",
              "## 処理の流れ", "",
              "入力グレースケール画像 → 4×4の16領域に分割 → 平均輝度除去とHann窓 → 各領域の2次元FFT → 動径平均した1次元周波数スペクトル → FFT特徴量とロバスト外れ値スコア → 元画像上のヒートマップ", "",
              "各画像フォルダには、`01_4x4_regions.png`（領域番号）、`02_fft_16regions_2d.png`（方向も分かる2Dスペクトル）、`03_fft_16regions_1d.png`（周波数対強度の比較）、`04_anomaly_heatmap.png`、`05_summary.png`、`features.csv`、`radial_spectra.csv`、`summary.json` を保存しています。", "",
              "1Dスペクトルは2D FFT振幅の動径平均です。上段は平均振幅、下段は各半径ビンの非DCパワー比です。動径平均では方向情報が失われるため、方向の確認には2Dスペクトルも併用してください。", "",
              "FFT difference score は、非DCパワー、ピーク比、ピーク周波数、スペクトル広がり、高周波比などの特徴量を画像内16領域の中央値とMADで標準化し、絶対ロバストz-scoreを平均した相対値です。値が高い領域は他の領域と周波数特性が異なりますが、傷・欠陥を確定する判定ではありません。", "",
              "`全画像_16分割FFT_summary.csv` は全画像の全16領域を1行ずつ記録します。Hann窓なしで比較する場合は `python fft_4x4_region_comparison.py --window none` を実行してください。", ""]
    (output_root / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"Processed {len(paths)} images / {len(all_rows)} regions. Output: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
