"""Compare grayscale histograms of manually trimmed images with their originals."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data20260903" / "data20260903"
TRIM_DIR = SOURCE / "triming"
OUT = ROOT / "outputs" / "trimmed_histograms"


def read_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def normalized_hist(gray: np.ndarray) -> np.ndarray:
    counts = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    return counts * (100.0 / max(counts.sum(), 1.0))


def peak_bands(gray: np.ndarray):
    counts = np.bincount(gray.ravel(), minlength=256)
    smooth = gaussian_filter1d(counts.astype(float), 2)
    candidates, _ = find_peaks(smooth, distance=6, prominence=smooth.max() * .003)
    if len(candidates):
        prominences = peak_prominences(smooth, candidates)[0]
        order = np.argsort(prominences)[::-1][:3]
        peaks = sorted(int(candidates[i]) for i in order)
    else:
        peaks = [int(np.argmax(smooth))]
    bands = []
    for i, peak in enumerate(peaks):
        gaps = []
        if i:
            gaps.append(peak - peaks[i - 1])
        if i + 1 < len(peaks):
            gaps.append(peaks[i + 1] - peak)
        radius = int(np.clip(round(min(gaps) / 4), 4, 12)) if gaps else 8
        bands.append([max(0, peak - radius), min(255, peak + radius)])
    valleys = [left + int(np.argmin(smooth[left:right + 1]))
               for left, right in zip(peaks[:-1], peaks[1:])]
    return smooth, peaks, bands, valleys


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trimmed_paths = sorted(
        TRIM_DIR.glob("image*_triming.png"),
        key=lambda p: int(re.search(r"image(\d+)_triming", p.stem).group(1)),
    )
    if not trimmed_paths:
        raise FileNotFoundError(f"No image*_triming.png files in {TRIM_DIR}")

    records = []
    fig, axes = plt.subplots(4, 4, figsize=(18, 15), constrained_layout=True)
    axes = axes.ravel()
    for ax, trim_path in zip(axes, trimmed_paths):
        number = int(re.search(r"image(\d+)_triming", trim_path.stem).group(1))
        original_path = SOURCE / f"image{number}.png"
        if not original_path.exists():
            ax.axis("off")
            continue
        original, trimmed = read_gray(original_path), read_gray(trim_path)
        orig_pct, trim_pct = normalized_hist(original), normalized_hist(trimmed)
        trim_smooth, peaks, bands, valleys = peak_bands(trimmed)

        fig_one, one_ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
        levels = np.arange(256)
        one_ax.plot(levels, orig_pct, color="#9aa0a6", linewidth=1.2,
                    label=f"Original ({original.shape[1]}×{original.shape[0]})")
        one_ax.plot(levels, trim_pct, color="#1769aa", linewidth=1.6,
                    label=f"Trimmed ({trimmed.shape[1]}×{trimmed.shape[0]})")
        colors = ["#1769aa", "#ed6a24", "#14804a"]
        for i, (peak, (lo, hi)) in enumerate(zip(peaks, bands)):
            color = colors[i % len(colors)]
            one_ax.axvspan(lo - .5, hi + .5, color=color, alpha=.12)
            one_ax.axvline(peak, color=color, linewidth=1, linestyle="--")
        one_ax.set(xlim=(0, 255), xlabel="Brightness [0–255]",
                   ylabel="Pixel fraction [%]",
                   title=f"image{number}: original vs trimmed histogram")
        one_ax.grid(alpha=.2)
        one_ax.legend(fontsize=9)
        fig_one.savefig(OUT / f"image{number}_histogram_comparison.png", dpi=150)
        plt.close(fig_one)

        ax.plot(levels, orig_pct, color="#aab0b5", linewidth=.85, label="Original")
        ax.plot(levels, trim_pct, color="#1769aa", linewidth=1.05, label="Trimmed")
        for i, (peak, (lo, hi)) in enumerate(zip(peaks, bands)):
            color = colors[i % len(colors)]
            ax.axvspan(lo - .5, hi + .5, color=color, alpha=.12)
            ax.axvline(peak, color=color, linewidth=.65, linestyle="--")
        ax.set(xlim=(0, 255), title=f"image{number}: peaks {', '.join(map(str, peaks))}")
        ax.grid(alpha=.18)

        records.append({
            "image_number": number,
            "original_file": str(original_path.relative_to(ROOT)),
            "trimmed_file": str(trim_path.relative_to(ROOT)),
            "original_size_wh": [original.shape[1], original.shape[0]],
            "trimmed_size_wh": [trimmed.shape[1], trimmed.shape[0]],
            "original_mean_brightness": float(original.mean()),
            "trimmed_mean_brightness": float(trimmed.mean()),
            "trimmed_peak_levels": peaks,
            "trimmed_peak_bands_inclusive": bands,
            "trimmed_peak_pixel_counts": [int(((trimmed >= lo) & (trimmed <= hi)).sum()) for lo, hi in bands],
            "trimmed_histogram_valleys": valleys,
            "note": "Histogram curves are normalized to percent of pixels so original and cropped images of different sizes are comparable.",
        })

    for ax in axes[len(trimmed_paths):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", ncol=1,
                   bbox_to_anchor=(.99, .99), fontsize=9)
    fig.suptitle("Original vs manually trimmed grayscale histograms (normalized pixel fraction)")
    fig.savefig(OUT / "all_trimmed_histogram_comparisons.png", dpi=170)
    plt.close(fig)

    (OUT / "trimmed_histogram_summary.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{
        "image": f"image{r['image_number']}",
        "trim_size_wh": r["trimmed_size_wh"],
        "original_mean": round(r["original_mean_brightness"], 2),
        "trimmed_mean": round(r["trimmed_mean_brightness"], 2),
        "trim_peaks": r["trimmed_peak_levels"],
        "trim_bands": r["trimmed_peak_bands_inclusive"],
    } for r in records], ensure_ascii=False, indent=2))
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
