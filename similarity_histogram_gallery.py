"""Histogram gallery for the existing 128 px / log 2-D FFT cosine similarity maps.

The image order and 4-column placement match similarity_map_log_128_contact_sheet.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fft2_cosine_similarity as fft


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data20260903" / "data20260903"
OUT = ROOT / "outputs" / "fft_similarity_histograms_128_log"
BLOCK_SIZE = 128
STRIDE = BLOCK_SIZE // 2
GRID_SIZE = 129
FMAX = 0.125
DC_BINS = 5


def image_key(path: Path) -> tuple[int, int | str]:
    suffix = path.stem.removeprefix("image")
    return (0, int(suffix)) if suffix.isdigit() else (1, path.stem.casefold())


def values_for_image(path: Path) -> np.ndarray:
    gray = fft.read_gray(path)
    raw, amplitude = fft.fft_amplitude(gray, subtract_mean=True)
    common = fft.common_spectrum(amplitude, GRID_SIZE, FMAX)
    mask = fft.comparison_mask(GRID_SIZE, DC_BINS)
    reference = fft.vector(common, mask, "log")
    h, w = gray.shape
    values = []
    for y in range(0, h - BLOCK_SIZE + 1, STRIDE):
        for x in range(0, w - BLOCK_SIZE + 1, STRIDE):
            _local_raw, local_amplitude = fft.fft_amplitude(
                gray[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE], subtract_mean=True
            )
            local = fft.common_spectrum(local_amplitude, GRID_SIZE, FMAX)
            score = fft.cosine(reference, fft.vector(local, mask, "log"))
            if score is not None and np.isfinite(score):
                values.append(float(score))
    return np.asarray(values, dtype=float)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted((p for p in INPUT.iterdir() if p.suffix.lower() in fft.EXTENSIONS), key=image_key)
    summaries = []
    all_values: list[tuple[Path, np.ndarray]] = []
    for path in paths:
        values = values_for_image(path)
        all_values.append((path, values))
        summaries.append({
            "image": path.name, "block_size": BLOCK_SIZE, "stride": STRIDE,
            "similarity_mode": "log", "block_count": len(values),
            "min": values.min(), "p05": np.percentile(values, 5),
            "p25": np.percentile(values, 25), "median": np.median(values),
            "mean": values.mean(), "p75": np.percentile(values, 75),
            "p95": np.percentile(values, 95), "max": values.max(),
        })

    bins = np.linspace(0, 1, 31)
    fig, axes = plt.subplots(6, 4, figsize=(24, 6 * 4.2), squeeze=False)
    for ax, (path, values), row in zip(axes.flat, all_values, summaries):
        ax.hist(values, bins=bins, color="#3478b7", edgecolor="white", linewidth=.35)
        ax.axvline(row["median"], color="#f28e2b", lw=1.5, label=f"median={row['median']:.3f}")
        ax.axvline(row["mean"], color="#d62728", lw=1.2, ls="--", label=f"mean={row['mean']:.3f}")
        ax.set(xlim=(0, 1), xlabel="Cosine similarity (log FFT)", ylabel="Block count")
        ax.set_title(f"{path.name} | n={len(values)}", fontsize=11)
        ax.grid(axis="y", alpha=.22)
        ax.legend(fontsize=8, loc="upper left")
    for ax in axes.flat[len(all_values):]:
        ax.axis("off")
    fig.suptitle("128 px / log: similarity histograms (same image positions as similarity maps)", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(OUT / "similarity_histograms_128_log_contact_sheet.png", dpi=150)
    plt.close(fig)

    with (OUT / "similarity_histogram_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(OUT)


if __name__ == "__main__":
    main()
