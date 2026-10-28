"""Render a detection-score histogram for each image."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs" / "point_anomaly_peak_analysis" / "09_gaussian_s8_top10"
OUT = SOURCE / "10_score_histograms"
THRESHOLD = float(json.loads(
    (ROOT / "outputs" / "point_anomaly_peak_analysis" / "parameter_summary.json")
    .read_text(encoding="utf-8")
)["base_threshold"])


def image_number(name: str) -> int:
    return int(name.replace("image", ""))


def load_scores():
    grouped = defaultdict(list)
    with (SOURCE / "all_candidate_scores.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            grouped[row["image"]].append(float(row["score"]))
    return grouped


def render_histogram(name: str, scores: list[float], path: Path):
    values = np.sort(np.asarray(scores, dtype=float))[::-1]
    top10 = values[:10]
    rank10 = float(values[9])
    rank11 = float(values[10])
    threshold = THRESHOLD
    upper = float(values.max())
    bins = np.linspace(threshold, upper, 61)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.hist(values, bins=bins, color="#4C78A8", alpha=0.78,
            label=f"All candidates (n={len(values):,})")
    ax.hist(top10, bins=bins, color="#F28E2B", alpha=0.95,
            label="Top 10 candidates")
    ax.axvline(rank10, color="#D62728", linewidth=2, linestyle="--",
               label=f"Rank 10 = {rank10:.2f}")
    ax.axvline(rank11, color="#9467BD", linewidth=1.8, linestyle=":",
               label=f"Rank 11 = {rank11:.2f}")
    ax.axvline(threshold, color="black", linewidth=1.2, linestyle="--",
               label=f"Candidate threshold = {threshold:.2f}")
    ax.set_yscale("log")
    ax.set_title(f"{name}: Gaussian sigma=8 detection-score histogram")
    ax.set_xlabel("Detection score: max(GaussianBlur - original) [gray levels]")
    ax.set_ylabel("Connected components [count, log scale]")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    ax.text(
        0.98, 0.55,
        f"median all = {np.median(values):.2f}\n"
        f"rank 10 - rank 11 = {rank10-rank11:.3f}\n"
        f"relative gap = {(rank10-rank11)/rank11*100:.3f}%",
        transform=ax.transAxes, ha="right", va="top",
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def make_review_pages(paths: list[Path]):
    thumb = (800, 464)
    for page, start in enumerate(range(0, len(paths), 4), 1):
        sheet = Image.new("RGB", (1600, 928), "white")
        for index, path in enumerate(paths[start:start + 4]):
            image = Image.open(path).convert("RGB")
            image.thumbnail(thumb, Image.Resampling.LANCZOS)
            x = (index % 2) * 800 + (800 - image.width) // 2
            y = (index // 2) * 464 + (464 - image.height) // 2
            sheet.paste(image, (x, y))
        sheet.save(OUT / f"score_histograms_review_page_{page:02d}.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    grouped = load_scores()
    paths = []
    for name in sorted(grouped, key=image_number):
        path = OUT / f"{name}_gaussian_s8_score_histogram.png"
        render_histogram(name, grouped[name], path)
        paths.append(path)
        print(name, len(grouped[name]))
    make_review_pages(paths)


if __name__ == "__main__":
    main()
