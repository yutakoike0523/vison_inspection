"""Analyze whether Gaussian sigma=8 candidates have a natural top-10 score gap."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import point_anomaly_feature_analysis as fa
from render_gaussian_top10 import candidates


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data20260903" / "data20260903"
ANALYSIS = ROOT / "outputs" / "point_anomaly_peak_analysis"
OUT = ANALYSIS / "09_gaussian_s8_top10"


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q))


def describe(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "q25": percentile(values, 25),
        "median": float(np.median(values)),
        "q75": percentile(values, 75),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ANALYSIS / "parameter_summary.json").read_text(encoding="utf-8"))
    threshold = float(config["base_threshold"])
    all_rows = []
    image_rows = []

    for rec in fa.load_records(DATA):
        gray = rec["gray"].astype(np.float32)
        response = cv2.GaussianBlur(
            gray, (0, 0), 8, borderType=cv2.BORDER_REFLECT_101
        ) - gray
        comps = candidates(response, threshold)
        scores = np.asarray([row["score"] for row in comps], dtype=float)
        for rank, row in enumerate(comps, 1):
            all_rows.append({
                "image": rec["name"],
                "rank": rank,
                "group": "top10" if rank <= 10 else "rank11_plus",
                "score": row["score"],
                "area": row["area"],
                "component_x": row["component_x"],
                "component_y": row["component_y"],
            })

        rank10 = float(scores[9])
        rank11 = float(scores[10])
        gap = rank10 - rank11
        adjacent = scores[4:15 - 1] - scores[5:15]
        boundary_gaps = scores[:19] - scores[1:20]
        image_rows.append({
            "image": rec["name"],
            "candidate_count": len(comps),
            "rank1_score": float(scores[0]),
            "rank10_score": rank10,
            "rank11_score": rank11,
            "gap_10_11": gap,
            "gap_percent_of_rank11": 100.0 * gap / rank11,
            "median_adjacent_gap_rank5_to15": float(np.median(adjacent)),
            "gap_vs_local_median_ratio": (
                float(gap / np.median(adjacent)) if np.median(adjacent) > 0 else None
            ),
            "rank10_boundary_is_largest_gap_in_top20": bool(
                np.argmax(boundary_gaps) == 9
            ),
        })

    with (OUT / "all_candidate_scores.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with (OUT / "rank10_rank11_gap.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(image_rows[0]))
        writer.writeheader()
        writer.writerows(image_rows)

    top = [r["score"] for r in all_rows if r["group"] == "top10"]
    rest = [r["score"] for r in all_rows if r["group"] == "rank11_plus"]
    gaps = [r["gap_10_11"] for r in image_rows]
    gap_pct = [r["gap_percent_of_rank11"] for r in image_rows]
    summary = {
        "score_definition": "maximum of GaussianBlur(gray, sigma=8) - gray inside each thresholded connected component",
        "threshold": threshold,
        "top10": describe(top),
        "rank11_plus": describe(rest),
        "rank10_minus_rank11": describe(gaps),
        "rank10_minus_rank11_percent": describe(gap_pct),
        "images_gap_le_1": sum(g <= 1.0 for g in gaps),
        "images_gap_le_2": sum(g <= 2.0 for g in gaps),
        "images_gap_ge_5": sum(g >= 5.0 for g in gaps),
        "images_where_top10_boundary_is_largest_top20_gap": sum(
            r["rank10_boundary_is_largest_gap_in_top20"] for r in image_rows
        ),
        "images": len(image_rows),
    }
    (OUT / "score_gap_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    bins = np.linspace(threshold, max(top), 55)
    axes[0].hist(rest, bins=bins, alpha=0.65, label=f"rank 11+ (n={len(rest)})")
    axes[0].hist(top, bins=bins, alpha=0.65, label=f"top 10 (n={len(top)})")
    axes[0].axvline(threshold, color="black", linestyle="--", linewidth=1, label="threshold")
    axes[0].set_title("Gaussian sigma=8 candidate score distribution")
    axes[0].set_xlabel("score: max(blurred gray - original gray) [gray levels]")
    axes[0].set_ylabel("connected components [count]")
    axes[0].legend()

    x = np.arange(len(image_rows))
    axes[1].bar(x, gaps)
    axes[1].axhline(np.median(gaps), color="black", linestyle="--", linewidth=1,
                    label=f"median={np.median(gaps):.2f}")
    axes[1].set_title("Score gap between rank 10 and rank 11")
    axes[1].set_xlabel("image number")
    axes[1].set_ylabel("rank 10 score - rank 11 score [gray levels]")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([r["image"].replace("image", "") for r in image_rows], fontsize=8)
    axes[1].legend()
    fig.suptitle("Source: all 21 images; fixed threshold and component filters")
    fig.tight_layout()
    fig.savefig(OUT / "score_distribution_and_gap.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
