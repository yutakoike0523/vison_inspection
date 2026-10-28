"""Render per-image views and pooled metrics from a completed validation CSV.

Separating rendering from LOIO fitting keeps the reproducible numerical
experiment fast and lets all 21 high-resolution panels be generated in small
batches on Windows.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import background_surface_feature_validation as experiment


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "background_surface_feature_validation"
STRING_FIELDS = {"image", "lighting", "block_id", "true_label", "selected_rule", "predicted_label"}
BOOL_FIELDS = {"predicted_background", "evaluated"}


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            row = {}
            for key, value in raw.items():
                if key in STRING_FIELDS:
                    row[key] = value
                elif key in BOOL_FIELDS:
                    row[key] = value.casefold() == "true"
                else:
                    try:
                        row[key] = float(value)
                    except (TypeError, ValueError):
                        row[key] = value
            rows.append(row)
    return rows


def add_metrics(rows: list[dict]) -> None:
    output = []
    scopes = [("overall", rows)]
    scopes += [(f"lighting:{name}", [r for r in rows if r["lighting"] == name]) for name in ("on", "off")]
    scopes += [(f"image:{name}", [r for r in rows if r["image"] == name])
               for name in sorted({r["image"] for r in rows}, key=lambda s: int(s[5:]))]
    for scope, part in scopes:
        valid = [row for row in part if row["evaluated"]]
        if not valid:
            continue
        metrics = experiment.prediction_metrics(
            np.asarray([row["predicted_background"] for row in valid]),
            np.asarray([row["true_label"] for row in valid]),
        )
        output.append({"scope": scope, "evaluated_blocks": len(valid), **metrics})
    experiment.write_csv(OUT / "classification_metrics.csv", output)

    overall = output[0]
    matrix = np.array([[overall["tn_surface"], overall["fp_surface_as_background"]],
                       [overall["fn_background_as_surface"], overall["tp_background"]]])
    fig, ax = plt.subplots(figsize=(5.2, 4.3), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    for y in range(2):
        for x in range(2):
            ax.text(x, y, str(matrix[y, x]), ha="center", va="center", fontsize=14)
    ax.set(xticks=(0, 1), xticklabels=("pred. surface", "pred. background"),
           yticks=(0, 1), yticklabels=("true surface", "true background"),
           title="LOIO selected rule: pooled endpoint confusion matrix")
    fig.colorbar(image, ax=ax, label="block count")
    fig.savefig(OUT / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="", help="comma-separated image stems; blank renders all")
    args = parser.parse_args()
    rows = read_rows(OUT / "block_predictions.csv")
    add_metrics(rows)
    experiment.write_clean_japanese_report(
        read_rows(OUT / "feature_summary.csv"), read_rows(OUT / "loio_rule_results.csv"),
        rows, rows[0]["selected_rule"], OUT, 90.0,
    )
    selected = set(filter(None, args.images.split(",")))
    for image in sorted({row["image"] for row in rows}, key=lambda s: int(s[5:])):
        if selected and image not in selected:
            continue
        image_rows = [row for row in rows if row["image"] == image]
        destination = OUT / "per_image" / image
        destination.mkdir(parents=True, exist_ok=True)
        pred = np.asarray([row["predicted_background"] for row in image_rows])
        experiment.classification_figure(image_rows, pred, destination)
        experiment.heatmap_figure(image_rows, destination)
    print("rendered", args.images or "all")


if __name__ == "__main__":
    main()
