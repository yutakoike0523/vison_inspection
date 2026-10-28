#!/usr/bin/env python3
"""Low-frequency weighted radial-FFT analysis with per-image LOIO thresholds.

Uses the saved 256 px / stride 16 px / fixed-K=4 spectral scores in
``output_frequency_band_analysis``.  No FFT is recomputed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import evaluate_all_radial_fft as prev
import multiscale_fft_experiment as base
import stride_grid_overlap_experiment as old


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "output_frequency_band_analysis" / "fixed_references" / "images"
OUT = ROOT / "output_weighted_fft_loio_analysis"
ALPHAS = (0.00, 0.10, 0.25, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)
HIST_ALPHAS = (0.50, 0.75, 0.90, 0.95, 1.00)
DEFORMATION = prev.CLASSES[0]


def write_csv(path: Path, rows: list[dict]) -> None:
    prev.safe_csv(path, rows)


def stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "median": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(len(values)), "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def read_image(image: int) -> dict:
    path = base.DATA / f"image{image}.png"
    gray = np.asarray(Image.open(path).convert("L"))
    labels = prev.annotation(path.with_suffix(".json"), gray.shape)
    result = old.load_result(ROOT / "output_all_images_radial_fft" / f"image{image}" / "result.npz")
    normal, entries = prev.classification(labels, result)
    with np.load(SOURCE / f"image{image}" / "scores_and_selections.npz") as data:
        low = data["score_C"].astype(float)       # 0.01 <= f < 0.05
        high = data["score_B"].astype(float)      # 0.05 <= f <= 0.50
        all_band = data["score_A"].astype(float)  # 0.01 <= f <= 0.50
    return {"image": image, "gray": gray, "result": result, "normal": normal,
            "entries": entries, "low": low, "high": high, "baseline": all_band}


def weighted(low: np.ndarray, high: np.ndarray, alpha: float) -> np.ndarray:
    return np.sqrt(alpha * low * low + (1.0 - alpha) * high * high)


def loio_threshold(data: dict[int, dict], image: int, alpha: float) -> float | None:
    train = []
    for number, item in data.items():
        if number == image:
            continue
        scores = weighted(item["low"], item["high"], alpha)
        normal = scores[item["normal"] & np.isfinite(scores)]
        if len(normal) >= prev.MIN_NORMAL:
            train.append(normal)
    return prev.weighted_threshold(train, q=.99)


def save_score_map(path: Path, score: np.ndarray, item: dict, threshold: float | None, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    axes[0].imshow(item["gray"], cmap="gray")
    axes[0].set_title(f"image{item['image']} original")
    for ent in item["entries"]:
        if ent["label"] == DEFORMATION:
            axes[0].contour(ent["center"], [.5], colors="cyan", linewidths=.7)
    finite = score[np.isfinite(score)]
    lo, hi = np.percentile(finite, [1, 99])
    im = axes[1].imshow(score, extent=old.extent(item["result"], 16), cmap="inferno", vmin=lo, vmax=hi)
    axes[1].set_title(title)
    fig.colorbar(im, ax=axes[1], label="weighted RMS score (dB)")
    if threshold is not None:
        axes[1].contour(score > threshold, [.5], colors="cyan", linewidths=.6,
                        extent=old.extent(item["result"], 16))
    for ax in axes:
        ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def histogram_sheet(path: Path, data: dict[int, dict], alpha: float | None, loio: dict[int, float | None], label: str, order=range(1, 22)) -> None:
    all_scores = []
    for item in data.values():
        s = item["baseline"] if alpha is None else weighted(item["low"], item["high"], alpha)
        all_scores.append(s[item["normal"] & np.isfinite(s)])
    xmax = max(2.0, float(np.percentile(np.concatenate(all_scores), 99.9)) * 1.12)
    bins = np.linspace(0, xmax, 51)
    fig, axes = plt.subplots(7, 3, figsize=(18, 27), sharex=True)
    for ax, number in zip(axes.flat, order):
        if number not in data:
            ax.set_title(f"image{number}", fontsize=10, weight="bold")
            ax.text(.5, .5, "not evaluated\n(no saved radial FFT score map)", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        item = data[number]
        score = item["baseline"] if alpha is None else weighted(item["low"], item["high"], alpha)
        normal = score[item["normal"] & np.isfinite(score)]
        defect_masks = [e["center"] for e in item["entries"] if e["label"] == DEFORMATION]
        defect = score[np.logical_or.reduce(defect_masks) & np.isfinite(score)] if defect_masks else np.array([])
        ax.hist(normal, bins=bins, color="#4e79a7", alpha=.82, label="normal")
        if len(defect):
            ax.hist(defect, bins=bins, color="#e15759", alpha=.78, label="deformation")
        if loio.get(number) is not None:
            ax.axvline(loio[number], color="#222222", ls="--", lw=1, label="LOIO p99")
        ax.set_title(f"image{number}  n={len(normal):,}", fontsize=10, weight="bold")
        ax.set_ylabel("windows")
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=6, frameon=False)
    for ax in axes[-1]: ax.set_xlabel("anomaly score (dB RMS)")
    fig.suptitle(f"All-image anomaly-score histograms — {label}\nCommon score range; dashed line = image-specific LOIO 99th-percentile threshold", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, .965))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    for folder in (OUT / "weighted_scores" / "score_maps", OUT / "loio_audit", OUT / "histograms" / "baseline",
                   OUT / "threshold_sweep", OUT / "summary"):
        folder.mkdir(parents=True, exist_ok=True)
    # Histograms cover all 21 images.  LOIO is intentionally limited to the
    # 16 regular grid images (2–17), which contain the 15 target instances.
    data = {i: read_image(i) for i in range(1, 22) if (ROOT / "output_all_images_radial_fft" / f"image{i}" / "result.npz").exists()}
    loio_data = {i: data[i] for i in range(2, 18)}
    # The 15 deformation instances in grid images 2–17 are the LOIO evaluation set.
    defect_count = sum(sum(e["label"] == DEFORMATION for e in x["entries"]) for x in loio_data.values())
    if defect_count != 15:
        raise RuntimeError(f"Expected 15 deformation instances; found {defect_count}")

    alpha_rows: list[dict] = []
    defect_rows: list[dict] = []
    image_rows: list[dict] = []
    threshold_rows: list[dict] = []
    maps: dict[float, dict[int, np.ndarray]] = {}
    thresholds: dict[float, dict[int, float | None]] = {}
    for alpha in ALPHAS:
        maps[alpha] = {}
        thresholds[alpha] = {}
        for number, item in data.items():
            score = weighted(item["low"], item["high"], alpha)
            maps[alpha][number] = score
            threshold = loio_threshold(loio_data, number, alpha) if number in loio_data else None
            thresholds[alpha][number] = threshold
            normal = score[item["normal"] & np.isfinite(score)]
            nstats = stats(normal)
            above = normal > threshold if threshold is not None else np.zeros(len(normal), bool)
            image_rows.append(dict(image=number, alpha=alpha, loio_threshold=threshold,
                                   normal_exceed_rate=float(np.mean(above)) if len(normal) else None,
                                   normal_exceed_count=int(above.sum()), **{f"normal_{k}": v for k, v in nstats.items()}))
            for entry in item["entries"]:
                if entry["label"] != DEFORMATION:
                    continue
                values = score[entry["center"] & np.isfinite(score)]
                dstats = stats(values)
                detected = bool(np.any(values > threshold)) if threshold is not None and len(values) else False
                defect_rows.append(dict(image=number, instance=entry["index"], alpha=alpha, loio_threshold=threshold,
                                        detected=detected, defect_exceed_rate=float(np.mean(values > threshold)) if len(values) and threshold is not None else None,
                                        defect_max_minus_threshold=float(values.max() - threshold) if len(values) and threshold is not None else None,
                                        S=(dstats["median"] - nstats["median"]) / np.std(normal) if len(values) and np.std(normal) > 1e-9 else None,
                                        **{f"defect_{k}": v for k, v in dstats.items()}))
            np.savez_compressed(OUT / "weighted_scores" / "score_maps" / f"image{number}_alpha_{alpha:.2f}.npz",
                                score=score, alpha=alpha, low_score=item["low"], high_score=item["high"], loio_threshold=threshold)
        rows = [r for r in defect_rows if r["alpha"] == alpha and r["image"] in loio_data]
        fps = [r for r in image_rows if r["alpha"] == alpha and r["image"] in loio_data]
        alpha_rows.append(dict(alpha=alpha, detected=sum(r["detected"] for r in rows), evaluable=len(rows),
                               normal_exceed_rate_mean=float(np.mean([r["normal_exceed_rate"] for r in fps])),
                               normal_exceed_images=sum(r["normal_exceed_count"] > 0 for r in fps),
                               median_S=float(np.nanmedian([r["S"] for r in rows]))))
    write_csv(OUT / "weighted_scores" / "alpha_metrics.csv", alpha_rows)
    write_csv(OUT / "weighted_scores" / "defect_metrics.csv", defect_rows)
    write_csv(OUT / "weighted_scores" / "image_metrics.csv", image_rows)
    write_csv(OUT / "loio_audit" / "loio_thresholds.csv", [dict(image=i, alpha=a, threshold=t) for a, d in thresholds.items() for i, t in d.items()])
    write_csv(OUT / "threshold_sweep" / "threshold_metrics.csv", alpha_rows)

    # Baseline uses all-band RMS; alpha scores use the requested low/high weighting.
    baseline_thresholds = {i: prev.weighted_threshold([x["baseline"][x["normal"] & np.isfinite(x["baseline"])] for j, x in loio_data.items() if j != i]) if i in loio_data else None for i in data}
    histogram_sheet(OUT / "histograms" / "baseline" / "all_images_full.png", data, None, baseline_thresholds, "baseline: all band 0.01–0.50")
    for alpha in HIST_ALPHAS:
        directory = OUT / "histograms" / f"alpha_{alpha:.2f}"; directory.mkdir(exist_ok=True)
        histogram_sheet(directory / "all_images_full.png", data, alpha, thresholds[alpha], f"weighted alpha={alpha:.2f}")
    for alpha in (0.50, 0.75, 0.90, 0.95, 1.00):
        for number in (2, 5, 15, 17):
            save_score_map(OUT / "weighted_scores" / "score_maps" / f"image{number}_alpha_{alpha:.2f}.png", maps[alpha][number], data[number], thresholds[alpha][number], f"weighted alpha={alpha:.2f}; cyan = > LOIO threshold")

    # Summary figures.
    x = np.array([r["alpha"] for r in alpha_rows])
    fig, ax = plt.subplots(figsize=(8, 4.2)); ax.plot(x, [r["detected"] for r in alpha_rows], "o-", label="LOIO detected / 15")
    ax.set(xlabel="low-frequency weight alpha", ylabel="detected instances", xticks=x, ylim=(0, 15)); ax.grid(alpha=.25); ax.legend()
    fig.savefig(OUT / "summary" / "weight_vs_detection.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.2)); ax.plot(x, [r["normal_exceed_rate_mean"] * 100 for r in alpha_rows], "o-", color="#e15759")
    ax.set(xlabel="low-frequency weight alpha", ylabel="mean normal exceedance (%)", xticks=x); ax.grid(alpha=.25)
    fig.savefig(OUT / "summary" / "weight_vs_false_positive.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    keys = [(r["image"], r["instance"]) for r in defect_rows if r["alpha"] == ALPHAS[0] and r["image"] in loio_data]
    matrix = np.array([[next(r["detected"] for r in defect_rows if r["alpha"] == a and r["image"] == im and r["instance"] == inst) for a in ALPHAS] for im, inst in keys], float)
    fig, ax = plt.subplots(figsize=(10, 7)); im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set(xticks=range(len(ALPHAS)), xticklabels=[f"{a:.2f}" for a in ALPHAS], yticks=range(len(keys)), yticklabels=[f"image{i}:{j}" for i, j in keys], xlabel="low-frequency weight alpha", ylabel="deformation instance", title="LOIO detection")
    fig.colorbar(im, ax=ax, label="detected")
    fig.savefig(OUT / "summary" / "defect_detection_heatmap.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    with (OUT / "summary" / "final_report.md").open("w", encoding="utf-8") as f:
        best = max(alpha_rows, key=lambda r: (r["detected"], -r["normal_exceed_rate_mean"]))
        f.write("# Weighted low-frequency FFT LOIO analysis\n\n")
        f.write("Scores reuse the fixed-K=4, 256 px, stride-16 radial-FFT spectra. No FFT was recomputed.\n\n")
        f.write(f"Evaluation: {defect_count} deformation instances in images 2–17; LOIO threshold is the equal-image-weighted 99th percentile of normal windows.\n\n")
        f.write(f"Best tested alpha: {best['alpha']:.2f}; detected {best['detected']}/{best['evaluable']}; mean normal exceedance {best['normal_exceed_rate_mean']:.2%}.\n")
    print(OUT.resolve())


if __name__ == "__main__":
    main()
