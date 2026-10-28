"""Point-anomaly study: enlarge the center block and/or surrounding ring.

The experiment intentionally uses the same four light-off images and Labelme
evaluation masks as ring_block_point_anomaly_experiment.py.  It is descriptive
because common thresholds are selected on these same four images.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import fft_mask_ground_truth_evaluation as gt
from background_surface_hist_experiment import write_csv

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data20260903" / "data20260903"
OUT = ROOT / "outputs" / "point_anomaly_size_expansion"
TARGETS = ("image1", "image19", "image20", "image21")
POINT = list(dict.fromkeys(gt.DAMAGE_MAP.values()))[0]
THRESHOLDS = np.arange(0, 101, 5, dtype=float)

# A: ratio 1:3, no ring masking.  B: center fixed, both ring variants.
SPECS = (
    *({"experiment": "A_center_and_ring", "center": c, "outer": c * 3,
       "ring_mode": "all_pixels"} for c in (15, 21, 25, 31, 41)),
    *({"experiment": "B_fixed_center", "center": 15, "outer": o,
       "ring_mode": mode} for o in (45, 63, 81, 101, 123)
      for mode in ("all_pixels", "exclude_known")),
)


def key(spec: dict) -> str:
    suffix = "masked" if spec["ring_mode"] == "exclude_known" else "raw"
    return f"{spec['experiment']}_c{spec['center']}_o{spec['outer']}_{suffix}"


def evaluation_masks(rec):
    points = [s for s in rec["shapes"] if s["normalized_label"] == POINT]
    point_mask = np.zeros(rec["gray"].shape, dtype=bool)
    for shape in points:
        point_mask |= shape["mask"]
    # Unlabelled pixels are the inspection surface; all labelled non-point
    # defects, tag, and background are outside the normal evaluation class.
    normal = ~(rec["masks"]["tag"] | rec["masks"]["background"] | rec["masks"]["damage"])
    evaluable = normal | point_mask
    known_excluded = rec["masks"]["tag"] | rec["masks"]["background"] | rec["masks"]["damage"]
    return points, point_mask, normal, evaluable, known_excluded


def ring_score(gray: np.ndarray, center: int, outer: int, valid_ring: np.ndarray | None):
    """Return D, and fraction of available outer-ring pixels for each pixel."""
    image = gray.astype(np.float32)
    center_mean = cv2.boxFilter(image, -1, (center, center), normalize=True,
                                borderType=cv2.BORDER_REFLECT_101)
    if valid_ring is None:
        outer_mean = cv2.boxFilter(image, -1, (outer, outer), normalize=True,
                                   borderType=cv2.BORDER_REFLECT_101)
        ring_mean = (outer_mean * outer * outer - center_mean * center * center) / (outer * outer - center * center)
        return np.maximum(ring_mean - center_mean, 0), np.ones(gray.shape, np.float32)

    valid = valid_ring.astype(np.float32)
    outer_sum = cv2.boxFilter(image * valid, -1, (outer, outer), normalize=False,
                              borderType=cv2.BORDER_REFLECT_101)
    inner_sum = cv2.boxFilter(image * valid, -1, (center, center), normalize=False,
                              borderType=cv2.BORDER_REFLECT_101)
    outer_n = cv2.boxFilter(valid, -1, (outer, outer), normalize=False,
                            borderType=cv2.BORDER_REFLECT_101)
    inner_n = cv2.boxFilter(valid, -1, (center, center), normalize=False,
                            borderType=cv2.BORDER_REFLECT_101)
    ring_sum, ring_n = outer_sum - inner_sum, outer_n - inner_n
    full_n = float(outer * outer - center * center)
    available = ring_n / full_n
    ring_mean = np.divide(ring_sum, ring_n, out=np.zeros_like(ring_sum), where=ring_n > 0)
    score = np.maximum(ring_mean - center_mean, 0)
    # No usable neighbouring pixels means no local evidence of darkness.
    score[ring_n <= 0] = 0
    return score, available


def threshold_rows(image, condition, score, points, point_mask, normal, evaluable):
    rows = []
    for threshold in THRESHOLDS:
        predicted = (score > threshold) & evaluable
        _, labels, _, _ = cv2.connectedComponentsWithStats(predicted.astype(np.uint8), 8)
        point_labels = set(np.unique(labels[point_mask]).tolist()); point_labels.discard(0)
        normal_labels = set(np.unique(labels[normal]).tolist()); normal_labels.discard(0)
        box_hits = sum(bool((predicted & shape["mask"]).any()) for shape in points)
        rows.append({
            "image": image, "condition": condition, "threshold": float(threshold),
            "point_count": len(points), "point_box_hits": box_hits,
            "point_box_hit_rate": box_hits / max(1, len(points)),
            "point_pixel_recall": float((predicted & point_mask).sum() / max(1, point_mask.sum())),
            "normal_false_positive_rate": float((predicted & normal).sum() / max(1, normal.sum())),
            "normal_false_positive_pixels": int((predicted & normal).sum()),
            "normal_pixels": int(normal.sum()),
            "false_positive_components": len(normal_labels - point_labels),
        })
    return rows


def selection(rows):
    """Best hit rate at normal FPR <=1%, then prefer substantial overlap."""
    safe = [r for r in rows if r["normal_false_positive_rate"] <= 0.01]
    return max(safe or rows, key=lambda r: (r["point_box_hit_rate"], r["point_pixel_recall"], -r["normal_false_positive_rate"]))


def render(rec, score, points, point_mask, normal, condition, selected, output):
    predicted = score > selected["threshold"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    axes[0].imshow(rec["rgb"]); axes[0].set_title(f"{rec['name']}: point annotations")
    for shape in points:
        x1, y1, x2, y2 = shape["box"]
        axes[0].add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, ec="lime", lw=1.5))
    vmax = max(15.0, float(np.percentile(score, 99.9)))
    axes[1].imshow(score, cmap="magma", vmin=0, vmax=vmax); axes[1].set_title("local darkness D")
    axes[2].hist(score[normal], bins=80, range=(0, vmax), density=True, histtype="step", label="normal")
    axes[2].hist(score[point_mask], bins=80, range=(0, vmax), density=True, histtype="step", label="point boxes")
    axes[2].set(title="score distribution", xlabel="D"); axes[2].legend()
    axes[3].imshow(rec["rgb"]); axes[3].contour(predicted, levels=[0.5], colors="cyan", linewidths=0.5)
    axes[3].set_title("t={:.0f}; hit={:.2f}; normal FPR={:.3f}".format(
        selected["threshold"], selected["point_box_hit_rate"], selected["normal_false_positive_rate"]))
    for axis in (axes[0], axes[1], axes[3]): axis.axis("off")
    fig.savefig(output / f"{rec['name']}_{condition}_diagnostic.png", dpi=120)
    plt.close(fig)


def aggregate_common(rows):
    """Pool counts across images for a truly common threshold summary."""
    result = []
    for condition in sorted({r["condition"] for r in rows}):
        for threshold in THRESHOLDS:
            part = [r for r in rows if r["condition"] == condition and r["threshold"] == threshold]
            result.append({
                "condition": condition, "threshold": float(threshold),
                "point_box_hit_rate": sum(r["point_box_hits"] for r in part) / sum(r["point_count"] for r in part),
                "point_pixel_recall_mean": float(np.mean([r["point_pixel_recall"] for r in part])),
                "normal_false_positive_rate_pooled": sum(r["normal_false_positive_pixels"] for r in part) / sum(r["normal_pixels"] for r in part),
                "false_positive_components_mean": float(np.mean([r["false_positive_components"] for r in part])),
            })
    return result


def plot_series(summary, experiment, output):
    data = [r for r in summary if r["experiment"] == experiment]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for mode in sorted({r["ring_mode"] for r in data}):
        group = sorted([r for r in data if r["ring_mode"] == mode], key=lambda r: r["x_size"])
        label = "exclude known" if mode == "exclude_known" else "all pixels"
        axes[0].plot([r["x_size"] for r in group], [r["false_positive_components_mean"] for r in group], "o-", label=label)
        axes[1].plot([r["x_size"] for r in group], [r["point_box_hit_rate"] for r in group], "o-", label=label)
    xname = "center size (px)" if experiment == "A_center_and_ring" else "outer size (px)"
    axes[0].set(xlabel=xname, ylabel="false positive components / image", title="components at image-wise FPR <=1%")
    axes[1].set(xlabel=xname, ylabel="point-box hit rate", ylim=(-.05, 1.05), title="point detection at image-wise FPR <=1%")
    for axis in axes: axis.grid(alpha=.25); axis.legend()
    fig.savefig(output / ("center_size_comparison.png" if experiment == "A_center_and_ring" else "outer_size_comparison.png"), dpi=160)
    plt.close(fig)


def plot_tradeoffs(common, output):
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for condition in sorted({r["condition"] for r in common}):
        part = [r for r in common if r["condition"] == condition]
        ax.plot([r["normal_false_positive_rate_pooled"] for r in part], [r["point_box_hit_rate"] for r in part], "-o", ms=2, label=condition)
    ax.set(xlabel="pooled normal-surface false-positive rate", ylabel="point-box hit rate", title="common-threshold trade-offs")
    ax.grid(alpha=.25); ax.legend(fontsize=5, ncol=2)
    fig.savefig(output / "common_threshold_tradeoffs.png", dpi=170); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    threshold_data, score_data, imagewise = [], [], []
    for image_name in TARGETS:
        rec = gt.load_record(DATA / f"{image_name}.json")
        points, point_mask, normal, evaluable, known_excluded = evaluation_masks(rec)
        for spec in SPECS:
            condition = key(spec)
            valid = None if spec["ring_mode"] == "all_pixels" else ~known_excluded
            score, usable = ring_score(rec["gray"], spec["center"], spec["outer"], valid)
            rows = threshold_rows(image_name, condition, score, points, point_mask, normal, evaluable)
            threshold_data.extend(rows)
            selected = selection(rows)
            imagewise.append({**spec, "condition": condition, "image": image_name, **{
                "selected_" + k: v for k, v in selected.items() if k not in {"image", "condition"}
            }})
            score_data.append({
                **spec, "condition": condition, "image": image_name,
                "normal_median": float(np.median(score[normal])), "normal_p90": float(np.percentile(score[normal], 90)),
                "normal_p99": float(np.percentile(score[normal], 99)), "point_median": float(np.median(score[point_mask])),
                "point_p90": float(np.percentile(score[point_mask], 90)), "point_p99": float(np.percentile(score[point_mask], 99)),
                "ring_usable_fraction_normal_mean": float(np.mean(usable[normal])),
                "ring_usable_fraction_point_mean": float(np.mean(usable[point_mask])),
                "ring_fraction_below_50pct": float(np.mean(usable < .5)),
            })
            render(rec, score, points, point_mask, normal, condition, selected, args.output)

    write_csv(args.output / "threshold_results.csv", threshold_data)
    write_csv(args.output / "score_summary.csv", score_data)
    write_csv(args.output / "size_comparison_by_image.csv", imagewise)
    common = aggregate_common(threshold_data)
    write_csv(args.output / "common_threshold_results.csv", common)
    summary = []
    for spec in SPECS:
        condition = key(spec)
        rows = [r for r in imagewise if r["condition"] == condition]
        summary.append({**spec, "condition": condition,
                        "x_size": spec["center"] if spec["experiment"] == "A_center_and_ring" else spec["outer"],
                        # Aggregate the nine annotated boxes, rather than giving
                        # an image with one box the same weight as image20 (five).
                        "point_box_hit_rate": sum(r["selected_point_box_hits"] for r in rows) / sum(r["selected_point_count"] for r in rows),
                        "point_pixel_recall_mean": float(np.mean([r["selected_point_pixel_recall"] for r in rows])),
                        "normal_false_positive_rate_mean": float(np.mean([r["selected_normal_false_positive_rate"] for r in rows])),
                        "false_positive_components_mean": float(np.mean([r["selected_false_positive_components"] for r in rows]))})
    write_csv(args.output / "size_comparison.csv", summary)
    plot_series(summary, "A_center_and_ring", args.output); plot_series(summary, "B_fixed_center", args.output); plot_tradeoffs(common, args.output)

    baseline = next(r for r in summary if r["condition"] == "A_center_and_ring_c15_o45_raw")
    best = max(summary, key=lambda r: (r["point_box_hit_rate"], -r["false_positive_components_mean"], r["point_pixel_recall_mean"]))
    common_safe = [r for r in common if r["normal_false_positive_rate_pooled"] <= .01]
    best_common = max(common_safe, key=lambda r: (r["point_box_hit_rate"], r["point_pixel_recall_mean"], -r["false_positive_components_mean"]))
    lines = [
        "# 点状異常検出：中心・周辺ブロックサイズ拡大実験", "",
        "## 条件", "- 対象は格子ライトなし image1 / image19 / image20 / image21、点状異常は計9個。",
        "- D = max(μ_ring − μ_center, 0) を1画素ずつ計算した。正常面は未ラベル画素、background/tag/全傷ラベルは評価から除外した。",
        "- Aは中心と周辺を1:3で同時に拡大（周辺は除外なし）。Bは中心15×15固定、周辺のみ拡大し、周辺に既知のbackground/tag/傷を含める場合と除外する場合を分けた。",
        "- 画像別比較は正常面FPR 1%以下でボックス検出率を最大化した探索的値。共通しきい値は `common_threshold_results.csv` のプール結果であり、同じ4画像で選んだ記述的結果である。", "",
        "## 画像別に1%以下のFPRを満たす比較", "",
        "|条件|ボックス検出率|点状画素再現率|正常面FPR|誤検出成分/画像|", "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(summary, key=lambda r: (r["experiment"], r["center"], r["outer"], r["ring_mode"])):
        lines.append("|{}|{:.3f}|{:.3f}|{:.4f}|{:.1f}|".format(row["condition"], row["point_box_hit_rate"], row["point_pixel_recall_mean"], row["normal_false_positive_rate_mean"], row["false_positive_components_mean"]))
    lines += ["", "## 自動集計による要約",
              "- 基準（15/45・周辺除外なし）: 検出率 {:.3f}、誤検出成分 {:.1f}/画像。".format(baseline["point_box_hit_rate"], baseline["false_positive_components_mean"]),
              "- この比較規則での最良: `{}`（検出率 {:.3f}、誤検出成分 {:.1f}/画像、点状画素再現率 {:.3f}）。".format(best["condition"], best["point_box_hit_rate"], best["false_positive_components_mean"], best["point_pixel_recall_mean"]),
              "- プール正常面FPR 1%以下の共通しきい値で最良の行: `{}`、t={:.0f}、ボックス検出率 {:.3f}、点状画素再現率 {:.3f}、FPR {:.4f}。".format(best_common["condition"], best_common["threshold"], best_common["point_box_hit_rate"], best_common["point_pixel_recall_mean"], best_common["normal_false_positive_rate_pooled"]),
              "- Aでは15/45から21/63へ拡大すると、9点全てを維持したまま誤検出成分が減った。一方、25/75では誤検出が増え、31/93以降は点状異常の見逃しが出た。ブロックを大きくすれば一様に改善する結果ではない。",
              "- Bでは、周辺を45から123へ広げると（周辺除外なし）誤検出成分は減少傾向で、点状画素再現率は上がった。ただし改善は63以降で小さく、既知領域を除外した周辺平均は画素再現率を高める一方、選択されたしきい値では誤検出成分も増えた。",
              "- 共通しきい値では、FPR 1%以下で9点全てを検出する条件はなかった。全点検出に必要な最小のプールFPRは約2.1%であり、画像別しきい値の結果を固定しきい値へそのまま一般化できない。",
              "- 点状異常は9個だけで大きさの層別比較には不足する。今回確認できるのは、中心31 px以上では少なくとも一部の点状異常が平均化されて見逃される、という範囲までである。",
              "", "## 読み方", "- `center_size_comparison.png` と `outer_size_comparison.png` は、画像別1% FPR制約下のサイズ効果を分けて示す。",
              "- `*_diagnostic.png` は、正解ボックス、Dマップ、正常/点状スコア分布、候補輪郭を同一条件で示す。",
              "- `score_summary.csv` には周辺除外時の有効リング画素率を保存した。既知領域が多く有効率が低い場合は、周辺平均の信頼性が下がる。",
              "- 1画素でもボックスに反応すれば成功となる検出率は緩い。点状画素再現率と候補輪郭も併せて判断する。"]
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"conditions={len(SPECS)}, threshold rows={len(threshold_data)}")


if __name__ == "__main__":
    main()
