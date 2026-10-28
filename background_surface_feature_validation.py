"""Validate simple brightness-histogram rules for background vs. surface.

This analysis keeps LabelMe source data unchanged.  It uses 7 x 7 blocks and
image-wise leave-one-image-out (LOIO) validation: blocks from a test image
never set that image's threshold.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from background_surface_hist_experiment import DEFAULT_INPUT, image_number, write_csv
from occupancy_histogram_relation_analysis import block_features, save


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "background_surface_feature_validation"
BASE_FEATURES = ("mean", "std", "hist_peak_probability", "mode_brightness", "iqr", "entropy")
ADAPTIVE_FEATURES = BASE_FEATURES + ("mean_minus_image_mean", "mode_minus_image_median")
PAIR_FEATURES = (("std", "hist_peak_probability"), ("std", "entropy"),
                 ("iqr", "hist_peak_probability"), ("iqr", "entropy"))
LIGHT_OFF = {"image1", "image19", "image20", "image21"}
COLORS = {"background": "#1676c0", "surface": "#f28e2b", "mixed": "#9e9e9e"}


def endpoint_label(row: dict, purity: float) -> str:
    """Tags/damage are excluded; only pure background or pure surface is scored."""
    if row["tag_percent"] > 0 or row["damage_percent"] > 0:
        return "mixed"
    if row["background_percent"] >= purity:
        return "background"
    if row["surface_percent"] >= purity:
        return "surface"
    return "mixed"


def add_image_relative_features(rows: list[dict]) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["image"]].append(row)
    for image_rows in groups.values():
        image_mean = float(np.mean([r["mean"] for r in image_rows]))
        image_mode_median = float(np.median([r["mode_brightness"] for r in image_rows]))
        for row in image_rows:
            row["mean_minus_image_mean"] = row["mean"] - image_mean
            row["mode_minus_image_median"] = row["mode_brightness"] - image_mode_median


def prediction_metrics(pred_background: np.ndarray, labels: np.ndarray) -> dict:
    bg = labels == "background"
    surface = labels == "surface"
    tp = int(np.sum(pred_background & bg)); fp = int(np.sum(pred_background & surface))
    fn = int(np.sum(~pred_background & bg)); tn = int(np.sum(~pred_background & surface))
    precision_bg = tp / max(1, tp + fp); recall_bg = tp / max(1, tp + fn)
    precision_surface = tn / max(1, tn + fn); recall_surface = tn / max(1, tn + fp)
    f1_bg = 2 * precision_bg * recall_bg / max(1e-12, precision_bg + recall_bg)
    f1_surface = 2 * precision_surface * recall_surface / max(1e-12, precision_surface + recall_surface)
    return {
        "accuracy": (tp + tn) / max(1, len(labels)),
        "background_precision": precision_bg, "background_recall": recall_bg, "background_f1": f1_bg,
        "surface_precision": precision_surface, "surface_recall": recall_surface, "surface_f1": f1_surface,
        "macro_f1": (f1_bg + f1_surface) / 2,
        "tp_background": tp, "tn_surface": tn, "fp_surface_as_background": fp,
        "fn_background_as_surface": fn,
    }


def threshold_candidates(values: np.ndarray) -> np.ndarray:
    unique = np.unique(values)
    if len(unique) <= 41:
        return unique
    return np.unique(np.quantile(values, np.linspace(.01, .99, 41)))


def fit_single(train: list[dict], feature: str) -> tuple[float, str, dict]:
    values = np.asarray([row[feature] for row in train], float)
    labels = np.asarray([row["true_label"] for row in train])
    best = None
    for threshold in threshold_candidates(values):
        for direction in ("high", "low"):
            pred = values >= threshold if direction == "high" else values < threshold
            metrics = prediction_metrics(pred, labels)
            candidate = (metrics["macro_f1"], metrics["accuracy"], threshold, direction, metrics)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    assert best is not None
    return float(best[2]), best[3], best[4]


def apply_single(rows: list[dict], feature: str, threshold: float, direction: str) -> np.ndarray:
    value = np.asarray([row[feature] for row in rows], float)
    return value >= threshold if direction == "high" else value < threshold


def fit_pair(train: list[dict], first: str, second: str, operator: str) -> tuple[float, str, float, str, dict]:
    a = np.asarray([row[first] for row in train], float); b = np.asarray([row[second] for row in train], float)
    labels = np.asarray([row["true_label"] for row in train]); best = None
    # A 2-D exhaustive search across every observed value is needlessly large
    # for this explanatory experiment.  Eleven training-data quantiles retain
    # coarse threshold placement while keeping every LOIO fold reproducible.
    thresholds_a = np.unique(np.quantile(a, np.linspace(.02, .98, 11)))
    thresholds_b = np.unique(np.quantile(b, np.linspace(.02, .98, 11)))
    for ta in thresholds_a:
        for tb in thresholds_b:
            for da in ("high", "low"):
                pa = a >= ta if da == "high" else a < ta
                for db in ("high", "low"):
                    pb = b >= tb if db == "high" else b < tb
                    pred = pa & pb if operator == "AND" else pa | pb
                    metrics = prediction_metrics(pred, labels)
                    candidate = (metrics["macro_f1"], metrics["accuracy"], ta, da, tb, db, metrics)
                    if best is None or candidate[:2] > best[:2]:
                        best = candidate
    assert best is not None
    return float(best[2]), best[3], float(best[4]), best[5], best[6]


def apply_pair(rows: list[dict], first: str, second: str, operator: str,
               ta: float, da: str, tb: float, db: str) -> np.ndarray:
    a = apply_single(rows, first, ta, da); b = apply_single(rows, second, tb, db)
    return a & b if operator == "AND" else a | b


def loio(rows: list[dict], purity: float, mode: str) -> tuple[list[dict], dict[tuple[str, str], np.ndarray]]:
    """Fit on endpoint blocks from other images, return per-image predictions."""
    endpoints = [row for row in rows if row["true_label"] in {"background", "surface"}]
    images = sorted({row["image"] for row in rows}, key=lambda value: image_number(Path(value + ".png")))
    specs: list[tuple[str, tuple]] = [("single", (feature,)) for feature in
                                      (BASE_FEATURES if mode == "common" else ADAPTIVE_FEATURES)]
    # Two-feature rules are evaluated for the common rule.  The adaptive
    # comparison is intentionally limited to image-normalised single features.
    if mode == "common":
        specs += [("pair", (first, second, operator)) for first, second in PAIR_FEATURES for operator in ("AND", "OR")]
    results: list[dict] = []; predictions: dict[tuple[str, str], np.ndarray] = {}
    for test_image in images:
        test_all = [row for row in rows if row["image"] == test_image]
        test_endpoints = [row for row in test_all if row["true_label"] in {"background", "surface"}]
        train = [row for row in endpoints if row["image"] != test_image]
        for kind, parameters in specs:
            if kind == "single":
                feature, = parameters
                threshold, direction, train_metrics = fit_single(train, feature)
                pred_all = apply_single(test_all, feature, threshold, direction)
                rule_name = feature
                result = {"rule_type": "single", "feature_1": feature, "feature_2": "", "operator": "",
                          "threshold_1": threshold, "direction_1": direction, "threshold_2": np.nan, "direction_2": ""}
            else:
                first, second, operator = parameters
                ta, da, tb, db, train_metrics = fit_pair(train, first, second, operator)
                pred_all = apply_pair(test_all, first, second, operator, ta, da, tb, db)
                rule_name = f"{first} {operator} {second}"
                result = {"rule_type": "pair", "feature_1": first, "feature_2": second, "operator": operator,
                          "threshold_1": ta, "direction_1": da, "threshold_2": tb, "direction_2": db}
            pred_endpoint = pred_all[[i for i, row in enumerate(test_all) if row["true_label"] in {"background", "surface"}]]
            metrics = prediction_metrics(pred_endpoint, np.asarray([row["true_label"] for row in test_endpoints]))
            results.append({"scope": mode, "purity_threshold_percent": purity, "test_image": test_image,
                            "lighting": test_all[0]["lighting"], "rule": rule_name,
                            "train_endpoint_blocks": len(train), "test_endpoint_blocks": len(test_endpoints),
                            "train_macro_f1": train_metrics["macro_f1"], **result, **metrics})
            predictions[(rule_name, test_image)] = pred_all
    return results, predictions


def feature_summary(rows: list[dict]) -> list[dict]:
    output = []
    for lighting in ("all", "on", "off"):
        subset = [row for row in rows if row["true_label"] in {"background", "surface"} and
                  (lighting == "all" or row["lighting"] == lighting)]
        for feature in BASE_FEATURES:
            values = {label: np.asarray([row[feature] for row in subset if row["true_label"] == label], float)
                      for label in ("background", "surface")}
            if min(map(len, values.values())) == 0:
                continue
            pooled = np.sqrt((values["background"].var(ddof=1) + values["surface"].var(ddof=1)) / 2)
            output.append({"lighting": lighting, "feature": feature,
                           "background_n": len(values["background"]), "surface_n": len(values["surface"]),
                           "background_mean": values["background"].mean(), "background_median": np.median(values["background"]),
                           "background_std": values["background"].std(ddof=1), "surface_mean": values["surface"].mean(),
                           "surface_median": np.median(values["surface"]), "surface_std": values["surface"].std(ddof=1),
                           "cohens_d_background_minus_surface": (values["background"].mean() - values["surface"].mean()) / pooled if pooled else np.nan})
    return output


def distribution_figures(rows: list[dict], out: Path) -> None:
    endpoint = [row for row in rows if row["true_label"] in {"background", "surface"}]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, feature in zip(axes.ravel(), BASE_FEATURES):
        for label in ("background", "surface"):
            values = [row[feature] for row in endpoint if row["true_label"] == label]
            ax.hist(values, bins=25, density=True, histtype="step", lw=1.8, color=COLORS[label], label=label)
        ax.set(title=feature, xlabel=feature, ylabel="density"); ax.legend(); ax.grid(alpha=.2)
    fig.suptitle("Pure blocks only: background ≥90% vs surface ≥90% (tag/damage excluded)")
    save(fig, out / "feature_histograms.png")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, feature in zip(axes.ravel(), BASE_FEATURES):
        groups = [[row[feature] for row in endpoint if row["true_label"] == label] for label in ("background", "surface")]
        ax.boxplot(groups, tick_labels=("background", "surface"), patch_artist=True,
                   boxprops={"facecolor": "#dddddd"})
        ax.set(title=feature, ylabel=feature); ax.grid(axis="y", alpha=.2)
    fig.suptitle("Pure-block feature distributions")
    save(fig, out / "feature_boxplots.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, x, y in zip(axes, ("peak_concentration", "iqr"), ("entropy", "hist_peak_probability")):
        for label in ("background", "surface"):
            sample = [row for row in endpoint if row["true_label"] == label]
            ax.scatter([row[x] for row in sample], [row[y] for row in sample], s=18, color=COLORS[label], alpha=.75, label=label)
        ax.set(title=f"{x} × {y}", xlabel=x, ylabel=y); ax.legend(); ax.grid(alpha=.2)
    save(fig, out / "feature_scatter.png")


def classification_figure(image_rows: list[dict], pred: np.ndarray, out: Path) -> None:
    image_path = DEFAULT_INPUT / f"{image_rows[0]['image']}.png"
    gray = plt.imread(str(image_path))
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.4), constrained_layout=True)
    axes[0].imshow(gray, cmap="gray"); axes[0].set(title="Original")
    for ax, title in zip(axes[1:], ("Ground truth block class", "LOIO automatic result", "Classification result")):
        ax.imshow(gray, cmap="gray"); ax.set_title(title)
    for row, value in zip(image_rows, pred):
        rect = (row["x0"], row["y0"], row["x1"] - row["x0"], row["y1"] - row["y0"])
        truth = row["true_label"]
        axes[1].add_patch(Rectangle(rect[:2], rect[2], rect[3], facecolor=COLORS[truth], edgecolor="white", alpha=.42, lw=.6))
        predicted = "background" if value else "surface"
        axes[2].add_patch(Rectangle(rect[:2], rect[2], rect[3], facecolor=COLORS[predicted], edgecolor="white", alpha=.42, lw=.6,
                                    linestyle="--" if truth == "mixed" else "-"))
        if truth == "mixed":
            edge, label = "#777777", "mixed / excluded"
        elif predicted == truth:
            edge, label = "#37a148", "correct"
        elif truth == "background":
            edge, label = "#d62728", "background→surface"
        else:
            edge, label = "#c23b91", "surface→background"
        axes[3].add_patch(Rectangle(rect[:2], rect[2], rect[3], fill=False, edgecolor=edge, lw=2.0,
                                    linestyle="--" if truth == "mixed" else "-"))
        axes[3].text(row["x0"] + 3, row["y0"] + 14, row["block_id"], color=edge, fontsize=7,
                     bbox={"facecolor": "white", "alpha": .65, "pad": .4, "edgecolor": "none"})
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Blue=background, orange=surface, gray/dashed=mixed or excluded; right: green=correct, red=background→surface, magenta=surface→background")
    fig.savefig(out / "classification_overlay.png", dpi=110, bbox_inches="tight"); plt.close(fig)


def heatmap_figure(image_rows: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for ax, feature in zip(axes.ravel(), BASE_FEATURES):
        matrix = np.asarray([row[feature] for row in image_rows]).reshape(7, 7)
        im = ax.imshow(matrix, cmap="viridis")
        for row in range(7):
            for col in range(7):
                ax.text(col, row, f"{matrix[row, col]:.2g}", ha="center", va="center", fontsize=6, color="white")
        ax.set(title=feature, xticks=range(7), yticks=range(7), xlabel="block column", ylabel="block row")
        fig.colorbar(im, ax=ax, shrink=.75)
    fig.suptitle(f"{image_rows[0]['image']}: per-block luminance features")
    fig.savefig(out / "feature_heatmaps.png", dpi=110, bbox_inches="tight"); plt.close(fig)


def report(rows: list[dict], summary: list[dict], result_rows: list[dict], selected: dict, out: Path, purity: float) -> None:
    selected_rows = [row for row in result_rows if row["scope"] == "common" and row["rule"] == selected["rule"]]
    overall = prediction_metrics(
        np.asarray([row["predicted_background"] for row in selected["prediction_rows"]]),
        np.asarray([row["true_label"] for row in selected["prediction_rows"]]),
    )
    best_features = sorted([row for row in summary if row["lighting"] == "all"],
                           key=lambda row: abs(row["cohens_d_background_minus_surface"]), reverse=True)
    lines = ["# 背景・表面の輝度特徴量による単純判別", "", "## 1. 実験目的",
             "7×7ブロックの輝度ヒストグラム由来の基本特徴量だけで、背景と検査対象表面を分けられるかを検証した。FFT・クラスタリング・機械学習は使っていない。",
             "", "## 2. 正解ラベルと評価対象",
             f"- 全21画像、計{len(rows)}ブロック。", f"- 純粋背景：background占有率≥{purity:.0f}%、純粋表面：surface占有率≥{purity:.0f}%。",
             "- tagまたはdamageを1画素でも含むブロックは純粋比較・精度評価から除外した。その他は混在として除外した。",
             "", "## 3. 特徴量", "平均輝度、標準偏差、正規化ヒストグラムの最大ピーク確率、最頻輝度、IQR、Shannonエントロピーを使用した。",
             "", "## 4. 分離の大きい特徴量（全画像統合、Cohen's dの絶対値順）", "| 特徴量 | 背景平均 | 表面平均 | Cohen's d |", "|---|---:|---:|---:|"]
    lines += [f"| {row['feature']} | {row['background_mean']:.4g} | {row['surface_mean']:.4g} | {row['cohens_d_background_minus_surface']:.3f} |" for row in best_features]
    lines += ["", "## 5. 画像単位Leave-One-Image-Out検証",
              "各テスト画像の閾値は、残り20画像の純粋ブロックだけから決定した。テスト画像のアノテーションは閾値調整に使っていない。",
              f"- 探索的に最も高い平均macro F1となった共通ルール：`{selected['rule']}`。",
              f"- Accuracy={overall['accuracy']:.3f}, background F1={overall['background_f1']:.3f}, surface F1={overall['surface_f1']:.3f}, macro F1={overall['macro_f1']:.3f}。",
              "- これは同じ21画像群で複数ルールを比較して選んだ探索的な結果であり、独立した最終性能ではない。",
              "", "## 6. 誤分類と限界",
              "画像ごとの結果は `per_image/<image>/classification_overview.png`、誤分類の内訳は `block_predictions.csv` に記録した。明るさ・照明むら・混在境界によって分布が変わるため、単一の全画像共通閾値が安定しない場合は、画像全体平均との差分などの適応特徴を比較した。",
              "", "## 7. 出力", "- `block_occupancy_features.csv`：占有率・全特徴量・純粋/混在ラベル", "- `feature_summary.csv`：背景・表面の統計量と分離量", "- `loio_rule_results.csv`：ルール別・テスト画像別のLOIO評価", "- `block_predictions.csv`：最終選択ルールの全ブロック予測", "- `feature_histograms.png`、`feature_boxplots.png`、`feature_scatter.png`", "- `per_image/`：21画像の分類図・特徴量ヒートマップ"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_clean_japanese_report(summary, results, prediction_rows, selected_rule, out, purity):
    """Write an UTF-8 Japanese report without relying on terminal code pages."""
    import base64
    text = base64.b64decode("IyDog4zmma/jg7vooajpnaLjga7ovJ3luqbnibnlvrTph4/jgavjgojjgovljZjntJTliKTliKUKCiMjIOWun+mok+ebrueahAo3w5c344OW44Ot44OD44Kv44Gu5bmz5Z2H6Lyd5bqm44CB5qiZ5rqW5YGP5beu44CB5pyA5aSn44OU44O844Kv56K6546H44CB5pyA6aC76Lyd5bqm44CBSVFS44CB44Ko44Oz44OI44Ot44OU44O844Gg44GR44Gn44CB6IOM5pmv44Go5qSc5p+75a++6LGh6KGo6Z2i44KS5YiG44GR44KJ44KM44KL44GL44KS5qSc6Ki844GX44Gf44CCRkZU44O75qmf5qKw5a2m57+S44Gv5L2/55So44GX44Gm44GE44Gq44GE44CCCgojIyDjg4fjg7zjgr/jgajoqZXkvqHln7rmupYK5YWoMjHnlLvlg4/jgIExLDAyOeODluODreODg+OCr+OCkuS9v+eUqOOBl+OBn+OAgue0lOeyi+iDjOaZr+OBr2JhY2tncm91bmTljaDmnInnjoc5MCXku6XkuIrjgIHntJTnsovooajpnaLjga9zdXJmYWNl5Y2g5pyJ546HOTAl5Lul5LiK44Go44GX44Gf44CCdGFn44G+44Gf44GvZGFtYWdl44KSMeeUu+e0oOOBp+OCguWQq+OCgOODluODreODg+OCr+OBr+avlOi8g+ODu+ipleS+oeOBi+OCiemZpOWkluOBl+OAgeOBneOBruS7luOBr+a3t+WcqOODluODreODg+OCr+OBqOOBl+OBpuWPr+imluWMluOBruOBv+OCkuihjOOBo+OBn+OAggoKIyMg54m55b606YeP44Gu5YiG6Zui5bqmCnwg54m55b606YePIHwg6IOM5pmv5bmz5Z2HIHwg6KGo6Z2i5bmz5Z2HIHwgQ29oZW4ncyBk77yI6IOM5pmv4oiS6KGo6Z2i77yJIHwKfC0tLXwtLS06fC0tLTp8LS0tOnwKX19GRUFUVVJFX1JPV1NfXwoK57Wx5ZCI5YiG5biD44Gn5beu44GM5aSn44GN44GE54m55b606YeP44Go44CB55S75YOP44KS44G+44Gf44GE44Gn6Za+5YCk44Go44GX44Gm5qmf6IO944GZ44KL54m55b606YeP44Gv5b+F44Ga44GX44KC5ZCM44GY44Gn44Gv44Gq44GE44CCCgojIyBMZWF2ZS1PbmUtSW1hZ2UtT3V05qSc6Ki8CuODhuOCueODiOeUu+WDj+OBrumWvuWApOOBr+aui+OCijIw55S75YOP44Gu57SU57KL44OW44Ot44OD44Kv44Gg44GR44GL44KJ5rG644KB44CB44OG44K544OI55S75YOP44Gu44Ki44OO44OG44O844K344On44Oz44Gv6Za+5YCk6Kq/5pW044Gr5L2/55So44GX44Gm44GE44Gq44GE44CCCgotIOaOoue0oueahOOBq+mBuOOBsOOCjOOBn+WFqOeUu+WDj+WFsemAmuODq+ODvOODq++8ml9fUlVMRV9fCi0g5YWo57SU57KL44OW44Ot44OD44Kv44KS44G+44Go44KB44Gf6KmV5L6h77yaQWNjdXJhY3k9X19BQ0NfX+OAgeiDjOaZr0YxPV9fQkdGMV9f44CB6KGo6Z2iRjE9X19TVVJGRjFfX+OAgW1hY3JvIEYxPV9fTUFDUk9fX+OAggotIOiDjOaZr++8mumBqeWQiOeOhz1fX0JHUFJFQ19f44CB5YaN54++546HPV9fQkdSRUNfX+OAguiDjOaZr+OCkuihqOmdouOBqOOBl+OBn+imi+mAg+OBl+OBr19fRk5fX+ODluODreODg+OCr+OAggotIOagvOWtkOODqeOCpOODiOOBguOCiu+8mm1hY3JvIEYxPV9fT05NQUNST19f44CC5qC85a2Q44Op44Kk44OI44Gq44GX77yabWFjcm8gRjE9X19PRkZNQUNST19f44CCCi0g55S75YOP5YilbWFjcm8gRjHjga7ljZjntJTlubPlnYfjga9fX0ZPTERNQUNST19f44CC6IOM5pmv44Gu57SU57KL44OW44Ot44OD44Kv44GM5a2Y5Zyo44GX44Gq44GE55S75YOP44Gn44Gv6IOM5pmvRjHjgpIw44Go44GZ44KL44Gf44KB44CB44OX44O844Or44GX44Gf5YWo5L2T5YCk44Go44Gv55Ww44Gq44KL44CCCgojIyDntZDoq5YK44OS44K544OI44Kw44Op44Og5b2i54q244Gr44Gv6IOM5pmv44O76KGo6Z2i44Gu5beu44GM44GC44KL44CC44GX44GL44GX55S75YOP44GU44Go44Gu54Wn5piO44O75piO44KL44GV44O76IOM5pmv5qeL5oiQ44Gu5beu44GM5aSn44GN44GP44CB5Y2Y5LiA44Gu5YWo55S75YOP5YWx6YCa6Za+5YCk44Gg44GR44Gn6IOM5pmv44KS5Y+W44KK5YiH44KL44GT44Go44Gv6Zuj44GX44GE44CC6YG45oqe44Or44O844Or44Gv6KGo6Z2i44Gu5L+d5oyB44Gr44Gv5q+U6LyD55qE5by344GE5LiA5pa544CB6IOM5pmv44Gu6KaL6YCD44GX44GM5q6L44KL44CC55S75YOP5Y2Y5L2N44Gu5Z+65rqW6KOc5q2j44CB6KSH5pWw54m55b606YeP44CB44OS44K544OI44Kw44Op44Og5b2i54q26aGe5Ly85bqm44Gv5qyh44Gu5YCZ6KOc44Gn44GC44KL44CCCgrmnKzntZDmnpzjga/lkIzjgZgyMeeUu+WDj+e+pOOBp+ikh+aVsOODq+ODvOODq+OCkuavlOi8g+OBl+OBn+aOoue0oueahOipleS+oeOBp+OBguOCi+OAguacquefpeeUu+WDj+OBp+OBr+WbuuWumuODq+ODvOODq+OCkuWIpeeUu+WDj+e+pOOBp+aknOiovOOBmeOCi+W/heimgeOBjOOBguOCi+OAggoKIyMg5Ye65YqbCi0gYmxvY2tfb2NjdXBhbmN5X2ZlYXR1cmVzLmNzdu+8muWNoOacieeOh+OAgeeJueW+tOmHj+OAgee0lOeyi++8j+a3t+WcqOODqeODmeODqwotIGZlYXR1cmVfc3VtbWFyeS5jc3bvvJrog4zmma/jg7vooajpnaLjga7ntbHoqIjph4/jgahDb2hlbidzIGQKLSBsb2lvX3J1bGVfcmVzdWx0cy5jc3bjgIFydWxlX3N1bW1hcnkuY3N277ya44Or44O844Or5Yil44GuTE9JT+ipleS+oQotIGNsYXNzaWZpY2F0aW9uX21ldHJpY3MuY3N244CBY29uZnVzaW9uX21hdHJpeC5wbmfvvJrlhajkvZPjg7vjg6njgqTjg4jmnaHku7bjg7vnlLvlg4/liKXjga7oqZXkvqEKLSBibG9ja19wcmVkaWN0aW9ucy5jc3bvvJrmnIDntYLjg6vjg7zjg6vjgavjgojjgovlhajjg5bjg63jg4Pjgq/mjqjlrpoKLSBwZXJfaW1hZ2UvPGltYWdlPi9jbGFzc2lmaWNhdGlvbl9vdmVybGF5LnBuZ++8muWOn+eUu+WDj+OAgeato+ino+OAgeaOqOWumuOAgeiqpOWIhumhngotIHBlcl9pbWFnZS88aW1hZ2U+L2ZlYXR1cmVfaGVhdG1hcHMucG5n77yaNueJueW+tOmHj+OBruODkuODvOODiOODnuODg+ODlwo=").decode("utf-8")
    valid = [r for r in prediction_rows if r["evaluated"]]
    overall = prediction_metrics(np.asarray([r["predicted_background"] for r in valid]),
                                 np.asarray([r["true_label"] for r in valid]))
    by_light = {}
    for lighting in ("on", "off"):
        part = [r for r in valid if r["lighting"] == lighting]
        by_light[lighting] = prediction_metrics(np.asarray([r["predicted_background"] for r in part]),
                                                np.asarray([r["true_label"] for r in part]))
    ranked = sorted([r for r in summary if r["lighting"] == "all"],
                    key=lambda r: abs(r["cohens_d_background_minus_surface"]), reverse=True)
    feature_rows = "\n".join(f"| {r['feature']} | {r['background_mean']:.4g} | {r['surface_mean']:.4g} | {r['cohens_d_background_minus_surface']:.3f} |" for r in ranked)
    folds = [r for r in results if r["scope"] == "common" and r["rule"] == selected_rule]
    values = {"__FEATURE_ROWS__": feature_rows, "__RULE__": selected_rule, "__PURITY__": f"{purity:.0f}",
              "__ACC__": f"{overall['accuracy']:.3f}", "__BGF1__": f"{overall['background_f1']:.3f}",
              "__SURFF1__": f"{overall['surface_f1']:.3f}", "__MACRO__": f"{overall['macro_f1']:.3f}",
              "__BGPREC__": f"{overall['background_precision']:.3f}", "__BGREC__": f"{overall['background_recall']:.3f}",
              "__FN__": str(overall["fn_background_as_surface"]), "__ONMACRO__": f"{by_light['on']['macro_f1']:.3f}",
              "__OFFMACRO__": f"{by_light['off']['macro_f1']:.3f}", "__FOLDMACRO__": f"{np.mean([r['macro_f1'] for r in folds]):.3f}"}
    for old, new in values.items():
        text = text.replace(old, new)
    (out / "report.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--purity", type=float, default=90.0, help="pure endpoint occupancy percentage")
    args = parser.parse_args()
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    purity = float(args.purity)
    paths = sorted(args.input_dir.glob("image[0-9]*.png"), key=image_number)
    rows: list[dict] = []
    for path in paths:
        image_rows, _hists, _masks = block_features(path, rows=7, cols=7)
        rows.extend(image_rows)
    add_image_relative_features(rows)
    for row in rows:
        row["true_label"] = endpoint_label(row, purity)
    write_csv(out / "block_occupancy_features.csv", rows)
    summary = feature_summary(rows); write_csv(out / "feature_summary.csv", summary)
    distribution_figures(rows, out)
    common_results, common_predictions = loio(rows, purity, "common")
    adaptive_results, _adaptive_predictions = loio(rows, purity, "adaptive")
    all_results = common_results + adaptive_results
    write_csv(out / "loio_rule_results.csv", all_results)
    # Aggregate each candidate rule over held-out image folds.
    aggregate = []
    for scope in ("common", "adaptive"):
        for rule in sorted({row["rule"] for row in all_results if row["scope"] == scope}):
            part = [row for row in all_results if row["scope"] == scope and row["rule"] == rule]
            aggregate.append({"scope": scope, "rule": rule, "folds": len(part),
                              **{key: float(np.mean([row[key] for row in part])) for key in
                                 ("accuracy", "background_precision", "background_recall", "background_f1", "surface_precision", "surface_recall", "surface_f1", "macro_f1")}})
    write_csv(out / "rule_summary.csv", aggregate)
    common_aggregate = [row for row in aggregate if row["scope"] == "common"]
    selected_rule = max(common_aggregate, key=lambda row: (row["macro_f1"], row["accuracy"]))["rule"]
    prediction_rows: list[dict] = []
    for image in sorted({row["image"] for row in rows}, key=lambda value: image_number(Path(value + ".png"))):
        image_rows = [row for row in rows if row["image"] == image]
        pred = common_predictions[(selected_rule, image)]
        per = out / "per_image" / image; per.mkdir(parents=True, exist_ok=True)
        classification_figure(image_rows, pred, per)
        heatmap_figure(image_rows, per)
        for row, predicted in zip(image_rows, pred):
            prediction_rows.append({**row, "selected_rule": selected_rule,
                                    "predicted_label": "background" if predicted else "surface",
                                    "predicted_background": bool(predicted),
                                    "evaluated": row["true_label"] in {"background", "surface"}})
    write_csv(out / "block_predictions.csv", prediction_rows)
    selected = {"rule": selected_rule, "prediction_rows": [row for row in prediction_rows if row["evaluated"]]}
    write_clean_japanese_report(summary, all_results, prediction_rows, selected_rule, out, purity)
    print(f"completed {len(paths)} images, {len(rows)} blocks; selected common rule: {selected_rule}")


if __name__ == "__main__":
    main()
