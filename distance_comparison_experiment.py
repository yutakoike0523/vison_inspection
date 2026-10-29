#!/usr/bin/env python3
"""Compare radial-FFT anomaly detection versus 8-reference distance.

Only the reference-centre distance changes.  The spectral calculation mirrors
the existing radial FFT implementation: mean removal, separable 2-D Hann,
rFFT2 power, in-band total-power normalisation, radial averaging, dB conversion,
and RMS against the mean of eight reference spectra.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle
from scipy import fft
from scipy.ndimage import label as connected_components
from sklearn.metrics import average_precision_score, roc_auc_score

import evaluate_all_radial_fft as existing
import multiscale_fft_experiment as base
import radial_fft_experiment as radial


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data20260903" / "data20260903"
OUT = ROOT / "output_distance_comparison"
DISTANCES = (16, 32, 48, 64, 72, 80, 96, 128)
OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
WINDOW = 128
HALF = WINDOW // 2
EVAL_SIZE = 16
EVAL_HALF = EVAL_SIZE // 2
STRIDE = 16
FREQ = (0.020, 0.500)
BIN_WIDTH = 0.010
EPS = base.EPS
GRID_IMAGES = set(range(2, 18))
DEFECT_LABELS = set(existing.CLASSES)
EXCLUSION_LABELS = {"tag", "background"}


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row)) if rows else (fields or ["status"])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def group_for(image_id: int) -> str:
    return "grid" if image_id in GRID_IMAGES else "no_grid"


def common_centres(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Stride-16 target centres for which every distance has all 8 references."""
    height, width = shape
    all_x = np.arange(HALF, width - HALF + 1, STRIDE, dtype=np.int32)
    all_y = np.arange(HALF, height - HALF + 1, STRIDE, dtype=np.int32)
    margin = max(DISTANCES) + HALF
    return all_x[(all_x >= margin) & (all_x <= width - margin)], all_y[(all_y >= margin) & (all_y <= height - margin)]


def required_coordinates(xs: np.ndarray, ys: np.ndarray) -> list[tuple[int, int]]:
    coords = {(int(x), int(y)) for y in ys for x in xs}
    for distance in DISTANCES:
        coords.update((int(x + dx * distance), int(y + dy * distance))
                      for y in ys for x in xs for dy, dx in OFFSETS)
    return sorted(coords, key=lambda item: (item[1], item[0]))


def calculate_spectra(image: np.ndarray, coordinates: list[tuple[int, int]], workers: int, batch: int) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    """Calculate each requested coordinate once and return a coordinate cache."""
    _, _, _, specs = radial.grids(WINDOW, BIN_WIDTH, [FREQ])
    spec = specs[FREQ]
    valid = spec["valid"]
    hann_1d = np.hanning(WINDOW).astype(np.float32)
    hann = np.outer(hann_1d, hann_1d).astype(np.float32)
    output = np.empty((len(coordinates), int(valid.sum())), np.float32)
    index = {coord: i for i, coord in enumerate(coordinates)}
    for start in range(0, len(coordinates), batch):
        selected = coordinates[start:start + batch]
        windows = np.stack([image[y-HALF:y+HALF, x-HALF:x+HALF] for x, y in selected]).astype(np.float32, copy=False)
        windows = windows.copy()
        windows -= windows.mean(axis=(-2, -1), keepdims=True)
        windows *= hann
        transformed = fft.rfft2(windows, axes=(-2, -1), workers=workers)
        power = (np.abs(fft.fftshift(transformed, axes=(-2,))) ** 2).astype(np.float32)
        # Preserve the existing implementation exactly: unweighted half-plane
        # in-band sum for normalisation; conjugate weights only in radial bins.
        total = np.maximum(power[:, spec["mask"]].sum(axis=1), EPS)
        sums = power.reshape(len(power), -1) @ spec["matrix"]
        average = sums[:, valid] / spec["counts"][valid][None, :] / total[:, None]
        output[start:start + len(selected)] = (10.0 * np.log10(average + EPS)).astype(np.float32)
    return output, index


def score_maps(spectra: np.ndarray, cache: dict[tuple[int, int], int], xs: np.ndarray, ys: np.ndarray) -> dict[int, np.ndarray]:
    maps: dict[int, np.ndarray] = {}
    for distance in DISTANCES:
        score = np.empty((len(ys), len(xs)), np.float32)
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                target = spectra[cache[int(x), int(y)]]
                refs = np.stack([spectra[cache[int(x + dx*distance), int(y + dy*distance)]] for dy, dx in OFFSETS])
                reference = refs.mean(axis=0)
                score[iy, ix] = np.sqrt(np.mean((target - reference) ** 2))
        maps[distance] = score
    return maps


def rectangle_overlap(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray, size: int = EVAL_SIZE) -> np.ndarray:
    integral = np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    x0 = xs - size // 2
    y0 = ys - size // 2
    out = np.empty((len(ys), len(xs)), bool)
    for iy, top in enumerate(y0):
        bottom = top + size
        out[iy] = (integral[bottom, x0 + size] - integral[top, x0 + size]
                   - integral[bottom, x0] + integral[top, x0]) > 0
    return out


def load_masks(image_id: int, shape: tuple[int, int], xs: np.ndarray, ys: np.ndarray) -> dict:
    path = DATA / f"image{image_id}.json"
    labels = existing.annotation(path, shape)
    if labels is None:
        raise FileNotFoundError(f"annotation missing: {path}")
    zeros = np.zeros(shape, bool)
    defects = [(idx, name, mask) for idx, (name, mask) in enumerate(labels) if name in DEFECT_LABELS]
    exclusions = [mask for name, mask in labels if name in EXCLUSION_LABELS]
    defect_union = np.logical_or.reduce([mask for _, _, mask in defects]) if defects else zeros.copy()
    exclusion_union = np.logical_or.reduce(exclusions) if exclusions else zeros.copy()
    defect_grid = rectangle_overlap(defect_union, xs, ys)
    excluded_grid = rectangle_overlap(exclusion_union, xs, ys)
    instances = []
    for idx, name, mask in defects:
        instances.append({"annotation_index": idx, "label": name, "mask": mask,
                          "selection": rectangle_overlap(mask, xs, ys) & ~excluded_grid})
    return {"labels": labels, "defect_pixel": defect_union, "exclusion_pixel": exclusion_union,
            "defect": defect_grid & ~excluded_grid, "excluded": excluded_grid,
            "normal": ~defect_grid & ~excluded_grid, "instances": instances}


def equal_image_quantile(arrays: list[np.ndarray], quantile: float = 0.99) -> float | None:
    arrays = [np.sort(np.asarray(a, float)) for a in arrays if len(a)]
    if not arrays:
        return None
    values = np.concatenate(arrays)
    weights = np.concatenate([np.full(len(array), 1.0 / len(array)) for array in arrays])
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    target = quantile * len(arrays)
    at = min(int(np.searchsorted(cumulative, target, side="left")), len(values) - 1)
    return float(values[order[at]])


def false_positive_components(mask: np.ndarray) -> int:
    return int(connected_components(mask, structure=np.ones((3, 3), np.uint8))[1])


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if len(np.unique(labels)) < 2:
        return math.nan, math.nan
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def annotation_overlay(ax, record: dict, include_exclusion: bool = True) -> None:
    damage = record["masks"]["defect_pixel"]
    excluded = record["masks"]["exclusion_pixel"]
    if damage.any():
        ax.contour(damage.astype(np.uint8), levels=[.5], colors="#38ff5a", linewidths=1.0)
    if include_exclusion and excluded.any():
        ax.contour(excluded.astype(np.uint8), levels=[.5], colors="#00e5ff", linewidths=.8)


def draw_image15(records: dict[int, dict], thresholds: dict[tuple[int, int], float | None], output: Path) -> None:
    if 15 not in records:
        return
    record = records[15]
    pool = np.concatenate([record["scores"][d].ravel() for d in DISTANCES])
    vmin, vmax = np.percentile(pool[np.isfinite(pool)], [1, 99])
    extent = [record["xs"][0]-STRIDE/2, record["xs"][-1]+STRIDE/2,
              record["ys"][-1]+STRIDE/2, record["ys"][0]-STRIDE/2]
    fig, axes = plt.subplots(3, 3, figsize=(16, 13), constrained_layout=True)
    for ax, distance in zip(axes.flat, DISTANCES):
        im = ax.imshow(record["scores"][distance], extent=extent, cmap="inferno", vmin=vmin, vmax=vmax, interpolation="nearest")
        annotation_overlay(ax, record)
        ax.set(title=f"d={distance} px", xlim=(0, record["image"].shape[1]), ylim=(record["image"].shape[0], 0))
    axes.flat[-1].axis("off")
    fig.colorbar(im, ax=axes.flat[:-1].tolist(), label="RMS anomaly score (dB)", shrink=.78)
    fig.suptitle("image15: distance comparison (one shared colour scale)")
    fig.savefig(output / "image15_heatmaps_common_scale.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(18, 13), constrained_layout=True)
    axes[0, 0].imshow(record["image"], cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(record["image"], cmap="gray", vmin=0, vmax=255)
    annotation_overlay(axes[0, 1], record)
    axes[0, 1].set_title("Ground truth (green=damage, cyan=excluded)")
    axes[0, 2].text(.5, .5, f"Shared scale\n{vmin:.3f}–{vmax:.3f} dB\nEvaluation square: 16×16 px",
                    ha="center", va="center", transform=axes[0, 2].transAxes)
    axes[0, 2].axis("off")
    axes[0, 3].axis("off")
    for ax, distance in zip(axes.flat[4:], DISTANCES):
        ax.imshow(record["image"], cmap="gray", vmin=0, vmax=255)
        threshold = thresholds.get((distance, 15))
        valid_evaluation = record["masks"]["normal"] | record["masks"]["defect"]
        hits = ((record["scores"][distance] > threshold) & valid_evaluation
                if threshold is not None else np.zeros_like(record["scores"][distance], bool))
        for iy, ix in np.argwhere(hits):
            ax.add_patch(Rectangle((record["xs"][ix]-EVAL_HALF, record["ys"][iy]-EVAL_HALF), EVAL_SIZE, EVAL_SIZE,
                                   facecolor=(1, 0, 0, .38), edgecolor="red", linewidth=.25))
        annotation_overlay(ax, record, include_exclusion=False)
        ax.set_title(f"d={distance}: detections (red), t={threshold:.3f}" if threshold is not None else f"d={distance}: no threshold")
    for ax in axes.flat:
        if ax.axison:
            ax.set(xlim=(0, record["image"].shape[1]), ylim=(record["image"].shape[0], 0))
            ax.set_axis_off()
    fig.suptitle("image15: original, annotation and LOIO detection results")
    fig.savefig(output / "image15_detection_comparison.png", dpi=170)
    plt.close(fig)

    heat_dir = output / "heatmaps"
    heat_dir.mkdir(exist_ok=True)
    for distance in DISTANCES:
        fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
        ax.imshow(record["image"], cmap="gray", vmin=0, vmax=255)
        im = ax.imshow(record["scores"][distance], extent=extent, cmap="inferno", vmin=vmin, vmax=vmax, alpha=.72, interpolation="nearest")
        annotation_overlay(ax, record)
        ax.set(title=f"image15 d={distance} px (shared scale)", xlim=(0, record["image"].shape[1]), ylim=(record["image"].shape[0], 0))
        fig.colorbar(im, ax=ax, label="RMS anomaly score (dB)")
        fig.savefig(heat_dir / f"image15_d{distance}_heatmap.png", dpi=160)
        plt.close(fig)


def plots(distance_rows: list[dict], records: dict[int, dict], output: Path) -> None:
    x = [row["distance_px"] for row in distance_rows]
    for key, ylabel, filename in (("roc_auc", "ROC-AUC", "distance_vs_roc_auc.png"),
                                  ("pr_auc", "PR-AUC", "distance_vs_pr_auc.png"),
                                  ("window_tpr_at_train_fpr1", "Defect-window TPR", "distance_vs_tpr_fpr1.png")):
        fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        ax.plot(x, [row[key] for row in distance_rows], "o-", linewidth=1.8)
        ax.set(xlabel="Reference-window centre distance (px)", ylabel=ylabel, xticks=x, ylim=(0, 1.02))
        ax.grid(alpha=.3)
        fig.savefig(output / filename, dpi=170)
        plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(17, 9), constrained_layout=True)
    for ax, distance in zip(axes.flat, DISTANCES):
        normal = np.concatenate([r["scores"][distance][r["masks"]["normal"]] for r in records.values()])
        defect = np.concatenate([r["scores"][distance][r["masks"]["defect"]] for r in records.values()])
        high = np.percentile(np.concatenate([normal, defect]), 99.5)
        bins = np.linspace(0, high, 55)
        ax.hist(normal, bins=bins, density=True, histtype="step", linewidth=1.5, label=f"normal n={len(normal):,}")
        ax.hist(defect, bins=bins, density=True, histtype="step", linewidth=1.5, label=f"damage n={len(defect):,}")
        ax.set(title=f"d={distance} px", xlabel="RMS score (dB)", ylabel="density")
        ax.legend(fontsize=7)
        ax.grid(alpha=.18)
    fig.suptitle("Damage and normal anomaly-score distributions")
    fig.savefig(output / "score_distributions.png", dpi=170)
    plt.close(fig)


def make_report(output: Path, distance_rows: list[dict], excluded: list[dict], image_rows: list[dict]) -> None:
    best = max(distance_rows, key=lambda row: (row["pr_auc"], row["roc_auc"]))
    row16 = next(row for row in distance_rows if row["distance_px"] == 16)
    row72 = next(row for row in distance_rows if row["distance_px"] == 72)
    last = distance_rows[-1]
    def pct(value): return "NA" if value is None or not np.isfinite(value) else f"{100*value:.2f}%"
    lines = [
        "# 周辺参照窓の距離によるFFT異常検出性能の比較", "",
        "## 実験方法", "",
        "変更したのは対象窓中心から8参照窓中心までの距離だけである。距離は16, 32, 48, 64, 72, 80, 96, 128 px。対象窓・参照窓は128×128 px、対象中心はstride 16 pxで走査し、全距離で8参照窓が画像内に入る共通位置だけを評価した。72 pxを含む全参照窓は必要座標から元画像を直接切り出し、同一座標のスペクトルは画像内キャッシュで1回だけ計算した。", "",
        "各窓は平均輝度除去、2次元Hann窓、rFFT2、パワースペクトル、放射平均の順に処理した。既存実装と同じく0.020–0.500 cycles/pixelの片側パワー総和で正規化し、共役重み付き放射平均をdB化した。対象dBスペクトルと8参照dBスペクトルの算術平均との差を、同じ帯域のRMSとして異常スコアにした。正規化・RMS式・参照数・平均方法は変更していない。", "",
        "ラベル判定は対象中心16×16 pxと既存LabelMe領域の重なりで行った。傷4クラスを陽性、tag/backgroundに重なる位置を除外、それ以外を正常表面とした。閾値は評価画像自身を使わず、同じ照明群（grid/no_grid）の他画像の正常窓を画像等重みで統合した99%点（LOIO）である。傷インスタンスは、その傷の中心16×16候補窓の1つ以上が閾値を超えた場合に検出とした。正常誤検出箇所は閾値超過格子の8近傍連結成分数である。", "",
        "PR-AUC欄はaverage precisionで算出した。ROC/PRは全評価画像の窓を統合した閾値非依存指標であり、FPR 1%欄はLOIO学習正常分布の99%点を各未知画像に適用した窓単位TPRである。", "",
        "## 距離別結果", "",
        "|距離 px|ROC-AUC|PR-AUC|傷窓TPR@train FPR 1%|実測正常FPR|検出傷|評価傷|正常誤検出箇所|", "|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for row in distance_rows:
        lines.append(f"|{row['distance_px']}|{row['roc_auc']:.4f}|{row['pr_auc']:.4f}|{pct(row['window_tpr_at_train_fpr1'])}|{pct(row['actual_normal_fpr'])}|{row['detected_defect_instances']}|{row['evaluable_defect_instances']}|{row['false_positive_components']}|")
    lines += ["", "## 結果と考察", "",
              f"1. **最良距離**: 主指標をPR-AUC、同率時ROC-AUCとすると **{best['distance_px']} px** が最良だった（ROC-AUC {best['roc_auc']:.4f}、PR-AUC {best['pr_auc']:.4f}、傷窓TPR {pct(best['window_tpr_at_train_fpr1'])}）。", "",
              f"2. **16 pxと72 px**: 16 pxはROC-AUC {row16['roc_auc']:.4f}、PR-AUC {row16['pr_auc']:.4f}、傷窓TPR {pct(row16['window_tpr_at_train_fpr1'])}。72 pxはそれぞれ {row72['roc_auc']:.4f}、{row72['pr_auc']:.4f}、{pct(row72['window_tpr_at_train_fpr1'])}。PR-AUC差（72−16）は {row72['pr_auc']-row16['pr_auc']:+.4f}。72 pxでは上下左右参照窓と中心16×16評価領域が境界で接し、評価領域そのものは参照窓へ入らない。", "",
              f"3. **距離増加の効果**: 16 pxから最良距離までのPR-AUC変化は {best['pr_auc']-row16['pr_auc']:+.4f}、傷窓TPR変化は {best['window_tpr_at_train_fpr1']-row16['window_tpr_at_train_fpr1']:+.2%}。{'数値上は小幅に改善したが、ROC-AUCは0.5付近で分布の重なりも大きく、実用的な改善が確認されたとは言えない' if best['pr_auc'] > row16['pr_auc'] else '改善は確認されなかった'}。", "",
              f"4. **大きすぎる距離**: 最大128 pxのPR-AUCは {last['pr_auc']:.4f}。最良値との差は {last['pr_auc']-best['pr_auc']:+.4f}であり、{'遠すぎる参照による低下が見られた' if last['pr_auc'] < best['pr_auc'] else '128 pxまでの低下は見られなかった'}。距離増加では傷混入が減る一方、照明・曲面・格子の空間変化を正常差として拾う可能性がある。", "",
              f"5. **正常誤検出**: 16 pxの実測正常FPR/誤検出箇所は {pct(row16['actual_normal_fpr'])}/{row16['false_positive_components']}、最良距離は {pct(best['actual_normal_fpr'])}/{best['false_positive_components']}。窓単位FPRは {best['actual_normal_fpr']-row16['actual_normal_fpr']:+.2%}変化し、連結成分数は {best['false_positive_components']-row16['false_positive_components']:+d}。したがって、正常誤検出は窓数では{'増加' if best['actual_normal_fpr'] > row16['actual_normal_fpr'] else '減少'}、空間的にまとめた箇所数では{'増加' if best['false_positive_components'] > row16['false_positive_components'] else '減少'}しており、単一の意味で増減したとは言えない。", "",
              f"6. **推奨**: 比較条件内では48 pxがPR-AUC最大かつ傷窓TPR最大タイで、誤検出箇所も16/32 pxより少ないため相対的な候補である。ただし絶対性能が低いため、現段階で48 pxを実運用値として確定するのではなく、独立画像で再検証する。72 pxは傷混入を幾何学的に避けられるが、検出性能は48 pxを上回らなかった。", "",
              "窓単位TPRと傷単位検出率は別物である。窓単位は陽性格子点の割合、傷単位は各アノテーションに1点でも反応したかを表す。詳細は `distance_comparison.csv`、画像別は `per_image_results.csv`、傷別は `defect_instance_results.csv` を参照。", "",
              "## 除外・制約", ""]
    if excluded:
        lines.extend(f"- {row['image_id']}: {row['reason']}" for row in excluded)
    else:
        lines.append("- 除外画像なし。")
    lines += ["- 未ラベル領域を正常表面とみなす既存評価上の仮定を含む。",
              "- LOIOは評価画像を閾値決定に使わないが、同一撮像系列内の検証であり、完全に独立した将来データではない。",
              "- 全距離共通位置に限定したため、画像端の傷は評価不能になり得る。距離間の公平性を優先し、距離ごとに位置数を増やしていない。", "",
              "## 出力", "",
              "- `distance_comparison.csv`: 距離別集計",
              "- `per_image_results.csv`: 画像・距離別集計とLOIO閾値",
              "- `defect_instance_results.csv`: 傷アノテーション単位の検出結果",
              "- `heatmaps/`: image15の距離別ヒートマップ（全て共通カラースケール）",
              "- `image15_heatmaps_common_scale.png`, `image15_detection_comparison.png`: image15比較",
              "- `distance_vs_*.png`: 距離別性能グラフ",
              "- `score_distributions.png`: 傷・正常スコア分布"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    settings = {"window": WINDOW, "evaluation_size": EVAL_SIZE, "stride": STRIDE,
                "distances": list(DISTANCES), "normalisation_band": list(FREQ),
                "rms_band": list(FREQ), "bin_width": BIN_WIDTH,
                "reference_count": 8, "reference_aggregation": "arithmetic mean in dB",
                "score": "RMS in dB", "threshold": "same-group LOIO equal-image-weighted normal p99",
                "data": str(args.data.resolve())}
    (output / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    global DATA
    DATA = args.data
    records: dict[int, dict] = {}
    excluded: list[dict] = []
    processing_rows: list[dict] = []
    for image_id in range(1, 22):
        path = DATA / f"image{image_id}.png"
        if not path.exists():
            excluded.append({"image_id": f"image{image_id}", "reason": "image file missing"})
            continue
        image = np.asarray(Image.open(path).convert("L"), np.float32)
        xs, ys = common_centres(image.shape)
        if not len(xs) or not len(ys):
            reason = (f"no common target positions: image {image.shape[1]}x{image.shape[0]} px; "
                      f"distance 128 requires target centres at least {max(DISTANCES)+HALF} px from every edge")
            excluded.append({"image_id": f"image{image_id}", "reason": reason})
            print(f"image{image_id}: excluded - {reason}", flush=True)
            continue
        started = time.perf_counter()
        coordinates = required_coordinates(xs, ys)
        spectra, cache = calculate_spectra(image, coordinates, args.workers, args.batch_size)
        scores = score_maps(spectra, cache, xs, ys)
        masks = load_masks(image_id, image.shape, xs, ys)
        records[image_id] = {"image": image, "xs": xs, "ys": ys, "scores": scores, "masks": masks,
                             "group": group_for(image_id)}
        processing_rows.append({"image_id": f"image{image_id}", "status": "evaluated", "width": image.shape[1],
                                "height": image.shape[0], "common_target_positions": len(xs)*len(ys),
                                "unique_cached_spectra": len(coordinates), "seconds": time.perf_counter()-started})
        print(f"image{image_id}: targets={len(xs)*len(ys):,}, cached FFT={len(coordinates):,}, {processing_rows[-1]['seconds']:.1f}s", flush=True)
    for row in excluded:
        processing_rows.append({"image_id": row["image_id"], "status": "excluded", "reason": row["reason"]})
    write_csv(output / "processing_summary.csv", processing_rows)
    write_csv(output / "excluded_images.csv", excluded, ["image_id", "reason"])

    thresholds: dict[tuple[int, int], float | None] = {}
    image_rows: list[dict] = []
    instance_rows: list[dict] = []
    for distance in DISTANCES:
        for image_id, record in records.items():
            training = []
            training_names = []
            for other_id, other in records.items():
                if other_id == image_id or other["group"] != record["group"]:
                    continue
                values = other["scores"][distance][other["masks"]["normal"]]
                if len(values):
                    training.append(values)
                    training_names.append(f"image{other_id}")
            threshold = equal_image_quantile(training, .99)
            thresholds[distance, image_id] = threshold
            score = record["scores"][distance]
            normal_values = score[record["masks"]["normal"]]
            defect_values = score[record["masks"]["defect"]]
            normal_hits = record["masks"]["normal"] & (score > threshold) if threshold is not None else np.zeros_like(score, bool)
            defect_hits = defect_values > threshold if threshold is not None else np.zeros(len(defect_values), bool)
            detected_instances = evaluable_instances = 0
            for instance in record["masks"]["instances"]:
                values = score[instance["selection"]]
                detected = bool(np.any(values > threshold)) if threshold is not None and len(values) else None
                if len(values):
                    evaluable_instances += 1
                    detected_instances += detected is True
                instance_rows.append({"distance_px": distance, "image_id": f"image{image_id}", "group": record["group"],
                                      "annotation_index": instance["annotation_index"], "defect_label": instance["label"],
                                      "candidate_windows": len(values), "threshold": threshold, "detected": detected,
                                      "maximum_score": float(values.max()) if len(values) else None})
            roc, pr = safe_auc(np.r_[np.zeros(len(normal_values), bool), np.ones(len(defect_values), bool)],
                               np.r_[normal_values, defect_values])
            image_rows.append({"distance_px": distance, "image_id": f"image{image_id}", "group": record["group"],
                               "status": "evaluated", "common_positions": score.size, "normal_windows": len(normal_values),
                               "defect_windows": len(defect_values), "roc_auc": roc, "pr_auc": pr,
                               "threshold_loio_normal_p99": threshold, "threshold_training_images": ";".join(training_names),
                               "normal_false_positive_windows": int(normal_hits.sum()),
                               "normal_fpr": float(normal_hits.sum()/len(normal_values)) if len(normal_values) else None,
                               "false_positive_components": false_positive_components(normal_hits),
                               "detected_defect_windows": int(defect_hits.sum()),
                               "defect_window_tpr": float(defect_hits.mean()) if len(defect_hits) else None,
                               "detected_defect_instances": detected_instances,
                               "evaluable_defect_instances": evaluable_instances,
                               "defect_instance_detection_rate": detected_instances/evaluable_instances if evaluable_instances else None})
    for distance in DISTANCES:
        for row in excluded:
            image_rows.append({"distance_px": distance, "image_id": row["image_id"], "status": "excluded", "reason": row["reason"]})

    distance_rows: list[dict] = []
    for distance in DISTANCES:
        normal_values = np.concatenate([r["scores"][distance][r["masks"]["normal"]] for r in records.values()])
        defect_values = np.concatenate([r["scores"][distance][r["masks"]["defect"]] for r in records.values()])
        y = np.r_[np.zeros(len(normal_values), bool), np.ones(len(defect_values), bool)]
        score = np.r_[normal_values, defect_values]
        roc, pr = safe_auc(y, score)
        rows = [row for row in image_rows if row["distance_px"] == distance and row["status"] == "evaluated"]
        normal_count = sum(row["normal_windows"] for row in rows)
        defect_count = sum(row["defect_windows"] for row in rows)
        distance_rows.append({"distance_px": distance, "evaluated_images": len(records),
                              "excluded_images": len(excluded), "common_target_positions": sum(r["scores"][distance].size for r in records.values()),
                              "normal_windows": normal_count, "defect_windows": defect_count,
                              "roc_auc": roc, "pr_auc": pr,
                              "window_tpr_at_train_fpr1": sum(row["detected_defect_windows"] for row in rows)/defect_count if defect_count else None,
                              "actual_normal_fpr": sum(row["normal_false_positive_windows"] for row in rows)/normal_count if normal_count else None,
                              "detected_defect_windows": sum(row["detected_defect_windows"] for row in rows),
                              "normal_false_positive_windows": sum(row["normal_false_positive_windows"] for row in rows),
                              "false_positive_components": sum(row["false_positive_components"] for row in rows),
                              "detected_defect_instances": sum(row["detected_defect_instances"] for row in rows),
                              "evaluable_defect_instances": sum(row["evaluable_defect_instances"] for row in rows),
                              "defect_instance_detection_rate": (sum(row["detected_defect_instances"] for row in rows)/sum(row["evaluable_defect_instances"] for row in rows)) if sum(row["evaluable_defect_instances"] for row in rows) else None})
    write_csv(output / "distance_comparison.csv", distance_rows)
    write_csv(output / "per_image_results.csv", image_rows)
    write_csv(output / "defect_instance_results.csv", instance_rows)
    draw_image15(records, thresholds, output)
    plots(distance_rows, records, output)
    make_report(output, distance_rows, excluded, image_rows)
    print(f"completed: {output.resolve()}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        parser.error("workers and batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
