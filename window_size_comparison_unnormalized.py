#!/usr/bin/env python3
"""Unnormalised local radial-FFT comparison for five analysis-window sizes."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
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

import evaluate_all_radial_fft as labels_impl
import multiscale_fft_experiment as base
import radial_fft_experiment as radial


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data20260903" / "data20260903"
OUT = ROOT / "output_window_size_comparison_unnormalized"
WINDOWS = (32, 64, 128, 256, 512)
STRIDE = 16
EVAL_SIZE = 16
EVAL_HALF = EVAL_SIZE // 2
FREQ = (0.020, 0.500)
BIN_WIDTH = 0.010
EPS = base.EPS
OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
DEFECT_LABELS = set(labels_impl.CLASSES)
EXCLUSION_LABELS = {"tag", "background"}
GRID_IMAGES = set(range(2, 18))
SCOPE_WINDOWS = {"all_positions": WINDOWS, "common_64_128_256": (64, 128, 256), "common_all_windows": WINDOWS}


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    columns = list(dict.fromkeys(k for row in rows for k in row)) if rows else (fields or ["status"])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def group_for(image_id: int) -> str:
    return "grid" if image_id in GRID_IMAGES else "no_grid"


def target_centres(shape: tuple[int, int], window: int, reference_policy: str) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape; half = window // 2
    margin = window + half if reference_policy == "strict8" else half
    xs = np.arange(half, width-half+1, STRIDE, dtype=np.int32)
    ys = np.arange(half, height-half+1, STRIDE, dtype=np.int32)
    return xs[(xs >= margin) & (xs <= width-margin)], ys[(ys >= margin) & (ys <= height-margin)]


def required_coordinates(xs: np.ndarray, ys: np.ndarray, window: int, reference_policy: str) -> list[tuple[int, int]]:
    coords = {(int(x), int(y)) for y in ys for x in xs}
    if reference_policy == "strict8":
        coords.update((int(x+dx*window), int(y+dy*window)) for y in ys for x in xs for dy, dx in OFFSETS)
    return sorted(coords, key=lambda p: (p[1], p[0]))


def calculate_spectra(image: np.ndarray, coords: list[tuple[int, int]], window: int,
                      workers: int, requested_batch: int) -> tuple[np.ndarray, dict, int, int]:
    """Raw radial power in dB: deliberately no total-power normalisation."""
    _, _, _, specs = radial.grids(window, BIN_WIDTH, [FREQ]); spec = specs[FREQ]
    valid = spec["valid"]; half = window//2
    hann1 = np.hanning(window).astype(np.float32); hann = np.outer(hann1, hann1).astype(np.float32)
    spectra = np.empty((len(coords), int(valid.sum())), np.float32)
    cache = {coord: i for i, coord in enumerate(coords)}
    # Keep roughly the same input-pixel memory per FFT call at every W.
    batch = max(1, min(requested_batch, 2_000_000 // (window*window)))
    for start in range(0, len(coords), batch):
        chosen = coords[start:start+batch]
        values = np.stack([image[y-half:y+half, x-half:x+half] for x, y in chosen]).astype(np.float32, copy=False).copy()
        values -= values.mean(axis=(-2, -1), keepdims=True); values *= hann
        transformed = fft.rfft2(values, axes=(-2, -1), workers=workers)
        power = (np.abs(fft.fftshift(transformed, axes=(-2,)))**2).astype(np.float32)
        sums = power.reshape(len(power), -1) @ spec["matrix"]
        radial_mean_power = sums[:, valid] / spec["counts"][valid][None, :]
        spectra[start:start+len(chosen)] = (10.0*np.log10(radial_mean_power+EPS)).astype(np.float32)
    return spectra, cache, int(valid.sum()), batch


def calculate_score(spectra: np.ndarray, cache: dict, xs: np.ndarray, ys: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    score = np.empty((len(ys), len(xs)), np.float32)
    counts = np.empty((len(ys), len(xs)), np.uint8)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            target = spectra[cache[int(x), int(y)]]
            ref_coords = [(int(x+dx*window), int(y+dy*window)) for dy, dx in OFFSETS
                          if (int(x+dx*window), int(y+dy*window)) in cache]
            counts[iy, ix] = len(ref_coords)
            if not ref_coords:
                score[iy, ix] = np.nan
                continue
            refs = np.stack([spectra[cache[coord]] for coord in ref_coords])
            score[iy, ix] = np.sqrt(np.mean((target-refs.mean(axis=0))**2))
    return score, counts


def rectangle_overlap(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    integral = np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    x0 = xs-EVAL_HALF; y0 = ys-EVAL_HALF; out = np.empty((len(ys), len(xs)), bool)
    for iy, top in enumerate(y0):
        out[iy] = (integral[top+EVAL_SIZE, x0+EVAL_SIZE]-integral[top, x0+EVAL_SIZE]
                   -integral[top+EVAL_SIZE, x0]+integral[top, x0]) > 0
    return out


def masks_for(image_id: int, shape: tuple[int, int], xs: np.ndarray, ys: np.ndarray, data: Path) -> dict:
    labels = labels_impl.annotation(data/f"image{image_id}.json", shape)
    if labels is None: raise FileNotFoundError(f"image{image_id}.json")
    zeros = np.zeros(shape, bool)
    defects = [(idx, name, mask) for idx, (name, mask) in enumerate(labels) if name in DEFECT_LABELS]
    exclusion_masks = [mask for name, mask in labels if name in EXCLUSION_LABELS]
    defect_pixels = np.logical_or.reduce([m for _, _, m in defects]) if defects else zeros.copy()
    exclusion_pixels = np.logical_or.reduce(exclusion_masks) if exclusion_masks else zeros.copy()
    excluded = rectangle_overlap(exclusion_pixels, xs, ys); defect_grid = rectangle_overlap(defect_pixels, xs, ys)
    instances = [{"annotation_index": idx, "label": name, "selection": rectangle_overlap(mask, xs, ys)&~excluded}
                 for idx, name, mask in defects]
    return {"labels": labels, "defect_pixel": defect_pixels, "exclusion_pixel": exclusion_pixels,
            "excluded": excluded, "defect": defect_grid&~excluded, "normal": ~defect_grid&~excluded,
            "instances": instances}


def common_coordinate_masks(by_window: dict[int, dict], required: tuple[int, ...]) -> dict[int, np.ndarray] | None:
    if any(w not in by_window for w in required): return None
    sets = [{(int(x), int(y)) for y in by_window[w]["ys"] for x in by_window[w]["xs"]} for w in required]
    common = set.intersection(*sets)
    if not common: return None
    result = {}
    for w in required:
        r = by_window[w]
        result[w] = np.asarray([[(int(x), int(y)) in common for x in r["xs"]] for y in r["ys"]], bool)
    return result


def equal_image_quantile(arrays: list[np.ndarray], q: float=.99) -> float | None:
    arrays = [np.sort(np.asarray(a, float)) for a in arrays if len(a)]
    if not arrays: return None
    values = np.concatenate(arrays); weights = np.concatenate([np.full(len(a), 1/len(a)) for a in arrays])
    order = np.argsort(values); cumulative = np.cumsum(weights[order]); target = q*len(arrays)
    at = min(int(np.searchsorted(cumulative, target, side="left")), len(values)-1)
    return float(values[order[at]])


def auc_pair(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    return ((float(roc_auc_score(y, score)), float(average_precision_score(y, score)))
            if len(np.unique(y)) == 2 else (math.nan, math.nan))


def component_count(mask: np.ndarray) -> int:
    return int(connected_components(mask, structure=np.ones((3, 3), np.uint8))[1])


def overlay_annotations(ax, record: dict, exclusions: bool=True) -> None:
    if record["masks"]["defect_pixel"].any():
        ax.contour(record["masks"]["defect_pixel"].astype(np.uint8), [.5], colors="#38ff5a", linewidths=.8)
    if exclusions and record["masks"]["exclusion_pixel"].any():
        ax.contour(record["masks"]["exclusion_pixel"].astype(np.uint8), [.5], colors="#00e5ff", linewidths=.7)


def extent(record: dict) -> list[float]:
    return [record["xs"][0]-STRIDE/2, record["xs"][-1]+STRIDE/2,
            record["ys"][-1]+STRIDE/2, record["ys"][0]-STRIDE/2]


def percentile_range(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not len(values): return 0.0, 1.0
    lo, hi = np.percentile(values, [5, 99])
    if hi <= lo: hi = lo+1e-6
    return float(lo), float(hi)


def render_outputs(records: dict[int, dict[int, dict]], thresholds: dict, output: Path,
                   scale_rows: list[dict]) -> None:
    per_dir=output/"heatmaps_per_image_scale"; normal_dir=output/"heatmaps_normal_scale_auxiliary"; shared_dir=output/"heatmaps_shared_scale"; det_dir=output/"detection_comparison"
    for folder in (per_dir, normal_dir, shared_dir, det_dir): folder.mkdir(exist_ok=True)
    for image_id, by_window in records.items():
        image = next(iter(by_window.values()))["image"]; h, wimg = image.shape
        pooled_valid = np.concatenate([r["score"].ravel() for r in by_window.values()])
        shared_lo, shared_hi = percentile_range(pooled_valid)
        fig_shared, axes_shared = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
        fig_normal, axes_normal = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
        fig_det, axes_det = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
        for ax in axes_shared.flat: ax.axis("off")
        for ax in axes_normal.flat: ax.axis("off")
        for ax in axes_det.flat: ax.axis("off")
        for slot, window in enumerate(WINDOWS):
            if window not in by_window: continue
            r=by_window[window]; normal=r["score"][r["masks"]["normal"]]; all_values=r["score"].ravel()
            normal_lo, normal_hi=percentile_range(normal); all_lo, all_hi=percentile_range(all_values)
            scale_rows.append({"image_id":f"image{image_id}","window_size":window,
                               "normal_p05":normal_lo,"normal_p99":normal_hi,
                               "label_free_valid_p05":all_lo,"label_free_valid_p99":all_hi,
                               "p05_difference_label_free_minus_normal":all_lo-normal_lo,
                               "p99_difference_label_free_minus_normal":all_hi-normal_hi})
            fig, ax=plt.subplots(figsize=(10,7),constrained_layout=True); ax.imshow(image,cmap="gray",vmin=0,vmax=255)
            im=ax.imshow(r["score"],extent=extent(r),cmap="inferno",vmin=all_lo,vmax=all_hi,alpha=.68,interpolation="nearest")
            overlay_annotations(ax,r); threshold=thresholds.get((window,"all_positions",image_id))
            valid=r["masks"]["normal"]|r["masks"]["defect"]
            hits=(r["score"]>threshold)&valid if threshold is not None else np.zeros_like(r["score"],bool)
            for iy,ix in np.argwhere(hits):
                ax.add_patch(Rectangle((r["xs"][ix]-EVAL_HALF,r["ys"][iy]-EVAL_HALF),EVAL_SIZE,EVAL_SIZE,
                                       facecolor=(1,0,0,.35),edgecolor="red",linewidth=.2))
            ax.set(xlim=(0,wimg),ylim=(h,0),title=f"image{image_id}, W={window}: all-valid p05–p99 = {all_lo:.3f}–{all_hi:.3f} dB")
            fig.colorbar(im,ax=ax,label="unnormalised radial-spectrum RMS (dB)")
            fig.savefig(per_dir/f"image{image_id}_w{window}.png",dpi=125);plt.close(fig)
            ax=axes_normal.flat[slot];ax.axis("on");ax.imshow(image,cmap="gray",vmin=0,vmax=255)
            im_normal=ax.imshow(r["score"],extent=extent(r),cmap="inferno",vmin=normal_lo,vmax=normal_hi,alpha=.68,interpolation="nearest")
            overlay_annotations(ax,r);ax.set(xlim=(0,wimg),ylim=(h,0),title=f"W={window}: {normal_lo:.3f}–{normal_hi:.3f}");ax.set_axis_off()
            fig_normal.colorbar(im_normal,ax=ax,label="RMS (dB)",shrink=.72,pad=.02)
            ax=axes_shared.flat[slot];ax.axis("on");ax.imshow(image,cmap="gray",vmin=0,vmax=255)
            im_shared=ax.imshow(r["score"],extent=extent(r),cmap="inferno",vmin=shared_lo,vmax=shared_hi,alpha=.68,interpolation="nearest")
            overlay_annotations(ax,r);ax.set(xlim=(0,wimg),ylim=(h,0),title=f"W={window}");ax.set_axis_off()
            ax=axes_det.flat[slot];ax.axis("on");ax.imshow(image,cmap="gray",vmin=0,vmax=255)
            for iy,ix in np.argwhere(hits):
                ax.add_patch(Rectangle((r["xs"][ix]-EVAL_HALF,r["ys"][iy]-EVAL_HALF),EVAL_SIZE,EVAL_SIZE,
                                       facecolor=(1,0,0,.4),edgecolor="red",linewidth=.2))
            overlay_annotations(ax,r);ax.set(xlim=(0,wimg),ylim=(h,0),title=f"W={window}, t={threshold:.3f}" if threshold else f"W={window}");ax.set_axis_off()
        fig_shared.colorbar(im_shared,ax=list(axes_shared.flat),label="RMS score (dB)",shrink=.8)
        fig_shared.suptitle(f"image{image_id}: shared all-valid scale across W ({shared_lo:.3f}–{shared_hi:.3f} dB)")
        fig_shared.savefig(shared_dir/f"image{image_id}_shared_scale.png",dpi=135);plt.close(fig_shared)
        fig_normal.suptitle(f"image{image_id}: auxiliary normal-region p05–p99 scales")
        fig_normal.savefig(normal_dir/f"image{image_id}_normal_scale.png",dpi=135);plt.close(fig_normal)
        fig_det.suptitle(f"image{image_id}: LOIO detections (red), damage (green), tag/background (cyan)")
        fig_det.savefig(det_dir/f"image{image_id}_detections.png",dpi=135);plt.close(fig_det)


def make_plots(summary: list[dict], records: dict, processing: list[dict], output: Path) -> None:
    scope_styles={"all_positions":("-","all positions"),"common_64_128_256":("--","common 64/128/256"),"common_all_windows":(":","common all W")}
    metrics=(("roc_auc","ROC-AUC"),("pr_auc","PR-AUC (AP)"),("defect_instance_detection_rate","Defect-instance detection rate"),
             ("actual_normal_fpr","Actual normal FPR"),("false_positive_components","False-positive components"))
    fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    for ax,(key,label) in zip(axes.flat,metrics):
        for scope,(style,name) in scope_styles.items():
            rows=sorted([r for r in summary if r["scope"]==scope],key=lambda r:r["window_size"])
            if rows: ax.plot([r["window_size"] for r in rows],[r[key] for r in rows],marker="o",ls=style,label=name)
        ax.set(xlabel="Analysis window W (px)",ylabel=label,xticks=WINDOWS);ax.grid(alpha=.25);ax.legend(fontsize=7)
    ax=axes.flat[-1]
    med=[]
    for w in WINDOWS:
        vals=[r["seconds"] for r in processing if r.get("status")=="evaluated" and r.get("window_size")==w]
        med.append(float(np.median(vals)) if vals else np.nan)
    ax.plot(WINDOWS,med,"o-");ax.set(xlabel="Analysis window W (px)",ylabel="Median seconds / image",xticks=WINDOWS);ax.grid(alpha=.25)
    fig.savefig(output/"window_size_vs_metrics.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    for col,(key,label) in enumerate((('roc_auc','ROC-AUC'),('pr_auc','PR-AUC (AP)'))):
        rows=sorted([r for r in summary if r["scope"]=="all_positions"],key=lambda r:r["window_size"]);x=[r["window_size"] for r in rows];y=[r[key] for r in rows]
        axes[0,col].plot(x,y,"o-");axes[0,col].set(xlabel="W (px)",ylabel=label,xticks=WINDOWS,ylim=(0,1));axes[0,col].grid(alpha=.25)
        pad=max((max(y)-min(y))*.2,.002);axes[1,col].plot(x,y,"o-")
        for xx,yy in zip(x,y): axes[1,col].annotate(f"{yy:.4f}",(xx,yy),xytext=(0,6),textcoords="offset points",ha="center",fontsize=8)
        axes[1,col].set(xlabel="W (px)",ylabel=f"{label} (zoom)",xticks=WINDOWS,ylim=(min(y)-pad,max(y)+pad));axes[1,col].grid(alpha=.25)
    fig.savefig(output/"window_size_auc_full_and_zoom.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(20,4),constrained_layout=True)
    for ax,w in zip(axes,WINDOWS):
        normal=np.concatenate([r[w]["score"][r[w]["masks"]["normal"]] for r in records.values() if w in r])
        defect=np.concatenate([r[w]["score"][r[w]["masks"]["defect"]] for r in records.values() if w in r])
        hi=np.percentile(np.r_[normal,defect],99.5);bins=np.linspace(0,hi,50)
        ax.hist(normal,bins=bins,density=True,histtype="step",label=f"normal {len(normal):,}")
        ax.hist(defect,bins=bins,density=True,histtype="step",label=f"damage {len(defect):,}")
        ax.set(title=f"W={w}",xlabel="RMS score (dB)",ylabel="density");ax.legend(fontsize=6);ax.grid(alpha=.18)
    fig.savefig(output/"score_distributions.png",dpi=170);plt.close(fig)


def make_report(output: Path, summary: list[dict], processing: list[dict], excluded: list[dict], scale_rows: list[dict], image_rows: list[dict], reference_policy: str) -> None:
    primary=sorted([r for r in summary if r["scope"]=="all_positions"],key=lambda r:r["window_size"])
    core=sorted([r for r in summary if r["scope"]=="common_64_128_256"],key=lambda r:r["window_size"])
    best=max(core,key=lambda r:(r["pr_auc"],r["roc_auc"]))
    best_roc=max(core,key=lambda r:r["roc_auc"])
    best_detection=max(core,key=lambda r:(r["detected_defect_instances"],-r["actual_normal_fpr"]))
    def pct(v): return "NA" if v is None or not np.isfinite(v) else f"{100*v:.2f}%"
    grid_rows=[r for r in primary if r.get("group","")=="grid"]
    scale_p99=np.asarray([r["p99_difference_label_free_minus_normal"] for r in scale_rows],float)
    grid_all=[r for r in image_rows if r["scope"]=="all_positions" and r["group"]=="grid"]
    def grid_mean(window: int, key: str) -> float:
        values=np.asarray([r[key] for r in grid_all if r["window_size"]==window],float)
        return float(np.nanmean(values))
    timing_text=", ".join(f"W={w}: {np.median([r['seconds'] for r in processing if r.get('status')=='evaluated' and r.get('window_size')==w]):.2f}秒" for w in WINDOWS)
    reference_text=("8参照が全て画像内にある位置だけを使用" if reference_policy=="strict8"
                    else "各位置で画像内に完全に収まる参照窓（全体で1～8窓、フルサイズ画像では3～8窓）の算術平均を使用")
    lines=["# 局所FFT異常検出：解析窓サイズ比較（正規化なし・"+("8参照必須" if reference_policy=="strict8" else "利用可能参照平均")+"）","","## 1. 目的","",
           "現在の128×128 px窓が適切かを、32/64/128/256/512 pxで比較した。既存結果は上書きせず、新しい実験として実行した。Wと同時に参照距離d=Wも変わるため、結果は窓サイズ単独の因果効果ではない。","",
           "## 2. FFT・スコア条件","",
           "各窓をグレースケール化し、窓内平均除去、2次元Hann、SciPy rFFT2（既定backward scaling）、power=|F|²、共役重み付き放射平均、10log10、参照dBスペクトルの算術平均との差のRMSの順に処理した。総パワー除算は行っていない。帯域0.020–0.500 cycles/pixel、ビン幅0.01、stride 16、中心評価領域16×16 px、参照距離d=W。"+reference_text+"。画像外補間・ゼロ埋めは行っていない。","",
           "閾値は同じ照明群の他画像正常窓から、画像等重み99%点をLOIOで決めた。評価画像自身では調整していない。tag/backgroundと重なる中心16×16 pxは除外した。","",
           "## 3. 主結果：各Wで評価可能な全位置","",
           "|W px|有効放射ビン|画像|正常窓|傷窓|ROC-AUC|PR-AUC|傷窓TPR|検出傷/評価傷|正常FPR|誤検出箇所|","|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in primary:
        lines.append(f"|{r['window_size']}|{r['valid_frequency_bins']}|{r['evaluated_images']}|{r['normal_windows']}|{r['defect_windows']}|{r['roc_auc']:.4f}|{r['pr_auc']:.4f}|{pct(r['window_tpr_at_train_fpr1'])}|{r['detected_defect_instances']}/{r['evaluable_defect_instances']}|{pct(r['actual_normal_fpr'])}|{r['false_positive_components']}|")
    lines += ["","PR-AUCを主指標、ROC-AUCを副指標とした相対的な最良条件は " + f"**W={best['window_size']} px**（PR-AUC {best['pr_auc']:.4f}、ROC-AUC {best['roc_auc']:.4f}）だった。ただし傷単位検出、正常FPR、評価可能範囲も併記して判断する。","",
              "## 4. 重視する比較：64/128/256 px共通位置","",
              "512 pxは解析窓自体が大きく評価可能中心が減るため参考条件とした。窓サイズ候補の主判断は64/128/256 pxの共通評価位置で行う。`window_size_comparison.csv`では、A=各Wの全位置、B=64/128/256共通位置、C=全W共通位置を分けた。評価不能傷は失敗に含めず、`defect_instance_results.csv`に候補窓0として記録した。","",
              "|W px|共通位置|ROC-AUC|PR-AUC|傷窓TPR|検出傷/評価傷|正常FPR|","|---:|---:|---:|---:|---:|---:|---:|" ]
    for r in core:
        lines.append(f"|{r['window_size']}|{r['valid_positions']}|{r['roc_auc']:.4f}|{r['pr_auc']:.4f}|{pct(r['window_tpr_at_train_fpr1'])}|{r['detected_defect_instances']}/{r['evaluable_defect_instances']}|{pct(r['actual_normal_fpr'])}|")
    lines += ["",f"共通位置ではPR-AUC最大は{best['window_size']} px、ROC-AUC最大は{best_roc['window_size']} px。傷単位検出最多は{best_detection['window_size']} pxの{best_detection['detected_defect_instances']}/{best_detection['evaluable_defect_instances']}。正常FPRも含めると、分布分離の第一候補を{best['window_size']} px、傷検出側の比較候補を{best_detection['window_size']} px、現行基準を128 pxとして次の正規化実験へ残す。","",
              "## 5. 可視化スケール","",
              "主ヒートマップは正解ラベルを使わず、画像・Wごとの全有効スコアp5–p99を表示範囲とした。正常領域p5–p99は補助表示として別フォルダへ出力した。同一画像の全W共通表示も、全Wの全有効スコアを統合したp5–p99を使用した。全条件におけるラベル非依存p99−正常p99の中央値は " + f"{np.median(scale_p99):+.4f} dB（範囲 {np.min(scale_p99):+.4f}～{np.max(scale_p99):+.4f} dB）。表示上限を超えた値は最高色に飽和させたが、生スコアと評価値は変更していない。","",
              "## 6. 格子模様・位置精度・解析範囲","",
              "格子あり/なしは別群でLOIO閾値を学習した。全位置での格子あり画像の画像別ROC-AUC平均は " + ", ".join(f"W={w}: {grid_mean(w,'roc_auc'):.4f}" for w in WINDOWS) + "。格子あり群では128/256 pxの画像内順位性能が相対的に高かった一方、全画像統合ROC-AUCの順位とは一致しなかった。正規化なしでは画像間のスコア水準差が統合指標へ入るため、画像別値と統合値を区別する必要がある。小さいWは格子周期の一部だけを捉えて正常格子へ反応し得る一方、大きいWは複数周期・照明むら・背景構造を同時に含み得る。今回のWとdの同時変更だけでは因果分離できない。","",
              "スコアの割当格子は全Wで中心16×16 px・stride 16だが、スコアを生む画像支持領域はW×Wである。したがって大きいWほど表示格子が細かくても実効的な位置局在性は低い。available-reference条件では端部も評価できるが、参照数1～7の位置と8の内部位置で参照平均の安定性が異なる（フルサイズ画像の最小は3）。","",
              "## 7. 計算時間","",
              "画像ごとの解析時間中央値は " + timing_text + "。`processing_summary.csv`に画像・W別の総処理時間、参照数範囲と実効FFTバッチ件数を記録した。ここでのbatchは計算時に同時処理するFFT件数であり、解析窓サイズWとは別概念である。","",
              "## 8. 次の正規化実験への候補","",
              f"64/128/256 px共通位置における正規化なしの相対分離ではW={best['window_size']} pxが第一候補。ただし、傷単位検出率・正常FPR・評価可能範囲での順位が一致しない場合は複数候補を併記し、単一条件へ確定しない。次は同じW・d・評価位置を固定したまま総パワー正規化の有無だけを比較する。","",
              "## 9. 制約・今後","",
              "- Wと参照距離d=Wが同時に変化しているため、窓サイズ単独の効果ではない。",
              "- 端部では参照数が少ないため、内部8参照位置とのスコア分散差が性能へ影響し得る。",
              "- 未ラベル領域を正常とみなす。",
              "- 同一撮像系列内LOIOであり、完全な外部検証ではない。",
              "- FFTの定数スケーリングはW間で異なるが、同一Wの対象・参照dB差では定数項が相殺される。",
              "- 次段階では候補Wについてdを固定または境界余裕で対応付け、Wとdの交互作用を分離する。","",
              "## 10. 除外","" ]
    lines.extend([f"- {r['image_id']} W={r.get('window_size','')}: {r['reason']}" for r in excluded] or ["- なし"])
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    output=base.unique_dir(args.output)
    settings={"data":str(args.data.resolve()),"windows":list(WINDOWS),"stride":STRIDE,"evaluation_size":EVAL_SIZE,
              "power_normalisation":"none","fft_scaling":"scipy.fft.rfft2 default backward; power=abs(F)^2; no W division",
              "rms_frequency":list(FREQ),"radial_bin_width":BIN_WIDTH,
              "reference_count_policy":("exactly 8" if args.reference_policy=="strict8" else "all available in-image references; minimum 1"),"reference_distance":"d=W",
              "reference_spectrum":("arithmetic mean of 8 dB radial spectra" if args.reference_policy=="strict8" else "arithmetic mean of available in-image dB radial spectra"),"score":"RMS dB difference",
              "reference_policy":args.reference_policy,
              "threshold":"same-lighting-group LOIO equal-image-weighted normal p99","requested_fft_batch_size":args.batch_size}
    (output/"settings.json").write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding="utf-8")
    records:dict[int,dict[int,dict]]={};processing=[];excluded=[]
    for image_id in range(1,22):
        path=args.data/f"image{image_id}.png"
        if not path.exists(): excluded.append({"image_id":f"image{image_id}","reason":"image missing"});continue
        image=np.asarray(Image.open(path).convert("L"),np.float32);by_window={}
        for window in WINDOWS:
            xs,ys=target_centres(image.shape,window,args.reference_policy)
            if not len(xs) or not len(ys):
                reason=f"analysis window does not fit image; image={image.shape[1]}x{image.shape[0]}, W={window}"
                excluded.append({"image_id":f"image{image_id}","window_size":window,"reason":reason});processing.append({"image_id":f"image{image_id}","window_size":window,"status":"excluded","reason":reason});continue
            started=time.perf_counter();coords=required_coordinates(xs,ys,window,args.reference_policy)
            spectra,cache,valid_bins,effective_batch=calculate_spectra(image,coords,window,args.workers,args.batch_size)
            score,reference_counts=calculate_score(spectra,cache,xs,ys,window);masks=masks_for(image_id,image.shape,xs,ys,args.data)
            if not np.isfinite(score).any():
                reason=f"no in-image reference window at d=W; image={image.shape[1]}x{image.shape[0]}, W={window}"
                excluded.append({"image_id":f"image{image_id}","window_size":window,"reason":reason});processing.append({"image_id":f"image{image_id}","window_size":window,"status":"excluded","reason":reason});continue
            record={"image":image,"window":window,"xs":xs,"ys":ys,"score":score,"reference_counts":reference_counts,"masks":masks,"valid_bins":valid_bins,"group":group_for(image_id),"scopes":{"all_positions":np.isfinite(score)}}
            by_window[window]=record;seconds=time.perf_counter()-started
            processing.append({"image_id":f"image{image_id}","window_size":window,"status":"evaluated","width":image.shape[1],"height":image.shape[0],"valid_target_positions":int(np.isfinite(score).sum()),"unique_cached_spectra":len(coords),"valid_frequency_bins":valid_bins,"minimum_reference_count":int(reference_counts.min()),"maximum_reference_count":int(reference_counts.max()),"locations_with_8_references":int(np.sum(reference_counts==8)),"effective_fft_batch_size":effective_batch,"seconds":seconds})
            print(f"image{image_id} W={window}: target={score.size:,}, FFT={len(coords):,}, bins={valid_bins}, {seconds:.1f}s",flush=True)
        if by_window:
            common_b=common_coordinate_masks(by_window,(64,128,256));common_c=common_coordinate_masks(by_window,WINDOWS)
            if common_b:
                for window,mask in common_b.items():by_window[window]["scopes"]["common_64_128_256"]=mask
            if common_c:
                for window,mask in common_c.items():by_window[window]["scopes"]["common_all_windows"]=mask
            records[image_id]=by_window
    write_csv(output/"processing_summary.csv",processing);write_csv(output/"excluded_images.csv",excluded,["image_id","window_size","reason"])
    thresholds={};image_rows=[];instance_rows=[]
    for image_id,by_window in records.items():
      for window,r in by_window.items():
       for scope,scope_mask in r["scopes"].items():
        training=[];training_names=[]
        for other_id,other_windows in records.items():
            if other_id==image_id or window not in other_windows or other_windows[window]["group"]!=r["group"] or scope not in other_windows[window]["scopes"]:continue
            other=other_windows[window];mask=other["scopes"][scope]&other["masks"]["normal"];v=other["score"][mask]
            if len(v):training.append(v);training_names.append(f"image{other_id}")
        threshold=equal_image_quantile(training);thresholds[window,scope,image_id]=threshold
        valid=scope_mask;normal_mask=valid&r["masks"]["normal"];defect_mask=valid&r["masks"]["defect"]
        normal=r["score"][normal_mask];defect=r["score"][defect_mask]
        normal_hits=normal_mask&(r["score"]>threshold) if threshold is not None else np.zeros_like(valid)
        defect_hits=defect>threshold if threshold is not None else np.zeros(len(defect),bool)
        detected=evaluable=unassessable=0
        for inst in r["masks"]["instances"]:
            selection=valid&inst["selection"];values=r["score"][selection];hit=bool(np.any(values>threshold)) if threshold is not None and len(values) else None
            if len(values):evaluable+=1;detected+=hit is True
            else:unassessable+=1
            instance_rows.append({"window_size":window,"scope":scope,"image_id":f"image{image_id}","group":r["group"],"annotation_index":inst["annotation_index"],"defect_label":inst["label"],"candidate_windows":len(values),"threshold":threshold,"detected":hit,"maximum_score":float(values.max()) if len(values) else None,"status":"evaluable" if len(values) else "out_of_scope"})
        roc,pr=auc_pair(np.r_[np.zeros(len(normal),bool),np.ones(len(defect),bool)],np.r_[normal,defect])
        image_rows.append({"window_size":window,"scope":scope,"image_id":f"image{image_id}","group":r["group"],"status":"evaluated","valid_positions":int(valid.sum()),"valid_frequency_bins":r["valid_bins"],"normal_windows":len(normal),"defect_windows":len(defect),"roc_auc":roc,"pr_auc":pr,"threshold_loio_normal_p99":threshold,"threshold_training_images":";".join(training_names),"normal_false_positive_windows":int(normal_hits.sum()),"normal_fpr":float(normal_hits.sum()/len(normal)) if len(normal) else None,"false_positive_components":component_count(normal_hits),"detected_defect_windows":int(defect_hits.sum()),"defect_window_tpr":float(defect_hits.mean()) if len(defect_hits) else None,"detected_defect_instances":detected,"evaluable_defect_instances":evaluable,"unassessable_defect_instances":unassessable,"defect_instance_detection_rate":detected/evaluable if evaluable else None})
    summary=[]
    for scope in SCOPE_WINDOWS:
      for window in WINDOWS:
        rows=[r for r in image_rows if r["scope"]==scope and r["window_size"]==window]
        if not rows:continue
        samples=[];ysamp=[]
        for image_id,by_window in records.items():
            if window not in by_window or scope not in by_window[window]["scopes"]:continue
            r=by_window[window];valid=r["scopes"][scope];n=r["score"][valid&r["masks"]["normal"]];d=r["score"][valid&r["masks"]["defect"]]
            samples.extend((n,d));ysamp.extend((np.zeros(len(n),bool),np.ones(len(d),bool)))
        score=np.concatenate(samples);y=np.concatenate(ysamp);roc,pr=auc_pair(y,score);normal_count=sum(r["normal_windows"] for r in rows);defect_count=sum(r["defect_windows"] for r in rows);eval_inst=sum(r["evaluable_defect_instances"] for r in rows)
        summary.append({"window_size":window,"scope":scope,"evaluated_images":len(rows),"valid_positions":sum(r["valid_positions"] for r in rows),"valid_frequency_bins":rows[0]["valid_frequency_bins"],"normal_windows":normal_count,"defect_windows":defect_count,"roc_auc":roc,"pr_auc":pr,"window_tpr_at_train_fpr1":sum(r["detected_defect_windows"] for r in rows)/defect_count if defect_count else None,"actual_normal_fpr":sum(r["normal_false_positive_windows"] for r in rows)/normal_count if normal_count else None,"normal_false_positive_windows":sum(r["normal_false_positive_windows"] for r in rows),"false_positive_components":sum(r["false_positive_components"] for r in rows),"detected_defect_instances":sum(r["detected_defect_instances"] for r in rows),"evaluable_defect_instances":eval_inst,"unassessable_defect_instances":sum(r["unassessable_defect_instances"] for r in rows),"defect_instance_detection_rate":sum(r["detected_defect_instances"] for r in rows)/eval_inst if eval_inst else None})
    group_rows=[]
    for scope in SCOPE_WINDOWS:
      for window in WINDOWS:
       for group in ("grid","no_grid"):
        rows=[r for r in image_rows if r["scope"]==scope and r["window_size"]==window and r["group"]==group]
        if not rows: continue
        normal=sum(r["normal_windows"] for r in rows); defect=sum(r["defect_windows"] for r in rows); evaluable=sum(r["evaluable_defect_instances"] for r in rows)
        group_rows.append({"window_size":window,"scope":scope,"group":group,"images":len(rows),"images_with_both_classes":sum(np.isfinite(r["roc_auc"]) for r in rows),"mean_per_image_roc_auc":float(np.nanmean([r["roc_auc"] for r in rows])),"mean_per_image_pr_auc":float(np.nanmean([r["pr_auc"] for r in rows])),"normal_windows":normal,"defect_windows":defect,"actual_normal_fpr":sum(r["normal_false_positive_windows"] for r in rows)/normal if normal else None,"detected_defect_instances":sum(r["detected_defect_instances"] for r in rows),"evaluable_defect_instances":evaluable,"defect_instance_detection_rate":sum(r["detected_defect_instances"] for r in rows)/evaluable if evaluable else None})
    write_csv(output/"window_size_comparison.csv",summary);write_csv(output/"per_image_results.csv",image_rows);write_csv(output/"defect_instance_results.csv",instance_rows);write_csv(output/"group_metrics.csv",group_rows)
    scale_rows=[];render_outputs(records,thresholds,output,scale_rows);write_csv(output/"display_scale_comparison.csv",scale_rows)
    make_plots(summary,records,processing,output);make_report(output,summary,processing,excluded,scale_rows,image_rows,args.reference_policy)
    print(f"completed: {output.resolve()}",flush=True);return output


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--data",type=Path,default=DATA);parser.add_argument("--output",type=Path,default=OUT);parser.add_argument("--workers",type=int,default=4);parser.add_argument("--batch-size",type=int,default=64);parser.add_argument("--reference-policy",choices=("strict8","available"),default="strict8");args=parser.parse_args()
    if args.workers<1 or args.batch_size<1:parser.error("workers and batch-size must be positive")
    run(args)


if __name__=="__main__":main()
