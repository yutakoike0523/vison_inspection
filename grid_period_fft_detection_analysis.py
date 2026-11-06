#!/usr/bin/env python3
"""Integrated grid-period, band, notch and window-size FFT detection study.

New computations preserve the existing signal definitions.  W=128 band/notch
comparisons use stride 16, exactly eight references at 48 px, arithmetic mean
reference spectra, and both full-band-total normalised and raw radial spectra.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
from sklearn.metrics import average_precision_score, roc_auc_score

import all_images_grid_removal_fft_comparison as allcmp
import fft_grid_notch_removal_analysis as grid
import image15_grid_removal_fft_comparison as basecmp
import local_multiscale_grid_harmonic_analysis as local
import window_size_comparison_unnormalized as labels


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output_grid_period_fft_detection_analysis"
MAPS = OUT / "score_maps"
HEAT = OUT / "frequency_band_heatmaps"
WINDOW_FIG = OUT / "window_size_heatmaps"
IMAGE_IDS = tuple(range(2, 19))
REPRESENTATIVE = {2, 4, 5, 10, 11, 13, 14, 15}
WINDOW = 128
STRIDE = 16
SPACING = 48
FREQ = (0.01, 0.50)
BIN_WIDTH = 0.01
EPS = np.float32(1e-12)
OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1)
                for dx in (-1, 0, 1) if dy or dx)
BANDS = {
    "0.01-0.50": (0.01, 0.50),
    "0.02-0.50": (0.02, 0.50),
    "0.02-0.10": (0.02, 0.10),
    "0.01-0.05": (0.01, 0.05),
    "0.05-0.50": (0.05, 0.50),
}
CONDITIONS = {
    "none": "元画像",
    "fundamentals": "基本周期除去",
    "harmonics": "基本周期＋確認済み高調波除去",
}
plt.rcParams.update({"font.family": ["Yu Gothic", "Meiryo", "DejaVu Sans"],
                     "axes.unicode_minus": False})


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    cols = list(dict.fromkeys(k for row in rows for k in row)) if rows else (fields or ["status"])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=cols, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def grid_vectors(image_id: int, top: list[dict], family: list[dict]) -> list[tuple[float, float]]:
    bases = [(float(q["fx"]), float(q["fy"])) for q in local.priors(image_id, top, family)]
    out = list(bases)
    if len(bases) == 2:
        for m, n, _ in local.COMBOS:
            v = (m*bases[0][0]+n*bases[1][0], m*bases[0][1]+n*bases[1][1])
            if math.hypot(*v) < 0.5:
                out.append(v)
    return out


def vector_keep_mask(vectors: list[tuple[float, float]]) -> np.ndarray:
    fy = np.fft.fftshift(np.fft.fftfreq(WINDOW))
    fx = np.fft.rfftfreq(WINDOW)
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    keep = np.ones(xx.shape, bool)
    radius = 1.5 / WINDOW
    for vx, vy in vectors:
        # rFFT stores only fx >= 0; use the member of each conjugate pair there.
        if vx < 0:
            vx, vy = -vx, -vy
        keep &= (xx-vx)**2 + (yy-vy)**2 > radius*radius
    return keep


def spectra_four(image: np.ndarray, vectors: list[tuple[float, float]],
                 workers: int = 4, batch: int = 64) -> dict:
    """Regular and vector-excluded raw/normalised radial log spectra."""
    h, w = image.shape; half = WINDOW//2
    xs = np.arange(half, w-half+1, STRIDE, dtype=np.int32)
    ys = np.arange(half, h-half+1, STRIDE, dtype=np.int32)
    setup = basecmp.radial_setup(); valid = setup["valid"]
    keep = vector_keep_mask(vectors) & setup["mask"]
    kept_counts = np.asarray(keep.reshape(1, -1).astype(np.float32) @ setup["matrix"])[0]
    vector_valid = valid & (kept_counts >= 4)
    arrays = {k: np.empty((len(ys), len(xs), int(valid.sum())), np.float32)
              for k in ("raw", "norm", "vector_raw", "vector_norm")}
    hann1 = np.hanning(WINDOW).astype(np.float32)
    hann = np.outer(hann1, hann1).astype(np.float32)
    for iy, y in enumerate(ys):
        row = np.lib.stride_tricks.sliding_window_view(
            image[y-half:y+half], WINDOW, axis=1)
        row = np.moveaxis(row, 1, 0)[::STRIDE]
        for start in range(0, len(xs), batch):
            values = np.asarray(row[start:start+batch], np.float32).copy()
            values -= values.mean(axis=(-2, -1), keepdims=True); values *= hann
            z = fft.rfft2(values, axes=(-2, -1), workers=workers)
            power = (np.abs(fft.fftshift(z, axes=(-2,)))**2).astype(np.float32)
            total = np.maximum(power[:, setup["mask"]].sum(axis=1), EPS)
            sums = power.reshape(len(power), -1) @ setup["matrix"]
            means = sums[:, valid] / setup["counts"][valid][None, :]
            vp = power * keep[None, :, :]
            vsums = vp.reshape(len(vp), -1) @ setup["matrix"]
            vmeans = vsums[:, valid] / np.maximum(kept_counts[valid][None, :], 1)
            sl = slice(start, start+len(values))
            arrays["raw"][iy, sl] = 10*np.log10(means+EPS)
            arrays["norm"][iy, sl] = 10*np.log10(means/total[:, None]+EPS)
            arrays["vector_raw"][iy, sl] = 10*np.log10(vmeans+EPS)
            arrays["vector_norm"][iy, sl] = 10*np.log10(vmeans/total[:, None]+EPS)
    arrays.update(xs=xs, ys=ys, vector_select=vector_valid[valid])
    return arrays


def score_features(spec: np.ndarray, select: np.ndarray) -> np.ndarray:
    step = SPACING//STRIDE
    chosen = spec[..., select]
    target = chosen[step:-step, step:-step]
    refs = np.stack([
        chosen[step+dy*step:len(chosen)-step+dy*step,
               step+dx*step:chosen.shape[1]-step+dx*step]
        for dy, dx in OFFSETS])
    return np.sqrt(np.mean((target-refs.mean(axis=0))**2, axis=-1)).astype(np.float32)


def equal_image_quantile(arrays: list[np.ndarray], q: float) -> float:
    arrays = [np.sort(np.asarray(a, float)) for a in arrays if len(a)]
    values = np.concatenate(arrays)
    weights = np.concatenate([np.full(len(a), 1/len(a)) for a in arrays])
    order = np.argsort(values); cumulative = np.cumsum(weights[order])
    target = q*len(arrays)
    return float(values[order[min(np.searchsorted(cumulative, target), len(values)-1)]])


def period_summary(top: list[dict], family: list[dict]) -> list[dict]:
    observations = read_csv(ROOT/"analysis_local_multiscale_harmonic"/"multiscale_peak_observations.csv")
    rows = []
    for image_id in IMAGE_IDS:
        pri = local.priors(image_id, top, family)
        record = {"image": image_id}
        confidences = []
        for base_index, prior in enumerate(pri, 1):
            candidates = [r for r in observations if int(r["image"]) == image_id
                          and r["kind"] == "base" and int(r["base"]) == base_index
                          and int(r["window_size"]) == 512 and r["annotation_overlap"].lower() == "false"]
            source = "normal_512_windows"
            if not candidates:
                candidates = [r for r in observations if int(r["image"]) == image_id
                              and r["kind"] == "base" and int(r["base"]) == base_index
                              and int(r["window_size"]) == 512]
                source = "all_512_windows_fallback"
            periods = np.asarray([float(r["period_px"]) for r in candidates], float)
            proms = np.asarray([float(r["prom_db"]) for r in candidates], float)
            fx = float(np.median([float(r["fx"]) for r in candidates])) if candidates else float(prior["fx"])
            fy = float(np.median([float(r["fy"]) for r in candidates])) if candidates else float(prior["fy"])
            period = float(np.median(periods)) if len(periods) else 1/math.hypot(fx, fy)
            iqr = float(np.percentile(periods, 75)-np.percentile(periods, 25)) if len(periods) else math.nan
            cv = float(np.std(periods)/np.mean(periods)) if len(periods) and np.mean(periods) else math.nan
            prom = float(np.median(proms)) if len(proms) else math.nan
            confidence = "high" if len(candidates) >= 10 and prom >= 6 and cv <= .15 else ("medium" if len(candidates) >= 3 and prom >= 4 else "low")
            confidences.append(confidence)
            record.update({
                f"g{base_index}_fx": fx, f"g{base_index}_fy": fy,
                f"T{base_index}_px": period, f"T{base_index}_iqr_px": iqr,
                f"T{base_index}_cv": cv, f"g{base_index}_median_prominence_db": prom,
                f"g{base_index}_windows": len(candidates), f"g{base_index}_source": source,
            })
            for w in (128, 256, 512): record[f"periods_g{base_index}_in_W{w}"] = w/period
        record["confidence"] = "low" if "low" in confidences else ("medium" if "medium" in confidences else "high")
        rows.append(record)
    return rows


def radial_selector(centers: np.ndarray, vectors: list[tuple[float, float]]) -> np.ndarray:
    radii = [math.hypot(*v) for v in vectors if FREQ[0] <= math.hypot(*v) <= FREQ[1]]
    select = np.ones(len(centers), bool)
    for radius in radii: select &= np.abs(centers-radius) > 1/WINDOW
    return select


def render_band_heatmaps(image_id: int, images: dict[str, np.ndarray], maps: dict,
                         xs: np.ndarray, ys: np.ndarray, masks: dict) -> None:
    if image_id not in REPRESENTATIVE: return
    for norm_key, heading in (("norm", "正規化あり"), ("raw", "正規化なし")):
        keys = [(condition, band) for condition in CONDITIONS for band in BANDS]
        normal_values = np.concatenate([maps[(condition, norm_key, band)][masks["normal"]]
                                        for condition, band in keys])
        lo, hi = np.percentile(normal_values[np.isfinite(normal_values)], [5, 99])
        fig, axes = plt.subplots(3, 5, figsize=(23, 13), constrained_layout=True)
        for row, condition in enumerate(CONDITIONS):
            for col, band in enumerate(BANDS):
                ax = axes[row, col]; ax.imshow(images[condition], cmap="gray", vmin=0, vmax=255)
                heat = ax.imshow(maps[(condition, norm_key, band)],
                    extent=basecmp.extent(xs, ys), cmap="inferno", vmin=lo, vmax=hi,
                    alpha=.68, interpolation="nearest")
                if masks["defect_pixel"].any():
                    ax.contour(masks["defect_pixel"].astype(np.uint8), [.5], colors="#38ff5a", linewidths=.7)
                if masks["exclusion_pixel"].any():
                    ax.contour(masks["exclusion_pixel"].astype(np.uint8), [.5], colors="#00e5ff", linewidths=.6)
                ax.set(xlim=(0, images[condition].shape[1]), ylim=(images[condition].shape[0], 0))
                ax.set_title(f"{CONDITIONS[condition]} / {band}"); ax.set_axis_off()
        fig.colorbar(heat, ax=axes, label="スペクトル差RMS (dB)", shrink=.65)
        fig.suptitle(f"image{image_id}：周波数帯域比較（{heading}、全15条件で共通色域 {lo:.3f}–{hi:.3f} dB）")
        fig.savefig(HEAT/f"image{image_id}_{norm_key}_bands.png", dpi=120); plt.close(fig)


def aggregate_detection(records: dict, config_meta: dict) -> list[dict]:
    rows = []
    for config, by_image in records.items():
        normal_all = np.concatenate([r["score"][r["normal"]] for r in by_image.values()])
        defect_all = np.concatenate([r["score"][r["defect"]] for r in by_image.values()])
        y = np.r_[np.zeros(len(normal_all)), np.ones(len(defect_all))]
        s = np.r_[normal_all, defect_all]
        result = dict(config_meta[config], normal_windows=len(normal_all), defect_windows=len(defect_all),
                      roc_auc=float(roc_auc_score(y, s)) if len(defect_all) else math.nan,
                      pr_auc=float(average_precision_score(y, s)) if len(defect_all) else math.nan,
                      separation=(float(np.median(defect_all)-np.median(normal_all))/float(np.std(normal_all))) if np.std(normal_all) else math.nan)
        for fpr in (.01, .05):
            normal_hits=[]; defect_hits=[]; detected=0; evaluable=0
            for image_id, r in by_image.items():
                training=[q["score"][q["normal"]] for other,q in by_image.items() if other != image_id]
                threshold=equal_image_quantile(training, 1-fpr)
                normal_hits.append(r["score"][r["normal"]] > threshold)
                defect_hits.append(r["score"][r["defect"]] > threshold)
                for instance in r["instances"]:
                    values=r["score"][instance["selection"]]
                    if len(values):
                        evaluable += 1; detected += int(np.any(values > threshold))
            nh=np.concatenate(normal_hits); dh=np.concatenate(defect_hits)
            pct=int(fpr*100)
            result.update({f"actual_fpr_{pct}pct":float(np.mean(nh)),
                           f"defect_window_tpr_{pct}pct":float(np.mean(dh)) if len(dh) else math.nan,
                           f"detected_instances_{pct}pct":detected,
                           f"evaluable_instances_{pct}pct":evaluable,
                           f"instance_tpr_{pct}pct":detected/evaluable if evaluable else math.nan})
        rows.append(result)
    return rows


def existing_window_outputs(periods: list[dict]) -> tuple[list[dict], list[dict]]:
    source = read_csv(ROOT/"output_window_size_vs_defect_size"/"summary"/"defect_window_metrics.csv")
    rows=[]
    period_by_image={int(r["image"]):r for r in periods}
    for r in source:
        image_id=int(r["image"]); window=int(r["window"])
        if image_id not in IMAGE_IDS or r["group"] != "grid" or r["scope"] != "common" or r["selection"] != "center": continue
        period=max(float(period_by_image[image_id]["T1_px"]), float(period_by_image[image_id]["T2_px"]))
        q=dict(r); q.update(reference_distance_px=128, normalisation="full-band total power",
                            frequency_band="0.01-0.50", periods_in_window=window/period,
                            provenance="existing output_window_size_vs_defect_size")
        rows.append(q)
    map_summary=read_csv(ROOT/"output_strict_b8_window_maps"/"window_size_summary.csv")
    strict=[r for r in map_summary if int(r["image_id"].replace("image","")) in IMAGE_IDS and int(r["window_size"]) in (64,128,256,512)]
    return rows, strict


def adaptive_from_strict(periods: list[dict]) -> list[dict]:
    period_by_image={int(r["image"]):r for r in periods}; records={"fixed_W128":{},"period_adaptive":{}}
    decision=[]
    for image_id in IMAGE_IDS:
        longest=max(float(period_by_image[image_id]["T1_px"]), float(period_by_image[image_id]["T2_px"]))
        requested=next((w for w in (64,128,256,512) if w >= 4*longest),512)
        available=[]
        for w in (64,128,256,512):
            path=ROOT/"output_strict_b8_window_maps"/"maps"/f"image{image_id}_w{w}.npz"
            if path.exists(): available.append(w)
        if not available or 128 not in available:
            decision.append({"image":image_id,"Tmax_px":longest,"rule":"smallest W >= 4*Tmax",
                             "requested_window":requested,"selected_window":"","fallback":"",
                             "periods_in_selected_window":"","label_used_for_selection":False,
                             "status":"excluded","reason":"no W128 strict-8 saved map for paired fixed/adaptive evaluation"})
            continue
        selected=requested if requested in available else max([w for w in available if w <= requested], default=max(available))
        decision.append({"image":image_id,"Tmax_px":longest,"rule":"smallest W >= 4*Tmax","requested_window":requested,
                         "selected_window":selected,"fallback":selected!=requested,
                         "periods_in_selected_window":selected/longest,
                         "label_used_for_selection":False,"status":"evaluated","reason":""})
        for name,w in (("fixed_W128",128),("period_adaptive",selected)):
            path=ROOT/"output_strict_b8_window_maps"/"maps"/f"image{image_id}_w{w}.npz"
            with np.load(path) as z:
                score=z["anomaly_map"].astype(np.float32); xs=z["x_centers"].astype(int); ys=z["y_centers"].astype(int)
            masks=labels.masks_for(image_id, grid.load_gray(image_id).shape, xs, ys, grid.DATA)
            valid=np.isfinite(score)
            records[name][image_id]={"score":score,"normal":masks["normal"]&valid,"defect":masks["defect"]&valid,
                                     "instances":[dict(x,selection=x["selection"]&valid) for x in masks["instances"]]}
    meta={"fixed_W128":{"experiment":"window_adaptation","condition":"fixed_W128","reference_distance":"W"},
          "period_adaptive":{"experiment":"window_adaptation","condition":"period_adaptive","reference_distance":"W"}}
    return decision, aggregate_detection(records, meta)


def defect_score_outputs() -> list[dict]:
    rows=[]
    for image_id in IMAGE_IDS:
        path=MAPS/f"image{image_id}_w128_band_scores.npz"
        with np.load(path) as z:
            xs=z["xs"].astype(int); ys=z["ys"].astype(int)
            arrays={k:z[k] for k in z.files if "__" in k}
        shape=grid.load_gray(image_id).shape
        masks=labels.masks_for(image_id,shape,xs,ys,grid.DATA)
        for key,score in arrays.items():
            condition,norm_key,band=key.split("__")
            for instance in masks["instances"]:
                values=score[instance["selection"] & np.isfinite(score)]
                if not len(values): continue
                rows.append({"image":image_id,"annotation_index":instance["annotation_index"],
                             "defect_type":instance["label"],"removal":condition,
                             "normalisation":norm_key,"band":band,"windows":len(values),
                             "score_median":float(np.median(values)),"score_max":float(np.max(values)),
                             "score_p95":float(np.percentile(values,95))})
    write_csv(OUT/"defect_score_comparison.csv",rows)
    return rows


def summary_figures(periods: list[dict]) -> None:
    detection=read_csv(OUT/"detection_metrics.csv")
    chosen=[]
    wanted=[("none","norm","radial_band","0.01-0.50"),
            ("fundamentals","norm","radial_band","0.01-0.50"),
            ("harmonics","norm","radial_band","0.01-0.50"),
            ("none","norm","radial_band","0.01-0.05"),
            ("none","raw","radial_band","0.01-0.50"),
            ("none","norm","vector_grid_excluded","0.01-0.50")]
    for target in wanted:
        chosen.append(next(r for r in detection if tuple(r[k] for k in ("removal","normalisation","feature_method","band"))==target))
    labels_short=["元/正規化/全帯域","基本波除去/全帯域","高調波込み/全帯域","元/正規化/0.01–0.05","元/非正規化/全帯域","元/2D格子bin除外"]
    x=np.arange(len(chosen)); fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
    ax.bar(x-.2,[float(r["pr_auc"]) for r in chosen],.4,label="PR-AUC")
    ax.bar(x+.2,[float(r["instance_tpr_1pct"]) for r in chosen],.4,label="傷単位TPR @ FPR1%")
    ax.set(xticks=x,xticklabels=labels_short,ylim=(0,1),ylabel="指標",title="主要条件の傷検出性能（W128、距離48 px、画像単位LOIO）")
    ax.tick_params(axis="x",rotation=20);ax.legend();fig.savefig(OUT/"figure_detection_condition_comparison.png",dpi=150);plt.close(fig)

    window_rows=read_csv(OUT/"window_size_comparison.csv")
    fig,ax=plt.subplots(figsize=(9,6),constrained_layout=True)
    for w,color in zip((64,128,192,256),("#4c78a8","#f58518","#54a24b","#e45756")):
        q=[r for r in window_rows if int(r["window"])==w]
        ax.scatter([float(r["periods_in_window"]) for r in q],[float(r["S"]) for r in q],s=35,alpha=.75,label=f"W={w}",color=color)
    ax.axhline(0,color="gray",lw=.8);ax.set(xlabel="窓内の長い側の格子周期数 W/Tmax",ylabel="傷–正常分離度 S",
        title="格子周期数と傷分離度（既存実験：参照距離128 px）");ax.legend()
    fig.savefig(OUT/"figure_grid_periods_vs_detection.png",dpi=150);plt.close(fig)

    # Preserve directly relevant prior evidence without modifying it.
    for source,name in ((ROOT/"analysis_fft_notch_spectral_mechanism"/"representative_fixed_window_spectra.png","figure5_representative_spectra_prior.png"),
                        (ROOT/"analysis_fft_notch_spectral_mechanism"/"representative_fixed_window_rois.png","figure6_representative_rois_prior.png")):
        if source.exists(): shutil.copy2(source,OUT/name)


def main() -> None:
    for folder in (OUT, MAPS, HEAT, WINDOW_FIG): folder.mkdir(parents=True, exist_ok=True)
    top=grid.read_csv(local.PRIOR/"fft_peak_pairs_top20.csv"); family=grid.read_csv(local.PRIOR/"grid_family_candidates.csv")
    periods=period_summary(top, family); write_csv(OUT/"grid_period_summary.csv", periods)
    centers=(np.arange(51)+.5)*BIN_WIDTH
    setup=basecmp.radial_setup(); centers=centers[setup["valid"]]
    records=defaultdict(dict); meta={}; band_rows=[]

    for image_id in IMAGE_IDS:
        print(f"image{image_id}: removal + W128 band analysis", flush=True)
        original, windows, _=local.analyze_image(image_id, top, family, detailed=(image_id==15))
        fundamentals,_=local.ola(original,windows,"local_base"); harmonics,_=local.ola(original,windows,"local_all")
        images={"none":original.astype(np.float32),"fundamentals":fundamentals.astype(np.float32),"harmonics":harmonics.astype(np.float32)}
        vectors=grid_vectors(image_id,top,family); all_maps={}; xs=ys=None
        for condition,image in images.items():
            spectra=spectra_four(image,vectors); step=SPACING//STRIDE
            xs=spectra["xs"][step:-step]; ys=spectra["ys"][step:-step]
            masks=labels.masks_for(image_id,image.shape,xs,ys,grid.DATA)
            for norm_key in ("norm","raw"):
                for band,(lo,hi) in BANDS.items():
                    select=(centers>=lo)&((centers<hi) if hi<.5 else (centers<=hi))
                    score=score_features(spectra[norm_key],select); key=(condition,norm_key,band)
                    all_maps[key]=score; config="|".join(key)
                    meta[config]={"experiment":"frequency_band","removal":condition,"normalisation":norm_key,"feature_method":"radial_band","band":band,"bin_count":int(select.sum())}
                    records[config][image_id]={"score":score,"normal":masks["normal"],"defect":masks["defect"],"instances":masks["instances"]}
                    band_rows.append({"image":image_id,**meta[config],"normal_median":float(np.median(score[masks["normal"]])),
                                      "defect_median":float(np.median(score[masks["defect"]])) if masks["defect"].any() else math.nan})
                rselect=radial_selector(centers,vectors)
                for method,source,select in (("radial_grid_excluded",norm_key,rselect),("vector_grid_excluded","vector_"+norm_key,spectra["vector_select"])):
                    score=score_features(spectra[source],select); config="|".join((condition,norm_key,method))
                    meta[config]={"experiment":"grid_frequency_exclusion","removal":condition,"normalisation":norm_key,"feature_method":method,"band":"0.01-0.50","bin_count":"image-specific; see settings"}
                    records[config][image_id]={"score":score,"normal":masks["normal"],"defect":masks["defect"],"instances":masks["instances"]}
            del spectra
        render_band_heatmaps(image_id,images,all_maps,xs,ys,masks)
        np.savez_compressed(MAPS/f"image{image_id}_w128_band_scores.npz",xs=xs,ys=ys,
                            **{("__".join(k)):v for k,v in all_maps.items()})
        del images,all_maps,windows,original,fundamentals,harmonics

    detection=aggregate_detection(records,meta)
    write_csv(OUT/"frequency_band_comparison.csv",band_rows)
    write_csv(OUT/"detection_metrics.csv",detection)
    notch=read_csv(ROOT/"analysis_all_images_grid_removal_fft_comparison"/"all_images_score_summary.csv")
    write_csv(OUT/"notch_effect_summary.csv",notch)
    window_rows,strict_rows=existing_window_outputs(periods)
    write_csv(OUT/"window_size_comparison.csv",window_rows)
    write_csv(OUT/"window_size_map_summary.csv",strict_rows)
    decisions,adaptive_metrics=adaptive_from_strict(periods)
    write_csv(OUT/"adaptive_window_decisions.csv",decisions)
    write_csv(OUT/"adaptive_window_metrics.csv",adaptive_metrics)
    defect_score_outputs(); summary_figures(periods)
    shutil.copy2(ROOT/"analysis_all_images_grid_removal_fft_comparison"/"all_images_contact_sheet.png",OUT/"figure1_grid_removal_contact_sheet.png")
    settings={"new_w128_experiment":{"images":list(IMAGE_IDS),"window":WINDOW,"stride":STRIDE,"reference_distance":SPACING,
              "reference_count":8,"reference_aggregation":"arithmetic mean per frequency in dB","bands":BANDS,
              "normalisation":"P/sum(P over 0.01<=radius<=0.50), then conjugate-weighted radial mean, then 10log10",
              "raw":"same without total-power division","score":"RMS(target dB spectrum - mean of 8 reference dB spectra)",
              "evaluation_area":"16x16 px at target centre","band_boundaries":"left closed, right open except final 0.50 endpoint"},
              "existing_window_evidence":{"defect_metrics":"W64/128/192/256, reference distance 128",
              "strict_maps":"W64/128/256/512, reference distance equals W"},
              "limitations":["fixed-48 window-size sweep beyond W128 was not recomputed","region-adaptive window fusion was not calibrated"]}
    (OUT/"settings.json").write_text(json.dumps(settings,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(OUT,flush=True)


if __name__ == "__main__": main()
