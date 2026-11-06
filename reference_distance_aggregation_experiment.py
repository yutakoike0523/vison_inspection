#!/usr/bin/env python3
"""Reference-distance and mean/median aggregation experiments for local radial FFT."""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from scipy import fft
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

import grid_period_window_comparison as prior
import radial_fft_experiment as radial


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data20260903" / "data20260903"
OUT = ROOT / "output_reference_distance_and_aggregation"
STRIDE, CELL, BAND, BIN_WIDTH, EPS = 16, 16, (0.020, 0.500), 0.010, prior.EPS
OFFSETS = prior.OFFSETS
A_DISTANCES = (48, 64, 80, 96, 112, 128)
B_FACTORS = (("fixed48", None), ("0.5W", .5), ("0.75W", .75), ("1.0W", 1.0))
W_MODES = ("period_1x", "period_2x", "period_3x")
DEV_IMAGES, VALIDATION_IMAGES = tuple(range(2, 11)), tuple(range(11, 19))
REPRESENTATIVE_IMAGES = (7, 8, 11, 13)


def write_csv(path: Path, rows: list[dict], fields=None):
    cols = list(dict.fromkeys(k for r in rows for k in r)) if rows else (fields or ["status"])
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def safe(value):
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [safe(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if not np.isfinite(value) else float(value)
    return value


def distance_for(window: int, name: str, factor: float | None) -> int:
    return 48 if factor is None else prior.round16(window * factor, 16)


def condition_id(experiment: str, window_mode: str, distance_name: str, aggregation: str) -> str:
    return f"{experiment}__{window_mode}__{distance_name}__{aggregation}"


def aggregate_score(spectra: np.ndarray, distance: int, method: str) -> tuple[np.ndarray, np.ndarray]:
    step = distance // STRIDE; ny, nx = spectra.shape[:2]
    score = np.full((ny, nx), np.nan, np.float32); valid = np.zeros((ny, nx), bool)
    if step < 1 or ny <= 2*step or nx <= 2*step or spectra.shape[2] == 0: return score, valid
    target = spectra[step:ny-step, step:nx-step]
    refs = np.stack([spectra[step+dy*step:ny-step+dy*step, step+dx*step:nx-step+dx*step]
                     for dy, dx in OFFSETS])
    reference = refs.mean(axis=0) if method == "mean" else np.median(refs, axis=0)
    difference = target-reference
    score[step:ny-step, step:nx-step] = np.sqrt(np.mean(difference*difference, axis=2))
    valid[step:ny-step, step:nx-step] = True
    return score, valid


def spectra_for_window(image: np.ndarray, window: int, workers: int, requested_batch: int):
    h, w = image.shape; x0 = np.arange(0, w-window+1, STRIDE, dtype=np.int32); y0 = np.arange(0, h-window+1, STRIDE, dtype=np.int32)
    if not len(x0) or not len(y0): return None
    _, _, _, specs = radial.grids(window, BIN_WIDTH, [BAND]); spec = specs[BAND]; good = spec["valid"]
    out = np.empty((len(y0), len(x0), int(good.sum())), np.float32)
    h1 = np.hanning(window).astype(np.float32); hann = np.outer(h1, h1)
    batch = max(1, min(requested_batch, 2_000_000//(window*window)))
    for iy, top in enumerate(y0):
        for start in range(0, len(x0), batch):
            xx = x0[start:start+batch]
            patches = np.stack([image[top:top+window, left:left+window] for left in xx]).astype(np.float32)
            patches -= patches.mean(axis=(-2, -1), keepdims=True); patches *= hann
            z = fft.rfft2(patches, axes=(-2, -1), workers=workers)
            power = (np.abs(fft.fftshift(z, axes=(-2,)))**2).astype(np.float32)
            sums = power.reshape(len(power), -1) @ spec["matrix"]
            out[iy, start:start+len(xx)] = 10*np.log10(sums[:, good]/spec["counts"][good][None, :] + EPS)
    return {"xs": x0+window//2, "ys": y0+window//2, "spectra": out, "valid_bins": int(good.sum()), "batch": batch}


def auc(normal, defect):
    if not len(normal) or not len(defect): return math.nan, math.nan
    y = np.r_[np.zeros(len(normal), np.uint8), np.ones(len(defect), np.uint8)]; s = np.r_[normal, defect]
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def equal_image_quantile(arrays, q):
    arrays = [np.sort(np.asarray(a, float)) for a in arrays if len(a)]
    if not arrays: return None
    v = np.concatenate(arrays); weights = np.concatenate([np.full(len(a), 1/len(a)) for a in arrays])
    order = np.argsort(v); cumulative = np.cumsum(weights[order]); target = q*len(arrays)
    return float(v[order[min(int(np.searchsorted(cumulative, target)), len(v)-1)]])


def vals(r, mask): return r["score"][mask & r["valid"] & np.isfinite(r["score"])]


def stats(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return {"count": len(v), "mean": float(v.mean()) if len(v) else None, "max": float(v.max()) if len(v) else None,
            "p95": float(np.percentile(v, 95)) if len(v) else None, "p99": float(np.percentile(v, 99)) if len(v) else None}


def score_raster(r, shape): return prior.raster(r["score"], r["xs"], r["ys"], shape)


def common_for(records: dict, keys: list[str], shape: tuple[int, int]) -> np.ndarray:
    maps = [np.isfinite(score_raster(records[k], shape)) for k in keys if k in records]
    return np.logical_and.reduce(maps) if maps else np.zeros(shape, bool)


def attach_masks(r: dict, ann: dict):
    r["defect"] = prior.cell_overlap(ann["damage"], r["xs"], r["ys"])
    r["excluded"] = prior.cell_overlap(ann["exclusion"], r["xs"], r["ys"])
    r["normal"] = ~r["defect"] & ~r["excluded"]
    r["instances"] = [prior.cell_overlap(d["mask"], r["xs"], r["ys"]) for d in ann["defects"]]


def centre_selection(r: dict, defect: dict, valid: np.ndarray) -> tuple[int, int] | None:
    yy, xx = np.where(defect["mask"])
    if not len(xx): return None
    cx, cy = (xx.min()+xx.max())/2, (yy.min()+yy.max())/2
    candidate = valid & r["valid"]
    if not candidate.any(): return None
    gy, gx = np.where(candidate); dist = (r["xs"][gx]-cx)**2 + (r["ys"][gy]-cy)**2; j = int(np.argmin(dist))
    return int(gy[j]), int(gx[j])


def window_sum(mask: np.ndarray, cx: int, cy: int, window: int) -> int:
    h, w = mask.shape; half = window//2
    return int(mask[max(0,cy-half):min(h,cy+half), max(0,cx-half):min(w,cx+half)].sum())


def contamination_row(image_id: int, number: int, defect: dict, r: dict, iy: int, ix: int, target_kind: str, score: float) -> dict:
    cx, cy, window, distance = int(r["xs"][ix]), int(r["ys"][iy]), r["window"], r["distance"]
    target_area = window_sum(defect["mask"], cx, cy, window); ref_areas = []
    for dy, dx in OFFSETS: ref_areas.append(window_sum(defect["mask"], cx+dx*distance, cy+dy*distance, window))
    overlap = [max(0, window-abs(dx*distance))*max(0, window-abs(dy*distance))/(window*window) for dy, dx in OFFSETS]
    return {"image_id": f"image{image_id}", "defect_number": number, "defect_type": defect["type"],
            "condition_id": r["condition_id"], "experiment": r["experiment"], "window_mode": r["window_mode"],
            "window_size": window, "distance_name": r["distance_name"], "reference_distance_px": distance,
            "aggregation": r["aggregation"], "target_kind": target_kind, "target_center_x": cx, "target_center_y": cy,
            "rms_score": score, "target_defect_area_px": target_area, "target_defect_fraction": target_area/(window*window),
            "contaminated_reference_count": sum(a>0 for a in ref_areas), "reference_defect_area_sum_px": sum(ref_areas),
            "reference_defect_fraction_mean": float(np.mean(ref_areas))/(window*window),
            "reference_defect_fraction_max": max(ref_areas)/(window*window),
            "target_reference_overlap_mean": float(np.mean(overlap)), "target_reference_overlap_max": max(overlap)}


def scale(maps):
    available = [m[np.isfinite(m)] for m in maps if np.isfinite(m).any()]
    if not available: return 0, 1
    v = np.concatenate(available)
    lo, hi = np.percentile(v, [5, 99]); return float(lo), float(max(hi, lo+1e-6))


def heatmap_grid(image_id, image, ann, records, keys, titles, path, rows, cols, normalized=False):
    maps = [score_raster(records[k], image.shape) if k in records else np.full(image.shape, np.nan) for k in keys]
    shared = scale(maps); fig, axes = plt.subplots(rows, cols, figsize=(5.2*cols, 4.3*rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(rows, cols); last = None
    for ax, m, title in zip(axes.ravel(), maps, titles):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255); vmin, vmax = scale([m]) if normalized else shared
        last = ax.imshow(m, cmap="inferno", alpha=.67, vmin=vmin, vmax=vmax)
        if ann["damage"].any(): ax.contour(ann["damage"].astype(np.uint8), [.5], colors="#39ff66", linewidths=.8)
        ax.set_title(title); ax.set_axis_off()
        if normalized: fig.colorbar(last, ax=ax, fraction=.035, pad=.01)
    if last is not None and not normalized: fig.colorbar(last, ax=axes, fraction=.012, pad=.008, label="RMS dB difference")
    fig.suptitle(f"image{image_id} | green=damage | " + ("per-panel scale" if normalized else "shared scale"))
    fig.savefig(path, dpi=125); plt.close(fig)


def spectral_at(image, window, distance, cx, cy, aggregation):
    half=window//2; h1=np.hanning(window).astype(np.float32); hann=np.outer(h1,h1)
    centres=[(cx,cy)]+[(cx+dx*distance,cy+dy*distance) for dy,dx in OFFSETS]
    patches=np.stack([image[y-half:y+half,x-half:x+half] for x,y in centres]).astype(np.float32)
    patches-=patches.mean(axis=(-2,-1),keepdims=True); patches*=hann
    full=fft.fftshift(fft.fft2(patches,axes=(-2,-1)),axes=(-2,-1)); full_db=10*np.log10(np.abs(full)**2+EPS)
    halfz=fft.fftshift(fft.rfft2(patches,axes=(-2,-1)),axes=(-2,)); power=np.abs(halfz)**2
    _,_,_,specs=radial.grids(window,BIN_WIDTH,[BAND]); spec=specs[BAND]; good=spec["valid"]
    sums=power.reshape(9,-1)@spec["matrix"]; one=10*np.log10(sums[:,good]/spec["counts"][good][None,:]+EPS)
    reducer=np.mean if aggregation=="mean" else np.median
    ref2=reducer(full_db[1:],axis=0); ref1=reducer(one[1:],axis=0); target1=one[0]
    # Log-power has no phase and cannot itself be inverted.  For a spatial diagnostic,
    # retain the target phase and inverse-transform the target-minus-reference amplitude.
    reference_amplitude = np.sqrt(np.maximum(10.0**(ref2/10.0)-EPS, 0.0))
    target_phase = np.exp(1j*np.angle(full[0]))
    reference_with_target_phase = reference_amplitude*target_phase
    inverse_reference = fft.ifft2(fft.ifftshift(reference_with_target_phase,axes=(-2,-1)),axes=(-2,-1)).real
    amplitude_residual = (np.abs(full[0])-reference_amplitude)*target_phase
    inverse_residual = fft.ifft2(fft.ifftshift(amplitude_residual,axes=(-2,-1)),axes=(-2,-1)).real
    return {"centres":centres,"target2":full_db[0],"ref2":ref2,"diff2":full_db[0]-ref2,
            "f":(np.flatnonzero(good)+.5)*BIN_WIDTH,"target1":target1,"ref1":ref1,"diff1":target1-ref1,
            "inverse_reference":inverse_reference,"inverse_residual":inverse_residual}


def detailed_figures(image_id, image, ann, records, selected_pair, output):
    selected_mode, selected_distance = selected_pair
    rowspec=[("W128 d48 mean",condition_id("A","fixed128","d48","mean")),
             ("W128 d96 mean",condition_id("A","fixed128","d96","mean")),
             ("W128 d128 mean",condition_id("A","fixed128","d128","mean")),
             ("W128 d96 median",condition_id("C","fixed128","d96","median")),
             (f"{selected_mode} {selected_distance} mean",condition_id("B",selected_mode,selected_distance,"mean")),
             (f"{selected_mode} {selected_distance} median",condition_id("B",selected_mode,selected_distance,"median"))]
    for number, defect in enumerate(ann["defects"],1):
        # The W=128 rows share the coordinate valid at d=128; selected W rows share their own coordinate.
        anchor128=records[condition_id("A","fixed128","d128","mean")]
        pos128=centre_selection(anchor128,defect,anchor128["valid"])
        selected_record=records[condition_id("B",selected_mode,selected_distance,"mean")]
        possel=centre_selection(selected_record,defect,selected_record["valid"])
        fig,axes=plt.subplots(6,8,figsize=(33,24),constrained_layout=True); layout_fig,layout_axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
        for row,(label,key) in enumerate(rowspec):
            r=records[key]; pos=pos128 if r["window_mode"]=="fixed128" else possel
            if pos is None:
                for ax in axes[row]: ax.set_axis_off()
                layout_axes.ravel()[row].set_axis_off(); continue
            iy,ix=pos;cx,cy=int(r["xs"][ix]),int(r["ys"][iy]); detail=spectral_at(image,r["window"],r["distance"],cx,cy,r["aggregation"])
            half=r["window"]//2; span=half+r["distance"]+40;x0=max(0,cx-span);x1=min(image.shape[1],cx+span);y0=max(0,cy-span);y1=min(image.shape[0],cy+span)
            for ax in (axes[row,0],layout_axes.ravel()[row]):
                ax.imshow(image[y0:y1,x0:x1],cmap="gray",vmin=0,vmax=255)
                for j,(x,y) in enumerate(detail["centres"]):
                    color="red" if j==0 else "cyan";ax.add_patch(Rectangle((x-half-x0,y-half-y0),r["window"],r["window"],fill=False,edgecolor=color,linewidth=2 if j==0 else .7))
                ax.set_title(label+f"\nW={r['window']}, d={r['distance']}");ax.set_axis_off()
            combined=np.r_[detail["target2"].ravel(),detail["ref2"].ravel()];lo,hi=np.percentile(combined,[5,99.5]);lim=np.percentile(np.abs(detail["diff2"]),99)
            axes[row,1].imshow(detail["target2"],cmap="magma",vmin=lo,vmax=hi,origin="lower");axes[row,1].set_title("Target 2-D dB")
            axes[row,2].imshow(detail["ref2"],cmap="magma",vmin=lo,vmax=hi,origin="lower");axes[row,2].set_title(f"8-ref {r['aggregation']} 2-D dB")
            im=axes[row,3].imshow(detail["diff2"],cmap="coolwarm",vmin=-lim,vmax=lim,origin="lower");axes[row,3].set_title("Target - reference");fig.colorbar(im,ax=axes[row,3],fraction=.04)
            for c in (1,2,3):axes[row,c].set_xticks([]);axes[row,c].set_yticks([])
            axes[row,4].plot(detail["f"],detail["target1"],label="target",color="crimson");axes[row,4].plot(detail["f"],detail["ref1"],label="reference",color="navy");axes[row,4].legend(fontsize=7);axes[row,4].grid(alpha=.25);axes[row,4].set_title("Radial 1-D dB")
            axes[row,5].axhline(0,color="black",lw=.6);axes[row,5].plot(detail["f"],detail["diff1"],color="darkgreen");axes[row,5].grid(alpha=.25);axes[row,5].set_title(f"1-D diff RMS={np.sqrt(np.mean(detail['diff1']**2)):.3f}")
            ref_inv=detail["inverse_reference"];ref_lim=max(float(np.percentile(np.abs(ref_inv),99)),1e-12)
            ref_im=axes[row,6].imshow(ref_inv,cmap="coolwarm",vmin=-ref_lim,vmax=ref_lim)
            axes[row,6].set_title(f"IFFT 8-ref {r['aggregation']} 2-D dB\n(target phase retained)");axes[row,6].set_xticks([]);axes[row,6].set_yticks([]);fig.colorbar(ref_im,ax=axes[row,6],fraction=.04)
            inv=detail["inverse_residual"];inv_lim=max(float(np.percentile(np.abs(inv),99)),1e-12)
            inv_im=axes[row,7].imshow(inv,cmap="coolwarm",vmin=-inv_lim,vmax=inv_lim)
            axes[row,7].set_title("IFFT amplitude residual\n(target phase retained)");axes[row,7].set_xticks([]);axes[row,7].set_yticks([]);fig.colorbar(inv_im,ax=axes[row,7],fraction=.04)
        fig.suptitle(f"image{image_id} defect {number} {defect['type']} | centre fixed within W where possible | 0.02-0.50 cycles/pixel")
        fig.savefig(output/"spectral_comparison"/f"image{image_id}_defect{number:02d}_{defect['type']}.png",dpi=110);plt.close(fig)
        layout_fig.suptitle(f"image{image_id} defect {number}: red target, cyan 8 references")
        layout_fig.savefig(output/"reference_layout"/f"image{image_id}_defect{number:02d}_{defect['type']}.png",dpi=125);plt.close(layout_fig)


def evaluate(records_all, annotations, experiment_keys, common_masks, threshold_mode="loio", validation_only=False):
    image_rows=[];defect_rows=[];thresholds={}
    ids=VALIDATION_IMAGES if validation_only else tuple(records_all)
    for image_id in ids:
        for key in experiment_keys:
            if key not in records_all[image_id]:continue
            r=records_all[image_id][key]
            for scope in ("all_valid","common_region"):
                train=[]
                if threshold_mode=="development": train_ids=DEV_IMAGES
                else: train_ids=[i for i in records_all if i!=image_id]
                for other in train_ids:
                    if other not in records_all or key not in records_all[other]:continue
                    o=records_all[other][key];mask=o["valid"] if scope=="all_valid" else common_masks[other]
                    train.append(vals(o,mask&o["normal"]))
                thresholds[image_id,key,scope,1]=equal_image_quantile(train,.99);thresholds[image_id,key,scope,5]=equal_image_quantile(train,.95)
                valid=r["valid"] if scope=="all_valid" else common_masks[image_id];n=vals(r,valid&r["normal"]);d=vals(r,valid&r["defect"]);ns,ds=stats(n),stats(d);roc,pr=auc(n,d)
                t1,t5=thresholds[image_id,key,scope,1],thresholds[image_id,key,scope,5];det1=det5=evaluable=0
                for number,(defect,sel) in enumerate(zip(annotations[image_id]["defects"],r["instances"]),1):
                    dv=vals(r,valid&sel); fixed=centre_selection(r,defect,valid);fixed_score=float(r["score"][fixed]) if fixed is not None and np.isfinite(r["score"][fixed]) else None
                    hit1=bool(np.any(dv>t1)) if len(dv) and t1 is not None else None;hit5=bool(np.any(dv>t5)) if len(dv) and t5 is not None else None
                    if len(dv):evaluable+=1;det1+=hit1 is True;det5+=hit5 is True
                    st=stats(dv);defect_rows.append({"image_id":f"image{image_id}","defect_number":number,"defect_type":defect["type"],"condition_id":key,"experiment":r["experiment"],"window_mode":r["window_mode"],"window_size":r["window"],"distance_name":r["distance_name"],"reference_distance_px":r["distance"],"aggregation":r["aggregation"],"scope":scope,"status":"evaluable" if len(dv) else "out_of_scope","candidate_cells":len(dv),"maximum_score":st["max"],"mean_score":st["mean"],"p95_score":st["p95"],"fixed_center_score":fixed_score,"threshold_fpr1":t1,"detected_fpr1":hit1,"fixed_center_detected_fpr1":fixed_score>t1 if fixed_score is not None and t1 is not None else None,"threshold_fpr5":t5,"detected_fpr5":hit5})
                image_rows.append({"image_id":f"image{image_id}","condition_id":key,"experiment":r["experiment"],"window_mode":r["window_mode"],"window_size":r["window"],"distance_name":r["distance_name"],"reference_distance_px":r["distance"],"aggregation":r["aggregation"],"scope":scope,"valid_cells":int(valid.sum()),"normal_cells":len(n),"defect_cells":len(d),"defect_max":ds["max"],"defect_mean":ds["mean"],"normal_mean":ns["mean"],"normal_p95":ns["p95"],"normal_p99":ns["p99"],"roc_auc":roc,"pr_auc":pr,"threshold_fpr1":t1,"normal_fpr1":float(np.mean(n>t1)) if len(n) and t1 is not None else None,"defect_cell_tpr_fpr1":float(np.mean(d>t1)) if len(d) and t1 is not None else None,"threshold_fpr5":t5,"normal_fpr5":float(np.mean(n>t5)) if len(n) and t5 is not None else None,"defect_cell_tpr_fpr5":float(np.mean(d>t5)) if len(d) and t5 is not None else None,"detected_defects_fpr1":det1,"detected_defects_fpr5":det5,"evaluable_defects":evaluable,"unassessable_defects":len(annotations[image_id]["defects"])-evaluable})
    summary=[]
    for key in experiment_keys:
        sample=next((records_all[i][key] for i in ids if key in records_all[i]),None)
        if sample is None:continue
        for scope in ("all_valid","common_region"):
            nall=[];dall=[]
            for image_id in ids:
                if key not in records_all[image_id]:continue
                r=records_all[image_id][key];valid=r["valid"] if scope=="all_valid" else common_masks[image_id];nall.append(vals(r,valid&r["normal"]));dall.append(vals(r,valid&r["defect"]))
            n=np.concatenate([v for v in nall if len(v)]) if any(len(v) for v in nall) else np.array([]);d=np.concatenate([v for v in dall if len(v)]) if any(len(v) for v in dall) else np.array([]);roc,pr=auc(n,d)
            ir=[r for r in image_rows if r["condition_id"]==key and r["scope"]==scope];dr=[r for r in defect_rows if r["condition_id"]==key and r["scope"]==scope];ev=[r for r in dr if r["status"]=="evaluable"]
            summary.append({"aggregation_level":"overall","condition_id":key,"experiment":sample["experiment"],"window_mode":sample["window_mode"],"distance_name":sample["distance_name"],"aggregation":sample["aggregation"],"scope":scope,"image_subset":"validation" if validation_only else "all","evaluated_images":len(ir),"normal_cells":len(n),"defect_cells":len(d),"roc_auc":roc,"pr_auc":pr,"mean_image_roc_auc":float(np.nanmean([r["roc_auc"] for r in ir])),"mean_image_pr_auc":float(np.nanmean([r["pr_auc"] for r in ir])),"detected_defects_fpr1":sum(r["detected_fpr1"] is True for r in ev),"detected_defects_fpr5":sum(r["detected_fpr5"] is True for r in ev),"evaluable_defects":len(ev),"unassessable_defects":len(dr)-len(ev),"defect_detection_rate_fpr1":sum(r["detected_fpr1"] is True for r in ev)/len(ev) if ev else None,"defect_detection_rate_fpr5":sum(r["detected_fpr5"] is True for r in ev)/len(ev) if ev else None})
    return image_rows,defect_rows,summary,thresholds


def run(args):
    t0=time.perf_counter();output=args.output
    if output.exists():output=output.with_name(output.name+"_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True)
    for d in ("heatmaps_distance","heatmaps_window_distance","heatmaps_mean_vs_median","spectral_comparison","reference_layout","metric_plots","score_maps"): (output/d).mkdir()
    settings_rows=[r for i in range(2,19) for r in prior.settings_for(i)];lookup={(r["image_number"],r["window_mode"]):r for r in settings_rows}
    records_all={};annotations={};processing=[]
    for image_id in range(2,19):
        image=np.asarray(Image.open(args.data/f"image{image_id}.png").convert("L"),np.float32);ann=prior.load_annotation(args.data,image_id,image.shape);annotations[image_id]=ann;records={}
        modes_by_window=defaultdict(list);modes_by_window[128].append("fixed128")
        for mode in W_MODES:modes_by_window[lookup[image_id,mode]["actual_window_px"]].append(mode)
        for window,modes in modes_by_window.items():
            begin=time.perf_counter();computed=spectra_for_window(image,window,args.workers,args.batch_size)
            if computed is None:continue
            requests=[]
            if "fixed128" in modes:
                requests += [("A","fixed128",f"d{d}",d,"mean") for d in A_DISTANCES]
                requests += [("C","fixed128",f"d{d}",d,"median") for d in (48,96,128)]
            for mode in [m for m in modes if m in W_MODES]:
                for name,factor in B_FACTORS:
                    d=distance_for(window,name,factor)
                    requests += [("B",mode,name,d,"mean"),("B",mode,name,d,"median")]
            cache={}
            for experiment,mode,dname,d,aggregation in requests:
                cache_key=(d,aggregation)
                if cache_key not in cache:cache[cache_key]=aggregate_score(computed["spectra"],d,aggregation)
                score,valid=cache[cache_key];key=condition_id(experiment,mode,dname,aggregation)
                r={"condition_id":key,"experiment":experiment,"window_mode":mode,"window":window,"distance_name":dname,"distance":d,"aggregation":aggregation,"score":score.copy(),"valid":valid.copy(),"xs":computed["xs"],"ys":computed["ys"],"valid_bins":computed["valid_bins"]};attach_masks(r,ann);records[key]=r
            processing.append({"image_id":f"image{image_id}","window":window,"modes":";".join(modes),"fft_windows":len(computed["xs"])*len(computed["ys"]),"valid_bins":computed["valid_bins"],"seconds":time.perf_counter()-begin})
            print(f"image{image_id} W={window}: {time.perf_counter()-begin:.1f}s",flush=True)
        records_all[image_id]=records
        np.savez_compressed(output/"score_maps"/f"image{image_id}.npz",**{k:r["score"] for k,r in records.items()})
    write_csv(output/"processing_time.csv",processing)
    akeys=[condition_id("A","fixed128",f"d{d}","mean") for d in A_DISTANCES]
    bkeys=[condition_id("B",m,n,"mean") for m in W_MODES for n,_ in B_FACTORS]
    common_a={i:prior.cell_overlap(common_for(records_all[i],akeys,annotations[i]["damage"].shape),records_all[i][akeys[0]]["xs"],records_all[i][akeys[0]]["ys"]) for i in records_all}
    # Each condition can have a different grid. Store common as centre membership per condition later; evaluation expects one mask matching record.
    common_a_by={};common_b_by={}
    for i,recs in records_all.items():
        shape=annotations[i]["damage"].shape;ca=common_for(recs,akeys,shape);cb=common_for(recs,bkeys,shape)
        common_a_by[i]={k:recs[k]["valid"]&ca[recs[k]["ys"][:,None],recs[k]["xs"][None,:]] for k in akeys}
        common_b_by[i]={k:recs[k]["valid"]&cb[recs[k]["ys"][:,None],recs[k]["xs"][None,:]] for k in bkeys}
    # Adapter: evaluate accepts one per-image mask, so evaluate one condition at a time and concatenate.
    def eval_many(keys,by,**kw):
        ir=[];dr=[];su=[];th={}
        for key in keys:
            cm={i:by[i][key] for i in records_all};a,b,c,d=evaluate(records_all,annotations,[key],cm,**kw);ir+=a;dr+=b;su+=c;th.update(d)
        return ir,dr,su,th
    air,adr,asu,ath=eval_many(akeys,common_a_by)
    bir,bdr,bsu,bth=eval_many(bkeys,common_b_by)
    # Select B rule on development images only, common B region, mean aggregation, PR-AUC.
    dev_scores=[]
    for key in bkeys:
        n=[];d=[]
        for i in DEV_IMAGES:
            r=records_all[i][key];v=common_b_by[i][key];n.append(vals(r,v&r["normal"]));d.append(vals(r,v&r["defect"]))
        _,pr=auc(np.concatenate(n),np.concatenate(d));dev_scores.append((pr,key))
    selected_key=max(dev_scores)[1];_,selected_mode,selected_distance,_=selected_key.split("__")
    c_fixed_keys=[]
    for d in (48,96,128):c_fixed_keys += [condition_id("A","fixed128",f"d{d}","mean"),condition_id("C","fixed128",f"d{d}","median")]
    selected_keys=[condition_id("B",selected_mode,selected_distance,"mean"),condition_id("B",selected_mode,selected_distance,"median")]
    # C fixed common region.
    c_by={};sel_by={}
    for i,recs in records_all.items():
        shape=annotations[i]["damage"].shape;cm=common_for(recs,c_fixed_keys,shape);sm=common_for(recs,selected_keys,shape)
        c_by[i]={k:recs[k]["valid"]&cm[recs[k]["ys"][:,None],recs[k]["xs"][None,:]] for k in c_fixed_keys}
        sel_by[i]={k:recs[k]["valid"]&sm[recs[k]["ys"][:,None],recs[k]["xs"][None,:]] for k in selected_keys}
    cir,cdr,csu,cth=eval_many(c_fixed_keys,c_by)
    sir,sdr,ssu,sth=eval_many(selected_keys,sel_by,threshold_mode="development",validation_only=True)
    # Give C output rows unique identities; some underlying score maps intentionally reuse A/B calculations.
    for r in cir+cdr+csu:
        r["source_condition_id"] = r["condition_id"]; r["condition_id"] = "C_fixed__"+r["condition_id"]; r["experiment"] = "C_fixed"
    for r in sir+sdr+ssu:
        r["source_condition_id"] = r["condition_id"]; r["condition_id"] = "C_selected__"+r["condition_id"]; r["experiment"] = "C_selected"
    for r in ssu:r["selection_note"]="rule selected by development images 2-10; reported on validation images 11-18"
    # Per-defect and contamination across A, B mean, C fixed median, selected validation.
    all_defects=adr+bdr+[r for r in cdr if r["aggregation"]=="median"]+sdr
    contamination=[]
    relevant=set(akeys+bkeys+[k for k in c_fixed_keys if k.endswith("median")]+selected_keys)
    for i,recs in records_all.items():
        for key in relevant:
            if key not in recs:continue
            r=recs[key]
            for number,(defect,sel) in enumerate(zip(annotations[i]["defects"],r["instances"]),1):
                candidates=sel&r["valid"]&np.isfinite(r["score"])
                if candidates.any():iy,ix=np.unravel_index(np.argmax(np.where(candidates,r["score"],-np.inf)),r["score"].shape);contamination.append(contamination_row(i,number,defect,r,iy,ix,"maximum_score",float(r["score"][iy,ix])))
                pos=centre_selection(r,defect,r["valid"])
                if pos is not None:iy,ix=pos;contamination.append(contamination_row(i,number,defect,r,iy,ix,"fixed_center",float(r["score"][iy,ix])))
    write_csv(output/"reference_contamination.csv",contamination)
    # Correlations appended to summary.
    correlation=[]
    for agg in ("mean","median"):
        q=[r for r in contamination if r["aggregation"]==agg and r["target_kind"]=="fixed_center"]
        if q:
            rho,p=spearmanr([r["reference_defect_fraction_mean"] for r in q],[r["rms_score"] for r in q]);correlation.append({"aggregation":agg,"n":len(q),"spearman_contamination_vs_rms":rho,"p_value":p})
    # Type metrics.
    type_rows=[]
    for key in set(r["condition_id"] for r in all_defects):
        for scope in ("all_valid","common_region"):
            for typ in prior.TYPE_JA:
                q=[r for r in all_defects if r["condition_id"]==key and r["scope"]==scope and r["defect_type"]==typ];ev=[r for r in q if r["status"]=="evaluable"]
                sample=q[0] if q else None
                if sample:type_rows.append({"condition_id":key,"experiment":sample["experiment"],"window_mode":sample["window_mode"],"distance_name":sample["distance_name"],"aggregation":sample["aggregation"],"scope":scope,"defect_type":typ,"evaluable_defects":len(ev),"unassessable_defects":len(q)-len(ev),"detected_fpr1":sum(r["detected_fpr1"] is True for r in ev),"detection_rate_fpr1":sum(r["detected_fpr1"] is True for r in ev)/len(ev) if ev else None})
    # Valid region rows from image metrics.
    valid_rows=[]
    for r in air+bir+cir+sir:valid_rows.append({k:r.get(k) for k in ("image_id","condition_id","experiment","window_mode","window_size","distance_name","reference_distance_px","aggregation","scope","valid_cells","evaluable_defects","unassessable_defects")})
    write_csv(output/"experiment_a.csv",asu+air);write_csv(output/"experiment_b.csv",bsu+bir);write_csv(output/"experiment_c.csv",csu+ssu+cir+sir)
    write_csv(output/"per_defect_metrics.csv",all_defects);write_csv(output/"per_type_metrics.csv",type_rows);write_csv(output/"valid_region_summary.csv",valid_rows)
    overall=asu+bsu+csu+ssu
    for r in correlation:overall.append({"aggregation_level":"contamination_correlation",**r})
    write_csv(output/"overall_summary.csv",overall)
    # Inline rendering is optional because repeated large Matplotlib figures retain substantial memory on Windows.
    # The default reproducible path renders one image per process with reference_distance_aggregation_render.py.
    if args.render_inline:
        for i,recs in records_all.items():
            image=np.asarray(Image.open(args.data/f"image{i}.png").convert("L"),np.float32);ann=annotations[i]
            titles=[f"W=128 d={d} mean" for d in A_DISTANCES]
            heatmap_grid(i,image,ann,recs,akeys,titles,output/"heatmaps_distance"/f"image{i}_shared.png",2,3)
            heatmap_grid(i,image,ann,recs,akeys,titles,output/"heatmaps_distance"/f"image{i}_normalized.png",2,3,True)
            titles=[f"{m} {n}\nW={recs[condition_id('B',m,n,'mean')]['window']} d={recs[condition_id('B',m,n,'mean')]['distance']}" for m in W_MODES for n,_ in B_FACTORS]
            heatmap_grid(i,image,ann,recs,bkeys,titles,output/"heatmaps_window_distance"/f"image{i}_shared.png",3,4)
            heatmap_grid(i,image,ann,recs,bkeys,titles,output/"heatmaps_window_distance"/f"image{i}_normalized.png",3,4,True)
            cplot=c_fixed_keys+selected_keys;titles=[f"{recs[k]['window_mode']} {recs[k]['distance_name']} {recs[k]['aggregation']}" for k in cplot]
            heatmap_grid(i,image,ann,recs,cplot,titles,output/"heatmaps_mean_vs_median"/f"image{i}_shared.png",4,2)
            heatmap_grid(i,image,ann,recs,cplot,titles,output/"heatmaps_mean_vs_median"/f"image{i}_normalized.png",4,2,True)
            if i in REPRESENTATIVE_IMAGES:detailed_figures(i,image,ann,recs,(selected_mode,selected_distance),output)
    # Metric plots.
    aplot=[r for r in asu if r["scope"]=="common_region"]
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for ax,metric in zip(axes,("roc_auc","pr_auc","defect_detection_rate_fpr1")):ax.plot(A_DISTANCES,[next(r[metric] for r in aplot if r["distance_name"]==f"d{d}") for d in A_DISTANCES],marker="o");ax.set(xlabel="d (px)",title=metric);ax.grid(alpha=.3)
    fig.savefig(output/"metric_plots"/"experiment_a_distance.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,metric in zip(axes,("roc_auc","pr_auc")):
        for m in W_MODES:
            q=[r for r in bsu if r["scope"]=="common_region" and r["window_mode"]==m];ax.plot(range(4),[next(r[metric] for r in q if r["distance_name"]==n) for n,_ in B_FACTORS],marker="o",label=m)
        ax.set_xticks(range(4),[n for n,_ in B_FACTORS]);ax.set_title(metric);ax.grid(alpha=.3);ax.legend()
    fig.savefig(output/"metric_plots"/"experiment_b_window_distance.png",dpi=150);plt.close(fig)
    # Report.
    def best(rows,metric="pr_auc"):return max(rows,key=lambda r:r[metric] if r.get(metric) is not None and np.isfinite(r[metric]) else -1)
    aprimary=[r for r in asu if r["scope"]=="common_region"];bprimary=[r for r in bsu if r["scope"]=="common_region"];cvalidation=[r for r in ssu if r["scope"]=="common_region"]
    besta=best(aprimary);bestb=best(bprimary);bestc=best(cvalidation)
    d1=next(r for r in bprimary if r["window_mode"]=="period_1x" and r["distance_name"]=="1.0W");d2=next(r for r in bprimary if r["window_mode"]=="period_2x" and r["distance_name"]=="1.0W");d3=next(r for r in bprimary if r["window_mode"]=="period_3x" and r["distance_name"]=="1.0W")
    lines=["# 局所FFT：参照距離と参照スペクトル集約方法の検証","","## 現在の実装確認","","- 窓内平均輝度を引き、separable 2-D Hannを乗算。FFTサイズはW、ゼロパディングなし。","- SciPy `rfft2`（既定backward scaling）からpower=|F|²を計算。総パワー正規化なし。","- 共役成分の重みを付けて線形パワーを幅0.01 cycles/pixelで放射平均し、各窓を10log10でdB化。","- 参照は上下左右・斜めの8方向。基準meanは、**各参照窓をdB化した後、周波数ごとに算術平均（方式B）**。方式Aではない。medianも同じ8本のdBスペクトルに対する周波数別中央値。","- スコアは対象dB放射スペクトルと参照dBスペクトルの差の周波数方向RMS。ヒートマップは窓中心スコアを16×16 pxセルへ戻す。8参照が揃わない端部は無効。","","## 実験分離","","- A: W=128、mean、帯域・前処理固定でdだけ変更。","- B: W=1P/2P/3P、mean固定でd規則だけ変更。","- C: A/B後にmean/medianだけ変更。Bの規則はimage2〜10で選択し、image11〜18で独立検証。","", "## 主結果","",f"- 実験Aの共通領域PR-AUC最良: **d={besta['distance_name'][1:]} px** (ROC-AUC {besta['roc_auc']:.4f}, PR-AUC {besta['pr_auc']:.4f}, 傷検出@FPR1% {besta['detected_defects_fpr1']}/{besta['evaluable_defects']})。",f"- 実験Bの共通領域PR-AUC最良: **{bestb['window_mode']} / {bestb['distance_name']}** (PR-AUC {bestb['pr_auc']:.4f})。",f"- 選択用画像で選ばれた規則: **{selected_mode}/{selected_distance}**。独立検証でmean/medianの勝者は **{bestc['aggregation']}** (PR-AUC {bestc['pr_auc']:.4f})。","","## 最終回答","",f"- **Q1:** W=128のPR-AUC基準ではd={besta['distance_name'][1:]} px。ROC-AUC・固定FPR検出率との不一致はCSVに併記した。",f"- **Q2:** d=Wは一律には有効でない。1P/2P/3Pのd=1.0W PR-AUCはそれぞれ {d1['pr_auc']:.4f}/{d2['pr_auc']:.4f}/{d3['pr_auc']:.4f}。固定48および0.5W/0.75Wとの比較が必要。",f"- **Q3:** 2〜3Pで比例距離により改善する条件があれば参照混入・重なりが一因、dを広げても低い場合は大窓による傷成分希釈が主因と解釈する。実測の汚染率相関と条件別結果は `reference_contamination.csv` に保存した。",f"- **Q4:** 独立検証では **{bestc['aggregation']}** が優位。ただし選択規則と固定W128各dの結果を分けて判断する。",f"- **Q5:** 開発側で得た候補規則は {selected_mode}/{selected_distance}。画像別勝者の分散が大きい場合は単一規則ではなく周期帯または傷種別ルールが必要。","- **Q6:** `per_type_metrics.csv`に種類別結果を保存。指定範囲に面状異常は存在しないため、点状・線状・局所変形のみ比較可能。","","## 傷混入分析","",f"固定中心の傷混入率とRMSのSpearman相関: "+", ".join(f"{r['aggregation']} ρ={r['spearman_contamination_vs_rms']:.3f} (n={r['n']})" for r in correlation)+"。相関は窓サイズ・画像差を交絡するため因果効果ではない。最大スコア位置と傷中心固定位置を別行で保存した。","","## 注意","","- 閾値は画像別最適化せず、A/BはLOIO正常分位点、選択後Cはimage2〜10の正常分布だけで決めた。","- tag/backgroundは正常から除外し、傷との重複は傷優先。","- image2〜18の傷は36件、面状異常0件。条件ごとの評価不能傷は失敗と分離。","- 生スコアの窓間比較よりAUC・固定FPR・傷単位検出を優先。","",f"処理時間 { (time.perf_counter()-t0)/60:.1f} 分。実行条件は `settings.json`。"]
    (output/"report_ja.md").write_text("\n".join(lines),encoding="utf-8")
    runtime=time.perf_counter()-t0
    settings={"created_at":datetime.now().astimezone().isoformat(),"data":str(args.data.resolve()),"fft_order":"subtract mean -> 2D Hann -> rfft2 -> linear power -> radial mean -> 10log10 -> aggregate 8 dB spectra -> RMS difference","existing_reference_order":"B: dB per window, then arithmetic mean","band":BAND,"bin_width":BIN_WIDTH,"stride":STRIDE,"A_distances":A_DISTANCES,"B_factors":B_FACTORS,"development_images":DEV_IMAGES,"validation_images":VALIDATION_IMAGES,"selected_B_rule":{"window_mode":selected_mode,"distance_name":selected_distance},"thresholds":"A/B LOIO equal-image normal p99/p95; selected C development-only normal p99/p95","workers":args.workers,"batch_size":args.batch_size,"python":sys.version,"platform":platform.platform(),"command":" ".join(sys.argv),"runtime_seconds":runtime}
    (output/"settings.json").write_text(json.dumps(safe(settings),ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"completed {output.resolve()} {runtime/60:.1f} min",flush=True);return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data",type=Path,default=DATA);p.add_argument("--output",type=Path,default=OUT);p.add_argument("--workers",type=int,default=6);p.add_argument("--batch-size",type=int,default=64);p.add_argument("--render-inline",action="store_true",help="render all large figures in the analysis process; per-image renderer is safer on Windows");run(p.parse_args())


if __name__=="__main__":main()
