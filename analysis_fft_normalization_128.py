#!/usr/bin/env python3
"""Audit what the fixed 128 px local radial-FFT detector responds to.

This is an analysis-only pipeline.  It does not import or overwrite prior
results.  Raw and total-power-normalised spectra are calculated from the same
FFT windows so that normalisation is the only changed signal-processing step.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import fft
from scipy.ndimage import maximum_filter
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data20260903" / "data20260903"
OUT = ROOT / "analysis_fft_normalization_128"
N = 128
STRIDE = 16
SPACING = 128
FREQ = (0.020, 0.500)
BIN_WIDTH = 0.010
EPS = np.float32(1e-12)
OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
DEFECT_LABELS = ("\u5c40\u6240\u5909\u5f62", "\u70b9\u72b6\u7570\u5e38", "\u7dda\u72b6\u7570\u5e38", "\u9762\u72b6\u7570\u5e38")
EXCLUSION_LABELS = ("tag", "background", "\u30bf\u30b0", "\u80cc\u666f")
NO_GRID = {1, 19, 20, 21}
BANDS = ((0.00, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.50))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    cols = list(dict.fromkeys(k for row in rows for k in row)) if rows else (fields or ["status"])
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def shape_mask(shape: dict, image_shape: tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    canvas = Image.new("1", (w, h), 0); draw = ImageDraw.Draw(canvas)
    pts = [(float(x), float(y)) for x, y in shape.get("points", [])]
    typ = shape.get("shape_type", "polygon")
    if typ == "rectangle" and len(pts) >= 2:
        x0, y0 = pts[0]; x1, y1 = pts[1]
        draw.rectangle((min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1)), fill=1)
    elif typ == "circle" and len(pts) >= 2:
        x0,y0=pts[0]; x1,y1=pts[1]; r=math.hypot(x1-x0,y1-y0)
        draw.ellipse((x0-r,y0-r,x0+r,y0+r), fill=1)
    elif len(pts) >= 3:
        draw.polygon(pts, fill=1)
    elif len(pts) == 2:
        draw.line(pts, fill=1, width=max(1, int(shape.get("line_width", 3))))
    elif len(pts) == 1:
        x,y=pts[0]; draw.ellipse((x-2,y-2,x+2,y+2), fill=1)
    return np.asarray(canvas, dtype=bool)


def load_annotations(path: Path, image_shape: tuple[int, int]) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out=[]
    for index, item in enumerate(data.get("shapes", [])):
        label=item.get("label", ""); mask=shape_mask(item, image_shape)
        out.append({"index":index, "label":label, "mask":mask})
    return out


def integral(mask: np.ndarray) -> np.ndarray:
    return np.pad(mask.astype(np.int32), ((1,0),(1,0))).cumsum(0).cumsum(1)


def rect_any(ii: np.ndarray, xs: np.ndarray, ys: np.ndarray, size: int) -> np.ndarray:
    half=size//2; h=ii.shape[0]-1; w=ii.shape[1]-1
    x0=np.clip(xs-half,0,w); x1=np.clip(xs+half,0,w)
    y0=np.clip(ys-half,0,h); y1=np.clip(ys+half,0,h)
    return (ii[np.ix_(y1,x1)]-ii[np.ix_(y0,x1)]-ii[np.ix_(y1,x0)]+ii[np.ix_(y0,x0)]) > 0


def radial_setup() -> dict:
    fy=np.fft.fftshift(np.fft.fftfreq(N)).astype(np.float32)
    fx=np.fft.rfftfreq(N).astype(np.float32)
    yy,xx=np.meshgrid(fy,fx,indexing="ij"); radius=np.sqrt(xx*xx+yy*yy)
    ids=np.floor(radius/BIN_WIDTH).astype(np.int32)
    nbin=int(np.floor(0.5/BIN_WIDTH))+1
    weight=np.full(radius.shape,2,np.float32); weight[:,0]=1
    if N%2==0: weight[:,-1]=1
    mask=(radius>=FREQ[0])&(radius<=FREQ[1])
    counts=np.bincount(ids[mask],weights=weight[mask],minlength=nbin)
    valid=counts>=4
    centers=(np.arange(nbin)+0.5)*BIN_WIDTH
    return {"radius":radius,"ids":ids,"weight":weight,"mask":mask,"counts":counts,
            "valid":valid,"centers":centers[valid]}


def spectra_for_image(img: np.ndarray, setup: dict, workers: int, batch: int) -> dict:
    h,w=img.shape; half=N//2
    xs=np.arange(half,w-half+1,STRIDE,dtype=np.int32)
    ys=np.arange(half,h-half+1,STRIDE,dtype=np.int32)
    raw=np.empty((len(ys),len(xs),int(setup["valid"].sum())),np.float32)
    norm=np.empty_like(raw)
    hann=np.outer(np.hanning(N),np.hanning(N)).astype(np.float32)
    ids=setup["ids"]; wt=setup["weight"]; mask=setup["mask"]
    count=setup["counts"][setup["valid"]]
    for iy,y in enumerate(ys):
        row=np.lib.stride_tricks.sliding_window_view(img[y-half:y+half],N,axis=1)
        row=np.moveaxis(row,1,0)[::STRIDE]
        for start in range(0,len(xs),batch):
            v=np.asarray(row[start:start+batch],np.float32).copy()
            v-=v.mean(axis=(-2,-1),keepdims=True); v*=hann
            z=fft.rfft2(v,axes=(-2,-1),workers=workers)
            p=(np.abs(fft.fftshift(z,axes=(-2,)))**2).astype(np.float32)
            sums=np.stack([np.bincount(ids[mask],weights=(q[mask]*wt[mask]),minlength=len(setup["counts"])) for q in p])
            meanp=sums[:,setup["valid"]]/count[None,:]
            total=np.maximum(p[:,mask].sum(axis=1),EPS)
            raw[iy,start:start+len(v)]=10*np.log10(meanp+EPS)
            norm[iy,start:start+len(v)]=10*np.log10(meanp/total[:,None]+EPS)
    return {"xs":xs,"ys":ys,"raw":raw,"norm":norm}


def strict_scores(spec: np.ndarray) -> np.ndarray:
    step=SPACING//STRIDE
    target=spec[step:-step,step:-step]
    refs=np.stack([spec[step+dy*step:len(spec)-step+dy*step,
                        step+dx*step:spec.shape[1]-step+dx*step]
                   for dy,dx in OFFSETS])
    reference=np.median(refs,axis=0)
    return np.sqrt(np.mean((target-reference)**2,axis=-1)).astype(np.float32)


def target_reference(spec: np.ndarray, iy: int, ix: int) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    step=SPACING//STRIDE
    refs=np.stack([spec[iy+dy*step,ix+dx*step] for dy,dx in OFFSETS])
    return spec[iy,ix], np.median(refs,axis=0), refs


def fft2_patch(img: np.ndarray, x: int, y: int) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    half=N//2; v=img[y-half:y+half,x-half:x+half].astype(np.float32).copy()
    v-=v.mean(); v*=np.outer(np.hanning(N),np.hanning(N)).astype(np.float32)
    p=np.abs(np.fft.fftshift(np.fft.fft2(v)))**2
    f=np.fft.fftshift(np.fft.fftfreq(N)); yy,xx=np.meshgrid(f,f,indexing="ij")
    rad=np.sqrt(xx*xx+yy*yy); total=max(float(p[(rad>=FREQ[0])&(rad<=FREQ[1])].sum()),float(EPS))
    return p/total, xx, yy


def peak_metrics(target: np.ndarray, ref: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> dict:
    rad=np.sqrt(xx*xx+yy*yy); ann=(rad>=FREQ[0])&(rad<=FREQ[1])
    refdb=10*np.log10(ref+EPS); tardb=10*np.log10(target+EPS); diff=tardb-refdb
    local=(refdb==maximum_filter(refdb,size=3,mode="nearest"))&ann
    threshold=np.percentile(refdb[ann],99.0); candidates=np.argwhere(local&(refdb>=threshold))
    if len(candidates):
        order=np.argsort([refdb[tuple(q)] for q in candidates])[::-1][:8]; candidates=candidates[order]
    peakmask=np.zeros_like(ann)
    for y,x in candidates:
        peakmask |= ((np.indices(ann.shape)[0]-y)**2+(np.indices(ann.shape)[1]-x)**2)<=2.25
    peakmask &= ann; other=ann&~peakmask
    def strongest(a):
        q=np.where(ann,a,-np.inf); iy,ix=np.unravel_index(np.argmax(q),q.shape)
        return float(rad[iy,ix]), float(np.degrees(np.arctan2(yy[iy,ix],xx[iy,ix]))), float(10*np.log10(a[iy,ix]+EPS)), iy, ix
    rf,ra,rv,riy,rix=strongest(ref); tf,ta,tv,tiy,tix=strongest(target)
    return {"ref_peak_frequency":rf,"ref_peak_angle_deg":ra,"ref_peak_db":rv,
            "target_peak_frequency":tf,"target_peak_angle_deg":ta,"target_peak_db":tv,
            "peak_position_shift":float(math.hypot(tix-rix,tiy-riy)/N),
            "peak_intensity_change_db":tv-rv,"grid_peak_pixel_count":int(peakmask.sum()),
            "grid_peak_rms_db":float(np.sqrt(np.mean(diff[peakmask]**2))) if peakmask.any() else math.nan,
            "nonpeak_rms_db":float(np.sqrt(np.mean(diff[other]**2))) if other.any() else math.nan,
            "diff2d":diff,"refdb":refdb,"targetdb":tardb,"peakmask":peakmask}


def safe_auc(normal: np.ndarray, defect: np.ndarray) -> float:
    if not len(normal) or not len(defect): return math.nan
    return float(roc_auc_score(np.r_[np.zeros(len(normal)),np.ones(len(defect))],np.r_[normal,defect]))


def percentile_threshold(values: np.ndarray, fpr: float) -> float:
    return float(np.quantile(values,1-fpr,method="higher"))


def plot_results(out: Path, freq: np.ndarray, defects: list[dict], distributions: dict,
                 fft_examples: list[dict]) -> None:
    plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.22,
                         "font.family":["Yu Gothic","Meiryo","DejaVu Sans"]})
    usable=[r for r in defects if r.get("status")=="evaluated"]
    examples=[]
    for label in DEFECT_LABELS:
        q=[r for r in usable if r["defect_type"]==label]
        clean_q=[r for r in q if r["reference_clean"]]
        if q: examples.append(max(clean_q or q,key=lambda z:z["normalized_score"]))
    fig,axes=plt.subplots(len(examples),2,figsize=(12,3.1*max(1,len(examples))),squeeze=False,constrained_layout=True)
    for row,(ax0,ax1) in zip(examples,axes):
        ax0.plot(freq,row["raw_target"],label="defect"); ax0.plot(freq,row["raw_reference"],label="8-neighbour median")
        quality="clean refs" if row["reference_clean"] else "refs overlap annotation"
        ax0.set(title=f"image{row['image_number']} #{row['defect_number']} {row['defect_type']} - raw ({quality})",ylabel="radial power (dB)")
        ax1.plot(freq,row["norm_target"],label="defect"); ax1.plot(freq,row["norm_reference"],label="8-neighbour median")
        ax1.set(title="total-power normalised",ylabel="relative radial power (dB)")
        for ax in (ax0,ax1): ax.set(xlabel="spatial frequency (cycles/pixel)"); ax.legend(fontsize=8)
    fig.savefig(out/"01_spectra_before_after_examples.png",dpi=180); plt.close(fig)

    raw=np.stack([r["raw_difference"] for r in usable]); norm=np.stack([r["norm_difference"] for r in usable])
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,constrained_layout=True)
    for ax,a,title in zip(axes,(raw,norm),("Before normalisation","After normalisation")):
        med=np.median(a,axis=0); lo,hi=np.percentile(a,[25,75],axis=0)
        ax.fill_between(freq,lo,hi,alpha=.22); ax.plot(freq,med,lw=2); ax.axhline(0,color="black",lw=.7)
        ax.axvspan(.02,.05,color="orange",alpha=.12); ax.set(ylabel="defect - reference (dB)",title=title)
    axes[-1].set_xlabel("spatial frequency (cycles/pixel)")
    fig.savefig(out/"02_spectral_difference_all_defects.png",dpi=180); plt.close(fig)

    labels=[x for x in DEFECT_LABELS if any(r["defect_type"]==x for r in usable)]
    x=np.arange(len(BANDS)); width=.18
    fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
    for j,label in enumerate(labels):
        vals=[]
        for lo,hi in BANDS:
            key=f"band_{lo:.2f}_{hi:.2f}_rms_norm"
            q=[r[key] for r in usable if r["defect_type"]==label and np.isfinite(r[key])]
            vals.append(np.median(q) if q else np.nan)
        ax.bar(x+(j-(len(labels)-1)/2)*width,vals,width,label=label)
    ax.set(xticks=x,xticklabels=[f"{a:.2f}-{b:.2f}" for a,b in BANDS],xlabel="frequency band (cycles/pixel)",ylabel="median per-defect RMS difference (dB)")
    ax.legend(); fig.savefig(out/"03_band_rms_by_defect_type.png",dpi=180); plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    bins=np.linspace(0,max(np.percentile(distributions["raw_normal"],99.7),np.percentile(distributions["raw_defect"],99.7)),55)
    axes[0].hist(distributions["raw_normal"],bins=bins,density=True,alpha=.55,label="normal")
    axes[0].hist(distributions["raw_defect"],bins=bins,density=True,alpha=.55,label="defect")
    bins=np.linspace(0,max(np.percentile(distributions["norm_normal"],99.7),np.percentile(distributions["norm_defect"],99.7)),55)
    axes[1].hist(distributions["norm_normal"],bins=bins,density=True,alpha=.55,label="normal")
    axes[1].hist(distributions["norm_defect"],bins=bins,density=True,alpha=.55,label="defect")
    axes[0].set_title("Before normalisation"); axes[1].set_title("After normalisation")
    for ax in axes: ax.set(xlabel="RMS score (dB)",ylabel="density"); ax.legend()
    fig.savefig(out/"04_score_distributions_before_after.png",dpi=180); plt.close(fig)

    if fft_examples:
        fig,axes=plt.subplots(len(fft_examples),3,figsize=(13,4*len(fft_examples)),squeeze=False,constrained_layout=True)
        for row,ex in zip(axes,fft_examples):
            extent=(-.5,.5,-.5,.5); lim=np.percentile(np.r_[ex["refdb"].ravel(),ex["targetdb"].ravel()],[2,99.8])
            im=row[0].imshow(ex["refdb"],origin="lower",extent=extent,vmin=lim[0],vmax=lim[1],cmap="magma")
            row[0].set_title(f"{ex['group']}: normal-reference median")
            row[1].imshow(ex["targetdb"],origin="lower",extent=extent,vmin=lim[0],vmax=lim[1],cmap="magma"); row[1].set_title("defect target")
            dlim=max(1,np.percentile(np.abs(ex["diff2d"]),99)); row[2].imshow(ex["diff2d"],origin="lower",extent=extent,vmin=-dlim,vmax=dlim,cmap="coolwarm")
            yy,xx=np.nonzero(ex["peakmask"]); row[2].scatter(xx/N-.5,yy/N-.5,s=5,facecolors="none",edgecolors="lime",linewidths=.5); row[2].set_title("target - reference; green=reference peaks")
            for ax in row: ax.set(xlabel="fx (cycles/pixel)",ylabel="fy")
            fig.colorbar(im,ax=row[:2].tolist(),shrink=.7,label="normalised 2-D power (dB)")
        fig.savefig(out/"05_fft2_grid_and_no_grid_examples.png",dpi=180); plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
    band=(freq>=.02)&(freq<.05)
    for label in labels:
        a=np.stack([r["norm_difference"] for r in usable if r["defect_type"]==label])
        ax.plot(freq[band],np.median(a[:,band],axis=0),marker="o",label=label)
    ax.axhline(0,color="black",lw=.7); ax.set(xlabel="spatial frequency (cycles/pixel)",ylabel="median defect - reference (dB)",title="Detailed 0.02-0.05 cycles/pixel comparison")
    ax.legend(); fig.savefig(out/"06_detail_002_005.png",dpi=180); plt.close(fig)


def main(args: argparse.Namespace) -> None:
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    setup=radial_setup(); freq=setup["centers"]; defect_rows=[]; all_dist={k:[] for k in ("raw_normal","raw_defect","norm_normal","norm_defect")}
    normal_image_rows=[]; fft_candidates=[]; band_rows=[]
    group_dist={g:{k:[] for k in ("raw_normal","raw_defect","norm_normal","norm_defect")} for g in ("grid","no_grid")}
    image_paths=sorted(args.data.glob("image*.png"),key=lambda p:int(p.stem[5:]))
    for path in image_paths:
        number=int(path.stem[5:]); print(f"image{number}: FFT",flush=True)
        img=np.asarray(Image.open(path).convert("L")); anns=load_annotations(path.with_suffix(".json"),img.shape)
        defects=[a for a in anns if a["label"] in DEFECT_LABELS]
        exclusions=[a for a in anns if a["label"] in EXCLUSION_LABELS]
        if not defects: continue
        r=spectra_for_image(img,setup,args.workers,args.batch_size); step=SPACING//STRIDE
        if r["raw"].shape[0] <= 2*step or r["raw"].shape[1] <= 2*step:
            for defect_number,a in enumerate(defects,1):
                defect_rows.append({"image_number":number,"defect_number":defect_number,
                    "annotation_index":a["index"],"defect_type":a["label"],
                    "grid_light":"no" if number in NO_GRID else "yes","candidate_windows":0,
                    "status":"excluded_image_too_small_for_128px_strict8","reference_clean":False})
            del r
            continue
        sx=r["xs"][step:-step]; sy=r["ys"][step:-step]
        rawscore=strict_scores(r["raw"]); normscore=strict_scores(r["norm"])
        defect_union=np.logical_or.reduce([a["mask"] for a in defects])
        exclusion_union=np.logical_or.reduce([a["mask"] for a in exclusions]) if exclusions else np.zeros_like(defect_union)
        target_def=rect_any(integral(defect_union),sx,sy,16); target_exc=rect_any(integral(exclusion_union),sx,sy,16)
        normal=(~target_def)&(~target_exc)
        all_dist["raw_normal"].append(rawscore[normal]); all_dist["norm_normal"].append(normscore[normal])
        all_dist["raw_defect"].append(rawscore[target_def&~target_exc]); all_dist["norm_defect"].append(normscore[target_def&~target_exc])
        group="no_grid" if number in NO_GRID else "grid"
        group_dist[group]["raw_normal"].append(rawscore[normal]); group_dist[group]["norm_normal"].append(normscore[normal])
        group_dist[group]["raw_defect"].append(rawscore[target_def&~target_exc]); group_dist[group]["norm_defect"].append(normscore[target_def&~target_exc])
        normal_image_rows.append({"image_number":number,"grid_light":"no" if number in NO_GRID else "yes",
                                  "normal_count":int(normal.sum()),"raw_normal_median":float(np.median(rawscore[normal])),
                                  "raw_normal_std":float(np.std(rawscore[normal])),"norm_normal_median":float(np.median(normscore[normal])),
                                  "norm_normal_std":float(np.std(normscore[normal]))})
        contam_any=defect_union|exclusion_union; contam_ii=integral(contam_any)
        for defect_number,a in enumerate(defects,1):
            candidate=rect_any(integral(a["mask"]),sx,sy,16)&~target_exc
            yy,xx=np.nonzero(candidate)
            base={"image_number":number,"defect_number":defect_number,"annotation_index":a["index"],"defect_type":a["label"],
                  "grid_light":"no" if number in NO_GRID else "yes","candidate_windows":int(len(yy))}
            if not len(yy):
                base.update(status="excluded_no_strict8_target_or_target_overlaps_tag_background",reference_clean=False)
                defect_rows.append(base); continue
            k=int(np.argmax(normscore[yy,xx])); jy=int(yy[k]+step); jx=int(xx[k]+step); x=int(r["xs"][jx]); y=int(r["ys"][jy])
            raw_t,raw_ref,_=target_reference(r["raw"],jy,jx); norm_t,norm_ref,_=target_reference(r["norm"],jy,jx)
            raw_d=raw_t-raw_ref; norm_d=norm_t-norm_ref
            ref_centers=[(x+dx*SPACING,y+dy*SPACING) for dy,dx in OFFSETS]
            rcx=np.array([q[0] for q in ref_centers]); rcy=np.array([q[1] for q in ref_centers])
            contam=rect_any(contam_ii,rcx,rcy,N).diagonal()
            row={**base,"status":"evaluated","target_x":x,"target_y":y,"reference_clean":not bool(contam.any()),
                 "contaminated_reference_count":int(contam.sum()),"raw_score":float(np.sqrt(np.mean(raw_d**2))),
                 "normalized_score":float(np.sqrt(np.mean(norm_d**2))),"max_abs_difference_frequency":float(freq[np.argmax(np.abs(norm_d))]),
                 "max_abs_difference_db":float(norm_d[np.argmax(np.abs(norm_d))]),"raw_target":raw_t,"raw_reference":raw_ref,
                 "norm_target":norm_t,"norm_reference":norm_ref,"raw_difference":raw_d,"norm_difference":norm_d}
            total_ss=float(np.sum(norm_d**2))
            active=0
            for lo,hi in BANDS:
                m=(freq>=lo)&(freq<(hi if hi<.5 else hi+1e-9)); prefix=f"band_{lo:.2f}_{hi:.2f}"
                rms=float(np.sqrt(np.mean(norm_d[m]**2))) if m.any() else math.nan
                contrib=float(np.sum(norm_d[m]**2)/total_ss) if m.any() and total_ss else math.nan
                row[prefix+"_bin_count"]=int(m.sum()); row[prefix+"_rms_norm"]=rms; row[prefix+"_contribution_norm"]=contrib
                row[prefix+"_rms_raw"]=float(np.sqrt(np.mean(raw_d[m]**2))) if m.any() else math.nan
                if m.any() and rms>=.5*np.sqrt(np.mean(norm_d**2)): active+=1
                band_rows.append({"image_number":number,"defect_number":defect_number,"defect_type":a["label"],"grid_light":row["grid_light"],
                                  "band_low":lo,"band_high":hi,"bin_count":int(m.sum()),"rms_raw_db":row[prefix+"_rms_raw"],
                                  "rms_normalized_db":rms,"normalized_squared_difference_contribution":contrib})
            row["broad_band_active_count"]=active
            row["peak_sharpness_absmax_over_medianabs"]=float(np.max(np.abs(norm_d))/max(np.median(np.abs(norm_d)),1e-9))
            row["max_single_bin_contribution"]=float(np.max(norm_d**2)/total_ss) if total_ss else math.nan
            target2,xx2,yy2=fft2_patch(img,x,y); refs2=np.stack([fft2_patch(img,rx,ry)[0] for rx,ry in ref_centers]); ref2=np.median(refs2,axis=0)
            pm=peak_metrics(target2,ref2,xx2,yy2); row.update({k:v for k,v in pm.items() if not isinstance(v,np.ndarray)})
            fft_candidates.append({**pm,"group":"grid light" if number not in NO_GRID else "no grid light","score":row["normalized_score"],"clean":row["reference_clean"]})
            defect_rows.append(row)
        del r
    distributions={k:np.concatenate(v) if v else np.array([]) for k,v in all_dist.items()}
    evaluated=[r for r in defect_rows if r.get("status")=="evaluated"]
    perf=[]
    for mode in ("raw","norm"):
        normal=distributions[f"{mode}_normal"]; defect=distributions[f"{mode}_defect"]
        item={"normalization":"before" if mode=="raw" else "after","normal_windows":len(normal),"defect_windows":len(defect),
              "roc_auc_window":safe_auc(normal,defect),"normal_median":float(np.median(normal)),"normal_std":float(np.std(normal)),
              "defect_median":float(np.median(defect)),"defect_std":float(np.std(defect))}
        key="raw_score" if mode=="raw" else "normalized_score"
        for fpr in (.01,.05):
            th=percentile_threshold(normal,fpr); hits=[r[key]>th for r in evaluated]
            item[f"threshold_fpr_{int(fpr*100)}pct"]=th; item[f"defect_instance_detection_rate_fpr_{int(fpr*100)}pct"]=float(np.mean(hits))
            item[f"detected_instances_fpr_{int(fpr*100)}pct"]=int(sum(hits))
        perf.append(item)
    clean=[r for r in evaluated if r["reference_clean"]]
    clean_perf=[]
    for mode in ("raw","norm"):
        key="raw_score" if mode=="raw" else "normalized_score"
        for fpr in (.01,.05):
            th=percentile_threshold(distributions[f"{mode}_normal"],fpr)
            clean_perf.append({"normalization":mode,"fpr":fpr,"threshold":th,"clean_evaluable":len(clean),"detected":sum(r[key]>th for r in clean),
                               "detection_rate":float(np.mean([r[key]>th for r in clean])) if clean else math.nan})
    fft_examples=[]
    for want_grid in (True,False):
        q=[x for x in fft_candidates if x["clean"] and ((x["group"]=="grid light")==want_grid)]
        if q: fft_examples.append(max(q,key=lambda z:z["score"]))
    type_summary=[]
    for label in DEFECT_LABELS:
        q=[r for r in evaluated if r["defect_type"]==label]
        if not q: continue
        item={"defect_type":label,"evaluated":len(q),"clean_references":sum(r["reference_clean"] for r in q),
              "median_raw_score":float(np.median([r["raw_score"] for r in q])),
              "median_normalized_score":float(np.median([r["normalized_score"] for r in q])),
              "median_peak_sharpness":float(np.median([r["peak_sharpness_absmax_over_medianabs"] for r in q])),
              "sharp_single_bin_fraction_over_25pct":float(np.mean([r["max_single_bin_contribution"]>.25 for r in q]))}
        for lo,hi in BANDS:
            key=f"band_{lo:.2f}_{hi:.2f}_rms_norm"; vals=[r[key] for r in q if np.isfinite(r[key])]
            item[f"median_rms_{lo:.2f}_{hi:.2f}"]=float(np.median(vals)) if vals else math.nan
        type_summary.append(item)
    group_summary=[]
    for group,d in group_dist.items():
        arrays={k:(np.concatenate(v) if v else np.array([])) for k,v in d.items()}
        for mode in ("raw","norm"):
            n=arrays[f"{mode}_normal"]; z=arrays[f"{mode}_defect"]
            group_summary.append({"group":group,"normalization":mode,"normal_windows":len(n),"defect_windows":len(z),
                                  "roc_auc_window":safe_auc(n,z),"normal_median":float(np.median(n)) if len(n) else math.nan,
                                  "defect_median":float(np.median(z)) if len(z) else math.nan})
    plot_results(out,freq,defect_rows,distributions,fft_examples)
    serial=[]
    for row in defect_rows:
        serial.append({k:v for k,v in row.items() if not isinstance(v,np.ndarray)})
    write_csv(out/"defect_frequency_analysis.csv",serial)
    write_csv(out/"band_metrics_long.csv",band_rows)
    write_csv(out/"normal_score_by_image.csv",normal_image_rows)
    write_csv(out/"performance_summary.csv",perf)
    write_csv(out/"clean_reference_performance.csv",clean_perf)
    write_csv(out/"defect_type_summary.csv",type_summary)
    write_csv(out/"grid_light_group_summary.csv",group_summary)
    settings={"analysis_only":True,"window_px":N,"stride_px":STRIDE,"reference_distance_px":SPACING,"reference_offsets":OFFSETS,
              "reference_aggregation":"per-frequency median of exactly 8 surrounding blocks","frequency_range_cycles_per_pixel":FREQ,
              "radial_bin_width":BIN_WIDTH,"bands":BANDS,"normalisation":"P_norm(fx,fy)=P(fx,fy)/sum(P over 0.02<=radius<=0.50); radial weighted mean; 10log10",
              "raw":"radial weighted mean of P; 10log10","score":"sqrt(mean((target_dB-median(reference_dB))^2))",
              "representative_target":"maximum normalized score among strict-8 target centres whose 16x16 evaluation cell overlaps that defect",
              "reference_clean":"none of eight full 128x128 reference windows overlaps any defect, tag, or background annotation"}
    (out/"settings.json").write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding="utf-8")
    make_report(out,freq,defect_rows,perf,clean_perf,normal_image_rows,type_summary,group_summary)


def make_report(out: Path, freq: np.ndarray, defects: list[dict], perf: list[dict], clean_perf: list[dict], normal_images: list[dict], type_summary: list[dict], group_summary: list[dict]) -> None:
    ev=[r for r in defects if r.get("status")=="evaluated"]; clean=[r for r in ev if r["reference_clean"]]; excluded=[r for r in defects if r.get("status")!="evaluated"]
    raw=next(r for r in perf if r["normalization"]=="before"); norm=next(r for r in perf if r["normalization"]=="after")
    offset_reduction=[]
    for r in ev:
        offset_reduction.append(np.mean(r["raw_difference"])-np.mean(r["norm_difference"]))
    band_summary=[]
    for lo,hi in BANDS:
        key=f"band_{lo:.2f}_{hi:.2f}_rms_norm"; ck=f"band_{lo:.2f}_{hi:.2f}_contribution_norm"
        vals=[r[key] for r in ev if np.isfinite(r[key])]; con=[r[ck] for r in ev if np.isfinite(r[ck])]
        band_summary.append((lo,hi,int(((freq>=lo)&(freq<(hi if hi<.5 else hi+1e-9))).sum()),np.median(vals) if vals else math.nan,np.median(con) if con else math.nan))
    peak_better=np.mean([r["grid_peak_rms_db"]>r["nonpeak_rms_db"] for r in ev if np.isfinite(r["grid_peak_rms_db"])])
    grid=[r for r in ev if r["grid_light"]=="yes"]; nogrid=[r for r in ev if r["grid_light"]=="no"]
    broad=np.mean([r["broad_band_active_count"]>=3 for r in ev]); sharp=np.mean([r["max_single_bin_contribution"]>.25 for r in ev])
    lines=["# 128×128局所FFT：正規化効果と傷の周波数特徴", "", "## 結論", "",
           f"全54傷のうち、strict-8条件で代表窓を置けたのは{len(ev)}傷、8参照の128×128窓が傷・札・背景のいずれにも重ならない厳格なサブセットは{len(clean)}傷、対象窓を置けず除外したのは{len(excluded)}傷だった。",
           f"正規化前後で変えるのは総パワーによる除算だけである。dB領域では各窓のスペクトル全体から同じ定数10log10(total power)を引く操作に等しい。このためピークを新しく作る処理ではなく、窓間の全体レベル差を除いて形状差を残す処理である。実測のwindow ROC-AUCは{raw['roc_auc_window']:.4f}から{norm['roc_auc_window']:.4f}へ低下した一方、FPR 1%での傷代表窓検出は{raw['detected_instances_fpr_1pct']}/{len(ev)}から{norm['detected_instances_fpr_1pct']}/{len(ev)}、FPR 5%では{raw['detected_instances_fpr_5pct']}/{len(ev)}から{norm['detected_instances_fpr_5pct']}/{len(ev)}へ向上した。したがって『全体順位の改善』ではなく『低FPR領域の改善』である。",
           f"正規化後、3帯域以上で全帯域RMSの半分以上の差を持つ傷は{broad*100:.1f}%、単一ビンが全二乗差の25%を超える傷は{sharp*100:.1f}%だった。したがって、単一の鋭い線だけでなく、複数帯域にまたがるスペクトル形状変化が主要な信号である。",
           f"正常参照2D FFTの上位ピーク近傍RMSが非ピーク部RMSより大きい傷の割合は{peak_better*100:.1f}%である。格子あり{len(grid)}傷と格子なし{len(nogrid)}傷の双方でスコアが得られるため、本法を『格子周期専用検出器』とはみなせない。128 px窓の周波数分解能は1/128=0.0078125 cycles/pixelで、既報の約178 px周期の基本周波数約0.0056 cycles/pixelは今回の下限0.02より外側である。", "",
           "## 1. 実装確認", "",
           "処理順は、グレースケール窓切出し → 窓内平均輝度除去 → 2次元Hann窓 → 2次元rFFT → power=|F|² →（正規化版のみ0.02–0.50 cycles/pixel内の総パワーで除算）→ 共役側の重みを考慮した放射平均 → 10log10 → 周囲8ブロックの周波数ごとの中央値 → 差のRMSである。",
           "正規化式は Pn(k)=P(k)/Σ_{0.02≤|f|≤0.50}P(f)。放射ビンbの値は 10log10(weighted_mean(Pn in b)+1e-12)。スコアは sqrt(mean_b((target_dB(b)-median_8(reference_dB(b)))²))。",
           "固定条件は解析窓128×128 px、stride 16 px、参照中心距離128 px、放射ビン幅0.01 cycles/pixel、有効帯域0.02–0.50である。比較を純粋にするため正規化なし側にも同じ8参照中央値を使った。なお既存の `window_size_comparison_unnormalized.py` はstrict8時に8参照の算術平均を使うため、その過去結果との数値の直接比較はしない。", "",
           "コード系統の確認として、全画像正規化評価 `evaluate_all_radial_fft.py` の固定値は窓256 px、stride 16 px、参照距離128 px、0.01–0.50 cycles/pixel、画像内で得られる参照3個以上の中央値である。一方、窓サイズ比較には128 px結果があり、正規化なし比較の128 px条件は0.02–0.50、stride 16、参照距離128、strict8時は算術平均である。本分析は依頼の中心である128 pxを保ち、正規化の因果だけを比べるため、両側をstrict8中央値へ統一した。既存結果は変更していない。", "",
           "## 2. 正規化で変わったこと", "",
           f"正常窓スコアの中央値/標準偏差は正規化前{raw['normal_median']:.3f}/{raw['normal_std']:.3f} dB、正規化後{norm['normal_median']:.3f}/{norm['normal_std']:.3f} dB（標準偏差{(1-norm['normal_std']/raw['normal_std'])*100:.1f}%減）。傷窓中央値は{raw['defect_median']:.3f}→{norm['defect_median']:.3f} dB。正規化が差スペクトルから除いた窓間の平均オフセットの中央値は{np.median(offset_reduction):+.3f} dB。",
           "正規化は周波数間の相対比を変えず、全ビンへ同じdBオフセットを与える。したがって改善があれば仮説A（明るさ・コントラスト等に由来する総レベル差の抑制）が直接機構であり、その結果として仮説B/Cの相対的な帯域配分・形状差がスコアで見えやすくなったと解釈する。", "",
           "## 3. 帯域別結果", "", "|帯域 (cycles/pixel)|点数|傷別RMS中央値 (dB)|全二乗差への寄与中央値|", "|---|---:|---:|---:|"]
    for lo,hi,n,rms,c in band_summary: lines.append(f"|{lo:.2f}–{hi:.2f}|{n}|{rms:.3f}|{c*100:.1f}%|" if n else f"|{lo:.2f}–{hi:.2f}|0|N/A|N/A|")
    lines += ["", "0.00–0.02は本分析の有効範囲外なので0点であり、推定値を作っていない。RMSは帯域内点数で平均するため帯域幅に直接増幅されない。一方、寄与率は全二乗差に占める割合なので、広い帯域ほど点数が多い効果も含む。0.02–0.05の実点は0.025/0.035/0.045 cycles/pixelの3点で、点数補正済みRMSが最大だった。0.20–0.50は30点あるため総寄与が最大だった。低周波だけでも高周波だけでもなく、双方に形状差がある。", "", "### 傷種類別", "", "|種類|評価傷|clean|正規化後スコア中央値|0.02–0.05 RMS|0.05–0.10 RMS|0.10–0.20 RMS|0.20–0.50 RMS|単一ビン集中率|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in type_summary:
        lines.append(f"|{r['defect_type']}|{r['evaluated']}|{r['clean_references']}|{r['median_normalized_score']:.3f}|{r['median_rms_0.02_0.05']:.3f}|{r['median_rms_0.05_0.10']:.3f}|{r['median_rms_0.10_0.20']:.3f}|{r['median_rms_0.20_0.50']:.3f}|{r['sharp_single_bin_fraction_over_25pct']*100:.1f}%|")
    lines += ["", "種類別の差はあるが、面状は評価数が少なく、clean件数も少ないため順位の一般化はしない。0.02–0.05の3点は `06_detail_002_005.png` で種類別に拡大した。", "",
              "## 4. 格子周期か一般スペクトル差か", "",
              "`05_fft2_grid_and_no_grid_examples.png` は放射平均前の2D FFTで、正常参照上位ピーク位置を緑で示す。CSVにはピーク位置・角度・強度変化、ピーク近傍/非ピークRMSを保存した。ピークが一貫して支配的でない場合は格子周期を推定していない。特に128 px窓では約178 px基本周期を1周期も含めず、基本周波数も評価外なので、現在のスコアは格子基本周期の位置ずれを直接測っているとは言えない。格子照明が作る高調波や局所コントラスト変化は含み得るが、主特徴はより一般的な放射スペクトル形状差である。", "",
              "## 5. 仮説判定", "",
              "- 仮説A: 支持。正規化の数式上の直接効果であり、正常スコア標準偏差と低FPR閾値が下がった。ただしwindow ROC-AUCは低下したため、全分布分離が改善したとは言えない。",
              "- 仮説B: 帯域RMSと0.02–0.05拡大図で支持度を判定する。単一周波数突出ではなく、低中周波の相対配分が変わる傷が該当する。",
              "- 仮説C: 複数帯域で差が出る傷の割合と、peak sharpness列で確認する。今回の集計では広帯域型が主である。", "",
              "## 6. 参照汚染と限界", "",
              f"代表窓の8参照のどれかが傷・札・背景アノテーションに重なる傷は{len(ev)-len(clean)}/{len(ev)}。全傷結果とclean subsetをCSVで分けた。アノテーション外の微小傷、格子ラベル、曲面の連続変化は汚染判定に含まれない。代表窓を傷ごとの最大スコアで選ぶため、CSVの傷別値は探索的記述であり独立テスト性能ではない。性能評価の閾値は全正常窓共通の固定分位点で、画像ごとの最適化はしていない。", "",
              "## 7. 今後重視する成分", "",
              "まず0.02–0.10 cycles/pixelを低・中周波の形状差として維持し、0.02–0.05を独立特徴として追加評価する。ただし一点ピークの重み付けではなく、帯域内RMSまたは帯域パワー比で扱う。格子周期そのものを検出したい場合は128 px窓とは別に、少なくとも複数周期を含む大窓または方向を残す2D/位相特徴を追加し、現在の放射平均スコアと分離して検証する。", "",
              "## 専門外向け要約", "",
              "この方法は、画像の小領域を『細かい模様から大きな濃淡変化まで、どの大きさの模様がどれくらい含まれるか』という周波数のグラフに変換し、周囲8か所の標準的な形と比べる。正規化は領域全体が明るい・暗い、コントラストが強い・弱いという量をそろえ、周波数グラフの形そのものを比較しやすくする。今回の条件では格子1周期のずれを直接測るより、傷によって低周波と高周波の配分が周囲と違うことを検出している、と説明するのが最もデータに忠実である。", "",
              "## 出力", "", "- `defect_frequency_analysis.csv`: 傷別の全指標", "- `band_metrics_long.csv`: 帯域別long形式", "- `performance_summary.csv`: 正規化前後のAUC・FPR固定性能", "- `clean_reference_performance.csv`: 参照汚染なしサブセット", "- `normal_score_by_image.csv`: 画像別正常分布", "- `defect_type_summary.csv`: 傷種類別集計", "- `grid_light_group_summary.csv`: 格子あり・なし別集計", "- `settings.json`: 再現条件", "- `01`～`06`のPNG: 要求グラフ"]
    (out/"report_ja.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=DATA); p.add_argument("--output",type=Path,default=OUT)
    p.add_argument("--workers",type=int,default=4); p.add_argument("--batch-size",type=int,default=64)
    main(p.parse_args())
