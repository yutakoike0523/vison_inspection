#!/usr/bin/env python3
"""Phase-preserving inverse-FFT visualisation for selected 128 px defects.

The existing detector and its outputs are read-only inputs.  This script keeps
complex FFT coefficients, applies radial masks, and reconstructs spatial
brightness deviations for interpretation.
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
from PIL import Image

import analysis_fft_normalization_128 as prior


ROOT=Path(__file__).resolve().parent
PRIOR=ROOT/"analysis_fft_normalization_128"
OUT=ROOT/"analysis_inverse_fft_bands_128"
BANDS=((0.00,0.02),(0.02,0.05),(0.05,0.10),(0.10,0.20),(0.20,0.50))
N=128; HALF=N//2
TYPE_EN={"\u5c40\u6240\u5909\u5f62":"local deformation","\u70b9\u72b6\u7570\u5e38":"point anomaly",
         "\u7dda\u72b6\u7570\u5e38":"linear anomaly","\u9762\u72b6\u7570\u5e38":"area anomaly"}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields=list(dict.fromkeys(k for r in rows for k in r)) if rows else ["status"]
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def load_prior_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
    for r in rows:
        for k in ("image_number","defect_number","target_x","target_y","candidate_windows","contaminated_reference_count"):
            if r.get(k): r[k]=int(float(r[k]))
        for k in ("band_0.02_0.05_rms_norm","normalized_score","raw_score"):
            if r.get(k): r[k]=float(r[k])
    return rows


def patch_at(img: np.ndarray,x: int,y: int) -> np.ndarray:
    return img[y-HALF:y+HALF,x-HALF:x+HALF].astype(np.float64)


def preprocess(patch: np.ndarray, hann: bool=True) -> np.ndarray:
    x=patch-patch.mean()
    if hann: x=x*np.outer(np.hanning(N),np.hanning(N))
    return x


def frequency_grid() -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    f=np.fft.fftshift(np.fft.fftfreq(N)); fy,fx=np.meshgrid(f,f,indexing="ij")
    return fx,fy,np.sqrt(fx*fx+fy*fy)


FX,FY,RADIUS=frequency_grid()
ANALYZED=RADIUS<=0.5


def band_mask(lo: float,hi: float) -> np.ndarray:
    return (RADIUS>=lo)&(RADIUS<(hi if hi<0.5 else hi+1e-12))


def transform(patch: np.ndarray,hann: bool=True) -> dict:
    x=preprocess(patch,hann); z=np.fft.fftshift(np.fft.fft2(x))
    recon={}
    for lo,hi in BANDS:
        filtered=np.where(band_mask(lo,hi),z,0)
        recon[(lo,hi)]=np.fft.ifft2(np.fft.ifftshift(filtered)).real
    return {"input":x,"fft":z,"power":np.abs(z)**2,"recon":recon}


def clean_reference_centres(img: np.ndarray,anns: list[dict],x: int,y: int) -> tuple[list[tuple[int,int]],list[bool]]:
    bad_masks=[a["mask"] for a in anns if a["label"] in prior.DEFECT_LABELS+prior.EXCLUSION_LABELS]
    bad=np.logical_or.reduce(bad_masks) if bad_masks else np.zeros(img.shape,bool)
    ii=prior.integral(bad)
    centres=[(x+dx*prior.SPACING,y+dy*prior.SPACING) for dy,dx in prior.OFFSETS]
    cx=np.array([q[0] for q in centres]); cy=np.array([q[1] for q in centres])
    contaminated=prior.rect_any(ii,cx,cy,N).diagonal().tolist()
    return centres,contaminated


def energy_share(t: dict,band=(0.02,0.05)) -> float:
    p=t["power"]; return float(p[band_mask(*band)].sum()/max(p.sum(),1e-30))


def choose_normal(img: np.ndarray,anns: list[dict],x: int,y: int) -> dict:
    centres,contam=clean_reference_centres(img,anns,x,y); candidates=[]
    for index,((rx,ry),bad) in enumerate(zip(centres,contam),1):
        if bad: continue
        p=patch_at(img,rx,ry); t=transform(p,True)
        candidates.append({"reference_index":index,"x":rx,"y":ry,"patch":p,"transform":t,"share":energy_share(t)})
    if not candidates: return {"status":"no_clean_reference","clean_count":0,"contaminated":contam}
    shares=np.array([c["share"] for c in candidates]); med=np.median(shares)
    chosen=candidates[int(np.argmin(np.abs(shares-med)))]
    chosen.update(status="clean_reference_selected",clean_count=len(candidates),contaminated=contam)
    return chosen


def select_examples(rows: list[dict],per_type: int) -> list[dict]:
    selected=[]
    image_cache={}
    for label in prior.DEFECT_LABELS:
        q=[r for r in rows if r.get("status")=="evaluated" and r["defect_type"]==label]
        q.sort(key=lambda r:r["band_0.02_0.05_rms_norm"],reverse=True)
        with_clean=[]; without=[]
        for r in q:
            n=r["image_number"]
            if n not in image_cache:
                img=np.asarray(Image.open(prior.DATA/f"image{n}.png").convert("L"))
                anns=prior.load_annotations(prior.DATA/f"image{n}.json",img.shape); image_cache[n]=(img,anns)
            img,anns=image_cache[n]
            normal=choose_normal(img,anns,r["target_x"],r["target_y"])
            item={**r,"normal_selection":normal}
            (with_clean if normal["status"]=="clean_reference_selected" else without).append(item)
        selected.extend((with_clean+without)[:per_type])
    return selected


def weighted_shape(a: np.ndarray) -> dict:
    w=np.abs(a); total=float(w.sum())
    if total<=0: return {"centroid_x":math.nan,"centroid_y":math.nan,"orientation_deg":math.nan,"anisotropy":math.nan,"spread_px":math.nan}
    yy,xx=np.indices(a.shape); cx=float((w*xx).sum()/total); cy=float((w*yy).sum()/total)
    dx=xx-cx; dy=yy-cy
    cov=np.array([[(w*dx*dx).sum(),(w*dx*dy).sum()],[(w*dx*dy).sum(),(w*dy*dy).sum()]])/total
    vals,vecs=np.linalg.eigh(cov); order=np.argsort(vals)[::-1]; vals=vals[order]; v=vecs[:,order[0]]
    return {"centroid_x":cx,"centroid_y":cy,"orientation_deg":float(np.degrees(np.arctan2(v[1],v[0]))),
            "anisotropy":float(1-vals[1]/max(vals[0],1e-30)),"spread_px":float(np.sqrt(vals.sum()))}


def band_metrics(t: dict,lo: float,hi: float) -> dict:
    a=t["recon"][(lo,hi)]; energy=float(np.sum(a*a)); full=float(np.sum(t["input"]**2))
    analyzed=float(t["power"][ANALYZED].sum()/(N*N))
    shape=weighted_shape(a)
    return {"rms":float(np.sqrt(np.mean(a*a))),"minimum":float(a.min()),"maximum":float(a.max()),
            "peak_to_peak":float(a.max()-a.min()),"energy":energy,"total_energy":full,
            "energy_fraction_total":energy/max(full,1e-30),"energy_fraction_analyzed_0_05":energy/max(analyzed,1e-30),**shape}


def panel_data(example: dict) -> dict:
    n=example["image_number"]; img=np.asarray(Image.open(prior.DATA/f"image{n}.png").convert("L"))
    anns=prior.load_annotations(prior.DATA/f"image{n}.json",img.shape)
    defect_patch=patch_at(img,example["target_x"],example["target_y"])
    dt=transform(defect_patch,True); dm=transform(defect_patch,False)
    normal=example["normal_selection"]
    if normal["status"]=="clean_reference_selected":
        normal_patch=normal["patch"]; nt=normal["transform"]; nm=transform(normal_patch,False)
    else: normal_patch=None; nt=None; nm=None
    return {"image":img,"annotations":anns,"defect_patch":defect_patch,"defect":dt,"defect_mean_only":dm,
            "normal_patch":normal_patch,"normal":nt,"normal_mean_only":nm,"normal_selection":normal}


def show_original(ax,a,title,vmin=None,vmax=None):
    if a is None: ax.text(.5,.5,"No clean reference",ha="center",va="center",transform=ax.transAxes); ax.set_title(title); ax.axis("off"); return
    ax.imshow(a,cmap="gray",vmin=vmin,vmax=vmax); ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])


def show_diverging(ax,a,title,limit=None):
    if a is None: ax.text(.5,.5,"No clean reference",ha="center",va="center",transform=ax.transAxes); ax.set_title(title); ax.axis("off"); return None
    if limit is None: limit=max(float(np.percentile(np.abs(a),99.5)),1e-9)
    im=ax.imshow(a,cmap="coolwarm",vmin=-limit,vmax=limit); ax.set_title(f"{title}\n±{limit:.2f} gray levels"); ax.set_xticks([]); ax.set_yticks([]); return im


def comparison_figure(ex: dict,d: dict,path: Path,shared: bool) -> None:
    fig,axes=plt.subplots(2,7,figsize=(20,6),constrained_layout=True)
    patches=[d["defect_patch"],d["normal_patch"]]; trans=[d["defect"],d["normal"]]
    names=["defect","normal"]
    if shared and d["normal_patch"] is not None:
        vals=np.r_[d["defect_patch"].ravel(),d["normal_patch"].ravel()]; gray=np.percentile(vals,[1,99])
    else: gray=None
    for row,(patch,t,name) in enumerate(zip(patches,trans,names)):
        if gray is None and patch is not None: scale=np.percentile(patch,[1,99])
        else: scale=gray
        show_original(axes[row,0],patch,f"{name}: original",*(scale if scale is not None else (None,None)))
        if shared and d["normal"] is not None:
            lim=max(np.percentile(np.abs(d["defect"]["input"]),99.5),np.percentile(np.abs(d["normal"]["input"]),99.5))
        else: lim=None
        show_diverging(axes[row,1],None if t is None else t["input"],"mean removed + Hann",lim)
        for col,band in enumerate(BANDS,2):
            if shared and d["normal"] is not None:
                blim=max(np.percentile(np.abs(d["defect"]["recon"][band]),99.5),np.percentile(np.abs(d["normal"]["recon"][band]),99.5))
            else: blim=None
            show_diverging(axes[row,col],None if t is None else t["recon"][band],f"{band[0]:.2f}-{band[1]:.2f} cyc/px",blim)
    status=d["normal_selection"]["status"]; fig.suptitle(f"image{ex['image_number']} defect {ex['defect_number']} {TYPE_EN[ex['defect_type']]} | {status} | {'shared scales' if shared else 'panel auto scales'}",fontsize=14)
    fig.savefig(path,dpi=170); plt.close(fig)


def fft_ring_figure(ex: dict,d: dict,path: Path) -> None:
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True); extent=(-.5,.5,-.5,.5)
    if d["normal"] is not None:
        powers=[d["normal"]["power"],d["defect"]["power"]]
        normed=[10*np.log10(p/max(p[ANALYZED].sum(),1e-30)+1e-20) for p in powers]
        lo,hi=np.percentile(np.r_[normed[0][ANALYZED],normed[1][ANALYZED]],[2,99.7])
        for ax,a,title in zip(axes[:2],normed,("normal 2-D FFT power","defect 2-D FFT power")):
            im=ax.imshow(a,origin="lower",extent=extent,cmap="magma",vmin=lo,vmax=hi)
            ax.contour(FX,FY,RADIUS,levels=[.02,.05],colors=["cyan","lime"],linewidths=1.2)
            ax.set(title=title,xlabel="fx (cycles/pixel)",ylabel="fy")
        diff=normed[1]-normed[0]; lim=max(np.percentile(np.abs(diff[ANALYZED]),99),1)
        im2=axes[2].imshow(diff,origin="lower",extent=extent,cmap="coolwarm",vmin=-lim,vmax=lim)
        axes[2].contour(FX,FY,RADIUS,levels=[.02,.05],colors=["cyan","lime"],linewidths=1.3)
        axes[2].set(title="defect - normal (dB)\nring=.02-.05",xlabel="fx",ylabel="fy")
        fig.colorbar(im,ax=axes[:2].tolist(),shrink=.75,label="normalised power (dB)"); fig.colorbar(im2,ax=axes[2],shrink=.75,label="difference (dB)")
    else:
        a=10*np.log10(d["defect"]["power"]/max(d["defect"]["power"][ANALYZED].sum(),1e-30)+1e-20)
        axes[0].text(.5,.5,"No clean normal reference",ha="center",va="center",transform=axes[0].transAxes); axes[0].axis("off")
        axes[1].imshow(a,origin="lower",extent=extent,cmap="magma"); axes[1].contour(FX,FY,RADIUS,levels=[.02,.05],colors=["cyan","lime"]); axes[1].set_title("defect FFT; ring=.02-.05")
        mask=band_mask(.02,.05); axes[2].imshow(mask,origin="lower",extent=extent,cmap="gray"); axes[2].set_title("inverse-FFT coefficient mask")
    fig.suptitle(f"image{ex['image_number']} defect {ex['defect_number']} {TYPE_EN[ex['defect_type']]}")
    fig.savefig(path,dpi=180); plt.close(fig)


def type_montages(examples: list[tuple[dict,dict]],out: Path) -> None:
    for label in prior.DEFECT_LABELS:
        q=[z for z in examples if z[0]["defect_type"]==label]
        if not q: continue
        fig,axes=plt.subplots(3,len(q),figsize=(4*len(q),10),squeeze=False,constrained_layout=True)
        for col,(ex,d) in enumerate(q):
            show_original(axes[0,col],d["defect_patch"],f"image{ex['image_number']} #{ex['defect_number']} original")
            lim=np.percentile(np.abs(d["defect"]["recon"][(.02,.05)]),99.5)
            show_diverging(axes[1,col],d["defect"]["recon"][(.02,.05)],"defect 0.02-0.05",lim)
            show_diverging(axes[2,col],None if d["normal"] is None else d["normal"]["recon"][(.02,.05)],"normal 0.02-0.05",lim if d["normal"] is not None else None)
        fig.suptitle(f"{TYPE_EN[label]}: selected examples (defect scale used for both band images)",fontsize=14)
        safe={"\u5c40\u6240\u5909\u5f62":"local_deformation","\u70b9\u72b6\u7570\u5e38":"point","\u7dda\u72b6\u7570\u5e38":"line","\u9762\u72b6\u7570\u5e38":"area"}[label]
        fig.savefig(out/f"montage_{safe}.png",dpi=170); plt.close(fig)


def hann_figure(examples: list[tuple[dict,dict]],out: Path) -> None:
    chosen=[]
    for label in prior.DEFECT_LABELS:
        q=[z for z in examples if z[0]["defect_type"]==label and z[1]["normal"] is not None]
        if q: chosen.append(q[0])
    fig,axes=plt.subplots(len(chosen),4,figsize=(14,3.5*len(chosen)),squeeze=False,constrained_layout=True)
    for row,(ex,d) in enumerate(chosen):
        arrays=[d["defect_mean_only"]["recon"][(.02,.05)],d["defect"]["recon"][(.02,.05)],d["normal_mean_only"]["recon"][(.02,.05)],d["normal"]["recon"][(.02,.05)]]
        lim=max(np.percentile(np.abs(a),99.5) for a in arrays)
        for ax,a,title in zip(axes[row],arrays,("defect: mean only","defect: mean+Hann","normal: mean only","normal: mean+Hann")):
            show_diverging(ax,a,title,lim)
        axes[row,0].set_ylabel(f"{TYPE_EN[ex['defect_type']]}\nimage{ex['image_number']} #{ex['defect_number']}")
    fig.suptitle("Hann-window influence on 0.02-0.05 cycles/pixel reconstruction (shared scale per row)")
    fig.savefig(out/"hann_window_influence.png",dpi=180); plt.close(fig)


def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True,exist_ok=True); (args.output/"examples").mkdir(exist_ok=True)
    rows=load_prior_rows(args.prior/"defect_frequency_analysis.csv"); selected=select_examples(rows,args.examples_per_type)
    metrics=[]; selections=[]; prepared=[]
    for seq,ex in enumerate(selected,1):
        print(f"{seq}/{len(selected)} image{ex['image_number']} defect{ex['defect_number']}",flush=True)
        d=panel_data(ex); prepared.append((ex,d)); stem=f"image{ex['image_number']}_defect{ex['defect_number']}_{seq:02d}"
        comparison_figure(ex,d,args.output/"examples"/f"{stem}_shared_scale.png",True)
        comparison_figure(ex,d,args.output/"examples"/f"{stem}_auto_scale.png",False)
        fft_ring_figure(ex,d,args.output/"examples"/f"{stem}_fft_ring.png")
        normal=d["normal_selection"]
        selections.append({"selection_order":seq,"image_number":ex["image_number"],"defect_number":ex["defect_number"],"defect_type":ex["defect_type"],
                           "grid_light":"no" if ex["image_number"] in prior.NO_GRID else "yes","prior_band_002_005_rms_db":ex["band_0.02_0.05_rms_norm"],
                           "normal_status":normal["status"],"clean_reference_count":normal["clean_count"],"normal_reference_index":normal.get("reference_index",""),
                           "defect_x":ex["target_x"],"defect_y":ex["target_y"],"normal_x":normal.get("x",""),"normal_y":normal.get("y","")})
        for lo,hi in BANDS:
            dm=band_metrics(d["defect"],lo,hi); dmo=band_metrics(d["defect_mean_only"],lo,hi)
            nm=band_metrics(d["normal"],lo,hi) if d["normal"] is not None else None
            nmo=band_metrics(d["normal_mean_only"],lo,hi) if d["normal_mean_only"] is not None else None
            row={"image_number":ex["image_number"],"defect_number":ex["defect_number"],"defect_type":ex["defect_type"],"grid_light":"no" if ex["image_number"] in prior.NO_GRID else "yes",
                 "band_low":lo,"band_high":hi,"normal_status":normal["status"],"clean_reference_count":normal["clean_count"],
                 **{f"defect_{k}":v for k,v in dm.items()},"defect_mean_only_rms":dmo["rms"],"defect_hann_to_mean_only_rms_ratio":dm["rms"]/max(dmo["rms"],1e-30),
                 "prior_radial_spectrum_band_rms_db":ex["band_0.02_0.05_rms_norm"] if (lo,hi)==(.02,.05) else ""}
            if nm is not None:
                row.update({f"normal_{k}":v for k,v in nm.items()}); row.update({"normal_mean_only_rms":nmo["rms"],"normal_hann_to_mean_only_rms_ratio":nm["rms"]/max(nmo["rms"],1e-30),
                    "band_energy_difference":dm["energy"]-nm["energy"],"band_energy_ratio_defect_over_normal":dm["energy"]/max(nm["energy"],1e-30),
                    "normalized_band_energy_fraction_difference":dm["energy_fraction_total"]-nm["energy_fraction_total"],
                    "normalized_band_energy_fraction_ratio":dm["energy_fraction_total"]/max(nm["energy_fraction_total"],1e-30),
                    "rms_amplitude_difference":dm["rms"]-nm["rms"],"orientation_difference_deg":abs(((dm["orientation_deg"]-nm["orientation_deg"]+90)%180)-90)})
            metrics.append(row)
    type_montages(prepared,args.output); hann_figure(prepared,args.output)
    write_csv(args.output/"inverse_fft_band_metrics.csv",metrics); write_csv(args.output/"selected_examples.csv",selections)
    make_summary_and_report(args.output,metrics,selections)
    settings={"window_px":N,"stride_px":prior.STRIDE,"reference_distance_px":prior.SPACING,"bands":BANDS,
              "preprocess":"subtract arithmetic mean, then multiply separable 2-D Hann","transform":"full complex fft2; fftshift radial mask; zero other complex coefficients; ifft2; real part",
              "display":"coolwarm centred at zero; shared-scale and per-panel auto-scale versions","energy":"sum of squared inverse-FFT image; total is sum of squared preprocessed image",
              "normal_selection":"among individually clean surrounding 8 blocks, choose band-energy fraction nearest their median",
              "selection":"highest prior 0.02-0.05 radial-spectrum RMS among examples with at least one clean reference; fill with no-clean examples only if needed"}
    (args.output/"settings.json").write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding="utf-8")


def make_summary_and_report(out: Path,metrics: list[dict],selections: list[dict]) -> None:
    target=[r for r in metrics if r["band_low"]==.02 and r["band_high"]==.05 and r.get("normal_rms") not in (None,"")]
    summary=[]
    for label in prior.DEFECT_LABELS:
        q=[r for r in target if r["defect_type"]==label]
        if not q: continue
        summary.append({"group":"defect_type","name":label,"examples":len(q),"median_defect_rms":float(np.median([r["defect_rms"] for r in q])),
            "median_normal_rms":float(np.median([r["normal_rms"] for r in q])),"median_rms_ratio":float(np.median([r["defect_rms"]/max(r["normal_rms"],1e-30) for r in q])),
            "median_energy_fraction_ratio":float(np.median([r["normalized_band_energy_fraction_ratio"] for r in q])),
            "median_orientation_difference_deg":float(np.median([r["orientation_difference_deg"] for r in q])),
            "median_hann_effect_defect":float(np.median([r["defect_hann_to_mean_only_rms_ratio"] for r in q]))})
    for group in ("yes","no"):
        q=[r for r in target if r["grid_light"]==group]
        if q: summary.append({"group":"grid_light","name":group,"examples":len(q),"median_defect_rms":float(np.median([r["defect_rms"] for r in q])),
            "median_normal_rms":float(np.median([r["normal_rms"] for r in q])),"median_rms_ratio":float(np.median([r["defect_rms"]/max(r["normal_rms"],1e-30) for r in q])),
            "median_energy_fraction_ratio":float(np.median([r["normalized_band_energy_fraction_ratio"] for r in q])),
            "median_orientation_difference_deg":float(np.median([r["orientation_difference_deg"] for r in q])),
            "median_hann_effect_defect":float(np.median([r["defect_hann_to_mean_only_rms_ratio"] for r in q]))})
    write_csv(out/"inverse_fft_summary.csv",summary)
    clean=sum(s["normal_status"]=="clean_reference_selected" for s in selections); no_clean=len(selections)-clean
    overall={k:float(np.median([r[k] for r in target])) for k in ("defect_rms","normal_rms","normalized_band_energy_fraction_ratio","orientation_difference_deg","defect_hann_to_mean_only_rms_ratio")}
    overall["paired_rms_ratio"]=float(np.median([r["defect_rms"]/max(r["normal_rms"],1e-30) for r in target]))
    lines=["# 128×128 帯域別逆FFT可視化レポート", "", "## 結論", "",
      "0.02–0.05 cycles/pixelは、おおよそ20–50 pxの波長に対応する。逆FFT像では、1画素の傷輪郭そのものではなく、数十px幅の明暗ローブ、帯、緩やかな勾配と、その周囲の正負の振動として現れた。これは位相を保持した結果であり、表示上の色は元画像の絶対輝度ではなく、平均除去・Hann後の符号付き輝度偏差である。",
      f"選択{len(selections)}例中、個別に汚染のない周囲参照を確保できたのは{clean}例、確保できなかったのは{no_clean}例だった。clean比較の0.02–0.05帯では、対応ペアごとの傷/正常RMS比の中央値は{overall['paired_rms_ratio']:.3f}、総エネルギーに占める帯域割合の比は{overall['normalized_band_energy_fraction_ratio']:.3f}だった。",
      "したがって前回の放射スペクトル差は、単に傷側の低周波振幅が常に大きいという意味ではない。明暗ローブの位置、広がり、方向、位相、および全帯域に対する相対配分が周囲と異なり、それらをdBスペクトルの形状差として検出している。", "",
      "## 1. 方法", "",
      "128×128 px窓から平均輝度を引き、2次元Hann窓を掛け、複素2D FFTを計算した。周波数半径 f=sqrt(fx²+fy²) で各リングだけを残し、それ以外の複素係数を0として逆FFTした。パワーから逆変換しておらず、位相は保持している。0.00–0.02は従来スコア範囲外だが可視化用として追加した。", "",
      "同一帯域の傷・正常で共通カラースケールを使う `shared_scale` と、各パネルの構造を見やすくする `auto_scale` を別保存した。共通表示は振幅比較、自動表示は形状比較に使う。", "",
      "## 2. 0.02–0.05帯が表す明暗パターン", "",
      "この帯域は20–50 px程度の空間スケールなので、局所的な明点・暗点をそのまま再現するのではなく、それを数十pxへぼかしたような正負のパターンを表す。帯域を急峻なリングで切り出すため、傷の周囲に波状のリンギングが生じる。これは物理的な凹凸の輪郭と断定できない。Hann窓により窓端は弱まり、中心寄りの構造が強調される。", "",
      "点状異常でも低周波が変わる例がある。小さな局所変化は広い周波数へ成分を持つため、その低周波部分だけを残すと広いローブになる。したがって逆FFT像のローブ径を傷寸法とみなしてはいけない。線状異常では帯状・方向性のあるローブ、面状異常と局所変形ではより広い勾配や非対称ローブが現れやすいが、例数と参照汚染の制約がある。", "",
      "## 3. 種類別定量結果（0.02–0.05）", "", "|種類|clean例|傷RMS|正常RMS|RMS比|帯域割合比|方向差|Hann/平均除去のみ|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        if r["group"]=="defect_type": lines.append(f"|{r['name']}|{r['examples']}|{r['median_defect_rms']:.3f}|{r['median_normal_rms']:.3f}|{r['median_rms_ratio']:.3f}|{r['median_energy_fraction_ratio']:.3f}|{r['median_orientation_difference_deg']:.1f}°|{r['median_hann_effect_defect']:.3f}|")
    lines += ["", "RMS振幅は空間像の強さ、帯域割合比は総エネルギーに対する相対配分である。前回の放射平均dBスペクトルのRMS差とは別指標であり、数値は直接比較しない。", "",
      "## 4. 格子ライトあり・なし", "", "|格子ライト|clean例|傷/正常RMS比|帯域割合比|方向差|", "|---|---:|---:|---:|---:|"]
    for r in summary:
        if r["group"]=="grid_light": lines.append(f"|{r['name']}|{r['examples']}|{r['median_rms_ratio']:.3f}|{r['median_energy_fraction_ratio']:.3f}|{r['median_orientation_difference_deg']:.1f}°|")
    lines += ["", "格子あり・なしの双方で20–50 pxスケールのローブ差が見える。128 px窓では既報約178 pxの格子基本周期を1周期含まず、その基本周波数約0.0056 cycles/pixelは0.02–0.05リング外である。この可視化だけから格子高調波の影響を完全には除けないが、格子周期の位置ずれ専用応答とは考えにくい。", "",
      "## 5. Hann窓の影響", "",
      f"0.02–0.05帯の傷RMSについて、Hann適用/平均除去のみの中央値は{overall['defect_hann_to_mean_only_rms_ratio']:.3f}だった。`hann_window_influence.png` では端部由来の強い構造がHannで抑えられ、中心側へ重みが移る様子を同一スケールで比較した。Hann後の逆FFT像は元パッチそのものではなく、窓関数で重み付けされた信号の帯域像である。", "",
      "## 6. 現在の検出器が見ているもの", "",
      "現在の局所FFTは、傷の形を直接テンプレート照合しているのではない。傷や反射変化によって生じる数十pxスケールの明暗ローブと、より細かい成分とのエネルギー配分が、周囲8ブロックの典型形状からどれだけ変わったかを検出している。0.02–0.05の差は特に大きいが、高周波側にも差があるため、低周波単独の検出器ではなく広帯域スペクトル形状比較器と説明するのが妥当である。", "",
      "## 注意", "", "- 逆FFT像の明暗を物理的な凹凸・反射原因へ直接対応付けない。", "- 急峻なリングマスクのリンギングとHann端部減衰を含む。", "- 正常参照がない例は傷像と周波数マスクだけを表示し、正常比較の統計から除外した。", "- 代表例は0.02–0.05差が大きい順の探索的選択で、母集団の性能推定ではない。", "",
      "## 出力", "", "- `examples/*_shared_scale.png`: 振幅共通比較", "- `examples/*_auto_scale.png`: 構造自動表示", "- `examples/*_fft_ring.png`: 2D FFTと0.02–0.05リング", "- `montage_*.png`: 種類別代表一覧", "- `hann_window_influence.png`: 平均除去のみ/Hann比較", "- `inverse_fft_band_metrics.csv`: 帯域別定量値", "- `selected_examples.csv`: 選択・参照状態", "- `inverse_fft_summary.csv`: 種類・格子有無集計"]
    (out/"report_ja.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--prior",type=Path,default=PRIOR); p.add_argument("--output",type=Path,default=OUT); p.add_argument("--examples-per-type",type=int,default=3)
    main(p.parse_args())
