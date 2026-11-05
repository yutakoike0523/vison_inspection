"""Evaluate global FFT notch filtering using previously confirmed grid peak pairs.

Creates a new analysis directory only.  Peak positions are read from the prior
full-image FFT analysis; no source images or prior outputs are modified.
"""
from __future__ import annotations

import csv, json, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter
from sklearn.metrics import roc_auc_score

import analysis_fft_normalization_128 as detector

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data20260903" / "data20260903"
PRIOR = ROOT / "analysis_full_image_fft_peaks"
OUT = ROOT / "analysis_fft_grid_notch_removal"
IMAGES = list(range(2, 19))
EVAL_IMAGES = list(range(2, 18))  # image18 is too small for strict-8 128px evaluation.
SELECTED = "top2_hard_r2"  # chosen from image15 spatial grid suppression, not labels.
EPS = 1e-12


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f: return list(csv.DictReader(f))


def as_int(row: dict, key: str) -> int: return int(float(row[key]))


def load_gray(n: int) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(DATA / f"image{n}.png").convert("L"), dtype=np.float64)


def peak_sets(n: int, toprows: list[dict], familyrows: list[dict]) -> dict[str, list[dict]]:
    top = [r for r in toprows if as_int(r,"image") == n and r["window"] == "none"]
    top.sort(key=lambda r: as_int(r,"rank_all"))
    fam = [r for r in familyrows if as_int(r,"image") == n and r["window"] == "none" and r["kind"] in ("base","harmonic")]
    # The compact family CSV stores frequency coordinates but not pixel indices.
    # Join only actually observed family peaks back to the top-20 coordinate list;
    # never synthesize an unobserved harmonic location.
    unique = {}
    for fr in fam:
        matches=[r for r in top if abs(float(r["fx"])-float(fr["fx"]))<1e-12 and abs(float(r["fy"])-float(fr["fy"]))<1e-12]
        if matches:
            r=matches[0]; unique[(as_int(r,"y"),as_int(r,"x"))] = r
    return {"top1":top[:1], "top2":top[:2], "top3":top[:3], "top5":top[:5], "family":list(unique.values())}


def locations(rows: list[dict], shape: tuple[int,int]) -> list[tuple[int,int]]:
    pts=[]
    for r in rows:
        y,x=as_int(r,"y"),as_int(r,"x")
        sy,sx=(as_int(r,"sym_y"),as_int(r,"sym_x")) if r.get("sym_y") else ((-y)%shape[0],(-x)%shape[1])
        pts += [(y,x),(sy,sx)]
    return pts


def notch(shape: tuple[int,int], rows: list[dict], mode: str, width: float) -> np.ndarray:
    """Real, centrosymmetric attenuation map in shifted FFT coordinates."""
    h,w=shape; yy,xx=np.ogrid[:h,:w]; response=np.ones(shape,np.float64)
    for y,x in locations(rows,shape):
        d2=(yy-y)**2+(xx-x)**2
        if mode == "hard": response[d2 <= width*width] = 0.0
        else:  # Gaussian attenuation; width is sigma in FFT bins.
            response *= 1.0 - np.exp(-d2/(2.0*width*width))
    return response


def apply_notch(gray: np.ndarray, rows: list[dict], mode: str, width: float) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    mean=float(gray.mean()); coeff=np.fft.fft2(gray-mean); shifted=np.fft.fftshift(coeff)
    m=notch(gray.shape,rows,mode,width); out=np.fft.ifft2(np.fft.ifftshift(shifted*m)).real+mean
    return out, shifted, shifted*m, m


def retained_grid(shifted: np.ndarray, rows: list[dict], radius: int=2) -> np.ndarray:
    m=1.0-notch(shifted.shape,rows,"hard",radius)
    return np.fft.ifft2(np.fft.ifftshift(shifted*m)).real


def mask_from_annotations(n: int, shape: tuple[int,int]) -> tuple[np.ndarray,np.ndarray,np.ndarray,list[dict]]:
    anns=detector.load_annotations(DATA/f"image{n}.json",shape)
    defects=[a for a in anns if a["label"] in detector.DEFECT_LABELS]
    excluded=[a for a in anns if a["label"] in detector.EXCLUSION_LABELS]
    defect=np.logical_or.reduce([a["mask"] for a in defects]) if defects else np.zeros(shape,bool)
    exc=np.logical_or.reduce([a["mask"] for a in excluded]) if excluded else np.zeros(shape,bool)
    grid=np.zeros(shape,bool)
    Y,X=np.ogrid[:shape[0],:shape[1]]
    # image15 includes manually recorded grid points from the former experiment.
    for a in anns:
        if str(a["label"]).startswith("grid"):
            # shape mask for a point is tiny; recover its centroid and make a local grid patch.
            py,px=np.nonzero(a["mask"])
            if len(px):
                grid |= (Y-int(round(py.mean())))**2+(X-int(round(px.mean())))**2 <= 38**2
    return defect, exc, grid & ~defect & ~exc, defects


def local_metrics(before: np.ndarray, after: np.ndarray, defect: np.ndarray, exc: np.ndarray, grid: np.ndarray) -> dict:
    valid=~(defect|exc)
    delta=after-before
    # Use a 31 px dilation ring as local normal comparison around all annotated defects.
    ring=binary_dilation(defect,iterations=31)&~binary_dilation(defect,iterations=3)&~exc
    def stat(im,m,fun): return float(fun(im[m])) if m.any() else math.nan
    return {
        "defect_pixels":int(defect.sum()), "normal_pixels":int(valid.sum()), "grid_normal_pixels":int(grid.sum()),
        "defect_mean_before":stat(before,defect,np.mean), "defect_mean_after":stat(after,defect,np.mean),
        "ring_mean_before":stat(before,ring,np.mean), "ring_mean_after":stat(after,ring,np.mean),
        "defect_rms_before":stat(before,defect,np.std), "defect_rms_after":stat(after,defect,np.std),
        "ring_rms_before":stat(before,ring,np.std), "ring_rms_after":stat(after,ring,np.std),
        "defect_abs_change":stat(np.abs(delta),defect,np.mean), "normal_abs_change":stat(np.abs(delta),valid,np.mean),
        "grid_abs_change":stat(np.abs(delta),grid,np.mean),
        "defect_contrast_before":abs(stat(before,defect,np.mean)-stat(before,ring,np.mean)),
        "defect_contrast_after":abs(stat(after,defect,np.mean)-stat(after,ring,np.mean)),
    }


def grid_metrics(before_shift: np.ndarray, after_shift: np.ndarray, rows: list[dict], before: np.ndarray, after: np.ndarray, grid: np.ndarray) -> dict:
    # Peak-region energy and a spatial modulation proxy (std after a 20px trend removal).
    mask=1.0-notch(before_shift.shape,rows,"hard",2)
    e0=float(np.sum(np.abs(before_shift[mask>0])**2)); e1=float(np.sum(np.abs(after_shift[mask>0])**2))
    def modulation(a):
        hp=a-gaussian_filter(a,20)
        return float(np.std(hp[grid])) if grid.any() else math.nan
    m0,m1=modulation(before),modulation(after)
    return {"peak_region_power_before":e0,"peak_region_power_after":e1,
            "peak_power_reduction_pct":100*(1-e1/max(e0,EPS)),
            "grid_modulation_before":m0,"grid_modulation_after":m1,
            "grid_modulation_reduction_pct":100*(1-m1/m0) if np.isfinite(m0) and m0 else math.nan}


def down(a: np.ndarray, step:int=3)->np.ndarray: return a[::step,::step]


def plot_image15_overview(gray, shifted, retained, filtered, after_shift, rows, name, out: Path):
    lim=np.percentile(gray,[1,99]); d=filtered-gray; dlim=np.percentile(abs(d),99.5)
    p0=10*np.log10(abs(shifted)**2+EPS); p1=10*np.log10(abs(after_shift)**2+EPS); plim=np.percentile(p0,[5,99.8])
    fig,ax=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    specs=[(gray,"Original"),(retained,"Retained grid peaks only"),(filtered,f"Notch output: {name}"),
           (d,"Output − original"),(p0,"FFT power before (dB)"),(p1,"FFT power after (dB)")]
    for i,(im,title) in enumerate(specs):
        a=ax.ravel()[i]
        if i==3: a.imshow(down(im),cmap="coolwarm",vmin=-dlim,vmax=dlim)
        elif i>=4: a.imshow(down(im),cmap="magma",vmin=plim[0],vmax=plim[1])
        else: a.imshow(down(im),cmap="gray",vmin=lim[0],vmax=lim[1])
        a.set_title(title);a.set_xticks([]);a.set_yticks([])
    fig.savefig(out,dpi=120);plt.close(fig)


def plot_image15_candidates(gray, rows_by_set, out: Path):
    lim=np.percentile(gray,[1,99]); fig,ax=plt.subplots(2,3,figsize=(14,9),constrained_layout=True)
    ax[0,0].imshow(down(gray),cmap="gray",vmin=lim[0],vmax=lim[1]);ax[0,0].set_title("Original")
    for a,(name,rows) in zip(ax.ravel()[1:],rows_by_set.items()):
        after,_,_,_=apply_notch(gray,rows,"hard",2)
        a.imshow(down(after),cmap="gray",vmin=lim[0],vmax=lim[1]);a.set_title(f"{name}: hard radius 2")
    for a in ax.ravel():a.set_xticks([]);a.set_yticks([])
    fig.savefig(out,dpi=120);plt.close(fig)


def plot_image15_widths(gray, rows, out:Path):
    configs=[("none",None,0),("hard r0","hard",0),("hard r1","hard",1),("hard r2","hard",2),("hard r3","hard",3),("Gaussian σ0.7","gauss",.7),("Gaussian σ1.2","gauss",1.2),("Gaussian σ2.0","gauss",2.0)]
    lim=np.percentile(gray,[1,99]);fig,ax=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
    for a,(name,mode,w) in zip(ax.ravel(),configs):
        im=gray if mode is None else apply_notch(gray,rows,mode,w)[0]
        a.imshow(down(im),cmap="gray",vmin=lim[0],vmax=lim[1]);a.set_title(name);a.set_xticks([]);a.set_yticks([])
    fig.savefig(out,dpi=120);plt.close(fig)


def plot_defect_rois(n, gray, filtered, defects, out:Path):
    boxes=[]
    for a in defects:
        yy,xx=np.nonzero(a["mask"])
        if len(xx): boxes.append((max(0,xx.min()-45),max(0,yy.min()-45),min(gray.shape[1],xx.max()+46),min(gray.shape[0],yy.max()+46),a["label"]))
    fig,ax=plt.subplots(max(1,len(boxes)),3,figsize=(12,4*max(1,len(boxes))),squeeze=False,constrained_layout=True)
    lim=np.percentile(gray,[1,99]); dlim=np.percentile(abs(filtered-gray),99.5)
    for row,b in zip(ax,boxes):
        x0,y0,x1,y1,label=b; sl=np.s_[y0:y1,x0:x1]
        for a,im,title,cmap in zip(row,(gray,filtered,filtered-gray),("Before","Top2 hard r2","Difference"),("gray","gray","coolwarm")):
            if cmap=="gray":a.imshow(im[sl],cmap=cmap,vmin=lim[0],vmax=lim[1])
            else:a.imshow(im[sl],cmap=cmap,vmin=-dlim,vmax=dlim)
            a.set_title(f"{label}: {title}");a.set_xticks([]);a.set_yticks([])
    fig.savefig(out,dpi=150);plt.close(fig)


def detector_scores(img: np.ndarray, defects: list[dict], exc: np.ndarray) -> tuple[np.ndarray,np.ndarray,list[dict]]:
    setup=detector.radial_setup(); r=detector.spectra_for_image(img,setup,workers=4,batch=64); step=detector.SPACING//detector.STRIDE
    if r["norm"].shape[0]<=2*step or r["norm"].shape[1]<=2*step:return np.array([]),np.array([]),[]
    score=detector.strict_scores(r["norm"]); sx=r["xs"][step:-step];sy=r["ys"][step:-step]
    union=np.logical_or.reduce([a["mask"] for a in defects]) if defects else np.zeros(img.shape,bool)
    target=detector.rect_any(detector.integral(union),sx,sy,16); excluded=detector.rect_any(detector.integral(exc),sx,sy,16)
    normal=score[(~target)&(~excluded)]; dscore=score[target&~excluded]
    rows=[]
    for k,a in enumerate(defects,1):
        m=detector.rect_any(detector.integral(a["mask"]),sx,sy,16)&~excluded
        rows.append({"defect_number":k,"defect_type":a["label"],"candidate_windows":int(m.sum()),
                     "instance_score":float(score[m].max()) if m.any() else math.nan})
    return normal,dscore,rows


def write_csv(path:Path, rows:list[dict]):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction="ignore");w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True); (OUT/"per_image").mkdir(exist_ok=True)
    top, family=read_csv(PRIOR/"fft_peak_pairs_top20.csv"),read_csv(PRIOR/"grid_family_candidates.csv")
    # image15 candidate exploration, selected without looking at any defect metric.
    gray15=load_gray(15); sets15=peak_sets(15,top,family); defect15,exc15,grid15,defs15=mask_from_annotations(15,gray15.shape)
    selection=[]
    for setname, rows in sets15.items():
        for mode,width in [("hard",x) for x in (0,1,2,3)]+[("gauss",x) for x in (.7,1.2,2.0)]:
            out,sh,ash,_=apply_notch(gray15,rows,mode,width); m={"image":15,"peak_set":setname,"notch":mode,"width_bins":width,"peak_pairs":len(rows)}
            m.update(grid_metrics(sh,ash,rows,gray15,out,grid15));m.update(local_metrics(gray15,out,defect15,exc15,grid15));selection.append(m)
    write_csv(OUT/"image15_notch_parameter_comparison.csv",selection)
    selected15=sets15["top2"]; f15,sh15,ash15,_=apply_notch(gray15,selected15,"hard",2); ret15=retained_grid(sh15,selected15)
    plot_image15_overview(gray15,sh15,ret15,f15,ash15,selected15,SELECTED,OUT/"image15_selected_notch_overview.png")
    plot_image15_candidates(gray15,sets15,OUT/"image15_peak_set_comparison_r2.png")
    plot_image15_widths(gray15,selected15,OUT/"image15_notch_width_comparison.png")
    plot_defect_rois(15,gray15,f15,defs15,OUT/"image15_defect_roi_before_after.png")

    image_rows=[]; removal_rows=[]; perf_data={"baseline":{"normal":[],"defect":[],"instances":[]},SELECTED:{"normal":[],"defect":[],"instances":[]}}
    for n in IMAGES:
        print(f"image{n}",flush=True); gray=load_gray(n); sets=peak_sets(n,top,family); rows=sets["top2"]
        defect,exc,grid,defs=mask_from_annotations(n,gray.shape); after,sh,ash,_=apply_notch(gray,rows,"hard",2)
        rm={"image":n,"condition":SELECTED,"peak_set":"top2","peak_pairs":len(rows),"notch":"hard","width_bins":2}
        rm.update(grid_metrics(sh,ash,rows,gray,after,grid));rm.update(local_metrics(gray,after,defect,exc,grid));removal_rows.append(rm)
        # representative visual using a shared input scale plus a signed difference.
        ret=retained_grid(sh,rows); plot_image15_overview(gray,sh,ret,after,ash,rows,SELECTED,OUT/"per_image"/f"image{n:02d}_representative.png")
        if n in EVAL_IMAGES and defs:
            for name,im in (("baseline",gray),(SELECTED,after)):
                normal,dscore,instances=detector_scores(im,defs,exc)
                perf_data[name]["normal"].append(normal);perf_data[name]["defect"].append(dscore)
                for r in instances: perf_data[name]["instances"].append({"image":n,"condition":name,**r})
        image_rows.append({"image":n,"height":gray.shape[0],"width":gray.shape[1],"top2_pairs":len(rows),"selected_condition":SELECTED})
    write_csv(OUT/"notch_removal_metrics_by_image.csv",removal_rows);write_csv(OUT/"per_image_conditions.csv",image_rows)
    # Fixed detector setup, pooled window evaluation and per-type instance detection.
    perf=[]; inst=[]
    for name,d in perf_data.items():
        normal=np.concatenate(d["normal"]); defect=np.concatenate(d["defect"])
        auc=float(roc_auc_score(np.r_[np.zeros(len(normal)),np.ones(len(defect))],np.r_[normal,defect]))
        result={"condition":name,"evaluation_unit":"window","normal_windows":len(normal),"defect_windows":len(defect),"roc_auc":auc}
        for fpr in (.01,.05):
            t=float(np.quantile(normal,1-fpr,method="higher")); result[f"threshold_fpr_{int(fpr*100)}pct"]=t
            result[f"actual_fpr_{int(fpr*100)}pct"]=float(np.mean(normal>t));result[f"window_tpr_fpr_{int(fpr*100)}pct"]=float(np.mean(defect>t))
            for r in d["instances"]: r[f"hit_fpr_{int(fpr*100)}pct"]=bool(r["instance_score"]>t)
        perf.append(result);inst.extend(d["instances"])
    write_csv(OUT/"local_fft_performance_summary.csv",perf);write_csv(OUT/"local_fft_instance_scores.csv",inst)
    type_rows=[]
    for name in ("baseline",SELECTED):
        q=[r for r in inst if r["condition"]==name and np.isfinite(r["instance_score"])]
        for typ in sorted(set(r["defect_type"] for r in q)):
            x=[r for r in q if r["defect_type"]==typ]
            type_rows.append({"condition":name,"defect_type":typ,"instances":len(x),
                "detection_rate_fpr_1pct":np.mean([r["hit_fpr_1pct"] for r in x]),"detection_rate_fpr_5pct":np.mean([r["hit_fpr_5pct"] for r in x]),
                "median_instance_score":float(np.median([r["instance_score"] for r in x]))})
    write_csv(OUT/"local_fft_performance_by_defect_type.csv",type_rows)
    (OUT/"settings.json").write_text(json.dumps({"prior_peak_files":[str(PRIOR/"fft_peak_pairs_top20.csv"),str(PRIOR/"grid_family_candidates.csv")],"selected_condition":SELECTED,"selection_rule":"image15 spatial grid suppression/visual comparison only; no defect label used","notch":"hard circular radius 2 FFT bins around prior top2 conjugate pairs","detector":"existing analysis_fft_normalization_128.py: normalized radial 128x128 FFT, stride 16, strict-8 reference, unchanged settings","evaluation":"pooled windows; instance=max score among annotation-overlapping 16px evaluation cells; thresholds per condition pooled normal windows"},ensure_ascii=False,indent=2),encoding="utf-8")
    make_report(perf,type_rows,removal_rows,selection)


def make_report(perf,types,rm,selection):
    p={x["condition"]:x for x in perf}; b=p["baseline"]; s=p[SELECTED]
    m15=[x for x in rm if x["image"]==15][0]
    lines=["# FFT格子ピーク除去の検証", "", "## 結論", "",
      f"- image15では、前回確認した直交する上位2共役ピーク対を半径2 FFT-binでノッチすると、選択ピーク領域のパワーは **{m15['peak_power_reduction_pct']:.1f}%**低下した。上位1組では縞が残り、上位2組で二方向格子を抑えられる。",
      "- 半径を広げると格子成分はより減るが、周辺の実画像成分も除去する。Gaussianノッチは急峻なゼロ化より穏やかだが、格子の残りも多い。image15の比較は傷ラベルを使わずに行い、全画像には固定の `top2 + hard radius 2` を適用した。",
      f"- 同一の既存局所FFT検出器（128px、stride 16px、strict-8、総パワー正規化）では、プールしたwindow ROC-AUCは **{b['roc_auc']:.4f} → {s['roc_auc']:.4f}**、FPR 1%のwindow TPRは **{b['window_tpr_fpr_1pct']:.4f} → {s['window_tpr_fpr_1pct']:.4f}**、FPR 5%では **{b['window_tpr_fpr_5pct']:.4f} → {s['window_tpr_fpr_5pct']:.4f}**。したがって、改善の有無はこの数値で判断し、格子が視覚的に弱まっただけで性能向上とは結論しない。",
      "- ノッチの周波数は格子だけに専有されない。傷の局所コントラストとRMSもCSVで同時に測定しており、傷が残る／弱まる効果を分離して確認する必要がある。", "",
      "## 出力の読み方", "", "- `image15_selected_notch_overview.png`: 元画像、格子ピークのみ、除去結果、差分、FFT前後。", "- `image15_peak_set_comparison_r2.png`: 除去するピーク組数の比較。", "- `image15_notch_width_comparison.png`: hard r0〜r3 と Gaussian σ0.7/1.2/2.0。", "- `image15_defect_roi_before_after.png`: 傷アノテーションの拡大。", "- `per_image/`: image2〜18の固定条件の代表図。image18は小画像のため、検出器性能集計には含めない。", "",
      "## 限界", "", "全画像性能値は、image15でラベルを使わずに決めた固定条件を対象にした比較であるが、同じ画像群での探索的な評価であり完全な独立テストではない。次段階では画像単位のleave-one-image-outでノッチ幅やピーク組を固定して検証する必要がある。", "",
      "## CSV", "", "- `image15_notch_parameter_comparison.csv`: 全ピーク組・ノッチ幅の空間指標（傷指標も参考として併記）。", "- `notch_removal_metrics_by_image.csv`: 画像ごとの格子抑制・傷／正常領域変化。", "- `local_fft_performance_summary.csv`: window単位のROC-AUC、FPR固定TPR、誤検出率。", "- `local_fft_instance_scores.csv`: 傷インスタンス単位。", "- `local_fft_performance_by_defect_type.csv`: 傷種類別。"]
    (OUT/"report_ja.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__": main()
