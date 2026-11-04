#!/usr/bin/env python3
"""Compare each selected defect with all eight 0.02-0.05 inverse-FFT references."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from scipy.stats import spearmanr

import analysis_fft_normalization_128 as prior
import visualize_inverse_fft_bands_128 as inv


ROOT=Path(__file__).resolve().parent
SELECTIONS=ROOT/"analysis_inverse_fft_bands_128"/"selected_examples.csv"
OUT=ROOT/"analysis_lowfreq_defect_vs_8refs"
BAND=(0.02,0.05); N=128; HALF=64
DIRECTIONS=("upper-left","up","upper-right","left","right","lower-left","down","lower-right")


def write_csv(path: Path,rows: list[dict]) -> None:
    fields=list(dict.fromkeys(k for r in rows for k in r)) if rows else ["status"]
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def read_selections(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
    for r in rows:
        for k in ("selection_order","image_number","defect_number","defect_x","defect_y","clean_reference_count"):
            r[k]=int(float(r[k])) if r.get(k) else 0
        r["prior_band_002_005_rms_db"]=float(r["prior_band_002_005_rms_db"])
    return rows


def choose_target_from_annotation(img: np.ndarray,mask: np.ndarray) -> tuple[int,int]:
    xs=np.arange(HALF,img.shape[1]-HALF+1,prior.STRIDE,dtype=int); ys=np.arange(HALF,img.shape[0]-HALF+1,prior.STRIDE,dtype=int)
    candidates=prior.rect_any(prior.integral(mask),xs,ys,16); yy,xx=np.nonzero(candidates)
    if len(yy):
        values=[]
        for iy,ix in zip(yy,xx):
            t=inv.transform(inv.patch_at(img,int(xs[ix]),int(ys[iy])),True); values.append(inv.band_metrics(t,*BAND)["rms"])
        k=int(np.argmax(values)); return int(xs[xx[k]]),int(ys[yy[k]])
    py,px=np.nonzero(mask); cx=int(np.clip(np.mean(px) if len(px) else img.shape[1]/2,HALF,img.shape[1]-HALF)); cy=int(np.clip(np.mean(py) if len(py) else img.shape[0]/2,HALF,img.shape[0]-HALF))
    return int(xs[np.argmin(abs(xs-cx))]),int(ys[np.argmin(abs(ys-cy))])


def all_defect_selections(prior_csv: Path) -> list[dict]:
    with prior_csv.open(encoding="utf-8-sig") as f: old=list(csv.DictReader(f))
    by_key={(int(r["image_number"]),int(r["defect_number"])):r for r in old}; result=[]; order=0
    for number in range(1,22):
        img=np.asarray(Image.open(prior.DATA/f"image{number}.png").convert("L")); anns=prior.load_annotations(prior.DATA/f"image{number}.json",img.shape)
        defects=[a for a in anns if a["label"] in prior.DEFECT_LABELS]
        for defect_number,a in enumerate(defects,1):
            order+=1; oldrow=by_key[number,defect_number]
            if oldrow.get("status")=="evaluated": x,y=int(float(oldrow["target_x"])),int(float(oldrow["target_y"])); source="prior_representative"
            else: x,y=choose_target_from_annotation(img,a["mask"]); source="derived_max_band_rms"
            score=float(oldrow["band_0.02_0.05_rms_norm"]) if oldrow.get("band_0.02_0.05_rms_norm") else math.nan
            result.append({"selection_order":order,"image_number":number,"defect_number":defect_number,"defect_type":a["label"],"defect_x":x,"defect_y":y,
                           "clean_reference_count":0,"prior_band_002_005_rms_db":score,"selection_source":source,"prior_status":oldrow.get("status","")})
    return result


def overlap_labels(anns: list[dict],x: int,y: int) -> list[str]:
    result=[]
    groups=((prior.DEFECT_LABELS,"defect"),("tag","tag"),("\u30bf\u30b0","tag"),("background","background"),("\u80cc\u666f","background"))
    x0,x1=x-HALF,x+HALF; y0,y1=y-HALF,y+HALF
    for labels,name in groups:
        labels=(labels,) if isinstance(labels,str) else labels
        masks=[a["mask"] for a in anns if a["label"] in labels]
        if masks and np.logical_or.reduce(masks)[max(0,y0):y1,max(0,x0):x1].any() and name not in result: result.append(name)
    return result


def window_record(img: np.ndarray,anns: list[dict],x: int,y: int,role: str,direction: str) -> dict:
    if x-HALF<0 or y-HALF<0 or x+HALF>img.shape[1] or y+HALF>img.shape[0]:
        return {"role":role,"direction":direction,"x":x,"y":y,"patch":None,"transform":None,"recon":None,"available":False,
                "clean_flag":False,"overlap_type":"out_of_image",**{k:math.nan for k in ("rms","minimum","maximum","peak_to_peak","energy","total_energy","energy_fraction_total","energy_fraction_analyzed_0_05","centroid_x","centroid_y","orientation_deg","anisotropy","spread_px")}}
    patch=inv.patch_at(img,x,y); trans=inv.transform(patch,True); met=inv.band_metrics(trans,*BAND)
    overlaps=overlap_labels(anns,x,y); recon=trans["recon"][BAND]
    return {"role":role,"direction":direction,"x":x,"y":y,"patch":patch,"transform":trans,"recon":recon,
            "available":True,"clean_flag":len(overlaps)==0,"overlap_type":";".join(overlaps) if overlaps else "none",**met}


def prepare(sel: dict) -> tuple[np.ndarray,list[dict],list[dict]]:
    img=np.asarray(Image.open(prior.DATA/f"image{sel['image_number']}.png").convert("L")); anns=prior.load_annotations(prior.DATA/f"image{sel['image_number']}.json",img.shape)
    x,y=sel["defect_x"],sel["defect_y"]
    windows=[window_record(img,anns,x,y,"defect","center")]
    for index,((dy,dx),direction) in enumerate(zip(prior.OFFSETS,DIRECTIONS),1):
        windows.append(window_record(img,anns,x+dx*prior.SPACING,y+dy*prior.SPACING,f"ref{index}",direction))
    return img,anns,windows


def annotate_location(ax,img: np.ndarray,windows: list[dict]) -> None:
    x=windows[0]["x"]; y=windows[0]["y"]; margin=HALF+prior.SPACING
    x0=max(0,x-margin); x1=min(img.shape[1],x+margin); y0=max(0,y-margin); y1=min(img.shape[0],y+margin)
    ax.imshow(img,cmap="gray",extent=(0,img.shape[1],img.shape[0],0),vmin=np.percentile(img[y0:y1,x0:x1],1),vmax=np.percentile(img[y0:y1,x0:x1],99))
    for w in windows:
        color="red" if w["role"]=="defect" else ("lime" if w["clean_flag"] else "orange")
        ax.add_patch(Rectangle((w["x"]-HALF,w["y"]-HALF),N,N,fill=False,edgecolor=color,linewidth=2))
        suffix="clean" if w["clean_flag"] else w["overlap_type"]
        ax.text(w["x"],w["y"]-HALF+4,f"{w['role']}\n{suffix}",color=color,ha="center",va="top",fontsize=8,
                bbox=dict(facecolor="black",alpha=.55,pad=1,edgecolor="none"))
    ax.set(xlim=(x0,x1),ylim=(y1,y0),title="Window layout: red=defect, green=clean ref, orange=annotation overlap",xlabel="x (px)",ylabel="y (px)")


def figure(sel: dict,img: np.ndarray,windows: list[dict],path: Path,shared: bool) -> None:
    fig=plt.figure(figsize=(27,10),constrained_layout=True); gs=GridSpec(3,9,figure=fig,height_ratios=(1.7,1,1))
    info=fig.add_subplot(gs[0,0:3]); info.axis("off")
    info_lines=["128 x 128 px windows", "reference spacing: 128 px", "band: 0.02-0.05 cycles/pixel", "", "Annotation audit:"]
    info_lines += [f"{w['role']}: {'clean' if w['clean_flag'] else w['overlap_type']}" for w in windows]
    info.text(.04,.96,"\n".join(info_lines),va="top",ha="left",fontsize=11,family="monospace")
    top=fig.add_subplot(gs[0,3:6]); annotate_location(top,img,windows)
    bars=fig.add_subplot(gs[0,6:9]); vals=[w["rms"] for w in windows]
    colors=["red"]+["#39a844" if w["clean_flag"] else "#f39c12" for w in windows[1:]]
    bars.bar(range(9),vals,color=colors); bars.axhline(np.nanmedian(vals[1:]),color="black",ls="--",label="available-ref median")
    bars.set(xticks=range(9),xticklabels=["D"]+[f"R{i}" for i in range(1,9)],ylabel="band RMS amplitude",title="Defect vs 8-reference strength"); bars.legend(); bars.grid(axis="y",alpha=.2)
    available=[w for w in windows if w["available"]]; allpix=np.concatenate([w["patch"].ravel() for w in available]); gray=np.percentile(allpix,[1,99])
    shared_lim=max(np.percentile(np.abs(w["recon"]),99.5) for w in available)
    for col,w in enumerate(windows):
        ax=fig.add_subplot(gs[1,col])
        if w["available"]: ax.imshow(w["patch"],cmap="gray",vmin=gray[0],vmax=gray[1])
        else: ax.text(.5,.5,"OUT OF IMAGE",ha="center",va="center",transform=ax.transAxes,color="orange",weight="bold")
        suffix="clean" if w["clean_flag"] else w["overlap_type"]
        ax.set_title(f"{w['role']} ({w['direction']})\n{suffix}",fontsize=9,color="black" if w["clean_flag"] or w["role"]=="defect" else "darkorange"); ax.set_xticks([]); ax.set_yticks([])
        ax=fig.add_subplot(gs[2,col]); lim=shared_lim if shared or not w["available"] else max(np.percentile(np.abs(w["recon"]),99.5),1e-9)
        if w["available"]:
            ax.imshow(w["recon"],cmap="coolwarm",vmin=-lim,vmax=lim); ax.set_title(f"RMS={w['rms']:.2f}, frac={w['energy_fraction_total']*100:.2f}%\nscale ±{lim:.2f}",fontsize=8)
        else: ax.text(.5,.5,"UNAVAILABLE",ha="center",va="center",transform=ax.transAxes,color="orange",weight="bold"); ax.set_title("RMS=N/A",fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_edgecolor("red" if w["role"]=="defect" else ("limegreen" if w["clean_flag"] else "orange")); spine.set_linewidth(2)
    fig.suptitle(f"image{sel['image_number']} defect {sel['defect_number']} {inv.TYPE_EN[sel['defect_type']]} | 0.02-0.05 cycles/pixel | {'shared inverse-FFT scale' if shared else 'per-panel auto scale'}",fontsize=16)
    fig.savefig(path,dpi=160); plt.close(fig)


def summarize(sel: dict,windows: list[dict]) -> dict:
    defect=windows[0]; allrefs=windows[1:]; refs=[w for w in allrefs if w["available"]]; values=np.array([w["rms"] for w in refs]); clean=[w for w in refs if w["clean_flag"]]
    mean=float(values.mean()); std=float(values.std(ddof=0)); med=float(np.median(values))
    corr=[]
    a=defect["recon"].ravel(); an=np.linalg.norm(a)
    for w in refs:
        b=w["recon"].ravel(); corr.append(float(np.dot(a,b)/max(an*np.linalg.norm(b),1e-30)))
    row={"image_id":f"image{sel['image_number']}","defect_id":sel["defect_number"],"defect_type":sel["defect_type"],"grid_light":"no" if sel["image_number"] in prior.NO_GRID else "yes",
         "prior_radial_band_rms_db":sel["prior_band_002_005_rms_db"],"defect_band_rms":defect["rms"],"ref_median_rms":med,
         "defect_minus_ref_median":defect["rms"]-med,"defect_div_ref_median":defect["rms"]/max(med,1e-30),"ref_mean_rms":mean,"ref_std":std,
         "ref_coefficient_of_variation":std/max(mean,1e-30),"z_score_like":(defect["rms"]-mean)/std if std>0 else math.nan,
         "defect_overlap_type":defect["overlap_type"],"defect_has_tag_or_background":("tag" in defect["overlap_type"] or "background" in defect["overlap_type"]),
         "available_reference_count":len(refs),"unavailable_reference_count":8-len(refs),"clean_reference_count":len(clean),"contaminated_reference_count":len(refs)-len(clean),"median_spatial_pattern_correlation_to_refs":float(np.median(corr)),
         "median_abs_spatial_pattern_correlation_to_refs":float(np.median(np.abs(corr))),"defect_orientation_deg":defect["orientation_deg"],
         "ref_median_orientation_deg":float(np.median([w["orientation_deg"] for w in refs]))}
    if clean:
        cv=np.array([w["rms"] for w in clean]); cm=float(cv.mean()); cs=float(cv.std(ddof=0)); cmed=float(np.median(cv))
        row.update(clean_ref_median_rms=cmed,defect_div_clean_ref_median=defect["rms"]/max(cmed,1e-30),clean_ref_std=cs,
                   clean_z_score_like=(defect["rms"]-cm)/cs if cs>0 else math.nan)
    return row


def overview(summary: list[dict],window_rows: list[dict],out: Path) -> None:
    safe_names={"\u5c40\u6240\u5909\u5f62":"local_deformation","\u70b9\u72b6\u7570\u5e38":"point","\u7dda\u72b6\u7570\u5e38":"line","\u9762\u72b6\u7570\u5e38":"area"}
    for label in prior.DEFECT_LABELS:
        group=[s for s in summary if s["defect_type"]==label]; cols=3; rows=math.ceil(len(group)/cols)
        fig,axes=plt.subplots(rows,cols,figsize=(15,3.3*rows),constrained_layout=True); axes=np.atleast_1d(axes).ravel()
        for ax,s in zip(axes,group):
            q=[r for r in window_rows if r["image_id"]==s["image_id"] and str(r["defect_id"])==str(s["defect_id"])]
            vals=[r["band_rms"] for r in q]; colors=["red"]+["#39a844" if r["clean_flag"] else "#f39c12" for r in q[1:]]
            ax.bar(range(9),vals,color=colors); ax.axhline(s["ref_median_rms"],color="black",ls="--",lw=1,label="available-ref median")
            ax.set(xticks=range(9),xticklabels=["D"]+[f"R{i}" for i in range(1,9)],ylabel="band RMS",title=f"{s['image_id']} #{s['defect_id']} ratio={s['defect_div_ref_median']:.2f}, z={s['z_score_like']:.2f}")
            ax.legend(fontsize=7); ax.grid(axis="y",alpha=.2)
        for ax in axes[len(group):]: ax.axis("off")
        fig.suptitle(f"{inv.TYPE_EN[label]}: 0.02-0.05 inverse-FFT RMS\nred=defect, green=clean, orange=overlap/out-of-image")
        fig.savefig(out/f"rms_comparison_{safe_names[label]}.png",dpi=180); plt.close(fig)


def make_report(out: Path,summary: list[dict],window_rows: list[dict]) -> None:
    amp=np.array([abs(math.log(max(r["defect_div_ref_median"],1e-12))) for r in summary]); z=np.abs([r["z_score_like"] for r in summary]); shape=np.array([1-abs(r["median_spatial_pattern_correlation_to_refs"]) for r in summary]); prior_score=np.array([r["prior_radial_band_rms_db"] for r in summary])
    valid=np.isfinite(prior_score); ca=spearmanr(prior_score[valid],amp[valid]).statistic; cz=spearmanr(prior_score[valid],z[valid]).statistic; cs=spearmanr(prior_score[valid],shape[valid]).statistic
    stronger=sum(r["defect_div_ref_median"]>1 for r in summary); out2=sum(abs(r["z_score_like"])>=2 for r in summary); target_overlap=sum(r["defect_has_tag_or_background"] for r in summary)
    refs=[r for r in window_rows if r["window_role"]!="defect"]; available_refs=[r for r in refs if r["overlap_type"]!="out_of_image"]; clean_refs=sum(r["clean_flag"] for r in available_refs); total=len(summary)
    type_rows=[]
    for label in prior.DEFECT_LABELS:
        q=[r for r in summary if r["defect_type"]==label]
        type_rows.append({"defect_type":label,"examples":len(q),"median_rms_ratio":float(np.nanmedian([r["defect_div_ref_median"] for r in q])),"median_z":float(np.nanmedian([r["z_score_like"] for r in q])),
                          "median_ref_cv":float(np.median([r["ref_coefficient_of_variation"] for r in q])),"median_abs_pattern_correlation":float(np.median([r["median_abs_spatial_pattern_correlation_to_refs"] for r in q])),
                          "median_clean_count":float(np.median([r["clean_reference_count"] for r in q]))})
    write_csv(out/"defect_type_summary.csv",type_rows)
    lines=["# 傷窓と周囲8参照の0.02–0.05逆FFT比較", "", "## 結論", "",
      f"全{total}傷のうち、傷窓の帯域RMSが取得可能な周囲参照中央値より大きかったのは{stronger}/{total}、参照RMS分布から|z|≥2の外れ値だったのは{out2}/{total}だった。",
      f"128×128 FFT窓全体で監査すると、傷窓自体が札または背景アノテーションにも重なる例が{target_overlap}/{total}あった。この場合の強い低周波は傷だけに帰属できない。図と `defect_summary.csv` の `defect_overlap_type` を併せて確認する必要がある。",
      f"全{len(refs)}参照スロット中、画像内で取得可能なのは{len(available_refs)}窓、そのうちcleanは{clean_refs}窓、傷・札・背景との重なりありは{len(available_refs)-clean_refs}窓、画像外は{len(refs)-len(available_refs)}窓だった。重なり参照を含む統計は実際の検出条件を表すが、正常分布の解釈にはclean-only列も併用する。",
      "共通スケール図では振幅差、自動スケール図ではローブの位置・符号・方向の違いを分離して確認できる。傷窓が強くない例でも、周囲参照とは異なる非対称ローブや帯方向を持つ場合があり、放射スペクトル差を逆FFT RMSだけで説明することはできない。", "",
      "## 種類別集計", "", "|種類|例数|傷/8参照RMS比|z相当|参照RMS CV|参照との絶対相関|clean参照数|", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in type_rows: lines.append(f"|{r['defect_type']}|{r['examples']}|{r['median_rms_ratio']:.3f}|{r['median_z']:.3f}|{r['median_ref_cv']:.3f}|{r['median_abs_pattern_correlation']:.3f}|{r['median_clean_count']:.1f}|")
    lines += ["", "参照RMSのCVが大きい種類・例では、周囲正常自体の低周波振幅が一定でない。オレンジ枠の参照は傷・札・背景アノテーションと重なるため、正常ばらつきの推定には注意が必要である。要約CSVには全8参照統計に加えてclean参照だけの中央値・z相当値も保存した。", "",
      "## 放射スペクトル差との関係", "",
      f"従来スコアを持つ{int(valid.sum())}傷でのSpearman相関は、前回の0.02–0.05放射スペクトルRMS差に対して、振幅比の絶対対数ρ={ca:.3f}、RMS外れ度|z|ρ={cz:.3f}、空間パターン非類似度ρ={cs:.3f}だった。",
      "放射平均は方向と位相を捨て、周波数半径ごとのパワー形状を比較する。一方、逆FFT RMSは帯域全体の振幅量である。このため放射スペクトル差は、(1)帯域振幅の差、(2)帯域内での周波数配分の差、(3)周囲参照スペクトルからの外れ、の組合せに対応する。逆FFT像の向き・位置は理解には有用だが、放射平均スコアへ直接は入っていない。", "",
      "## 図の読み方", "", "- 上段は3×3配置。赤が傷窓、緑がclean参照、橙が傷・札・背景との重なりあり。", "- 中段は同じグレースケールの元128×128パッチ。", "- 下段は位相保持した複素FFTの0.02–0.05リングだけを逆変換した実部。赤青は符号付き輝度偏差。", "- `shared_scale` は9窓共通振幅、`auto_scale` は各窓の形状確認用。", "",
      "## 注意", "", "急峻なリングマスクによるリンギング、Hann窓による中心強調を含む。逆FFT像を物理的な凹凸・反射原因と断定しない。放射平均スペクトルRMS差と逆FFT RMSは別指標である。画像外参照は統計から除外し、図とCSVではunavailableとした。", "",
      "## 出力", "", "- `examples/*_shared_scale.png` / `*_auto_scale.png`: 傷ごとの9窓比較", "- `rms_comparison_*.png`: 種類別の全傷RMS棒グラフ", "- `window_band_metrics.csv`: 傷×9窓の定量値・汚染状態", "- `defect_summary.csv`: 傷別の参照比較", "- `defect_type_summary.csv`: 種類別集計"]
    (out/"report_ja.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True,exist_ok=True); (args.output/"examples").mkdir(exist_ok=True)
    selections=all_defect_selections(args.prior_results) if args.all_defects else read_selections(args.selections); window_rows=[]; summaries=[]
    for i,sel in enumerate(selections,1):
        print(f"{i}/{len(selections)} image{sel['image_number']} defect{sel['defect_number']}",flush=True)
        img,anns,windows=prepare(sel); stem=f"image{sel['image_number']}_defect{sel['defect_number']}_{i:02d}"
        figure(sel,img,windows,args.output/"examples"/f"{stem}_shared_scale.png",True); figure(sel,img,windows,args.output/"examples"/f"{stem}_auto_scale.png",False)
        for w in windows:
            window_rows.append({"image_id":f"image{sel['image_number']}","defect_id":sel["defect_number"],"defect_type":sel["defect_type"],"window_role":w["role"],"direction":w["direction"],
                "center_x":w["x"],"center_y":w["y"],"clean_flag":w["clean_flag"],"overlap_type":w["overlap_type"],"band_rms":w["rms"],"band_max":w["maximum"],"band_min":w["minimum"],
                "band_energy":w["energy"],"total_energy":w["total_energy"],"band_energy_fraction":w["energy_fraction_total"],"orientation_deg":w["orientation_deg"],"anisotropy":w["anisotropy"],"spread_px":w["spread_px"]})
        summaries.append(summarize(sel,windows))
    write_csv(args.output/"window_band_metrics.csv",window_rows); write_csv(args.output/"defect_summary.csv",summaries)
    write_csv(args.output/"all_defect_selections.csv",selections)
    overview(summaries,window_rows,args.output); make_report(args.output,summaries,window_rows)
    settings={"window_px":N,"stride_px":prior.STRIDE,"reference_spacing_px":prior.SPACING,"reference_order":dict(zip([f"ref{i}" for i in range(1,9)],DIRECTIONS)),
              "band_cycles_per_pixel":BAND,"preprocess":"subtract patch mean then separable 2-D Hann","inverse":"keep complex fft2 coefficients in radial ring, zero the rest, ifft2 real part",
              "z_score_like":"(defect RMS - mean of all 8 reference RMS)/population std of all 8 reference RMS","clean_flag":"no overlap with any defect, tag, or background annotation"}
    (args.output/"settings.json").write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--selections",type=Path,default=SELECTIONS); p.add_argument("--prior-results",type=Path,default=ROOT/"analysis_fft_normalization_128"/"defect_frequency_analysis.csv"); p.add_argument("--all-defects",action="store_true"); p.add_argument("--output",type=Path,default=OUT); main(p.parse_args())
