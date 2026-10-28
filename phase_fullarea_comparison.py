"""Compare existing masked phase analysis with full-area phase/amplitude scoring.

Run: python phase_fullarea_comparison.py [--image PATH] [--output PATH]
The C condition reuses B maps and changes evaluation/display support only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.stats import rankdata

from phase_anomaly_detection import Config, analyze, find_image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output_phase_fullarea_comparison"
REGION_IDS = [171, 154, 155, 156, 170, 172, 186, 187, 188]
EDGE_MARGINS = [50, 75, 100]


def region_boxes(shape):
    h, w = shape
    boxes = {}
    for rid in REGION_IDS:
        row, col = divmod(rid - 1, 16)
        boxes[rid] = [col*w//16, row*h//16, (col+1)*w//16, (row+1)*h//16]
    return boxes


def rank_scores(result, eval_mask=None):
    if eval_mask is None:
        eval_mask = np.ones(result["gray"].shape, bool)
    boxes = region_boxes(result["gray"].shape)
    rows = []
    for rid, (x0, y0, x1, y1) in boxes.items():
        mask = eval_mask[y0:y1, x0:x1]
        row = {"region_id": rid, "role": "candidate" if rid == 171 else "control",
               "box_xyxy": [x0, y0, x1, y1]}
        for key, name in [("normalized_phase_score", "phase"),
                          ("normalized_amplitude_score", "amplitude"), ("total_score", "total")]:
            values = result[key][y0:y1, x0:x1]
            values = values[mask & np.isfinite(values)]
            row[name + "_median"] = float(np.median(values)) if values.size else None
            row[name + "_mean"] = float(np.mean(values)) if values.size else None
            row[name + "_p95"] = float(np.percentile(values, 95)) if values.size else None
        rows.append(row)
    for name in ("phase", "amplitude", "total"):
        # Rank by regional mean so sparse positive responses are not all tied at median zero.
        # Average ranks make genuine ties explicit instead of favoring the first region ID.
        ranked = [r for r in rows if r[name+"_mean"] is not None]
        values = [-r[name+"_mean"] for r in ranked]
        for row, rank in zip(ranked, rankdata(values, method="average")):
            row[name+"_rank"] = float(rank)
    return rows


def top_percent_metrics(score, eligible, margins=(50, 100)):
    v = eligible & np.isfinite(score)
    if not v.any():
        return {"count": 0}
    threshold = float(np.percentile(score[v], 99))
    top = v & (score >= threshold)
    yy, xx = np.nonzero(top)
    h, w = score.shape
    edge_d = np.minimum.reduce([xx, yy, w-1-xx, h-1-yy])
    return {"count": int(len(xx)), "threshold_p99": threshold,
            "max_xy": [int(xx[np.argmax(score[top])]), int(yy[np.argmax(score[top])])],
            "top_xy_sample": [[int(x), int(y)] for x, y in zip(xx[::max(1, len(xx)//300)], yy[::max(1, len(yy)//300)])],
            "near_image_edge_fraction": {str(m): float(np.mean(edge_d <= m)) for m in margins}}


def boundary_metrics(score, valid):
    v = valid & np.isfinite(score)
    if not v.any():
        return {"count": 0}
    distance = distance_transform_edt(v)
    threshold = float(np.percentile(score[v], 99))
    top = v & (score >= threshold)
    y, x = np.unravel_index(np.nanargmax(np.where(v, score, np.nan)), score.shape)
    return {"max_xy": [int(x), int(y)], "max_distance_to_validity_boundary_px": float(distance[y, x]),
            "top1pct_within_50px_of_validity_boundary_fraction": float(np.mean(distance[top] <= 50)),
            "valid_pixels_within_50px_of_validity_boundary_fraction": float(np.mean(distance[v] <= 50)),
            "top1pct_count": int(top.sum())}


def save_score_maps(result, folder, condition, eval_mask=None):
    folder.mkdir(parents=True, exist_ok=True)
    gray = result["gray"]
    boxes = region_boxes(gray.shape)
    score_keys = [("normalized_phase_score", "phase_score", "Phase score (robust z)"),
                  ("normalized_amplitude_score", "amplitude_score", "Amplitude score (robust z)"),
                  ("total_score", "total_score", "Combined score")]
    for key, stem, title in score_keys:
        data = result[key].copy()
        if eval_mask is not None:
            data[~eval_mask] = np.nan
        finite = data[np.isfinite(data)]
        upper = max(6.0, float(np.percentile(finite, 99.5))) if finite.size else 6.0
        cmap = plt.get_cmap("inferno").copy(); cmap.set_bad("#777777")
        fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
        ax.imshow(data, cmap=cmap, vmin=0, vmax=upper)
        ax.imshow(gray, cmap="gray", vmin=0, vmax=255, alpha=.22)
        for rid, (x0,y0,x1,y1) in boxes.items():
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,
                                   edgecolor="cyan" if rid == 171 else "lime",linewidth=1.1 if rid == 171 else .6))
            ax.text(x0+2,y0+12,str(rid),color="cyan" if rid == 171 else "lime",fontsize=7)
        if condition == "C" and eval_mask is not None:
            margin = int(np.flatnonzero(eval_mask.any(axis=1))[0]) if eval_mask.any() else 0
            h,w = gray.shape
            ax.add_patch(Rectangle((margin,margin),w-2*margin,h-2*margin,fill=False,
                                   edgecolor="white",linestyle="--",linewidth=1.2))
        ax.set_title(title + (" — gray = outside evaluation area" if condition == "C" else ""))
        ax.set_axis_off(); fig.colorbar(ax.images[0], ax=ax, shrink=.75, label="positive robust z")
        fig.savefig(folder/(stem+".png"), dpi=150); plt.close(fig)
    total=result["total_score"]
    valid=np.isfinite(total)
    amount=np.clip(np.nan_to_num(total)/6.0,0,1)
    base=np.repeat((gray/255)[...,None],3,axis=2)
    color=plt.get_cmap("inferno")(amount)[...,:3]
    alpha=(.72*amount*valid)[...,None]
    Image.fromarray(np.uint8(np.clip((base*(1-alpha)+color*alpha)*255,0,255))).save(folder/"overlay.png")
    # Neutral grayscale with R171 clearly identified.
    fig, ax = plt.subplots(figsize=(10,8), layout="constrained")
    ax.imshow(gray,cmap="gray",vmin=0,vmax=255)
    for rid,(x0,y0,x1,y1) in boxes.items():
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor="cyan" if rid==171 else "lime",linewidth=1.1 if rid==171 else .6))
        ax.text(x0+2,y0+12,str(rid),color="cyan" if rid==171 else "lime",fontsize=7)
    ax.set_title("image15: R171 (cyan) and eight fixed controls (green)"); ax.set_axis_off()
    fig.savefig(folder/"regions.png",dpi=150); plt.close(fig)


def comparison_plot(results, output, key, filename, title, masks=None):
    fig, axes = plt.subplots(1,3,figsize=(18,6),layout="constrained")
    arrays=[]
    for r,m in zip(results,masks or [None]*3):
        a=r[key].copy()
        if m is not None: a[~m]=np.nan
        arrays.append(a)
    finite=np.concatenate([a[np.isfinite(a)] for a in arrays if np.isfinite(a).any()])
    vmax=max(6,float(np.percentile(finite,99.5)))
    cm=plt.get_cmap("inferno").copy();cm.set_bad("#777777")
    for ax,a,label in zip(axes,arrays,["A: current", "B: full-area", "C: full-area, edge-excluded evaluation"]):
        im=ax.imshow(a,cmap=cm,vmin=0,vmax=vmax);ax.set_title(label);ax.axis("off")
    fig.suptitle(title);fig.colorbar(im,ax=axes,shrink=.75,label="positive robust z")
    fig.savefig(output/filename,dpi=150);plt.close(fig)


def save_rank_plot(all_rows, output):
    names=["A_current","B_full","C_margin75"]
    labels=["A current","B full","C margin75"]
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout="constrained")
    for ax,metric,title in zip(axes,["phase","amplitude","total"],["Phase rank","Amplitude rank","Combined rank"]):
        vals=[]
        for name in names:
            row=next(x for x in all_rows[name] if x["region_id"]==171)
            vals.append(row.get(metric+"_rank",np.nan))
        ax.bar(labels,vals,color=["#4c78a8","#f58518","#54a24b"]);ax.set_ylim(9.5,.5);ax.set_yticks(range(1,10));ax.set_ylabel("Rank (1 = highest score)");ax.set_title(title)
        for i,v in enumerate(vals):ax.text(i,v-.25,f"{v:g}" if np.isfinite(v) else "NA",ha="center")
    fig.suptitle("R171 rank by mean score among R171 and eight controls (ties averaged)");fig.savefig(output/"r171_rank_comparison.png",dpi=160);plt.close(fig)


def save_top_locations(results, output, masks):
    fig,axes=plt.subplots(1,3,figsize=(18,6),layout="constrained")
    for ax,r,mask,label in zip(axes,results,masks,["A: validity boundary", "B: image edge", "C: margin75 eval area"]):
        score=r["total_score"];valid=np.isfinite(score) if mask is None else np.isfinite(score)&mask
        threshold=np.percentile(score[valid],99); yy,xx=np.nonzero(valid&(score>=threshold))
        ax.imshow(r["gray"],cmap="gray",vmin=0,vmax=255)
        ax.scatter(xx,yy,s=3,c="red",alpha=.65)
        ax.add_patch(Rectangle((705,894),70,89,fill=False,edgecolor="cyan",linewidth=1.4));ax.text(707,890,"R171",color="cyan")
        if label.startswith("A"):
            boundary=~r["phase_valid"]
            ax.contour(boundary,levels=[.5],colors="yellow",linewidths=.35)
        elif label.startswith("C"):
            ax.add_patch(Rectangle((75,75),r["gray"].shape[1]-150,r["gray"].shape[0]-150,fill=False,edgecolor="white",linestyle="--"))
        ax.set_title(label+f"\nred = top 1% combined score ({len(xx)} px)");ax.axis("off")
    fig.savefig(output/"top1percent_location_comparison.png",dpi=160);plt.close(fig)


def save_summary(results, output, masks):
    r0,r1,r2=results
    fig,axes=plt.subplots(3,4,figsize=(18,13),layout="constrained")
    axes[0,0].imshow(r0["gray"],cmap="gray",vmin=0,vmax=255);axes[0,0].set_title("Original grayscale")
    keys=["total_score","normalized_phase_score","normalized_amplitude_score"]
    row_titles=["Combined", "Phase", "Amplitude"]
    for row,(key,rowtitle) in enumerate(zip(keys,row_titles)):
        for col,(r,label,mask) in enumerate(zip(results,["A current","B full-area","C edge-excluded eval"],masks),1):
            if row==0: axes[0,col].imshow(r["gray"],cmap="gray",vmin=0,vmax=255)
            a=r[key].copy()
            if mask is not None:a[~mask]=np.nan
            cm=plt.get_cmap("inferno").copy();cm.set_bad("#777777")
            axes[row,col].imshow(a,cmap=cm,vmin=0,vmax=6,alpha=.88)
            if row==0: axes[row,col].set_title(label)
            if row>0 and col==1: axes[row,col].set_ylabel(rowtitle)
        # show phase row's first cell also; first column of rows 1 and 2 is A
        if row>0:
            a=r0[key].copy();cm=plt.get_cmap("inferno").copy();cm.set_bad("#777777")
            axes[row,0].imshow(a,cmap=cm,vmin=0,vmax=6)
            axes[row,0].set_title(rowtitle+" A")
    for ax in axes.ravel():
        ax.add_patch(Rectangle((705,894),70,89,fill=False,edgecolor="cyan",linewidth=.9));ax.axis("off")
    fig.suptitle("image15 full-area experiment — cyan box marks R171; gray in C is excluded from evaluation")
    fig.savefig(output/"image15_fullarea_summary.png",dpi=150);plt.close(fig)


def amplitude_phase_diagnostics(result):
    amp=np.mean([d["amplitude_corrected"] for d in result["directions"]],axis=0)
    phase=result["normalized_phase_score"]
    valid=np.isfinite(phase)&np.isfinite(amp)
    a=amp[valid];p=phase[valid]
    corr=float(np.corrcoef(a,p)[0,1]) if a.size>1 and np.std(a)>0 and np.std(p)>0 else None
    cuts=np.quantile(a,[1/3,2/3])
    groups=[a<=cuts[0],(a>cuts[0])&(a<=cuts[1]),a>cuts[1]]
    return {"pearson_corr_amplitude_with_phase_score":corr,
            "phase_score_by_amplitude_tertile": [{"amplitude_range": [float(np.min(a[g])),float(np.max(a[g]))],
                "pixel_count":int(g.sum()),"phase_score_median":float(np.median(p[g])),"phase_score_mean":float(np.mean(p[g]))} for g in groups]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image",type=Path,default=None)
    parser.add_argument("--output",type=Path,default=OUT)
    parser.add_argument("--edge-margins",type=int,nargs="+",default=EDGE_MARGINS)
    parser.add_argument("--use-hann",action=argparse.BooleanOptionalAction,default=True, help="Use Hann window for B/C; A always keeps existing condition")
    parser.add_argument("--no-padding",action="store_true",help="Padding comparison is intentionally not enabled yet")
    args=parser.parse_args()
    source=args.image or find_image()
    gray=np.asarray(Image.open(source).convert("L"),dtype=float)
    config=Config()
    print(f"Input {source}, shape={gray.shape}; auto-detecting FFT carriers",flush=True)
    a=analyze(gray,config)
    print("Computing B: all image pixels participate in phase/amplitude scoring",flush=True)
    b=analyze(gray,config,[(p["dx"],p["dy"]) for p in a["peaks"]],full_area=True,use_hann=args.use_hann)
    margin=int(75 if 75 in args.edge_margins else args.edge_margins[0])
    h,w=gray.shape
    yy,xx=np.indices(gray.shape)
    edge_distance=np.minimum.reduce([xx,yy,w-1-xx,h-1-yy])
    c_mask=edge_distance>=margin
    root=args.output
    folders={"A":root/"01_condition_A_current","B":root/"02_condition_B_full","C":root/"03_condition_C_full_edge_removed"}
    for label,res in [("A",a),("B",b),("C",b)]:save_score_maps(res,folders[label],label,c_mask if label=="C" else None)
    masks=[None,None,c_mask]
    comp=root/"04_comparison";comp.mkdir(parents=True,exist_ok=True)
    comparison_plot([a,b,b],comp,"normalized_phase_score","phase_comparison.png","Phase anomaly score: A vs B/C",masks)
    comparison_plot([a,b,b],comp,"normalized_amplitude_score","amplitude_comparison.png","Amplitude anomaly score: A vs B/C",masks)
    comparison_plot([a,b,b],comp,"total_score","total_comparison.png","Combined score: A vs B/C",masks)
    rows={"A_current":rank_scores(a),"B_full":rank_scores(b),"C_margin75":rank_scores(b,c_mask)}
    save_rank_plot(rows,comp);save_top_locations([a,b,b],comp,masks);save_summary([a,b,b],root,masks)
    # Boundary concentration distinguishes the old validity boundary from the outer image edge.
    boundary={"A_current":{k:boundary_metrics(a[k],a["phase_valid"] if k!="normalized_amplitude_score" else a["amplitude_valid"])
                          for k in ["normalized_phase_score","normalized_amplitude_score","total_score"]},
              "B_full":{k:top_percent_metrics(b[k],np.ones(gray.shape,bool),[50,100])
                        for k in ["normalized_phase_score","normalized_amplitude_score","total_score"]},
              "C_full_edge_removed":{str(m):{k:top_percent_metrics(b[k],edge_distance>=m,[50,100])
                                            for k in ["normalized_phase_score","normalized_amplitude_score","total_score"]}
                                      for m in args.edge_margins}}
    c_rows=rank_scores(b,c_mask)
    # Make the requested A/B/C comparison metrics explicit and report all edge-margin sensitivities.
    report={"image":source.name,"image_path":str(source.resolve()),"image_shape_hw":list(gray.shape),
            "conditions":{"A_current":{"description":"Existing Hann + valid support + low-amplitude exclusion; scoring/evaluation on valid pixels.","r171_and_controls":rows["A_current"]},
                          "B_full":{"description":"Hann retained; no spatial-validity or low-amplitude exclusion in final phase/amplitude score.","phase_valid_fraction":float(b["phase_valid"].mean()),"amplitude_valid_fraction":float(b["amplitude_valid"].mean()),"r171_and_controls":rows["B_full"]},
                          "C_full_edge_removed":{"description":"Same score maps as B; only final evaluation excludes outer border.","edge_margin_px":margin,"r171_and_controls":c_rows}},
            "selected_fft_peaks":a["peaks"],"hann_enabled_for_B_C":args.use_hann,
            "edge_margin_sensitivity":{str(m):rank_scores(b,edge_distance>=m) for m in args.edge_margins},
            "boundary_concentration":boundary,"amplitude_phase_relation":{"A_current":amplitude_phase_diagnostics(a),"B_full":amplitude_phase_diagnostics(b)},
            "padding_comparison":"deferred" if not args.no_padding else "disabled by CLI",
            "interpretation_limits":["R171 and controls are fixed review areas, not verified ground-truth labels.",
                "A and B each robust-normalize against their own valid population, so absolute score magnitudes are not directly comparable; compare spatial concentration, ranks, and distributions.",
                "B deliberately includes weak-carrier and edge pixels; high phase scores there may be unstable estimates, not defects.",
                "C does not recompute FFT or phase; it is a reporting/evaluation mask only.",
                "Only image15 was analyzed; findings do not establish defect-detection performance."]}
    root.mkdir(parents=True,exist_ok=True)
    (root/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    with (root/"all_regions_summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=["condition","region_id","role","phase_median","phase_mean","phase_p95","phase_rank","amplitude_median","amplitude_mean","amplitude_p95","amplitude_rank","total_median","total_mean","total_p95","total_rank"]);writer.writeheader()
        for condition,items in rows.items():
            for r in items:writer.writerow({"condition":condition,"region_id":r["region_id"],"role":r["role"],
                "phase_median":r["phase_median"],"phase_mean":r["phase_mean"],"phase_p95":r["phase_p95"],"phase_rank":r.get("phase_rank"),"amplitude_median":r["amplitude_median"],"amplitude_mean":r["amplitude_mean"],"amplitude_p95":r["amplitude_p95"],"amplitude_rank":r.get("amplitude_rank"),"total_median":r["total_median"],"total_mean":r["total_mean"],"total_p95":r["total_p95"],"total_rank":r.get("total_rank")})
    readme = f"""# image15: Full-area phase and amplitude comparison

## Purpose
The existing method excludes weak Hann-window edges and low-amplitude carrier pixels. This experiment tests whether scores concentrate on those artificial validity boundaries and whether that changes R171 rankings.

## Conditions
- **A**: Existing method: Hann window, validity support, and low-amplitude exclusion.
- **B**: Full-area scores; no spatial-validity or low-amplitude exclusion. Hann enabled: {args.use_hann}.
- **C**: Reuses B score maps and excludes only the outer {margin}px during final evaluation. Margin sensitivity ({", ".join(map(str,args.edge_margins))} px) is in `report.json`.
- Toggle the B/C window with `--use-hann` or `--no-use-hann`. Padding comparisons are deferred.

## Figures
- `image15_fullarea_summary.png`: Top row: grayscale and A/B/C combined scores; middle: phase; bottom: amplitude. Cyan rectangle is R171; gray in C is outside the evaluation area.
- `04_comparison/phase_comparison.png`, `amplitude_comparison.png`, `total_comparison.png`: A/B/C side-by-side on shared display scales.
- `04_comparison/r171_rank_comparison.png`: R171 rank among the fixed eight controls. Rank uses region mean robust-z score; ties receive average rank.
- `04_comparison/top1percent_location_comparison.png`: Locations of top 1% combined scores. Yellow contour in A marks the validity boundary; dashed box in C marks the evaluation margin.
- Per-condition `phase_score.png`, `amplitude_score.png`, `total_score.png`, `overlay.png`, and `regions.png` show individual maps and region positions.

## Interpretation
A and B normalize scores over different pixel populations, so compare spatial distribution, rankings, and boundary concentration rather than treating colors as absolute units. B intentionally includes weak-carrier and edge pixels; a high phase score there can be an unstable estimate rather than a defect. C does not recompute the FFT or phase; it is an evaluation-only mask. R171 is a visual candidate, not a verified defect label. This is a one-image experiment and does not establish detection performance.

`report.json` contains region medians, means, 95th percentiles and ranks; top-score location and edge concentration; amplitude/phase correlation; and phase-score summaries by amplitude tertile. `all_regions_summary.csv` contains A/B/C scores for R171 and the eight controls.
"""
    (root/"README.md").write_text(readme,encoding="utf-8")
    print(json.dumps({"output":str(root.resolve()),"A_R171":next(r for r in rows["A_current"] if r["region_id"]==171),
                      "B_R171":next(r for r in rows["B_full"] if r["region_id"]==171),
                      "C_R171":next(r for r in c_rows if r["region_id"]==171)},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
