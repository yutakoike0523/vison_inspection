"""Initial dark point-anomaly validation on light-off images.

Uses only D=max(GaussianBlur(gray)-gray, 0) before any component filtering.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import fft_mask_ground_truth_evaluation as gt
from background_surface_hist_experiment import write_csv

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data20260903"/"data20260903"
OUT=ROOT/"outputs"/"dark_point_anomaly_validation"
TARGETS=("image1","image19","image20","image21")
POINT_LABEL=list(dict.fromkeys(gt.DAMAGE_MAP.values()))[0]

def score(gray,sigma):
    background=cv2.GaussianBlur(gray.astype(np.float32),(0,0),sigma,borderType=cv2.BORDER_REFLECT_101)
    return np.maximum(background-gray,0),background

def run_image(path,sigma,out):
    rec=gt.load_record(path); gray=rec["gray"].astype(np.float32); diff,blur=score(gray,sigma)
    point_shapes=[s for s in rec["shapes"] if s["normalized_label"]==POINT_LABEL]
    point_mask=np.zeros(gray.shape,bool)
    for s in point_shapes: point_mask|=s["mask"]
    # Normal comparison is strictly unlabeled surface: background/tag/any damage excluded.
    normal_mask=~(rec["masks"]["tag"]|rec["masks"]["background"]|rec["masks"]["damage"])
    rows=[]
    for index,s in enumerate(point_shapes,1):
        x1,y1,x2,y2=s["box"]; x0,y0=max(0,int(np.floor(x1))),max(0,int(np.floor(y1))); xa,ya=min(rec["width"],int(np.ceil(x2))),min(rec["height"],int(np.ceil(y2)))
        values=diff[y0:ya,x0:xa]
        rows.append({"image":rec["name"],"point_id":f"{rec['name']}_point_{index:02d}","x":x0,"y":y0,"width":xa-x0,"height":ya-y0,"area_px2":(xa-x0)*(ya-y0),"diff_mean":float(values.mean()),"diff_max":float(values.max()),"diff_p95":float(np.percentile(values,95)),"diff_p99":float(np.percentile(values,99)),"gray_mean":float(gray[y0:ya,x0:xa].mean())})
    # Diagnostic figure: original, local-darkness score, then pixel distributions.
    fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    axes[0].imshow(rec["rgb"]); axes[0].set_title(f"{rec['name']}: annotated point boxes")
    for i,s in enumerate(point_shapes,1):
        x1,y1,x2,y2=s["box"]; axes[0].add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor="lime",lw=2)); axes[0].text(x1,y1-4,str(i),color="lime",fontsize=9)
    axes[1].imshow(diff,cmap="magma",vmin=0,vmax=max(20,float(np.percentile(diff,99.9)))); axes[1].set_title(f"D=max(Gaussian(sigma={sigma}) - gray, 0)")
    for ax in axes[:2]: ax.axis("off")
    axes[2].hist(diff[normal_mask],bins=100,range=(0,max(40,float(np.percentile(diff,99.9)))),density=True,histtype="step",lw=1.5,label=f"normal surface (n={normal_mask.sum():,})")
    axes[2].hist(diff[point_mask],bins=100,range=(0,max(40,float(np.percentile(diff,99.9)))),density=True,histtype="step",lw=1.8,label=f"point boxes (n={point_mask.sum():,})")
    axes[2].set(xlabel="local darkness D (gray levels)",ylabel="density",title="Pixel score distribution"); axes[2].legend(); axes[2].grid(alpha=.25)
    fig.savefig(out/f"{rec['name']}_diagnostic.png",dpi=170); plt.close(fig)
    # Exploratory pixel threshold sweep; not used to tune detection on these images.
    sweep=[]
    for threshold in range(0,81):
        pred=diff>threshold
        sweep.append({"image":rec["name"],"threshold":threshold,"point_pixel_recall":float((pred&point_mask).sum()/max(1,point_mask.sum())),"normal_surface_false_positive_rate":float((pred&normal_mask).sum()/max(1,normal_mask.sum())),"point_box_hit_rate":float(sum((pred&s["mask"]).any() for s in point_shapes)/max(1,len(point_shapes)))} )
    return rows,sweep,{"image":rec["name"],"point_count":len(point_shapes),"normal_pixels":int(normal_mask.sum()),"point_pixels":int(point_mask.sum()),"normal_diff_median":float(np.median(diff[normal_mask])),"normal_diff_p99":float(np.percentile(diff[normal_mask],99)),"point_diff_median":float(np.median(diff[point_mask])) if point_mask.any() else None,"point_diff_p99":float(np.percentile(diff[point_mask],99)) if point_mask.any() else None}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--sigma",type=float,default=15); ap.add_argument("--output",type=Path,default=OUT); args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    points=[]; sweep=[]; summary=[]
    for name in TARGETS:
        p,s,info=run_image(DATA/f"{name}.json",args.sigma,args.output); points+=p; sweep+=s; summary.append(info)
    write_csv(args.output/"point_scores.csv",points); write_csv(args.output/"threshold_sweep.csv",sweep); write_csv(args.output/"image_score_summary.csv",summary)
    # Combined threshold curves (diagnostic only).
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for name in TARGETS:
        r=[x for x in sweep if x["image"]==name]; axes[0].plot([x["normal_surface_false_positive_rate"] for x in r],[x["point_pixel_recall"] for x in r],label=name); axes[1].plot([x["threshold"] for x in r],[x["point_box_hit_rate"] for x in r],label=name)
    axes[0].set(xlabel="normal-surface false-positive rate",ylabel="point-pixel recall",title="Threshold tradeoff"); axes[1].set(xlabel="D threshold",ylabel="point-box hit rate",title="Any-pixel hit per annotated point box")
    for ax in axes: ax.grid(alpha=.25); ax.legend()
    fig.savefig(args.output/"threshold_tradeoff.png",dpi=170); plt.close(fig)
    lines=["# Dark point-anomaly: initial local-darkness validation","",f"Images: {', '.join(TARGETS)}; Gaussian sigma={args.sigma}. D=max(blur-gray,0).", "", "## Per-image score summary", "| image | annotated points | normal D p99 | point D median | point D p99 |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {r['image']} | {r['point_count']} | {r['normal_diff_p99']:.2f} | {r['point_diff_median']:.2f} | {r['point_diff_p99']:.2f} |" for r in summary]
    lines += ["", "## Interpretation", "This run deliberately applies no area, shape, or connected-component filter. The threshold sweep is diagnostic; it must not be chosen from the same images for a final performance claim. Background/tag/damage annotations are excluded from the normal-surface comparison.", "", "## Files", "- *_diagnostic.png: source, local-darkness score, and normal-versus-point score distribution", "- point_scores.csv: rectangle-level score statistics", "- threshold_sweep.csv and threshold_tradeoff.png: diagnostic threshold behavior"]
    (args.output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"points={len(points)} output={args.output}")
if __name__=="__main__":main()
