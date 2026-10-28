"""Relate annotated block occupancy to luminance-histogram shape.

This is analysis only: images and Labelme JSON are read, never changed.
Run: python occupancy_histogram_relation_analysis.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

import fft_mask_ground_truth_evaluation as gt
from background_surface_hist_experiment import DEFAULT_INPUT, bounds, histogram, image_number, read_gray, write_csv


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "occupancy_histogram_relation_analysis"
LIGHT_OFF = {"image1", "image19", "image20", "image21"}
FEATURES = ("iqr", "std", "peak_concentration", "entropy", "hist_peak_probability", "mean")
BINS = ((0, 10, "0-10%"), (10, 50, "10-50%"), (50, 90, "50-90%"), (90, 101, "90-100%"))


def exclusive_masks(record):
    """Return exclusive masks and raw overlap counts, priority damage>tag>background>surface."""
    raw = record["masks"]
    assigned = np.zeros_like(raw["tag"], bool)
    result = {}
    for name in ("damage", "tag", "background", "surface"):
        result[name] = raw[name] & ~assigned
        assigned |= result[name]
    # Annotation convention: any unlabelled pixel is inspection surface.
    result["explicit_surface"] = result["surface"].copy()
    result["surface"] = result["explicit_surface"] | ~assigned
    result["inferred_surface"] = result["surface"] & ~result["explicit_surface"]
    result["unknown"] = np.zeros_like(assigned, bool)
    result["background_tag_union"] = raw["background"] | raw["tag"]
    result["raw_overlap_pixels"] = int(sum((raw[a] & raw[b]).sum() for i, a in enumerate(("damage","tag","background","surface")) for b in ("damage","tag","background","surface")[i+1:]))
    return result


def block_features(path: Path, rows=7, cols=7):
    record = gt.load_record(path.with_suffix(".json"))
    gray = record["gray"].astype(float)
    masks = exclusive_masks(record)
    results, hists = [], []
    for row in range(rows):
        for col in range(cols):
            x0,y0,x1,y1 = bounds(*gray.shape, row, col, rows, cols)
            block = gray[y0:y1,x0:x1]
            hist = histogram(block)
            peak = int(np.argmax(hist)); q1,q3=np.percentile(block,[25,75])
            fractions={name: float(mask[y0:y1,x0:x1].mean()) for name,mask in masks.items() if name in {"background","tag","damage","surface","unknown","background_tag_union"}}
            p=hist[hist>0]
            results.append({"image":path.stem,"lighting":"off" if path.stem in LIGHT_OFF else "on","block_id":f"B{row+1},{col+1}","row":row+1,"col":col+1,"x0":x0,"y0":y0,"x1":x1,"y1":y1,"pixel_count":block.size,
                            **{f"{k}_percent":100*v for k,v in fractions.items()},
                            "mean":float(block.mean()),"std":float(block.std()),"iqr":float(q3-q1),"mode_brightness":peak,"hist_peak_probability":float(hist[peak]),"peak_concentration":float(hist[max(0,peak-10):min(256,peak+11)].sum()),"entropy":float(-(p*np.log2(p)).sum()),"white_255_fraction":float(hist[255]),"raw_annotation_overlap_pixels":masks["raw_overlap_pixels"]})
            hists.append(hist)
    # Invariant: every pixel belongs to exactly one exclusive class.
    exclusive_total=sum(masks[name].astype(np.uint8) for name in ("background","tag","damage","surface","unknown"))
    if not np.all(exclusive_total == 1): raise RuntimeError(f"exclusive mask error: {path.name}")
    return results, np.vstack(hists), masks


def save(fig,path): fig.savefig(path,dpi=160,bbox_inches="tight"); plt.close(fig)


def scatter_figure(rows,path,title):
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True); damage=np.array([r["damage_percent"] for r in rows])
    for ax,feature in zip(axes.ravel(),FEATURES):
        point=ax.scatter([r["background_tag_union_percent"] for r in rows],[r[feature] for r in rows],c=damage,cmap="Reds",vmin=0,vmax=max(1,damage.max()),s=18,edgecolors="none")
        ax.set(xlabel="background + tag occupancy (%)",ylabel=feature,title=feature); ax.grid(alpha=.2)
    fig.colorbar(point,ax=axes.ravel().tolist(),label="damage occupancy (%)"); fig.suptitle(title); save(fig,path)


def histograms_by_bin(rows,hists,path,title):
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    x=np.arange(256)
    for ax,(lo,hi,label) in zip(axes.ravel(),BINS):
        ids=[i for i,r in enumerate(rows) if lo <= r["background_tag_union_percent"] < hi]
        for i in ids: ax.plot(x,hists[i],color="#6f6f6f",lw=.45)
        if ids: ax.plot(x,np.mean(hists[ids],axis=0),color="#d95f02",lw=2,label=f"mean (n={len(ids)})")
        ax.set(title=f"background+tag {label}",xlabel="brightness",ylabel="probability"); ax.grid(alpha=.2)
        if ids: ax.legend(fontsize=8)
    fig.suptitle(title); save(fig,path)


def occupancy_heatmap(rows,path,title):
    value=np.array([r["background_tag_union_percent"] for r in rows]).reshape(7,7)
    fig,ax=plt.subplots(figsize=(6,5),constrained_layout=True); im=ax.imshow(value,vmin=0,vmax=100,cmap="Blues")
    for r in range(7):
        for c in range(7): ax.text(c,r,f"{value[r,c]:.0f}",ha="center",va="center",fontsize=8,color="black" if value[r,c]<55 else "white")
    ax.set(title=title,xlabel="column",ylabel="row",xticks=range(7),yticks=range(7)); fig.colorbar(im,ax=ax,label="background + tag occupancy (%)"); save(fig,path)


def corr_rows(rows):
    result=[]
    for subset_name,subset in (("all",rows),("damage_zero",[r for r in rows if r["damage_percent"]==0])):
        x=np.array([r["background_tag_union_percent"] for r in subset])
        for feature in FEATURES:
            y=np.array([r[feature] for r in subset])
            if len(x)>2 and x.std()>0 and y.std()>0:
                pear=pearsonr(x,y); spear=spearmanr(x,y)
                result.append({"subset":subset_name,"feature":feature,"n":len(x),"pearson_r":pear.statistic,"pearson_p":pear.pvalue,"spearman_rho":spear.statistic,"spearman_p":spear.pvalue})
    return result


def bin_summary(rows):
    result=[]
    for lo,hi,label in BINS:
        subset=[r for r in rows if lo<=r["background_tag_union_percent"]<hi]
        for feature in FEATURES:
            values=np.array([r[feature] for r in subset])
            if len(values): result.append({"occupancy_bin":label,"feature":feature,"n":len(values),"mean":values.mean(),"median":np.median(values),"q25":np.percentile(values,25),"q75":np.percentile(values,75)})
    return result


def threshold_rule(train,test,feature,positive_high):
    candidates=np.unique([r[feature] for r in train]); best=None
    for threshold in candidates:
        p=np.array([r[feature]>=threshold if positive_high else r[feature]<threshold for r in train]); y=np.array([r["background_tag_union_percent"]>=90 for r in train])
        tp=(p&y).sum(); fp=(p&~y).sum(); fn=(~p&y).sum(); f1=2*tp/max(1,2*tp+fp+fn)
        if best is None or f1>best[0]: best=(f1,threshold)
    threshold=best[1]; p=np.array([r[feature]>=threshold if positive_high else r[feature]<threshold for r in test]); y=np.array([r["background_tag_union_percent"]>=90 for r in test]); tp=int((p&y).sum());fp=int((p&~y).sum());fn=int((~p&y).sum());tn=int((~p&~y).sum())
    return {"rule":feature,"direction":"high" if positive_high else "low","train_threshold":threshold,"train_f1":best[0],"test_n":len(test),"test_accuracy":(tp+tn)/max(1,len(test)),"test_precision":tp/max(1,tp+fp),"test_recall":tp/max(1,tp+fn),"test_f1":2*tp/max(1,2*tp+fp+fn),"tp":tp,"tn":tn,"fp":fp,"fn":fn}


def simple_rules(rows):
    # Pure endpoints only, excluding uncertain unknown occupancy and damage.
    data=[r for r in rows if r["damage_percent"]==0 and r["unknown_percent"]<=10 and (r["background_tag_union_percent"]<=10 or r["background_tag_union_percent"]>=90)]
    train=[r for r in data if int(r["image"].replace("image",""))%2]; test=[r for r in data if not int(r["image"].replace("image",""))%2]
    if not train or not test or len({r["background_tag_union_percent"]>=90 for r in train})<2 or len({r["background_tag_union_percent"]>=90 for r in test})<2:
        return [{"rule":"not_evaluable","reason":"Need damage-free, low-unknown pure surface and pure background blocks in disjoint image sets."}]
    return [threshold_rule(train,test,"iqr",False),threshold_rule(train,test,"peak_concentration",True),threshold_rule(train,test,"entropy",False)]


def report(rows,corr,rules,out,image_count):
    ranked=sorted([r for r in corr if r["subset"]=="damage_zero"],key=lambda r:abs(r["spearman_rho"]),reverse=True)
    counts={label:sum(lo<=r["background_tag_union_percent"]<hi for r in rows) for lo,hi,label in BINS}
    high=sorted(rows,key=lambda r:r["background_tag_union_percent"],reverse=True)[:5]
    lines=["# Occupancy and histogram-shape relation", "", "## Scope", f"{image_count} images, 7x7 blocks, no classifier. Raw masks are recorded separately; exclusive occupancy uses damage > tag > background > explicit surface, then assigns all remaining pixels to inferred surface according to the stated annotation convention.", "", "## Data quality", f"- blocks: {len(rows)}", f"- occupancy bins: {counts}", "- each block's exclusive background, tag, damage, and surface percentages sum to 100% (validated).", "", "## Strongest damage-free relations", "| feature | Pearson r | Spearman rho | n |", "|---|---:|---:|---:|"]
    lines += [f"| {r['feature']} | {r['pearson_r']:.3f} | {r['spearman_rho']:.3f} | {r['n']} |" for r in ranked]
    lines += ["", "## Highest background/tag examples", ", ".join(f"{r['image']} {r['block_id']}: bg+tag={r['background_tag_union_percent']:.1f}%, IQR={r['iqr']:.1f}, peak concentration={r['peak_concentration']:.3f}" for r in high), "", "## Simple image-separated rule check", "Odd image numbers are training images and even numbers are evaluation images; only damage-free, unknown<=10%, endpoint occupancy blocks are eligible."]
    for rule in rules: lines.append("- " + ", ".join(f"{k}={v}" for k,v in rule.items()))
    lines += ["", "## Interpretation", "High background+tag occupancy is not a universal proxy for one histogram feature: inspect per-image scatter plots and occupancy-group histograms. Damage-containing blocks are shown separately; they are excluded from the primary relation check.", "", "## Outputs", "- block_occupancy_features.csv: occupancy and histogram features", "- correlation_summary.csv and occupancy_bin_summary.csv", "- occupancy_feature_scatter.png, occupancy_histograms.png, feature_distributions.png", "- per_image/: image-specific scatter, histogram, and occupancy heatmap"]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--input-dir",type=Path,default=DEFAULT_INPUT); ap.add_argument("--output-dir",type=Path,default=OUT); ap.add_argument("--images",default=""); ap.add_argument("--aggregate-only",action="store_true",help="rebuild only combined outputs; leave existing per-image outputs untouched")
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True); per=args.output_dir/"per_image"; per.mkdir(exist_ok=True)
    wanted=set(filter(None,args.images.split(","))); paths=sorted(args.input_dir.glob("image[0-9]*.png"),key=image_number); paths=[p for p in paths if not wanted or p.stem in wanted]
    all_rows=[]; all_hists=[]
    for path in paths:
        rows,hists,_=block_features(path); all_rows.extend(rows); all_hists.extend(hists)
        if not args.aggregate_only:
            dest=per/path.stem; dest.mkdir(exist_ok=True)
            scatter_figure(rows,dest/"occupancy_feature_scatter.png",f"{path.stem}: occupancy vs histogram features")
            histograms_by_bin(rows,hists,dest/"occupancy_histograms.png",f"{path.stem}: individual and mean histograms by occupancy")
            occupancy_heatmap(rows,dest/"background_tag_occupancy_heatmap.png",f"{path.stem}: background + tag occupancy")
    all_hists=np.vstack(all_hists); write_csv(args.output_dir/"block_occupancy_features.csv",all_rows)
    scatter_figure(all_rows,args.output_dir/"occupancy_feature_scatter.png","All images: occupancy vs histogram features")
    histograms_by_bin(all_rows,all_hists,args.output_dir/"occupancy_histograms.png","All images: individual and mean histograms by occupancy")
    # Damage-free companion visual required for isolating damage influence.
    keep=[i for i,r in enumerate(all_rows) if r["damage_percent"]==0]
    scatter_figure([all_rows[i] for i in keep],args.output_dir/"occupancy_feature_scatter_damage_zero.png","Damage-free blocks: occupancy vs histogram features")
    histograms_by_bin([all_rows[i] for i in keep],all_hists[keep],args.output_dir/"occupancy_histograms_damage_zero.png","Damage-free blocks: histograms by occupancy")
    corr=corr_rows(all_rows); write_csv(args.output_dir/"correlation_summary.csv",corr); write_csv(args.output_dir/"occupancy_bin_summary.csv",bin_summary(all_rows)); rules=simple_rules(all_rows); write_csv(args.output_dir/"simple_rule_results.csv",rules)
    # Endpoint feature distributions.
    pure_surface=[r for r in all_rows if r["background_tag_union_percent"]<=10 and r["damage_percent"]==0 and r["unknown_percent"]<=10]
    pure_background=[r for r in all_rows if r["background_tag_union_percent"]>=90 and r["damage_percent"]==0 and r["unknown_percent"]<=10]
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,feature in zip(axes,("iqr","peak_concentration","entropy")):
        if pure_surface and pure_background:
            ax.boxplot([[r[feature] for r in pure_surface],[r[feature] for r in pure_background]],tick_labels=["0-10% bg+tag","90-100% bg+tag"]); ax.set(title=feature,ylabel=feature)
        else: ax.text(.5,.5,"Not evaluable: no low-unknown endpoint blocks",ha="center",va="center",transform=ax.transAxes); ax.set(title=feature)
    save(fig,args.output_dir/"feature_distributions.png")
    for lighting in ("on","off"):
        ids=[i for i,r in enumerate(all_rows) if r["lighting"]==lighting]
        if ids: histograms_by_bin([all_rows[i] for i in ids],all_hists[ids],args.output_dir/f"occupancy_histograms_light_{lighting}.png",f"Light {lighting}: histograms by occupancy")
    report(all_rows,corr,rules,args.output_dir,len(paths)); print(f"completed {len(paths)} images, {len(all_rows)} blocks")


if __name__=="__main__": main()
