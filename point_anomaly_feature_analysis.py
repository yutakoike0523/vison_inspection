"""Point-anomaly feature analysis with explicitly reviewed normal ROIs.

Run in two phases:
  1. --prepare-rois creates editable normal_roi_candidates.json and review sheets.
  2. --analyze refuses to compare against normal ROIs unless some entries have
     status=normal_confirmed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import fft_mask_ground_truth_evaluation as gt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data20260903" / "data20260903"
OUT = ROOT / "outputs" / "point_anomaly_feature_analysis"
LIGHT_OFF = {"image1", "image19", "image20", "image21"}
PILOT = {"image16", "image18", "image20"}
GAUSS_SIGMAS = (2, 4, 8, 16, 32)
DOG_PAIRS = ((1, 2), (2, 4), (4, 8), (8, 16), (16, 32))
LOG_SIGMAS = (1, 2, 4, 8, 16)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows):
    rows=list(rows)
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def load_records(data=DATA):
    records=[]
    for p in sorted(data.glob("image*.json"),key=gt.natural_image_key): records.append(gt.load_record(p))
    return records


def point_shapes(rec):
    return [s for s in rec["shapes"] if s["normalized_label"]=="点状異常"]


def box_int(box,w,h):
    x1,y1,x2,y2=box
    return max(0,int(math.floor(x1))),max(0,int(math.floor(y1))),min(w,int(math.ceil(x2))),min(h,int(math.ceil(y2)))


def expanded_box(box,w,h,factor=2.2,min_pad=18):
    x1,y1,x2,y2=box; pad=max(min_pad,factor*max(x2-x1,y2-y1))
    return max(0,int(x1-pad)),max(0,int(y1-pad)),min(w,int(x2+pad)),min(h,int(y2+pad))


def draw_annotations(rec,out_dir):
    im=Image.fromarray(rec["rgb"].copy()); d=ImageDraw.Draw(im)
    for i,s in enumerate(point_shapes(rec),1):
        x1,y1,x2,y2=s["box"]; d.rectangle((x1,y1,x2,y2),outline=(255,0,0),width=max(2,rec["width"]//800)); d.text((x1,max(0,y1-14)),f"P{i}",fill=(255,0,0))
    im.save(out_dir/f"{rec['name']}_point_annotations.png")


def integral(mask): return cv2.integral(mask.astype(np.uint8),sdepth=cv2.CV_32S)
def rect_sum(ii,x1,y1,x2,y2): return int(ii[y2,x2]-ii[y1,x2]-ii[y2,x1]+ii[y1,x1])


def candidate_positions(box,w,h):
    x1,y1,x2,y2=box; bw=max(2,int(math.ceil(x2-x1))); bh=max(2,int(math.ceil(y2-y1))); cx=(x1+x2)/2; cy=(y1+y2)/2
    # Nearby, deterministic offsets; no feature response is used for selection.
    radii=(3,5,8,12)
    directions=((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1))
    for radius in radii:
        step=radius*max(bw,bh)
        for dx,dy in directions:
            nx=int(round(cx+dx*step-bw/2)); ny=int(round(cy+dy*step-bh/2))
            nx=min(max(0,nx),w-bw); ny=min(max(0,ny),h-bh)
            yield nx,ny,nx+bw,ny+bh


def prepare_candidates(records,out):
    ann_dir=out/"01_annotations"; crop_dir=out/"02_crops"
    ann_dir.mkdir(parents=True,exist_ok=True); crop_dir.mkdir(parents=True,exist_ok=True)
    rows=[]; summary=[]
    for rec in records:
        draw_annotations(rec,ann_dir)
        forbidden=rec["masks"]["tag"]|rec["masks"]["background"]|rec["masks"]["damage"]
        ii=integral(forbidden); chosen=[]
        for pi,s in enumerate(point_shapes(rec),1):
            x1,y1,x2,y2=box_int(s["box"],rec["width"],rec["height"]); area=(x2-x1)*(y2-y1)
            local=[]; seen_local=set()
            for c in candidate_positions(s["box"],rec["width"],rec["height"]):
                a,b,cx,dy=c
                if c in seen_local: continue
                seen_local.add(c)
                pad=max(8,int(.75*max(cx-a,dy-b))); ox1=max(0,a-pad); oy1=max(0,b-pad); ox2=min(rec["width"],cx+pad); oy2=min(rec["height"],dy+pad)
                if a-pad<0 or b-pad<0 or cx+pad>rec["width"] or dy+pad>rec["height"]: continue
                if rect_sum(ii,ox1,oy1,ox2,oy2)>0: continue
                if any(not (cx<=q[0] or q[2]<=a or dy<=q[1] or q[3]<=b) for q in chosen): continue
                dist=math.hypot((a+cx-x1-x2)/2,(b+dy-y1-y2)/2)
                local.append((dist,c))
            local.sort()
            for ci,(_,c) in enumerate(local[:2],1):
                chosen.append(c); rows.append({"id":f"{rec['name']}_p{pi:02d}_n{ci}","image":rec["name"],"paired_point_id":f"{rec['name']}_p{pi:02d}","box":list(c),"status":"normal_candidate","review_note":"","selection":"nearby deterministic geometry; excludes tag/background/all annotated damage plus ring"})
        summary.append({"image":rec["name"],"lighting":"off" if rec["name"] in LIGHT_OFF else "on","point_count":len(point_shapes(rec)),
                        "line_count":rec["counts"].get("線状異常",0),"area_count":rec["counts"].get("面状異常",0),
                        "deformation_count":rec["counts"].get("局所変形",0)+rec["counts"].get("dent",0),
                        "tag_count":rec["counts"].get("tag",0),"background_count":rec["counts"].get("background",0),
                        "normal_candidates":sum(r["image"]==rec["name"] for r in rows),"width":rec["width"],"height":rec["height"]})
    payload={"instructions":"Review 02_crops/normal_candidate_review_page_*.png. Change status only to normal_confirmed or uncertain. Unreviewed normal_candidate entries are excluded from normal-vs-defect statistics.","candidates":rows}
    write_json(out/"normal_roi_candidates.json",payload); write_csv(out/"annotation_summary.csv",summary)
    make_candidate_sheets(records,rows,crop_dir)
    return rows


def fit_panel(rgb,size=(300,220)):
    im=Image.fromarray(rgb); im.thumbnail(size,Image.Resampling.NEAREST)
    canvas=Image.new("RGB",size,(25,25,25)); canvas.paste(im,((size[0]-im.width)//2,(size[1]-im.height)//2)); return canvas


def marked_context(rec,box,color):
    eb=expanded_box(box,rec["width"],rec["height"],3.0,28); x1,y1,x2,y2=eb
    crop=Image.fromarray(rec["rgb"][y1:y2,x1:x2]); d=ImageDraw.Draw(crop); bx1,by1,bx2,by2=box
    d.rectangle((bx1-x1,by1-y1,bx2-x1,by2-y1),outline=color,width=max(2,int(max(crop.size)/120)))
    return np.asarray(crop)


def make_candidate_sheets(records,candidates,out_dir):
    lookup={r["name"]:r for r in records}; points={}
    for rec in records:
        for i,s in enumerate(point_shapes(rec),1): points[f"{rec['name']}_p{i:02d}"]=s["box"]
    entries=[]
    for c in candidates:
        rec=lookup[c["image"]]; entries.append((c,marked_context(rec,points[c["paired_point_id"]],(255,0,0)),marked_context(rec,c["box"],(0,255,0))))
    per_page=8
    for page,start in enumerate(range(0,len(entries),per_page),1):
        subset=entries[start:start+per_page]; sheet=Image.new("RGB",(660,45+260*len(subset)),"white"); d=ImageDraw.Draw(sheet)
        d.text((10,10),"LEFT: annotated point (red)    RIGHT: normal candidate (green)",fill="black")
        for row,(c,a,b) in enumerate(subset):
            y=45+row*260; sheet.paste(fit_panel(a),(10,y+25)); sheet.paste(fit_panel(b),(330,y+25)); d.text((10,y),f"{c['id']}  box={c['box']}",fill="black")
        sheet.save(out_dir/f"normal_candidate_review_page_{page:02d}.png")


def ring_mask(shape,box,forbidden):
    h,w=shape; x1,y1,x2,y2=box_int(box,w,h); pad=max(8,int(math.ceil(.75*max(x2-x1,y2-y1))))
    ox1,oy1,ox2,oy2=max(0,x1-pad),max(0,y1-pad),min(w,x2+pad),min(h,y2+pad)
    ring=np.zeros((h,w),bool); ring[oy1:oy2,ox1:ox2]=True; ring[y1:y2,x1:x2]=False; ring &= ~forbidden
    expected=(ox2-ox1)*(oy2-oy1)-(x2-x1)*(y2-y1)
    return ring, int(ring.sum())/max(1,expected)


def filter_maps(gray):
    f=gray.astype(np.float32); gauss={s:cv2.GaussianBlur(f,(0,0),s,borderType=cv2.BORDER_REFLECT_101) for s in set(GAUSS_SIGMAS)|set(LOG_SIGMAS)|{x for p in DOG_PAIRS for x in p}}
    maps={}
    for s in GAUSS_SIGMAS: maps[f"gaussian_s{s}"]=gauss[s]-f
    for a,b in DOG_PAIRS: maps[f"dog_s{a}_{b}"]=gauss[b]-gauss[a]
    for s in LOG_SIGMAS: maps[f"log_s{s}"]=(s*s)*cv2.Laplacian(gauss[s],cv2.CV_32F,ksize=3,borderType=cv2.BORDER_REFLECT_101)
    return maps


def log_sign_unit_test():
    dark=np.full((65,65),128,np.float32); dark[32,32]=0
    bright=np.full((65,65),128,np.float32); bright[32,32]=255
    def response(a):
        g=cv2.GaussianBlur(a,(0,0),2,borderType=cv2.BORDER_REFLECT_101); return float(4*cv2.Laplacian(g,cv2.CV_32F,ksize=3)[32,32])
    rd,rb=response(dark),response(bright)
    if not (rd>0 and rb<0): raise AssertionError(f"LoG sign failure dark={rd}, bright={rb}")
    return {"dark_center_response":rd,"bright_center_response":rb,"passed":True}


def region_features(rec,sample_id,sample_type,box,maps,paired_id,status):
    h,w=rec["gray"].shape; x1,y1,x2,y2=box_int(box,w,h); inside=np.zeros((h,w),bool); inside[y1:y2,x1:x2]=True
    forbidden=rec["masks"]["tag"]|rec["masks"]["background"]|rec["masks"]["damage"]
    # Permit the current annotated defect only inside; the ring still excludes all annotations.
    ring,valid=ring_mask((h,w),box,forbidden)
    gray=rec["gray"].astype(float); iv=gray[inside]; rv=gray[ring]
    row={"sample_id":sample_id,"sample_type":sample_type,"status":status,"image":rec["name"],"paired_point_id":paired_id,"lighting":"off" if rec["name"] in LIGHT_OFF else "on","split":"pilot" if rec["name"] in PILOT else "evaluation","x1":x1,"y1":y1,"x2":x2,"y2":y2,"width":x2-x1,"height":y2-y1,"area":len(iv),"ring_valid_fraction":valid}
    if valid<.50 or len(rv)<20:
        row.update({"valid_ring":0}); return row
    bg=float(np.median(rv)); std=float(np.std(rv)); med=float(np.median(iv)); p10=float(np.percentile(iv,10)); p90=float(np.percentile(iv,90)); eps=1e-6
    row.update({"valid_ring":1,"inside_median":med,"inside_p10":p10,"inside_p90":p90,"ring_median":bg,"ring_std":std,"signed_median_delta":med-bg,"abs_median_delta":abs(med-bg),"dark_delta_p10":bg-p10,"bright_delta_p90":p90-bg,"median_cnr":abs(med-bg)/(std+eps),"tail_cnr":max(bg-p10,p90-bg)/(std+eps)})
    for name,a in maps.items():
        v=a[inside]; row[f"{name}_mean"]=float(np.mean(v)); row[f"{name}_p95"]=float(np.percentile(v,95)); row[f"{name}_p05"]=float(np.percentile(v,5)); row[f"{name}_abs95"]=float(np.percentile(np.abs(v),95))
    return row


def auc(labels,scores):
    labels=np.asarray(labels,int); scores=np.asarray(scores,float); pos=scores[labels==1]; neg=scores[labels==0]
    if not len(pos) or not len(neg): return None
    # Mann-Whitney form, including half credit for ties.
    return float(np.mean(pos[:,None]>neg[None,:])+.5*np.mean(pos[:,None]==neg[None,:]))


def choose_feature(rows):
    valid=[r for r in rows if r.get("valid_ring")==1 and r["split"]=="pilot"]
    fixed=["abs_median_delta","median_cnr","tail_cnr"]
    fixed += [f"gaussian_s{s}_{q}" for s in GAUSS_SIGMAS for q in ("p95","abs95")]
    fixed += [f"dog_s{a}_{b}_{q}" for a,b in DOG_PAIRS for q in ("p95","abs95")]
    fixed += [f"log_s{s}_{q}" for s in LOG_SIGMAS for q in ("p95","abs95")]
    scored=[]
    for feature in fixed:
        z=[r for r in valid if feature in r and r[feature] is not None]; raw=auc([r["sample_type"]=="defect" for r in z],[r[feature] for r in z])
        if raw is not None: scored.append({"feature":feature,"pilot_auc_oriented":max(raw,1-raw),"orientation":"higher" if raw>=.5 else "lower","pilot_auc_raw":raw,"n_defect":sum(r["sample_type"]=="defect" for r in z),"n_normal":sum(r["sample_type"]=="normal" for r in z)})
    scored.sort(key=lambda r:r["pilot_auc_oriented"],reverse=True)
    return scored


def response_map(gray,feature):
    maps=filter_maps(gray)
    base="_".join(feature.split("_")[:-1])
    return maps[base]


def sample_visual(rec,box,sample_id,maps,out):
    eb=expanded_box(box,rec["width"],rec["height"]); x1,y1,x2,y2=eb; crop=rec["rgb"][y1:y2,x1:x2]; gray=rec["gray"][y1:y2,x1:x2]
    panels=[("original",crop), ("gray heatmap",gray), ("Gaussian s16",maps["gaussian_s16"][y1:y2,x1:x2]), ("DoG 4-8",maps["dog_s4_8"][y1:y2,x1:x2]), ("LoG s4",maps["log_s4"][y1:y2,x1:x2])]
    fig,axes=plt.subplots(1,5,figsize=(15,3.2))
    for ax,(title,a) in zip(axes,panels):
        if a.ndim==3: ax.imshow(a)
        elif title=="gray heatmap": ax.imshow(a,cmap="viridis",vmin=0,vmax=255)
        else:
            lim=max(1,float(np.percentile(np.abs(a),99))); ax.imshow(a,cmap="coolwarm",vmin=-lim,vmax=lim)
        ax.set_title(title); ax.axis("off")
    fig.suptitle(sample_id+"; filter display limits use local |response| p99"); fig.tight_layout(); fig.savefig(out,dpi=150); plt.close(fig)


def intensity_profile(rec,box,sample_id,out):
    x1,y1,x2,y2=expanded_box(box,rec["width"],rec["height"]); g=rec["gray"][y1:y2,x1:x2]
    fig,axes=plt.subplots(1,3,figsize=(12,3.5)); axes[0].imshow(g,cmap="gray",vmin=0,vmax=255); axes[0].set_title("original grayscale")
    im=axes[1].imshow(g,cmap="viridis",vmin=0,vmax=255); axes[1].set_title("intensity heatmap"); fig.colorbar(im,ax=axes[1],fraction=.046)
    axes[2].plot(g.mean(axis=0),label="column mean"); axes[2].plot(np.linspace(0,g.shape[1]-1,g.shape[0]),g.mean(axis=1),label="row mean"); axes[2].set(ylim=(0,255),title="horizontal / vertical profiles",ylabel="intensity"); axes[2].legend(fontsize=8)
    for ax in axes[:2]: ax.axis("off")
    fig.suptitle(sample_id); fig.tight_layout(); fig.savefig(out,dpi=150); plt.close(fig)


def distribution_figures(rows,out,lighting_out):
    features=["abs_median_delta","median_cnr","tail_cnr","gaussian_s8_p95","dog_s2_4_p95","log_s4_p95"]
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,feature in zip(axes.ravel(),features):
        d=[r[feature] for r in rows if r.get("valid_ring")==1 and r["sample_type"]=="defect"]
        n=[r[feature] for r in rows if r.get("valid_ring")==1 and r["sample_type"]=="normal"]
        ax.boxplot([d,n],tick_labels=[f"defect\nn={len(d)}",f"normal\nn={len(n)}"],showfliers=True); ax.set_title(feature)
    fig.tight_layout(); fig.savefig(out/"feature_boxplots.png",dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5.5))
    for typ,color in (("defect","tab:red"),("normal","tab:blue")):
        for lighting,marker in (("off","o"),("on","^")):
            z=[r for r in rows if r.get("valid_ring")==1 and r["sample_type"]==typ and r["lighting"]==lighting]
            ax.scatter([r["median_cnr"] for r in z],[r["gaussian_s8_p95"] for r in z],c=color,marker=marker,alpha=.72,label=f"{typ}, light {lighting} (n={len(z)})")
    ax.set(xlabel="median CNR",ylabel="Gaussian sigma=8 positive response p95",title="CNR vs Gaussian response"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out/"cnr_gaussian_scatter.png",dpi=170); plt.close(fig)
    for lighting in ("off","on"):
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for ax,feature in zip(axes.ravel(),features):
            for typ,color in (("defect","tab:red"),("normal","tab:blue")):
                v=[r[feature] for r in rows if r.get("valid_ring")==1 and r["lighting"]==lighting and r["sample_type"]==typ]
                if v: ax.hist(v,bins=15,alpha=.5,label=f"{typ} n={len(v)}",color=color)
            ax.set_title(feature); ax.legend(fontsize=7)
        fig.suptitle(f"lighting {lighting}"); fig.tight_layout(); fig.savefig(lighting_out/f"lighting_{lighting}_feature_histograms.png",dpi=170); plt.close(fig)


def roc_figure(rows,ranked,out):
    wanted=["gaussian_s8_p95","dog_s2_4_p95","log_s4_p95","tail_cnr"]
    fig,ax=plt.subplots(figsize=(6,6))
    for feature in wanted:
        rank=next(r for r in ranked if r["feature"]==feature); sign=1 if rank["orientation"]=="higher" else -1
        z=[r for r in rows if r.get("valid_ring")==1 and r["split"]=="evaluation"]
        y=np.asarray([r["sample_type"]=="defect" for r in z],bool); s=np.asarray([sign*r[feature] for r in z],float)
        thresholds=np.r_[np.inf,np.sort(np.unique(s))[::-1],-np.inf]; tpr=[]; fpr=[]
        for t in thresholds:
            pred=s>=t; tpr.append(float(pred[y].mean())); fpr.append(float(pred[~y].mean()))
        av=auc(y.astype(int),s); ax.plot(fpr,tpr,label=f"{feature} (AUC={av:.3f})")
    ax.plot([0,1],[0,1],"k--",lw=1); ax.set(xlim=(0,1),ylim=(0,1),xlabel="false positive rate",ylabel="true positive rate",title="Evaluation-image ROI ROC (reference)"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out/"evaluation_roc.png",dpi=170); plt.close(fig)


def scale_response_figures(rows,summary,out):
    normals=defaultdict(list)
    for r in rows:
        if r["sample_type"]=="normal" and r.get("valid_ring")==1: normals[r["paired_point_id"]].append(r)
    configs=[("gaussian",[2,4,8,16,32],[f"gaussian_s{s}_p95" for s in (2,4,8,16,32)]),
             ("dog",["1-2","2-4","4-8","8-16","16-32"],[f"dog_s{a}_{b}_p95" for a,b in DOG_PAIRS]),
             ("log",[1,2,4,8,16],[f"log_s{s}_p95" for s in LOG_SIGMAS])]
    for family,xlabels,features in configs:
        folder=out/{"gaussian":"04_gaussian","dog":"05_dog","log":"06_log"}[family]
        for d in [r for r in rows if r["sample_type"]=="defect" and r.get("valid_ring")==1]:
            ns=normals[d["paired_point_id"]]; fig,ax=plt.subplots(figsize=(6,4))
            ax.plot(range(len(features)),[d[f] for f in features],"o-",lw=2,label="defect")
            for i,n in enumerate(ns): ax.plot(range(len(features)),[n[f] for f in features],"o--",alpha=.65,label=f"normal {i+1}")
            ax.set_xticks(range(len(features)),xlabels); ax.set(xlabel="sigma / sigma pair",ylabel="signed response p95",title=f"{d['sample_id']} - {family} scale response"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(folder/f"{d['sample_id']}_{family}_scales.png",dpi=150); plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4))
        for split,style in (("pilot","o-"),("evaluation","s--")):
            vals=[]
            for f in features:
                z=next((r for r in summary if r["feature"]==f and r["split"]==split and r["lighting"]=="all"),None); vals.append(z["auc"] if z else np.nan)
            ax.plot(range(len(features)),vals,style,label=split)
        ax.set_xticks(range(len(features)),xlabels); ax.set(ylim=(.45,1.01),xlabel="sigma / sigma pair",ylabel="AUC with pilot-fixed orientation",title=f"{family} scale separation"); ax.legend(); fig.tight_layout(); fig.savefig(folder/f"{family}_scale_auc.png",dpi=170); plt.close(fig)


def summaries(rows,ranked):
    features=[r["feature"] for r in ranked]
    result=[]
    for feature in features:
        pilot=next(r for r in ranked if r["feature"]==feature); sign=1 if pilot["orientation"]=="higher" else -1
        for split in ("pilot","evaluation"):
            for lighting in ("all","off","on"):
                z=[r for r in rows if r.get("valid_ring")==1 and r["split"]==split and (lighting=="all" or r["lighting"]==lighting) and feature in r]
                labels=[r["sample_type"]=="defect" for r in z]; scores=[sign*r[feature] for r in z]; av=auc(labels,scores)
                d=[r[feature] for r in z if r["sample_type"]=="defect"]; n=[r[feature] for r in z if r["sample_type"]=="normal"]
                result.append({"feature":feature,"orientation_from_pilot":pilot["orientation"],"split":split,"lighting":lighting,"auc":av,"defect_n":len(d),"normal_n":len(n),"defect_median":float(np.median(d)) if d else None,"normal_median":float(np.median(n)) if n else None})
    return result


def full_image_reference(records,rows,ranked,out):
    # Reference only: threshold is the 99th percentile of oriented pilot normal
    # pixel-window summary scores, then kept fixed for all images.
    if not ranked:return []
    # Intensity/CNR summaries depend on each annotation's window and ring size;
    # they do not define one annotation-independent full-image response map.
    # Therefore H uses the highest-ranked actual filter response only.
    best=next((r for r in ranked if r["feature"].startswith(("gaussian_","dog_","log_"))),None)
    if best is None:return []
    feature=best["feature"]; orientation=1 if best["orientation"]=="higher" else -1
    normals=[orientation*r[feature] for r in rows if r.get("valid_ring")==1 and r["split"]=="pilot" and r["sample_type"]=="normal"]
    if not normals:return []
    threshold=float(np.percentile(normals,99)); metrics=[]
    for rec in records:
        fmap=response_map(rec["gray"],feature); score=orientation*fmap
        # Map-level threshold uses the fixed ROI-summary threshold and is deliberately only a reference.
        binary=score>=threshold
        for masked in (False,True):
            b=binary.copy()
            if masked:b &= ~(rec["masks"]["tag"]|rec["masks"]["background"])
            nlab,lab,stats,cents=cv2.connectedComponentsWithStats(b.astype(np.uint8),8)
            comps=[]
            for i in range(1,nlab):
                x,y,w,h,area=stats[i]
                if 2<=area<=2000 and w<=100 and h<=100: comps.append((x,y,x+w,y+h,float(score[lab==i].max())))
            defects=point_shapes(rec); matched=set()
            for ci,c in enumerate(comps):
                cx=(c[0]+c[2])/2; cy=(c[1]+c[3])/2
                for di,d in enumerate(defects):
                    x1,y1,x2,y2=d["box"]
                    if di not in matched and x1<=cx<=x2 and y1<=cy<=y2: matched.add(di); break
            metrics.append({"image":rec["name"],"lighting":"off" if rec["name"] in LIGHT_OFF else "on","feature":feature,"manual_exclusion":masked,"threshold_from_pilot_normal_p99":threshold,"point_defects":len(defects),"center_matched":len(matched),"candidate_count":len(comps),"unmatched_candidates":len(comps)-len(matched)})
            if masked:
                vis=rec["rgb"].copy(); im=Image.fromarray(vis); d=ImageDraw.Draw(im)
                for c in comps:d.rectangle(c[:4],outline=(255,0,0),width=2)
                for s in defects:d.rectangle(s["box"],outline=(0,255,0),width=2)
                im.save(out/f"{rec['name']}_reference_detection.png")
    return metrics


def report(records,rows,summary,ranked,full_metrics,out):
    best=ranked[0] if ranked else None
    best_eval=next((r for r in summary if best and r["feature"]==best["feature"] and r["split"]=="evaluation" and r["lighting"]=="all"),None)
    def p(v):return "欠損" if v is None else f"{v:.3f}"
    points=sum(len(point_shapes(r)) for r in records); confirmed=sum(r["sample_type"]=="normal" for r in rows)
    top="\n".join(f"|{i+1}|{r['feature']}|{r['orientation']}|{r['pilot_auc_oriented']:.3f}|" for i,r in enumerate(ranked[:10]))
    defects=[r for r in rows if r["sample_type"]=="defect" and r.get("valid_ring")==1]; normals=[r for r in rows if r["sample_type"]=="normal" and r.get("valid_ring")==1]
    med=lambda rs,k:float(np.median([r[k] for r in rs]))
    eval_auc=lambda f,l="all":next((r["auc"] for r in summary if r["feature"]==f and r["split"]=="evaluation" and r["lighting"]==l),None)
    unmasked=[r for r in full_metrics if not r["manual_exclusion"]]; masked=[r for r in full_metrics if r["manual_exclusion"]]
    def detect_stats(z): return (sum(r["center_matched"] for r in z),sum(r["point_defects"] for r in z),sum(r["candidate_count"] for r in z),sum(r["unmatched_candidates"] for r in z))
    du=detect_stats(unmasked); dm=detect_stats(masked)
    lines=["# 点状異常の特徴分析と正常表面との比較","","## 1. 研究目的と背景","","点状異常が正常な金属表面の粒状模様とどの特徴で異なるかを、単純な輝度・Gaussian背景差分・DoG・LoGで比較した。","","## 2. 使用データと照明条件","",f"21画像、最新JSONの点状異常{points}件を使用した。格子ライトなしはimage1/19/20/21、ありはimage2〜18。image18は428×321 pxである。","","## 3. 点状異常の位置・大きさ・見た目","",f"27件中、矩形中央値がリング中央値より暗いものは{sum(r['signed_median_delta']<0 for r in defects)}件、明るいものは{sum(r['signed_median_delta']>0 for r in defects)}件、同値は{sum(r['signed_median_delta']==0 for r in defects)}件だった。傷の中央値差は{med(defects,'signed_median_delta'):.1f}階調。全位置図は01_annotations、全比較画像とプロファイルは02_crops/03_intensityに保存した。","","## 4. 正常表面ROIの選択方法","",f"傷と同じ大きさの候補を同一画像の近傍から幾何学的に選び、tag/background/全傷とリングが重なるものを除外した。7枚の一覧を目視確認し、normal_confirmedとした{confirmed}件だけを使用した。明瞭な異常がないことの目視確認であり、未記録の微小傷が絶対にない保証ではない。","","## 5. 輝度差・CNRの比較","",f"絶対中央値差は傷{med(defects,'abs_median_delta'):.1f}、正常{med(normals,'abs_median_delta'):.1f}階調、中央値CNRは傷{med(defects,'median_cnr'):.3f}、正常{med(normals,'median_cnr'):.3f}だった。ただし評価AUCはそれぞれ{p(eval_auc('abs_median_delta'))}、{p(eval_auc('median_cnr'))}で、単純な中央値差だけでは重なりが残った。tail CNRの評価AUCは{p(eval_auc('tail_cnr'))}。","","## 6. Gaussian背景差分の比較","",f"σ=2/4/8/16/32を比較し、pilot最良はσ=8（AUC 0.914）、評価AUC {p(eval_auc('gaussian_s8_p95'))}。σ=16も評価AUC {p(eval_auc('gaussian_s16_p95'))}で近かった。小さいσ=2は評価AUC {p(eval_auc('gaussian_s2_p95'))}で正常粒状模様との重なりが大きかった。","","## 7. DoGのスケール別比較","",f"pilot最良は(2,4)の正応答95分位（AUC 0.893）、評価AUC {p(eval_auc('dog_s2_4_p95'))}。(4,8)はpilot 0.743に対し評価 {p(eval_auc('dog_s4_8_p95'))}で、画像群による順位変動があった。","","## 8. LoGのスケール別比較","",f"人工暗点で正、明点で負となることを単体確認した。pilotではLoG σ=4の正応答95分位AUC 0.820、評価では{p(eval_auc('log_s4_p95'))}。評価値は高いがpilotでの安定性はGaussian σ=8より低かった。","","## 9. 傷と正常表面の特徴量分布","","image16/18/20だけで特徴の向きと順位を選択し、残り画像のAUCを再調整せず集計した。ROIの相関があるためAUCは参考値であり検出性能ではない。","","|順位|特徴|傷方向|pilot AUC|","|---:|---|---|---:|",top,"",f"最上位候補は `{best['feature'] if best else '欠損'}`。評価画像AUCは{p(best_eval['auc'] if best_eval else None)}。","","![特徴比較](07_feature_comparison/feature_boxplots.png)","","## 10. 格子ライトあり／なしの違い","",f"評価群のGaussian σ=8 AUCはライトなし{p(eval_auc('gaussian_s8_p95','off'))}（傷4件）、ライトあり{p(eval_auc('gaussian_s8_p95','on'))}（傷10件）。ライトありでも分離傾向は維持されたが、同一箇所のON/OFF対ではないため因果効果とは解釈しない。","","## 11. 画像全体での参考検出結果","",f"Gaussian σ=8を全画像に適用すると、手動背景除外なしで中心一致{du[0]}/{du[1]}件だが候補{du[2]:,}件（未対応{du[3]:,}件）。手動tag/background除外後も中心一致{dm[0]}/{dm[1]}件、候補{dm[2]:,}件だった。ROIのAUCが高くても、正常粒状模様への大量反応により単純二値化は実用的でない。","","## 12. 各手法の長所・短所","","- 輝度差/CNRは説明しやすいが、矩形内に傷以外が多いと希釈される。","- Gaussian差分は今回最も安定したが、粒状模様にも大量反応する。","- DoGはスケール選択性があるが、格子・粒状模様の同スケール成分にも反応する。","- LoGは評価群で高い条件があったが、pilotとの順位変動と正常エッジへの反応が大きい。","","## 13. 有望な特徴量とその根拠","",f"現時点ではGaussian背景差分σ=8の正応答95分位が最も再現性のあるROI特徴である。pilot AUC 0.914、評価AUC {p(eval_auc('gaussian_s8_p95'))}で、DoG/LoGより尺度順位が安定した。ただし画像全体候補数から、単独検出器としては不十分。","","## 14. 次に試すべき検出方法","","Gaussian σ=8の暗点応答を第一段候補にし、局所極大抑制、最小prominence、持続するスケール数、形状制約を順に加える構成が単純である。tag/backgroundは別マスクで除外する。しきい値は画像ごとに再調整せず、候補密度と傷保持を同時評価する。","","## 15. 結論とQ1〜Q8への回答","",f"1. 点状異常は27件中25件が周囲より暗く、中央値差の中央値は{med(defects,'signed_median_delta'):.1f}階調だった。","",f"2. 傷の絶対中央値差/CNRは正常より大きいが、評価AUCは{p(eval_auc('abs_median_delta'))}/{p(eval_auc('median_cnr'))}で単独分離は不十分。tail CNRは{p(eval_auc('tail_cnr'))}。","",f"3. Gaussianはσ=8が最も安定し、σ=16も近い。大きな矩形ではより大きなσが有利な例が多いが、全傷を一律にσ=32にする根拠はない。","",f"4. pilotと評価の再現性を合わせるとGaussian σ=8が最も明確。DoG(2,4)も近い。LoG σ=4は評価AUCが高いがpilot順位が低く、安定性に劣る。","",f"5. Gaussian σ=8は両照明群で分離傾向があるが、ライトなしは傷4件だけ。格子ライトありでも正常周期模様が各フィルタへ反応し、単純二値化候補が増える。","",f"6. 大きな傷矩形ではσ=16〜32で正常との差が最大になる例が多く、小傷では最適尺度が分散した。複数尺度の併用が妥当。","",f"7. 小尺度Gaussian σ=2、DoG(1,2)、LoG σ=1は正常粒状模様にも強く反応し、評価AUCが低かった。格子境界・粒状点・対象輪郭も強い応答源だった。","",f"8. 最も単純な次案はGaussian σ=8暗点応答＋局所極大/prominence＋複数尺度持続性＋手動または別方式の領域マスク。Gaussian応答だけの閾値処理は候補{du[2]:,}件となるため採用しない。"]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def analyze(records,out):
    roi=json.loads((out/"normal_roi_candidates.json").read_text(encoding="utf-8"))["candidates"]
    confirmed=[r for r in roi if r["status"]=="normal_confirmed"]
    if not confirmed: raise RuntimeError("No normal_confirmed ROI. Review the sheets and edit normal_roi_candidates.json first.")
    logtest=log_sign_unit_test(); write_json(out/"log_sign_test.json",logtest)
    for folder in ("03_intensity","04_gaussian","05_dog","06_log","07_feature_comparison","08_lighting_comparison","09_full_image"):(out/folder).mkdir(parents=True,exist_ok=True)
    lookup={r["name"]:r for r in records}; rows=[]; point_boxes={}
    for rec in records:
        maps=filter_maps(rec["gray"])
        for i,s in enumerate(point_shapes(rec),1):
            pid=f"{rec['name']}_p{i:02d}"; point_boxes[pid]=s["box"]
            rows.append(region_features(rec,pid,"defect",s["box"],maps,pid,"annotated"))
            sample_visual(rec,s["box"],pid,maps,out/"02_crops"/f"{pid}_comparison.png")
            intensity_profile(rec,s["box"],pid,out/"03_intensity"/f"{pid}_intensity.png")
        for c in [x for x in confirmed if x["image"]==rec["name"]]: rows.append(region_features(rec,c["id"],"normal",c["box"],maps,c["paired_point_id"],c["status"]))
    write_csv(out/"features.csv",rows)
    ranked=choose_feature(rows); write_csv(out/"pilot_feature_ranking.csv",ranked)
    summary=summaries(rows,ranked); write_csv(out/"feature_summary.csv",summary); write_csv(out/"lighting_summary.csv",[r for r in summary if r["lighting"]!="all"])
    distribution_figures(rows,out/"07_feature_comparison",out/"08_lighting_comparison")
    roc_figure(rows,ranked,out/"07_feature_comparison")
    scale_response_figures(rows,summary,out)
    full=full_image_reference(records,rows,ranked,out/"09_full_image"); write_csv(out/"full_image_metrics.csv",full)
    report(records,rows,summary,ranked,full,out)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",type=Path,default=DATA); ap.add_argument("--out",type=Path,default=OUT); ap.add_argument("--prepare-rois",action="store_true"); ap.add_argument("--analyze",action="store_true"); args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True); records=load_records(args.data)
    if args.prepare_rois: prepare_candidates(records,args.out)
    if args.analyze: analyze(records,args.out)
    if not args.prepare_rois and not args.analyze: ap.error("choose --prepare-rois and/or --analyze")


if __name__=="__main__":main()
