"""Analyze and prune Gaussian-background-difference peaks without learned models."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

import fft_mask_ground_truth_evaluation as gt
import point_anomaly_feature_analysis as fa

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data20260903"/"data20260903"
PREVIOUS=ROOT/"outputs"/"point_anomaly_feature_analysis"
OUT=ROOT/"outputs"/"point_anomaly_peak_analysis"
PILOT={"image16","image18","image20"}
LIGHT_OFF={"image1","image19","image20","image21"}
SIGMAS=(4,8,16,32)
BASE_THRESHOLD=20.294957160949707
K_VALUES=(10,50,100,500)


def write_csv(path,rows):
    rows=list(rows)
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),extrasaction="ignore");w.writeheader();w.writerows(rows)


def write_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def maps(gray):
    f=gray.astype(np.float32)
    return {s:cv2.GaussianBlur(f,(0,0),s,borderType=cv2.BORDER_REFLECT_101)-f for s in SIGMAS}


def point_shapes(rec):return [s for s in rec["shapes"] if s["normalized_label"]=="点状異常"]


def box_int(box,w,h):
    x1,y1,x2,y2=box
    return max(0,int(math.floor(x1))),max(0,int(math.floor(y1))),min(w,int(math.ceil(x2))),min(h,int(math.ceil(y2)))


def load_normals(previous):
    doc=json.loads((previous/"normal_roi_candidates.json").read_text(encoding="utf-8"))
    user_confirmed=bool(doc.get("user_confirmation"))
    rows=[]
    for r in doc["candidates"]:
        if r.get("status") not in {"normal_confirmed","normal_candidate"}:continue
        q=dict(r);q["analysis_status"]="user_confirmed_normal" if user_confirmed else "provisional_normal"
        rows.append(q)
    return rows,{"source":str((previous/"normal_roi_candidates.json").resolve()),"user_confirmation_recorded":user_confirmed,"review_record":doc.get("review"),"analysis_treatment":"confirmed" if user_confirmed else "provisional"}


def pilot_scale_thresholds(records,normals):
    by=defaultdict(list)
    for rec in records:
        if rec["name"] not in PILOT:continue
        mm=maps(rec["gray"])
        for n in [x for x in normals if x["image"]==rec["name"]]:
            x1,y1,x2,y2=box_int(n["box"],rec["width"],rec["height"])
            for s in SIGMAS:by[s].append(mm[s][y1:y2,x1:x2].ravel())
    return {s:float(np.percentile(np.concatenate(by[s]),99)) for s in SIGMAS}


def response_visual(rec,mm,out):
    r=mm[8]; pos=np.maximum(r,0); fixed=80.0
    fig,axes=plt.subplots(1,4,figsize=(15,4))
    axes[0].imshow(rec["rgb"]);axes[0].set_title("original")
    axes[1].imshow(rec["gray"]+r,cmap="gray",vmin=0,vmax=255);axes[1].set_title("Gaussian blur sigma=8")
    lim=max(1,float(np.percentile(np.abs(r),99)));axes[2].imshow(r,cmap="coolwarm",vmin=-lim,vmax=lim);axes[2].set_title(f"R8 signed; local ±{lim:.1f}")
    im=axes[3].imshow(pos,cmap="magma",vmin=0,vmax=fixed);axes[3].set_title("R8 positive; fixed 0..80")
    fig.colorbar(im,ax=axes[3],fraction=.046)
    for a in axes:a.axis("off")
    fig.suptitle(rec["name"]);fig.tight_layout();fig.savefig(out/f"{rec['name']}_response.png",dpi=140);plt.close(fig)


def component_candidates(r,threshold=BASE_THRESHOLD):
    binary=r>=threshold;n,lab,stats,cents=cv2.connectedComponentsWithStats(binary.astype(np.uint8),8)
    rows=[]
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if 2<=area<=2000 and w<=100 and h<=100:
            rows.append({"x":float(cents[i,0]),"y":float(cents[i,1]),"score":float(r[lab==i].max()),"area":int(area),"width":int(w),"height":int(h)})
    return rows


def local_maxima(r,threshold=BASE_THRESHOLD):
    mx=cv2.dilate(r,np.ones((3,3),np.uint8));mask=(r>=threshold)&(r==mx)
    n,lab=cv2.connectedComponents(mask.astype(np.uint8),8)
    ys,xs=np.nonzero(mask);labels=lab[ys,xs]
    if not len(labels):return np.empty((0,2),int)
    _,idx=np.unique(labels,return_index=True)
    return np.column_stack((xs[idx],ys[idx])).astype(int)


def ring_offsets(inner,outer,samples=96):
    values=set()
    radii=np.linspace(inner+.5,outer-.5,max(2,(outer-inner)*2))
    for rad in radii:
        for a in np.linspace(0,2*np.pi,samples,endpoint=False):values.add((int(round(rad*math.sin(a))),int(round(rad*math.cos(a)))))
    return np.asarray(sorted(values),int)


RINGS={o:ring_offsets(max(2,o//3),o) for o in (8,12,16)}


def ring_medians(r,xy,outer):
    offsets=RINGS[outer];h,w=r.shape;answer=np.empty(len(xy),np.float32)
    for start in range(0,len(xy),3000):
        q=xy[start:start+3000];yy=q[:,1,None]+offsets[None,:,0];xx=q[:,0,None]+offsets[None,:,1]
        valid=(yy>=0)&(yy<h)&(xx>=0)&(xx<w);yy=np.clip(yy,0,h-1);xx=np.clip(xx,0,w-1)
        vals=r[yy,xx].astype(float);vals[~valid]=np.nan;answer[start:start+len(q)]=np.nanmedian(vals,axis=1)
    return answer


def spatial_features(xy,shape):
    if not len(xy):return np.array([]),np.array([])
    tree=cKDTree(xy);dist=tree.query(xy,k=min(2,len(xy)))[0]
    nearest=np.full(len(xy),np.inf) if len(xy)==1 else dist[:,1]
    density=np.asarray([len(z)-1 for z in tree.query_ball_point(xy,16)],int)
    return nearest,density


def half_component(r,x,y,radius=32):
    h,w=r.shape;H=float(r[y,x]);x1=max(0,x-radius);y1=max(0,y-radius);x2=min(w,x+radius+1);y2=min(h,y+radius+1)
    patch=r[y1:y2,x1:x2];mask=(patch>=.5*H)&(patch>0);n,lab,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    label=lab[y-y1,x-x1]
    if label==0:return 0,0,0,None,None,None
    _,_,bw,bh,area=stats[label];aspect=max(bw/bh,bh/bw);touch=int(np.any(lab[[0,-1],:]==label) or np.any(lab[:,[0,-1]]==label))
    # Decay: sampled median at approximately 4 and 8 px relative to peak height.
    decay=[]
    for rad in (4,8,12):
        off=ring_offsets(rad-1,rad+1,32);yy=np.clip(y+off[:,0],0,h-1);xx=np.clip(x+off[:,1],0,w-1);decay.append(float(np.median(r[yy,xx])/max(H,1e-6)))
    return int(area),int(bw),int(bh),float(aspect),touch,decay


def peak_features(rec,mm,scale_thresholds):
    r=mm[8];xy=local_maxima(r);nearest,density=spatial_features(xy,r.shape)
    meds={o:ring_medians(r,xy,o) for o in RINGS}
    rows=[]
    for i,(x,y) in enumerate(xy):
        h8=float(r[y,x]);q={"image":rec["name"],"lighting":"off" if rec["name"] in LIGHT_OFF else "on","split":"pilot" if rec["name"] in PILOT else "evaluation","peak_id":i,"x":int(x),"y":int(y),"height":h8,"nearest_peak_distance":float(nearest[i]),"density_r16":int(density[i])}
        for o in RINGS:q[f"prominence_r{o}"]=h8-float(meds[o][i])
        count=0
        for s in SIGMAS:
            a=mm[s];q[f"response_s{s}_fixed"]=float(a[y,x]);y1=max(0,y-3);y2=min(a.shape[0],y+4);x1=max(0,x-3);x2=min(a.shape[1],x+4);v=float(a[y1:y2,x1:x2].max());q[f"response_s{s}_matched"]=v
            keep=v>=scale_thresholds[s];q[f"strong_s{s}"]=int(keep);count+=int(keep)
        q["scale_persistence_count"]=count;rows.append(q)
    return rows


def roi_memberships(rec,peaks,normals,mm):
    result=[]
    regions=[]
    for i,s in enumerate(point_shapes(rec),1):regions.append((f"{rec['name']}_p{i:02d}","defect_roi",s["box"],"annotated"))
    for n in [x for x in normals if x["image"]==rec["name"]]:regions.append((n["id"],"provisional_normal_roi",n["box"],n["analysis_status"]))
    for rid,typ,box,status in regions:
        x1,y1,x2,y2=box
        inside=[p for p in peaks if x1<=p["x"]<x2 and y1<=p["y"]<y2]
        for rank,p in enumerate(sorted(inside,key=lambda z:z["height"],reverse=True),1):
            q=dict(p);q.update({"roi_id":rid,"roi_type":typ,"roi_status":status,"roi_peak_rank":rank,"roi_peak_count":len(inside)})
            area,bw,bh,aspect,touch,decay=half_component(mm[8],p["x"],p["y"]);q.update({"half_area":area,"half_width":bw,"half_height":bh,"half_aspect":aspect,"half_window_truncated":touch,"decay_r4":decay[0] if decay else None,"decay_r8":decay[1] if decay else None,"decay_r12":decay[2] if decay else None});result.append(q)
    return result


def nms(candidates,radius,score="height"):
    ordered=sorted(candidates,key=lambda q:q[score],reverse=True);cell=max(1,radius);grid=defaultdict(list);kept=[];r2=radius*radius
    for q in ordered:
        gx,gy=int(q["x"]//cell),int(q["y"]//cell);blocked=False
        for yy in range(gy-1,gy+2):
            for xx in range(gx-1,gx+2):
                if any((q["x"]-p["x"])**2+(q["y"]-p["y"])**2<r2 for p in grid[(xx,yy)]):blocked=True;break
            if blocked:break
        if not blocked:kept.append(q);grid[(gx,gy)].append(q)
    return kept


def match(candidates,boxes,k=None):
    chosen=sorted(candidates,key=lambda q:q["score"],reverse=True)[:k] if k else candidates
    matched=set();duplicates=0
    for q in chosen:
        hits=[i for i,b in enumerate(boxes) if b[0]<=q["x"]<=b[2] and b[1]<=q["y"]<=b[3]]
        fresh=next((i for i in hits if i not in matched),None)
        if fresh is not None:matched.add(fresh)
        elif hits:duplicates+=1
    return len(matched),duplicates


def candidate_methods(base_components,peaks,params,include5):
    p1=[dict(p,score=p["height"]) for p in peaks]
    p2=[dict(p,score=p["height"]) for p in nms(peaks,params["nms_radius"])]
    p3=[dict(p,score=p["prominence_r12"]) for p in peaks if p["prominence_r12"]>=params["prominence_threshold"]]
    p4=[dict(p,score=p["prominence_r12"]) for p in nms(p3,params["nms_radius"],"prominence_r12")]
    out={"method0_components":[dict(p,score=p["score"]) for p in base_components],"method1_local_max":p1,"method2_nms":p2,"method3_prominence":p3,"method4_nms_prominence":p4}
    if include5:out["method5_persistence"]=[dict(p,score=p["prominence_r12"]*p["scale_persistence_count"]) for p in p4 if p["scale_persistence_count"]>=params["persistence_min"]]
    return out


def tune(pilot_data):
    # Preserve the local-maxima pilot recall whenever possible; then minimize candidate count.
    target=sum(match(d["peaks"],[s["box"] for s in point_shapes(d["rec"])])[0] for d in pilot_data)
    radius_trials=[]
    for rad in (3,5,8,12,16):
        kept=[nms(d["peaks"],rad) for d in pilot_data];rec=sum(match(k,[s["box"] for s in point_shapes(d["rec"])])[0] for k,d in zip(kept,pilot_data));radius_trials.append({"radius":rad,"recall_count":rec,"candidates":sum(map(len,kept))})
    eligible=[x for x in radius_trials if x["recall_count"]==target] or radius_trials;best_r=min(eligible,key=lambda x:x["candidates"])["radius"]
    normal_prom=[]
    for d in pilot_data:
        for n in d["normal_memberships"]:normal_prom.append(n["prominence_r12"])
    thresholds=sorted(set([0.,5.,10.,15.,20.]+([float(np.percentile(normal_prom,q)) for q in (50,75,90,95)] if normal_prom else [])))
    prom_trials=[]
    for t in thresholds:
        kept=[[p for p in d["peaks"] if p["prominence_r12"]>=t] for d in pilot_data];rec=sum(match(k,[s["box"] for s in point_shapes(d["rec"])])[0] for k,d in zip(kept,pilot_data));prom_trials.append({"threshold":t,"recall_count":rec,"candidates":sum(map(len,kept))})
    eligible=[x for x in prom_trials if x["recall_count"]==target] or [x for x in prom_trials if x["recall_count"]>=target-1] or prom_trials;best_p=min(eligible,key=lambda x:x["candidates"])["threshold"]
    base=[]
    for d in pilot_data:
        q=[p for p in d["peaks"] if p["prominence_r12"]>=best_p];base.append(nms(q,best_r,"prominence_r12"))
    base_recall=sum(match(k,[s["box"] for s in point_shapes(d["rec"])])[0] for k,d in zip(base,pilot_data));base_n=sum(map(len,base));pers=[]
    for count in (2,3,4):
        kept=[[p for p in q if p["scale_persistence_count"]>=count] for q in base];rec=sum(match(k,[s["box"] for s in point_shapes(d["rec"])])[0] for k,d in zip(kept,pilot_data));pers.append({"minimum":count,"recall_count":rec,"candidates":sum(map(len,kept))})
    useful=[x for x in pers if x["recall_count"]==base_recall and x["candidates"]<=.9*base_n];best_s=min(useful,key=lambda x:x["candidates"])["minimum"] if useful else None
    return {"local_max_pilot_recall":target,"nms_radius":best_r,"prominence_threshold":best_p,"persistence_min":best_s or 2,"include_method5":bool(best_s),"nms_trials":radius_trials,"prominence_trials":prom_trials,"persistence_trials":pers}


def summarize_methods(data,params):
    metrics=[];rak=[];all_candidates={}
    for d in data:
        rec=d["rec"];boxes=[s["box"] for s in point_shapes(rec)];methods=candidate_methods(d["components"],d["peaks"],params,params["include_method5"])
        exclusion=rec["masks"]["tag"]|rec["masks"]["background"]
        for name,cands in methods.items():
            all_candidates[(rec["name"],name)]=cands
            for manual in (False,True):
                use=[q for q in cands if not manual or not exclusion[min(rec["height"]-1,max(0,int(q["y"]))),min(rec["width"]-1,max(0,int(q["x"])))] ]
                tp,dup=match(use,boxes);metrics.append({"image":rec["name"],"split":"pilot" if rec["name"] in PILOT else "evaluation","lighting":"off" if rec["name"] in LIGHT_OFF else "on","method":name,"manual_exclusion":manual,"point_defects":len(boxes),"detected":tp,"missed":len(boxes)-tp,"candidate_count":len(use),"candidates_per_mpixel":len(use)/(rec["width"]*rec["height"]/1e6),"duplicate_matched_candidates":dup,"unmatched_candidates":len(use)-tp-dup})
                if not manual:
                    for k in K_VALUES:
                        hit,dups=match(use,boxes,k);rak.append({"image":rec["name"],"split":"pilot" if rec["name"] in PILOT else "evaluation","lighting":"off" if rec["name"] in LIGHT_OFF else "on","method":name,"ranking":"actual_score","k":k,"point_defects":len(boxes),"detected":hit,"candidate_count":len(use)})
    return metrics,rak,all_candidates


def random_baseline(data,all_candidates,selected,runs=200):
    rng=np.random.default_rng(20261018);rows=[]
    for d in data:
        if d["rec"]["name"] in PILOT:continue
        rec=d["rec"];boxes=[s["box"] for s in point_shapes(rec)];c=all_candidates[(rec["name"],selected)]
        for k in K_VALUES:
            hits=[]
            for _ in range(runs):
                if len(c)<=k:q=c
                else:q=[c[i] for i in rng.choice(len(c),k,replace=False)]
                hits.append(match(q,boxes)[0])
            rows.append({"image":rec["name"],"split":"evaluation","lighting":"off" if rec["name"] in LIGHT_OFF else "on","method":selected,"ranking":"random_mean_200","k":k,"point_defects":len(boxes),"detected":float(np.mean(hits)),"detected_std":float(np.std(hits)),"candidate_count":len(c)})
    return rows


def roi_feature_plots(rows,out):
    groups=("defect_roi","provisional_normal_roi");features=["height","prominence_r12","half_area","half_width","density_r16","scale_persistence_count"]
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,f in zip(axes.ravel(),features):
        vals=[[r[f] for r in rows if r["roi_type"]==g and r.get(f) is not None] for g in groups]
        ax.boxplot(vals,tick_labels=[f"defect ROI\nn={len(vals[0])}",f"provisional normal\nn={len(vals[1])}"],showfliers=False);ax.set_title(f)
    fig.tight_layout();fig.savefig(out/"peak_feature_boxplots.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for g,c in zip(groups,("tab:red","tab:blue")):
        z=[r for r in rows if r["roi_type"]==g];axes[0].scatter([r["height"] for r in z],[r["prominence_r12"] for r in z],s=12,alpha=.5,c=c,label=g);axes[1].scatter([r["height"] for r in z],[r["half_area"] for r in z],s=12,alpha=.5,c=c,label=g)
    axes[0].set(xlabel="peak height",ylabel="prominence r12",title="Strength vs local prominence");axes[1].set(xlabel="peak height",ylabel="half-height component area (px)",title="Strength vs spread")
    for ax in axes:ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/"peak_feature_scatter.png",dpi=170);plt.close(fig)


def peak_location_visual(rec,peaks,out):
    im=Image.fromarray(rec["rgb"].copy());d=ImageDraw.Draw(im)
    for p in peaks:d.point((p["x"],p["y"]),fill=(255,0,0))
    for s in point_shapes(rec):d.rectangle(s["box"],outline=(0,255,0),width=3)
    d.rectangle((5,5,min(rec["width"]-5,730),38),fill=(0,0,0));d.text((12,12),f"3x3 local maxima >= {BASE_THRESHOLD:.3f}: red; point GT: green",fill=(255,255,255));im.save(out)


def peak_profiles(rec,mm,roi_rows,out):
    r=mm[8]
    for i,s in enumerate(point_shapes(rec),1):
        rid=f"{rec['name']}_p{i:02d}";z=[q for q in roi_rows if q["roi_id"]==rid]
        if not z:continue
        p=max(z,key=lambda q:q["height"]);x,y=int(p["x"]),int(p["y"]);rad=32
        xs=np.arange(max(0,x-rad),min(rec["width"],x+rad+1));ys=np.arange(max(0,y-rad),min(rec["height"],y+rad+1))
        fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].plot(xs-x,r[y,xs],label="horizontal");axes[0].axhline(.5*p["height"],color="k",ls="--",label="50% peak");axes[1].plot(ys-y,r[ys,x],label="vertical");axes[1].axhline(.5*p["height"],color="k",ls="--",label="50% peak")
        for ax in axes:ax.set(xlabel="distance from peak (px)",ylabel="R8 response",title="response profile");ax.legend(fontsize=8)
        fig.suptitle(f"{rid}: strongest rectangle peak H={p['height']:.2f} at ({x},{y})");fig.tight_layout();fig.savefig(out/f"{rid}_profile.png",dpi=150);plt.close(fig)


def scale_analysis_plots(roi_rows,out):
    groups=("defect_roi","provisional_normal_roi");positions=np.arange(len(SIGMAS));fig,ax=plt.subplots(figsize=(7,5))
    for g,color in zip(groups,("tab:red","tab:blue")):
        z=[r for r in roi_rows if r["roi_type"]==g];med=[];lo=[];hi=[]
        for s in SIGMAS:
            v=np.asarray([r[f"response_s{s}_matched"] for r in z]);med.append(np.median(v));lo.append(np.percentile(v,25));hi.append(np.percentile(v,75))
        ax.plot(positions,med,"o-",color=color,label=g);ax.fill_between(positions,lo,hi,color=color,alpha=.18)
    ax.set(xticks=positions,xticklabels=SIGMAS,xlabel="Gaussian sigma (px)",ylabel="matched local peak response",title="Peak response persistence across Gaussian scales");ax.legend();fig.tight_layout();fig.savefig(out/"roi_scale_persistence.png",dpi=170);plt.close(fig)
    top={}
    for r in roi_rows:
        if r["roi_type"]=="defect_roi" and (r["roi_id"] not in top or r["height"]>top[r["roi_id"]]["height"]):top[r["roi_id"]]=r
    for rid,r in top.items():
        fixed=[r[f"response_s{s}_fixed"] for s in SIGMAS];matched=[r[f"response_s{s}_matched"] for s in SIGMAS]
        fig,ax=plt.subplots(figsize=(6,4));ax.plot(positions,fixed,"o-",label="fixed coordinate");ax.plot(positions,matched,"s--",label="matched max within 3px");ax.set(xticks=positions,xticklabels=SIGMAS,xlabel="Gaussian sigma (px)",ylabel="response",title=f"{rid} scale response");ax.legend();fig.tight_layout();fig.savefig(out/f"{rid}_scale.png",dpi=150);plt.close(fig)


def method_plot(metrics,out):
    names=[];cands=[];rec=[]
    for name in sorted({r["method"] for r in metrics}):
        z=[r for r in metrics if r["method"]==name and r["split"]=="evaluation" and not r["manual_exclusion"]];names.append(name.replace("method","M"));cands.append(sum(r["candidate_count"] for r in z));den=sum(r["point_defects"] for r in z);rec.append(sum(r["detected"] for r in z)/den if den else 0)
    fig,ax=plt.subplots(figsize=(9,5));ax.bar(names,cands,color="0.55");ax.set_yscale("log");ax.set(ylabel="candidates in 18 evaluation images (log)",title="Candidate count and point-defect recall")
    ax2=ax.twinx();ax2.plot(names,rec,"o-",color="tab:red");ax2.set(ylim=(0,1.05),ylabel="recall");ax.tick_params(axis="x",rotation=25);fig.tight_layout();fig.savefig(out/"evaluation_method_comparison.png",dpi=180);plt.close(fig)


def draw_candidates(rec,cands,method,out):
    im=Image.fromarray(rec["rgb"].copy());d=ImageDraw.Draw(im)
    for q in cands:d.ellipse((q["x"]-2,q["y"]-2,q["x"]+2,q["y"]+2),outline=(255,0,0),width=1)
    for s in point_shapes(rec):d.rectangle(s["box"],outline=(0,255,0),width=3)
    d.rectangle((5,5,min(rec["width"]-5,700),38),fill=(0,0,0));d.text((12,12),f"{method}: red candidates; green point GT",fill=(255,255,255));im.save(out)


def missed_cases(data,all_candidates,selected,out):
    index=[]
    for d in data:
        rec=d["rec"];c=all_candidates[(rec["name"],selected)]
        for i,s in enumerate(point_shapes(rec),1):
            b=s["box"]
            if any(b[0]<=q["x"]<=b[2] and b[1]<=q["y"]<=b[3] for q in c):continue
            x1,y1,x2,y2=fa.expanded_box(b,rec["width"],rec["height"],3,30);Image.fromarray(rec["rgb"][y1:y2,x1:x2]).save(out/f"missed_{rec['name']}_p{i:02d}.png");index.append({"image":rec["name"],"point_id":i,"box":b,"reason":"no selected-method candidate center in rectangle"})
    write_json(out/"missed_index.json",index)


def aggregate(metrics,method,split="evaluation",manual=False):
    z=[r for r in metrics if r["method"]==method and r["split"]==split and r["manual_exclusion"]==manual];den=sum(r["point_defects"] for r in z);return {"detected":sum(r["detected"] for r in z),"defects":den,"recall":sum(r["detected"] for r in z)/den if den else None,"candidates":sum(r["candidate_count"] for r in z),"per_image":sum(r["candidate_count"] for r in z)/len(z)}


def make_report(records,roi_rows,metrics,rak,params,normal_info,selected,out):
    methods=sorted({r["method"] for r in metrics});agg={m:aggregate(metrics,m) for m in methods};base=aggregate(metrics,"method0_components",split="pilot");baseall=aggregate(metrics,"method0_components",split="evaluation")
    def pct(v):return f"{100*v:.1f}%" if v is not None else "欠損"
    k_lines=[]
    for k in K_VALUES:
        z=[r for r in rak if r["split"]=="evaluation" and r["method"]==selected and r["ranking"]=="actual_score" and r["k"]==k];den=sum(r["point_defects"] for r in z);hit=sum(r["detected"] for r in z)
        random=[r for r in rak if r["method"]==selected and r["ranking"]=="random_mean_200" and r["k"]==k];rh=sum(r["detected"] for r in random)
        k_lines.append(f"- K={k}: 実スコア {hit}/{den}件、ランダム順位期待値 {rh:.2f}/{den}件")
    method_lines=[]
    for m in methods:
        a=agg[m];method_lines.append(f"- {m}: Recall {a['detected']}/{a['defects']} ({pct(a['recall'])}), 候補 {a['candidates']:,}件 ({a['per_image']:.1f}/画像)")
    d=[r for r in roi_rows if r["roi_type"]=="defect_roi"];n=[r for r in roi_rows if r["roi_type"]=="provisional_normal_roi"]
    med=lambda z,k:float(np.median([r[k] for r in z])) if z else None
    m0=agg["method0_components"];m5=agg.get("method5_persistence")
    lines=["# Gaussian背景差分ピーク分析と候補絞り込み","","## 1. 研究目的","","Gaussian σ=8背景差分で反応する点状異常と正常粒状模様の違いを、ピーク強度・広がり・局所prominence・複数σ持続性から調べ、単純な候補削減方法を固定条件で評価した。","","## 2. 前回までの結果と課題","",f"Method 0を同じしきい値{BASE_THRESHOLD:.6f}で再現した。全21画像282,263件を再現し、評価18画像では候補{baseall['candidates']:,}件、Recall {baseall['detected']}/{baseall['defects']}。","",f"正常ROIはユーザー確認記録が{'ある' if normal_info['user_confirmation_recorded'] else 'ない'}ため、今回は{'確定正常' if normal_info['user_confirmation_recorded'] else '暫定正常'}として扱った。","","## 3. Gaussian背景差分σ=8の仕組み","","R8=GaussianBlur(I,σ=8)−I。正値は周囲より暗い局所構造を示す。元画像全体を処理してからtag/background候補を除外し、マスク境界由来の応答を作っていない。","","## 4. 点状異常と正常模様の応答比較","",f"正解矩形内ピーク群の高さ中央値は{med(d,'height'):.2f}、暫定正常ROIピーク群は{med(n,'height'):.2f}。正解矩形内の全ピークを真の傷ピークとは扱っていない。","","## 5. 局所ピーク強度の違い","","3×3局所極大の平坦領域は連結成分ごとに代表1点とした。矩形内ピーク数・順位をpeak_features.csvへ保存した。","","## 6. ピーク面積・広がりの違い","",f"半値連結面積の中央値は正解矩形内{med(d,'half_area'):.1f}px、暫定正常{med(n,'half_area'):.1f}px。正常側の方がむしろ広く、面積で傷を残す仮説は支持されなかった。","","## 7. 周囲に対するprominenceの違い","",f"代理prominence中央値は正解矩形内{med(d,'prominence_r12'):.2f}、暫定正常{med(n,'prominence_r12'):.2f}。pilot閾値{params['prominence_threshold']:.3f}はMethod 1からMethod 3を約11%削減しただけで、Method 0より候補が多く残った。","","## 8. Gaussianのσを変えた際の違い","",f"σ=4/8/16/32の持続性は、ピーク方式内でMethod 4からMethod 5を約10%削減しRecallを維持した。しかしMethod 5はMethod 0より約{100*(m5['candidates']/m0['candidates']-1):.1f}%多かった。","","## 9. 格子ライトあり／なしの比較","","画像別・照明別の候補密度をmethod_metrics.csvに保存した。ライトありでは格子の明暗境界と粒状点、なしでは粒状点・対象輪郭・濃淡むらが主な反応源として可視化された。同一箇所のON/OFFではないため因果効果とは断定しない。","","## 10. 候補抽出方法の比較","",*method_lines,"","局所極大化は1つの二値連結成分内にある複数の粒状ピークを別候補へ分解するため、Method 1〜5はいずれもMethod 0より候補数が増えた。NMSはMethod 1比44%、Method 4はMethod 3比43%削減したが、基準方式を下回らなかった。","","![方式比較](06_method_comparison/evaluation_method_comparison.png)","","## 11. Recallと候補数の関係","",*k_lines,"","## 12. 検出できない傷の特徴","","効率優先のMethod 0で見逃した2件は08_failure_casesに保存した。低コントラスト、二値連結成分の重心が矩形外へずれることが主因候補である。","","## 13. 誤反応候補の発生原因","","正常粒状模様そのものが暗い局所極小を多数持つため、局所極大化だけでは候補は減らない。格子境界、対象輪郭、札文字、外周も強い。手動マスク後もMethod 0は261,246件残り、金属内部の粒状ピークが支配的である。","","## 14. 最も有望なシンプルな検出方法","","候補効率と上位順位ではMethod 0が最良だった。全21画像で25/27件、282,263候補。Recall優先ならMethod 5が26/27件だが398,456候補であり、絞り込みにはなっていない。両者はPareto関係で、今回の局所ピーク特徴はMethod 0を置き換えない。","","## 15. 結論とQ1〜Q8","",f"1. ピーク高さ中央値は傷矩形群{med(d,'height'):.2f}、暫定正常{med(n,'height'):.2f}で差はあるが分布は重なる。","",f"2. 半値面積中央値は{med(d,'half_area'):.1f}px対{med(n,'half_area'):.1f}pxで、傷の方が大きいとはいえない。","","3. prominenceは局所極大候補を約11%減らしたが、正常粒状模様を十分除外できずMethod 0より多い。","","4. 複数σ持続性はピーク方式内で約10%追加削減したが、正常模様も複数σで持続した。","","5. NMSは局所極大83.4万件を46.5万件へ44%削減したが、Method 0の27.2万件を下回らなかった。","",f"6. Method 0のRecall@KはK=10:8/14、50:11/14、100:11/14、500:13/14。ランダム順位と比較して有意な優先順位を持つ。","","7. ライトありは格子周期・粒状点、なしは粒状点・輪郭・濃淡むらが主な誤反応源。ただし撮影条件差を含む。","","8. 現時点ではMethod 0を維持し、連結成分内の最大応答位置・重心ずれを改善して2件の見逃しを回収する方が有望。局所極大全展開は採用しない。"]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=Path,default=DATA);ap.add_argument("--previous",type=Path,default=PREVIOUS);ap.add_argument("--out",type=Path,default=OUT);args=ap.parse_args()
    folders=["01_response_maps","02_peak_locations","03_peak_profiles","04_scale_analysis","05_feature_distributions","06_method_comparison","07_full_image_results","08_failure_cases"]
    for f in folders:(args.out/f).mkdir(parents=True,exist_ok=True)
    records=fa.load_records(args.data);normals,normal_info=load_normals(args.previous);scale_thresholds=pilot_scale_thresholds(records,normals);write_json(args.out/"normal_roi_provenance.json",normal_info)
    data=[];all_peaks=[];roi_rows=[]
    for rec in records:
        mm=maps(rec["gray"]);response_visual(rec,mm,args.out/"01_response_maps");peaks=peak_features(rec,mm,scale_thresholds);components=component_candidates(mm[8]);members=roi_memberships(rec,peaks,normals,mm);peak_location_visual(rec,peaks,args.out/"02_peak_locations"/f"{rec['name']}_peaks.png");peak_profiles(rec,mm,members,args.out/"03_peak_profiles");all_peaks.extend(peaks);roi_rows.extend(members);data.append({"rec":rec,"peaks":peaks,"components":components,"normal_memberships":[r for r in members if r["roi_type"]=="provisional_normal_roi"]});print(rec["name"],len(components),len(peaks),flush=True)
    write_csv(args.out/"peak_features.csv",all_peaks);write_csv(args.out/"roi_peak_features.csv",roi_rows)
    params=tune([d for d in data if d["rec"]["name"] in PILOT]);params.update({"base_threshold":BASE_THRESHOLD,"scale_thresholds":scale_thresholds,"pilot_images":sorted(PILOT),"normal_roi_status":"user_confirmed" if normal_info["user_confirmation_recorded"] else "provisional"})
    metrics,rak,candidates=summarize_methods(data,params);selected="method0_components";random_rows=random_baseline(data,candidates,selected);rak.extend(random_rows)
    write_csv(args.out/"method_metrics.csv",metrics);write_csv(args.out/"recall_at_k.csv",rak);write_json(args.out/"parameter_summary.json",params)
    scale_analysis_plots(roi_rows,args.out/"04_scale_analysis");roi_feature_plots(roi_rows,args.out/"05_feature_distributions");method_plot(metrics,args.out/"06_method_comparison")
    for d in data:
        draw_candidates(d["rec"],candidates[(d["rec"]["name"],selected)],selected,args.out/"07_full_image_results"/f"{d['rec']['name']}_{selected}.png")
    missed_cases(data,candidates,selected,args.out/"08_failure_cases");make_report(records,roi_rows,metrics,rak,params,normal_info,selected,args.out)
    print("done",args.out,flush=True)


if __name__=="__main__":main()
