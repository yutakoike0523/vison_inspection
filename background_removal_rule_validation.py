"""Image-wise validation of concentration/entropy background-removal rules."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from background_surface_hist_experiment import DEFAULT_INPUT, image_number, write_csv
from occupancy_histogram_relation_analysis import block_features, save

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs"/"background_removal_rule_validation"
RULES=("concentration","entropy","and","or")

def true_labels(rows):
    return np.array(["background" if r["background_tag_union_percent"]>=90 and r["damage_percent"]==0 else "surface" if r["surface_percent"]>=90 and r["damage_percent"]==0 else "mixed" for r in rows],object)

def predict(c,e,tc,te,rule):
    if rule=="concentration": return c>tc
    if rule=="entropy": return e<te
    if rule=="and": return (c>tc)&(e<te)
    return (c>tc)|(e<te)

def metric(pred,truth,damage):
    bg=truth=="background"; surf=truth=="surface"
    tp=(pred&bg).sum(); fp=(pred&~bg).sum(); fn=(~pred&bg).sum(); tn=(~pred&~bg).sum()
    return {"background_precision":tp/max(1,tp+fp),"background_recall":tp/max(1,tp+fn),"background_f1":2*tp/max(1,2*tp+fp+fn),"surface_false_removal_rate":(pred&surf).sum()/max(1,surf.sum()),"damage_false_removal_rate":(pred&damage).sum()/max(1,damage.sum()),"tp":int(tp),"tn":int(tn),"fp":int(fp),"fn":int(fn)}

def fit(train,rule):
    c=np.array([r["peak_concentration"] for r in train]); e=np.array([r["entropy"] for r in train]); truth=true_labels(train); damage=np.array([r["damage_percent"]>0 for r in train])
    cs=np.unique(np.quantile(c,np.linspace(0,1,31))); es=np.unique(np.quantile(e,np.linspace(0,1,31)))
    candidates=[]
    for tc in (cs if rule in ("concentration","and","or") else [np.nan]):
        for te in (es if rule in ("entropy","and","or") else [np.nan]):
            p=predict(c,e,tc,te,rule); m=metric(p,truth,damage)
            # Safety constraints predeclared before looking at held-out images.
            if m["surface_false_removal_rate"]<=.01 and m["damage_false_removal_rate"]<=.01:
                candidates.append((m["background_recall"],m["background_precision"],tc,te,m))
    if not candidates: return None
    _,_,tc,te,_=max(candidates,key=lambda x:(x[0],x[1])); return float(tc),float(te)

def folds(rows,lighting_specific):
    images=sorted({r["image"] for r in rows},key=lambda x:int(x[5:]))
    for test_image in images:
        test=[r for r in rows if r["image"]==test_image]
        train=[r for r in rows if r["image"]!=test_image and (not lighting_specific or r["lighting"]==test[0]["lighting"])]
        yield test_image,train,test

def validate(rows,lighting_specific):
    results=[]; predictions={}
    for image,train,test in folds(rows,lighting_specific):
        for rule in RULES:
            fit_result=fit(train,rule)
            if fit_result is None: continue
            tc,te=fit_result; p=predict(np.array([r["peak_concentration"] for r in test]),np.array([r["entropy"] for r in test]),tc,te,rule); m=metric(p,true_labels(test),np.array([r["damage_percent"]>0 for r in test]))
            results.append({"scope":"lighting_specific" if lighting_specific else "common","test_image":image,"lighting":test[0]["lighting"],"rule":rule,"Tc":tc,"Te":te,**m})
            predictions[("lighting_specific" if lighting_specific else "common",rule,image)]=p
    return results,predictions

def summary(results):
    out=[]
    for scope in sorted({r["scope"] for r in results}):
        for rule in RULES:
            part=[r for r in results if r["scope"]==scope and r["rule"]==rule]
            if part: out.append({"scope":scope,"rule":rule,"folds":len(part),**{k:float(np.mean([r[k] for r in part])) for k in ("background_precision","background_recall","background_f1","surface_false_removal_rate","damage_false_removal_rate")}})
    return out

def overlays(rows,pred,out):
    gray=plt.imread(str(DEFAULT_INPUT/(rows[0]["image"]+".png")))
    fig,ax=plt.subplots(figsize=(9,6),constrained_layout=True); ax.imshow(gray)
    for r,value in zip(rows,pred):
        color="#1976d2" if value else "#2e7d32"; ax.add_patch(plt.Rectangle((r["x0"],r["y0"]),r["x1"]-r["x0"],r["y1"]-r["y0"],fill=False,lw=2,edgecolor=color))
        if value and r["damage_percent"]>0: ax.add_patch(plt.Rectangle((r["x0"]+3,r["y0"]+3),r["x1"]-r["x0"]-6,r["y1"]-r["y0"]-6,fill=False,lw=1.5,edgecolor="red"))
    ax.set(title="Blue=predicted background; green=kept; red inset=damage falsely removed"); ax.axis("off"); save(fig,out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",type=Path,default=OUT); args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True); per=args.output_dir/"per_image"; per.mkdir(exist_ok=True)
    rows=[]
    for path in sorted(DEFAULT_INPUT.glob("image[0-9]*.png"),key=image_number): rows.extend(block_features(path)[0])
    write_csv(args.output_dir/"block_occupancy_features.csv",rows)
    # Feature comparisons for pure endpoints.
    labels=true_labels(rows); fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    features=("peak_concentration","entropy","iqr","std","mean")
    for ax,feature in zip(axes.ravel(),features):
        vals=[[r[feature] for r,l in zip(rows,labels) if l==kind] for kind in ("surface","background")]; ax.boxplot(vals,tick_labels=["surface >=90%","background+tag >=90%"]); ax.set(title=feature,ylabel=feature)
    axes.ravel()[-1].axis("off"); save(fig,args.output_dir/"feature_distributions.png")
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for ax,x,y in zip(axes,("peak_concentration","iqr"),("entropy","peak_concentration")):
        for kind,color in (("surface","#2e7d32"),("background","#1976d2")):
            q=[r for r,l in zip(rows,labels) if l==kind]; ax.scatter([r[x] for r in q],[r[y] for r in q],s=18,c=color,label=kind)
        ax.set(xlabel=x,ylabel=y,title=f"{x} x {y}"); ax.legend()
    save(fig,args.output_dir/"peak_entropy_scatter.png")
    common,cp=validate(rows,False); lighting,lp=validate(rows,True); results=common+lighting; write_csv(args.output_dir/"simple_rule_results.csv",results); summ=summary(results); write_csv(args.output_dir/"rule_summary.csv",summ)
    # Predeclared safe choice: among common rules, highest mean recall then precision.
    safe=[r for r in summ if r["scope"]=="common" and r["surface_false_removal_rate"]<=.01 and r["damage_false_removal_rate"]<=.01]
    chosen=max(safe,key=lambda r:(r["background_recall"],r["background_precision"])) if safe else max([r for r in summ if r["scope"]=="common"],key=lambda r:r["background_f1"])
    false=[]; truth=[]; pred_all=[]
    for image in sorted({r["image"] for r in rows},key=lambda x:int(x[5:])):
        image_rows=[r for r in rows if r["image"]==image]; p=cp[("common",chosen["rule"],image)]; (per/image).mkdir(exist_ok=True); overlays(image_rows,p,per/image/"classification_overlay.png")
        for r,v,l in zip(image_rows,p,true_labels(image_rows)):
            truth.append(l);pred_all.append(v)
            if v and (l=="surface" or r["damage_percent"]>0): false.append({**r,"predicted_background":True,"true_endpoint_label":l,"false_removal_reason":"damage" if r["damage_percent"]>0 else "surface"})
    write_csv(args.output_dir/"false_removal_blocks.csv",false)
    # Aggregate confusion matrix only for pure endpoint labels.
    valid=np.array([x in ("background","surface") for x in truth]); y=np.array(truth)[valid]; p=np.array(pred_all)[valid]; mat=np.array([[(~p & (y=="surface")).sum(),(p & (y=="surface")).sum()],[(~p & (y=="background")).sum(),(p & (y=="background")).sum()]])
    fig,ax=plt.subplots(figsize=(5,4),constrained_layout=True); im=ax.imshow(mat,cmap="Blues");
    for i in range(2):
        for j in range(2): ax.text(j,i,str(mat[i,j]),ha="center",va="center")
    ax.set(xticks=(0,1),xticklabels=("kept","predicted bg"),yticks=(0,1),yticklabels=("surface","background"),title=f"LOO common rule: {chosen['rule']}"); fig.colorbar(im,ax=ax); save(fig,args.output_dir/"confusion_matrix.png")
    lines=["# Background-removal rule validation","",f"Chosen common rule: `{chosen}`.","",f"False removals: {len(false)}; see false_removal_blocks.csv.","", "Rules use leave-one-image-out threshold fitting; thresholds are never adjusted on the test image. Safety constraints during fitting: surface and damage false-removal rates <=1%.","", "Lighting-specific and common-rule fold results are in simple_rule_results.csv."]
    (args.output_dir/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"completed {len(rows)} blocks; chosen={chosen['rule']}")
if __name__=="__main__":main()
