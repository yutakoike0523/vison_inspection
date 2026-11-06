#!/usr/bin/env python3
"""Render one image from cached scores of the reference-distance experiment."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import reference_distance_aggregation_experiment as exp
import grid_period_window_comparison as prior


def main():
    p=argparse.ArgumentParser();p.add_argument("--image",type=int,required=True);p.add_argument("--data",type=Path,default=exp.DATA);p.add_argument("--output",type=Path,default=exp.OUT);p.add_argument("--selected-mode",default="period_3x");p.add_argument("--selected-distance",default="1.0W");p.add_argument("--spectral-only",action="store_true");args=p.parse_args()
    i=args.image;image=np.asarray(Image.open(args.data/f"image{i}.png").convert("L"),np.float32);ann=prior.load_annotation(args.data,i,image.shape)
    lookup={r["window_mode"]:r for r in prior.settings_for(i)};cached=np.load(args.output/"score_maps"/f"image{i}.npz");records={}
    for key in cached.files:
        experiment,mode,dname,aggregation=key.split("__");window=128 if mode=="fixed128" else int(lookup[mode]["actual_window_px"])
        if experiment in ("A","C"):distance=int(dname[1:])
        else:
            factor=dict(exp.B_FACTORS)[dname];distance=exp.distance_for(window,dname,factor)
        xs=np.arange(0,image.shape[1]-window+1,exp.STRIDE,dtype=np.int32)+window//2
        ys=np.arange(0,image.shape[0]-window+1,exp.STRIDE,dtype=np.int32)+window//2
        score=cached[key];r={"condition_id":key,"experiment":experiment,"window_mode":mode,"window":window,"distance_name":dname,"distance":distance,"aggregation":aggregation,"score":score,"valid":np.isfinite(score),"xs":xs,"ys":ys,"valid_bins":0};exp.attach_masks(r,ann);records[key]=r
    akeys=[exp.condition_id("A","fixed128",f"d{d}","mean") for d in exp.A_DISTANCES]
    bkeys=[exp.condition_id("B",m,n,"mean") for m in exp.W_MODES for n,_ in exp.B_FACTORS]
    cfixed=[]
    for d in (48,96,128):cfixed += [exp.condition_id("A","fixed128",f"d{d}","mean"),exp.condition_id("C","fixed128",f"d{d}","median")]
    selected=[exp.condition_id("B",args.selected_mode,args.selected_distance,"mean"),exp.condition_id("B",args.selected_mode,args.selected_distance,"median")]
    if args.spectral_only:
        if i in exp.REPRESENTATIVE_IMAGES:exp.detailed_figures(i,image,ann,records,(args.selected_mode,args.selected_distance),args.output)
        print(f"rendered spectral image{i}");return
    exp.heatmap_grid(i,image,ann,records,akeys,[f"W=128 d={d} mean" for d in exp.A_DISTANCES],args.output/"heatmaps_distance"/f"image{i}_shared.png",2,3)
    exp.heatmap_grid(i,image,ann,records,akeys,[f"W=128 d={d} mean" for d in exp.A_DISTANCES],args.output/"heatmaps_distance"/f"image{i}_normalized.png",2,3,True)
    titles=[f"{m} {n}\nW={records[exp.condition_id('B',m,n,'mean')]['window']} d={records[exp.condition_id('B',m,n,'mean')]['distance']}" for m in exp.W_MODES for n,_ in exp.B_FACTORS]
    exp.heatmap_grid(i,image,ann,records,bkeys,titles,args.output/"heatmaps_window_distance"/f"image{i}_shared.png",3,4)
    exp.heatmap_grid(i,image,ann,records,bkeys,titles,args.output/"heatmaps_window_distance"/f"image{i}_normalized.png",3,4,True)
    cplot=cfixed+selected;titles=[f"{records[k]['window_mode']} {records[k]['distance_name']} {records[k]['aggregation']}" for k in cplot]
    exp.heatmap_grid(i,image,ann,records,cplot,titles,args.output/"heatmaps_mean_vs_median"/f"image{i}_shared.png",4,2)
    exp.heatmap_grid(i,image,ann,records,cplot,titles,args.output/"heatmaps_mean_vs_median"/f"image{i}_normalized.png",4,2,True)
    if i in exp.REPRESENTATIVE_IMAGES:exp.detailed_figures(i,image,ann,records,(args.selected_mode,args.selected_distance),args.output)
    print(f"rendered image{i}")


if __name__=="__main__":main()
