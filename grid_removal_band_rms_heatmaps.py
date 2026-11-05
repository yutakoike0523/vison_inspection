#!/usr/bin/env python3
"""Whole-image 0.02--0.05 cyc/px inverse-FFT RMS maps, before/after grid notch."""
from __future__ import annotations
import csv, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import analysis_fft_normalization_128 as det
import grid_removal_bandpass_window_comparison as grid

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'data20260903'/'data20260903'; OUT=ROOT/'analysis_grid_removal_band_rms_heatmaps_002_005'
N=128; STRIDE=16; H=N//2; EPS=1e-20
F=np.fft.fftshift(np.fft.fftfreq(N)); FY,FX=np.meshgrid(F,F,indexing='ij'); BAND=(np.hypot(FX,FY)>=.02)&(np.hypot(FX,FY)<=.05); HANN=np.outer(np.hanning(N),np.hanning(N))

def write(path,rows,fields=None):
 c=list(dict.fromkeys(k for r in rows for k in r)) if rows else (fields or ['status'])
 with open(path,'w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=c,extrasaction='ignore');w.writeheader();w.writerows(rows)

def classification(anns,x0,y0,shape):
 labs=[a['label'] for a in anns if a['mask'][y0:y0+N,x0:x0+N].any()]
 hasdef=any(t in det.DEFECT_LABELS for t in labs); hasexc=any(t in det.EXCLUSION_LABELS for t in labs)
 if hasdef and hasexc:return 'mixed','|'.join(labs)
 if hasdef:return 'defect','|'.join(labs)
 if hasexc:return 'excluded','|'.join(labs)
 return 'clean',''

def rmsmap(image,anns,condition,n):
 h,w=image.shape; xs=np.arange(0,w-N+1,STRIDE);ys=np.arange(0,h-N+1,STRIDE);arr=np.full((len(ys),len(xs)),np.nan); rows=[]
 # vectorisation would consume substantial memory for 2048x1536. Row batches retain float64 fidelity.
 for iy,y in enumerate(ys):
  blocks=np.moveaxis(np.lib.stride_tricks.sliding_window_view(image[y:y+N],N,axis=1),1,0)[::STRIDE].astype(float,copy=True)
  blocks-=blocks.mean(axis=(1,2),keepdims=True); blocks*=HANN
  z=np.fft.fftshift(np.fft.fft2(blocks,axes=(1,2)),axes=(1,2)); b=np.fft.ifft2(np.fft.ifftshift(np.where(BAND,z,0),axes=(1,2)),axes=(1,2)).real
  rs=np.sqrt(np.mean(b*b,axis=(1,2)));arr[iy]=rs
  for ix,(x,r) in enumerate(zip(xs,rs)):
   cls,labs=classification(anns,int(x),int(y),image.shape)
   rows.append(dict(image=n,condition=condition,window_x=int(x),window_y=int(y),center_x=int(x+H),center_y=int(y+H),rms=float(r),classification=cls,overlap_labels=labs))
 return xs,ys,arr,rows

def draw_annotations(ax,anns,scale=1):
 for a in anns:
  m=a['mask']; y,x=np.nonzero(m)
  if not len(x):continue
  col='red' if a['label'] in det.DEFECT_LABELS else ('orange' if a['label'] in det.EXCLUSION_LABELS and a['label'] in ('tag','タグ') else 'deepskyblue')
  # contour preserves polygons without inventing an interpolated heatmap.
  ax.contour(m.astype(float),levels=[.5],colors=[col],linewidths=1.1)

def map_image(ax, arr,xs,ys,title,cmap,vmin,vmax,anns=None):
 extent=(xs[0]+H,xs[-1]+H,ys[-1]+H,ys[0]+H)
 im=ax.imshow(arr,origin='upper',extent=extent,interpolation='nearest',aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax)
 if anns:draw_annotations(ax,anns)
 ax.set(title=title,xlabel='x (px)',ylabel='y (px)'); return im

def summaries(rows,n):
 out=[]
 for cond in ('none','base'):
  q=[r for r in rows if r['image']==n and r['condition']==cond]
  d={k:np.array([r['rms'] for r in q if r['classification']==k],float) for k in ('clean','defect')}
  a=np.array([r['rms'] for r in q],float)
  row=dict(image=n,condition=cond,all_mean=float(a.mean()),all_median=float(np.median(a)))
  for k,v in d.items():
   row[f'{k}_count']=len(v);row[f'{k}_mean']=float(v.mean()) if len(v) else math.nan;row[f'{k}_median']=float(np.median(v)) if len(v) else math.nan
   if k=='clean':row['clean_p95']=float(np.percentile(v,95)) if len(v) else math.nan;row['clean_p99']=float(np.percentile(v,99)) if len(v) else math.nan
  row['defect_minus_clean']=row['defect_mean']-row['clean_mean'];row['defect_to_clean']=row['defect_mean']/(row['clean_mean']+EPS)
  out.append(row)
 return out

def main():
 plt.rcParams.update({'font.family':['Yu Gothic','Meiryo','DejaVu Sans']});OUT.mkdir(exist_ok=True)
 allrows=[]; sums=[]; peakrows=[]; vals=[]; cached={}
 for n in (15,18):
  original=grid.load(n); anns=det.load_annotations(DATA/f'image{n}.json',original.shape); obs,_,_=grid.peaks(original);base,angle=grid.select_base(obs)
  if len(base)!=2:raise RuntimeError(f'image{n}: basic pair not found')
  filtered,removed,mask,z=grid.filter_image(original,base); recon=np.max(np.abs(filtered+removed-original))
  idir=OUT/f'image{n}';idir.mkdir(exist_ok=True)
  Image.fromarray(original.astype(np.uint8)).save(idir/'original.png');Image.fromarray(np.clip(filtered,0,255).astype(np.uint8)).save(idir/'grid_removed_base.png');np.save(idir/'grid_removed_base.npy',filtered);np.save(idir/'removed_base.npy',removed)
  peakrows += [dict(image=n,peak_id=p['peak_id'],fx=p['fx'],fy=p['fy'],period_px=p['period_px'],angle_deg=p['angle_deg'],prominence_db=p['prominence_db'],angle_separation_deg=angle,notch_radius_bins=2,removed_fft_bins=int((mask==0).sum()),reconstruction_max_abs_error=recon) for p in base]
  x0,y0,a,ra=rmsmap(original,anns,'none',n);x1,y1,b,rb=rmsmap(filtered,anns,'base',n);assert np.array_equal(x0,x1) and np.array_equal(y0,y1)
  allrows+=ra+rb;s=summaries(ra+rb,n);sums+=s;vals+=list(a.ravel())+list(b.ravel());cached[n]=(original,filtered,anns,x0,y0,a,b,removed)
  np.save(idir/'band_rms_none.npy',a);np.save(idir/'band_rms_base.npy',b);np.save(idir/'band_rms_difference.npy',b-a)
 # same RMS range across both images, and shared within each comparison.
 lo,hi=np.percentile(vals,[1,99]);diffmax=max(np.percentile(np.abs(v[6]-v[5]),99) for v in cached.values())
 for n,(original,filtered,anns,xs,ys,a,b,removed) in cached.items():
  idir=OUT/f'image{n}';d=b-a; lim=np.percentile(original,[1,99])
  for name,arr,cmap,v0,v1 in [('band_rms_none.png',a,'viridis',lo,hi),('band_rms_base.png',b,'viridis',lo,hi),('band_rms_difference.png',d,'coolwarm',-diffmax,diffmax)]:
   fig,ax=plt.subplots(figsize=(9,6),constrained_layout=True);im=map_image(ax,arr,xs,ys,name.replace('_',' '),cmap,v0,v1,anns);fig.colorbar(im,ax=ax,label='band inverse-FFT RMS');fig.savefig(idir/name,dpi=180);plt.close(fig)
  fig,ax=plt.subplots(1,4,figsize=(20,5),constrained_layout=True)
  ax[0].imshow(original,cmap='gray',vmin=lim[0],vmax=lim[1]);draw_annotations(ax[0],anns);ax[0].set_title('Original');ax[0].axis('off')
  ims=[map_image(ax[1],a,xs,ys,'RMS: none','viridis',lo,hi,anns),map_image(ax[2],b,xs,ys,'RMS: base','viridis',lo,hi,anns),map_image(ax[3],d,xs,ys,'base − none','coolwarm',-diffmax,diffmax,anns)]
  fig.colorbar(ims[0],ax=ax[1:3],label='RMS');fig.colorbar(ims[2],ax=ax[3],label='RMS difference');fig.savefig(idir/'heatmap_comparison.png',dpi=180);plt.close(fig)
  for name,arr,title in [('heatmap_overlay_none.png',a,'none'),('heatmap_overlay_base.png',b,'base')]:
   fig,ax=plt.subplots(figsize=(9,6),constrained_layout=True);im=ax.imshow(original,cmap='gray',vmin=lim[0],vmax=lim[1]);over=map_image(ax,arr,xs,ys,f'RMS overlay: {title}','viridis',lo,hi,anns);fig.colorbar(over,ax=ax,label='RMS');fig.savefig(idir/name,dpi=180);plt.close(fig)
  # one overview crop enclosing every defect, or a 256px crop around the first.
  dm=np.logical_or.reduce([q['mask'] for q in anns if q['label'] in det.DEFECT_LABELS]); yy,xx=np.nonzero(dm);xa=max(0,xx.min()-96);xb=min(original.shape[1],xx.max()+97);ya=max(0,yy.min()-96);yb=min(original.shape[0],yy.max()+97)
  fig,ax=plt.subplots(1,4,figsize=(16,4),constrained_layout=True)
  ax[0].imshow(original[ya:yb,xa:xb],cmap='gray',vmin=lim[0],vmax=lim[1]);ax[0].set_title('Original defect crop')
  for aa,arr,t,cm,v0,v1 in zip(ax[1:],(a,b,d),('none','base','base − none'),('viridis','viridis','coolwarm'),(lo,lo,-diffmax),(hi,hi,diffmax)):
   im=map_image(aa,arr,xs,ys,t,cm,v0,v1,anns);aa.set(xlim=(xa,xb),ylim=(yb,ya))
  for aa in ax:aa.axis('off')
  fig.savefig(idir/'defect_zoom_comparison.png',dpi=200);plt.close(fig)
 write(OUT/'window_band_rms_all.csv',allrows);write(OUT/'per_image_rms_summary.csv',sums);write(OUT/'defect_vs_clean_summary.csv',sums);write(OUT/'grid_removal_peak_summary.csv',peakrows)
 lines=['# 0.02–0.05 cycles/pixel 帯域RMSヒートマップ','', '局所FFTの周囲参照差スコアではなく、各128×128px窓の帯域限定逆FFT画像のRMSを16px格子へ割り当てた結果である。','']
 for n in (15,18):
  q=[r for r in sums if r['image']==n];a,b=q
  lines += [f'## image{n}',f"clean RMS平均: {a['clean_mean']:.4g} → {b['clean_mean']:.4g}; 傷RMS平均: {a['defect_mean']:.4g} → {b['defect_mean']:.4g}。",f"傷/clean比: {a['defect_to_clean']:.4g} → {b['defect_to_clean']:.4g}。"]
 lines += ['','image15では基本波が帯域より低く、RMSマップの変化は限定的である。image18では基本波が解析帯域内なので、正常と傷の双方が低下し得る。比の変化は局所的な分離の補助所見であり、検出性能の証明ではない。']
 (OUT/'report_ja.md').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__':main()
