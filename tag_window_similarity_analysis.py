"""Tag-impact and FFT-window audit for 4-neighbour spectral similarity."""
from __future__ import annotations
import csv, json, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import evaluate_all_radial_fft as prev
import multiscale_fft_experiment as base

ROOT=Path(__file__).resolve().parent; DATA=base.DATA; OUT=ROOT/'output_tag_window_similarity'
WINS=(128,192,256,320,384); MODES={'fixed128':128,'half_window':None}; LABEL=prev.CLASSES[0]

def integral(mask): return np.pad(mask.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
def overlap(ii,x,y,n):
    x0=np.clip(x-n//2,0,ii.shape[1]-1); x1=np.clip(x+n//2,0,ii.shape[1]-1); y0=np.clip(y-n//2,0,ii.shape[0]-1); y1=np.clip(y+n//2,0,ii.shape[0]-1)
    return (ii[y1,x1]-ii[y0,x1]-ii[y1,x0]+ii[y0,x0])/(n*n)
def load(i,n):
    p=ROOT/'output_window_size_vs_defect_size'/f'image{i}'/f'window_{n}.npz'
    if not p.exists(): p=ROOT/'output_large_window_radial'/f'image{i}'/f'window_{n}.npz'
    if not p.exists(): return None
    with np.load(p) as z:
        return {k:z[k].copy() for k in z.files}
def category(v,t):
    d=np.array([np.sqrt(np.mean((v[a]-v[b])**2)) for a,b in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))])
    pairs=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
    if np.all(d<=t): return '4_similar',None,d
    for out in range(4):
        keep=[j for j in range(4) if j!=out]; q=[d[k] for k,p in enumerate(pairs) if p[0] in keep and p[1] in keep]; r=[d[k] for k,p in enumerate(pairs) if out in p]
        if max(q)<=t and min(r)>t: return '3_plus_1',out,d
    for a,b in ((0,1),(0,2),(0,3)):
        rem=[j for j in range(4) if j not in (a,b)]; within=[]; cross=[]
        for x,p in enumerate(pairs):
            (within if (p==(a,b) or p==(rem[0],rem[1])) else cross).append(d[x])
        if max(within)<=t and min(cross)>t: return '2_plus_2',None,d
    return 'other',None,d
def zone(x,y,w,h,n):
    edge=min(x,y,w-x,h-y); return 'corner' if x<n and y<n or x<n and y>h-n or x>w-n and y<n or x>w-n and y>h-n else ('edge' if edge<n else 'center')
def main():
    OUT.mkdir(exist_ok=True); rows=[]; conditions=[]; timings=[]
    # First pass: collect all pair distances so each image/window/spacing uses an auditable robust threshold.
    cache={}
    for n in WINS:
      for mode,fixed in MODES.items():
       spacing=fixed or n//2
       for i in range(2,18):
        r=load(i,n)
        if r is None: continue
        step=spacing//16; sp=r['radial_spectra']; ny,nx=sp.shape[:2]
        ds=[]; points=[]
        for y in range(step,ny-step):
         for x in range(step,nx-step):
          v=np.stack((sp[y-step,x],sp[y+step,x],sp[y,x-step],sp[y,x+step])); d=[np.sqrt(np.mean((v[a]-v[b])**2)) for a,b in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))]; ds+=d; points.append((y,x,v))
        if not ds: continue
        q1,q3=np.percentile(ds,[25,75]); t=float(np.median(ds)+.5*(q3-q1)); cache[(i,n,mode)]=(r,points,t); conditions.append(dict(image=i,window=n,mode=mode,spacing=spacing,threshold=t,all_four_reference_points=len(points),grid_points=ny*nx,all_four_fraction=len(points)/(ny*nx),frequency_resolution=1/n,valid_frequency_bins=sp.shape[-1]))
    for (i,n,mode),(r,points,t) in cache.items():
        started=time.perf_counter(); gray=np.asarray(Image.open(DATA/f'image{i}.png').convert('L')); labels=prev.annotation(DATA/f'image{i}.json',gray.shape); masks={'tag':np.zeros(gray.shape,bool),'background':np.zeros(gray.shape,bool)}
        for name,m in labels: 
            if name in masks: masks[name]|=m
        ii={k:integral(v) for k,v in masks.items()}; xs=r['x_centers']; ys=r['y_centers']; h,w=gray.shape
        for y,x,v in points:
            coords=((int(xs[x]),int(ys[y-n//16 if False else y])),) # retained only for clarity
            cx,cy=int(xs[x]),int(ys[y]); step=(128 if mode=='fixed128' else n//2)//16
            refs=[(int(xs[x]),int(ys[y-step])),(int(xs[x]),int(ys[y+step])),(int(xs[x-step]),int(ys[y])),(int(xs[x+step]),int(ys[y]))]
            tags=[overlap(ii['tag'],a,b,n)>0 for a,b in refs]; target_tag=overlap(ii['tag'],cx,cy,n)>0; bg=overlap(ii['background'],cx,cy,n)>0
            kind,out,d=category(v,t); rows.append(dict(image=i,window=n,mode=mode,spacing=128 if mode=='fixed128' else n//2,x=cx,y=cy,category=kind,outlier_direction=out,threshold=t,mean_pair_distance=float(np.mean(d)),target_tag=target_tag,reference_tag_count=sum(tags),outlier_is_tag=(tags[out] if out is not None else None),target_background=bg,zone=zone(cx,cy,w,h,n)))
        timings.append(dict(image=i,window=n,mode=mode,seconds=time.perf_counter()-started))
    prev.safe_csv(OUT/'point_classification.csv',rows); prev.safe_csv(OUT/'condition_metrics.csv',conditions); prev.safe_csv(OUT/'processing_times.csv',timings)
    groups=[]
    for n in WINS:
     for mode in MODES:
      for tagbin in ('ref_0','ref_1','ref_2plus','target_tag','target_no_tag'):
       r=[z for z in rows if z['window']==n and z['mode']==mode and ((tagbin=='ref_0' and z['reference_tag_count']==0) or (tagbin=='ref_1' and z['reference_tag_count']==1) or (tagbin=='ref_2plus' and z['reference_tag_count']>=2) or (tagbin=='target_tag' and z['target_tag']) or (tagbin=='target_no_tag' and not z['target_tag']))]
       if r:
        groups.append(dict(window=n,mode=mode,tag_group=tagbin,n=len(r),**{k:sum(z['category']==k for z in r)/len(r) for k in ('4_similar','3_plus_1','2_plus_2','other')},three_plus_one_outlier_tag_rate=(sum(z['outlier_is_tag'] is True for z in r if z['category']=='3_plus_1')/max(1,sum(z['category']=='3_plus_1' for z in r)))))
    prev.safe_csv(OUT/'tag_group_rates.csv',groups)
    # Tag-free geographic/background audit for the required 256 px / spacing 128 baseline.
    audit=[]
    for z in rows:
     if z['window']==256 and z['mode']=='fixed128' and not z['target_tag'] and z['reference_tag_count']==0:
      audit.append(z)
    summary=[]
    for field in ('zone','target_background'):
     for value in sorted(set(str(z[field]) for z in audit)):
      q=[z for z in audit if str(z[field])==value]; summary.append(dict(scope='256_tag_free',field=field,value=value,n=len(q),**{k:sum(z['category']==k for z in q)/len(q) for k in ('4_similar','3_plus_1','2_plus_2','other')}))
    prev.safe_csv(OUT/'tag_free_geometry_background_rates.csv',summary)
    # Compact figures: tag impact at baseline, and window/mode class composition.
    baseg=[z for z in groups if z['window']==256 and z['mode']=='fixed128' and z['tag_group'] in ('ref_0','ref_1','ref_2plus','target_tag')]
    cats=('4_similar','3_plus_1','2_plus_2','other'); fig,ax=plt.subplots(figsize=(9,4)); bottom=np.zeros(len(baseg))
    for c in cats: vals=np.array([z[c] for z in baseg]); ax.bar([z['tag_group'] for z in baseg],vals,bottom=bottom,label=c); bottom+=vals
    ax.set(ylabel='share of locations',title='256 px / 128 px spacing: class share by tag exposure',ylim=(0,1)); ax.legend(ncol=4,fontsize=8); fig.savefig(OUT/'tag_impact_256.png',dpi=160,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4));
    for mode in MODES:
     q=[z for z in groups if z['mode']==mode and z['tag_group']=='ref_0']; ax.plot([z['window'] for z in q],[z['4_similar'] for z in q],marker='o',label=f'{mode}: tag-free refs')
    ax.set(xlabel='FFT window (px)',ylabel='4-similar share',xticks=WINS,title='Tag-free reference locations'); ax.legend(); ax.grid(alpha=.25); fig.savefig(OUT/'window_comparison_tag_free.png',dpi=160,bbox_inches='tight'); plt.close(fig)
    report=['# Tag and FFT-window similarity audit','', 'Pairwise distance = radial-spectrum RMS difference (dB). Four-neighbour classes use a per-image/window/spacing robust threshold: median pair distance + 0.5 × IQR. This is a diagnostic rule, not a trained classifier.','', 'The 256 px / 128 px condition is the primary tag audit. Tag and background labels are used only to explain observed classes, never to select references.','', f'Rows: {len(rows):,}; 256 px tag-free rows: {len(audit):,}.', '', 'See tag_group_rates.csv for the requested tag strata and 3+1 singleton-tag rate; tag_free_geometry_background_rates.csv separates image geometry from annotated background.']
    (OUT/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(OUT)
if __name__=='__main__': main()
