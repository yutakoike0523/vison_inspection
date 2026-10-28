#!/usr/bin/env python3
"""Full-image radial FFT stride experiment for image15/image17."""
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
import multiscale_fft_experiment as base
import evaluate_all_radial_fft as prev

ROOT=base.ROOT; OUT=ROOT/'output_stride_grid_overlap'; PRIOR=ROOT/'output_all_images_radial_fft'
IMAGE_IDS=(15,17); STRIDES=(16,8,4); WINDOW=256; SPACING=128


def run_one(i,stride,args):
    dest=args.output/f'image{i}'/f'stride_{stride}'; dest.mkdir(parents=True,exist_ok=True)
    result=dest/'result.npz'; meta=dest/'timing.json'
    if result.exists() and meta.exists(): print(f'image{i} stride {stride}: complete, skip',flush=True); return
    img=np.asarray(Image.open(args.data/f'image{i}.png').convert('L'),dtype=np.float32)
    if stride==16 and (PRIOR/f'image{i}'/'result.npz').exists():
        settings=json.loads((PRIOR/'settings.json').read_text(encoding='utf-8'))
        expected=(settings['window']==WINDOW and settings['stride']==16 and settings['reference_spacing']==SPACING and settings['frequency']==[.01,.5] and settings['bin_width']==.01 and settings['workers']==args.workers and settings['batch_size']==args.batch_size and Path(settings['data']).resolve()==args.data.resolve())
        if expected:
            import shutil
            shutil.copy2(PRIOR/f'image{i}'/'result.npz',result)
            old=json.loads((PRIOR/f'image{i}'/'metrics.json').read_text(encoding='utf-8'))
            meta.write_text(json.dumps({'origin':'previous fixed-condition run','timings':old['timings']},ensure_ascii=False,indent=2),encoding='utf-8')
            print(f'image{i} stride 16: reused',flush=True); return
    prev.N=WINDOW; prev.STRIDE=stride; prev.SPACING=SPACING
    print(f'image{i} stride {stride}: full image FFT',flush=True)
    r=prev.analyze(img,args.workers,args.batch_size,False)
    np.savez_compressed(result,anomaly_map=r['scores'],x_centers=r['xs'],y_centers=r['ys'],reference_counts=r['counts'],radial_spectra=r['spectra'],bin_counts=r['bin_counts'],bin_valid=r['bin_valid'])
    timing={k:r[k] for k in ('fft_seconds','radial_seconds','median_seconds','score_seconds','map_seconds','total_seconds','peak_memory_bytes')}
    meta.write_text(json.dumps({'origin':'new full-image run','timings':timing},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'image{i} stride {stride}: {r["total_seconds"]:.1f}s',flush=True)


def load_result(path):
    with np.load(path) as z:
        return {'xs':z['x_centers'].copy(),'ys':z['y_centers'].copy(),'scores':z['anomaly_map'].copy(),'spectra':z['radial_spectra'].copy(),'bin_counts':z['bin_counts'].copy(),'bin_valid':z['bin_valid'].copy()}


def reference(r,y,x,stride):
    step=SPACING//stride; ny,nx=r['scores'].shape
    neighbors=[r['spectra'][y+dy*step,x+dx*step] for dy in (-1,0,1) for dx in (-1,0,1) if (dy or dx) and 0<=y+dy*step<ny and 0<=x+dx*step<nx]
    if len(neighbors)<3: return None
    return np.median(np.stack(neighbors),axis=0)


def prior_thresholds():
    path=PRIOR/'summary_20261026_172732'/'common_threshold_metrics.csv'
    if not path.exists(): path=PRIOR/'summary'/'common_threshold_metrics.csv'
    with path.open(encoding='utf-8-sig',newline='') as f:
        return {int(r['number']):float(r['threshold']) for r in csv.DictReader(f) if r['selection']=='center' and r['image'] in ('image15','image17') and r['threshold']}


def mask_info(mask):
    yy,xx=np.nonzero(mask); return (float(np.mean(xx)),float(np.mean(yy))), (int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max()))


def per_image(i,args,threshold,summary):
    image=args.data/f'image{i}.png'; img=np.asarray(Image.open(image).convert('L'),dtype=np.float32)
    labels=prev.annotation(image.with_suffix('.json'),img.shape); masks=[m for name,m in labels if name=='局所変形']
    if len(masks)!=1: raise ValueError(f'image{i} expected one deformation annotation, got {len(masks)}')
    mask=masks[0]; (cx,cy),box=mask_info(mask)
    hann=np.hanning(WINDOW).astype(np.float32); hann=np.outer(hann,hann)
    weight=fftconvolve(mask.astype(np.float32),hann[::-1,::-1],mode='valid')
    # Integral gives exact integer pixel occupancy; FFT convolution only computes Hann sums.
    integ=np.pad(mask.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
    data={s:load_result(args.output/f'image{i}'/f'stride_{s}'/'result.npz') for s in STRIDES}
    r16=data[16]; metrics=[]; windows=[]; times=[]; common_checks=[]
    for stride,r in data.items():
        prev.N=WINDOW; prev.STRIDE=stride; normal,entries=prev.classification(labels,r); defect=next(e for e in entries if e['label']=='局所変形')
        common_x=np.isin(r['xs'],r16['xs']); common_y=np.isin(r['ys'],r16['ys']); common=common_y[:,None]&common_x[None,:]
        for domain,domain_mask in [('all',np.ones(r['scores'].shape,bool)),('common16',common)]:
            normal_values=prev.values(r,normal&domain_mask); ns=prev.summary_stats(normal_values)
            for mode in ('center','overlap'):
                selection=defect[mode]&domain_mask&np.isfinite(r['scores']); yy,xx=np.nonzero(selection); values=r['scores'][yy,xx]
                d=prev.defect_stats(values,ns); k=int(np.argmax(values)) if len(values) else None
                metrics.append({'image':i,'stride':stride,'domain':domain,'selection':mode,'count':len(values),'normal_count':ns['count'],'normal_median':ns['median'],'normal_std':ns['std'],'median':d['median'],'maximum':d['max'],'p95':d['p95'],'S':d['S'],'maximum_x':int(r['xs'][xx[k]]) if k is not None else None,'maximum_y':int(r['ys'][yy[k]]) if k is not None else None,'own_normal_p99_exceed_fraction':d['normal_p99_exceed_rate'],'loio_threshold_stride16_training':threshold,'loio_exceed_fraction':float(np.mean(values>threshold)) if len(values) else None,'loio_detected':bool(np.any(values>threshold)) if len(values) else None})
        if stride!=16:
            yi=np.searchsorted(r['ys'],r16['ys']); xi=np.searchsorted(r['xs'],r16['xs'])
            assert np.array_equal(r['ys'][yi],r16['ys']) and np.array_equal(r['xs'][xi],r16['xs'])
            d=r['scores'][np.ix_(yi,xi)]-r16['scores']; finite=np.isfinite(d)
            common_checks.append({'image':i,'stride':stride,'common_count':int(finite.sum()),'max_abs_difference':float(np.max(np.abs(d[finite]))),'matches':bool(np.allclose(r['scores'][np.ix_(yi,xi)],r16['scores'],rtol=1e-5,atol=1e-5,equal_nan=True))})
        yy,xx=np.nonzero(defect['overlap']&np.isfinite(r['scores'])); y0=r['ys'][yy].astype(int)-WINDOW//2; x0=r['xs'][xx].astype(int)-WINDOW//2
        occupancy=integ[y0+WINDOW,x0+WINDOW]-integ[y0,x0+WINDOW]-integ[y0+WINDOW,x0]+integ[y0,x0]
        hann_sum=weight[y0,x0]; distance=np.hypot(r['xs'][xx]-cx,r['ys'][yy]-cy)
        for j in range(len(yy)):
            windows.append({'image':i,'stride':stride,'x':int(r['xs'][xx[j]]),'y':int(r['ys'][yy[j]]),'center_in_defect':bool(defect['center'][yy[j],xx[j]]),'distance_to_defect_centroid_px':float(distance[j]),'overlap_pixels':int(occupancy[j]),'defect_fraction_of_window':float(occupancy[j]/WINDOW**2),'mean_hann_weight_on_defect':float(np.clip(hann_sum[j]/occupancy[j],0,1)) if occupancy[j] else None,'score':float(r['scores'][yy[j],xx[j]])})
        t=json.loads((args.output/f'image{i}'/f'stride_{stride}'/'timing.json').read_text(encoding='utf-8'))
        times.append({'image':i,'stride':stride,'locations':int(r['scores'].size),'origin':t['origin'],**t['timings']})
    return {'img':img,'labels':labels,'mask':mask,'box':box,'centroid':(cx,cy),'data':data,'metrics':metrics,'windows':windows,'times':times,'checks':common_checks}


def extent(r,stride):
    return [r['xs'][0]-stride/2,r['xs'][-1]+stride/2,r['ys'][-1]+stride/2,r['ys'][0]-stride/2]


def visuals(i,v,args):
    folder=args.output/f'image{i}'/'comparison'; folder.mkdir(parents=True,exist_ok=True)
    img=v['img']; h,w=img.shape; mask=v['mask']; left,top,right,bottom=v['box']
    selected=[next(x for x in v['metrics'] if x['stride']==s and x['domain']=='all' and x['selection']=='center') for s in STRIDES]
    values=np.concatenate([v['data'][s]['scores'][np.isfinite(v['data'][s]['scores'])] for s in STRIDES]); limits=(float(np.percentile(values,1)),float(np.percentile(values,99)))
    for normalized in (False,True):
        fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
        for ax,s,met in zip(axes,STRIDES,selected):
            r=v['data'][s]; a=r['scores']; a=(a-met['normal_median'])/met['normal_std'] if normalized else a
            im=ax.imshow(a,extent=extent(r,s),cmap='inferno',interpolation='nearest',vmin=-2 if normalized else limits[0],vmax=6 if normalized else limits[1])
            ax.contour(mask.astype(np.uint8),levels=[.5],colors='cyan',linewidths=1,origin='upper',extent=[0,w,h,0])
            ax.scatter([met['maximum_x']],[met['maximum_y']],c='white',marker='x',s=45)
            ax.set(title=f'stride {s}px',xlim=(0,w),ylim=(h,0),xlabel='x (px)',ylabel='y (px)'); fig.colorbar(im,ax=ax,shrink=.7)
        fig.suptitle('Normal median/std normalization' if normalized else 'Shared raw display scale (1–99 percentile; not threshold)')
        fig.savefig(folder/('stride_comparison_normalized.png' if normalized else 'stride_comparison.png'),dpi=130); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    for ax,s,met in zip(axes,STRIDES,selected):
        r=v['data'][s]; ax.imshow(img,cmap='gray',extent=[0,w,h,0]); ax.imshow(r['scores'],extent=extent(r,s),cmap='inferno',alpha=.55,vmin=limits[0],vmax=limits[1],interpolation='nearest')
        ax.contour(mask.astype(np.uint8),levels=[.5],colors='cyan',linewidths=1,origin='upper',extent=[0,w,h,0]); ax.scatter([met['maximum_x']],[met['maximum_y']],c='white',marker='x')
        ax.set(xlim=(max(0,left-180),min(w,right+180)),ylim=(min(h,bottom+180),max(0,top-180)),title=f'stride {s}')
    fig.savefig(folder/'defect_zoom.png',dpi=130); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,7)); ax.imshow(img,cmap='gray',extent=[0,w,h,0]); ax.contour(mask.astype(np.uint8),levels=[.5],colors='cyan',linewidths=1,origin='upper',extent=[0,w,h,0])
    for color,s,met in zip(['orange','white','magenta'],STRIDES,selected):
        x,y=met['maximum_x'],met['maximum_y']; ax.add_patch(Rectangle((x-128,y-128),256,256,fill=False,edgecolor=color,label=f'stride {s}')); ax.scatter([x],[y],c=color,marker='x')
    ax.set(xlim=(max(0,left-250),min(w,right+250)),ylim=(min(h,bottom+250),max(0,top-250)),title='Maximum center windows'); ax.legend(); fig.savefig(folder/'fft_window_positions.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    met=selected[-1]; x,y=met['maximum_x'],met['maximum_y']
    fig,ax=plt.subplots(figsize=(9,7)); ax.imshow(img,cmap='gray',extent=[0,w,h,0]); ax.contour(mask.astype(np.uint8),levels=[.5],colors='cyan',linewidths=1,origin='upper',extent=[0,w,h,0])
    ax.add_patch(Rectangle((x-128,y-128),256,256,fill=False,edgecolor='yellow',linewidth=2,label='Target'))
    for dy in (-128,0,128):
        for dx in (-128,0,128):
            if (dx or dy) and 128<=x+dx<=w-128 and 128<=y+dy<=h-128:
                ax.add_patch(Rectangle((x+dx-128,y+dy-128),256,256,fill=False,edgecolor='lime',alpha=.75))
    ax.set(xlim=(max(0,x-270),min(w,x+270)),ylim=(min(h,y+270),max(0,y-270)),title='Stride 4 maximum: target (yellow) and reference windows (green)'); ax.legend(); fig.savefig(folder/'neighbor_window_layout.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,met in zip(axes,selected):
        x,y=met['maximum_x'],met['maximum_y']; ax.imshow(img[y-128:y+128,x-128:x+128],cmap='gray',vmin=0,vmax=255); ax.set_title(f"stride {met['stride']} ({x},{y})"); ax.axis('off')
    fig.savefig(folder/'maximum_windows.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for s in STRIDES:
        rows=[x for x in v['windows'] if x['stride']==s]
        axes[0].scatter([x['distance_to_defect_centroid_px'] for x in rows],[x['score'] for x in rows],s=3,alpha=.22,label=f'{s}px')
        axes[1].scatter([x['mean_hann_weight_on_defect'] for x in rows],[x['score'] for x in rows],s=3,alpha=.22,label=f'{s}px')
    axes[0].set(xlabel='Distance to annotated centroid (px)',ylabel='RMS dB'); axes[1].set(xlabel='Mean Hann weight in overlapping defect pixels',ylabel='RMS dB'); axes[0].legend(); fig.savefig(folder/'window_position_vs_score.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for s in STRIDES:
        r=v['data'][s]; prev.N=WINDOW; prev.STRIDE=s; normal,_=prev.classification(v['labels'],r)
        a=prev.values(r,normal); defect=prev.values(r,base.selections(mask,r,WINDOW)[0]);
        if s==4: ax.hist(a,bins=70,density=True,alpha=.4,label='Normal stride 4'); ax.hist(defect,bins=35,density=True,alpha=.45,label='Defect center stride 4')
    ax.set(xlabel='RMS dB',ylabel='Density'); ax.legend(); fig.savefig(folder/'distributions.png',dpi=130,bbox_inches='tight'); plt.close(fig)


def spectra_and_contribution(i,v,args):
    folder=args.output/f'image{i}'/'comparison'; r=v['data'][4]; prev.N=WINDOW; prev.STRIDE=4
    normal,_=prev.classification(v['labels'],r); center=base.selections(v['mask'],r,WINDOW)[0]
    yy,xx=np.nonzero(center & np.isfinite(r['scores'])); j=int(np.argmax(r['scores'][yy,xx])); defect=(int(yy[j]),int(xx[j]))
    yy,xx=np.nonzero(normal & np.isfinite(r['scores'])); vals=r['scores'][yy,xx]; j=int(np.argmin(np.abs(vals-np.median(vals)))); normalpos=(int(yy[j]),int(xx[j]))
    freq=(np.arange(len(r['bin_valid']))+.5)*.01; freq=freq[r['bin_valid']]
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True); fig2,ax2=plt.subplots(figsize=(9,4)); rows=[]; representatives=[]
    for kind,(y,x),color in [('defect',defect,'firebrick'),('normal',normalpos,'steelblue')]:
        target=r['spectra'][y,x]; ref=reference(r,y,x,4); diff=target-ref; square=diff**2; total=float(square.sum()); contribution=square/total if total>0 else np.zeros_like(square)
        axes[0,0].plot(freq,target,color=color,label=f'{kind} target'); axes[0,0].plot(freq,ref,color=color,ls='--',label=f'{kind} ref')
        axes[0,1].plot(freq,diff,color=color,label=kind); axes[1,0].plot(freq,square,color=color,label=kind); axes[1,1].plot(freq,contribution,color=color,label=kind); ax2.plot(freq,contribution,color=color,label=kind)
        representatives.append({'image':i,'kind':kind,'x':int(r['xs'][x]),'y':int(r['ys'][y]),'score':float(r['scores'][y,x]),'peak_frequency':float(freq[np.argmax(contribution)]),'peak_contribution':float(np.max(contribution))})
        for f,t,rf,d,c in zip(freq,target,ref,diff,contribution): rows.append({'image':i,'kind':kind,'stride':4,'x':int(r['xs'][x]),'y':int(r['ys'][y]),'frequency_bin_center':float(f),'target_db':float(t),'reference_db':float(rf),'difference_db':float(d),'difference_squared_db2':float(d*d),'contribution':float(c)})
    for ax,title in zip(axes.flat,['Target and reference','Difference','Squared difference','RMS contribution']): ax.set(title=title,xlabel='r (cycles/pixel)'); ax.legend(fontsize=7)
    fig.savefig(folder/'radial_spectra.png',dpi=130); plt.close(fig); ax2.legend(); ax2.set(xlabel='r (cycles/pixel)',ylabel='Fraction of squared RMS'); fig2.savefig(folder/'frequency_contribution.png',dpi=130,bbox_inches='tight'); plt.close(fig2)
    return rows,representatives


def compare_brightness(images,summary):
    crops={}
    for i,v in images.items():
        img=v['img']; h,w=img.shape; cx,cy=v['centroid']; radius=192
        x0=max(0,int(cx)-radius); x1=min(w,int(cx)+radius); y0=max(0,int(cy)-radius); y1=min(h,int(cy)+radius)
        crops[i]=(img[y0:y1,x0:x1],x0,y0)
    pooled=np.concatenate([c[0].ravel() for c in crops.values()]); lo,hi=np.percentile(pooled,[2,98])
    fig,axes=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
    fig2,ax2=plt.subplots(figsize=(10,4)); stats=[]
    for row,i in enumerate(IMAGE_IDS):
        v=images[i]; img=v['img']; mask=v['mask']; patch,x0,y0=crops[i]; h,w=img.shape; cx,cy=v['centroid']; ymin=max(0,int(round(cy))-2); ymax=min(h,int(round(cy))+3)
        axes[row,0].imshow(patch,cmap='gray',vmin=0,vmax=255); axes[row,0].set_title(f'image{i} original')
        axes[row,1].imshow(patch,cmap='gray',vmin=lo,vmax=hi); axes[row,1].set_title('Common 2–98% contrast')
        axes[row,2].imshow(patch,cmap='gray',vmin=0,vmax=255); axes[row,2].contour(mask[y0:y0+patch.shape[0],x0:x0+patch.shape[1]],levels=[.5],colors='cyan',linewidths=1); axes[row,2].set_title('Annotation')
        xlo=max(0,int(cx)-192); xhi=min(w,int(cx)+192); profile=img[ymin:ymax,xlo:xhi].mean(axis=0)
        ax2.plot(np.arange(xlo,xhi)-cx,profile,label=f'image{i}')
        area=img[mask]; nearby=patch.ravel()
        stats.append({'image':i,'defect_pixels':len(area),'defect_mean_intensity':float(np.mean(area)),'defect_median_intensity':float(np.median(area)),'defect_p10':float(np.percentile(area,10)),'defect_p90':float(np.percentile(area,90)),'local_mean_intensity':float(np.mean(nearby)),'local_median_intensity':float(np.median(nearby)),'profile_y_center':float(cy)})
    for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(summary/'image15_vs_image17.png',dpi=130); plt.close(fig)
    ax2.set(xlabel='Horizontal position relative to annotated centroid (px)',ylabel='Gray level: mean of 5 rows',ylim=(0,255)); ax2.legend(); fig2.savefig(summary/'brightness_profiles.png',dpi=130,bbox_inches='tight'); plt.close(fig2)
    prev.safe_csv(summary/'brightness_metrics.csv',stats)
    return stats


def summarize(args):
    args.output.mkdir(parents=True,exist_ok=True); summary=base.unique_dir(args.output/'summary')
    thresholds=prior_thresholds()
    if set(thresholds)!=set(IMAGE_IDS): raise ValueError('Missing prior stride-16 LOIO thresholds')
    images={i:per_image(i,args,thresholds[i],summary) for i in IMAGE_IDS}
    metrics=sum((v['metrics'] for v in images.values()),[]); windows=sum((v['windows'] for v in images.values()),[]); times=sum((v['times'] for v in images.values()),[]); checks=sum((v['checks'] for v in images.values()),[])
    contrib=[]; reps=[]
    for i,v in images.items():
        visuals(i,v,args); c,r=spectra_and_contribution(i,v,args); contrib.extend(c); reps.extend(r)
    brightness=compare_brightness(images,summary)
    prev.safe_csv(summary/'stride_metrics.csv',metrics); prev.safe_csv(summary/'window_position_metrics.csv',windows); prev.safe_csv(summary/'frequency_contribution.csv',contrib)
    prev.safe_csv(summary/'processing_times.csv',times); prev.safe_csv(summary/'common_coordinate_checks.csv',checks); prev.safe_csv(summary/'representative_spectra.csv',reps)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for i in IMAGE_IDS:
        rows=[r for r in metrics if r['image']==i and r['domain']=='all' and r['selection']=='center']; rows.sort(key=lambda r:r['stride'])
        for ax,key in zip(axes,['maximum','median','S']): ax.plot([r['stride'] for r in rows],[r[key] for r in rows],marker='o',label=f'image{i}')
    for ax,title in zip(axes,['Maximum RMS','Median RMS','S']): ax.set(xlabel='Stride (px)',ylabel=title,xticks=[4,8,16]); ax.legend()
    fig.savefig(summary/'stride_score_comparison.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    lines=['# ストライドと格子模様の局所変形検出', '', '## 解析方法', 'image15・image17の全画像で窓256 px、stride 16・8・4 px、周辺3×3（中央除外、最低3参照）、参照間隔128 px、0.01〜0.50 cycles/pixel、放射ビン幅0.01を使用。画像全体を計算し、ストライド以外の前処理・正規化・放射平均・基準を維持。傷中心と窓重なりを別評価。正常は窓が全アノテーションと重ならない領域。', '全地点と、中心座標がstride16に一致する地点を区別。LOIO閾値は前回の格子ライトあり学習画像のstride16正常窓を画像均等に重み付けした99%点であり、評価画像のラベルで調整していない。stride4・8用に全学習画像を高密度再計算した閾値ではなく、同一の外部参照閾値を全ストライドに適用した比較。画像ごとの学習側は評価画像を除いたLOIO foldであり、fold間に共通の固定値ではない。', 'Hann平均は各FFT窓内で傷マスクと重なる画素の重みの平均。傷占有率は傷の重なり画素数/65536。窓と画素の強い重なりがあり、点は独立ではない。FFT畳み込みによるHann和には浮動小数点誤差がある。', '', '## ストライド別の傷中心評価']
    for i in IMAGE_IDS:
        for s in STRIDES:
            r=next(x for x in metrics if x['image']==i and x['stride']==s and x['domain']=='all' and x['selection']=='center')
            lines.append(f"- image{i} stride{s}: 中央値 {r['median']:.4f}、最大 {r['maximum']:.4f} ({r['maximum_x']}, {r['maximum_y']})、S {r['S']:.3f}、正常99%超 {r['own_normal_p99_exceed_fraction']:.1%}、外部LOIO閾値超 {r['loio_exceed_fraction']:.1%}。")
    lines+=['','## 共通座標の一致']
    for c in checks: lines.append(f"- image{c['image']} stride{c['stride']}: 共通{c['common_count']}点、最大差 {c['max_abs_difference']:.9g} dB、一致 {c['matches']}（rtol=atol=1e-5）。")
    lines.append('stride4では指定した厳密な許容誤差を満たさない地点がある。最大差は上記のとおり小さいが、完全一致として扱わず、差の原因は未特定である。最大値の比較にはこの数値差も含まれる。')
    lines+=['','## 代表スペクトルと輝度']
    for r in reps: lines.append(f"- image{r['image']} {r['kind']} ({r['x']}, {r['y']}): score {r['score']:.3f} dB、寄与最大ビン {r['peak_frequency']:.3f} cycles/pixel、寄与率 {r['peak_contribution']:.1%}。")
    for b in brightness: lines.append(f"- image{b['image']}: ラベル内部輝度中央値 {b['defect_median_intensity']:.1f}、局所拡大領域中央値 {b['local_median_intensity']:.1f}。絶対輝度は照明条件の影響を受ける。")
    lines+=['','## 窓位置と計算資源']
    for i in IMAGE_IDS:
        rows=[r for r in windows if r['image']==i and r['stride']==4 and r['mean_hann_weight_on_defect'] is not None]
        corr=float(np.corrcoef([r['mean_hann_weight_on_defect'] for r in rows],[r['score'] for r in rows])[0,1])
        lines.append(f'- image{i}: 傷と重なるstride4窓のHann平均重みとスコアの相関係数 {corr:.3f}（{len(rows)}窓、重複した従属観測）。窓位置と局所模様も変化しているため因果関係は示さない。')
    for t in times: lines.append(f"- image{t['image']} stride{t['stride']}: 解析 {t['total_seconds']:.2f} 秒、プロセス最大作業セット {t['peak_memory_bytes']/1024**2:.1f} MiB（{t['origin']}）。")
    lines.append('stride16は以前の実行結果を再利用し、stride8・4は2画像を同時に実行したため、処理時間は厳密な性能比較ではない。')
    lines+=['','## ストライド仮説の直接比較']
    for i in IMAGE_IDS:
        a=next(r for r in metrics if r['image']==i and r['stride']==16 and r['domain']=='all' and r['selection']=='center')
        b=next(r for r in metrics if r['image']==i and r['stride']==4 and r['domain']=='all' and r['selection']=='center')
        new=(b['maximum_x']-128)%16!=0 or (b['maximum_y']-128)%16!=0
        lines.append(f"- image{i}: stride16→4で最大値 {a['maximum']:.4f}→{b['maximum']:.4f} dB（差 {b['maximum']-a['maximum']:+.4f}）、最大位置 ({a['maximum_x']},{a['maximum_y']})→({b['maximum_x']},{b['maximum_y']})、新規座標 {new}。外部LOIO閾値 {b['loio_threshold_stride16_training']:.4f} dB、検出 {a['loio_detected']}→{b['loio_detected']}。")
    target16=next(r for r in metrics if r['image']==17 and r['stride']==16 and r['domain']=='all' and r['selection']=='center')
    target4=next(r for r in metrics if r['image']==17 and r['stride']==4 and r['domain']=='all' and r['selection']=='center')
    if not target16['loio_detected'] and not target4['loio_detected']:
        lines.append('この2画像・固定条件では、image17は高密度化後も外部LOIO閾値を超えず、stride16の粗さだけが未検出の原因とは支持されない。一方、格子と傷の重なりや方向特徴が主因であることは本実験だけでは証明されない。')
    elif not target16['loio_detected'] and target4['loio_detected']:
        lines.append('image17では高密度化によって外部LOIO閾値を超える新規中心点が現れ、stride16の座標間隔がこの条件での未検出に寄与したことを支持する。ただし他画像へ一般化できず、格子との重なりの影響も別途検証が必要。')
    else:
        lines.append('image17の閾値超過は条件ごとの上記実測値を参照。格子との重なりの因果効果や他画像への一般化は未検証。')
    lines+=['','## 限界と解釈','小さいストライドで点数や最大値が増えても、それ自体は検出精度の改善を意味しない。common16結果が一致すれば追加座標に起因する違いと判断できるが、窓の違いの物理的原因までは示さない。FFT窓と傷の距離・占有率・Hann重みは互いに相関し、周囲模様も同時に変わるため、相関はHann窓の因果効果を証明しない。', '黒い縞の数や格子方向の変化は自動分類・計測していない。原画像、同一表示条件の拡大図と輝度プロファイルから確認できる範囲に限り、縞との重なりが主因と断定しない。放射平均は方向情報を落とすが、この実験は方向保持手法と比較していない。', '各地点・窓は強く重複し独立標本ではない。2画像だけでは汎化は判断できない。ピーク作業セットは各実行プロセスの生涯値でありFFTのみのメモリではない。解析時間にはFFT、特徴計算、周辺比較が含まれ、画像読込・図保存は含まない。']
    (summary/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Summary:',summary,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--data',type=Path,default=base.DATA); p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--workers',type=int,default=4); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--summary-only',action='store_true')
    args=p.parse_args()
    if args.workers!=4 or args.batch_size!=64: p.error('fixed workers=4 and batch size=64')
    args.output.mkdir(parents=True,exist_ok=True)
    settings={'images':list(IMAGE_IDS),'window':WINDOW,'strides':list(STRIDES),'spacing':SPACING,'frequency':[.01,.5],'bin_width':.01,'workers':args.workers,'batch_size':args.batch_size,'data':str(args.data.resolve())}
    path=args.output/'settings.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=settings: p.error('Existing output has different settings; choose a new directory')
    if not path.exists(): path.write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
    if not args.summary_only:
        for i in IMAGE_IDS:
            for stride in STRIDES:
                # A separate process gives an independent peak memory measurement.
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--internal',str(i),str(stride),'--data',str(args.data),'--output',str(args.output)],check=True)
    summarize(args)


if __name__=='__main__':
    if '--internal' in sys.argv:
        p=argparse.ArgumentParser(); p.add_argument('--internal',nargs=2,type=int,required=True); p.add_argument('--data',type=Path,default=base.DATA); p.add_argument('--output',type=Path,default=OUT)
        a=p.parse_args(); a.workers=4; a.batch_size=64; run_one(a.internal[0],a.internal[1],a)
    else: main()
