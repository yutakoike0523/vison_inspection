#!/usr/bin/env python3
"""Resume-safe image-level evaluation of fixed radial FFT settings."""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import psutil
from PIL import Image
from scipy import fft
from numpy.lib.stride_tricks import sliding_window_view
import multiscale_fft_experiment as base
import radial_fft_experiment as radial

N=256; STRIDE=16; SPACING=128; FREQ=(.01,.5); WIDTH=.01; EPS=base.EPS
MIN_REFERENCES=3; MIN_NORMAL=100; MIN_STD=1e-6
NO_GRID={1,19,20,21}
CLASSES=['局所変形','点状異常','線状異常','面状異常']
OUT=base.ROOT/'output_all_images_radial_fft'


def canon(label):
    s=unicodedata.normalize('NFKC',label).strip().lower().replace(' ','').replace('　','')
    mapping={'局所変形':'局所変形','局部変形':'局所変形','点状異常':'点状異常','線状異常':'線状異常','面状異常':'面状異常','tag':'tag','タグ':'tag','background':'background','背景':'background'}
    return mapping.get(s,s)


def annotation(path,shape):
    if not path.exists(): return None
    entries=base.load_labels(path,shape)
    return [(canon(label),mask) for label,mask in entries]


def neighbor_features(features,mask=None):
    ny,nx=features.shape[:2]; out=np.full((ny,nx),np.nan,np.float32); counts=np.zeros((ny,nx),np.uint8)
    median_time=score_time=0.; step=SPACING//STRIDE
    offsets=[(dy*step,dx*step) for dy in (-1,0,1) for dx in (-1,0,1) if dy or dx]
    for y in range(ny):
        for x in range(nx):
            adjacent=[features[y+dy,x+dx] for dy,dx in offsets if 0<=y+dy<ny and 0<=x+dx<nx]
            counts[y,x]=len(adjacent)
            if len(adjacent)<MIN_REFERENCES: continue
            t=time.perf_counter(); ref=np.median(np.stack(adjacent),axis=0); median_time+=time.perf_counter()-t
            t=time.perf_counter(); diff=features[y,x]-ref; diff=diff[mask] if mask is not None else diff
            out[y,x]=np.sqrt(np.mean(diff**2)); score_time+=time.perf_counter()-t
    return out,counts,median_time,score_time


def references(shape):
    h,w=shape; ny=max(0,(h-N)//STRIDE+1); nx=max(0,(w-N)//STRIDE+1); step=SPACING//STRIDE
    return np.asarray([[sum(0<=y+dy*step<ny and 0<=x+dx*step<nx for dy in (-1,0,1) for dx in (-1,0,1) if dy or dx) for x in range(nx)] for y in range(ny)])


def analyze(img,workers,batch,also_2d):
    started=time.perf_counter(); h,w=img.shape; ys=np.arange(0,h-N+1,STRIDE); xs=np.arange(0,w-N+1,STRIDE)
    _,_,_,specs=radial.grids(N,WIDTH,[FREQ]); spec=specs[FREQ]
    spectra=np.empty((len(ys),len(xs),spec['valid'].sum()),np.float32)
    hann=np.hanning(N).astype(np.float32); hann=np.outer(hann,hann)
    fft_s=rad_s=0.; memmap=None; temp=None
    if also_2d:
        import tempfile
        temp=tempfile.TemporaryDirectory(prefix='all_radial_fft_')
        memmap=np.memmap(Path(temp.name)/'spectra.dat',dtype=np.float32,mode='w+',shape=(len(ys),len(xs),N,N//2+1))
    for iy,y in enumerate(ys):
        windows=np.moveaxis(sliding_window_view(img[y:y+N],N,axis=1),1,0)[::STRIDE]
        for ix in range(0,len(xs),batch):
            v=np.asarray(windows[ix:ix+batch],dtype=np.float32).copy(); v-=v.mean(axis=(-2,-1),keepdims=True); v*=hann
            t=time.perf_counter(); z=fft.rfft2(v,axes=(-2,-1),workers=workers); fft_s+=time.perf_counter()-t
            power=(np.abs(fft.fftshift(z,axes=(-2,)))**2).astype(np.float32)
            total=np.maximum(power[:,spec['mask']].sum(axis=1),EPS)
            if also_2d:
                norm=power/total[:,None,None]; memmap[iy,ix:ix+batch]=(10*np.log10(norm+EPS)).astype(np.float32)
            t=time.perf_counter(); sums=power.reshape(len(power),-1)@spec['matrix']
            avg=sums[:,spec['valid']]/spec['counts'][spec['valid']][None,:]/total[:,None]
            spectra[iy,ix:ix+batch]=(10*np.log10(avg+EPS)).astype(np.float32); rad_s+=time.perf_counter()-t
    t=time.perf_counter(); scores,counts,median_s,score_s=neighbor_features(spectra); map_s=time.perf_counter()-t
    scores2=None; median2=score2=None
    if also_2d:
        scores2,_,median2,score2=neighbor_features(memmap,spec['mask']); del memmap; temp.cleanup()
    elapsed=time.perf_counter()-started
    return dict(xs=xs+N//2,ys=ys+N//2,scores=scores,counts=counts,spectra=spectra,scores2=scores2,bin_counts=spec['counts'],bin_valid=spec['valid'],fft_seconds=fft_s,radial_seconds=rad_s,median_seconds=median_s,score_seconds=score_s,map_seconds=map_s,total_seconds=elapsed,median_2d_seconds=median2,score_2d_seconds=score2,peak_memory_bytes=psutil.Process().memory_info().peak_wset if os.name=='nt' else None)


def summary_stats(v):
    v=v[np.isfinite(v)]
    if not len(v): return {'count':0,'mean':None,'median':None,'std':None,'p95':None,'p99':None,'max':None}
    return {'count':len(v),'mean':float(np.mean(v)),'median':float(np.median(v)),'std':float(np.std(v)),'p95':float(np.percentile(v,95)),'p99':float(np.percentile(v,99)),'max':float(np.max(v))}


def values(data,mask): return data['scores'][mask & np.isfinite(data['scores'])]


def defect_stats(v,normal):
    stats=summary_stats(v); usable=normal['count']>=MIN_NORMAL and normal['std'] is not None and normal['std']>=MIN_STD
    stats.update({'median_difference':stats['median']-normal['median'] if stats['count'] and normal['count'] else None,
                  'S':(stats['median']-normal['median'])/normal['std'] if stats['count'] and usable else None,
                  'normal_p99_exceed_count':int(np.sum(v>normal['p99'])) if len(v) and usable else None,
                  'normal_p99_exceed_rate':float(np.mean(v>normal['p99'])) if len(v) and usable else None})
    return stats


def classification(labels,data):
    if labels is None: return None,[]
    allmask=np.logical_or.reduce([m for _,m in labels]) if labels else np.zeros((int(data['ys'][-1]+N//2),int(data['xs'][-1]+N//2)),bool)
    normal=~base.selections(allmask,data,N)[1]
    entries=[]
    for idx,(name,mask) in enumerate(labels):
        if name not in CLASSES: continue
        yy,xx=np.nonzero(mask); center,overlap=base.selections(mask,data,N)
        entries.append(dict(index=idx,label=name,center=center,overlap=overlap,width=int(xx.max()-xx.min()+1) if len(xx) else 0,height=int(yy.max()-yy.min()+1) if len(yy) else 0,area=int(mask.sum())))
    return normal,entries


def produce_images(path,img,labels,data,normal,entries):
    shutil.copy2(path,data['folder']/'original.png')
    folder=data['folder']; r=data; score=data['scores']
    for typ in ('anomaly_map','anomaly_overlay','annotations'):
        fig,ax=plt.subplots(figsize=(9,7))
        if typ!='anomaly_map': ax.imshow(img,cmap='gray',extent=[0,img.shape[1],img.shape[0],0])
        if typ!='annotations':
            im=ax.imshow(score,extent=base.extent(r),cmap='inferno',interpolation='nearest',alpha=.55 if typ=='anomaly_overlay' else 1)
            fig.colorbar(im,ax=ax,label='Radial RMS (dB)')
        if typ!='anomaly_map':
            for label,mask in labels or []:
                if label=='局所変形' and mask.any(): ax.contour(mask.astype(np.uint8),levels=[.5],colors='cyan',linewidths=1,origin='upper',extent=[0,img.shape[1],img.shape[0],0])
        ax.set(xlim=(0,img.shape[1]),ylim=(img.shape[0],0),xlabel='x (px)',ylabel='y (px)',title=typ)
        fig.savefig(folder/f'{typ}.png',dpi=110,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    if normal is not None:
        v=values(data,normal)
        if len(v): ax.hist(v,bins=60,density=True,alpha=.55,label='Normal')
        for e in entries:
            if e['label']=='局所変形':
                v=values(data,e['center'])
                if len(v): ax.hist(v,bins=30,density=True,alpha=.5,label=f"Deformation {e['index']} center")
        ax.legend()
    else: ax.text(.5,.5,'Annotation unavailable',transform=ax.transAxes,ha='center')
    ax.set(xlabel='Radial RMS (dB)',ylabel='Density'); fig.savefig(folder/'distribution.png',dpi=120,bbox_inches='tight'); plt.close(fig)
    x=(np.arange(len(data['bin_counts']))+.5)*WIDTH; x=x[data['bin_valid']]; step=SPACING//STRIDE
    for e in entries:
        if e['label']!='局所変形': continue
        for mode in ('center','overlap'):
            yy,xx=np.nonzero(e[mode]&np.isfinite(score))
            if len(yy): break
        if not len(yy): continue
        k=int(np.argmax(score[yy,xx])); iy=int(yy[k]); ix=int(xx[k]); target=data['spectra'][iy,ix]
        near=[data['spectra'][iy+dy*step,ix+dx*step] for dy in (-1,0,1) for dx in (-1,0,1) if (dy or dx) and 0<=iy+dy*step<len(data['ys']) and 0<=ix+dx*step<len(data['xs'])]
        ref=np.median(np.stack(near),axis=0); diff=target-ref; biggest=int(np.argmax(np.abs(diff)))
        e['representative']={'x':int(data['xs'][ix]),'y':int(data['ys'][iy]),'selection':mode,'peak_frequency':float(x[biggest]),'peak_difference_db':float(diff[biggest])}
        fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True); axes[0].plot(x,target,label='Target'); axes[0].plot(x,ref,label='Neighbor median'); axes[0].legend(); axes[0].set(ylabel='Power (dB)')
        axes[1].plot(x,diff); axes[1].axhline(0,color='gray',lw=.5); axes[1].set(xlabel='Radius (cycles/pixel)',ylabel='Difference (dB)')
        fig.suptitle(f"Deformation {e['index']}, {mode} ({data['xs'][ix]}, {data['ys'][iy]})"); fig.savefig(folder/f"spectrum_defect_{e['index']}.png",dpi=120,bbox_inches='tight'); plt.close(fig)


def process_one(number,args):
    name=f'image{number}'; image_path=args.data/f'{name}.png'; destination=args.output/name
    if destination.exists() and (destination/'metrics.json').exists():
        print(f'{name}: completed, skip',flush=True); return
    staging=base.unique_dir(args.output/(name+'_partial')); final=destination
    info={'image':name,'number':number,'group':'no_grid' if number in NO_GRID else 'grid','status':'','reason':'','label_available':False}
    if not image_path.is_file():
        info.update(status='excluded',reason='image missing')
    else:
        img=np.asarray(Image.open(image_path).convert('L'),dtype=np.float32); info['shape']=list(img.shape)
        refs=references(img.shape); info['max_available_references']=int(refs.max()) if refs.size else 0
        if not refs.size or info['max_available_references']<MIN_REFERENCES:
            info.update(status='excluded',reason=f'No windows with at least {MIN_REFERENCES} references; maximum {info["max_available_references"]}')
        else:
            labels=annotation(image_path.with_suffix('.json'),img.shape); info['label_available']=labels is not None
            data=analyze(img,args.workers,args.batch_size,args.two_d); data['folder']=staging
            normal,entries=classification(labels,data)
            normal_stats=summary_stats(values(data,normal)) if normal is not None else summary_stats(np.array([]))
            produce_images(image_path,img,labels,data,normal,entries)
            instances=[]
            for e in entries:
                result={k:v for k,v in e.items() if k not in ('center','overlap')}
                result['long_side']=max(e['width'],e['height'])
                for mode in ('center','overlap'): result[mode]=defect_stats(values(data,e[mode]),normal_stats)
                instances.append(result)
            aggregates={}
            for kind in CLASSES:
                selected=[e for e in entries if e['label']==kind]
                aggregates[kind]={'annotation_count':len(selected)}
                for mode in ('center','overlap'):
                    union=np.logical_or.reduce([e[mode] for e in selected]) if selected else np.zeros(data['scores'].shape,bool)
                    aggregates[kind][mode]=defect_stats(values(data,union),normal_stats)
            info.update(status='analyzed',reason='',normal=normal_stats,instances=instances,aggregates=aggregates,
                timings={k:data[k] for k in ('fft_seconds','radial_seconds','median_seconds','score_seconds','map_seconds','total_seconds','median_2d_seconds','score_2d_seconds','peak_memory_bytes')})
            payload={'anomaly_map':data['scores'],'x_centers':data['xs'],'y_centers':data['ys'],'reference_counts':data['counts'],'radial_spectra':data['spectra'],'bin_counts':data['bin_counts'],'bin_valid':data['bin_valid']}
            if data['scores2'] is not None: payload['anomaly_map_2d']=data['scores2']
            np.savez_compressed(staging/'result.npz',**payload)
    (staging/'metrics.json').write_text(json.dumps(info,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if final.exists(): raise FileExistsError(f'Cannot replace existing directory: {final}')
    staging.rename(final)
    print(f'{name}: {info["status"]}',flush=True)


def weighted_threshold(normal_sets,q=.99):
    # Equal image weight: mixture of each image's empirical distribution.
    arrays=[np.sort(a) for a in normal_sets if len(a)>=MIN_NORMAL]
    if len(arrays)<2: return None
    samples=np.concatenate(arrays); weights=np.concatenate([np.full(len(a),1/len(a)) for a in arrays])
    order=np.argsort(samples); cum=np.cumsum(weights[order]); index=np.searchsorted(cum,q*len(arrays),side='left')
    return float(samples[order[min(index,len(order)-1)]])


def load_evaluation(args,info):
    path=args.data/(info['image']+'.png'); image=np.asarray(Image.open(path).convert('L'))
    labels=annotation(path.with_suffix('.json'),image.shape)
    with np.load(args.output/info['image']/'result.npz') as d:
        data={'xs':d['x_centers'].copy(),'ys':d['y_centers'].copy(),'scores':d['anomaly_map'].copy()}
    normal,entries=classification(labels,data)
    return data,normal,entries


def safe_csv(path,rows,fields=None):
    if rows:
        columns=list(dict.fromkeys(k for row in rows for k in row))
    else: columns=fields or ['image']
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns); writer.writeheader(); writer.writerows(rows)


def plot_summary(records,datasets,instances,thresholds,summary):
    analyzed=[r for r in records if r['status']=='analyzed']; local=[r for r in analyzed if r['aggregates']['局所変形']['annotation_count']]
    fig,ax=plt.subplots(figsize=(11,4))
    for group,color in [('grid','steelblue'),('no_grid','darkorange')]:
        rs=[r for r in local if r['group']==group and r['aggregates']['局所変形']['center']['S'] is not None]
        ax.bar([r['number'] for r in rs],[r['aggregates']['局所変形']['center']['S'] for r in rs],color=color,label=group)
    ax.axhline(0,color='black',lw=.6); ax.set(xlabel='Image number',ylabel='S (center)',xticks=range(1,22)); ax.legend(); fig.savefig(summary/'separation_scores.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5)); ticks=[]; labels=[]
    for r in local:
        d,normal,entries=datasets[r['image']]; union=np.logical_or.reduce([e['center'] for e in entries if e['label']=='局所変形'])
        a=values(d,normal); b=values(d,union); x=r['number']*3
        if len(a): ax.boxplot(a,positions=[x-.4],widths=.7,showfliers=False,patch_artist=True,boxprops=dict(facecolor='skyblue'))
        if len(b): ax.boxplot(b,positions=[x+.4],widths=.7,showfliers=False,patch_artist=True,boxprops=dict(facecolor='salmon'))
        ticks.append(x); labels.append(str(r['number']))
    ax.set_xticks(ticks,labels); ax.set(xlabel='Image number (blue normal, red deformation center)',ylabel='Radial RMS (dB)'); fig.savefig(summary/'anomaly_distributions.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots()
    for group,marker in [('grid','o'),('no_grid','x')]:
        rows=[i for i in instances if i['group']==group and i['center_S'] is not None and i['label']=='局所変形']
        ax.scatter([i['long_side'] for i in rows],[i['center_S'] for i in rows],marker=marker,label=group)
    ax.set(xlabel='Annotation long side (px)',ylabel='S (center)'); ax.legend(); fig.savefig(summary/'defect_size_vs_score.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    for r in local:
        t=next((x for x in thresholds if x['image']==r['image'] and x['selection']=='center'),None)
        own=r['aggregates']['局所変形']['center']['normal_p99_exceed_rate']
        if own is not None: axes[0].scatter(r['number']-.12,own,color='steelblue',label='Own normal p99' if r is local[0] else None)
        if t and t['defect_exceed_rate'] is not None: axes[0].scatter(r['number']+.12,t['defect_exceed_rate'],color='darkorange',label='LOIO' if r is local[0] else None)
    axes[0].legend(); axes[0].set(ylabel='Center point exceedance')
    for group,color in [('grid','steelblue'),('no_grid','darkorange')]:
        rows=[t for t in thresholds if t['selection']=='center' and t['group']==group]
        axes[1].scatter([t['number'] for t in rows if t['normal_exceed_rate'] is not None],[t['normal_exceed_rate'] for t in rows if t['normal_exceed_rate'] is not None],c=color,marker='o',label=f'{group} normal')
        axes[1].scatter([t['number'] for t in rows if t['instance_detection_rate'] is not None],[t['instance_detection_rate'] for t in rows if t['instance_detection_rate'] is not None],c=color,marker='x',label=f'{group} instance')
    axes[1].set(xlabel='Image number',ylabel='LOIO rate',xticks=range(1,22),ylim=(-.05,1.05)); axes[1].legend(ncol=2)
    fig.savefig(summary/'threshold_exceedance.png',dpi=130,bbox_inches='tight'); fig.savefig(summary/'detection_per_image.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    fig,axes=plt.subplots(5,5,figsize=(16,13),constrained_layout=True)
    for ax,i in zip(axes.flat,range(1,26)):
        if i>21: ax.axis('off'); continue
        r=next((x for x in records if x['number']==i),None)
        if r and r['status']=='analyzed':
            d=datasets[r['image']][0]; im=ax.imshow(d['scores'],extent=base.extent(d),cmap='inferno'); ax.set_title(f"image{i} ({r['group']})"); fig.colorbar(im,ax=ax,shrink=.65)
        else: ax.text(.5,.5,f'image{i}: excluded',ha='center',va='center',transform=ax.transAxes); ax.set_axis_off()
    fig.suptitle('Per-image color scales, not common thresholds'); fig.savefig(summary/'all_anomaly_maps.png',dpi=120); plt.close(fig)


def summarize_all(args):
    root=args.output; records=[]
    for i in range(1,22):
        p=root/f'image{i}'/'metrics.json'
        if p.exists(): records.append(json.loads(p.read_text(encoding='utf-8')))
        else: records.append({'image':f'image{i}','number':i,'group':'no_grid' if i in NO_GRID else 'grid','status':'failed_or_unprocessed','reason':'No completed metrics.json'})
    summary=base.unique_dir(root/'summary'); datasets={}
    for r in records:
        if r['status']=='analyzed': datasets[r['image']]=load_evaluation(args,r)
    normal_rows=[]; image_rows=[]; instances=[]; timing=[]; ranges=[]
    for r in records:
        row={k:r.get(k) for k in ('image','number','group','status','reason','label_available','shape','max_available_references')}; row.update(window=N,stride=STRIDE,reference_spacing=SPACING,min_references=MIN_REFERENCES,frequency_min=FREQ[0],frequency_max=FREQ[1],bin_width=WIDTH)
        if r['status']=='analyzed':
            ns=r['normal']; normal_rows.append({'image':r['image'],'group':r['group'],**ns})
            timing.append({'image':r['image'],'group':r['group'],**r['timings']})
            for kind,agg in r['aggregates'].items():
                for mode in ('center','overlap'):
                    ranges.append({'image':r['image'],'group':r['group'],'label':kind,'selection':mode,'annotation_count':agg['annotation_count'],**agg[mode]})
            agg=r['aggregates']['局所変形']; row['deformation_annotations']=agg['annotation_count']
            for mode in ('center','overlap'):
                row.update({f'{mode}_{k}':v for k,v in agg[mode].items()})
            for item in r['instances']:
                flat={'image':r['image'],'number':r['number'],'group':r['group'],'label':item['label'],'annotation_index':item['index'],'width_px':item['width'],'height_px':item['height'],'area_pixels':item['area'],'long_side':item['long_side']}
                for mode in ('center','overlap'): flat.update({f'{mode}_{k}':v for k,v in item[mode].items()})
                if 'representative' in item: flat.update({f'representative_{k}':v for k,v in item['representative'].items()})
                instances.append(flat)
        image_rows.append(row)
    thresholds=[]; instance_threshold=[]; overlap_diagnostics=[]
    for r in records:
        if r['status']!='analyzed' or not r['label_available']: continue
        train=[]; train_names=[]
        for other in records:
            if other['status']!='analyzed' or not other['label_available'] or other['group']!=r['group'] or other['image']==r['image']: continue
            d,n,_=datasets[other['image']]; arr=values(d,n)
            if len(arr)>=MIN_NORMAL: train.append(arr); train_names.append(other['image'])
        threshold=weighted_threshold(train); d,n,entries=datasets[r['image']]; normal=values(d,n)
        local=[e for e in entries if e['label']=='局所変形']
        for mode in ('center','overlap'):
            union=np.logical_or.reduce([e[mode] for e in local]) if local else np.zeros(d['scores'].shape,bool)
            points=values(d,union); detected=0; missed=0; unassessable=0
            for e in local:
                arr=values(d,e[mode]); hit=bool(np.any(arr>threshold)) if threshold is not None and len(arr) else None
                if hit is True: detected+=1
                elif hit is False: missed+=1
                else: unassessable+=1
                instance_threshold.append({'image':r['image'],'group':r['group'],'annotation_index':e['index'],'selection':mode,'threshold':threshold,'point_count':len(arr),'detected':hit,'max_score':float(np.max(arr)) if len(arr) else None})
                if mode=='overlap' and threshold is not None:
                    image_path=args.data/(r['image']+'.png'); image_shape=tuple(r['shape'])
                    labs=annotation(image_path.with_suffix('.json'),image_shape)
                    unwanted=[mask for label,mask in labs if label in ('tag','background')]
                    contaminated=base.selections(np.logical_or.reduce(unwanted),d,N)[1] if unwanted else np.zeros(d['scores'].shape,bool)
                    hit_windows=e['overlap'] & np.isfinite(d['scores']) & (d['scores']>threshold)
                    overlap_diagnostics.append({'image':r['image'],'group':r['group'],'annotation_index':e['index'],'threshold':threshold,'hit_windows':int(hit_windows.sum()),'hit_windows_center_outside':int(np.sum(hit_windows & ~e['center'])),'hit_windows_touch_tag_or_background':int(np.sum(hit_windows & contaminated)),'hit_windows_clean_center_outside':int(np.sum(hit_windows & ~e['center'] & ~contaminated))})
            thresholds.append({'image':r['image'],'number':r['number'],'group':r['group'],'selection':mode,'threshold':threshold,'training_images':','.join(train_names),'training_image_count':len(train_names),'normal_count':len(normal),'normal_exceed_rate':float(np.mean(normal>threshold)) if threshold is not None and len(normal) else None,'defect_count':len(points),'defect_exceed_rate':float(np.mean(points>threshold)) if threshold is not None and len(points) else None,'detected_instances':detected,'missed_instances':missed,'unassessable_instances':unassessable,'instance_detection_rate':detected/(detected+missed) if detected+missed else None})
    safe_csv(summary/'image_metrics.csv',image_rows); safe_csv(summary/'normal_metrics.csv',normal_rows); safe_csv(summary/'defect_instance_metrics.csv',instances)
    safe_csv(summary/'class_metrics.csv',ranges); safe_csv(summary/'common_threshold_metrics.csv',thresholds); safe_csv(summary/'instance_threshold_metrics.csv',instance_threshold); safe_csv(summary/'processing_times.csv',timing)
    safe_csv(summary/'overlap_hit_diagnostics.csv',overlap_diagnostics)
    plot_summary(records,datasets,instances,thresholds,summary)
    valid=[r for r in records if r['status']=='analyzed']; excluded=[r for r in records if r['status']!='analyzed']
    lines=['# 放射方向平均FFT：全画像評価','', '## 条件',f'256×256 px、stride 16、参照3×3中央除外、参照間隔128 px、0.01〜0.50 cycles/pixel、半径幅0.01。元コードと同じ片側FFTパワー総和で正規化し、左右対称重み1または2を使った線形パワーの放射平均をdB化し周辺中央値との差をRMS集計。成分数4未満のビンは除外。前処理と式はimage5実験と共通。',f'最低参照数 {MIN_REFERENCES}。正常窓は全アノテーションと非重複。正常点{MIN_NORMAL}未満または標準偏差{MIN_STD:g}未満のSは評価不能。未ラベルを正常とみなす仮定を含む。', '共通閾値は各評価画像を除いた同じグループの正常窓から、画像ごとに合計重みが等しい経験分布の99%点として計算。各fold内では単一の閾値を使用し、fold間は変わる。参照側のラベルは正常窓を選ぶために使用し、評価画像自身のラベルを閾値設定に使用しない。正常99%点およびLOIOの境界は未知画像に事前固定された1個の閾値ではない。格子あり・なしは独立に学習・集計。傷インスタンス検出は該当窓の1つ以上が閾値超の場合。重なった窓を独立した傷として数えない。', 'image5はパラメータ選択に使われたため完全な独立評価ではない。他画像も同一データ収集の可能性があり、結果を一般化するには別データで検証が必要。', '', '## 解析可否']
    lines.append(f'解析 {len(valid)}/21画像。除外・失敗: '+(', '.join(f"{r['image']} ({r['reason']})" for r in excluded) or 'なし'))
    for group in ('grid','no_grid'):
        group_records=[r for r in valid if r['group']==group]; with_defect=[r for r in group_records if r['aggregates']['局所変形']['annotation_count']]
        number=sum(r['aggregates']['局所変形']['annotation_count'] for r in group_records)
        lines+=['',f'## {group} グループ',f'解析 {len(group_records)}画像、局所変形あり {len(with_defect)}画像、変形アノテーション {number}個。']
        for r in with_defect:
            d=r['aggregates']['局所変形']['center']; t=next((t for t in thresholds if t['image']==r['image'] and t['selection']=='center'),None)
            if d['S'] is None: lines.append(f"- {r['image']}: 中心基準は評価不能（中心点または正常点不足）。")
            else: lines.append(f"- {r['image']}: S={d['S']:.3f}、画像内正常99%超 {d['normal_p99_exceed_count']}/{d['count']}、LOIO閾値 {t['threshold']:.3f}、LOIO中心窓超過 {t['defect_exceed_rate']:.1%}、検出インスタンス {t['detected_instances']}/{t['detected_instances']+t['missed_instances']}。")
        ts=[t for t in thresholds if t['group']==group and t['selection']=='center']; detected=sum(t['detected_instances'] for t in ts); missed=sum(t['missed_instances'] for t in ts); na=sum(t['unassessable_instances'] for t in ts)
        if detected+missed: lines.append(f'LOIO中心基準: 検出 {detected}/{detected+missed} インスタンス、評価不能 {na}。窓単位の結果はCSV参照。')
        if ts: lines.append('正常窓の閾値超過率（画像平均）: '+f"{np.mean([t['normal_exceed_rate'] for t in ts if t['normal_exceed_rate'] is not None]):.2%}。")
    if timing: lines+=['','## 計算時間・メモリ',f"解析時間中央値 {np.median([t['total_seconds'] for t in timing]):.2f} 秒/画像。CSVにFFT、放射変換、中央値、スコア、マップ全体を記録。入力読込と図・NPZ保存は含まない。メモリは各画像の独立プロセスでの解析完了時点までのピーク作業セットで、図・保存は含まない。"]
    lines += ['', '## 考察', 'Sは検出率ではなく正常分布との中央値差を標準偏差で割った値。S、画像内99%超、LOIOの窓超過率と傷単位の検出を併せて解釈する。256 px窓と128 px離れた参照は重なるため、対象と基準の双方に変形が入る可能性がある。未ラベルの傷、撮像条件、札に近い参照窓、欠陥の位置・サイズは結果に影響し得るが、ここでは原因を個別に検証していない。', '傷の画素面積と長辺をインスタンスCSVに記録。傷が少なく画像や傷の条件も揃っていないため、大きさと性能の因果関係は断定できない。代表スペクトルの最大差周波数はインスタンスCSVに保存し、各代表点のスペクトル図と併せて確認する。', '放射平均は方向情報を失うが、方向情報を使う方法や格子点位置の直接計測との比較は行っていないため、欠点や改善効果は実証されていない。画像間のばらつきが大きい場合は現条件の一般化を留保し、低周波成分と方向情報・格子点の位置ずれを独立画像で比較する。マルチスケールは今回の固定条件とは別実験で検証する。']
    grid_center=[t for t in thresholds if t['group']=='grid' and t['selection']=='center']
    grid_overlap=[t for t in thresholds if t['group']=='grid' and t['selection']=='overlap']
    gd=[r for r in overlap_diagnostics if r['group']=='grid' and r['hit_windows']]
    if grid_center:
        center_hits=sum(t['detected_instances'] for t in grid_center); center_total=sum(t['detected_instances']+t['missed_instances'] for t in grid_center)
        overlap_hits=sum(t['detected_instances'] for t in grid_overlap); overlap_total=sum(t['detected_instances']+t['missed_instances'] for t in grid_overlap)
        tag_only=sum(r['hit_windows']==r['hit_windows_touch_tag_or_background'] for r in gd)
        lines += ['', '## 今回の実測からの判断',f'格子ありの中心基準は {center_hits}/{center_total} インスタンス、重なり基準は {overlap_hits}/{overlap_total}。重なり基準で検出扱いのうち、全ての閾値超過窓が札・背景に重なるものが {tag_only} 件ある。窓の重なりによる検出は位置の正確さを保証しない（overlap_hit_diagnostics.csv）。']
        lookup={(r['image'],r['annotation_index']):r for r in instances if r['group']=='grid' and r['label']=='局所変形'}
        detected_sizes=[]; missed_sizes=[]
        for t in instance_threshold:
            key=(t['image'],t['annotation_index'])
            if t['group']=='grid' and t['selection']=='center' and key in lookup:
                if t['detected'] is True: detected_sizes.append(lookup[key]['long_side'])
                elif t['detected'] is False: missed_sizes.append(lookup[key]['long_side'])
        if detected_sizes and missed_sizes:
            lines.append(f'格子ありの中心検出例の長辺中央値は {np.median(detected_sizes):g} px、未検出例は {np.median(missed_sizes):g} px。標本が少なく、大きさ以外の条件も異なるため因果関係は判断できない。')
        frequencies=[r['representative_peak_frequency'] for r in lookup.values() if r.get('representative_peak_frequency') is not None]
        if frequencies: lines.append(f'代表変形点の最大差ビンが0.05 cycles/pixel未満だったのは {sum(f<.05 for f in frequencies)}/{len(frequencies)} 件。選択した代表点と周辺中央値の差であり、低周波だけで検出できることは示さない。')
        lines.append('今回の固定条件は画像間で安定した中心検出を示さなかったため、単独採用は保留。低周波の要因分析を優先し、方向情報・格子点のずれ・複数窓サイズを、別画像で比較する。')
    (summary/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Summary:',summary,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=base.DATA); p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--workers',type=int,default=4); p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--two-d',action='store_true',help='Also compute original 2D spectrum RMS at the same frequency range (much slower)')
    p.add_argument('--single',type=int,choices=range(1,22),help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.workers<1 or args.batch_size<1: p.error('workers and batch-size must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    settings={'window':N,'stride':STRIDE,'reference_spacing':SPACING,'min_references':MIN_REFERENCES,'frequency':FREQ,'bin_width':WIDTH,'workers':args.workers,'batch_size':args.batch_size,'two_d':args.two_d,'data':str(args.data.resolve()),'version':1}
    path=args.output/'settings.json'
    if path.exists():
        previous=json.loads(path.read_text(encoding='utf-8'))
        if previous!=json.loads(json.dumps(settings)): p.error('Output directory settings differ; choose a new --output directory')
    else: path.write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.single:
        process_one(args.single,args); return
    print('Full FFT symmetry validation:',radial.check_full_fft(WIDTH),flush=True)
    for i in range(1,22):
        if (args.output/f'image{i}'/'metrics.json').exists(): print(f'image{i}: completed, skip',flush=True); continue
        cmd=[sys.executable,str(Path(__file__).resolve()),'--single',str(i),'--data',str(args.data),'--output',str(args.output),'--workers',str(args.workers),'--batch-size',str(args.batch_size)]
        if args.two_d: cmd.append('--two-d')
        result=subprocess.run(cmd)
        if result.returncode: print(f'image{i}: failed (exit {result.returncode}); continuing',flush=True)
    summarize_all(args)


if __name__=='__main__': main()
