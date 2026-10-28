#!/usr/bin/env python3
"""Evaluate radial FFT window size against annotated deformation size."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import pearsonr, spearmanr
import multiscale_fft_experiment as base
import evaluate_all_radial_fft as prev
import radial_fft_experiment as radial

WINDOWS=(64,128,192,256)
OUT=base.ROOT/'output_window_size_vs_defect_size'
MIN_DEFECT_POINTS=4
PREVIOUS=base.ROOT/'output_all_images_radial_fft'


def group(i): return 'small_image' if i==18 else ('no_grid' if i in prev.NO_GRID else 'grid')


def save_result(path,r):
    np.savez_compressed(path,anomaly_map=r['scores'],x_centers=r['xs'],y_centers=r['ys'],reference_counts=r['counts'],radial_spectra=r['spectra'],bin_counts=r['bin_counts'],bin_valid=r['bin_valid'])


def load_result(path,features=False):
    with np.load(path) as z:
        result={'xs':z['x_centers'].copy(),'ys':z['y_centers'].copy(),'scores':z['anomaly_map'].copy(),'counts':z['reference_counts'].copy(),'bin_counts':z['bin_counts'].copy(),'bin_valid':z['bin_valid'].copy()}
        if features: result['spectra']=z['radial_spectra'].copy()
        return result


def prev_256_compatible(args):
    try: settings=json.loads((PREVIOUS/'settings.json').read_text(encoding='utf-8'))
    except (FileNotFoundError,ValueError): return False
    return (settings['window']==256 and settings['stride']==16 and settings['reference_spacing']==128 and settings['min_references']==3 and settings['frequency']==[.01,.5] and settings['bin_width']==.01 and settings['workers']==args.workers and settings['batch_size']==args.batch_size and Path(settings['data']).resolve()==args.data.resolve())


def process_image(i,args):
    imgpath=args.data/f'image{i}.png'; folder=args.output/f'image{i}'; folder.mkdir(parents=True,exist_ok=True)
    if not imgpath.exists():
        (folder/'missing_image.json').write_text(json.dumps({'status':'missing_image'}),encoding='utf-8'); return
    img=np.asarray(Image.open(imgpath).convert('L'),dtype=np.float32)
    for n in WINDOWS:
        output=folder/f'window_{n}.npz'; meta=folder/f'window_{n}.json'
        if output.exists() and meta.exists(): print(f'image{i}, {n}: resume skip',flush=True); continue
        prev.N=n; refs=prev.references(img.shape)
        if not refs.size or int(refs.max())<prev.MIN_REFERENCES:
            if not meta.exists(): meta.write_text(json.dumps({'status':'excluded','reason':f'maximum reference count {int(refs.max()) if refs.size else 0} < {prev.MIN_REFERENCES}'}),encoding='utf-8')
            print(f'image{i}, {n}: insufficient references',flush=True); continue
        oldnpz=PREVIOUS/f'image{i}'/'result.npz'; oldmeta=PREVIOUS/f'image{i}'/'metrics.json'
        if n==256 and prev_256_compatible(args) and oldnpz.exists() and oldmeta.exists():
            old=json.loads(oldmeta.read_text(encoding='utf-8'))
            shutil.copy2(oldnpz,output)
            timing=old['timings']; origin='previous fixed-condition experiment'
        else:
            r=prev.analyze(img,args.workers,args.batch_size,False); save_result(output,r)
            timing={k:r[k] for k in ('fft_seconds','radial_seconds','median_seconds','score_seconds','map_seconds','total_seconds','peak_memory_bytes')}
            origin='new measurement'
        metadata={'status':'analyzed','window':n,'image':i,'origin':origin,'timings':timing}
        meta.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'image{i}, {n}: {origin} {timing["total_seconds"]:.2f}s',flush=True)


def draw_image_maps(i,args,records,labels,img):
    folder=args.output/f'image{i}'; h,w=img.shape
    expected=[folder/'comparison.png',folder/'normalized_comparison.png']+[folder/f'anomaly_{n}.png' for n in records]
    if all(p.exists() for p in expected): return
    common=(128,w-128,128,h-128) if 256 in records else None
    fig,axes=plt.subplots(1,5,figsize=(20,5),constrained_layout=True)
    axes[0].imshow(img,cmap='gray',extent=[0,w,h,0]); axes[0].set_title('Image + deformation')
    for name,mask in labels or []:
        if name=='局所変形' and mask.any(): axes[0].contour(mask.astype(np.uint8),levels=[.5],colors='cyan',linewidths=1,origin='upper',extent=[0,w,h,0])
    fig2,axes2=plt.subplots(1,4,figsize=(16,4),constrained_layout=True)
    for n,ax,ax2 in zip(WINDOWS,axes[1:],axes2):
        if n not in records:
            ax.set_title(f'{n} px: excluded'); ax2.set_title(f'{n} px: excluded'); continue
        r=records[n]; a=r['scores']; ext=base.extent(r)
        im=ax.imshow(a,extent=ext,cmap='inferno',interpolation='nearest'); fig.colorbar(im,ax=ax,shrink=.7)
        ax.set_title(f'{n} px, raw'); ax.set(xlim=(0,w),ylim=(h,0))
        single=folder/f'anomaly_{n}.png'
        if not single.exists():
            figx,one=plt.subplots(figsize=(8,6)); imx=one.imshow(a,extent=ext,cmap='inferno',interpolation='nearest'); figx.colorbar(imx,ax=one,label='RMS dB'); one.set(xlim=(0,w),ylim=(h,0),xlabel='x (px)',ylabel='y (px)'); figx.savefig(single,dpi=110,bbox_inches='tight'); plt.close(figx)
        if common:
            xmin,xmax,ymin,ymax=common; reg=(r['xs'][None,:]>=xmin)&(r['xs'][None,:]<=xmax)&(r['ys'][:,None]>=ymin)&(r['ys'][:,None]<=ymax)
            vals=a[reg & np.isfinite(a)]; med=np.median(vals); std=np.std(vals)
            z=(a-med)/std if std>0 else np.full(a.shape,np.nan)
            im2=ax2.imshow(z,extent=ext,cmap='inferno',vmin=-2,vmax=4,interpolation='nearest'); fig2.colorbar(im2,ax=ax2,shrink=.7)
            ax2.set(xlim=(xmin,xmax),ylim=(ymax,ymin),title=f'{n} px')
        else: ax2.text(.5,.5,'256 px unavailable',ha='center',transform=ax2.transAxes)
    axes[0].set(xlim=(0,w),ylim=(h,0))
    if not (folder/'comparison.png').exists(): fig.savefig(folder/'comparison.png',dpi=110)
    plt.close(fig)
    fig2.suptitle('Common center region: (score - median) / std; display range -2 to 4')
    if not (folder/'normalized_comparison.png').exists(): fig2.savefig(folder/'normalized_comparison.png',dpi=110)
    plt.close(fig2)


def center_region(r,shape):
    h,w=shape; xmin=ymin=128; xmax=w-128; ymax=h-128
    return (r['xs'][None,:]>=xmin)&(r['xs'][None,:]<=xmax)&(r['ys'][:,None]>=ymin)&(r['ys'][:,None]<=ymax)


def val(r,mask): return r['scores'][mask & np.isfinite(r['scores'])]


def build_data(args):
    all_data={}; status=[]; timings=[]; bin_rows=[]
    for i in range(1,22):
        path=args.data/f'image{i}.png'; folder=args.output/f'image{i}'; maps={}
        if not path.exists(): status.append({'image':i,'window':'','status':'missing_image','reason':'file missing'}); continue
        img=np.asarray(Image.open(path).convert('L'),dtype=np.float32); labels=prev.annotation(path.with_suffix('.json'),img.shape)
        for n in WINDOWS:
            p=folder/f'window_{n}.npz'; m=folder/f'window_{n}.json'
            meta=json.loads(m.read_text(encoding='utf-8')) if m.exists() else {'status':'unprocessed','reason':'result missing'}
            status.append({'image':i,'window':n,'group':group(i),'status':meta['status'],'reason':meta.get('reason','')})
            if meta['status']!='analyzed' or not p.exists(): continue
            r=load_result(p); maps[n]=r
            timings.append({'image':i,'window':n,'group':group(i),'origin':meta['origin'],**meta['timings']})
            for k,(count,valid) in enumerate(zip(r['bin_counts'],r['bin_valid'])):
                bin_rows.append({'image':i,'window':n,'frequency_resolution':1/n,'bin_start':k*.01,'bin_end':(k+1)*.01,'component_count_full_fft':int(count),'used':bool(valid)})
            prev.N=n
            normal,entries=prev.classification(labels,r)
            if normal is None: continue
            scopes={'full':np.ones(r['scores'].shape,bool)}
            if (folder/'window_256.npz').exists(): scopes['common']=center_region(r,img.shape)
            for scope,region in scopes.items():
                all_data[(i,n,scope)]={'r':r,'normal':val(r,normal & region),'entries':entries,'region':region}
        if maps: draw_image_maps(i,args,maps,labels,img)
    return all_data,status,timings,bin_rows


def calculate_metrics(data):
    rows=[]; threshold_rows=[]; cache={}
    for (i,n,scope),item in data.items():
        same=[v['normal'] for (j,win,sc),v in data.items() if j!=i and win==n and sc==scope and group(j)==group(i)]
        threshold=prev.weighted_threshold(same) if group(i)!='small_image' else None
        cache[(i,n,scope)]=threshold
        normal=item['normal']; normal_stats=prev.summary_stats(normal)
        for e in item['entries']:
            if e['label']!='局所変形': continue
            for mode in ('center','overlap'):
                selection=e[mode]&item['region']; v=val(item['r'],selection); s=prev.defect_stats(v,normal_stats)
                reason=''
                if len(v)<MIN_DEFECT_POINTS:
                    reason=f'fewer than {MIN_DEFECT_POINTS} {mode} locations in {scope} region'; s['S']=None
                elif normal_stats['count']<prev.MIN_NORMAL or normal_stats['std'] is None or normal_stats['std']<prev.MIN_STD:
                    reason='insufficient normal locations or near-zero normal standard deviation'
                if threshold is None: detected=None; above=None
                elif len(v)<MIN_DEFECT_POINTS: detected=None; above=None
                else: detected=bool(np.any(v>threshold)); above=float(np.mean(v>threshold))
                width=e['width']; height=e['height']; long_side=max(width,height)
                rows.append({'image':i,'group':group(i),'instance':e['index'],'window':n,'scope':scope,'selection':mode,'width_px':width,'height_px':height,'long_side_px':long_side,'short_side_px':min(width,height),'annotation_area_px2':e['area'],'bbox_area_px2':width*height,'size_ratio':long_side/n,'count':len(v),'median':s['median'],'maximum':s['max'],'p95':s['p95'],'normal_count':normal_stats['count'],'normal_median':normal_stats['median'],'normal_std':normal_stats['std'],'normal_p99':normal_stats['p99'],'S':s['S'],'normal_p99_exceed_fraction':s['normal_p99_exceed_rate'],'loio_threshold':threshold,'loio_exceed_fraction':above,'loio_detected':detected,'reason':reason})
        summary={'image':i,'group':group(i),'window':n,'scope':scope,'threshold':threshold,'normal_count':len(normal),'normal_mean':normal_stats['mean'],'normal_median':normal_stats['median'],'normal_std':normal_stats['std'],'normal_p95':normal_stats['p95'],'normal_p99':normal_stats['p99'],'normal_exceed_fraction':float(np.mean(normal>threshold)) if threshold is not None and len(normal) else None,'training_image_count':sum(len(a)>=prev.MIN_NORMAL for a in same),'training_policy':'equal image weight, same group; held-out image excluded'}
        local=[e for e in item['entries'] if e['label']=='局所変形']
        for mode in ('center','overlap'):
            union=np.logical_or.reduce([e[mode] for e in local]) if local else np.zeros(item['r']['scores'].shape,bool)
            points=val(item['r'],union & item['region'])
            selected=[r for r in rows if r['image']==i and r['window']==n and r['scope']==scope and r['selection']==mode]
            summary[f'{mode}_point_count']=len(points)
            summary[f'{mode}_point_median']=float(np.median(points)) if len(points) else None
            summary[f'{mode}_point_exceed_fraction']=float(np.mean(points>threshold)) if threshold is not None and len(points) else None
            summary[f'{mode}_detected_instances']=sum(r['loio_detected'] is True for r in selected)
            summary[f'{mode}_missed_instances']=sum(r['loio_detected'] is False for r in selected)
            summary[f'{mode}_unassessable_instances']=sum(r['loio_detected'] is None for r in selected)
        threshold_rows.append(summary)
    return rows,threshold_rows


def pearson_spearman(x,y):
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float); valid=np.isfinite(x)&np.isfinite(y); x=x[valid]; y=y[valid]
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0: return len(x),None,None
    return len(x),float(pearsonr(x,y).statistic),float(spearmanr(x,y).statistic)


def derived(rows):
    focused=[r for r in rows if r['group']=='grid' and r['scope']=='common' and r['selection']=='center' and r['S'] is not None]
    all_focused=[r for r in rows if r['group'] in ('grid','no_grid') and r['scope']=='common' and r['selection']=='center' and r['S'] is not None]
    keys=sorted({(r['image'],r['instance']) for r in all_focused}); opt=[]
    for image,instance in keys:
        group_rows=[r for r in all_focused if r['image']==image and r['instance']==instance]
        if not group_rows: continue
        best=max(group_rows,key=lambda r:r['S']); opt.append({'image':image,'instance':instance,'group':best['group'],'long_side_px':best['long_side_px'],'area_px2':best['annotation_area_px2'],'best_window':best['window'],'best_S':best['S'],'available_windows':len(group_rows),'interpretation':'descriptive maximum; not an independent selection or a detection guarantee'})
    grid_opt=[r for r in opt if r['group']=='grid']
    corr=[]
    for n in WINDOWS:
        rs=[r for r in focused if r['window']==n]
        for field in ('long_side_px','annotation_area_px2'):
            count,p,s=pearson_spearman([r[field] for r in rs],[r['S'] for r in rs]); corr.append({'relationship':f'{field} vs S','window':n,'N':count,'pearson':p,'spearman':s,'note':'descriptive only; instances from one image may not be independent'})
    for name,x,y in [('long_side vs best_window',[r['long_side_px'] for r in grid_opt],[r['best_window'] for r in grid_opt]),('size_ratio vs S (repeated instances)',[r['size_ratio'] for r in focused],[r['S'] for r in focused])]:
        count,p,s=pearson_spearman(x,y); corr.append({'relationship':name,'window':'all','N':count,'pearson':p,'spearman':s,'note':'descriptive only; repeated windows not independent; no significance test'})
    return focused,opt,corr


def plot_summary(summary,focused,opt,thresholds,timing):
    colors=dict(zip(WINDOWS,['#4c78a8','#f58518','#54a24b','#b279a2']))
    opt=[r for r in opt if r['group']=='grid']
    fig,ax=plt.subplots(figsize=(9,6))
    for n in WINDOWS:
        rs=[r for r in focused if r['window']==n]; ax.scatter([r['long_side_px'] for r in rs],[r['S'] for r in rs],label=f'{n} px',color=colors[n],alpha=.8)
    ax.set(xlabel='Defect long side (px)',ylabel='S'); ax.legend(); fig.savefig(summary/'defect_size_vs_score.png',dpi=140,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6))
    for key in sorted({(r['image'],r['instance']) for r in focused}):
        rs=sorted([r for r in focused if (r['image'],r['instance'])==key],key=lambda x:x['window']); ax.plot([r['window'] for r in rs],[r['S'] for r in rs],marker='o',label=f'image{key[0]}:{key[1]}')
    ax.set(xlabel='Window size (px)',ylabel='S',xticks=WINDOWS); ax.legend(fontsize=7,ncol=2); fig.savefig(summary/'window_size_vs_score.png',dpi=140,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(); ax.scatter([r['long_side_px'] for r in opt],[r['best_window'] for r in opt]); ax.set(xlabel='Defect long side (px)',ylabel='Window with largest S (px)',yticks=WINDOWS); fig.savefig(summary/'optimal_window_vs_defect_size.png',dpi=140,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots()
    for n in WINDOWS:
        rs=[r for r in focused if r['window']==n]; ax.scatter([r['size_ratio'] for r in rs],[r['S'] for r in rs],color=colors[n],label=f'{n} px')
    ax.set(xlabel='Defect long side / window',ylabel='S'); ax.legend(); fig.savefig(summary/'size_ratio_vs_score.png',dpi=140,bbox_inches='tight'); plt.close(fig)
    keys=sorted({(r['image'],r['instance']) for r in focused},key=lambda k: next(r['long_side_px'] for r in focused if (r['image'],r['instance'])==k))
    mat=np.full((len(keys),len(WINDOWS)),np.nan)
    for r in focused: mat[keys.index((r['image'],r['instance'])),WINDOWS.index(r['window'])]=r['S']
    fig,ax=plt.subplots(figsize=(8,max(5,len(keys)*.35))); im=ax.imshow(mat,aspect='auto',cmap='viridis'); ax.set_xticks(range(4),WINDOWS); ax.set_yticks(range(len(keys)),[f'{i}:{j}' for i,j in keys]); ax.set(xlabel='Window size (px)',ylabel='Image : annotation index'); fig.colorbar(im,ax=ax,label='S'); fig.savefig(summary/'defect_window_heatmap.png',dpi=140,bbox_inches='tight'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for n in WINDOWS:
        rs=[r for r in focused if r['window']==n and r['loio_detected'] is not None]
        hits=sum(r['loio_detected'] for r in rs); axes[0].bar(str(n),hits/len(rs) if rs else 0,color=colors[n]); axes[0].text(str(n),hits/len(rs) if rs else 0,f'{hits}/{len(rs)}',ha='center',va='bottom')
        normals=[r['normal_exceed_fraction'] for r in thresholds if r['window']==n and r['scope']=='common' and r['group']=='grid' and r['normal_exceed_fraction'] is not None]
        axes[1].bar(str(n),np.mean(normals) if normals else 0,color=colors[n])
    axes[0].set(xlabel='Window (px)',ylabel='LOIO detected fraction',ylim=(0,1.1)); axes[1].set(xlabel='Window (px)',ylabel='Mean normal exceedance'); fig.savefig(summary/'detection_by_window.png',dpi=140,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots()
    for n in WINDOWS:
        vals=[r['total_seconds'] for r in timing if r['window']==n and r['group']=='grid']; ax.bar(str(n),np.median(vals) if vals else 0,color=colors[n])
    ax.set(xlabel='Window (px)',ylabel='Median analysis time (s)'); fig.savefig(summary/'processing_time_comparison.png',dpi=140,bbox_inches='tight'); plt.close(fig)


def spacing_experiment(args,data,focused,baseline_rows):
    # Additional test uses the better LOIO detection count of 192/256 px, chosen on this dataset.
    def hits(n): return sum(r['loio_detected'] is True for r in focused if r['window']==n)
    n=max((192,256),key=lambda window:(hits(window),-window)); collected=[]
    for r in baseline_rows:
        if r['window']==n: collected.append({'reference_spacing':128,**r})
    prev.N=n
    for spacing in (192,256):
        prev.SPACING=spacing; changed={}
        for i in range(1,22):
            if (i,n,'full') not in data: continue
            folder=args.output/f'image{i}'; p=folder/f'window_{n}.npz'; r=load_result(p,features=True)
            extra=folder/f'window_{n}_spacing_{spacing}.npz'
            if extra.exists():
                with np.load(extra) as z: r['scores']=z['anomaly_map'].copy(); r['counts']=z['reference_counts'].copy()
            else:
                r['scores'],r['counts'],_,_=prev.neighbor_features(r['spectra'])
                np.savez_compressed(extra,anomaly_map=r['scores'],reference_counts=r['counts'])
            del r['spectra']
            shape=np.asarray(Image.open(args.data/f'image{i}.png').convert('L')).shape
            labels=prev.annotation((args.data/f'image{i}.png').with_suffix('.json'),shape)
            normal,entries=prev.classification(labels,r)
            if normal is None: continue
            for scope in ('full','common'):
                if (i,n,scope) not in data: continue
                region=data[(i,n,scope)]['region']
                changed[(i,n,scope)]={'r':r,'normal':val(r,normal & region),'entries':entries,'region':region}
        spacing_rows,_=calculate_metrics(changed)
        collected.extend({'reference_spacing':spacing,**r} for r in spacing_rows)
        print(f'Additional reference spacing {spacing}px: complete',flush=True)
    prev.SPACING=128
    return n,collected


def summarize(args):
    summary=base.unique_dir(args.output/'summary')
    data,status,timing,bin_rows=build_data(args); rows,thresholds=calculate_metrics(data); focused,opt,corr=derived(rows)
    prev.safe_csv(summary/'defect_window_metrics.csv',rows); prev.safe_csv(summary/'optimal_window_per_defect.csv',opt)
    prev.safe_csv(summary/'loio_metrics.csv',thresholds); prev.safe_csv(summary/'correlation_results.csv',corr)
    prev.safe_csv(summary/'processing_times.csv',timing); prev.safe_csv(summary/'analysis_status.csv',status); prev.safe_csv(summary/'frequency_bins.csv',bin_rows)
    plot_summary(summary,focused,opt,thresholds,timing)
    lines=['# 局所変形サイズとFFT窓サイズの比較', '', '## 目的・方法', '同一の局所変形インスタンスを64・128・192・256 pxで比較し、サイズ依存性を検討する。窓以外はstride 16 px、3×3参照（中央を除く最大8、最低3）、間隔128 px、0.01〜0.50 cycles/pixel、半径ビン幅0.01で固定。元コードの片側FFT範囲総パワー正規化、共役重み付き線形放射平均、dB化、参照中央値とRMSを使用。', '正常領域は窓全体が全アノテーションと非重複、未ラベル領域を正常と仮定。中心と重なりを別評価する。正常窓100未満、正常標準偏差1e-6未満、傷の評価窓4未満はSを評価不能とした。4窓共通の画像中心座標範囲での結果を主とし、各窓の全範囲もCSVに保存。', 'LOIOは評価画像を除いた同グループ正常窓の画像均等重み付き99%点をfoldごとに決める。各foldでは一つの閾値だが、fold間で変わる。検出は傷中心窓の1点以上が閾値を超えることと定義。窓の重なりによる検出は札等に反応する可能性があり、別に扱う。', '窓ごとに周波数分解能と有効ビンが異なる。64 pxは1/64=0.015625 cycles/pixelで、0.01未満・一部のビンは観測できず補間しない。frequency_bins.csvに周波数ビン・個数を保存。192・256 px窓は128 px離れた参照と重なる。', 'image1・19〜21の格子なし画像は別集計、image18は小画像として条件別に扱い、主解析・LOIO閾値学習には含めない。image5は条件選択に使っており完全な独立検証ではない。', '', '## 解析可否']
    for n in WINDOWS:
        ok=[r for r in status if r.get('window')==n and r['status']=='analyzed']; bad=[r for r in status if r.get('window')==n and r['status']!='analyzed']
        lines.append(f"- {n} px: {len(ok)}画像解析。除外・未処理: "+(', '.join(f"image{r['image']} ({r['reason']})" for r in bad) or 'なし'))
    lines+=['','## 格子あり画像の共通座標・中心評価']
    for n in WINDOWS:
        rs=[r for r in focused if r['window']==n]; hits=sum(r['loio_detected'] is True for r in rs); evaluable=sum(r['loio_detected'] is not None for r in rs)
        normals=[r['normal_exceed_fraction'] for r in thresholds if r['window']==n and r['scope']=='common' and r['group']=='grid' and r['normal_exceed_fraction'] is not None]
        lines.append(f"- {n} px: S中央値 {np.median([r['S'] for r in rs]):.3f}（{len(rs)}傷）、LOIO検出 {hits}/{evaluable}、正常窓超過率の画像平均 {np.mean(normals):.2%}。" if rs and normals else f'- {n} px: 評価可能なデータなし。')
    lines+=['','## 傷ごとの最大S（探索的）']
    for r in opt: lines.append(f"- {r['group']} image{r['image']}:{r['instance']}、長辺 {r['long_side_px']} px：最大S {r['best_S']:.3f}（{r['best_window']} px）。")
    lines+=['','## 相関（記述統計）']
    for r in corr:
        if r['pearson'] is not None: lines.append(f"- {r['relationship']} / {r['window']}: n={r['N']}, Pearson {r['pearson']:.3f}, Spearman {r['spearman']:.3f}。")
    if focused:
        best_counts={n:sum(r['best_window']==n and r['group']=='grid' for r in opt) for n in WINDOWS}
        detections={n:sum(r['loio_detected'] is True for r in focused if r['window']==n) for n in WINDOWS}
        union={(r['image'],r['instance']) for r in focused if r['loio_detected'] is True}
        lines+=['','## 観測された窓サイズの違い',f'傷ごとの事後的な最大Sの窓数: {best_counts}。固定窓のLOIO中心検出数: {detections}。4条件のいずれかで閾値超となる傷は {len(union)} 件（OR統合の誤検出率は未校正）。', '異なる窓が最大Sとなること自体はマルチスケールの検討材料だが、傷ごとの事後選択は未知画像の検出率ではない。傷の大きさだけで窓を決める規則や、単調なサイズ依存をこれだけで確定できない。']
        if len(union)>max(detections.values()):
            lines.append('結論：傷の大きさに合わせて窓を変更する規則は、現時点では採用しない。研究の次候補はB（複数窓の統合）とし、256 px単独を基準に比較する。ただしOR統合の誤検出率は未校正であり、Bの実用上の優位性は未実証。')
        else:
            lines.append('結論：傷の大きさに合わせて窓を変更する規則を採用する根拠はない。まず固定窓を基準に独立画像で再検証し、複数窓の統合は誤検出率を含めた追加実験で判断する。')
    if args.extra_spacing and focused:
        chosen,extra=spacing_experiment(args,data,focused,rows)
        prev.safe_csv(summary/'reference_spacing_metrics.csv',extra)
        lines+=['','## 追加実験：参照間隔',f'192・256 px窓のうち同データでLOIO検出数が多かった {chosen} pxを探索的に選び、参照間隔128・192・256 pxを比較。既存のFFT特徴量を再利用し、周辺中央値とスコアだけ再計算。条件選択自体は独立検証ではない。']
        for spacing in (128,192,256):
            rs=[r for r in extra if r['group']=='grid' and r['scope']=='common' and r['selection']=='center' and r['reference_spacing']==spacing and r['S'] is not None]
            hits=sum(r['loio_detected'] is True for r in rs); count=sum(r['loio_detected'] is not None for r in rs)
            lines.append(f"- 間隔 {spacing} px: S中央値 {np.median([r['S'] for r in rs]):.3f}、LOIO中心検出 {hits}/{count}。" if rs else f'- 間隔 {spacing} px: 評価不能。')
        lines.append('参照間隔を広げると重複だけでなく参照する画像位置と有効な窓数も変わるため、差を重複だけの因果効果とは解釈できない。')
    lines+=['','## 計算時間・メモリ',f'処理時間は画像読込・描画・保存を含まない解析時間。256 pxは既存の同条件結果を再利用した場合がある（CSVのorigin列）。ワーカー数{args.workers}、batch {args.batch_size}。メモリはWindowsプロセス生涯のピーク作業セットで、同一プロセスで窓を順番に実行した場合は各窓独立のピークではない。','', '## 解釈・今後', '最大Sの窓は評価対象の正解ラベルを見て選んだ事後的な結果で、未知画像で使える固定設定の性能ではない。傷ごとの4窓は繰り返し観測であり、単純な独立標本とした有意性検定は行わない。傷数は少なく、傷の大きさ・場所・形状・照明は交絡する。中心窓や参照窓が札と重なる可能性、参照窓との重複、窓ごとの周波数分解能の違いも結果に影響しうる。', '固定スケールのLOIO結果と、傷ごとの事後的な最大Sのばらつきを比較し、マルチスケール統合を検討する。未知画像の傷サイズを使わず、訓練画像だけから各窓スコアを標準化し、最大値などの統合と閾値を訓練データ内で決定したうえで、画像単位の外側LOIOで評価する。統合後の誤検出と欠陥検出を別途検証する必要がある。']
    (summary/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Summary:',summary,flush=True)
    return summary,rows,thresholds,opt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=base.DATA); p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--workers',type=int,default=4); p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--summary-only',action='store_true')
    p.add_argument('--no-extra-spacing',dest='extra_spacing',action='store_false',default=True)
    args=p.parse_args()
    if args.workers<1 or args.batch_size<1: p.error('workers and batch size must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    settings={'data':str(args.data.resolve()),'windows':list(WINDOWS),'stride':16,'spacing':128,'min_references':3,'frequency':[.01,.5],'bin_width':.01,'workers':args.workers,'batch_size':args.batch_size,'minimum_defect_points':MIN_DEFECT_POINTS}
    path=args.output/'settings.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=settings: p.error('Output directory has different settings; use a new --output directory')
    else: path.write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
    if not args.summary_only:
        for i in range(1,22): process_image(i,args)
    summarize(args)


if __name__=='__main__': main()
