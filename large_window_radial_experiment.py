#!/usr/bin/env python3
"""Fixed 256/320/384 radial FFT comparison with 128/384 reference spacing."""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import fft
import multiscale_fft_experiment as base
import evaluate_all_radial_fft as prev
import radial_fft_experiment as radial

OUT=base.ROOT/'output_large_window_radial'
SOURCE=base.ROOT/'output_all_images_radial_fft'
SIZES=(256,320,384); SPACINGS=(128,384); IMAGES=tuple(range(2,18))


def feature_data(path):
    with np.load(path) as z:
        return {'xs':z['x_centers'].copy(),'ys':z['y_centers'].copy(),'scores':z['anomaly_map'].copy(),'counts':z['reference_counts'].copy(),'spectra':z['radial_spectra'].copy(),'bin_counts':z['bin_counts'].copy(),'bin_valid':z['bin_valid'].copy()}


def process(i,args):
    image=args.data/f'image{i}.png'; folder=args.output/f'image{i}'; folder.mkdir(parents=True,exist_ok=True)
    if not image.exists(): print(f'image{i}: missing',flush=True); return
    img=np.asarray(Image.open(image).convert('L'),dtype=np.float32)
    try: settings=json.loads((SOURCE/'settings.json').read_text(encoding='utf-8'))
    except (FileNotFoundError,ValueError): settings={}
    source_compatible=(settings.get('window')==256 and settings.get('stride')==16 and settings.get('reference_spacing')==128 and settings.get('min_references')==3 and settings.get('frequency')==[.01,.5] and settings.get('bin_width')==.01 and settings.get('workers')==args.workers and settings.get('batch_size')==args.batch_size and Path(settings.get('data','')).resolve()==args.data.resolve())
    for n in SIZES:
        p=folder/f'window_{n}.npz'; meta=folder/f'window_{n}.json'
        if p.exists() and meta.exists(): print(f'image{i} {n}: skipped',flush=True); continue
        prev.N=n; prev.SPACING=128
        if n==256 and source_compatible and (SOURCE/f'image{i}'/'result.npz').exists():
            shutil.copy2(SOURCE/f'image{i}'/'result.npz',p)
            old=json.loads((SOURCE/f'image{i}'/'metrics.json').read_text(encoding='utf-8'))
            timing=old['timings']; origin='previous experiment'
        else:
            r=prev.analyze(img,args.workers,args.batch_size,False)
            np.savez_compressed(p,anomaly_map=r['scores'],x_centers=r['xs'],y_centers=r['ys'],reference_counts=r['counts'],radial_spectra=r['spectra'],bin_counts=r['bin_counts'],bin_valid=r['bin_valid'])
            timing={k:r[k] for k in ('fft_seconds','radial_seconds','median_seconds','score_seconds','map_seconds','total_seconds','peak_memory_bytes')}; origin='new measurement'
        meta.write_text(json.dumps({'image':i,'window':n,'origin':origin,'timings':timing},ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'image{i} window {n}: {origin} ({timing["total_seconds"]:.1f}s)',flush=True)
    prev.SPACING=128


def values(r,mask): return r['scores'][mask & np.isfinite(r['scores'])]


def common_region(r,shape):
    h,w=shape; return (r['xs'][None,:]>=192)&(r['xs'][None,:]<=w-192)&(r['ys'][:,None]>=192)&(r['ys'][:,None]<=h-192)


def verify_radial():
    rows=[]
    for n in SIZES:
        x=np.random.default_rng(n).normal(size=(n,n)).astype(np.float32)
        p=np.abs(fft.fftshift(fft.rfft2(x),axes=0))**2
        spec=radial.grids(n,.01,[(.01,.5)])[3][(.01,.5)]
        half=(p.ravel()@spec['matrix'])/np.maximum(spec['counts'],1)
        full=np.abs(fft.fft2(x))**2; f=np.fft.fftfreq(n).astype(np.float32)
        yy,xx=np.meshgrid(f,f,indexing='ij'); rr=np.sqrt(xx**2+yy**2); mask=(rr>=.01)&(rr<=.5); index=np.floor(rr/.01).astype(int)
        count=np.bincount(index[mask],minlength=len(spec['counts']))
        avg=np.bincount(index[mask],weights=full[mask],minlength=len(count))/np.maximum(count,1)
        valid=spec['valid']; error=float(np.max(np.abs(half[valid]-avg[valid])/np.maximum(avg[valid],1e-12)))
        if not np.array_equal(count,spec['counts']) or error>1e-5: raise ValueError(f'Full-FFT radial verification failed: {n}, {error}')
        rows.append({'window':n,'full_fft_count_match':True,'max_relative_error':error})
    return rows


def reference_at(r,y,x,spacing):
    k=spacing//16; ny,nx=r['scores'].shape
    blocks=[r['spectra'][y+dy*k,x+dx*k] for dy in (-1,0,1) for dx in (-1,0,1) if (dy or dx) and 0<=y+dy*k<ny and 0<=x+dx*k<nx]
    return np.median(np.stack(blocks),axis=0) if len(blocks)>=3 else None


def normal_bin_differences(r,normal,spacing):
    yy,xx=np.nonzero(normal & np.isfinite(r['scores'])); step=max(1,int(np.ceil(len(yy)/512)))
    diff=[]
    for y,x in zip(yy[::step],xx[::step]):
        ref=reference_at(r,int(y),int(x),spacing)
        if ref is not None: diff.append(np.abs(r['spectra'][y,x]-ref))
    return np.median(np.stack(diff),axis=0) if diff else np.full(int(r['bin_valid'].sum()),np.nan)


def compare13_15(args,records,labels,images,summary):
    rows=[]; fig,axes=plt.subplots(2,4,figsize=(18,8),constrained_layout=True)
    fig2,axes2=plt.subplots(2,3,figsize=(15,7),constrained_layout=True)
    fig3,axes3=plt.subplots(2,3,figsize=(15,7),constrained_layout=True)
    fig4,axes4=plt.subplots(2,3,figsize=(15,7),constrained_layout=True)
    for row,i in enumerate((13,15)):
        img=images[i]; h,w=img.shape; lab=labels[i]; local=[m for label,m in lab if label=='局所変形']
        axes[row,0].imshow(img,cmap='gray',extent=[0,w,h,0]); axes[row,0].set_title(f'image{i}: annotation')
        for mask in local: axes[row,0].contour(mask.astype(np.uint8),levels=[.5],colors='cyan',linewidths=1,origin='upper',extent=[0,w,h,0])
        for col,n in enumerate(SIZES):
            r=records[(i,n,128)]; center=base.selections(local[0],r,n)[0] & common_region(r,img.shape)
            yy,xx=np.nonzero(center & np.isfinite(r['scores']))
            if not len(yy): continue
            k=int(np.argmax(r['scores'][yy,xx])); y=int(yy[k]); x=int(xx[k]); ref=reference_at(r,y,x,128)
            fy=(np.arange(len(r['bin_valid']))+.5)*.01; fy=fy[r['bin_valid']]
            target=r['spectra'][y,x]; diff=target-ref; index=int(np.argmax(np.abs(diff)))
            px=int(r['xs'][x]); py=int(r['ys'][y]); patch=img[py-n//2:py+n//2,px-n//2:px+n//2]
            axes[row,col+1].imshow(patch,cmap='gray'); axes[row,col+1].set_title(f'{n} px: ({px},{py})')
            ax=axes2[row,col]; ax.plot(fy,target,label='Target'); ax.plot(fy,ref,label='Neighbor median'); ax.set(title=f'image{i}, {n}px',xlabel='r (cycles/pixel)',ylabel='dB'); ax.legend(fontsize=7)
            tags=[m for label,m in lab if label in ('tag','background')]
            tag_overlap=base.selections(np.logical_or.reduce(tags),r,n)[1] if tags else np.zeros(r['scores'].shape,bool)
            step=128//16; ny,nx=r['scores'].shape
            near=[(y+dy*step,x+dx*step) for dy in (-1,0,1) for dx in (-1,0,1) if (dy or dx) and 0<=y+dy*step<ny and 0<=x+dx*step<nx]
            prev.N=n; normal,_=prev.classification(lab,r); normal &= common_region(r,img.shape)
            nyy,nxx=np.nonzero(normal & np.isfinite(r['scores']))
            normal_values=r['scores'][nyy,nxx]; ni=int(np.argmin(np.abs(normal_values-np.median(normal_values))))
            ny0=int(nyy[ni]); nx0=int(nxx[ni]); normal_ref=reference_at(r,ny0,nx0,128); normal_diff=r['spectra'][ny0,nx0]-normal_ref
            nxpix=int(r['xs'][nx0]); nypix=int(r['ys'][ny0]); normal_patch=img[nypix-n//2:nypix+n//2,nxpix-n//2:nxpix+n//2]
            axes4[row,col].imshow(normal_patch,cmap='gray'); axes4[row,col].set_title(f'image{i} {n}px: normal ({nxpix},{nypix})'); axes4[row,col].set_xticks([]); axes4[row,col].set_yticks([])
            axes3[row,col].plot(fy,diff,label='Deformation'); axes3[row,col].plot(fy,normal_diff,label='Normal'); axes3[row,col].axhline(0,color='gray',lw=.5); axes3[row,col].set(title=f'image{i}, {n}px',xlabel='r (cycles/pixel)',ylabel='Target - reference (dB)'); axes3[row,col].legend(fontsize=7)
            rows.append({'image':i,'window':n,'spacing':128,'representative_x':px,'representative_y':py,'selection':'max score among defect centers','score':float(r['scores'][y,x]),'max_abs_difference_bin_center':float(fy[index]),'signed_difference_db':float(diff[index]),'target_overlaps_tag_background':bool(tag_overlap[y,x]),'reference_count':len(near),'reference_overlaps_tag_background_count':sum(bool(tag_overlap[a,b]) for a,b in near),'normal_representative_x':nxpix,'normal_representative_y':nypix,'normal_score':float(r['scores'][ny0,nx0])})
    for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(summary/'image13_image15_patches.png',dpi=120); plt.close(fig)
    fig2.savefig(summary/'image13_image15_spectra.png',dpi=130); plt.close(fig2)
    fig3.savefig(summary/'image13_image15_spectral_differences.png',dpi=130); plt.close(fig3)
    fig4.savefig(summary/'image13_image15_normal_patches.png',dpi=120); plt.close(fig4)
    prev.safe_csv(summary/'image13_image15_representatives.csv',rows)
    return rows


def build_data(args,summary):
    data={}; timing=[]; records={}; labels={}; images={}; bin_rows=[]; normal_bins=[]
    for i in IMAGES:
        imgpath=args.data/f'image{i}.png'
        if not imgpath.exists(): continue
        img=np.asarray(Image.open(imgpath).convert('L'),dtype=np.float32); lab=prev.annotation(imgpath.with_suffix('.json'),img.shape)
        if lab is None: continue
        labels[i]=lab
        if i in (13,15): images[i]=img
        anchor_path=args.output/f'image{i}'/'window_384.npz'
        if not anchor_path.exists():
            print(f'image{i}: 384 px unavailable; common-size comparison skipped',flush=True); continue
        anchor=feature_data(anchor_path); prev.N=384; prev.SPACING=384
        anchor_alt,_,_,_=prev.neighbor_features(anchor['spectra'])
        anchor_normal,_=prev.classification(lab,anchor)
        anchor_normal &= np.isfinite(anchor['scores']) & np.isfinite(anchor_alt) & common_region(anchor,img.shape)
        for n in SIZES:
            path=args.output/f'image{i}'/f'window_{n}.npz'; meta=args.output/f'image{i}'/f'window_{n}.json'
            if not path.exists() or not meta.exists(): continue
            prev.N=n; prev.SPACING=128; r=feature_data(path)
            for k,(count,valid) in enumerate(zip(r['bin_counts'],r['bin_valid'])):
                bin_rows.append({'window':n,'bin_start':k*.01,'bin_end':(k+1)*.01,'frequency_resolution':1/n,'full_fft_component_count':int(count),'used':bool(valid),'image':i})
            timing.append({'image':i,'window':n,**json.loads(meta.read_text(encoding='utf-8'))['timings']})
            prev.SPACING=384; alt_scores,alt_counts,_,_=prev.neighbor_features(r['spectra'])
            paired=np.isfinite(r['scores']) & np.isfinite(alt_scores) & common_region(r,img.shape)
            normal,entries=prev.classification(lab,r)
            iy=np.searchsorted(r['ys'],anchor['ys']); ix=np.searchsorted(r['xs'],anchor['xs'])
            if np.any(iy>=len(r['ys'])) or np.any(ix>=len(r['xs'])) or not np.array_equal(r['ys'][iy],anchor['ys']) or not np.array_equal(r['xs'][ix],anchor['xs']):
                raise ValueError('Window center grids are not aligned')
            common_normal=np.zeros(r['scores'].shape,bool); common_normal[np.ix_(iy,ix)]=anchor_normal
            for spacing in SPACINGS:
                q={**r,'scores':r['scores'] if spacing==128 else alt_scores,'counts':r['counts'] if spacing==128 else alt_counts}
                normal_mask=normal & paired & common_normal; norm=values(q,normal_mask)
                sampled=normal_bin_differences(q,normal_mask,spacing)
                freq=(np.arange(len(q['bin_valid']))+.5)*.01; freq=freq[q['bin_valid']]
                for f,d in zip(freq,sampled): normal_bins.append({'image':i,'window':n,'spacing':spacing,'frequency_bin_center':float(f),'median_absolute_target_reference_difference_db':float(d)})
                data[(i,n,spacing)]={'normal':norm,'scores':q['scores'],'entries':entries,'region':paired}
                if i in (13,15) and spacing==128: records[(i,n,128)]=q
            print(f'image{i} window {n}: scoring and paired normal spectra complete',flush=True)
            # Only image13/15 keep their spectra for representative plots.
            if i not in (13,15): del r
    prev.SPACING=128
    prev.safe_csv(summary/'normal_frequency_differences.csv',normal_bins)
    prev.safe_csv(summary/'frequency_bins.csv',bin_rows)
    prev.safe_csv(summary/'processing_times.csv',timing)
    return data,records,labels,images,timing,normal_bins


def metrics(data):
    normal_rows=[]; defects=[]
    for (i,n,spacing),v in data.items():
        train=[other['normal'] for (j,size,s),other in data.items() if j!=i and size==n and s==spacing]
        threshold=prev.weighted_threshold(train); normal=v['normal']; stats=prev.summary_stats(normal)
        local=[e for e in v['entries'] if e['label']=='局所変形']
        normal_rows.append({'image':i,'window':n,'spacing':spacing,'normal_count':stats['count'],'normal_mean':stats['mean'],'normal_median':stats['median'],'normal_std':stats['std'],'normal_p95':stats['p95'],'normal_p99':stats['p99'],'normal_max':stats['max'],'loio_threshold':threshold,'normal_loio_exceed_fraction':float(np.mean(normal>threshold)) if threshold is not None and len(normal) else None})
        for e in local:
            for mode in ('center','overlap'):
                selected=e[mode]&v['region']&np.isfinite(v['scores']); points=v['scores'][selected]; d=prev.defect_stats(points,stats)
                usable=len(points)>=4 and d['S'] is not None
                defects.append({'image':i,'window':n,'spacing':spacing,'instance':e['index'],'selection':mode,'long_side':max(e['width'],e['height']),'point_count':len(points),'median':d['median'],'p95':d['p95'],'max':d['max'],'normal_median':stats['median'],'normal_std':stats['std'],'S':d['S'] if usable else None,'own_normal_p99_exceed_fraction':d['normal_p99_exceed_rate'] if usable else None,'loio_threshold':threshold,'loio_point_exceed_fraction':float(np.mean(points>threshold)) if usable and threshold is not None else None,'loio_detected':bool(np.any(points>threshold)) if usable and threshold is not None else None,'reason':'' if usable else 'fewer than 4 defect points or insufficient normal data'})
    return normal_rows,defects


def plot_results(summary,data,normal_rows,defects,normal_bins):
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,spacing in zip(axes,SPACINGS):
        vals=[[r['normal_std'] for r in normal_rows if r['window']==n and r['spacing']==spacing and r['normal_std'] is not None] for n in SIZES]
        ax.boxplot(vals,labels=[str(n) for n in SIZES]); ax.set(title=f'Reference spacing {spacing} px',xlabel='Window size (px)',ylabel='Normal score standard deviation (dB)')
    fig.savefig(summary/'normal_variability.png',dpi=140,bbox_inches='tight'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    for ax,spacing in zip(axes,SPACINGS):
        samples=[]; positions=[]
        for col,n in enumerate(SIZES):
            pooled_normal=[]; pooled_defect=[]
            for (i,window,sp),entry in data.items():
                if window!=n or sp!=spacing: continue
                normal=entry['normal']
                if len(normal): pooled_normal.extend(normal[np.linspace(0,len(normal)-1,min(256,len(normal)),dtype=int)])
                local=[e for e in entry['entries'] if e['label']=='局所変形']
                if local:
                    mask=np.logical_or.reduce([e['center'] for e in local]) & entry['region'] & np.isfinite(entry['scores'])
                    v=entry['scores'][mask]
                    if len(v): pooled_defect.extend(v[np.linspace(0,len(v)-1,min(256,len(v)),dtype=int)])
            samples.extend([pooled_normal,pooled_defect]); positions.extend([col*3, col*3+1])
        ax.boxplot(samples,positions=positions,widths=.7,showfliers=False)
        ax.set_xticks([.5,3.5,6.5],[str(n) for n in SIZES]); ax.set(title=f'Spacing {spacing}px',xlabel='Window px (left normal, right deformation)',ylabel='RMS difference (dB)')
    fig.suptitle('Up to 256 points per image and class; spatial windows overlap')
    fig.savefig(summary/'score_distributions_all_images.png',dpi=130); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for row,i in enumerate((13,15)):
        for col,n in enumerate(SIZES):
            v=data[(i,n,128)]; norm=v['normal']; local=[e for e in v['entries'] if e['label']=='局所変形']; ax=axes[row,col]
            if local:
                union=np.logical_or.reduce([e['center'] for e in local]) & v['region'] & np.isfinite(v['scores']); defect=v['scores'][union]
                if len(norm): ax.hist(norm,bins=55,density=True,alpha=.55,label='Normal')
                if len(defect): ax.hist(defect,bins=25,density=True,alpha=.55,label='Deformation center')
                ax.legend(fontsize=7)
            ax.set(title=f'image{i} {n}px',xlabel='RMS dB',ylabel='Density')
    fig.savefig(summary/'image13_image15_distributions.png',dpi=130); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,spacing in zip(axes,SPACINGS):
        for n in SIZES:
            vals=[r for r in normal_bins if r['window']==n and r['spacing']==spacing]
            freqs=sorted({r['frequency_bin_center'] for r in vals}); med=[np.median([r['median_absolute_target_reference_difference_db'] for r in vals if r['frequency_bin_center']==f]) for f in freqs]
            ax.plot(freqs,med,label=f'{n}px')
        ax.set(title=f'Spacing {spacing}px',xlabel='r (cycles/pixel)',ylabel='Median |target - reference| (dB)'); ax.legend()
    fig.savefig(summary/'normal_frequency_difference.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,spacing in zip(axes,SPACINGS):
        for n in SIZES:
            d=[r for r in defects if r['window']==n and r['spacing']==spacing and r['selection']=='center' and r['loio_detected'] is not None]
            hit=sum(r['loio_detected'] for r in d); ax.bar(str(n),hit/len(d) if d else 0); ax.text(str(n),hit/len(d) if d else 0,f'{hit}/{len(d)}',ha='center',va='bottom')
        ax.set(title=f'Spacing {spacing}px',xlabel='Window (px)',ylabel='LOIO detection fraction',ylim=(0,1.1))
    fig.savefig(summary/'loio_detection.png',dpi=130,bbox_inches='tight'); plt.close(fig)


def report(args,summary,data,normal,defects,timing,reps):
    lines=['# 大きいFFT窓と正常表面の安定性', '', '## 条件と評価', '格子ライトありのimage2〜image17を対象。256・320・384 px、stride 16 px、周辺3×3（中央除外・最低3参照）、周波数0.01〜0.50 cycles/pixel、放射ビン幅0.01、FFTワーカー4、バッチ64。画像のリサイズはしない。256 pxは既存の同条件結果を使用した場合がある。', '2次元FFTパワーを従来どおり片側パワー総和で正規化し、共役の重みを付けた線形放射平均をdB化。各ビンの周辺中央値との差のRMSをスコアとする。ビンの有無と成分数はfrequency_bins.csvに保存。', '参照間隔128 pxと384 pxを別々に実施。384 px間隔は3種類すべてで対象と参照の窓領域が重ならない。中心座標は384 px窓の共通範囲に限定し、両参照間隔とも参照3点以上の地点だけを対象にする。参照間隔変更は参照領域の位置も変えるため、重複の因果効果だけを分離できない。', '正常評価には384 px窓が全ラベルに重ならない中心を用い、同じ画像座標の窓を3サイズで比較する。未ラベルは正常と仮定。変形は中心と窓の重なりを別に保存。Sは正常窓100点以上・標準偏差1e-6以上・変形窓4点以上で計算。LOIOは評価画像を除く正常窓を画像均等重みで集計した99%点をfold単位で使用し、少なくとも1つの変形中心窓が閾値を超えれば傷インスタンスを検出とする。image5は条件選択に使用済みで独立検証ではない。', '正常周波数差は各画像・条件で正常窓を走査順に等間隔で最大512点抽出し、各点の周辺中央値との差の絶対値をビンごとに集計した記述統計である。無作為標本や独立標本とはみなさない。', '', '## 正常領域・LOIOの結果']
    for spacing in SPACINGS:
        lines.append(f'### 参照間隔 {spacing} px')
        for n in SIZES:
            nr=[r for r in normal if r['window']==n and r['spacing']==spacing and r['normal_std'] is not None]
            dr=[r for r in defects if r['window']==n and r['spacing']==spacing and r['selection']=='center' and r['S'] is not None]
            if not nr: lines.append(f'- {n} px: 評価不能'); continue
            fpr=[r['normal_loio_exceed_fraction'] for r in nr if r['normal_loio_exceed_fraction'] is not None]
            hits=sum(r['loio_detected'] is True for r in dr); total=sum(r['loio_detected'] is not None for r in dr)
            lines.append(f"- {n} px：正常スコア標準偏差の画像中央値 {np.median([r['normal_std'] for r in nr]):.3f} dB、変形Sの中央値 {np.median([r['S'] for r in dr]):.3f}、LOIO検出 {hits}/{total}、正常超過率の画像平均 {np.mean(fpr):.2%}。")
    lines+=['','## image13 と image15']
    for i in (13,15):
        for n in SIZES:
            nr=next(r for r in normal if r['image']==i and r['window']==n and r['spacing']==128)
            dr=[r for r in defects if r['image']==i and r['window']==n and r['spacing']==128 and r['selection']=='center']
            if not dr: continue
            d=dr[0]; rep=next((r for r in reps if r['image']==i and r['window']==n),None)
            lines.append(f"- image{i}、{n} px：正常中央値 {nr['normal_median']:.3f} dB、標準偏差 {nr['normal_std']:.3f}、変形中央値 {d['median']:.3f}、S {d['S']:.3f}、LOIO検出 {d['loio_detected']}。"+(f" 代表変形点 ({rep['representative_x']}, {rep['representative_y']}) の最大スペクトル差ビン {rep['max_abs_difference_bin_center']:.3f} cycles/pixel、差 {rep['signed_difference_db']:+.3f} dB。対象窓の札・背景重複 {rep['target_overlaps_tag_background']}、参照窓の札・背景重複 {rep['reference_overlaps_tag_background_count']}/{rep['reference_count']}。" if rep else ''))
    lines += ['', '## 解釈と限界', '正常スコアの標準偏差が小さい場合、正常表面の「この異常度」の画像内ばらつきが小さいことを示す。窓を大きくすると含まれる模様や周波数分解能、使用するビン、同じ変形が参照窓に混入する程度も変わるため、その減少を単独の原因に帰することはできない。normal_frequency_difference.pngは周波数別の差を確認するためのものであり、因果的な分解ではない。', 'image13・15の代表点は各変形中心のスコア最大点として選択しており、代表点同士の差や最大周波数だけから、変形の物理的原因や格子の方向変化を断定できない。選択した窓と参照窓が札や他の構造を含むか、局所画像・スペクトル図と併せて確認する。', '各ウィンドウのSとLOIOの検出を併せて評価する。Sや画像内正常99%点は未知画像に直接適用できる閾値ではない。重なり窓は札・背景を同時に含む可能性があり、検出位置の一致を保証しない。大きな窓やマルチスケールの優劣は、追加画像と正常誤検出の検証なしに結論づけない。', '計算時間は入力読込・図保存を除く。メモリはプロセス生涯のピーク作業セットであり、同一プロセスで複数窓を実行した場合は窓ごとに独立していない。']
    (summary/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def summarize(args):
    summary=base.unique_dir(args.output/'summary'); prev.safe_csv(summary/'full_fft_verification.csv',verify_radial())
    data,records,labels,images,timing,normal_bins=build_data(args,summary)
    normal,defects=metrics(data); prev.safe_csv(summary/'normal_metrics.csv',normal); prev.safe_csv(summary/'defect_metrics.csv',defects)
    plot_results(summary,data,normal,defects,normal_bins)
    reps=compare13_15(args,records,labels,images,summary)
    report(args,summary,data,normal,defects,timing,reps)
    print('Summary:',summary,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=base.DATA); p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--workers',type=int,default=4); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--summary-only',action='store_true')
    args=p.parse_args()
    if args.workers!=4 or args.batch_size!=64: p.error('This comparison fixes workers=4 and batch size=64')
    args.output.mkdir(parents=True,exist_ok=True)
    settings={'images':list(IMAGES),'windows':list(SIZES),'spacings':list(SPACINGS),'stride':16,'frequency':[.01,.5],'bin_width':.01,'workers':args.workers,'batch_size':args.batch_size,'data':str(args.data.resolve())}
    path=args.output/'settings.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=settings: p.error('Existing output has different settings; choose a new directory')
    if not path.exists(): path.write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
    if not args.summary_only:
        for i in IMAGES: process(i,args)
    summarize(args)


if __name__=='__main__': main()
