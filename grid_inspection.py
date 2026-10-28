"""Image-independent periodic-grid inspection pipeline. See GRID_INSPECTION.md."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from scipy.ndimage import (binary_closing, binary_fill_holes, distance_transform_edt,
                           gaussian_filter, label, map_coordinates, maximum_filter,
                           uniform_filter)
from scipy.optimize import least_squares, root

NAMES = ['黒の交点', '白の中央', '黒1本・方向1', '黒1本・方向2']
KEYS = ['black_cross', 'white_center', 'black_single_1', 'black_single_2']
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']


@dataclass
class Config:
    max_dimension: int = 1024
    min_peak_fraction: float = .025
    min_combined_fraction: float = .15
    min_coherence: float = .38
    min_amplitude_fraction: float = .28
    crop_fraction: float = .25
    crop_size: int | None = None
    min_period: float | None = None
    max_period: float | None = None
    figures: bool = True


class GridNotFound(Exception):
    pass


def at(field, xy):
    return float(map_coordinates(field, [[xy[1]], [xy[0]]], order=1, mode='nearest')[0])


def fft_candidates(data, scale_xy, config):
    h,w=data.shape
    transform=np.fft.fftshift(np.fft.fft2(data*np.outer(np.hanning(h),np.hanning(w))))
    power=abs(transform)**2
    fx=np.fft.fftshift(np.fft.fftfreq(w));fy=np.fft.fftshift(np.fft.fftfreq(h))
    xx,yy=np.meshgrid(fx,fy)
    radius=np.hypot(xx,yy)
    scale=float(np.mean(scale_xy))
    low=1/(config.max_period*scale) if config.max_period else 3/min(h,w)
    high=1/(config.min_period*scale) if config.min_period else 1/6
    band=(radius>low)&(radius<high)
    if high<=low or not band.any(): raise GridNotFound('周期の探索範囲が画像サイズに対して不適切です。')
    peak_mask=(power==maximum_filter(power,5))&band&(yy>0)
    peaks=sorted(np.argwhere(peak_mask),key=lambda p:power[tuple(p)],reverse=True)
    if not peaks or power[band].sum()<=1e-12: raise GridNotFound('周期成分がありません。')
    chosen=[];vectors=[]
    for y,x in peaks:
        v=np.array([fx[x],fy[y]])
        if vectors:
            angle=abs(np.linalg.det(np.stack([vectors[0],v])))/(np.linalg.norm(v)*np.linalg.norm(vectors[0]))
            ratio=np.linalg.norm(v)/np.linalg.norm(vectors[0])
            if angle<.65 or not(.5<ratio<2):continue
        chosen.append((int(y),int(x)));vectors.append(v)
        if len(vectors)==2:break
    if len(vectors)<2:raise GridNotFound('独立した2方向の周期成分が見つかりません。')
    fractions=[]
    for y,x in chosen:
        patch=power[max(y-2,0):y+3,max(x-2,0):x+3]
        fractions.append(float(2*patch.sum()/power[band].sum()))
    diagnostic=dict(seed_vectors=np.array(vectors).tolist(),peak_fractions=fractions,
                    search_frequency_range=[float(low),float(high)])
    accepted=min(fractions)>=config.min_peak_fraction and sum(fractions)>=config.min_combined_fraction
    return np.array(vectors),power,fx,fy,diagnostic,accepted


def refine_vectors(data,seeds):
    h,w=data.shape;yy,xx=np.indices(data.shape)
    stride=max(2,int(np.sqrt(data.size/18000)))
    sx=xx[::stride,::stride].ravel()/w-.5;sy=yy[::stride,::stride].ravel()/h-.5
    values=data[::stride,::stride].ravel()
    q=(seeds*np.array([w,h])).ravel()
    def design(v):
        a=2*np.pi*(v[0]*sx+v[1]*sy);b=2*np.pi*(v[2]*sx+v[3]*sy)
        return np.column_stack([np.cos(a),np.sin(a),np.cos(b),np.sin(b),np.ones_like(a)])
    coeff=np.linalg.lstsq(design(q),values,rcond=None)[0]
    fit=least_squares(lambda p:design(p[:4])@p[4:]-values,np.r_[q,coeff],
                      bounds=(np.r_[q-.9,[-np.inf]*5],np.r_[q+.9,[np.inf]*5]),
                      loss='soft_l1',f_scale=max(2,np.std(values)*.3),max_nfev=65)
    return fit.x[:4].reshape(2,2)/[w,h],bool(fit.success)


def local_grid(data,vectors,valid=None):
    yy,xx=np.indices(data.shape)
    period=float(np.median(1/np.linalg.norm(vectors,axis=1)))
    sigma=max(2,period*.35)
    components=[];amplitudes=[];phases=[];carriers=[]
    weight=np.ones_like(data) if valid is None else valid.astype(float)
    denominator=gaussian_filter(weight,sigma)
    for v in vectors:
        carrier=2*np.pi*(v[0]*xx+v[1]*yy)
        demod=data*weight*np.exp(-1j*carrier)
        env=(gaussian_filter(demod.real,sigma)+1j*gaussian_filter(demod.imag,sigma))/np.maximum(denominator,.01)
        phases.append(carrier+np.unwrap(np.unwrap(np.angle(env),axis=1),axis=0))
        amplitudes.append(2*abs(env));components.append(2*np.real(env*np.exp(1j*carrier)));carriers.append(carrier)
    energy=gaussian_filter(data**2,sigma)
    coherent=(amplitudes[0]**2+amplitudes[1]**2)/2
    coherence=np.clip(coherent/np.maximum(energy,1),0,1)
    return period,phases,amplitudes,components,coherence,carriers


def analysis_mask(amplitudes,coherence,period,config):
    thresholds=[max(1.5,float(np.percentile(a,85))*config.min_amplitude_fraction) for a in amplitudes]
    mask=(amplitudes[0]>thresholds[0])&(amplitudes[1]>thresholds[1])&(coherence>config.min_coherence)
    mask=binary_closing(mask,iterations=max(1,round(period*.04)))
    mask=binary_fill_holes(mask)
    labels,n=label(mask);sizes=np.bincount(labels.ravel())
    keep=np.flatnonzero(sizes>max(64,period**2*2));keep=keep[keep!=0]
    mask=np.isin(labels,keep) if len(keep) else np.zeros_like(mask)
    # Remove the uncertain transition where demodulation can leak across paper edges.
    mask=distance_transform_edt(mask)>period*.10
    return mask,thresholds


def find_centers(phases,carriers,vectors,mask,scale_xy,original_mask,crop_size):
    h,w=mask.shape
    offsets=np.array([np.median((p-c)[mask]) for p,c in zip(phases,carriers)])
    matrix=2*np.pi*vectors
    rows=[];seen=set()
    sx,sy=scale_xy
    for category,phase_offset in enumerate([(np.pi,np.pi),(0,0),(np.pi,0),(0,np.pi)]):
        ranges=[range(int(np.floor((p[mask].min()-off)/(2*np.pi)))-1,int(np.ceil((p[mask].max()-off)/(2*np.pi)))+2) for p,off in zip(phases,phase_offset)]
        for i in ranges[0]:
            for j in ranges[1]:
                wanted=2*np.pi*np.array([i,j])+phase_offset
                seed=np.linalg.solve(matrix,wanted-offsets)
                if not(0<seed[0]<w-1 and 0<seed[1]<h-1):continue
                result=root(lambda xy:np.array([at(p,xy) for p in phases])-wanted,seed)
                x,y=result.x
                if not result.success or not(0<x<w-1 and 0<y<h-1) or not mask[round(y),round(x)]:continue
                if np.linalg.norm([at(p,result.x)-t for p,t in zip(phases,wanted)])>1e-3:continue
                cx=int(round((x+.5)/sx-.5));cy=int(round((y+.5)/sy-.5))
                x0=cx-crop_size//2;y0=cy-crop_size//2
                if x0<0 or y0<0 or x0+crop_size>original_mask.shape[1] or y0+crop_size>original_mask.shape[0]:continue
                if not original_mask[y0:y0+crop_size,x0:x0+crop_size].all():continue
                key=(category,cx,cy)
                if key in seen:continue
                seen.add(key)
                rows.append(dict(category=KEYS[category],category_index=category,center_xy=[cx,cy],
                                 top_left_xy=[x0,y0],size=crop_size))
    return rows


def radial_power(patch):
    n=patch.shape[0]
    if patch.shape!=(n,n) or n<8:raise ValueError('FFT crop must be square and at least 8 pixels.')
    window=np.outer(np.hanning(n),np.hanning(n))
    spectrum=np.fft.fftshift(np.fft.fft2((patch-patch.mean())*window))
    power=abs(spectrum/n**2)**2
    f=np.fft.fftshift(np.fft.fftfreq(n));xx,yy=np.meshgrid(f,f);r=np.hypot(xx,yy)
    bins=np.rint(r*n).astype(int);valid=r<=.5
    count=np.bincount(bins[valid],minlength=n//2+1)
    radial=np.bincount(bins[valid],weights=power[valid],minlength=n//2+1)/np.maximum(count,1)
    return np.arange(n//2+1)/n,radial


def savefig(fig,out,name):
    fig.savefig(out/name,dpi=140,bbox_inches='tight',pad_inches=.12);plt.close(fig)


def make_figures(out,raw,work,detrended,power,fx,fy,vectors,mask,phases,components,coherence,rows,freq,features,labels):
    plt.rcParams.update({'font.family':['Yu Gothic','DejaVu Sans'],'font.size':10,'axes.unicode_minus':False})
    fig,axes=plt.subplots(1,3,figsize=(13,5),layout='constrained')
    for ax,a,title in zip(axes,[work,gaussian_filter(work,1),detrended],['入力画像（検出用に縮小）','細かな反射を抑制','緩やかな明るさむらを除去']):
        limits=(0,255) if a is not detrended else (-np.percentile(abs(detrended),99),np.percentile(abs(detrended),99))
        ax.imshow(a,cmap='gray',vmin=limits[0],vmax=limits[1]);ax.set_title(title);ax.set_axis_off()
    savefig(fig,out,'01_preprocessing.png')
    fig,ax=plt.subplots(figsize=(7,6),layout='constrained')
    im=ax.imshow(10*np.log10(power/max(float(power.max()),1e-12)+1e-12),origin='lower',cmap='magma',vmin=-50,vmax=0,
                 extent=[fx[0],fx[-1],fy[0],fy[-1]])
    extent=min(.5,float(np.max(abs(vectors)))*3.3)
    for i,v in enumerate(vectors):
        ax.scatter([v[0],-v[0]],[v[1],-v[1]],facecolors='none',edgecolors=COLORS[i],s=100,label=f'方向{i+1}')
    ax.set(xlim=(-extent,extent),ylim=(-extent,extent),xlabel='検出画像内 fx [cycles/pixel]',ylabel='検出画像内 fy [cycles/pixel]',title='02 周期の検出（画像ごとに推定）');ax.legend()
    fig.colorbar(im,ax=ax,label='最大値基準 [dB]');savefig(fig,out,'02_fft.png')
    fig,axes=plt.subplots(1,3,figsize=(13,5),layout='constrained')
    axes[0].imshow(work,cmap='gray',vmin=0,vmax=255);axes[0].contour(mask,levels=[.5],colors=['#e41a32'],linewidths=1)
    axes[0].set_title('推定範囲（材料全体の保証はない）')
    axes[1].imshow(mask,cmap='gray',vmin=0,vmax=1);axes[1].set_title('格子解析に使うマスク')
    im=axes[2].imshow(coherence,cmap='viridis',vmin=0,vmax=1);axes[2].set_title('周期成分の局所的な強さ')
    for ax in axes:ax.set_axis_off()
    fig.colorbar(im,ax=axes[2],shrink=.7);savefig(fig,out,'03_analysis_mask.png')
    fig,axes=plt.subplots(1,2,figsize=(11,6),layout='constrained')
    axes[0].imshow(work,cmap='gray',vmin=0,vmax=255)
    for i,p in enumerate(phases):
        levels=np.arange(np.ceil((p[mask].min()-np.pi)/(2*np.pi)),np.floor((p[mask].max()-np.pi)/(2*np.pi))+1)*2*np.pi+np.pi
        axes[0].contour(np.ma.array(p,mask=~mask),levels=levels,colors=[COLORS[i]],linewidths=.7,linestyles=['solid','dashed'][i])
    axes[0].set_title('黒い線の中心を重ね合わせ')
    model=sum(components);lim=np.percentile(abs(detrended[mask]),99)
    axes[1].imshow(np.ma.array(model,mask=~mask),cmap='gray',vmin=-lim,vmax=lim);axes[1].set_title('周期成分から再構成した格子')
    for ax in axes:ax.set_axis_off()
    savefig(fig,out,'04_grid_overlay.png')
    fig,ax=plt.subplots(figsize=(10,8),layout='constrained');ax.imshow(raw,cmap='gray',vmin=0,vmax=255)
    for i in range(4):
        subset=[r for r in rows if r['category_index']==i]
        style={'facecolors':'none','edgecolors':COLORS[i]} if i in [1,3] else {'color':COLORS[i]}
        ax.scatter([r['center_xy'][0] for r in subset],[r['center_xy'][1] for r in subset],s=20,marker=['x','o','+','s'][i],label=f'{NAMES[i]} ({len(subset)})',**style)
    ax.legend(loc='upper left',bbox_to_anchor=(1,1));ax.set(title='05 格子位置の分類（傷の有無は別途指定）',xlabel='元画像 x [pixel]',ylabel='元画像 y [pixel]')
    savefig(fig,out,'05_categories.png')
    fig,axes=plt.subplots(2,4,figsize=(12,6),layout='constrained')
    for i in range(4):
        subset=[r for r in rows if r['category_index']==i]
        subset=sorted(subset,key=lambda r:r['spectral_difference_db'])
        for row,p in enumerate(([subset[len(subset)//2],subset[-1]] if subset else [None,None])):
            ax=axes[row,i]
            if p:
                x,y=p['top_left_xy'];n=p['size'];ax.imshow(raw[y:y+n,x:x+n],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                ax.set_title(f"{NAMES[i]} / {p['id']}\n"+('中間的な例' if row==0 else '同分類内でスペクトル差が大きい例'))
            ax.set_axis_off()
    fig.suptitle('06 切り出し例（スペクトル差だけで傷とは判定しない）');savefig(fig,out,'06_crop_examples.png')
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    wavelength=1/freq[1:];order=np.argsort(wavelength)
    for i,ax in enumerate(axes.flat):
        idx=[j for j,r in enumerate(rows) if r['category_index']==i]
        if idx:
            values=features[idx,1:][:,order];lo,med,hi=np.percentile(values,[10,50,90],axis=0)
            ax.plot(wavelength[order],med,color=COLORS[i],marker='.',label='中央値')
            ax.fill_between(wavelength[order],lo,hi,color=COLORS[i],alpha=.15,label='10〜90パーセンタイル')
        ax.set(title=NAMES[i],xlabel='空間波長 [pixel／周期]',ylabel='線形パワー [a.u.]',ylim=(0,None));ax.grid(alpha=.15);ax.legend()
    fig.suptitle('07 各格子位置のスペクトル分布（正常・傷の混在は未判定）');savefig(fig,out,'07_spectrum_by_category.png')
    if labels:
        fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
        for i,ax in enumerate(axes.flat):
            for truth,color in [('normal','#0072B2'),('defect','#D55E00')]:
                idx=[j for j,r in enumerate(rows) if r['category_index']==i and labels.get(r['id'])==truth]
                if idx:ax.plot(wavelength[order],np.median(features[idx,1:],axis=0)[order],color=color,label=('傷なし' if truth=='normal' else '傷あり')+f' n={len(idx)}')
            ax.set(title=NAMES[i],xlabel='空間波長 [pixel／周期]',ylabel='線形パワー [a.u.]',ylim=(0,None));ax.grid(alpha=.15)
            if ax.get_legend_handles_labels()[0]:ax.legend()
        fig.suptitle('08 ユーザーが指定した傷あり／傷なしの比較');savefig(fig,out,'08_labeled_comparison.png')


def process_image(path:Path,out:Path,config:Config,manual_mask:Path|None=None,user_labels=None):
    out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    raw=np.asarray(Image.open(path).convert('L'),dtype=float)
    oh,ow=raw.shape
    if min(oh,ow)<32:raise GridNotFound('画像が小さすぎます（短辺32画素未満）。')
    scale=min(1,config.max_dimension/max(oh,ow));w=round(ow*scale);h=round(oh*scale)
    work=np.asarray(Image.fromarray(raw.astype(np.uint8)).resize((w,h),Image.Resampling.LANCZOS),dtype=float)
    scales=np.array([w/ow,h/oh])
    detrended=gaussian_filter(work,1)-gaussian_filter(work,min(h,w)/15)
    report={'source':str(path.resolve()),'source_size_wh':[ow,oh],'config':asdict(config),'mask_mode':'manual' if manual_mask else 'automatic_periodic_support','status':'processing','warnings':[]}
    try:
        seeds,power,fx,fy,diagnostic,accepted=fft_candidates(detrended,scales,config)
        report.update(diagnostic)
        if not accepted:raise GridNotFound('2方向の周期成分が不十分です。格子なし、または検出条件外の可能性があります。')
        vectors,fit_ok=refine_vectors(detrended,seeds)
        period,phases,amps,components,coherence,carriers=local_grid(detrended,vectors)
        mask,thresholds=analysis_mask(amps,coherence,period,config)
        if manual_mask:
            original_mask=np.asarray(Image.open(manual_mask).convert('L'))>0
            if original_mask.shape!=raw.shape:raise ValueError('手動マスクは元画像と同じサイズである必要があります。')
            mask=np.asarray(Image.fromarray(original_mask).resize((w,h),Image.Resampling.NEAREST),bool)
        else:
            original_mask=np.asarray(Image.fromarray(mask).resize((ow,oh),Image.Resampling.NEAREST),bool)
        if mask.sum()<max(period**2*3,mask.size*.025):raise GridNotFound('信頼できる格子の範囲が不足しています。')
        original_vectors=vectors*scales
        original_period=float(np.median(1/np.linalg.norm(original_vectors,axis=1)))
        n=config.crop_size if config.crop_size is not None else max(8,int(original_period*config.crop_fraction)//2*2)
        if n<8 or n%2:raise ValueError('切り出しサイズは8以上の偶数にしてください。')
        if n<16:report['warnings'].append('切り出しが16画素未満で、スペクトルの周波数点数が少なくなります。')
        rows=find_centers(phases,carriers,vectors,mask,scales,original_mask,n)
        if len(rows)<8 or len({r['category'] for r in rows})<4:raise GridNotFound('4種類の格子位置を十分に切り出せませんでした。')
        features=[]
        for j,r in enumerate(rows):
            r['id']=f'r{j:04d}';x,y=r['top_left_xy'];patch=raw[y:y+n,x:x+n]
            freq,p=radial_power(patch);features.append(p)
            r['mean_gray']=float(patch.mean());r['label']=(user_labels or {}).get(r['id'])
        features=np.array(features);db=10*np.log10(features+1e-12)
        for i in range(4):
            idx=[j for j,r in enumerate(rows) if r['category_index']==i]
            reference=np.median(db[idx,1:],axis=0)
            for j in idx:rows[j]['spectral_difference_db']=float(np.sqrt(np.mean((db[j,1:]-reference)**2)))
        counts=[sum(r['category_index']==i for r in rows) for i in range(4)]
        Image.fromarray(original_mask.astype(np.uint8)*255).save(out/'analysis_mask.png')
        cutout=np.dstack([np.where(original_mask,raw,255).astype(np.uint8)]*3+[original_mask.astype(np.uint8)*255])
        Image.fromarray(cutout).save(out/'analysis_cutout.png')
        for r in rows:
            folder=out/'crops'/r['category'];folder.mkdir(parents=True,exist_ok=True)
            filename=folder/(r['id']+'.png');x,y=r['top_left_xy']
            Image.fromarray(raw[y:y+n,x:x+n].astype(np.uint8)).save(filename)
            r['file']=str(filename.relative_to(out))
        np.savez_compressed(out/'spectra.npz',frequency=freq,linear_power=features,db=db,region_ids=np.array([r['id'] for r in rows]))
        (out/'labels_template.json').write_text(json.dumps({path.name:{r['id']:None for r in rows}},indent=2),encoding='utf-8')
        if config.figures:make_figures(out,raw,work,detrended,power,fx,fy,vectors,mask,phases,components,coherence,rows,freq,features,user_labels)
        if not fit_ok:report['warnings'].append('全体の周波数調整が規定回数内に収束しませんでした。重ね合わせを確認してください。')
        report.update(status='needs_review',periods_pixels=(1/np.linalg.norm(original_vectors,axis=1)).tolist(),
                      vectors_cycles_per_original_pixel=original_vectors.tolist(),crop_size=n,counts=counts,
                      mask_fraction=float(original_mask.mean()),amplitude_thresholds=thresholds,
                      median_crop_gray=[float(np.median([r['mean_gray'] for r in rows if r['category_index']==i])) for i in range(4)],
                      regions=rows,meaning='Automated candidates; mask and grid must be reviewed. No automatic defect ground truth.')
    except GridNotFound as error:
        report.update(status='grid_not_found',reason=str(error))
        if config.figures:
            fig,ax=plt.subplots(figsize=(9,6),layout='constrained');ax.imshow(raw,cmap='gray');ax.set_axis_off()
            ax.set_title('Grid not found / review required');savefig(fig,out,'00_grid_not_found.png')
    report['elapsed_seconds']=round(time.monotonic()-started,3)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def collect_inputs(inputs):
    files=[]
    for item in inputs:
        p=Path(item)
        if p.is_dir():files.extend(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'})
        elif p.is_file():files.append(p)
        else:raise FileNotFoundError(p)
    return sorted(set(p.resolve() for p in files),key=lambda p:[int(t) if t.isdigit() else t.lower() for t in re.split('(\\d+)',str(p))])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='+',help='画像ファイル、または画像が入ったフォルダ（直下のみ）')
    parser.add_argument('--output',type=Path,default=Path('outputs/general_grid'))
    parser.add_argument('--max-dimension',type=int,default=1024)
    parser.add_argument('--crop-size',type=int,default=None,help='全画像共通の偶数画素数。省略時は格子周期の約1/4。')
    parser.add_argument('--min-period',type=float,default=None,help='元画像での最小格子周期[pixel]')
    parser.add_argument('--max-period',type=float,default=None,help='元画像での最大格子周期[pixel]')
    parser.add_argument('--roi-mask',type=Path,help='任意：元画像と同サイズの白黒マスク。入力画像1枚の場合のみ。')
    parser.add_argument('--labels',type=Path,help='任意：{"image.png":{"r0000":"normal","r0001":"defect"}}')
    parser.add_argument('--no-figures',action='store_true',help='一括検証用。マスク・切り出し・数値結果は保存。')
    args=parser.parse_args()
    if args.max_dimension<64:parser.error('--max-dimension は64以上が必要です。')
    if args.crop_size is not None and (args.crop_size<8 or args.crop_size%2):parser.error('--crop-size は8以上の偶数が必要です。')
    files=collect_inputs(args.inputs)
    if not files:parser.error('入力画像がありません。')
    if args.roi_mask and len(files)!=1:parser.error('--roi-mask は入力画像1枚の場合だけ使えます。')
    truth=json.loads(args.labels.read_text(encoding='utf-8')) if args.labels else {}
    config=Config(max_dimension=args.max_dimension,crop_size=args.crop_size,min_period=args.min_period,max_period=args.max_period,figures=not args.no_figures)
    args.output.mkdir(parents=True,exist_ok=True)
    batch=[]
    # Separate output directories and disallow accidental stale crops on reruns.
    names=set()
    for path in files:
        name=path.stem
        if name in names:parser.error('同じ名前の入力画像があります。別々に実行してください。')
        names.add(name)
        out=args.output/name
        if out.exists() and any(out.iterdir()):parser.error(f'出力先が空ではありません。別の --output を指定してください: {out}')
    for path in files:
        print(f'Processing {path.name}',flush=True)
        try:
            image_labels=truth.get(path.name,{})
            if any(v not in {None,'normal','defect'} for v in image_labels.values()):raise ValueError('ラベルは normal / defect / null のみ使用できます。')
            report=process_image(path,args.output/path.stem,config,args.roi_mask,image_labels)
        except Exception as error:
            report={'source':str(path),'status':'error','reason':str(error)}
            out=args.output/path.stem;out.mkdir(parents=True,exist_ok=True)
            (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        batch.append({k:v for k,v in report.items() if k!='regions'})
        print(json.dumps({k:report[k] for k in ['status','counts','mask_fraction','reason'] if k in report},ensure_ascii=False),flush=True)
    (args.output/'batch_summary.json').write_text(json.dumps(batch,ensure_ascii=False,indent=2),encoding='utf-8')
    if any(r['status']=='error' for r in batch):raise SystemExit(1)


if __name__=='__main__':main()
