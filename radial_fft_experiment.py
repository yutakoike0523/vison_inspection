#!/usr/bin/env python3
"""Compare original 2-D FFT neighbor scores with radial power spectra."""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import psutil
from PIL import Image
from scipy import fft
from scipy.sparse import csr_matrix
from numpy.lib.stride_tricks import sliding_window_view
import multiscale_fft_experiment as base

RANGES = [(0.005,0.50),(0.01,0.50),(0.02,0.50),(0.02,0.20)]
MAIN = (0.02,0.50)
STRIDE, SPACING, EPS = base.STRIDE, base.SPACING, base.EPS


def grids(n, width, ranges=RANGES):
    fy=np.fft.fftshift(np.fft.fftfreq(n)).astype(np.float32)
    fx=np.fft.rfftfreq(n).astype(np.float32)
    yy,xx=np.meshgrid(fy,fx,indexing='ij'); radius=np.sqrt(xx**2+yy**2)
    bins=np.floor(radius/width).astype(int)
    count_bins=int(np.floor(.5/width))+1
    weight=np.full(radius.shape,2,np.float32); weight[:,0]=1
    if n%2==0: weight[:,-1]=1
    specs={}
    for low,high in ranges:
        mask=(radius>=low)&(radius<=high); indexes=bins[mask]
        counts=np.bincount(indexes,weights=weight[mask],minlength=count_bins)
        valid=counts>=4
        ids=np.flatnonzero(mask.ravel()); selected=bins.ravel()[ids]
        # Linear power summation with conjugate multiplicities. Empty/underpopulated bins omitted.
        mat=csr_matrix((weight.ravel()[ids],(ids,selected)),shape=(radius.size,count_bins),dtype=np.float32)
        specs[(low,high)]={'mask':mask,'counts':counts,'valid':valid,'matrix':mat}
    return fy,fx,radius,specs


def check_full_fft(width):
    errors=[]
    for n in (64,128,192,256):
        rng=np.random.default_rng(1200+n); x=rng.normal(size=(n,n)).astype(np.float32)
        p=np.abs(fft.fftshift(fft.rfft2(x),axes=0))**2
        _,_,_,spec=grids(n,width,[MAIN]); s=spec[MAIN]
        half=(p.ravel()@s['matrix'])/np.maximum(s['counts'],1)
        full=np.abs(fft.fft2(x))**2
        f=np.fft.fftfreq(n).astype(np.float32); yy,xx=np.meshgrid(f,f,indexing='ij'); rad=np.sqrt(xx**2+yy**2)
        bins=np.floor(rad/width).astype(int); mask=(rad>=MAIN[0])&(rad<=MAIN[1]); count=np.bincount(bins[mask],minlength=len(s['counts']))
        full_avg=np.bincount(bins[mask],weights=full[mask],minlength=len(count))/np.maximum(count,1)
        good=s['valid']; err=float(np.max(np.abs(half[good]-full_avg[good])/np.maximum(full_avg[good],1e-12)))
        if not np.array_equal(count,s['counts']) or err>1e-5: raise AssertionError(f'Full FFT radial check failed for {n}: {err}')
        errors.append({'window':n,'max_relative_error':err,'weighted_counts_match':True})
    return errors


def neighbor_compare(features, mask=None):
    ny,nx=features.shape[:2]; scores=np.full((ny,nx),np.nan,np.float32); medtime=scoretime=0.0
    step=SPACING//STRIDE
    offsets=[(y*step,x*step) for y in (-1,0,1) for x in (-1,0,1) if y or x]
    for iy in range(ny):
        for ix in range(nx):
            neighbors=[features[iy+y,ix+x] for y,x in offsets if 0<=iy+y<ny and 0<=ix+x<nx]
            if not neighbors: continue
            t=time.perf_counter(); ref=np.median(np.stack(neighbors),axis=0); medtime+=time.perf_counter()-t
            t=time.perf_counter(); d=features[iy,ix]-ref; d=d[mask] if mask is not None else d
            scores[iy,ix]=np.sqrt(np.mean(d**2)); scoretime+=time.perf_counter()-t
    return scores,medtime,scoretime


def do_window(img,n,width,workers,batch,folder,labels):
    t0=time.perf_counter(); h,w=img.shape
    xs=np.arange(0,w-n+1,STRIDE); ys=np.arange(0,h-n+1,STRIDE)
    if not len(xs) or not len(ys): raise ValueError('Window exceeds image dimensions')
    fy,fx,radius,spec=grids(n,width); hann=np.hanning(n).astype(np.float32); hann=np.outer(hann,hann)
    radial={r:np.empty((len(ys),len(xs),int(s['valid'].sum())),np.float32) for r,s in spec.items()}
    fft_time=rad_time=0.0
    with tempfile.TemporaryDirectory(prefix='radial_fft_') as tmp:
        twod=np.memmap(Path(tmp)/'spectra.dat',mode='w+',dtype=np.float32,shape=(len(ys),len(xs),n,n//2+1))
        for iy,y in enumerate(ys):
            win=np.moveaxis(sliding_window_view(img[y:y+n],n,axis=1),1,0)[::STRIDE]
            for ix in range(0,len(xs),batch):
                v=np.asarray(win[ix:ix+batch],dtype=np.float32).copy(); v-=v.mean(axis=(-2,-1),keepdims=True); v*=hann
                t=time.perf_counter(); f=fft.rfft2(v,axes=(-2,-1),workers=workers); fft_time+=time.perf_counter()-t
                p=(np.abs(fft.fftshift(f,axes=(-2,)))**2).astype(np.float32)
                s=spec[MAIN]; total=np.maximum(p[:,s['mask']].sum(axis=1,keepdims=True),EPS)
                norm=p/total[:,:,None]; twod[iy,ix:ix+batch]=(10*np.log10(norm+EPS)).astype(np.float32)
                t=time.perf_counter()
                for ran,s in spec.items():
                    denom=np.maximum(p[:,s['mask']].sum(axis=1),EPS)
                    sums=(p.reshape(len(p),-1)@s['matrix'])
                    vals=sums[:,s['valid']]/s['counts'][s['valid']][None,:]/denom[:,None]
                    radial[ran][iy,ix:ix+batch]=(10*np.log10(vals+EPS)).astype(np.float32)
                rad_time+=time.perf_counter()-t
        print(f'  {n}: compare 2D',flush=True)
        a,med2,score2=neighbor_compare(twod,spec[MAIN]['mask'])
        results={}; timing={}
        for ran in RANGES:
            results[ran],median,scoring=neighbor_compare(radial[ran]); timing[ran]=(median,scoring)
        centers={'xs':xs+n//2,'ys':ys+n//2}
        allmask=np.logical_or.reduce([m for _,m in labels]) if labels else np.zeros(img.shape,bool)
        normal=~base.selections(allmask,centers,n)[1] if labels else np.ones(a.shape,bool)
        local=[m for name,m in labels or [] if name=='局所変形']
        candidates=base.selections(local[0],centers,n)[0] if local else np.zeros(a.shape,bool)
        # Representative defect: highest radial score with center in annotation; representative normal: median normal score.
        rep={}
        for kind,sel in [('defect',candidates),('normal',normal)]:
            yy,xx=np.nonzero(sel & np.isfinite(results[MAIN]))
            if not len(yy): continue
            vals=results[MAIN][yy,xx]
            k=int(np.argmax(vals)) if kind=='defect' else int(np.argmin(np.abs(vals-np.median(vals))))
            jy,jx=int(yy[k]),int(xx[k]); step=SPACING//STRIDE
            neighbors=[radial[MAIN][jy+dy*step,jx+dx*step] for dy in (-1,0,1) for dx in (-1,0,1) if (dy or dx) and 0<=jy+dy*step<len(ys) and 0<=jx+dx*step<len(xs)]
            ref=np.median(np.stack(neighbors),axis=0)
            rep[kind]={'iy':jy,'ix':jx,'target':radial[MAIN][jy,jx].copy(),'reference':ref,'power2d':twod[jy,jx].copy()}
        del twod
    elapsed=time.perf_counter()-t0
    memory=psutil.Process().memory_info().peak_wset if os.name=='nt' else None
    data={'xs':centers['xs'],'ys':centers['ys'],'map_2d':a,'radial_maps':results,'radial':radial[MAIN], 'spec':spec,'fy':fy,'fx':fx,'representatives':rep,'fft_seconds':fft_time,'radial_all_ranges_seconds':rad_time,'median_2d_seconds':med2,'score_2d_seconds':score2,'timing_radial':timing,'total_seconds':elapsed,'peak_memory_bytes':memory}
    np.savez_compressed(folder/'result.npz',x_centers=data['xs'],y_centers=data['ys'],map_2d=a,**{f'map_radial_{lo:g}_{hi:g}':results[(lo,hi)] for lo,hi in RANGES},**{f'bin_counts_{lo:g}_{hi:g}':spec[(lo,hi)]['counts'] for lo,hi in RANGES},bin_width=width,window=n,stride=STRIDE,reference_spacing=SPACING,fft_workers=workers,batch_size=batch)
    return data


def evaluate(scores, normal, local, centers, n, method, freq):
    vals=scores[normal & np.isfinite(scores)]
    row={'window':n,'method':method,'freq_min':freq[0],'freq_max':freq[1],
         'count':int(np.isfinite(scores).sum()),'mean':float(np.nanmean(scores)),
         'median':float(np.nanmedian(scores)),'std':float(np.nanstd(scores)),
         'normal_count':len(vals),'normal_mean':float(np.mean(vals)) if len(vals) else np.nan,
         'normal_median':float(np.median(vals)) if len(vals) else np.nan,
         'normal_std':float(np.std(vals)) if len(vals) else np.nan,
         'normal_p99':float(np.percentile(vals,99)) if len(vals) else np.nan}
    defects=[]
    for index,mask in local:
        center,overlap=base.selections(mask,centers,n)
        for mode,selected in [('center',center),('overlap',overlap)]:
            yy,xx=np.nonzero(selected & np.isfinite(scores)); v=scores[yy,xx]
            sd=row['normal_std']; delta=float(np.median(v)-row['normal_median']) if len(v) else np.nan
            idx=int(np.argmax(v)) if len(v) else None
            defects.append({'window':n,'method':method,'freq_min':freq[0],'freq_max':freq[1],
                'annotation_index':index,'selection':mode,'count':len(v),'mean':float(np.mean(v)) if len(v) else np.nan,
                'median':float(np.median(v)) if len(v) else np.nan,'maximum':float(np.max(v)) if len(v) else np.nan,
                'p95':float(np.percentile(v,95)) if len(v) else np.nan,
                'max_x':int(centers['xs'][xx[idx]]) if idx is not None else '',
                'max_y':int(centers['ys'][yy[idx]]) if idx is not None else '',
                'normal_median':row['normal_median'],'normal_std':sd,'normal_p99':row['normal_p99'],
                'median_difference':delta,'separation_S':delta/sd if sd>0 else np.nan,
                'above_normal_p99_count':int(np.sum(v>row['normal_p99'])) if len(vals) else '',
                'above_normal_p99_fraction':float(np.mean(v>row['normal_p99'])) if len(vals) and len(v) else np.nan})
    return row,defects


def draw_map(ax,a,r,fullshape,overlay=None,labels=None,limits=None):
    if overlay is not None: ax.imshow(overlay,cmap='gray',extent=[0,fullshape[1],fullshape[0],0])
    im=ax.imshow(a,extent=base.extent(r),origin='upper',cmap='inferno',interpolation='nearest',
                 alpha=.55 if overlay is not None else 1, vmin=limits[0] if limits else None,vmax=limits[1] if limits else None)
    if overlay is not None:
        for label,mask in labels or []:
            if label=='局所変形' and mask.any():
                ax.contour(mask.astype(np.uint8),levels=[.5],colors='cyan',linewidths=1.2,origin='upper',extent=[0,fullshape[1],fullshape[0],0])
    ax.set(xlabel='x (px)',ylabel='y (px)',xlim=(0,fullshape[1]),ylim=(fullshape[0],0))
    return im


def window_plots(img,n,r,folder,labels,width):
    for method,a in [('2d',r['map_2d']),('radial',r['radial_maps'][MAIN])]:
        for overlay in (False,True):
            fig,ax=plt.subplots(figsize=(10,7)); im=draw_map(ax,a,r,img.shape,img if overlay else None,labels)
            fig.colorbar(im,ax=ax,label='RMS dB'); ax.set_title(f'{n}px {method}'); fig.savefig(folder/(f'overlay_{method}.png' if overlay else f'anomaly_{method}.png'),dpi=130,bbox_inches='tight'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,(method,a) in zip(axes,[('2D',r['map_2d']),('Radial',r['radial_maps'][MAIN])]):
        im=draw_map(ax,a,r,img.shape); ax.set_title(f'{method} (own scale)'); fig.colorbar(im,ax=ax,label='RMS dB')
    fig.savefig(folder/'method_comparison.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    bins=(np.arange(len(r['spec'][MAIN]['counts']))+.5)*width
    x=bins[r['spec'][MAIN]['valid']]
    for kind,rep in r['representatives'].items():
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        im=axes[0].imshow(rep['power2d'],origin='lower',extent=[r['fx'][0],r['fx'][-1],r['fy'][0],r['fy'][-1]],aspect='auto')
        fig.colorbar(im,ax=axes[0],label='Normalized 2D power (dB)')
        axes[0].set(xlabel='fx',ylabel='fy',title='2D spectrum')
        axes[1].plot(x,rep['target']); axes[1].set(xlabel='r (cycles/pixel)',ylabel='Normalized radial power (dB)',title='Linear mean then dB')
        fig.suptitle(f"{kind}: ({r['xs'][rep['ix']]}, {r['ys'][rep['iy']]})"); fig.savefig(folder/f'conversion_{kind}.png',dpi=140,bbox_inches='tight'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4)); fig2,axes2=plt.subplots(1,2,figsize=(12,4))
    for ax,ax2,kind in zip(axes,axes2,['normal','defect']):
        if kind not in r['representatives']: ax.set_title(f'{kind}: unavailable'); ax2.set_title(f'{kind}: unavailable'); continue
        rep=r['representatives'][kind]; ax.plot(x,rep['target'],label='Target'); ax.plot(x,rep['reference'],label='Neighbor median'); ax.legend()
        ax.set(title=kind,xlabel='r (cycles/pixel)',ylabel='dB'); ax2.plot(x,rep['target']-rep['reference']); ax2.axhline(0,color='gray',lw=.5)
        ax2.set(title=kind,xlabel='r (cycles/pixel)',ylabel='Target - reference (dB)')
    fig.savefig(folder/'radial_spectrum_comparison.png',dpi=140,bbox_inches='tight'); plt.close(fig)
    fig2.savefig(folder/'radial_spectrum_difference.png',dpi=140,bbox_inches='tight'); plt.close(fig2)


def comparison_plots(results,folder,width):
    xmin=max(r['xs'][0] for r in results.values()); xmax=min(r['xs'][-1] for r in results.values())
    ymin=max(r['ys'][0] for r in results.values()); ymax=min(r['ys'][-1] for r in results.values())
    arrays=[]
    for n,r in results.items():
        for method,a in [('2D',r['map_2d']),('Radial',r['radial_maps'][MAIN])]:
            common=a[np.ix_((r['ys']>=ymin)&(r['ys']<=ymax),(r['xs']>=xmin)&(r['xs']<=xmax))]
            arrays.append((n,method,common))
    for style in ('shared','own','normalized'):
        fig,axes=plt.subplots(len(results),2,figsize=(12,4*len(results)),constrained_layout=True)
        axes=np.asarray(axes).reshape(-1)
        arr=[(a-np.nanmedian(a))/np.nanstd(a) if style=='normalized' and np.nanstd(a)>0 else a for _,_,a in arrays]
        lo=min(np.nanmin(a) for a in arr); hi=max(np.nanpercentile(a,99) for a in arr)
        for ax,(n,method,_),a in zip(axes,arrays,arr):
            lim=(np.nanmin(a),np.nanpercentile(a,99)) if style=='own' else (lo,hi)
            im=ax.imshow(a,extent=[xmin-8,xmax+8,ymax+8,ymin-8],cmap='inferno',vmin=lim[0],vmax=lim[1],interpolation='nearest')
            ax.set(title=f'{n} px {method}',xlabel='x (px)',ylabel='y (px)'); fig.colorbar(im,ax=ax,shrink=.8)
        fig.suptitle({'shared':'Raw RMS dB: shared scale','own':'Raw RMS dB: each map own scale','normalized':'(score - common-region median) / common-region std; shared scale'}[style])
        name={'shared':'anomaly_comparison.png','own':'anomaly_comparison_individual_scale.png','normalized':'normalized_comparison.png'}[style]
        fig.savefig(folder/name,dpi=120); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for n,r in results.items():
        for ax,kind in zip(axes,['normal','defect']):
            if kind not in r['representatives']: continue
            bins=(np.arange(len(r['spec'][MAIN]['counts']))+.5)*width
            ax.plot(bins[r['spec'][MAIN]['valid']],r['representatives'][kind]['target'],label=f'{n} px')
            ax.set(title=kind,xlabel='r (cycles/pixel)',ylabel='Normalized radial power (dB)')
    for ax in axes: ax.legend()
    fig.savefig(folder/'radial_spectra_comparison.png',dpi=140,bbox_inches='tight'); plt.close(fig)


def frequency_detail(img,n,r,labels,ran,width,workers,folder):
    """Recompute representative spectra for a selected supplemental range."""
    local=[mask for label,mask in labels or [] if label=='局所変形']
    if not local: return None
    allmask=np.logical_or.reduce([mask for _,mask in labels])
    normal=~base.selections(allmask,r,n)[1]; defect=base.selections(local[0],r,n)[0]
    scores=r['radial_maps'][ran]; spec=grids(n,width,[ran])[3][ran]
    hann=np.hanning(n).astype(np.float32); hann=np.outer(hann,hann)
    chosen={}
    for kind,mask in [('normal',normal),('defect',defect)]:
        yy,xx=np.nonzero(mask & np.isfinite(scores))
        if not len(yy): return None
        vals=scores[yy,xx]; k=int(np.argmax(vals)) if kind=='defect' else int(np.argmin(np.abs(vals-np.median(vals))))
        iy,ix=int(yy[k]),int(xx[k]); step=SPACING//STRIDE
        def spectrum(jy,jx):
            y=int(r['ys'][jy]-n//2); x=int(r['xs'][jx]-n//2)
            v=img[y:y+n,x:x+n].copy(); v-=v.mean(); v*=hann
            power=(np.abs(fft.fftshift(fft.rfft2(v,workers=workers),axes=0))**2).astype(np.float32)
            total=max(float(power[spec['mask']].sum()),float(EPS))
            sums=power.ravel()@spec['matrix']; vals=sums[spec['valid']]/spec['counts'][spec['valid']]/total
            return (10*np.log10(vals+EPS)).astype(np.float32)
        target=spectrum(iy,ix)
        neighbors=[spectrum(iy+dy*step,ix+dx*step) for dy in (-1,0,1) for dx in (-1,0,1) if (dy or dx) and 0<=iy+dy*step<len(r['ys']) and 0<=ix+dx*step<len(r['xs'])]
        chosen[kind]=(iy,ix,target,np.median(np.stack(neighbors),axis=0))
    x=(np.arange(len(spec['counts']))+.5)*width; x=x[spec['valid']]
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for col,kind in enumerate(['normal','defect']):
        iy,ix,target,ref=chosen[kind]; axes[0,col].plot(x,target,label='Target'); axes[0,col].plot(x,ref,label='Neighbor median'); axes[0,col].legend()
        axes[0,col].set(title=f"{kind}: ({r['xs'][ix]}, {r['ys'][iy]})",ylabel='Normalized power (dB)')
        axes[1,col].plot(x,target-ref); axes[1,col].axhline(0,color='gray',lw=.5); axes[1,col].set(xlabel='r (cycles/pixel)',ylabel='Target - reference (dB)')
    fig.suptitle(f'{n} px, {ran[0]:g}–{ran[1]:g} cycles/pixel'); fig.savefig(folder/'selected_range_spectra.png',dpi=140); plt.close(fig)
    iy,ix,target,ref=chosen['defect']; k=int(np.argmax(np.abs(target-ref)))
    return {'window':n,'range':ran,'x':int(r['xs'][ix]),'y':int(r['ys'][iy]),'frequency':float(x[k]),'difference_db':float((target-ref)[k])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,default=base.DATA/'image5.png')
    p.add_argument('--output',type=Path,default=base.ROOT/'output_radial_fft')
    p.add_argument('--windows',type=int,nargs='+',default=[64,128,192,256])
    p.add_argument('--bin-width',type=float,default=.01)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=64)
    args=p.parse_args()
    if not 0<args.bin_width<=.1 or args.workers<1 or args.batch_size<1 or any(n<2 or n%2 for n in args.windows) or len(set(args.windows))!=len(args.windows): p.error('positive bin width <= 0.1, positive workers and batch size, unique even windows >= 2 required')
    img=np.asarray(Image.open(args.image).convert('L'),dtype=np.float32)
    labels=base.load_labels(args.image.with_suffix('.json'),img.shape)
    check=check_full_fft(args.bin_width)
    out=base.unique_dir(args.output/args.image.stem); comp=out/'comparison'; comp.mkdir()
    (comp/'settings.json').write_text(json.dumps({'image':str(args.image.resolve()),'windows':args.windows,'stride':STRIDE,'reference_grid':3,'reference_spacing_px':SPACING,'fft_workers':args.workers,'batch_size':args.batch_size,'bin_width':args.bin_width,'frequency_ranges':RANGES,'min_full_fft_component_count':4,'radial_bin_rule':'floor(radius/bin_width), include only components in frequency range','full_fft_checks':check},ensure_ascii=False,indent=2),encoding='utf-8')
    results={}; methods=[]; defects=[]; freq_metrics=[]; counts=[]; times=[]
    for n in args.windows:
        print(f'Window {n}: start',flush=True); folder=out/f'window_{n}'; folder.mkdir()
        r=do_window(img,n,args.bin_width,args.workers,args.batch_size,folder,labels); results[n]=r
        allmask=np.logical_or.reduce([m for _,m in labels]) if labels else np.zeros(img.shape,bool)
        normal=~base.selections(allmask,r,n)[1] if labels else np.zeros(r['map_2d'].shape,bool)
        local=[(i,mask) for i,(name,mask) in enumerate(labels or []) if name=='局所変形']
        for method,arr in [('2d',r['map_2d']),('radial',r['radial_maps'][MAIN])]:
            row,detail=evaluate(arr,normal,local,r,n,method,MAIN); methods.append(row); defects.extend(detail)
        for ran,arr in r['radial_maps'].items():
            row,detail=evaluate(arr,normal,local,r,n,'radial',ran)
            for d in detail:
                freq_metrics.append({**row,**{f'defect_{key}':val for key,val in d.items() if key not in ('window','method','freq_min','freq_max')}})
            if not detail: freq_metrics.append(row)
        for ran,s in r['spec'].items():
            for i,c in enumerate(s['counts']): counts.append({'window':n,'freq_min':ran[0],'freq_max':ran[1],'bin_index':i,'bin_start':i*args.bin_width,'bin_end':(i+1)*args.bin_width,'full_fft_component_count':int(c),'used':bool(s['valid'][i])})
        times.append({'window':n,'fft_seconds':r['fft_seconds'],'radial_conversion_4_ranges_seconds':r['radial_all_ranges_seconds'],'neighbor_median_2d_seconds':r['median_2d_seconds'],'score_2d_seconds':r['score_2d_seconds'],'neighbor_median_radial_main_seconds':r['timing_radial'][MAIN][0],'score_radial_main_seconds':r['timing_radial'][MAIN][1],'total_shared_all_ranges_seconds':r['total_seconds'],'process_lifetime_peak_wset_bytes':r['peak_memory_bytes']})
        window_plots(img,n,r,folder,labels,args.bin_width)
        print(f"Window {n}: {r['total_seconds']:.1f}s",flush=True)
    base.csv_write(comp/'method_metrics.csv',methods); base.csv_write(comp/'defect_metrics.csv',defects)
    # All frequency rows have the same columns if annotation exists.
    base.csv_write(comp/'frequency_range_metrics.csv',freq_metrics); base.csv_write(comp/'bin_counts.csv',counts); base.csv_write(comp/'timings.csv',times)
    comparison_plots(results,comp,args.bin_width)
    candidates=[d for d in freq_metrics if d.get('defect_selection')=='center' and np.isfinite(d.get('defect_separation_S',np.nan))]
    best=max(candidates,key=lambda d:d['defect_separation_S']) if candidates else None
    detail=frequency_detail(img,int(best['window']),results[int(best['window'])],labels,(best['freq_min'],best['freq_max']),args.bin_width,args.workers,comp) if best else None
    lines=['# 2次元FFTと放射方向平均の比較',f'対象: {args.image}。目的: 局所変形を正常表面より目立たせられるか比較する。',
           '', '## 方法・条件',f'窓 {args.windows} px、stride {STRIDE} px、周辺3×3（中央除く最大8）、間隔 {SPACING} px。主比較の範囲は0.02〜0.50 cycles/pixel。元方式は既存の片側FFTの指定範囲内の総パワーで正規化した後にdB化し、周辺中央値との差のRMSを算出。',
           f'放射平均は同じ線形パワーを同じ片側総パワーで正規化してから距離ごとに平均し、dB化。rfft2のfx=0・ナイキスト列は重み1、その他は2。ビンは原点から幅 {args.bin_width:g} cycles/pixelで共通化し、境界をまたぐビンには範囲内の成分のみ使用。重み付き成分数4未満のビンは除外。空ビンを補間しない。既定の0.01は全窓で共通軸を保ちつつ0.005より疎なビンを減らすための選択で、64 pxの分解能0.015625より細く、空ビンが生じる。各ビン数はbin_counts.csv参照。',
           '片側正規化は元コードとの整合を優先したため、フルFFTの総パワーで1に正規化する定義とは異なる。FFTのフルスペクトルと重み付き片側スペクトルは成分数と放射平均を合成画像で照合済み（settings.json）。',
           '正常は全ラベルとの窓全体の重なりを除き、未ラベル領域を正常と仮定。変形評価では中心と重なりを別集計。検出マップ自体は札・背景を除外しない。表示の共通範囲は全窓で解析可能な中心座標範囲。正規化画像は各マップの共通範囲の(値−中央値)/標準偏差。色の上限は各99%点に基づく表示設定で判定閾値ではない。', '', '## 実測結果']
    for row in methods:
        n=row['window']; lines.append(f"- {n} px {row['method']}: 正常中央値 {row['normal_median']:.3f} dB、正常標準偏差 {row['normal_std']:.3f}、正常99%点 {row['normal_p99']:.3f}。")
    if defects:
        lines.extend(['','## 局所変形'])
        for d in defects:
            lines.append(f"- {d['window']} px {d['method']} {d['selection']}: 中央値 {d['median']:.3f}、95% {d['p95']:.3f}、最大 {d['maximum']:.3f} dB、S={d['separation_S']:.3f}、正常99%超 {d['above_normal_p99_count']}/{d['count']}。")
        lines.extend(['','## 周波数範囲'])
        for d in freq_metrics:
            if d.get('defect_selection')=='center': lines.append(f"- {d['window']} px {d['freq_min']:.3f}–{d['freq_max']:.2f}: S={d['defect_separation_S']:.3f}、正常99%超 {d['defect_above_normal_p99_count']}/{d['defect_count']}。")
        for n,r in results.items():
            if 'defect' in r['representatives']:
                rep=r['representatives']['defect']; diff=rep['target']-rep['reference']; k=int(np.argmax(np.abs(diff))); bins=(np.arange(len(r['spec'][MAIN]['counts']))+.5)*args.bin_width; f=bins[r['spec'][MAIN]['valid']][k]
                lines.append(f"- {n} px の代表変形点 ({r['xs'][rep['ix']]}, {r['ys'][rep['iy']]}) の最大絶対差ビン中心は {f:.3f} cycles/pixel（差 {diff[k]:+.3f} dB）。これは代表点と周辺中央値の差であり、正常代表点との差そのものではない。")
    else: lines.append('局所変形アノテーションがないため、検出性能は評価しない。')
    lines+=['','## 時間・メモリ', '同一FFT結果を4周波数範囲と2D方式で共有。timings.csvの放射変換時間は4範囲合計で、単独方式の全体時間ではない。比較時間は中央値計算とRMS計算を個別に計測。全体は4範囲の計算・一時ファイルを含み、画像読込・描画・保存を除く。メモリは同一プロセス生涯ピーク作業セットで、方式別のピークを分離できない。']
    for t in times: lines.append(f"- {t['window']} px: FFT {t['fft_seconds']:.3f} s、放射変換4範囲 {t['radial_conversion_4_ranges_seconds']:.3f} s、中央値2D {t['neighbor_median_2d_seconds']:.3f} s、放射 {t['neighbor_median_radial_main_seconds']:.3f} s、全体 {t['total_shared_all_ranges_seconds']:.3f} s。")
    lines += ['', '## 考察・今後', '正常標準偏差、S、正常99%超の数はいずれも分布の重なりを併せて評価する。2Dと放射のRMSは成分数や集約が違うため絶対値だけでは優劣を判定できない。放射平均は方向情報を捨てるため、格子の方向変化に感度が落ちる可能性があるが、この結果だけでは原因を特定できない。192・256 pxでは対象窓と128 px離れた参照窓が重なる。周波数分解能は窓サイズの逆数であり、空の低周波ビンは生成していない。', '局所変形が1箇所の場合、他画像への汎化や大きさ別の性能は判断できない。比較結果から有望な方法を検討し、異なる変形・画像で再評価する。方向情報を残す特徴量や格子点の位置ずれは未測定である。']
    if best:
        lines.append(f"今回探索した範囲の中心S最大は放射平均 {best['window']} px・{best['freq_min']:.3f}–{best['freq_max']:.2f} cycles/pixelで {best['defect_separation_S']:.3f}、正常99%超は {best['defect_above_normal_p99_count']}/{best['defect_count']}。同じ画像上で選んだ探索結果のため、独立画像での性能ではない。Bを次の検証候補とし、Aを基準として維持する。C・D・Eは未実測で、優劣は判断できない。")
    if detail:
        lines.append(f"この選択条件の代表変形点 ({detail['x']}, {detail['y']}) の周辺中央値との差の最大は {detail['frequency']:.3f} cycles/pixel（{detail['difference_db']:+.3f} dB）。selected_range_spectra.pngに正常代表点と変形代表点を表示する。変形代表点は中心内の最大スコア点、正常代表点は正常中央値に最も近い点である。")
    (comp/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Output:',out,flush=True)


if __name__=='__main__': main()
