#!/usr/bin/env python3
"""Batch version of FFT_radial_full_image.ipynb (global median radial spectrum)."""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import fft
from scipy.sparse import csr_matrix
from numpy.lib.stride_tricks import sliding_window_view

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data20260903'/'data20260903'
OUT=ROOT/'output_radial_full_image'
WINDOW=128; STRIDE=4; BATCH=64; WORKERS=-1; BINS=64; FMAX=.5


def radial_matrix(window,bins,fmax):
    center=window//2; y,x=np.indices((window,window))
    radius=np.sqrt((x-center)**2+(y-center)**2)/window
    edges=np.linspace(0.,fmax,bins+1); index=np.digitize(radius.ravel(),edges)-1
    valid=(index>=0)&(index<bins); count=np.bincount(index[valid],minlength=bins).astype(np.float32)
    count[count==0]=1
    selected=np.flatnonzero(valid)
    matrix=csr_matrix((1.0/count[index[valid]],(selected,index[valid])),shape=(window*window,bins),dtype=np.float32)
    return matrix,(edges[:-1]+edges[1:])/2


def analyze(img,window,stride,batch,workers):
    h,w=img.shape
    if h<window or w<window: raise ValueError('image smaller than window')
    matrix,freq=radial_matrix(window,window//2,FMAX)
    hann=np.hanning(window).astype(np.float32); hann=np.outer(hann,hann).astype(np.float32)
    xs=np.arange(0,w-window+1,stride); ys=np.arange(0,h-window+1,stride)
    features=np.empty((len(ys),len(xs),window//2),np.float32)
    t=time.perf_counter()
    for iy,y in enumerate(ys):
        win=np.moveaxis(sliding_window_view(img[y:y+window],window,axis=1),1,0)[::stride]
        for ix in range(0,len(xs),batch):
            patches=np.asarray(win[ix:ix+batch],dtype=np.float32)
            v=patches-patches.mean(axis=(1,2),keepdims=True); v*=hann[None]
            z=fft.fftshift(fft.fft2(v,axes=(-2,-1),workers=workers),axes=(-2,-1))
            power=(np.abs(z)**2).astype(np.float32)
            radial=power.reshape(len(power),-1)@matrix
            features[iy,ix:ix+batch]=(10*np.log10(radial+1e-12)).astype(np.float32)
    reference=np.median(features.reshape(-1,window//2),axis=0)
    delta=features-reference[None,None,:]
    anomaly=np.sqrt(np.mean(delta**2,axis=2))
    return {'features':features,'xs':xs+window/2,'ys':ys+window/2,'freq':freq,'reference':reference,'anomaly':anomaly,'seconds':time.perf_counter()-t}


def figure_map(a,r,path,label):
    fig,ax=plt.subplots(figsize=(10,7)); im=ax.imshow(a,origin='upper',extent=[r['xs'][0],r['xs'][-1],r['ys'][-1],r['ys'][0]],aspect='equal',cmap='inferno')
    fig.colorbar(im,ax=ax,label=label); ax.set(xlabel='x (px)',ylabel='y (px)'); fig.savefig(path,dpi=120,bbox_inches='tight'); plt.close(fig)


def process(path,out,args):
    img=np.asarray(Image.open(path).convert('L'),dtype=np.float32)
    r=analyze(img,args.window,args.stride,args.batch_size,args.workers)
    np.savez_compressed(out/'radial_fft_feature_map.npz',feature_map=r['features'],x_centers=r['xs'],y_centers=r['ys'],frequency_axis=r['freq'],window_size=np.int32(args.window),stride_x=np.int32(args.stride),stride_y=np.int32(args.stride),reference_spectrum=r['reference'],anomaly_map=r['anomaly'])
    freq_mask=(r['freq']>=.03)&(r['freq']<=.08)
    figure_map(r['features'][:,:,freq_mask].mean(axis=2),r,out/'frequency_band_map.png','Mean radial power (dB)')
    figure_map(r['anomaly'],r,out/'anomaly_map.png','RMS difference from image median (dB)')
    fig,ax=plt.subplots(); iy=len(r['ys'])//2; ix=len(r['xs'])//2
    ax.plot(r['freq'],r['features'][iy,ix]); ax.set(xlim=(0,.15),xlabel='Frequency (cycles/pixel)',ylabel='Radial power (dB)',title=f"Center block: x={r['xs'][ix]:.0f}, y={r['ys'][iy]:.0f}")
    ax.grid(alpha=.3); fig.savefig(out/'center_spectrum.png',dpi=120,bbox_inches='tight'); plt.close(fig)
    metadata={'image':str(path.resolve()),'image_width':img.shape[1],'image_height':img.shape[0],'feature_shape':r['features'].shape,'analysis_seconds':r['seconds'],'anomaly_mean':float(np.mean(r['anomaly'])),'anomaly_max':float(np.max(r['anomaly']))}
    (out/'metrics.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    return metadata


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    settings={'data':str(args.data.resolve()),'window':args.window,'stride':args.stride,'batch_size':args.batch_size,'workers':args.workers,'radial_freq_max':FMAX,'hann':True,'algorithm':'full FFT; unnormalized power; linear radial mean then dB; global median reference'}
    settings_path=args.output/'settings.json'
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding='utf-8'))!=settings: raise ValueError('Existing output settings differ; use a new output directory')
    else: settings_path.write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
    records=[]
    for i in range(args.start_number,args.end_number+1):
        name=f'image{i}'; image=args.data/f'{name}.png'; out=args.output/name
        if (out/'metrics.json').exists():
            print(f'{name}: already complete',flush=True); records.append(json.loads((out/'metrics.json').read_text(encoding='utf-8'))); continue
        if not image.exists(): print(f'{name}: input missing',flush=True); continue
        stage=args.output/f'{name}_partial'; counter=1
        while stage.exists(): stage=args.output/f'{name}_partial_{counter}'; counter+=1
        stage.mkdir()
        try:
            print(f'{name}: analyzing',flush=True); result=process(image,stage,args)
            if out.exists(): raise FileExistsError(f'Refusing to replace {out}')
            stage.rename(out); records.append(result); print(f"{name}: {result['analysis_seconds']:.1f} seconds",flush=True)
        except Exception as exc:
            print(f'{name}: failed: {exc}',flush=True)
    print(f'Completed {len(records)}/{args.end_number-args.start_number+1} images. Output: {args.output}',flush=True)
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=DATA); p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--window',type=int,default=WINDOW); p.add_argument('--stride',type=int,default=STRIDE)
    p.add_argument('--batch-size',type=int,default=BATCH); p.add_argument('--workers',type=int,default=WORKERS)
    p.add_argument('--start-number',type=int,default=1); p.add_argument('--end-number',type=int,default=21)
    args=p.parse_args()
    if args.window<2 or args.window%2 or args.stride<1 or args.batch_size<1 or args.workers==0 or not 1<=args.start_number<=args.end_number<=21: p.error('invalid window, stride, batch, workers, or image number range')
    run(args)


if __name__=='__main__': main()
