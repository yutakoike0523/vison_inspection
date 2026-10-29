"""Radial-FFT RMS maps using every available one of the 8 surrounding blocks."""
from __future__ import annotations
import csv, time
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import evaluate_all_radial_fft as prev
import multiscale_fft_experiment as base

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'output_strict_b8_window_maps'; WS=(32,64,128,256,512); STRIDE=16
def rms8(spectra, w):
    """Score every location from the mean of its in-image 8-neighbour subset."""
    step = w // STRIDE
    ny, nx, _ = spectra.shape
    out = np.full((ny, nx), np.nan, np.float32)
    counts = np.zeros((ny, nx), np.uint8)
    if step < 1:
        return out, counts

    ref_sum = np.zeros_like(spectra, dtype=np.float32)
    # NW, N, NE, W, E, SW, S and SE, one window width from the target.
    for dy, dx in ((-step, -step), (-step, 0), (-step, step),
                   (0, -step),                  (0, step),
                   (step, -step),  (step, 0),   (step, step)):
        target_y = slice(max(0, -dy), min(ny, ny - dy))
        target_x = slice(max(0, -dx), min(nx, nx - dx))
        ref_y = slice(max(0, dy), min(ny, ny + dy))
        ref_x = slice(max(0, dx), min(nx, nx + dx))
        ref_sum[target_y, target_x] += spectra[ref_y, ref_x]
        counts[target_y, target_x] += 1

    valid = counts > 0
    reference_mean = np.zeros_like(spectra, dtype=np.float32)
    reference_mean[valid] = ref_sum[valid] / counts[valid, None]
    out[valid] = np.sqrt(np.mean((spectra[valid] - reference_mean[valid]) ** 2, axis=-1))
    return out, counts
def main():
    (OUT/'maps').mkdir(parents=True,exist_ok=True); (OUT/'comparisons').mkdir(exist_ok=True); rows=[]
    oldN,oldS=prev.N,prev.SPACING
    for w in WS:
      maps={}
      for i in range(1,22):
        p=base.DATA/f'image{i}.png'; img=np.asarray(Image.open(p).convert('L'),dtype=np.float32); start=time.perf_counter(); prev.N=w; prev.SPACING=w
        # One adjacent reference only needs two window widths in each direction.
        if img.shape[0]<2*w or img.shape[1]<2*w:
          rows.append(dict(image_id=f'image{i}',window_size=w,stride=16,status='not evaluated',num_valid_locations=0,anomaly_min=None,anomaly_max=None,anomaly_mean=None,anomaly_median=None,anomaly_std=None,seconds=0)); continue
        r=prev.analyze(img,workers=1,batch=32,also_2d=False); score, ref_counts=rms8(r['spectra'],w); vals=score[np.isfinite(score)]; maps[i]=(img,r,score)
        np.savez_compressed(OUT/'maps'/f'image{i}_w{w}.npz',anomaly_map=score,x_centers=r['xs'],y_centers=r['ys'],window=w,stride=16,reference_spacing=w,reference_count=8,reference_counts=ref_counts)
        rows.append(dict(image_id=f'image{i}',window_size=w,stride=16,status='ok',num_valid_locations=len(vals),anomaly_min=float(vals.min()),anomaly_max=float(vals.max()),anomaly_mean=float(vals.mean()),anomaly_median=float(np.median(vals)),anomaly_std=float(vals.std()),seconds=time.perf_counter()-start)); print(i,w,flush=True)
      # all-image contact sheet; each image has its own robust display scale
      fig,axes=plt.subplots(7,3,figsize=(18,27),constrained_layout=True)
      for i,ax in zip(range(1,22),axes.flat):
        if i not in maps: ax.set_facecolor('black'); ax.set_title(f'image{i}: not evaluated'); ax.set_axis_off(); continue
        _,r,s=maps[i]; v=s[np.isfinite(s)]; lo,hi=np.percentile(v,[1,99]); im=ax.imshow(s,extent=base.extent(r),cmap='inferno',vmin=lo,vmax=hi); ax.set_title(f'image{i} n={len(v):,}'); ax.set_axis_off()
      fig.suptitle(f'Strict B method: W={w}px, stride=16px, 8 references at ±W, radial-spectrum RMS',fontsize=15,weight='bold'); fig.savefig(OUT/f'window_{w}_all_images.png',dpi=160); plt.close(fig)
      # per-image comparison built after all windows below
      maps=None
    prev.N,prev.SPACING=oldN,oldS
    with (OUT/'window_size_summary.csv').open('w',encoding='utf-8-sig',newline='') as f: writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    # load maps and create image-wise shared/local scale comparisons
    for i in range(1,22):
      img=np.asarray(Image.open(base.DATA/f'image{i}.png').convert('L')); items=[]
      for w in WS:
        p=OUT/'maps'/f'image{i}_w{w}.npz'
        if p.exists():
          with np.load(p) as z: items.append((w,z['anomaly_map'],z['x_centers'],z['y_centers']))
      if not items: continue
      pool=np.concatenate([a[np.isfinite(a)] for _,a,_,_ in items]); lo,hi=np.percentile(pool,[1,99])
      for shared in (False,True):
        fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True);axes[0,0].imshow(img,cmap='gray');axes[0,0].set_title('original');axes[0,0].set_axis_off()
        for ax,(w,a,x,y) in zip(axes.flat[1:],items):
          v=a[np.isfinite(a)]; l,h=(lo,hi) if shared else np.percentile(v,[1,99]); ax.imshow(a,extent=[x[0]-8,x[-1]+8,y[-1]+8,y[0]-8],cmap='inferno',vmin=l,vmax=h);ax.set_title(f'W={w}px');ax.set_axis_off()
        fig.suptitle(f'image{i}: '+('shared scale across windows' if shared else 'individual scale'));fig.savefig(OUT/'comparisons'/f'image{i}_window_comparison_{"shared" if shared else "individual"}.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));
    for w in WS:
      q=[r for r in rows if r['window_size']==w and r['status']=='ok'];ax.scatter([w]*len(q),[r['num_valid_locations'] for r in q],alpha=.6)
    ax.set(xlabel='window size (px)',ylabel='valid locations (at least 1 reference)');fig.savefig(OUT/'window_size_valid_locations.png',dpi=150,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));
    for w in WS:
      q=[r['seconds'] for r in rows if r['window_size']==w and r['status']=='ok'];ax.bar(str(w),np.median(q) if q else 0)
    ax.set(xlabel='window size (px)',ylabel='median processing seconds / image');fig.savefig(OUT/'window_size_runtime_summary.png',dpi=150,bbox_inches='tight');plt.close(fig)
    (OUT/'report.md').write_text('## Strict B8 radial-FFT maps\n\nEach block is mean-subtracted, Hann-windowed, and transformed with rFFT2. Power is normalized by total in-band power then radial-bin averaged; log10 power spectra over 0.01–0.50 cycles/pixel are compared. Score = RMS(target spectrum − mean of 8 reference spectra). Only all-8-reference locations are valid.\n',encoding='utf-8')
    (OUT/'report.md').write_text(
        '## Available-neighbour radial-FFT maps\n\n'
        'Each block is mean-subtracted, Hann-windowed, and transformed with rFFT2. '
        'Power is normalized by total in-band power then radial-bin averaged. '
        'Score = RMS(target spectrum minus the mean of available surrounding reference spectra). '
        'The candidates are NW, N, NE, W, E, SW, S and SE at one window-width separation; '
        'at edges and corners, only in-image candidates are used.\n',
        encoding='utf-8',
    )
if __name__=='__main__':main()
