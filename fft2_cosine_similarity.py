"""Single-file visual-inspection workflow.

Run: python fft2_cosine_similarity.py
It performs global/local 2D FFT comparison, reference-peak matching, LabelMe
rectangle comparison, and the follow-up 1D FFT period check.  Dependencies:
numpy, scipy, matplotlib, and opencv-python.  No training or decision threshold
is applied.
"""
import argparse
import csv
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from time import perf_counter
import numpy as np
import cv2
from scipy.sparse import csr_matrix
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# This file is deliberately self-contained.  It is the single entry point for
# the inspection workflow, so it must not depend on the earlier experiment
# scripts that remain in this directory as historical references.
ROOT = Path(__file__).resolve().parent
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def read_gray(path):
    """Read a finite grayscale image without relying on a separate utility."""
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8),
                         cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    if image is None or not np.isfinite(image).all():
        raise ValueError(f'Cannot read finite grayscale image: {path}')
    return image


def write_csv(path, rows):
    """Write UTF-8 CSV only when there is data to write."""
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def intersect(a, b):
    return area((max(a[0], b[0]), max(a[1], b[1]),
                 min(a[2], b[2]), min(a[3], b[3])))


def load_labels(path, shape):
    """Load validated LabelMe rectangle annotations; no labels means unknown."""
    path = Path(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding='utf-8'))
    if (data['imageHeight'], data['imageWidth']) != shape:
        raise ValueError(f'{path.name}: annotation/image dimensions differ')
    labels = []
    for index, item in enumerate(data['shapes']):
        if item['shape_type'] != 'rectangle' or len(item['points']) != 2:
            raise ValueError(f'{path.name}: unsupported annotation shape')
        points = np.array(item['points'], dtype=float)
        if not np.isfinite(points).all():
            raise ValueError(f'{path.name}: nonfinite annotation')
        x0, y0 = points.min(0)
        x1, y1 = points.max(0)
        box = (float(x0), float(y0), float(x1), float(y1))
        if area(box) <= 0 or x0 < 0 or y0 < 0 or x1 > shape[1] or y1 > shape[0]:
            raise ValueError(f'{path.name}: invalid rectangle')
        labels.append(dict(id=f'{path.stem}_{index + 1:02d}', label=item['label'], box=box))
    return labels


def nearby(rows, label, labels, n):
    """Return nearby unlabeled blocks for comparison, never inferred normals."""
    box = label['box']
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    eligible = []
    for row in rows:
        block = (row['x'], row['y'], row['x'] + n, row['y'] + n)
        distance = np.hypot((block[0] + block[2]) / 2 - cx,
                            (block[1] + block[3]) / 2 - cy)
        if distance <= 3 * n and all(intersect(block, other['box']) == 0 for other in labels):
            eligible.append((distance, row))
    return [row for _, row in sorted(eligible, key=lambda item: (
        item[0], item[1]['block_row'], item[1]['block_col']))[:12]]


def overlap_map(shape, rows, key):
    """Average finite block values over their covered pixels."""
    total = np.zeros((shape[0] + 1, shape[1] + 1))
    count = np.zeros_like(total)
    for row in rows:
        value = row.get(key)
        if value is None or not np.isfinite(value):
            continue
        # In-memory records use x/y, while records reloaded from CSV use
        # x_start/y_start strings.  Supporting both keeps report generation
        # in the same single-file workflow.
        x = int(row.get('x', row.get('x_start')))
        y = int(row.get('y', row.get('y_start')))
        n = int(row['block_size'])
        for yy, xx, sign in ((y, x, 1), (y + n, x, -1),
                             (y, x + n, -1), (y + n, x + n, 1)):
            total[yy, xx] += sign * value
            count[yy, xx] += sign
    total = total.cumsum(0).cumsum(1)[:-1, :-1]
    count = count.cumsum(0).cumsum(1)[:-1, :-1]
    return np.divide(total, count, out=np.full(shape, np.nan), where=count > 0), count

# Include a larger window to test whether finer frequency resolution preserves
# the lattice periodicity better. All sizes use 50% overlap.
SIZES=(512,256,128)


def peak_candidates(raw, dc_radius, peak_percentile, local_max_size):
    """Return local maxima of an fftshifted amplitude spectrum outside DC."""
    h,w=raw.shape; cy,cx=h//2,w//2
    yy,xx=np.indices(raw.shape)
    eligible=(xx-cx)**2+(yy-cy)**2 > dc_radius**2
    values=raw[eligible]
    if not len(values) or float(values.max())<=1e-15: return []
    threshold=float(np.percentile(values,peak_percentile))
    local=cv2.dilate(raw,np.ones((local_max_size,local_max_size),np.uint8))
    found=eligible & (raw>=threshold) & (raw>=local-1e-12) & (raw>1e-15)
    return sorted([(int(y),int(x)) for y,x in np.argwhere(found)],
                  key=lambda p:(-float(raw[p]),p))


def reference_peaks(raw, args):
    """Select up to 20 local maxima as complete conjugate pairs.

    dx/dy are saved in full-image FFT bins. Matching uses fx/fy so the same
    periodic component maps correctly to the smaller block FFT grid.
    """
    h,w=raw.shape; cy,cx=h//2,w//2
    yy,xx=np.indices(raw.shape)
    in_center=(np.abs(xx-cx)<=args.peak_radius)&(np.abs(yy-cy)<=args.peak_radius)
    candidates=set(p for p in peak_candidates(raw,args.peak_dc_radius,args.peak_percentile,args.peak_local_max_size)
                   if in_center[p])
    selected=[]; selected_coords=[]; visited=set()
    for y,x in sorted(candidates,key=lambda p:(-float(raw[p]),p)):
        partner=((2*cy-y)%h,(2*cx-x)%w)
        key=tuple(sorted(((y,x),partner)))
        if key in visited: continue
        visited.add(key)
        if partner==(y,x) or partner not in candidates or len(selected)+2>args.reference_peak_count: continue
        if any(np.hypot(py-sy,px-sx)<args.peak_min_distance
               for py,px in key for sy,sx in selected_coords): continue
        pair_id=len(selected)//2+1
        for py,px in key:
            selected.append(dict(peak_id=len(selected)+1,pair_id=pair_id,
                dx=int(px-cx),dy=int(py-cy),fx=float((px-cx)/w),fy=float((py-cy)/h),
                amplitude=float(raw[py,px])))
            selected_coords.append((py,px))
    return selected


def block_peak_matches(raw, references, args):
    """Match every reference frequency to one detected local block peak."""
    h,w=raw.shape; cy,cx=h//2,w//2
    candidates=peak_candidates(raw,args.peak_dc_radius,args.peak_percentile,args.peak_local_max_size)
    matched=[]; used_candidates=set()
    for ref in references:
        ey,ex=cy+ref['fy']*h,cx+ref['fx']*w
        hits=[(float(np.hypot(x-ex,y-ey)),y,x) for y,x in candidates
              if (y,x) not in used_candidates and np.hypot(x-ex,y-ey)<=args.peak_tolerance]
        if hits:
            distance,y,x=min(hits)
            used_candidates.add((y,x))
            matched.append(dict(reference_peak_id=ref['peak_id'],pair_id=ref['pair_id'],matched=True,
                expected_dx=float(ex-cx),expected_dy=float(ey-cy),block_dx=int(x-cx),block_dy=int(y-cy),distance_bins=distance))
        else:
            matched.append(dict(reference_peak_id=ref['peak_id'],pair_id=ref['pair_id'],matched=False,
                expected_dx=float(ex-cx),expected_dy=float(ey-cy),block_dx=None,block_dy=None,distance_bins=None))
    return matched,candidates


def fft_amplitude(gray, subtract_mean=True):
    data=gray.astype(np.float64)
    if subtract_mean: data=data-data.mean()
    window=np.outer(np.hanning(data.shape[0]),np.hanning(data.shape[1]))
    raw=np.abs(np.fft.fftshift(np.fft.fft2(data*window)))
    # Linear cosine is scale invariant. Use a common amplitude unit BEFORE log1p.
    return raw, raw/window.sum()


@lru_cache(maxsize=64)
def area_weights(length, grid_size, fmax):
    """Frequency-cell area averaging: conserve narrow peak contribution, no peak skipping.
    Native FFT bin centers are fftshift(fftfreq(N)); cells have width 1/N.
    Common cells have width 2*fmax/(grid_size-1). Sparse rows sum to one.
    """
    native=np.fft.fftshift(np.fft.fftfreq(length))
    common=np.linspace(-fmax,fmax,grid_size)
    step=common[1]-common[0]
    left=np.maximum(common[:,None]-step/2,native[None,:]-.5/length)
    right=np.minimum(common[:,None]+step/2,native[None,:]+.5/length)
    overlap=np.maximum(0,right-left)
    if np.any(overlap.sum(1)<=0): raise ValueError('Frequency grid outside native spectrum')
    return csr_matrix(overlap/overlap.sum(1,keepdims=True))


def common_spectrum(amplitude, grid_size=129, fmax=.125):
    wy=area_weights(amplitude.shape[0],grid_size,fmax)
    wx=area_weights(amplitude.shape[1],grid_size,fmax)
    return (wx @ (wy @ amplitude).T).T


def comparison_mask(grid_size,dc_bins):
    y,x=np.indices((grid_size,grid_size)); c=grid_size//2
    return (np.abs(x-c)>dc_bins) | (np.abs(y-c)>dc_bins)


def vector(spectrum,mask,mode):
    a=spectrum[mask]
    return np.log1p(a) if mode=='log' else a


def cosine(a,b):
    na,nb=float(np.linalg.norm(a)),float(np.linalg.norm(b))
    if not np.isfinite(na+nb) or min(na,nb)<=1e-15: return None
    return float(np.clip(np.dot(a/na,b/nb),0,1))


def percentile(values,value):
    if value is None or not len(values): return None
    # Midrank empirical CDF; lower = relatively less similar. Ties get same value.
    return float(100*(np.sum(values<value)+.5*np.sum(values==value))/len(values))


def raw_energy(raw,dc_limit):
    fy=np.fft.fftshift(np.fft.fftfreq(raw.shape[0]))
    fx=np.fft.fftshift(np.fft.fftfreq(raw.shape[1]))
    keep=(np.abs(fy[:,None])>dc_limit)|(np.abs(fx[None,:])>dc_limit)
    return float(np.sum(raw[keep]**2))


def labels_on(ax,labels):
    for l in labels:
        x,y,x1,y1=l['box']; ax.add_patch(Rectangle((x,y),x1-x,y1-y,fill=False,edgecolor='red',lw=1.5))
        ax.text(x,y,l['id']+' '+l['label'],color='red',fontsize=6,backgroundcolor='white')


def save_global(gray,raw,common,mask,labels,dest,args):
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,show,extent,title in [
        (axes[0],np.log1p(raw),[-.5,.5,.5,-.5],'Full global FFT: log1p(raw amplitude)'),
        (axes[1],np.ma.masked_where(~mask,np.log1p(common)),[-args.fmax,args.fmax,args.fmax,-args.fmax],'Common frequency cells; DC excluded')]:
        data=show.compressed() if np.ma.isMaskedArray(show) else show.ravel()
        lo,hi=np.percentile(data,[1,99.8]); hi=max(hi,lo+1e-10)
        im=ax.imshow(show,cmap='magma',extent=extent,vmin=lo,vmax=hi,interpolation='nearest')
        ax.set(title=title,xlabel='fx [cycles/pixel]',ylabel='fy [cycles/pixel]'); fig.colorbar(im,ax=ax,shrink=.75)
    fig.tight_layout(); fig.savefig(dest/'global_fft.png',dpi=110); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,7)); ax.imshow(gray,cmap='gray'); labels_on(ax,labels)
    ax.set_title('Original + annotated defect rectangles (unlabeled = unknown)')
    fig.tight_layout(); fig.savefig(dest/'original_labels.png',dpi=110); plt.close(fig)
    # Save an actual grayscale image at its original pixel dimensions.
    import cv2
    success,encoded=cv2.imencode('.png',gray)
    if not success: raise ValueError('PNG encode failed')
    (dest/'original.png').write_bytes(encoded.tobytes())


def maps(gray,rows,labels,dest,n,mode,primary,include_overlay=True):
    key='cosine_similarity_'+mode
    m,_=overlap_map(gray.shape,rows,key)
    coverage=overlap_map(gray.shape,rows,'block_mean')[1]
    suffix='' if mode==primary else '_'+mode
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    for ax in axes:
        ax.set_facecolor('#bdbdbd')
        im=ax.imshow(np.ma.masked_invalid(m),cmap='RdYlBu',vmin=0,vmax=1)
    labels_on(axes[1],labels)
    axes[0].set_title('Similarity (overlap averaged)\nGrey = undefined / unanalysed',fontsize=10)
    axes[1].set_title('Similarity + user defect labels\nLow similarity does NOT mean defect',fontsize=10)
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7,label='Cosine similarity; low=red')
    fig.suptitle(f'{dest.name} | {n}px stride={n//2} | {mode}')
    fig.savefig(dest/('similarity_map'+suffix+'.png'),dpi=110); plt.close(fig)
    if not include_overlay: return
    fig,ax=plt.subplots(figsize=(16,12)); ax.imshow(gray,cmap='gray')
    ax.imshow(np.ma.masked_where(coverage>0,np.ones(gray.shape)),cmap='Greys',vmin=0,vmax=1,alpha=.4)
    im=ax.imshow(np.ma.masked_invalid(m),cmap='RdYlBu',vmin=0,vmax=1,alpha=.45)
    for r in rows:
        x,y=r['x'],r['y']; value=r[key]
        ax.add_patch(Rectangle((x-.5,y-.5),n,n,fill=False,edgecolor='red' if r['label']!='unknown' else 'white',lw=.6 if r['label']!='unknown' else .15))
        ax.text(x+n/2,y+n/2,f'{value:.3f}' if value is not None else 'NA',ha='center',va='center',fontsize=3.8 if n==128 else 5.5,
                color='black',bbox=dict(facecolor='white',alpha=.65,edgecolor='none',pad=.1))
    labels_on(ax,labels); fig.colorbar(im,ax=ax,shrink=.7)
    ax.set_title(f'{dest.name}: {n}px, {mode}; number=individual block similarity\nRed block outline=intersects defect bbox; all other blocks=unknown')
    fig.tight_layout(); fig.savefig(dest/('overlay'+suffix+'.png'),dpi=160); plt.close(fig)


def distribution(rows,n,mode,out,primary):
    key='cosine_similarity_'+mode
    groups=[np.array([r[key] for r in rows if r['block_size']==n and (r['label']!='unknown')==positive and r[key] is not None]) for positive in (True,False)]
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    axes[0].boxplot([a if len(a) else [np.nan] for a in groups],tick_labels=['labeled bbox','unknown'],showfliers=False)
    rng=np.random.default_rng(0)
    for i,(name,a,color) in enumerate(zip(('labeled bbox','unknown'),groups,('crimson','steelblue'))):
        axes[1].hist(a,bins=np.linspace(0,1,31),density=bool(len(a)),alpha=.5,label=f'{name} (n={len(a)})',color=color)
        axes[2].scatter(rng.normal(i,.06,len(a)),a,s=4,alpha=.25,color=color)
    axes[1].legend(); axes[1].set(xlabel='Cosine similarity',ylabel='Density',xlim=(0,1))
    axes[0].set(ylabel='Cosine similarity',ylim=(0,1)); axes[2].set(xticks=[0,1],xticklabels=['labeled bbox','unknown'],ylim=(0,1))
    fig.suptitle(f'{n}px / {mode}: unknown is NOT normal; overlapping blocks are not independent')
    fig.tight_layout(); suffix='' if mode==primary else '_'+mode
    fig.savefig(out/f'scratch_vs_unknown_{n}{suffix}.png',dpi=120); plt.close(fig)


def peak_score_visuals(gray, rows, raw, references, labels, dest, n, args):
    """Save the requested score maps, reference view, block FFT examples, and scatter."""
    score_map,_=overlap_map(gray.shape,rows,'peak_match_score')
    fig,ax=plt.subplots(figsize=(12,9)); ax.set_facecolor('#bdbdbd')
    im=ax.imshow(np.ma.masked_invalid(score_map),cmap='viridis',vmin=0,vmax=1)
    labels_on(ax,labels); ax.set_title(f'Peak-match score heatmap ({n}px; overlap averaged)')
    fig.colorbar(im,ax=ax,label='matched reference peaks / reference peaks')
    fig.tight_layout(); fig.savefig(dest/'peak_score_heatmap.png',dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(16,12)); ax.imshow(gray,cmap='gray')
    im=ax.imshow(np.ma.masked_invalid(score_map),cmap='viridis',vmin=0,vmax=1,alpha=.38)
    for r in rows:
        ax.add_patch(Rectangle((r['x']-.5,r['y']-.5),n,n,fill=False,edgecolor='white',lw=.25))
        ax.text(r['x']+n/2,r['y']+n/2,f"{r['matched_peak_count']}/{r['reference_peak_count']}",ha='center',va='center',fontsize=4 if n==128 else 5.5,
                color='black',bbox=dict(facecolor='white',alpha=.7,edgecolor='none',pad=.1))
    labels_on(ax,labels); fig.colorbar(im,ax=ax,label='Peak-match score')
    ax.set_title(f'Peak-match scores by block ({n}px)'); fig.tight_layout(); fig.savefig(dest/'peak_score_text.png',dpi=160); plt.close(fig)
    h,w=raw.shape; cy,cx=h//2,w//2; radius=args.peak_radius
    y0,y1=max(0,cy-radius),min(h,cy+radius+1); x0,x1=max(0,cx-radius),min(w,cx+radius+1)
    display=np.log1p(raw[y0:y1,x0:x1]); lo,hi=np.percentile(display,[1,99.8]); hi=max(hi,lo+1e-10)
    fig,ax=plt.subplots(figsize=(9,8)); ax.imshow(display,cmap='magma',extent=[x0-cx-.5,x1-cx-.5,y1-cy-.5,y0-cy-.5],vmin=lo,vmax=hi)
    ax.scatter([p['dx'] for p in references],[p['dy'] for p in references],s=55,facecolors='none',edgecolors='cyan',label=f'Reference peaks (n={len(references)})')
    for p in references: ax.annotate(str(p['peak_id']),(p['dx'],p['dy']),xytext=(3,3),textcoords='offset points',color='cyan',fontsize=7)
    ax.set(title='Reference FFT peaks (relative FFT bins)',xlabel='dx [full-image FFT bins]',ylabel='dy [full-image FFT bins]'); ax.legend(loc='upper right'); fig.tight_layout(); fig.savefig(dest/'reference_fft_peaks.png',dpi=150); plt.close(fig)
    selected=sorted(rows,key=lambda r:(r['peak_match_score'],r['block_id']))
    selected=(selected[:max(1,args.debug_blocks//2)]+selected[-max(1,args.debug_blocks-args.debug_blocks//2):])[:args.debug_blocks]
    for r in selected:
        block=gray[r['y']:r['y']+n,r['x']:r['x']+n]; br,_=fft_amplitude(block,not args.keep_mean)
        matches,_=block_peak_matches(br,references,args); d=br.shape[0]//2
        shown=np.log1p(br); lo,hi=np.percentile(shown,[1,99.8]); hi=max(hi,lo+1e-10)
        fig,ax=plt.subplots(figsize=(8,7)); ax.imshow(shown,cmap='magma',vmin=lo,vmax=hi)
        for m in matches:
            color='lime' if m['matched'] else 'red'; ex=d+m['expected_dx']; ey=d+m['expected_dy']
            ax.scatter(ex,ey,s=50,facecolors='none',edgecolors=color,linewidths=1.2)
            if m['matched']: ax.scatter(d+m['block_dx'],d+m['block_dy'],s=12,c=color)
        ax.set(title=f"{r['block_id']} | score={r['peak_match_score']:.2f}; green=matched, red=unmatched\nCircles=expected reference frequency; dots=matched local peaks",xlabel='FFT x bin',ylabel='FFT y bin')
        fig.tight_layout(); fig.savefig(dest/f"debug_block_fft_{r['block_id']}.png",dpi=150); plt.close(fig)
    x=np.array([r['cosine_similarity'] for r in rows if r['cosine_similarity'] is not None]); y=np.array([r['peak_match_score'] for r in rows if r['cosine_similarity'] is not None])
    fig,ax=plt.subplots(figsize=(7,6)); ax.scatter(x,y,s=14,alpha=.65); ax.set(xlabel='cosine_similarity',ylabel='peak_match_score',xlim=(0,1),ylim=(0,1),title=f'Cosine similarity vs peak-match score ({n}px)'); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(dest/'cosine_vs_peak_score.png',dpi=150); plt.close(fig)


def refresh_peak_text_outputs(output_dir, input_dir):
    """Regenerate original-image score labels from saved block CSVs."""
    import csv
    output_dir,input_dir=Path(output_dir),Path(input_dir)
    all_rows=list(csv.DictReader((output_dir/'blocks_peak_scores.csv').open(encoding='utf-8-sig',newline='')))
    for image_name in sorted({r['image_name'] for r in all_rows}):
        source=input_dir/image_name
        gray=read_gray(source); labels=load_labels(source.with_suffix('.json'),gray.shape)
        for n in SIZES:
            rows=[r for r in all_rows if r['image_name']==image_name and int(r['block_size'])==n]
            if not rows: continue
            rows=[dict(r,x=int(r['x']),y=int(r['y']),x_start=int(r['x_start']),y_start=int(r['y_start']),block_size=n,
                       matched_peak_count=int(r['matched_peak_count']),
                       reference_peak_count=int(r['reference_peak_count']),
                       peak_match_score=float(r['peak_match_score'])) for r in rows]
            score_map,_=overlap_map(gray.shape,rows,'peak_match_score')
            dest=output_dir/str(n)/Path(image_name).stem; dest.mkdir(parents=True,exist_ok=True)
            fig,ax=plt.subplots(figsize=(16,12)); ax.imshow(gray,cmap='gray')
            im=ax.imshow(np.ma.masked_invalid(score_map),cmap='viridis',vmin=0,vmax=1,alpha=.38)
            for r in rows:
                ax.add_patch(Rectangle((r['x']-.5,r['y']-.5),n,n,fill=False,edgecolor='white',lw=.25))
                ax.text(r['x']+n/2,r['y']+n/2,
                        f"{r['matched_peak_count']}/{r['reference_peak_count']}",
                        ha='center',va='center',fontsize=4 if n==128 else 5.5,
                        color='black',bbox=dict(facecolor='white',alpha=.7,edgecolor='none',pad=.1))
            labels_on(ax,labels); fig.colorbar(im,ax=ax,label='Peak-match score (matched / reference)')
            ax.set_title(f'Original + matched reference peaks per block ({n}px; label = matched/total)')
            fig.tight_layout(); fig.savefig(dest/'peak_score_text.png',dpi=160); plt.close(fig)


def similarity_map_gallery(output_dir, image_paths, block_size=256, mode='log', input_dir=None):
    """Create a gallery from the labeled right-hand cosine map only."""
    import csv
    output_dir=Path(output_dir); input_dir=Path(input_dir) if input_dir else None
    name=f'similarity_map_{mode}_{block_size}'
    panel_dir=output_dir/f'{name}_right'; panel_dir.mkdir(parents=True,exist_ok=True)
    key=f'cosine_similarity_{mode}'; available=[]
    def image_order(path):
        suffix=path.stem.removeprefix('image')
        return (0,int(suffix)) if suffix.isdigit() else (1,path.stem.casefold())
    for path in sorted(image_paths,key=image_order):
        csv_path=output_dir/str(block_size)/path.stem/'blocks.csv'
        if not csv_path.is_file(): continue
        with csv_path.open(encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream))
        if not rows or key not in rows[0]: continue
        rows=[dict(r,x_start=int(r['x_start']),y_start=int(r['y_start']),block_size=int(r['block_size']),
                   **{key:float(r[key]) if r[key] else np.nan}) for r in rows]
        gray=read_gray((input_dir/path.name) if input_dir else path)
        labels=load_labels((input_dir/path.stem).with_suffix('.json'),gray.shape) if input_dir else []
        score_map,_=overlap_map(gray.shape,rows,key)
        fig,ax=plt.subplots(figsize=(10,7)); ax.set_facecolor('#bdbdbd')
        im=ax.imshow(np.ma.masked_invalid(score_map),cmap='RdYlBu',vmin=0,vmax=1)
        labels_on(ax,labels)
        ax.set_title(f'{path.name} | {block_size}px | {mode}\nSimilarity + defect labels; low similarity does not mean defect')
        ax.set_xlabel('x [pixels]'); ax.set_ylabel('y [pixels]')
        fig.colorbar(im,ax=ax,shrink=.85,label='Cosine similarity (0–1)')
        fig.tight_layout(); fig.savefig(panel_dir/f'{path.stem}.png',dpi=150); plt.close(fig)
        available.append(path)
    if not available: return
    cols=4; rows_count=6
    fig,axes=plt.subplots(rows_count,cols,figsize=(24,rows_count*4.2),squeeze=False)
    for ax,path in zip(axes.flat,available):
        ax.imshow(plt.imread(panel_dir/f'{path.stem}.png')); ax.set_title(path.name,fontsize=11); ax.axis('off')
    for ax in list(axes.flat)[len(available):]: ax.axis('off')
    fig.suptitle(f'{block_size}px / {mode}: labeled similarity maps (right panel only)',fontsize=17)
    fig.tight_layout(rect=(0,0,1,.98)); fig.savefig(output_dir/f'{name}_contact_sheet.png',dpi=150); plt.close(fig)
    cards=[]
    for path in available:
        rel=f'{name}_right/{path.stem}.png'
        cards.append(f'<a class="card" href="{rel}"><b>{path.name}</b><img loading="lazy" src="{rel}" alt="{path.name} labeled cosine similarity map"></a>')
    page=f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{block_size}px {mode} similarity maps</title><style>body{{font-family:system-ui,sans-serif;margin:24px;background:#f4f5f7}}h1{{margin-bottom:4px}}p{{color:#555}}.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}.card{{background:white;padding:8px;border-radius:8px;color:#222;text-decoration:none;box-shadow:0 1px 5px #bbb;min-width:0}}img{{display:block;width:100%;margin-top:8px}}</style><h1>{block_size}px / {mode} コサイン類似度マップ（右パネルのみ）</h1><p>全{len(available)}画像。横4列×縦6行で表示しています。カードを選ぶと個別画像を開きます。</p><div class="grid">{''.join(cards)}</div></html>'''
    (output_dir/f'{name}_gallery.html').write_text(page,encoding='utf-8')


def classify_similarity_blocks(output_dir, input_dir, image_paths):
    """Apply the requested fixed-high / per-image lower-quintile 3-class rule."""
    import csv
    output_dir,input_dir=Path(output_dir),Path(input_dir)
    source=output_dir/'blocks_peak_scores.csv'
    with source.open(encoding='utf-8-sig',newline='') as stream:
        all_rows=list(csv.DictReader(stream))
    grouped={}
    for row in all_rows:
        grouped.setdefault((row['image_name'],int(row['block_size'])),[]).append(row)
    classified=[]; summaries=[]
    class_names=('grid','background','review')
    class_labels={'grid':'grid_candidate','background':'background_candidate','review':'review_candidate'}
    class_colors=('#20b45b','#ed6548','#9b9b9b')
    for (image_name,n),rows in sorted(grouped.items()):
        valid=[float(r['cosine_similarity_log']) for r in rows if r['cosine_similarity_log'] not in ('',None)]
        if not valid: continue
        p20=float(np.percentile(valid,20)); group_results=[]
        for row in rows:
            sim=float(row['cosine_similarity_log']) if row['cosine_similarity_log'] not in ('',None) else None
            if sim is None: category='review'
            elif sim>=.90: category='grid'
            elif sim<=p20: category='background'
            else: category='review'
            result=dict(row,similarity_mode='log',image_p20_threshold=f'{p20:.10g}',
                        similarity_class=category,similarity_class_label=class_labels[category])
            classified.append(result); group_results.append(result)
        counts={name:sum(1 for r in group_results if r['similarity_class']==name) for name in class_names}
        summaries.append(dict(image_name=image_name,block_size=n,similarity_mode='log',fixed_grid_threshold=.90,
            image_p20_threshold=p20,block_count=len(rows),grid_count=counts['grid'],background_count=counts['background'],
            review_count=counts['review'],grid_ratio=counts['grid']/len(rows),background_ratio=counts['background']/len(rows),
            review_ratio=counts['review']/len(rows),thresholds_overlap=bool(p20>=.90)))
        image_path=input_dir/image_name; gray=read_gray(image_path)
        labels=load_labels(image_path.with_suffix('.json'),gray.shape)
        # Overlapping blocks are summarized per pixel by their majority class.
        votes=np.zeros((3,gray.shape[0],gray.shape[1]),dtype=np.uint8)
        for r in rows:
            score=float(r['cosine_similarity_log']) if r['cosine_similarity_log'] not in ('',None) else None
            category='review' if score is None else ('grid' if score>=.90 else ('background' if score<=p20 else 'review'))
            ci=class_names.index(category); x,y=int(r['x']),int(r['y']); x1=min(gray.shape[1],x+n); y1=min(gray.shape[0],y+n)
            votes[ci,y:y1,x:x1]+=1
        covered=votes.sum(axis=0)>0; class_map=np.argmax(votes,axis=0)
        rgba=np.zeros((*gray.shape,4),dtype=float)
        colors=np.array([[32/255,180/255,91/255],[237/255,101/255,72/255],[155/255,155/255,155/255]])
        rgba[covered,:3]=colors[class_map[covered]]; rgba[covered,3]=.34
        dest=output_dir/str(n)/Path(image_name).stem; dest.mkdir(parents=True,exist_ok=True)
        fig,ax=plt.subplots(figsize=(10,7)); ax.imshow(gray,cmap='gray'); ax.imshow(rgba)
        for r in rows:
            ax.add_patch(Rectangle((int(r['x'])-.5,int(r['y'])-.5),n,n,fill=False,edgecolor='white',lw=.25))
        labels_on(ax,labels)
        from matplotlib.patches import Patch
        legend=[Patch(facecolor=class_colors[i],alpha=.6,label=class_labels[name]) for i,name in enumerate(class_names)]
        ax.legend(handles=legend,loc='upper right',fontsize=8)
        ax.set_title(f'{image_name} | {n}px | log cosine 3-class map | p20={p20:.3f}')
        ax.axis('off'); fig.tight_layout(); fig.savefig(dest/'similarity_3class.png',dpi=150); plt.close(fig)
        with (dest/'blocks_similarity_classes.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(classified[-len(rows)].keys())); writer.writeheader()
            writer.writerows(classified[-len(rows):])
    if classified:
        with (output_dir/'similarity_3class_blocks.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(classified[0].keys())); writer.writeheader(); writer.writerows(classified)
        with (output_dir/'similarity_3class_summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(summaries[0].keys())); writer.writeheader(); writer.writerows(summaries)
        (output_dir/'similarity_3class_summary.json').write_text(json.dumps(dict(similarity_mode='log',rule=dict(grid='similarity >= 0.90',background='similarity < 0.90 and similarity <= per-image p20',review='otherwise; grid takes precedence if thresholds overlap'),images=summaries),ensure_ascii=False,indent=2),encoding='utf-8')
    def image_order(path):
        suffix=path.stem.removeprefix('image')
        return (0,int(suffix)) if suffix.isdigit() else (1,path.stem.casefold())
    for n in SIZES:
        available=sorted((p for p in image_paths if (output_dir/str(n)/p.stem/'similarity_3class.png').is_file()),key=image_order)
        if not available: continue
        cols=4; rows_count=6
        fig,axes=plt.subplots(rows_count,cols,figsize=(24,rows_count*4.2),squeeze=False)
        for ax,path in zip(axes.flat,available):
            ax.imshow(plt.imread(output_dir/str(n)/path.stem/'similarity_3class.png')); ax.set_title(path.name,fontsize=11); ax.axis('off')
        for ax in list(axes.flat)[len(available):]: ax.axis('off')
        fig.suptitle(f'{n}px 3-class cosine similarity maps | grid / background / review',fontsize=16)
        fig.tight_layout(rect=(0,0,1,.98)); fig.savefig(output_dir/f'similarity_3class_{n}_contact_sheet.png',dpi=140); plt.close(fig)
    write_interactive_threshold_viewer(output_dir,input_dir,image_paths,classified,summaries)


def write_interactive_threshold_viewer(output_dir, input_dir, image_paths, classified, summaries):
    """Write a local HTML gallery whose block classes respond to a threshold slider."""
    output_dir,input_dir=Path(output_dir),Path(input_dir)
    summary_map={(r['image_name'],int(r['block_size'])):r for r in summaries}
    grouped={}
    for row in classified:
        grouped.setdefault((row['image_name'],int(row['block_size'])),[]).append(row)
    payload=[]
    def image_order(path):
        suffix=path.stem.removeprefix('image')
        return (0,int(suffix)) if suffix.isdigit() else (1,path.stem.casefold())
    for path in sorted(image_paths,key=image_order):
        try: gray=read_gray(input_dir/path.name)
        except (OSError,ValueError): continue
        item=dict(name=path.name,stem=path.stem,width=int(gray.shape[1]),height=int(gray.shape[0]),sizes={})
        for n in SIZES:
            key=(path.name,n)
            if key not in grouped: continue
            item['sizes'][str(n)]=dict(p20=float(summary_map[key]['image_p20_threshold']),
                blocks=[dict(x=int(r['x']),y=int(r['y']),s=float(r['cosine_similarity_log']))
                        for r in grouped[key] if r['cosine_similarity_log'] not in ('',None)])
        payload.append(item)
    data=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    template='''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>類似度しきい値ビューア</title><style>
body{font-family:system-ui,sans-serif;margin:20px;background:#f2f4f5;color:#202124}.controls{position:sticky;top:0;z-index:5;background:#fff;padding:14px 18px;border-radius:10px;box-shadow:0 2px 8px #0002}.line{display:flex;align-items:center;gap:18px;flex-wrap:wrap}.line label{font-weight:650}input[type=range]{width:min(48vw,560px)}select{font-size:16px;padding:5px}.legend{display:flex;gap:16px;margin-top:8px}.swatch{display:inline-block;width:13px;height:13px;margin-right:5px;vertical-align:-2px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-top:16px}.card{background:#fff;border-radius:8px;padding:8px;box-shadow:0 1px 5px #0002;min-width:0}.card h3{font-size:14px;margin:2px 0 4px}.meta{font-size:11px;color:#555;min-height:30px}.card canvas{display:block;width:100%;height:auto;background:#ddd}@media(max-width:900px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:520px){.grid{grid-template-columns:1fr}}
</style><main><h1>類似度しきい値で格子候補を確認</h1><div class="controls"><div class="line"><label>格子候補しきい値 <output id="thresholdValue">90%</output></label><input id="threshold" type="range" min="0" max="100" step="1" value="90"><label>ブロックサイズ</label><select id="blockSize"><option value="256">256 px</option><option value="128">128 px</option></select></div><div class="legend"><span><i class="swatch" style="background:#20b45b"></i>格子候補（類似度 ≥ しきい値）</span><span><i class="swatch" style="background:#ed6548"></i>背景候補（しきい値未満かつ画像内p20以下）</span><span><i class="swatch" style="background:#9b9b9b"></i>判定保留</span></div><small>スライダーを動かすと全画像を即時更新します。画像カードをクリックすると個別画像を開きます。重複ブロック領域は色が重なって見える場合があります。</small></div><div id="gallery" class="grid"></div></main><script>
const images=__DATA__;const slider=document.getElementById('threshold'),sizeSelect=document.getElementById('blockSize'),gallery=document.getElementById('gallery'),thresholdValue=document.getElementById('thresholdValue');
function render(){const threshold=Number(slider.value)/100,size=sizeSelect.value;thresholdValue.value=slider.value+'%';thresholdValue.textContent=slider.value+'%';gallery.replaceChildren();for(const item of images){const entry=item.sizes[size];if(!entry)continue;const card=document.createElement('a');card.className='card';card.href=size+'/'+item.stem+'/similarity_map_log.png';card.target='_blank';card.style.color='inherit';card.style.textDecoration='none';const title=document.createElement('h3');title.textContent=item.name;const meta=document.createElement('div');meta.className='meta';const canvas=document.createElement('canvas');const ctx=canvas.getContext('2d');const image=new Image();image.src=size+'/'+item.stem+'/original.png';image.onload=()=>{canvas.width=600;canvas.height=Math.round(600*item.height/item.width);ctx.drawImage(image,0,0,canvas.width,canvas.height);const sx=canvas.width/item.width,sy=canvas.height/item.height;let grid=0,bg=0,review=0;for(const b of entry.blocks){let kind;if(b.s>=threshold){kind=0;grid++}else if(b.s<=entry.p20){kind=1;bg++}else{kind=2;review++}const colors=['32,180,91','237,101,72','155,155,155'];ctx.fillStyle='rgba('+colors[kind]+',0.30)';ctx.fillRect(b.x*sx,b.y*sy,Number(size)*sx,Number(size)*sy);ctx.strokeStyle='rgba(255,255,255,0.75)';ctx.lineWidth=0.7;ctx.strokeRect(b.x*sx,b.y*sy,Number(size)*sx,Number(size)*sy)}meta.textContent='p20='+ (entry.p20*100).toFixed(1)+'% | 格子 '+grid+' / 背景 '+bg+' / 保留 '+review;};card.append(title,meta,canvas);gallery.append(card)}}
slider.addEventListener('input',render);sizeSelect.addEventListener('change',render);render();</script></html>'''
    template=template.replace('<option value="256">','<option value="512">512 px</option><option value="256">')
    (output_dir/'interactive_similarity_threshold.html').write_text(template.replace('__DATA__',data),encoding='utf-8')


def one_d_block_spectra(block, low_frequency_bins=1):
    """Mean row-wise x FFT and column-wise y FFT amplitudes; omit low bins."""
    values=block.astype(np.float64)
    sx=np.abs(np.fft.rfft(values,axis=1)).mean(axis=0)
    sy=np.abs(np.fft.rfft(values,axis=0)).mean(axis=1)
    return sx[low_frequency_bins:],sy[low_frequency_bins:]


def cosine_shape_similarity(a,b):
    """Cosine similarity compares spectrum shape independent of total strength."""
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    denom=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a,b)/denom) if denom>1e-15 else None


def _finite_stats(values):
    values=np.asarray([v for v in values if v is not None and np.isfinite(v)],dtype=float)
    if not len(values): return dict(mean=None,median=None,min=None,max=None)
    return dict(mean=float(values.mean()),median=float(np.median(values)),min=float(values.min()),max=float(values.max()))


def analyze_1d_fft_blocks(output_dir,input_dir,image_paths,top_k=10,low_frequency_bins=1):
    """Rank grid/uncertain blocks against per-image grid-only 1D FFT references."""
    import csv
    output_dir,input_dir=Path(output_dir),Path(input_dir)
    source=output_dir/'similarity_3class_blocks.csv'
    with source.open(encoding='utf-8-sig',newline='') as stream:
        classified=list(csv.DictReader(stream))
    grouped={}
    for row in classified:
        grouped.setdefault((row['image_name'],int(row['block_size'])),[]).append(row)
    result_root=output_dir/'results_1d_fft'; result_root.mkdir(parents=True,exist_ok=True)
    all_scores=[]; summaries=[]
    def image_order(path):
        suffix=path.stem.removeprefix('image')
        return (0,int(suffix)) if suffix.isdigit() else (1,path.stem.casefold())
    paths={p.name:p for p in image_paths}
    for (image_name,n),rows in sorted(grouped.items(),key=lambda v:(image_order(paths.get(v[0][0],Path(v[0][0]))),v[0][1])):
        image_path=paths.get(image_name,input_dir/image_name)
        gray=read_gray(image_path); height,width=gray.shape
        blocks=[]
        for row in rows:
            x,y=int(row['x']),int(row['y'])
            block=gray[y:y+n,x:x+n]
            if block.shape!=(n,n): continue
            sx,sy=one_d_block_spectra(block,low_frequency_bins)
            category={'grid':'grid','background':'background','review':'uncertain'}.get(row['similarity_class'],row['similarity_class'])
            blocks.append(dict(source=row,x=x,y=y,sx=sx,sy=sy,background_class=category))
        grids=[b for b in blocks if b['background_class']=='grid']
        backgrounds=[b for b in blocks if b['background_class']=='background']
        uncertain=[b for b in blocks if b['background_class']=='uncertain']
        dest=result_root/str(n)/image_path.stem; dest.mkdir(parents=True,exist_ok=True)
        fx=np.fft.rfftfreq(n)[low_frequency_bins:]; fy=np.fft.rfftfreq(n)[low_frequency_bins:]
        reference_x=np.mean([b['sx'] for b in grids],axis=0) if grids else None
        reference_y=np.mean([b['sy'] for b in grids],axis=0) if grids else None
        scores=[]
        for b in blocks:
            row=b['source']; eligible=b['background_class']!='background'
            sim_x=cosine_shape_similarity(reference_x,b['sx']) if eligible and reference_x is not None else None
            sim_y=cosine_shape_similarity(reference_y,b['sy']) if eligible and reference_y is not None else None
            minimum=min(sim_x,sim_y) if sim_x is not None and sim_y is not None else None
            scores.append(dict(image_name=image_name,block_size=n,block_id=row['block_id'],row=int(row['block_row']),col=int(row['block_col']),
                x0=b['x'],y0=b['y'],x1=b['x']+n,y1=b['y']+n,background_class=b['background_class'],
                cosine_similarity_2d=float(row['cosine_similarity_log']) if row['cosine_similarity_log'] not in ('',None) else None,
                sim_x=sim_x,sim_y=sim_y,min_similarity=minimum,rank=None))
        eligible=[r for r in scores if r['min_similarity'] is not None]
        eligible.sort(key=lambda r:(r['min_similarity'],r['row'],r['col']))
        for rank,row in enumerate(eligible,1): row['rank']=rank
        all_scores.extend(scores)
        if reference_x is not None and reference_y is not None:
            np.savez_compressed(dest/'reference_spectra.npz',frequency_x=fx,reference_x=reference_x,frequency_y=fy,reference_y=reference_y,
                grid_spectra_x=np.stack([b['sx'] for b in grids]),grid_spectra_y=np.stack([b['sy'] for b in grids]))
            for direction,freq,ref,key in [('x',fx,reference_x,'sx'),('y',fy,reference_y,'sy')]:
                fig,ax=plt.subplots(figsize=(9,4.5))
                for b in grids: ax.plot(freq,b[key],color='#3687b8',alpha=.10,lw=.7)
                ax.plot(freq,ref,color='#d62728',lw=2.2,label=f'grid mean (n={len(grids)})')
                ax.set(title=f'{image_name} / {n}px / reference {direction.upper()} FFT',xlabel='Frequency bin / block length (cycles/pixel)',ylabel='Mean FFT amplitude')
                ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(dest/f'reference_{direction}_fft.png',dpi=150); plt.close(fig)
            # Similarity maps average values where blocks overlap; background is masked.
            map_rows=[]
            by_id={r['block_id']:r for r in scores}
            for b in blocks:
                score=by_id[b['source']['block_id']]
                map_rows.append(dict(x_start=b['x'],y_start=b['y'],block_size=n,
                    sim_x=score['sim_x'],sim_y=score['sim_y'],min_similarity=score['min_similarity']))
            for metric in ('sim_x','sim_y','min_similarity'):
                heat,_=overlap_map(gray.shape,map_rows,metric)
                fig,ax=plt.subplots(figsize=(10,7)); ax.imshow(gray,cmap='gray')
                masked=np.ma.masked_invalid(heat)
                im=ax.imshow(masked,cmap='viridis',vmin=0,vmax=1,alpha=.78)
                ax.set_title(f'{image_name} / {n}px / {metric} (background masked)'); ax.axis('off')
                fig.colorbar(im,ax=ax,fraction=.035,pad=.02,label='Cosine similarity')
                fig.tight_layout(); fig.savefig(dest/f'{metric}_heatmap.png',dpi=140); plt.close(fig)
            original=gray
            for idx,candidate in enumerate(eligible[:top_k],1):
                b=next(b for b in blocks if b['source']['block_id']==candidate['block_id'])
                fig=plt.figure(figsize=(13,10)); grid=fig.add_gridspec(3,2,height_ratios=(1.0,1.0,.22))
                ax0=fig.add_subplot(grid[0,0]); ax1=fig.add_subplot(grid[0,1]); axx=fig.add_subplot(grid[1,0]); axy=fig.add_subplot(grid[1,1]); meta=fig.add_subplot(grid[2,:])
                ax0.imshow(original,cmap='gray'); ax0.add_patch(Rectangle((b['x'],b['y']),n,n,fill=False,edgecolor='red',lw=2))
                ax0.set_title('Block location (red box)'); ax0.axis('off')
                ax1.imshow(original[b['y']:b['y']+n,b['x']:b['x']+n],cmap='gray'); ax1.set_title('Block crop'); ax1.axis('off')
                axx.plot(fx,reference_x,color='#d62728',lw=2,label='reference_x')
                axx.plot(fx,b['sx'],color='#1f77b4',lw=1.3,label='block Sx')
                axx.set(title='x direction FFT',xlabel='Frequency bin / block length',ylabel='Mean amplitude'); axx.grid(alpha=.2); axx.legend()
                axy.plot(fy,reference_y,color='#d62728',lw=2,label='reference_y')
                axy.plot(fy,b['sy'],color='#ff7f0e',lw=1.3,label='block Sy')
                axy.set(title='y direction FFT',xlabel='Frequency bin / block length',ylabel='Mean amplitude'); axy.grid(alpha=.2); axy.legend()
                meta.axis('off'); meta.text(.01,.6,f"{image_name} | {n}px | block_id={candidate['block_id']} | row={candidate['row']} col={candidate['col']} | 2D={candidate['background_class']} (cos={candidate['cosine_similarity_2d']:.4f}) | sim_x={candidate['sim_x']:.4f} sim_y={candidate['sim_y']:.4f} min={candidate['min_similarity']:.4f} | rank={candidate['rank']}",fontsize=10)
                fig.tight_layout(); fig.savefig(dest/f'suspicious_block_{idx:02d}.png',dpi=150); plt.close(fig)
        else:
            # Explicitly record that the image has no grid blocks to define references.
            for metric in ('reference_x_fft.png','reference_y_fft.png','sim_x_heatmap.png','sim_y_heatmap.png','min_similarity_heatmap.png'):
                fig,ax=plt.subplots(figsize=(8,4)); ax.text(.5,.5,'No 2D grid-classified blocks; reference unavailable',ha='center',va='center'); ax.axis('off'); fig.savefig(dest/metric,dpi=120); plt.close(fig)
        with (dest/'blocks_1d_fft_scores.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            fields=list(scores[0]) if scores else []
            writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(scores)
        summaries.append(dict(image_name=image_name,block_size=n,grid_threshold=.90,background_percentile=20,
            grid_block_count=len(grids),background_block_count=len(backgrounds),uncertain_block_count=len(uncertain),
            reference_grid_block_count=len(grids),top_k=top_k,low_frequency_bins_excluded=low_frequency_bins,
            sim_x=_finite_stats([r['sim_x'] for r in scores]),sim_y=_finite_stats([r['sim_y'] for r in scores]),
            min_similarity=_finite_stats([r['min_similarity'] for r in scores])))
    if all_scores:
        with (result_root/'blocks_1d_fft_scores.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(all_scores[0])); writer.writeheader(); writer.writerows(all_scores)
    summary=dict(method=dict(reference='mean spectrum across grid-classified blocks, independently for each image and block size',
        x='rFFT magnitude of every row along x, averaged over rows',y='rFFT magnitude of every column along y, averaged over columns',
        similarity='cosine similarity after excluding configured lowest-frequency bins; no intensity weighting',
        ranking='ascending min(sim_x, sim_y) among grid and uncertain blocks; low score is a periodicity deviation, not a defect label'),
        settings=dict(block_sizes=list(SIZES),grid_threshold=.90,background_percentile=20,top_k=top_k,low_frequency_bins_excluded=low_frequency_bins),images=summaries)
    (result_root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (result_root/'README.md').open('w',encoding='utf-8') as stream:
        stream.write('# 1D FFT 格子周期類似度\n\n'
            '各画像・ブロックサイズごとに、2D分類がgridのブロックだけのx/y平均スペクトルを基準にします。xはブロック内の各行を横方向にrFFTして振幅を平均、yは各列を縦方向にrFFTして振幅を平均します。指定したDC/低周波binを除外し、コサイン類似度で形を比較します。sim_xとsim_yは独立に保持し、backgroundは1D解析対象外です。\n\n'
            'gridとuncertainをmin(sim_x, sim_y)の昇順で順位付けし、先頭TOP_Kを可視化します。低類似度は代表周期からの逸脱を示すだけで、傷と断定しません。傷ラベルは基準スペクトル作成・ランキングに使用しません。\n\n'
            '全ブロックCSVは`blocks_1d_fft_scores.csv`、画像別解析は`<block_size>/<image>/`、集計は`summary.json`です。')
    print(f'1D FFT analysis saved: {result_root}',flush=True)


def run(args):
    started_at=perf_counter()
    modes=['linear','log'] if args.spectrum_mode=='both' else [args.spectrum_mode]
    primary=modes[0]; step=2*args.fmax/(args.grid_size-1); dc_limit=(args.dc_bins+.5)*step
    mask=comparison_mask(args.grid_size,args.dc_bins)
    # Keep the result root intentionally small.  Open index.html or README.md
    # first; every intermediate image and CSV is placed under details/.
    result_root=args.output.resolve()
    if result_root in (ROOT, ROOT/'outputs'):
        raise ValueError('Choose a dedicated output directory, for example outputs/result.')
    # A rerun of identical input/settings updates the same managed result
    # folder.  Only files created by this program are removed.
    if result_root.exists():
        for name in ('details','index.html','README.md'):
            target=result_root/name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.is_file():
                target.unlink()
    result_root.mkdir(parents=True, exist_ok=True)
    out=result_root/'details'
    out.mkdir()
    print(f'OUTPUT {result_root}',flush=True)
    files=sorted(p for p in args.input.iterdir() if p.suffix.lower() in EXTENSIONS and not p.name.startswith('._') and (not args.images or p.stem in args.images))
    if not files: raise ValueError('No input images')
    allrows=[]; annotations=[]; perimage=[]; statuses=[]; errors=[]
    for path in files:
        try:
            gray=read_gray(path); h,w=gray.shape; labels=load_labels(path.with_suffix('.json'),gray.shape)
            raw,amp=fft_amplitude(gray,not args.keep_mean); common=common_spectrum(amp,args.grid_size,args.fmax)
            global_vectors={mode:vector(common,mask,mode) for mode in modes}
            ge=raw_energy(raw,dc_limit)
            references=reference_peaks(raw,args)
            if not references: raise ValueError('No reference FFT peak pairs passed the current peak settings')
            for n in SIZES:
                if min(h,w)<n:
                    print(f'{path.name} {n}: skipped (image smaller than block)',flush=True)
                    continue
                dest=out/str(n)/path.stem; dest.mkdir(parents=True)
                if args.details:
                    save_global(gray,raw,common,mask,labels,dest,args)
                    np.savez_compressed(dest/'reference_spectrum.npz',amplitude=common,mask=mask,
                        frequency=np.linspace(-args.fmax,args.fmax,args.grid_size))
                positions=[(row,col,x,y) for row,y in enumerate(range(0,h-n+1,n//2))
                           for col,x in enumerate(range(0,w-n+1,n//2))]
                fft_started=perf_counter()

                def analyze_one_block(position):
                    """Pure per-block work: safe to execute concurrently."""
                    row,col,x,y=position
                    block=gray[y:y+n,x:x+n]
                    br,ba=fft_amplitude(block,not args.keep_mean)
                    bs=common_spectrum(ba,args.grid_size,args.fmax)
                    peak_matches,_=block_peak_matches(br,references,args)
                    matched_count=sum(match['matched'] for match in peak_matches)
                    pair_ids=sorted({match['pair_id'] for match in peak_matches})
                    matched_pairs=sum(all(match['matched'] for match in peak_matches if match['pair_id']==pair_id)
                                      for pair_id in pair_ids)
                    hits=[label for label in labels if intersect((x,y,x+n,y+n),label['box'])>0]
                    scores={mode:cosine(global_vectors[mode],vector(bs,mask,mode)) for mode in modes}
                    return dict(image_name=path.name,block_size=n,block_id=f'r{row:02d}_c{col:02d}',block_row=row,block_col=col,
                        x=x,y=y,width=n,height=n,x_start=x,y_start=y,stride=n//2,block_mean=float(block.mean()),
                        cosine_similarity=scores[primary],label=';'.join(sorted(set(label['label'] for label in hits))) if hits else 'unknown',
                        annotation_ids=';'.join(label['id'] for label in hits),global_fft_energy=ge,block_fft_energy=raw_energy(br,dc_limit),
                        matched_peak_count=matched_count,peak_match_score=float(matched_count/len(references)),
                        matched_pair_count=matched_pairs,pair_match_score=float(matched_pairs/len(pair_ids)) if pair_ids else None,
                        reference_peak_count=len(references),reference_pair_count=len(pair_ids),
                        **{'cosine_similarity_'+mode:scores[mode] for mode in modes})

                if args.workers == 1:
                    rows=[analyze_one_block(position) for position in positions]
                else:
                    # executor.map preserves position order, making CSVs and figures reproducible.
                    with ThreadPoolExecutor(max_workers=args.workers) as executor:
                        rows=list(executor.map(analyze_one_block,positions))
                print(f'{path.name} {n}: FFT for {len(rows)} blocks took {perf_counter()-fft_started:.1f}s',flush=True)
                for mode in modes:
                    key='cosine_similarity_'+mode
                    valid=[r for r in rows if r[key] is not None]; values=np.array([r[key] for r in valid])
                    ranks=rankdata(values,method='average'); med=float(np.median(values)) if len(values) else None
                    for r in rows:
                        r['similarity_rank_'+mode]=None; r['similarity_percentile_'+mode]=None; r['similarity_minus_image_median_'+mode]=None
                    for r,rank in zip(valid,ranks):
                        r['similarity_rank_'+mode]=float(rank)
                        r['similarity_percentile_'+mode]=float(100*(rank-.5)/len(values))
                        r['similarity_minus_image_median_'+mode]=r[key]-med
                    for r in rows:
                        if mode==primary:
                            r['similarity_rank']=r['similarity_rank_'+mode]; r['similarity_percentile']=r['similarity_percentile_'+mode]
                            r['similarity_minus_image_median']=r['similarity_minus_image_median_'+mode]
                    pos=np.array([r[key] for r in valid if r['label']!='unknown']); unk=np.array([r[key] for r in valid if r['label']=='unknown'])
                    perimage.append(dict(image_name=path.name,block_size=n,mode=mode,blocks=len(rows),valid_blocks=len(valid),
                        labeled_blocks=len(pos),unknown_blocks=len(unk),image_median=med,
                        labeled_median=float(np.median(pos)) if len(pos) else None,unknown_median=float(np.median(unk)) if len(unk) else None))
                    for l in labels:
                        hits=[r for r in rows if l['id'] in r['annotation_ids'].split(';')]
                        vals=np.array([r[key] for r in hits if r[key] is not None])
                        refs=nearby(rows,l,labels,n); refvals=np.array([r[key] for r in refs if r[key] is not None])
                        lm=float(np.median(vals)) if len(vals) else None
                        b=l['box']; cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2
                        center_score=None; center_coverage=None
                        if min(gray.shape)>=n:
                            xx=max(0,min(w-n,int(round(cx-n/2)))); yy=max(0,min(h-n,int(round(cy-n/2))))
                            _,ca=fft_amplitude(gray[yy:yy+n,xx:xx+n],not args.keep_mean)
                            center_score=cosine(global_vectors[mode],vector(common_spectrum(ca,args.grid_size,args.fmax),mask,mode))
                            center_coverage=intersect((xx,yy,xx+n,yy+n),b)/area(b)
                        refmed=float(np.median(refvals)) if len(refvals) else None
                        annotations.append(dict(image_name=path.name,annotation_id=l['id'],label=l['label'],block_size=n,mode=mode,
                            overlap_blocks=len(hits),valid_overlap_blocks=len(vals),cosine_similarity=lm,
                            image_percentile=percentile(values,lm),similarity_minus_image_median=lm-med if lm is not None and med is not None else None,
                            overlap_min=float(vals.min()) if len(vals) else None,overlap_max=float(vals.max()) if len(vals) else None,
                            nearby_unknown_count=len(refvals),nearby_unknown_median=refmed,
                            similarity_minus_nearby_unknown=lm-refmed if lm is not None and refmed is not None else None,
                            centered_similarity=center_score,centered_image_percentile=percentile(values,center_score),centered_bbox_coverage=center_coverage,
                            bbox_x0=b[0],bbox_y0=b[1],bbox_x1=b[2],bbox_y1=b[3]))
                    if args.details:
                        maps(gray,rows,labels,dest,n,mode,primary)
                write_csv(dest/'blocks.csv',rows); allrows.extend(rows)
                write_csv(dest/'blocks_peak_scores.csv',rows)
                if args.details:
                    peak_score_visuals(gray,rows,raw,references,labels,dest,n,args)
                cosine_values=np.array([r['cosine_similarity'] for r in rows if r['cosine_similarity'] is not None])
                score_values=np.array([r['peak_match_score'] for r in rows if r['cosine_similarity'] is not None])
                correlation=float(np.corrcoef(cosine_values,score_values)[0,1]) if len(score_values)>1 and np.std(score_values)>0 and np.std(cosine_values)>0 else None
                peak_summary=dict(image_name=path.name,block_size=n,reference_peak_count=len(references),reference_pair_count=len(references)//2,
                    reference_peaks=references,tolerance_fft_bins=args.peak_tolerance,block_count=len(rows),
                    peak_match_score=dict(mean=float(np.mean(score_values)),median=float(np.median(score_values)),min=float(np.min(score_values)),max=float(np.max(score_values))),
                    cosine_similarity_correlation=correlation,settings=dict(peak_radius=args.peak_radius,peak_dc_radius=args.peak_dc_radius,peak_percentile=args.peak_percentile,peak_local_max_size=args.peak_local_max_size,peak_min_distance=args.peak_min_distance,mean_subtraction=not args.keep_mean,hann_window=True))
                (dest/'summary.json').write_text(json.dumps(peak_summary,ensure_ascii=False,indent=2),encoding='utf-8')
                coverage=overlap_map(gray.shape,rows,'block_mean')[1]
                statuses.append(dict(image_name=path.name,block_size=n,blocks=len(rows),unanalysed_pixels=int(np.sum(coverage==0)),label_count=len(labels)))
                print(f'{path.name} {n}: {len(rows)} blocks',flush=True)
        except Exception as e:
            plt.close('all'); errors.append(dict(image_name=path.name,error=str(e))); print(f'ERROR {path.name}: {e}',flush=True)
    for name,data in [('all_blocks',allrows),('blocks_peak_scores',allrows),('annotation_percentiles',annotations),('image_summary',perimage),('image_status',statuses)]: write_csv(out/(name+'.csv'),data)
    if args.details:
        for n in SIZES:
            for mode in modes: distribution(allrows,n,mode,out,primary)
    settings=dict(input=str(args.input),sizes=SIZES,strides=[n//2 for n in SIZES],spectrum_modes=modes,primary_mode=primary,
        mean_subtraction=not args.keep_mean,grid_size=args.grid_size,frequency_max=args.fmax,frequency_step=step,
        dc_bins=args.dc_bins,dc_half_width_cycles_per_pixel=dc_limit,workers=args.workers,details=args.details,
        blocks=len(allrows),images=len(files),errors=errors,
        reference_peak_count=args.reference_peak_count,peak_tolerance_fft_bins=args.peak_tolerance,
        peak_radius=args.peak_radius,peak_dc_radius=args.peak_dc_radius,peak_percentile=args.peak_percentile,
        peak_local_max_size=args.peak_local_max_size,peak_min_distance=args.peak_min_distance)
    (out/'settings.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    aggregate=[]
    for image_name in sorted({r['image_name'] for r in allrows}):
        for n in SIZES:
            subset=[r for r in allrows if r['image_name']==image_name and r['block_size']==n]
            if not subset: continue
            values=np.array([r['peak_match_score'] for r in subset],dtype=float)
            cos=np.array([r['cosine_similarity'] for r in subset if r['cosine_similarity'] is not None],dtype=float)
            paired=np.array([r['peak_match_score'] for r in subset if r['cosine_similarity'] is not None],dtype=float)
            aggregate.append(dict(image_name=image_name,block_size=n,reference_peak_count=subset[0]['reference_peak_count'],
                reference_pair_count=subset[0]['reference_pair_count'],tolerance_fft_bins=args.peak_tolerance,
                block_count=len(subset),peak_match_score=dict(mean=float(values.mean()),median=float(np.median(values)),
                min=float(values.min()),max=float(values.max())),
                cosine_similarity_correlation=float(np.corrcoef(cos,paired)[0,1]) if len(paired)>1 and np.std(cos)>0 and np.std(paired)>0 else None))
    (out/'summary.json').write_text(json.dumps(dict(settings=settings,images=aggregate),ensure_ascii=False,indent=2),encoding='utf-8')
    if args.details:
        report(out,files,annotations,perimage,settings)
    classify_similarity_blocks(out,args.input,files)
    if args.details:
        analyze_1d_fft_blocks(out,args.input,files,top_k=args.top_k,low_frequency_bins=args.one_d_dc_bins)
    write_result_landing(result_root, out, settings, args.details)
    print(f'Total processing time: {perf_counter()-started_at:.1f}s',flush=True)
    if errors: raise SystemExit(1)


def write_result_landing(result_root, details, settings, full_details):
    """Make the output root a short, human-first entry point."""
    mode = ('詳細解析あり（1D FFT・デバッグ画像を含む）' if full_details
            else '通常解析（高速・必要最小限の出力）')
    extra = ('''\n- `details/index.html` — 2D FFT・ピーク照合・ラベル比較の詳細
- `details/results_1d_fft/README.md` — 1D FFT による周期性の再確認
''') if full_details else ''
    readme = f'''# 外観検査 結果一覧

最初に **`index.html`** を開いてください。画像ごとの色分け結果を確認できます。

- `details/similarity_3class_256_contact_sheet.png` — 全画像の結果一覧（256 px）
- `details/256/<画像名>/similarity_3class.png` — 各画像の結果
{extra}
実行モード: {mode}

`details` には途中計算のCSVなどもあります。通常は開く必要はありません。
注意: 類似度が低いことは、そのまま不良判定を意味しません。未ラベル領域は正常品ではなく「未確認」です。

解析対象: {settings['images']} 枚、ブロック数: {settings['blocks']}
'''
    (result_root/'README.md').write_text(readme, encoding='utf-8')
    extra_link = ('<a href="details/index.html">詳細解析を見る</a>'
                  if full_details else '')
    page = f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>外観検査 結果一覧</title>
<style>body{{font-family:system-ui,sans-serif;max-width:760px;margin:48px auto;padding:0 20px;line-height:1.7}}a{{display:block;margin:14px 0;padding:15px 18px;background:#f1f5f9;border-radius:9px;color:#0f3d68;text-decoration:none;font-weight:650}}small{{color:#556}}</style>
<h1>外観検査 結果一覧</h1><p>まず下の「全画像の結果を見る」を開いてください。</p>
<a href="details/similarity_3class_256_contact_sheet.png">全画像の結果を見る（256 px）</a>
<a href="README.md">結果の読み方を見る</a>{extra_link}
<small>{mode}。低い類似度だけで不良判定はしません。</small></html>'''
    (result_root/'index.html').write_text(page, encoding='utf-8')


def report(out,files,annotations,perimage,settings):
    modes=settings['spectrum_modes']; primary=settings['primary_mode']
    lines=['# 画像全体2D FFTと局所2D FFTのコサイン類似度','',
        f'画像{settings["images"]}枚、走査ブロック{settings["blocks"]}個。主表示={primary}、比較モード={modes}。',
        '', '## 方法',
        '元画像のグレースケール値をリサイズせず使用。既定では平均除去後に縦横Hann窓を掛け、fft2→fftshift→abs。全体・局所で同じ前処理。',
        '', '## 代表FFTピーク一致スコア',
        f'全体FFTの中心±{settings["peak_radius"]} binから、DC半径{settings["peak_dc_radius"]} binを除外して局所最大値を共役対として選び、最大{settings["reference_peak_count"]}点（10対）を基準ピークにする。各点の全体FFT相対座標(dx, dy)と周波数(fx, fy)を`summary.json`へ保存する。',
        f'各ブロックでも同じ局所ピーク検出を行い、基準周波数をそのブロックFFTのbin位置へ換算して、距離{settings["peak_tolerance_fft_bins"]} FFT bins以内の局所ピークを一致とする。`peak_match_score = matched_peak_count / reference_peak_count`を主指標、共役する両点が一致した組の割合を`pair_match_score`とする。',
        'これは「格子特有と仮定した代表ピークが何個再現されたか」の単純な記述指標であり、ピーク強度は重み付けしない。全体FFTの選択ピークが本当に格子由来か、背景でも高得点にならないかは、`reference_fft_peaks.png`、ヒートマップ、デバッグFFT図で確認する。',
        '追加の3クラス表示ではlogコサイン類似度を使い、0.90以上を格子候補、0.90未満かつ画像内p20以下を背景候補、それ以外を判定保留とする。条件が重なるときは格子候補を優先する。',
        '1D FFT後段解析では、各画像・サイズごとにgridブロックのみの平均スペクトルをx/y別に作る。xは各行の横方向rFFT振幅平均、yは各列の縦方向rFFT振幅平均。DCを含む最低1 binを除き、スペクトル形状のコサイン類似度sim_x/sim_yを別々に計算する。backgroundを除きmin(sim_x,sim_y)昇順にTOP_Kを可視化する。低類似度は周期構造の逸脱候補であり、傷とは断定しない。傷ラベルは使わない。',
        '振幅はHann窓総和で割って同じ輝度単位にする。linearではこの定数はcosineに影響しない。logではlog1pの前のスケールを統一するために重要。logのcosineは単純な乗算コントラスト変化にも完全には不変ではない。',
        f'共通座標は各軸±{settings["frequency_max"]} cycles/pixel、{settings["grid_size"]}点、間隔{settings["frequency_step"]:.8f}。',
        '周波数binを中心±半bin幅のセルとみなし、共通セルとの重なり面積に応じて振幅を平均する（縦横独立の疎行列）。大きいFFTの狭いピークを点サンプリングで飛ばすことを避ける。局所FFT側も同じ方法で変換し、同じ周波数座標で比較する。細かい共通座標にしても局所FFTの真の周波数分解能は増えない。',
        f'DCは共通グリッド中心±{settings["dc_bins"]}点の正方形を除外。セル境界までの半幅は{settings["dc_half_width_cycles_per_pixel"]:.8f} cycles/pixel。各元FFTで独立に±5 binsを除外する方法ではなく、同じ物理周波数帯を両者から除く。FFT数値は変更せず比較マスクを用いる。',
        '有効セルの振幅（またはlog1p振幅）をベクトル化し、各L2ノルムで割って内積を取る。零ノルムはNAとし、低類似度0に置き換えない。',
        '512/stride256、256/stride128、128/stride64。重複画素のヒートマップはブロック値の平均。peak_score_text.pngの元画像上には各ブロックの一致ピーク数/基準ピーク数（例：18/20）を表示し、CSVには各ブロックの元の値を保存する。',
        '', '## ラベルと順位',
        '矩形と面積が少しでも交差するブロックをユーザーのdent/scrapeラベルに対応付ける。未ラベルはすべてunknown。矩形の全画素が傷とは限らない。',
        '順位は画像・サイズ・モード別、有効ブロック内の昇順平均順位。パーセンタイル=100×(小さい個数+同値個数/2)/有効数。低いほど画像内で低類似度。',
        '傷22か所それぞれの主値は、矩形に重なる走査ブロックの類似度の中央値。その値が同じ画像・サイズの全ブロック分布の何パーセンタイルかを計算する。最小値の選択による過大評価を避け、min/maxと傷中心切り出しの別スコアも保存した。',
        '周辺比較は同じ画像の既知ラベルすべてに重ならない、傷中心から3ブロック幅以内の近い順最大12ブロック。unknownであり正常保証はない。',
        '画像全体FFTには背景・テープ・外枠・傷自身も含まれ、正常格子の基準とは限らない。傷中心の追加切り出しは走査位置と異なる場合があり、大きい傷では一部のみを含む（被覆率を保存）。',
        '', '## 集計：画像内・周辺との比較',
        '|サイズ|モード|傷中央値が画像中央値未満|傷中央値が周辺unknown中央値未満|',
        '|---:|---|---:|---:|']
    for n in SIZES:
        for mode in modes:
            a=[r for r in annotations if r['block_size']==n and r['mode']==mode]
            v=[r for r in a if r['similarity_minus_image_median'] is not None]; u=[r for r in a if r['similarity_minus_nearby_unknown'] is not None]
            lines.append(f'|{n}|{mode}|{sum(r["similarity_minus_image_median"]<0 for r in v)}/{len(v)}|{sum(r["similarity_minus_nearby_unknown"]<0 for r in u)}/{len(u)}|')
    lines+=['','## 画像による分布の違い',
        '|画像|サイズ|モード|画像中央値|ラベル交差ブロック中央値|unknown中央値|',
        '|---|---:|---|---:|---:|---:|']
    def fmt(v): return 'NA' if v is None else f'{v:.4f}'
    for r in perimage:
        lines.append(f'|{r["image_name"]}|{r["block_size"]}|{r["mode"]}|{fmt(r["image_median"])}|{fmt(r["labeled_median"])}|{fmt(r["unknown_median"])}|')
    lines+=['','## 傷ごとの画像内パーセンタイル（全モード）','|傷ID|種類|サイズ|モード|類似度（重なるブロック中央値）|画像内percentile|周辺との差|','|---|---|---:|---|---:|---:|---:|']
    for r in annotations:
        lines.append(f'|{r["annotation_id"]}|{r["label"]}|{r["block_size"]}|{r["mode"]}|{fmt(r["cosine_similarity"])}|{fmt(r["image_percentile"])}%|{fmt(r["similarity_minus_nearby_unknown"])}|')
    lines+=['','## 解釈と次に試すこと',
        '各ブロックサイズnの局所FFT周波数分解能は1/n cycles/pixel（512: 1/512、256: 1/256、128: 1/128）。窓によるピーク幅も異なるため、同じ座標に統一してもサイズによる類似度差は残る。傷の局在性と分解能のトレードオフを含む結果である。',
        '傷で必ず低下するかは上表と画像ごとのCSVを参照。これらは記述統計であり、検出成功数・精度ではない。unknownを正常として誤検出率やROCを算出していない。重複ブロックは独立標本ではない。',
        'この方法を傷判定として採用するには、正常格子の確認、ラベル別の局所比較、別画像での検証が必要。今回自動閾値は設定しない。',
        '次の実験候補：正常と確認した格子ROIを基準にする／共通周波数範囲とDC幅を変えて順位の安定性を確認／全体FFTを局所窓の分解能に合わせて平滑化する。現在の全体FFTは全画面Hannで中央が重く、端の局所格子と異なり得る。',
        '', '## 出力・再実行',
        '`index.html`は画像とサイズ別の元画像・FFT・マップ・ラベル重畳を横並び表示。`blocks_peak_scores.csv`はピーク一致列を含む全ブロック、ルートの`summary.json`は全画像・サイズの集計。`peak_score_text.png`では元画像上に一致数/基準数（例：18/20）を表示し、`peak_score_heatmap.png`、`reference_fft_peaks.png`、`debug_block_fft_*.png`、`cosine_vs_peak_score.png`で照合を確認できる。',
        '`interactive_similarity_threshold.html`ではスライダーで類似度しきい値を変え、全画像の3クラス表示を同時に確認できる。`similarity_3class_blocks.csv`と`similarity_3class_summary.csv`は初期90%設定の判定結果、`similarity_3class_256_contact_sheet.png`と`similarity_3class_128_contact_sheet.png`は元画像への色重ね一覧。緑=格子候補、赤=背景候補、灰=保留。',
        '`global_fft_energy`と`block_fft_energy`は同じDC周波数領域を除いた生の|FFT|²の和。画像面積で規模が変わる参考値であり、大小をそのまま比較する指標ではない。',
        '```powershell','python fft2_cosine_similarity.py','python fft2_cosine_similarity.py --spectrum-mode log --dc-bins 3','```',
        '既存出力は上書きせず日時付きフォルダに保存。`--keep-mean`で平均除去を無効化、`--fmax`と`--grid-size`で共通座標を変更可能。DCのbinsは共通座標の単位である。']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    cards=['<meta charset="utf-8"><style>body{font-family:sans-serif}.row{display:flex;gap:8px}.row a{width:25%}img{width:100%}</style><h1>Global / local 2D FFT cosine similarity</h1><p>Low similarity does not mean defect. Unlabeled = unknown.</p><p><a href="README.md">Report + all 22 annotation percentiles</a> / <a href="annotation_percentiles.csv">Annotation CSV</a> / <a href="all_blocks.csv">Block CSV</a> / <a href="similarity_map_log_256_gallery.html">All 256px log similarity maps</a> / <a href="results_1d_fft/README.md">1D FFT grid-period analysis</a> / <a href="results_1d_fft/summary.json">1D FFT summary</a></p>']
    for n in SIZES:
        for mode in modes:
            suffix='' if mode==primary else '_'+mode
            cards.append(f'<p><a href="scratch_vs_unknown_{n}{suffix}.png">{n}px {mode}: boxplot / histogram / strip plot</a></p>')
            for p in files:
                base=f'{n}/{p.stem}'
                if not (out/base).exists(): continue
                cards.append(f'<h2>{p.name} / {n}px / {mode}</h2><div class="row">'+''.join(f'<a href="{base}/{name}.png">{title}<img loading="lazy" src="{base}/{name}.png"></a>' for name,title in [('original_labels','Original + labels'),('global_fft','Global FFT'),('similarity_map'+suffix,'Similarity'),('overlay'+suffix,'Block scores + labels'),('peak_score_heatmap','Peak score heatmap'),('peak_score_text','Peak score text'),('reference_fft_peaks','Reference peaks'),('cosine_vs_peak_score','Cosine vs peak score')])+'</div>')
    (out/'index.html').write_text(''.join(cards),encoding='utf-8')
    similarity_map_gallery(out,files,block_size=256,mode='log',input_dir=settings['input'])
    print(json.dumps(settings),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'data20260903/data20260903')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/result',
                   help='Fixed result folder. Each run refreshes this program-managed folder.')
    p.add_argument('--spectrum-mode',choices=['linear','log','both'],default='both')
    p.add_argument('--grid-size',type=int,default=129)
    p.add_argument('--fmax',type=float,default=.125)
    p.add_argument('--dc-bins',type=int,default=5)
    p.add_argument('--keep-mean',action='store_true')
    p.add_argument('--reference-peak-count',type=int,default=20,help='Maximum individual reference peaks; selected as conjugate pairs.')
    p.add_argument('--peak-radius',type=int,default=150,help='Center-region radius in full-image FFT bins.')
    p.add_argument('--peak-dc-radius',type=float,default=10,help='DC exclusion radius in native FFT bins.')
    p.add_argument('--peak-percentile',type=float,default=99.,help='Local-peak amplitude percentile threshold.')
    p.add_argument('--peak-local-max-size',type=int,default=7,help='Odd local-max neighbourhood side length.')
    p.add_argument('--peak-min-distance',type=float,default=7.,help='Minimum separation between selected reference peaks.')
    p.add_argument('--peak-tolerance',type=float,default=3.,help='Allowed distance in block FFT bins for a peak match.')
    p.add_argument('--debug-blocks',type=int,default=4,help='Number of best/worst score block FFT debug images per image/size.')
    p.add_argument('--top-k',type=int,default=10,help='Number of lowest 1D FFT similarity blocks to visualize per image/size.')
    p.add_argument('--one-d-dc-bins',type=int,default=1,help='Number of lowest 1D FFT bins to exclude before cosine comparison (includes DC).')
    p.add_argument('--workers',type=int,default=min(8,max(1,os.cpu_count() or 1)),
                   help='Parallel FFT worker threads (default: up to 8; use 1 for sequential processing).')
    p.add_argument('--details',action='store_true',
                   help='Also generate slow diagnostic plots, debug FFT images, and 1D FFT follow-up results.')
    p.add_argument('--images',nargs='*')
    args=p.parse_args()
    if args.grid_size<5 or args.grid_size%2!=1 or not 0<=args.dc_bins<args.grid_size//2 or not 0<args.fmax<.49:
        p.error('Require odd grid-size >=5, 0 <= dc-bins < grid-size//2, 0 < fmax < .49')
    if args.reference_peak_count<2 or args.reference_peak_count%2 or args.peak_radius<1 or not 0<=args.peak_dc_radius<args.peak_radius or not 0<=args.peak_percentile<=100 or args.peak_local_max_size<1 or args.peak_local_max_size%2!=1 or args.peak_min_distance<0 or args.peak_tolerance<0 or args.debug_blocks<1:
        p.error('Require an even reference-peak-count >=2, valid peak radii/percentile, odd local-max size, and nonnegative peak distances.')
    if args.top_k<1 or args.one_d_dc_bins<1 or args.one_d_dc_bins>=65:
        p.error('Require top-k >=1 and one-d-dc-bins between 1 and 64 for the supported block sizes.')
    if args.workers < 1:
        p.error('Require workers >= 1.')
    run(args)


if __name__=='__main__': main()
