"""Compare full-image 1D projection FFT at 256/128 pixels, 50% overlap.
Run: python compare_multiscale_1d_fft.py
Reuses block_1d_fft_experiment; dependencies numpy, opencv-python, matplotlib.
"""
import argparse
import html
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from block_1d_fft_experiment import ROOT, EXTENSIONS, REPRESENTATIVES, read_gray, analyze, comparison, write_csv

SIZES = (256, 128)
METRICS = ('peak_amplitude', 'peak_frequency', 'band_energy_ratio', 'band_peak_amplitude')


def overlap_map(shape, rows, key):
    """Rectangle difference arrays; average only finite block values."""
    total = np.zeros((shape[0]+1, shape[1]+1))
    count = np.zeros_like(total)
    for r in rows:
        value = r[key]
        if value is None or not np.isfinite(value):
            continue
        x,y,n = r['x_start'],r['y_start'],r['block_size']
        for yy,xx,sign in ((y,x,1),(y+n,x,-1),(y,x+n,-1),(y+n,x+n,1)):
            total[yy,xx] += sign*value; count[yy,xx] += sign
    total = total.cumsum(0).cumsum(1)[:-1,:-1]
    count = count.cumsum(0).cumsum(1)[:-1,:-1]
    return np.divide(total,count,out=np.full(shape,np.nan),where=count>0), count


def atlas(records, out, band):
    """Self-contained, paginated SVG graphs for every block; both windows saved."""
    ymax = max(float(s[d+'_'+w+'_amplitude'].max()) for _,s in records for d in ('x','y') for w in ('plain','hann'))
    ymax = max(ymax,1e-12)
    links=[]
    for start in range(0,len(records),96):
        cards=[]
        for label,s in records[start:start+96]:
            traces=[]
            for d,color in [('x','#1776bb'),('y','#df7200')]:
                for w,dash in [('plain','3 2'),('hann','none')]:
                    points=' '.join(f'{35+f/.5*300:.2f},{140-a/ymax*115:.2f}' for f,a in zip(s['frequency'],s[f'{d}_{w}_amplitude']))
                    traces.append(f'<polyline fill="none" stroke="{color}" stroke-width="1" stroke-dasharray="{dash}" points="{points}"/>')
            cards.append(f'<div><b>{html.escape(label)}</b><svg viewBox="0 0 350 170"><rect x="{35+band[0]*600}" y="25" width="{(band[1]-band[0])*600}" height="115" fill="#e7eadb"/><path d="M35 25V140H335" fill="none" stroke="black"/><text x="0" y="28" font-size="10">{ymax:.2g}</text><text x="20" y="151" font-size="10">0</text><text x="315" y="151" font-size="10">0.5</text><text x="85" y="166" font-size="10">Frequency [cycles/pixel]</text>{"".join(traces)}</svg></div>')
        name=f'spectra_atlas_{start//96+1:02d}.html'; links.append(name)
        (out/name).write_text('<meta charset="utf-8"><style>body{font-family:sans-serif}.grid{display:grid;grid-template-columns:repeat(4,1fr)}svg{width:100%}b{font-size:12px}</style><h2>All block FFT spectra</h2><p>X blue / Y orange; Hann solid / plain dashed. Amplitude in gray levels; DC excluded. Shared amplitude scale across all pages for this image and size. Shaded exploratory band '+str(band)+' (not a confirmed lattice band).</p><div class="grid">'+''.join(cards)+'</div>',encoding='utf-8')
    return links


def size_run(gray, filename, n, out, band):
    out.mkdir(parents=True)
    rows=[]; records=[]; stride=n//2
    for row,y in enumerate(range(0,gray.shape[0]-n+1,stride)):
        for col,x in enumerate(range(0,gray.shape[1]-n+1,stride)):
            v,s=analyze(gray[y:y+n,x:x+n],band)
            rows.append(dict(filename=filename,block_row=row,block_col=col,x_start=x,y_start=y,
                             block_size=n,stride=stride,band_low=band[0],band_high=band[1],**v))
            records.append((f'r{row} c{col} | x={x}, y={y}',s))
    write_csv(out/'features.csv',rows)
    if not rows:
        (out/'status.json').write_text(json.dumps({'status':'image_smaller_than_block','blocks':0}),encoding='utf-8')
        return rows, []
    arrays={k:np.stack([s[k] for _,s in records]) for k in records[0][1] if k not in ('block','frequency')}
    np.savez_compressed(out/'profiles_spectra.npz',frequency=records[0][1]['frequency'],
                        positions=np.array([[r[k] for k in ('block_row','block_col','x_start','y_start')] for r in rows]),**arrays)
    pages=atlas(records,out,band)
    coverage=overlap_map(gray.shape,rows,'block_mean')[1]
    fig,ax=plt.subplots(figsize=(12,9)); ax.imshow(gray,cmap='gray')
    ax.imshow(np.ma.masked_where(coverage>0,np.ones(gray.shape)),cmap='Reds',vmin=0,vmax=1,alpha=.6)
    for r in rows:
        ax.add_patch(Rectangle((r['x_start']-.5,r['y_start']-.5),n,n,fill=False,edgecolor='cyan',lw=.25))
    ax.set_title(f'{filename}: size={n}, stride={stride}; red=unanalysed; coordinates in pixels')
    fig.tight_layout(); fig.savefig(out/'block_locations.png',dpi=110); plt.close(fig)
    (out/'settings.json').write_text(json.dumps(dict(block_size=n,stride=stride,blocks=len(rows),
        unanalysed_pixels=int((coverage==0).sum()),band=band,atlas_pages=pages),indent=2),encoding='utf-8')
    return rows,pages


def map_comparisons(gray, results, out, band):
    limits={}; summaries=[]
    for metric in METRICS:
        fig,axes=plt.subplots(2,2,figsize=(13,10),sharex=True,sharey=True)
        for ri,d in enumerate(('x','y')):
            key=f'{d}_hann_{metric}'
            maps=[overlap_map(gray.shape,results[n],key)[0] for n in SIZES]
            finite=np.concatenate([m[np.isfinite(m)] for m in maps])
            vmax=1. if metric=='band_energy_ratio' else max(float(finite.max()) if finite.size else 0,1e-12)
            limits[key]=[0,vmax]
            for ci,(n,m) in enumerate(zip(SIZES,maps)):
                ax=axes[ri,ci]; ax.imshow(gray,cmap='gray')
                # Red tint means no block coverage; undefined features are grey.
                cov=overlap_map(gray.shape,results[n],'block_mean')[1]
                ax.imshow(np.ma.masked_where(cov>0,np.ones(gray.shape)),cmap='Reds',vmin=0,vmax=1,alpha=.6)
                im=ax.imshow(np.ma.masked_invalid(m),cmap='viridis',vmin=0,vmax=vmax,alpha=.7)
                ax.set_title(f'{d.upper()} / {n}px stride={n//2}')
                values=np.array([r[key] for r in results[n] if r[key] is not None],dtype=float)
                summaries.append(dict(size=n,direction=d,metric=metric,minimum=float(values.min()) if values.size else None,
                    median=float(np.median(values)) if values.size else None,maximum=float(values.max()) if values.size else None))
            cax=fig.add_axes([.88,.56 if ri==0 else .14,.015,.29])
            fig.colorbar(im,cax=cax,label=metric)
        fig.suptitle(f'{out.name}: {metric} (Hann)\nExploratory band {band[0]}-{band[1]} cycles/pixel; NOT confirmed lattice band\nOverlap averaged per pixel; red=unanalysed; grey=undefined')
        fig.subplots_adjust(top=.88,wspace=.08,hspace=.14,right=.83)
        fig.savefig(out/f'compare_{metric}.png',dpi=110); plt.close(fig)
    (out/'color_limits.json').write_text(json.dumps(limits,indent=2),encoding='utf-8')
    write_csv(out/'distribution_summary.csv',summaries)


def representative_run(gray, stem, out, band):
    rows=[]; combined=[]
    for name,row,col,label in REPRESENTATIVES:
        if name!=stem: continue
        cx,cy=col*128+64,row*128+64
        samples=[]; fig,ax=plt.subplots(figsize=(11,8)); ax.imshow(gray,cmap='gray')
        for n,color in [(256,'cyan'),(128,'red')]:
            x,y=cx-n//2,cy-n//2
            if x<0 or y<0 or x+n>gray.shape[1] or y+n>gray.shape[0]: continue
            v,s=analyze(gray[y:y+n,x:x+n],band)
            samples.append((f'{stem}: {label}\n{n}px center=({cx},{cy})',s))
            combined.append(samples[-1])
            rows.append(dict(filename=stem,representative=f'r{row}_c{col}',label=label,center_x=cx,center_y=cy,
                             x_start=x,y_start=y,block_size=n,**v))
            ax.add_patch(Rectangle((x-.5,y-.5),n,n,fill=False,edgecolor=color,lw=2,label=str(n)))
        ax.legend(); ax.set_title(f'{label}: same center ({cx},{cy}); candidates, not ground truth')
        fig.tight_layout(); fig.savefig(out/f'representative_r{row}_c{col}_location.png',dpi=110); plt.close(fig)
        comparison(samples,out/f'representative_r{row}_c{col}_spectra.png')
        comparison(samples,out/f'representative_r{row}_c{col}_zoom.png',fmax=.12)
    write_csv(out/'representative_features.csv',rows)
    if combined:
        comparison(combined,out/'representative_all_shared_scale.png',fmax=.12)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'data20260903/data20260903')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/multiscale_1d_fft')
    p.add_argument('--band',type=float,nargs=2,default=[.01,.04])
    p.add_argument('--images',nargs='*')
    args=p.parse_args()
    if not 0<args.band[0]<args.band[1]<=.5: p.error('Require 0 < low < high <= .5')
    files=sorted(f for f in args.input.iterdir() if f.suffix.lower() in EXTENSIONS and not f.name.startswith('._') and (not args.images or f.stem in args.images))
    if not files: p.error('No input images')
    # Never overwrite old experiments, including repeated runs of this script.
    out=args.output
    if out.exists(): out=out.with_name(out.name+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True)
    print(f'OUTPUT {out}',flush=True)
    report=[]; errors=[]; allrows={n:[] for n in SIZES}; reps=[]
    for path in files:
        try:
            gray=read_gray(path); dest=out/path.stem; dest.mkdir()
            results={}; page_links={}
            for n in SIZES:
                results[n],page_links[n]=size_run(gray,path.name,n,dest/f'size_{n}',args.band)
                allrows[n].extend(results[n])
            map_comparisons(gray,results,dest,args.band)
            reps.extend(representative_run(gray,path.stem,dest,args.band))
            report.append(dict(filename=path.name,shape=list(gray.shape),counts={str(n):len(results[n]) for n in SIZES},atlas=page_links))
            (dest/'index.html').write_text('<meta charset="utf-8"><h1>'+path.stem+'</h1>'+''.join(f'<p><a href="compare_{m}.png">{m}: 256 / 128</a></p>' for m in METRICS)+''.join(f'<p><a href="{f.name}">{f.name}</a></p>' for f in sorted(dest.glob('representative_*.png')))+''.join(f'<h2>{n}px</h2><a href="size_{n}/block_locations.png">Block positions</a>'+''.join(f'<p><a href="size_{n}/{a}">{a}</a></p>' for a in page_links[n]) for n in SIZES),encoding='utf-8')
            print(f'{path.name}: '+str(report[-1]['counts']),flush=True)
        except Exception as e:
            plt.close('all'); errors.append(dict(filename=path.name,error=str(e))); print(f'ERROR {path.name}: {e}',flush=True)
    for n in SIZES: write_csv(out/f'all_features_{n}.csv',allrows[n])
    write_csv(out/'all_representative_features.csv',reps)
    (out/'run_summary.json').write_text(json.dumps(dict(input_dir=str(args.input.resolve()),band=args.band,sizes=SIZES,images=report,errors=errors,
        total_blocks={str(n):len(allrows[n]) for n in SIZES}),indent=2),encoding='utf-8')
    (out/'index.html').write_text('<meta charset="utf-8"><h1>256 / 128 FFT comparison</h1><p>Exploratory band '+str(args.band)+' cycles/pixel; no defect classification.</p><p><a href="README.md">Explanation and findings</a></p>'+''.join(f'<p><a href="overview_{i+1:02d}.png">Overview {i+1}</a></p>' for i in range((len(report)+5)//6))+''.join(f'<p><a href="{Path(r["filename"]).stem}/index.html">{r["filename"]}</a></p>' for r in report),encoding='utf-8')
    if errors: raise SystemExit(1)
    from review_multiscale_1d_fft import main as review
    review(['--output',str(out)])


if __name__=='__main__': main()
