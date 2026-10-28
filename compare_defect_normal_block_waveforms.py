"""Compare annotated defect blocks with nearby unlabeled grid blocks.

This is a visualization aid, not a supervised detector. Controls are unlabeled
2D-grid-classified blocks and must not be interpreted as confirmed normal.
"""
import argparse
import csv
import html
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from block_1d_fft_experiment import ROOT, EXTENSIONS, read_gray
from evaluate_labeled_fft import load_labels, intersect


def spectra(block, dc_bins=1):
    a=np.asarray(block,dtype=float)
    sx=np.abs(np.fft.rfft(a,axis=1)).mean(axis=0)[dc_bins:]
    sy=np.abs(np.fft.rfft(a,axis=0)).mean(axis=1)[dc_bins:]
    sx=sx/max(float(np.linalg.norm(sx)),1e-15)
    sy=sy/max(float(np.linalg.norm(sy)),1e-15)
    return sx,sy


def profile(block):
    a=np.asarray(block,dtype=float)
    return a.mean(axis=0)-a.mean(),a.mean(axis=1)-a.mean()


def run(output_root,input_dir,top_controls=2):
    output_root=Path(output_root); input_dir=Path(input_dir)
    classified_path=output_root/'similarity_3class_blocks.csv'
    with classified_path.open(encoding='utf-8-sig',newline='') as stream:
        classes=list(csv.DictReader(stream))
    one_d_path=output_root/'results_1d_fft'/'blocks_1d_fft_scores.csv'
    one_d={}
    if one_d_path.exists():
        with one_d_path.open(encoding='utf-8-sig',newline='') as stream:
            one_d={(r['image_name'],int(r['block_size']),r['block_id']):r for r in csv.DictReader(stream)}
    by_group={}
    for r in classes: by_group.setdefault((r['image_name'],int(r['block_size'])),[]).append(r)
    image_paths=sorted((p for p in input_dir.iterdir() if p.suffix.lower() in EXTENSIONS and not p.name.startswith('._')),
                       key=lambda p:(int(p.stem[5:]) if p.stem.startswith('image') and p.stem[5:].isdigit() else 10**9,p.name.casefold()))
    out=output_root/'results_1d_fft'/'waveform_comparisons'; out.mkdir(parents=True,exist_ok=True)
    index=[]; records=[]
    for path in image_paths:
        gray=read_gray(path); labels=load_labels(path.with_suffix('.json'),gray.shape)
        for size in (256,128):
            rows=by_group.get((path.name,size),[])
            if not rows: continue
            all_label_boxes=[l['box'] for l in labels]
            blocks=[]
            for r in rows:
                x,y=int(r['x']),int(r['y']); n=size
                crop=gray[y:y+n,x:x+n]
                if crop.shape!=(n,n): continue
                blocks.append(dict(row=r,x=x,y=y,crop=crop,box=(x,y,x+n,y+n),
                    one_d=one_d.get((path.name,size,r['block_id']),{})))
            controls=[b for b in blocks if b['row']['similarity_class']=='grid' and
                      all(intersect(b['box'],box)==0 for box in all_label_boxes)]
            for label in labels:
                overlaps=[(intersect(b['box'],label['box']),b) for b in blocks if intersect(b['box'],label['box'])>0]
                if not overlaps: continue
                # Use the block containing the largest annotated area; stable tie-break by center distance.
                lb=label['box']; cx,cy=(lb[0]+lb[2])/2,(lb[1]+lb[3])/2
                overlaps.sort(key=lambda q:(-q[0],np.hypot(q[1]['x']+size/2-cx,q[1]['y']+size/2-cy),q[1]['y'],q[1]['x']))
                target=overlaps[0][1]
                candidates=sorted(controls,key=lambda b:(np.hypot(b['x']+size/2-(target['x']+size/2),b['y']+size/2-(target['y']+size/2)),b['y'],b['x']))[:top_controls]
                samples=[('defect-labeled block',target,'#d62728')]+[(f'unlabeled grid control {i}',b,c) for i,(b,c) in enumerate(zip(candidates,['#2ca02c','#1f77b4']),1)]
                block_id=target['row']['block_id']
                filename=f'{path.stem}_{label["id"].split("_")[-1]}_{size}_{block_id}.png'
                fig,axes=plt.subplots(2,2,figsize=(14,9))
                # Profiles are mean intensity traces across rows/columns, mean-centered.
                for name,b,color in samples:
                    px,py=profile(b['crop'])
                    axes[0,0].plot(np.arange(size),px,color=color,lw=1.25,label=name)
                    axes[0,1].plot(np.arange(size),py,color=color,lw=1.25,label=name)
                    fx,fy=spectra(b['crop'])
                    freq=np.fft.rfftfreq(size)[1:]
                    axes[1,0].plot(freq,fx,color=color,lw=1.35,label=name)
                    axes[1,1].plot(freq,fy,color=color,lw=1.35,label=name)
                for ax,title,xlabel,ylabel in [
                    (axes[0,0],'x-direction intensity waveform (row-mean)','x pixel in block','mean-centered intensity'),
                    (axes[0,1],'y-direction intensity waveform (column-mean)','y pixel in block','mean-centered intensity'),
                    (axes[1,0],'x-direction 1D FFT shape (DC excluded)','frequency [cycles/pixel]','L2-normalized amplitude'),
                    (axes[1,1],'y-direction 1D FFT shape (DC excluded)','frequency [cycles/pixel]','L2-normalized amplitude')]:
                    ax.set(title=title,xlabel=xlabel,ylabel=ylabel); ax.grid(alpha=.22); ax.legend(fontsize=8)
                detail=[]
                for name,b,_ in samples:
                    r=b['row']; d=b['one_d']
                    dtext=f", 1D sim_x/y {float(d['sim_x']):.3f}/{float(d['sim_y']):.3f}" if d.get('sim_x') and d.get('sim_y') else ''
                    detail.append(f"{name}: {r['block_id']} ({r['similarity_class']}, 2D log cosine {float(r['cosine_similarity_log']):.3f}{dtext})")
                fig.suptitle(f'{path.name} | {label["id"]} ({label["label"]}) | {size}px blocks\n'+'\n'.join(detail),fontsize=11)
                fig.tight_layout(rect=(0,0,1,.90)); fig.savefig(out/filename,dpi=145); plt.close(fig)
                # Context image with target and control positions.
                context=out/f'{Path(filename).stem}_locations.png'
                fig,ax=plt.subplots(figsize=(9,6)); ax.imshow(gray,cmap='gray')
                for name,b,color in samples:
                    ax.add_patch(Rectangle((b['x'],b['y']),size,size,fill=False,edgecolor=color,lw=2,label=name))
                ax.set_title(f'{path.name} | {label["id"]}: red=label-overlap block, green/blue=unlabeled grid controls')
                ax.legend(loc='upper right',fontsize=8); ax.axis('off'); fig.tight_layout(); fig.savefig(context,dpi=120); plt.close(fig)
                record=dict(image_name=path.name,annotation_id=label['id'],defect_label=label['label'],block_size=size,
                    target_block_id=block_id,target_class=target['row']['similarity_class'],
                    target_cosine_similarity_2d=target['row']['cosine_similarity_log'],
                    target_sim_x=target['one_d'].get('sim_x'),target_sim_y=target['one_d'].get('sim_y'),
                    target_min_similarity=target['one_d'].get('min_similarity'),control_count=len(candidates))
                for i,b in enumerate(candidates,1):
                    record[f'control_{i}_block_id']=b['row']['block_id']
                    record[f'control_{i}_class']=b['row']['similarity_class']
                    record[f'control_{i}_cosine_similarity_2d']=b['row']['cosine_similarity_log']
                    record[f'control_{i}_sim_x']=b['one_d'].get('sim_x')
                    record[f'control_{i}_sim_y']=b['one_d'].get('sim_y')
                records.append(record)
                index.append(f'<article><h3>{html.escape(path.name)} / {html.escape(label["id"])} ({html.escape(label["label"])}) / {size}px</h3><p>{html.escape(" | ".join(detail))}</p><a href="{html.escape(filename)}"><img src="{html.escape(filename)}" loading="lazy"></a><a href="{html.escape(context.name)}"><img src="{html.escape(context.name)}" loading="lazy"></a></article>')
    with (out/'waveform_comparisons.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        if records:
            writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in records for k in r))); writer.writeheader(); writer.writerows(records)
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Defect and unlabeled block waveforms</title><style>body{font:15px system-ui;margin:24px;background:#f3f4f6;color:#222}article{background:white;padding:14px;margin:18px 0;border-radius:8px}img{width:min(100%,1000px);display:block;margin:10px 0}p{color:#444}</style><h1>Block-level waveform comparison</h1><p>Controls are unlabeled grid-classified blocks, not confirmed normal. The target is the block with the largest overlap with each defect annotation. 1D FFT curves are L2-normalized after excluding DC to compare shape.</p>'+''.join(index)+'</html>'
    (out/'index.html').write_text(page,encoding='utf-8')
    (out/'README.md').write_text('# Block-level waveform comparison\n\n各傷ラベルについて、ラベル矩形を最も多く含むブロックと、同一画像・同一サイズの最近傍unlabeled gridブロック最大2個を比較します。unlabeledは正常確定ではなく、比較用候補です。各図はx/yの明度プロファイルと、各行/各列の1D rFFT振幅平均（DC除外後にL2正規化）を重ねます。傷ラベルは比較対象の選択だけに使用し、スコアや基準スペクトルの学習には使用しません。\n\n`index.html`に全比較図、`waveform_comparisons.csv`にブロックIDと2D分類/類似度を保存しています。')
    print(f'Created {len(records)} defect-block comparisons: {out}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=ROOT/'outputs/fft2_peak_match')
    parser.add_argument('--input',type=Path,default=ROOT/'data20260903/data20260903')
    parser.add_argument('--top-controls',type=int,default=2)
    args=parser.parse_args()
    if args.top_controls<1: parser.error('--top-controls must be >=1')
    run(args.output_root,args.input,args.top_controls)


if __name__=='__main__': main()
