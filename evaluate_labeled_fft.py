"""Join LabelMe rectangles to existing 256/128 FFT features; no normal labels inferred.
Run: python evaluate_labeled_fft.py
"""
import argparse
import csv
import html
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from block_1d_fft_experiment import ROOT, read_gray, analyze, comparison, write_csv
from review_multiscale_1d_fft import read_rows

SIZES=(256,128)
KEYS=[f'{d}_hann_{m}' for d in ('x','y') for m in ('peak_amplitude','peak_frequency','peak_energy_ratio','band_energy_ratio','band_peak_amplitude')]


def area(box):
    return max(0,box[2]-box[0])*max(0,box[3]-box[1])


def intersect(a,b):
    return area((max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])))


def block_box(r):
    x,y,n=r['x_start'],r['y_start'],r['block_size']
    return (x,y,x+n,y+n)


def load_labels(path, shape):
    if not path.exists(): return []
    data=json.loads(path.read_text(encoding='utf-8'))
    if (data['imageHeight'],data['imageWidth'])!=shape:
        raise ValueError(f'{path.name}: annotation/image dimensions differ')
    labels=[]
    for i,s in enumerate(data['shapes']):
        if s['shape_type']!='rectangle' or len(s['points'])!=2:
            raise ValueError(f'{path.name}: unsupported shape')
        pts=np.array(s['points'],dtype=float)
        if not np.isfinite(pts).all(): raise ValueError('Nonfinite annotation')
        x0,y0=pts.min(0); x1,y1=pts.max(0)
        box=(float(x0),float(y0),float(x1),float(y1))
        if area(box)<=0 or x0<0 or y0<0 or x1>shape[1] or y1>shape[0]:
            raise ValueError(f'{path.name}: invalid rectangle')
        labels.append(dict(id=f'{path.stem}_{i+1:02d}',label=s['label'],box=box))
    return labels


def nearby(rows, label, labels, n):
    b=label['box']; cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2
    eligible=[]
    for r in rows:
        bb=block_box(r)
        distance=np.hypot((bb[0]+bb[2])/2-cx,(bb[1]+bb[3])/2-cy)
        if distance<=3*n and all(intersect(bb,l['box'])==0 for l in labels):
            eligible.append((distance,r))
    return [r for _,r in sorted(eligible,key=lambda t:(t[0],t[1]['block_row'],t[1]['block_col']))[:12]]


def summarize_directions(metricrows,out):
    summary=[]
    lines=['','## 今回の数値で分かったこと','',
        '傷と重なる走査ブロック群の中央値と、近傍の未ラベルブロック群の中央値の差を集計した。',
        '以下は差の符号の件数であり、差の有意性や検出成功件数ではない。小さい差も含む。',
        '|指標|256：周辺より低い/高い|128：周辺より低い/高い|サイズ間で符号逆転|',
        '|---|---:|---:|---:|']
    for key in [f'{d}_hann_{m}' for m in ('band_peak_amplitude','band_energy_ratio') for d in ('x','y')]:
        bysize={n:{r['annotation_id']:float(r['difference']) for r in metricrows if int(r['block_size'])==n and r['metric']==key and r['difference'] not in (None,'')} for n in SIZES}
        common=bysize[256].keys() & bysize[128].keys()
        reverse=sum(bysize[256][i]*bysize[128][i]<0 for i in common)
        row=dict(metric=key,count_256=len(bysize[256]),lower_256=sum(v<0 for v in bysize[256].values()),higher_256=sum(v>0 for v in bysize[256].values()),
            count_128=len(bysize[128]),lower_128=sum(v<0 for v in bysize[128].values()),higher_128=sum(v>0 for v in bysize[128].values()),sign_reversals=reverse)
        summary.append(row)
        lines.append(f'|{key}|{row["lower_256"]}/{row["higher_256"]}|{row["lower_128"]}/{row["higher_128"]}|{reverse}|')
    lines.extend(['', 'この比較では「ピーク振幅・帯域比が低い側だけを傷候補にする」方式は支持されない。大小の変化には照明、位置、ブロック内の傷の割合、平均化による打ち消し、周辺比較領域の内容が混ざる。',
        'したがって、256で低い値を示した場所だけを128へ進める方法は保留する。次は、比較相手のブロックが正常格子かを確認し、正常範囲から上下どちらにも離れるかを検証する。',
        '大きいラベルの全体を一つの中心ブロックで代表させない。中心比較図と、ラベルに重なる全走査ブロックの分布・被覆率を併せて確認する。'])
    write_csv(out/'direction_summary.csv',summary)
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'data20260903/data20260903')
    p.add_argument('--features',type=Path,default=ROOT/'outputs/multiscale_1d_fft')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/labeled_fft_comparison')
    args=p.parse_args()
    source=json.loads((args.features/'run_summary.json').read_text())
    band=source['band']; out=args.output
    if out.exists(): out=out.with_name(out.name+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True); print(f'OUTPUT {out}',flush=True)
    allblocks=[]; links=[]; summaries=[]; metricrows=[]; inventories=[]; errors=[]; labelcounts=Counter()
    for entry in source['images']:
        filename=entry['filename']; stem=Path(filename).stem
        try:
            gray=read_gray(args.input/filename); labels=load_labels(args.input/(stem+'.json'),gray.shape)
            inventories.append(dict(filename=filename,status='annotated_positive_locations' if labels else 'unknown_no_positive_annotation',label_count=len(labels)))
            labelcounts.update(l['label'] for l in labels)
            rows={n:read_rows(args.features/stem/f'size_{n}/features.csv') for n in SIZES}
            for n in SIZES:
                for r in rows[n]:
                    hits=[l for l in labels if intersect(block_box(r),l['box'])>0]
                    allblocks.append(dict(**r,annotation_status='intersects_positive_bbox' if hits else 'unlabeled_unknown',
                        annotation_ids=';'.join(l['id'] for l in hits),annotation_types=';'.join(l['label'] for l in hits)))
            if not labels: continue
            dest=out/stem; dest.mkdir()
            fig,axes=plt.subplots(1,2,figsize=(16,7))
            for ax,n in zip(axes,SIZES):
                ax.imshow(gray,cmap='gray')
                for r in rows[n]:
                    if any(intersect(block_box(r),l['box'])>0 for l in labels):
                        ax.add_patch(Rectangle((r['x_start'],r['y_start']),n,n,fill=False,edgecolor='orange',lw=.7))
                for l in labels:
                    x0,y0,x1,y1=l['box']; ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='red',lw=2))
                    ax.text(x0,y0,l['id']+' '+l['label'],color='red',fontsize=7,backgroundcolor='white')
                ax.set_title(f'{stem}: {n}px / stride {n//2}\nRed=annotation bbox; orange=overlapping FFT blocks')
            fig.tight_layout(); fig.savefig(dest/'label_block_locations.png',dpi=120); plt.close(fig)
            for l in labels:
                samples=[]
                for n in SIZES:
                    hit=[r for r in rows[n] if intersect(block_box(r),l['box'])>0]
                    refs=nearby(rows[n],l,labels,n)
                    b=l['box']; cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2
                    x=max(0,min(gray.shape[1]-n,int(round(cx-n/2))))
                    y=max(0,min(gray.shape[0]-n,int(round(cy-n/2))))
                    target,s=analyze(gray[y:y+n,x:x+n],band)
                    coverage=intersect((x,y,x+n,y+n),b)/area(b)
                    samples.append((f'{l["id"]} {l["label"]} | {n}px\nLabel-center patch; bbox covered {coverage:.0%}',s))
                    if refs:
                        rr=refs[0]; rx,ry=rr['x_start'],rr['y_start']
                        _,rs=analyze(gray[ry:ry+n,rx:rx+n],band)
                        samples.append((f'{n}px nearest unlabeled comparison\nNOT verified normal; x={rx}, y={ry}',rs))
                    summaries.append(dict(annotation_id=l['id'],filename=filename,label=l['label'],block_size=n,
                        bbox_x0=b[0],bbox_y0=b[1],bbox_x1=b[2],bbox_y1=b[3],overlap_block_count=len(hit),
                        reference_count=len(refs),center_patch_x=x,center_patch_y=y,center_bbox_coverage=coverage,
                        max_scanned_block_bbox_coverage=max((intersect(block_box(r),b)/area(b) for r in hit),default=0)))
                    for r in hit:
                        overlap=intersect(block_box(r),b)
                        links.append(dict(annotation_id=l['id'],filename=filename,label=l['label'],block_size=n,
                            block_row=r['block_row'],block_col=r['block_col'],x_start=r['x_start'],y_start=r['y_start'],
                            intersection_area=overlap,block_bbox_fraction=overlap/(n*n),annotation_coverage=overlap/area(b)))
                    for key in KEYS:
                        a=np.array([r[key] for r in hit if r[key] is not None],float)
                        ref=np.array([r[key] for r in refs if r[key] is not None],float)
                        am=float(np.median(a)) if len(a) else None
                        rm=float(np.median(ref)) if len(ref) else None
                        q1,q3=np.percentile(ref,[25,75]) if len(ref) else (None,None)
                        metricrows.append(dict(annotation_id=l['id'],filename=filename,label=l['label'],block_size=n,metric=key,
                            overlap_median=am,overlap_min=float(a.min()) if len(a) else None,overlap_max=float(a.max()) if len(a) else None,
                            unlabeled_reference_median=rm,reference_q25=q1,reference_q75=q3,
                            difference=am-rm if am is not None and rm is not None else None,
                            centered_value=target[key],centered_minus_reference=target[key]-rm if target[key] is not None and rm is not None else None,
                            reference_count=len(ref),reference_coordinates=json.dumps([[r['block_row'],r['block_col']] for r in refs])))
                comparison(samples,dest/(l['id']+'_spectra.png'))
                comparison(samples,dest/(l['id']+'_zoom.png'),fmax=.12)
            (dest/'index.html').write_text('<meta charset="utf-8"><h1>'+stem+'</h1><p>Unlabeled comparisons are NOT confirmed normal.</p><a href="label_block_locations.png">Label / block positions</a>'+''.join(f'<p><a href="{l["id"]}_zoom.png">{l["id"]}: {l["label"]} - low frequency</a> / <a href="{l["id"]}_spectra.png">full FFT</a></p>' for l in labels),encoding='utf-8')
            print(f'{stem}: {len(labels)} labels compared',flush=True)
        except Exception as e:
            plt.close('all'); errors.append(dict(filename=filename,error=str(e))); print(f'ERROR {filename}: {e}',flush=True)
    for name,data in [('block_features_with_labels',allblocks),('annotation_block_links',links),('annotation_summary',summaries),('metric_comparison',metricrows),('image_label_status',inventories)]:
        write_csv(out/(name+'.csv'),data)
    report=dict(images=len(inventories),annotated_images=sum(r['label_count']>0 for r in inventories),label_counts=dict(labelcounts),
        total_labels=sum(labelcounts.values()),blocks=len(allblocks),annotation_size_pairs=len(summaries),band=band,errors=errors,
        unknown_images=[r['filename'] for r in inventories if not r['label_count']],reference_rule='Nearest up to 12 same-size scanned blocks within 3*size from bbox center, intersecting no annotated bbox; NOT normal labels')
    (out/'run_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    # Display signed differences separately; do not combine into an anomaly score.
    ids=[r['annotation_id'] for r in summaries if r['block_size']==256]
    fig,axes=plt.subplots(2,2,figsize=(15,13))
    for ax,key in zip(axes.flat,[f'{d}_hann_{m}' for m in ('band_peak_amplitude','band_energy_ratio') for d in ('x','y')]):
        for n,color,offset in [(256,'steelblue',-.17),(128,'darkorange',.17)]:
            values=[next((r['difference'] for r in metricrows if r['annotation_id']==i and r['block_size']==n and r['metric']==key),None) for i in ids]
            ax.barh(np.arange(len(ids))+offset,[v if v is not None else np.nan for v in values],height=.33,label=str(n),color=color)
        ax.set_yticks(np.arange(len(ids)),ids,fontsize=7); ax.axvline(0,color='black',lw=.7); ax.set_title(key,fontsize=10)
        ax.set_xlabel('Overlapping block median - nearby UNLABELED median'); ax.legend()
    fig.suptitle('Annotated defects vs local unlabeled comparisons (not verified normal)\nExploratory band 0.01-0.04 cycles/pixel; no decision threshold')
    fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(out/'labeled_feature_differences.png',dpi=120); plt.close(fig)
    text=['# 傷ラベルと局所FFTの照合','',f'ラベル付き{report["annotated_images"]}画像、{report["total_labels"]}矩形。各256/128、計{len(summaries)}組。',
        '', '## 前提',
        'ラベル矩形と面積が少しでも重なる走査ブロックを抽出。矩形は傷画素マスクではない。重なり面積・ブロック内割合・矩形の被覆率を保存した。',
        '未ラベル画像・未ラベル領域は不明。正常とは扱わず、検出率や誤検出率・自動判定閾値は計算しない。',
        '周辺比較は同じ画像・サイズで全ラベル矩形に重ならない、ラベル中心から3ブロック幅以内の近い順12ブロックまで。背景、未知の傷を含み得る。重なり合う走査ブロックは独立標本ではない。',
        '各ラベルの中心付近を追加切り出しし、同じサイズの最も近い未ラベルブロックとFFTを比較。中心切り出しは走査ブロックの中央値とは別。大きい矩形は一つのブロックに収まらないので、被覆率も表示した。',
        '周辺との中央値差を256/128別に示す。正負の逆転や変化の大小は、傷以外の影響も含み、検出できた・見逃したという判定ではない。',
        '', '## 出力','- `index.html`：画像・傷ごとの比較図への入口。','- `labeled_feature_differences.png`：22ラベルの差を256/128で比較。',
        '- `metric_comparison.csv`：全帯域ピーク振幅・周波数・集中度、帯域比・帯域ピーク振幅のX/Y比較。',
        '- `block_features_with_labels.csv`：全ブロック特徴量と、矩形交差／未ラベル不明の区別。',
        '- `annotation_block_links.csv`：ラベルとブロックの対応、重なり割合。','- `annotation_summary.csv`：傷ごとの対象数・比較数・被覆率。',
        '- `image_label_status.csv`：ラベルなし画像も不明として記録。',
        '', '以前のimage15 r4,c8等の目視候補は正解ラベルに使っていない。今回のユーザーラベルのみを傷位置の基準にした。',
        '', '再実行：`python evaluate_labeled_fft.py`。既存出力があれば日時付きの新規フォルダに保存する。']
    (out/'README.md').write_text('\n'.join(text)+summarize_directions(metricrows,out),encoding='utf-8')
    (out/'index.html').write_text('<meta charset="utf-8"><h1>Annotated defect / FFT comparison</h1><p><a href="README.md">Explanation</a> / <a href="labeled_feature_differences.png">All label differences</a></p>'+''.join(f'<p><a href="{Path(r["filename"]).stem}/index.html">{r["filename"]}</a></p>' for r in inventories if r['label_count']),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)
    if errors: raise SystemExit(1)


if __name__=='__main__': main()
