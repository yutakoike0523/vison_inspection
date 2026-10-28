"""Add spatial localization and saved-experiment timing summaries."""
import csv
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import evaluate_all_radial_fft as prev
import multiscale_fft_experiment as base

out=Path('output_low_frequency_analysis'); freq=Path('output_frequency_band_analysis'); prev.N=256
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
loio=read(freq/'fixed_references'/'summary'/'loio_metrics.csv'); spatial=[]
for i in range(1,22):
    f=freq/'fixed_references'/'images'/f'image{i}'/'scores_and_selections.npz'
    if not f.exists(): continue
    image=np.asarray(Image.open(base.DATA/f'image{i}.png').convert('L')); labels=prev.annotation(base.DATA/f'image{i}.json',image.shape)
    if not labels: continue
    with np.load(f) as a:
        X,Y=np.meshgrid(a['x_centers'],a['y_centers']); masks=[m for name,m in labels if name=='局所変形']; defect=np.logical_or.reduce(masks) if masks else np.zeros(image.shape,bool); dist=distance_transform_edt(~defect)[Y,X] if masks else None
        for band in ('A','C'):
            row=next(z for z in loio if int(z['image'])==i and z['condition']==band); thr=float(row['threshold']); score=a[f'score_{band}']; exceed=np.isfinite(score)&(score>thr); count=int(exceed.sum())
            spatial.append(dict(image=i,condition=band,exceed_all_locations=count,exceed_centers_in_defect=int(np.sum(exceed&defect[Y,X])),exceed_within_128px_of_defect=int(np.sum(exceed&(dist<=128))) if dist is not None else None,exceed_farther_than_256px_from_defect=int(np.sum(exceed&(dist>=256))) if dist is not None else None,definition='centers within image-level union of deformation annotations; includes labeled non-normal areas'))
prev.safe_csv(out/'summary'/'spatial_localization_metrics.csv',spatial)
fixed=read(freq/'fixed_references'/'summary'/'processing_times.csv'); specific=read(freq/'band_specific_references'/'summary'/'processing_times.csv');
rows=[]
for name,src,condition in [('A',fixed,'A'),('B',fixed,'C'),('C',specific,'C')]:
    q=[float(z['band_score_and_selection_seconds']) for z in src if z['group']=='grid' and z['condition']==condition]
    rows.append(dict(condition=name,mean_seconds=float(np.mean(q)),scope='cached FFT; fixed uses saved selection; C includes band-specific selection'))
prev.safe_csv(out/'reference_selection'/'condition_processing_times.csv',rows)
changes=read(out/'detection_comparison'/'per_defect_detection.csv'); ref=read(out/'reference_selection'/'representative_selection_comparison.csv'); fp=read(out/'summary'/'image_false_positive_metrics.csv')
lines=['','## 空間分布と処理時間（追加集計）']
for i in (5,15,17):
    for band in ('A','C'):
        q=next(z for z in spatial if z['image']==i and z['condition']==band); lines.append(f"- image{i} {band}: 全超過 {q['exceed_all_locations']}地点、傷領域内中心 {q['exceed_centers_in_defect']}地点、傷から128 px以内 {q['exceed_within_128px_of_defect']}地点、256 px以上離れた地点 {q['exceed_farther_than_256px_from_defect']}。全ラベル領域を含む全超過地点の集計。")
for z in rows: lines.append(f"- {z['condition']}の画像当たり平均 {z['mean_seconds']:.3f}秒（既存特徴量からの帯域計算、Cは選択処理を含む。FFT・読込・図保存は含まない）。")
for img in (5,15,17):
    q=[z for z in ref if int(z['image'])==img]
    for z in q: lines.append(f"- image{img} {z['method']}: 代表点の選択参照 {z['ids']}、参照間の低周波RMS平均 {float(z['mean_pairwise_low_rms']):.3f} dB、背景平均 {float(z['mean_background_fraction']):.1%}、局所変形平均 {float(z['mean_deformation_fraction']):.1%}。")
p=out/'summary'/'report.md'; p.write_text(p.read_text(encoding='utf-8').rstrip()+'\n'+'\n'.join(lines)+'\n',encoding='utf-8')
