"""Add per-instance, per-image and bin summaries to the frequency-band report."""
import csv
from pathlib import Path
import numpy as np
import evaluate_all_radial_fft as prev

root=Path('output_frequency_band_analysis'); overall=root/'overall_comparison'; names=('A','B','C','D','E','F')
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
fixed=read(root/'fixed_references'/'summary'/'loio_metrics.csv'); specific=read(root/'band_specific_references'/'summary'/'loio_metrics.csv'); bins=read(root/'fixed_references'/'summary'/'band_definitions.csv'); metrics=read(root/'fixed_references'/'summary'/'defect_instance_metrics.csv')
combined=[]
for x in fixed:
    y=next(z for z in specific if z['image']==x['image'] and z['instance']==x['instance'] and z['condition']==x['condition'])
    combined.append(dict(image=x['image'],group=x['group'],instance=x['instance'],condition=x['condition'],fixed_threshold=x['threshold'],reselected_threshold=y['threshold'],fixed_detected=x['detected'],reselected_detected=y['detected'],fixed_normal_exceed_rate=x['normal_exceed_rate'],reselected_normal_exceed_rate=y['normal_exceed_rate'],fixed_defect_exceed_rate=x['defect_exceed_rate'],reselected_defect_exceed_rate=y['defect_exceed_rate']))
prev.safe_csv(overall/'fixed_vs_reselected_instances.csv',combined)
lines=['','## ビン数と代表的な画像の差']
for condition in names:
    q=[z for z in bins if z['condition']==condition]; lines.append(f"- {condition}: {len(q)}ビン、中心 {float(q[0]['frequency_center']):.3f}〜{float(q[-1]['frequency_center']):.3f} cycles/pixel。")
for img in (5,15,17):
    q=[z for z in fixed if z['image']==str(img) and z['instance']]; lines.append(f"- image{img}の参照固定：検出条件 "+(', '.join(z['condition'] for z in q if z['detected']=='True') or 'なし')+'。')
for img in (3,8,9,13):
    for inst in sorted({z['instance'] for z in fixed if z['image']==str(img) and z['instance']}):
        q=[z for z in fixed if z['image']==str(img) and z['instance']==inst]; lines.append(f"- image{img} 傷{inst}：参照固定で検出した帯域 "+(', '.join(z['condition'] for z in q if z['detected']=='True') or 'なし')+'。')
for mode,arr in [('固定',fixed),('再選択',specific)]:
    q=[z for z in arr if z['image']=='11' and z['condition'] in ('A','B','C')]; lines.append(f"- image11の正常窓超過率（{mode}、A/B/C）："+' / '.join(f"{float(z['normal_exceed_rate']):.2%}" for z in sorted(q,key=lambda z:z['condition']))+'。')
lines+=['','C（低周波のみ）は4ビンと狭く、このデータでは全帯域より検出数が増えた。これは候補帯域としての根拠にはなるが、個別画像の正常分布や見逃しも確認し、帯域選択を固定した後に未使用画像で再評価する必要がある。']
p=overall/'final_report.md'; p.write_text(p.read_text(encoding='utf-8').rstrip()+'\n'+'\n'.join(lines)+'\n',encoding='utf-8')
