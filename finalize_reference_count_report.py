"""Add reproducible supplemental summaries to an existing run."""
import csv
from pathlib import Path
import numpy as np

p=Path('output_reference_count_comparison/summary')
def read(name):
    with (p/name).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
m=read('defect_instance_metrics.csv'); t=read('processing_times.csv'); ag=read('reference_count_metrics.csv')
report=p/'report.md'; s=report.read_text(encoding='utf-8').split('\n## ????')[0]
extra=['','## 無格子画像（別LOIO、全有効地点）']
for z in ag:
    if z['group']=='no_grid' and z['domain']=='all': extra.append(f"- K={z['K']}: 検出 {z['detected']}/{z['evaluable']}、正常超過率の画像平均 {float(z['normal_exceed_image_mean']):.2%}、傷Sの中央値 {float(z['S_median']):.3f}。")
extra+=['','## image15・17と計算時間']
for i in (15,17):
    q=[z for z in m if z['group']=='grid' and z['domain']=='all' and int(z['image'])==i]
    extra.append(f"- image{i}の各KのS: "+', '.join(f"{z['K']}={float(z['S']):.3f}" for z in q)+'。')
for k in (1,4,5,8):
    q=[z for z in t if z['group']=='grid' and int(z['K'])==k]
    extra.append(f"- K={k}のFFT計算後の単独実行相当時間の画像平均 {np.mean([float(z['total_if_separate_seconds']) for z in q]):.3f}秒。")
extra+=['','image18は候補数3以上の窓が0点で、ラベルもないため性能評価はできない。image15・17の全地点でK=8のスコアは従来の全参照方式と完全一致、K=5は前回D方式と完全一致した。最上端中心y=128で3参照以上が得られる地点が有効になることも確認した。']
report.write_text(s.rstrip()+'\n'+'\n'.join(extra)+'\n',encoding='utf-8')
