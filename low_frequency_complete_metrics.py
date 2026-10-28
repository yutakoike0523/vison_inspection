"""Add the previously saved, band-specific S to the comparison tables."""
import csv
from pathlib import Path
import evaluate_all_radial_fft as prev

root=Path('output_low_frequency_analysis'); path=Path('output_frequency_band_analysis/band_specific_references/summary/defect_instance_metrics.csv')
with path.open(encoding='utf-8-sig',newline='') as f: metrics=list(csv.DictReader(f))
for p in [root/'detection_comparison'/'per_defect_detection.csv',root/'detection_comparison'/'detection_changes.csv',root/'summary'/'defect_metrics.csv']:
    with p.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    for r in rows:
        m=next(v for v in metrics if v['image']==r['image'] and v['instance']==r['instance'] and v['condition']=='C')
        r['low_reselected_S']=m['S']
    prev.safe_csv(p,rows)
