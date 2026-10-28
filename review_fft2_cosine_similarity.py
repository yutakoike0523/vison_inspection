"""Verify saved cosine results, refresh report, and render 22-label percentile chart."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from fft2_cosine_similarity import ROOT,SIZES,report,maps,read_gray,load_labels,fft_amplitude,common_spectrum,vector,cosine,percentile


def read_csv(path):
    rows=list(csv.DictReader(path.open(encoding='utf-8-sig')))
    for r in rows:
        for k,v in r.items():
            if v=='': r[k]=None
            else:
                try: r[k]=float(v)
                except ValueError: pass
        for k in ('block_size','x_start','y_start','block_row','block_col','x','y','width','height','stride'):
            if k in r: r[k]=int(r[k])
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'outputs/fft2_cosine_similarity')
    p.add_argument('--redraw-maps',action='store_true')
    args=p.parse_args(); out=args.output
    settings=json.loads((out/'settings.json').read_text()); modes=settings['spectrum_modes']
    blocks=read_csv(out/'all_blocks.csv'); annotations=read_csv(out/'annotation_percentiles.csv'); summaries=read_csv(out/'image_summary.csv')
    names=sorted(set(r['image_name'] for r in blocks)); files=[Path(settings['input'])/f for f in names]
    checks=[]
    for path in files:
        gray=read_gray(path); labels=load_labels(path.with_suffix('.json'),gray.shape)
        for n in SIZES:
            dest=out/str(n)/path.stem; rows=[r for r in blocks if r['image_name']==path.name and r['block_size']==n]
            expected=max(0,(gray.shape[0]-n)//(n//2)+1)*max(0,(gray.shape[1]-n)//(n//2)+1)
            assert len(rows)==expected
            ref=np.load(dest/'reference_spectrum.npz'); common,mask=ref['amplitude'],ref['mask']
            for mode in modes:
                key='cosine_similarity_'+mode
                valid=[r for r in rows if r[key] is not None]; values=np.array([r[key] for r in valid])
                assert np.all((values>=0)&(values<=1))
                np.testing.assert_allclose([r['similarity_rank_'+mode] for r in valid],rankdata(values))
                np.testing.assert_allclose([r['similarity_minus_image_median_'+mode] for r in valid],values-np.median(values))
                if rows:
                    r=rows[len(rows)//2]; x,y=r['x'],r['y']
                    _,a=fft_amplitude(gray[y:y+n,x:x+n],settings['mean_subtraction'])
                    score=cosine(vector(common,mask,mode),vector(common_spectrum(a,settings['grid_size'],settings['frequency_max']),mask,mode))
                    if score is None: assert r[key] is None
                    else: assert abs(score-r[key])<1e-12
                for a in annotations:
                    if a['image_name']==path.name and a['block_size']==n and a['mode']==mode:
                        v=[r[key] for r in valid if a['annotation_id'] in (r['annotation_ids'] or '').split(';')]
                        assert len(v)==a['valid_overlap_blocks']
                        if v:
                            assert abs(a['cosine_similarity']-np.median(v))<1e-12
                            assert abs(a['image_percentile']-percentile(values,float(np.median(v))))<1e-10
                if args.redraw_maps: maps(gray,rows,labels,dest,n,mode,settings['primary_mode'],include_overlay=False)
            if not labels: assert all(r['label']=='unknown' for r in rows)
            checks.append(dict(image=path.name,size=n,blocks=len(rows),status='passed'))
        print(f'Verified {path.name}',flush=True)
    assert len(blocks)==settings['blocks']; assert not settings['errors']
    report(out,files,annotations,summaries,settings)
    ids=list(dict.fromkeys(r['annotation_id'] for r in annotations))
    columns=[(n,m) for n in SIZES for m in modes]
    matrix=np.full((len(ids),len(columns)),np.nan)
    for i,identity in enumerate(ids):
        for j,(n,m) in enumerate(columns):
            r=next(r for r in annotations if r['annotation_id']==identity and r['block_size']==n and r['mode']==m)
            if r['image_percentile'] is not None: matrix[i,j]=r['image_percentile']
    fig,ax=plt.subplots(figsize=(9,max(5,.35*len(ids))))
    im=ax.imshow(np.ma.masked_invalid(matrix),cmap='RdYlBu',vmin=0,vmax=100,aspect='auto')
    ax.set_yticks(range(len(ids)),ids); ax.set_xticks(range(len(columns)),[f'{n} / {m}' for n,m in columns])
    for i in range(len(ids)):
        for j in range(len(columns)):
            ax.text(j,i,'NA' if np.isnan(matrix[i,j]) else f'{matrix[i,j]:.1f}%',ha='center',va='center',fontsize=8,
                    color='white' if matrix[i,j]<20 or matrix[i,j]>80 else 'black')
    ax.set_title('Annotated bbox overlap-median similarity: within-image percentile\nLower = relatively less similar; NOT detection accuracy')
    fig.colorbar(im,ax=ax,label='Within-image percentile [%]'); fig.tight_layout()
    fig.savefig(out/'annotation_percentile_comparison.png',dpi=130); plt.close(fig)
    index=out/'index.html'; t=index.read_text(encoding='utf-8')
    t=t.replace('<h1>', '<p><a href="annotation_percentile_comparison.png">22 defect locations: percentile comparison</a></p><h1>',1)
    index.write_text(t,encoding='utf-8')
    (out/'validation.json').write_text(json.dumps(dict(checks=checks,blocks=len(blocks),annotation_rows=len(annotations),annotation_count=len(ids),status='passed'),indent=2),encoding='utf-8')
    print(f'PASS: {len(blocks)} blocks, {len(ids)} labels, {len(annotations)} label/size/mode results',flush=True)


if __name__=='__main__': main()
