#!/usr/bin/env python3
"""Frequency-band RMS with fixed and band-reselected K=4 references."""
import json, time, warnings
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import evaluate_all_radial_fft as prev
import multiscale_fft_experiment as base
import stride_grid_overlap_experiment as old
import reference_selection_analysis as ra

ROOT=base.ROOT; PRIOR=ROOT/'output_all_images_radial_fft'; KC=ROOT/'output_reference_count_comparison'; OUT=ROOT/'output_frequency_band_analysis'
IDS=('A','B','C','D','E','F'); RANGE={'A':(.01,.50),'B':(.05,.50),'C':(.01,.05),'D':(.05,.10),'E':(.10,.30),'F':(.30,.50)}

def readnp(path):
    with np.load(path) as z: return {k:z[k].copy() for k in z.files}
def save(path,rows): prev.safe_csv(path,rows)
def locpath(i): return (KC/f'image{i}' if i in (15,17) else KC/'other_images'/f'image{i}')/'locations.npz'
def masks(r):
    index=np.flatnonzero(r['bin_valid']); centers=(index+.5)*.01; result={}
    for key,(lo,hi) in RANGE.items():
        selected=(centers>=lo)&(centers<hi)
        # Preserve the final nonempty original bin, even if its representative center lies at the endpoint.
        if hi==.50 and len(index): selected[-1]=True
        if not np.any(selected): raise ValueError(f'No bins in {key}')
        result[key]=selected
    return centers,result

def candidates(r):
    ny,nx=r['scores'].shape; yy,xx=np.indices((ny,nx)); idx=np.full((ny*nx,8),-1,np.int32)
    for j,(dx,dy) in enumerate(ra.OFF):
        y=yy+dy*8; x=xx+dx*8; good=(y>=0)&(y<ny)&(x>=0)&(x<nx); idx[:,j]=np.where(good,y*nx+x,-1).ravel()
    valid=idx>=0; spectra=r['spectra'].reshape(-1,r['spectra'].shape[-1]); return idx,valid,spectra,spectra[np.maximum(idx,0)]

def calc_band(spectra,ng,valid,bits,band,reselect):
    subset=ng[:,:,band]; target=spectra[:,band]; n=len(target); count=valid.sum(1)
    if reselect:
        diff=subset[:,:,None,:]-subset[:,None,:,:]; dist=np.sqrt(np.mean(diff*diff,axis=-1)); del diff
        pair=valid[:,:,None]&valid[:,None,:]; means=np.where(valid,np.sum(np.where(pair,dist,0),axis=2)/np.maximum(count[:,None],1),np.inf)
        medoid=np.argmin(means,axis=1); distance=np.where(valid,dist[np.arange(n),medoid],np.inf); order=np.argsort(distance,axis=1,kind='stable'); ranks=np.argsort(order,axis=1,kind='stable'); selected=valid&(ranks<4)
    else: selected=bits & valid
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning); reference=np.nanmedian(np.where(selected[:,:,None],subset,np.nan),axis=1)
    score=np.sqrt(np.mean((target-reference)**2,axis=1)); score[count<3]=np.nan
    return score,np.packbits(selected,axis=1,bitorder='little').ravel()

def analyze(mode,out):
    root=out/mode; (root/'images').mkdir(parents=True); summary=root/'summary'; (summary/'figures').mkdir(parents=True); data={}; timings=[]; bins=[]; agreements=[]; checks=[]
    for i in range(1,22):
        start=time.perf_counter(); p=base.DATA/f'image{i}.png'; img=np.asarray(Image.open(p).convert('L')); labels=prev.annotation(p.with_suffix('.json'),img.shape); loc=readnp(locpath(i)); folder=root/'images'/f'image{i}'; folder.mkdir(); group='no_grid' if i in prev.NO_GRID else ('small_reference' if i==18 else 'grid'); valid=loc['candidate_count']>=3
        if not np.any(valid) or labels is None:
            (folder/'status.txt').write_text('No valid locations with >=3 reference candidates or missing annotation.\n',encoding='utf-8'); data[i]={'group':group,'status':'no valid locations or annotation','valid':int(valid.sum())}; continue
        r=old.load_result(PRIOR/f'image{i}'/'result.npz'); centers,band=masks(r); idx,ok,sp,ng=candidates(r); fixed=np.unpackbits(loc['selected_bits_k4'].reshape(-1,1),axis=1,bitorder='little').astype(bool); result={}; selection={}; nvalid=int(valid.sum())
        if i==2:
            for name in IDS:
                for j in np.flatnonzero(band[name]): bins.append(dict(condition=name,lower=RANGE[name][0],upper=RANGE[name][1],bin_index=int(np.flatnonzero(r['bin_valid'])[j]),frequency_center=float(centers[j]),bin_count=int(band[name].sum())))
        for name in IDS:
            t=time.perf_counter(); scores,chosen=calc_band(sp,ng,ok,fixed,band[name],mode=='band_specific_references'); result[name]=scores.reshape(r['scores'].shape); selection[name]=chosen.reshape(r['scores'].shape)
            timings.append(dict(image=i,group=group,condition=name,mode=mode,band_score_and_selection_seconds=time.perf_counter()-t))
            matches=(chosen==loc['selected_bits_k4'].ravel()); agreements.append(dict(image=i,group=group,condition=name,mode=mode,common_valid=nvalid,exact_selection_agreement=float(np.mean(matches[valid.ravel()]))))
            if name=='A':
                base_score=loc['score_k4']; delta=np.abs(result[name]-base_score); err=float(np.nanmax(delta)); same=bool(np.allclose(result[name],base_score,rtol=1e-6,atol=1e-6,equal_nan=True)); checks.append(dict(image=i,mode=mode,max_abs_difference=err,equal=same));
                if not same: raise AssertionError(f'image{i}: all-band K4 mismatch {err}')
                if mode=='band_specific_references' and not np.all(matches[valid.ravel()]): raise AssertionError('All-band selection mismatch')
        normal,entries=prev.classification(labels,r); data[i]=dict(group=group,status='ok',valid=nvalid,r=r,img=img,labels=labels,normal=normal,entries=entries,scores=result,selected=selection,centers=centers,bands=band,idx=idx)
        np.savez_compressed(folder/'scores_and_selections.npz',x_centers=r['xs'],y_centers=r['ys'],candidate_count=loc['candidate_count'],**{f'score_{name}':result[name] for name in IDS},**{f'selection_{name}':selection[name] for name in IDS})
        print(mode,'image',i,'seconds',round(time.perf_counter()-start,2),flush=True)
    save(summary/'band_definitions.csv',bins); save(summary/'processing_times.csv',timings); save(summary/'reference_agreement.csv',agreements); save(summary/'consistency_checks.csv',checks)
    evaluate(root,data,mode); return data

def evaluate(root,data,mode):
    summary=root/'summary'; metric=[]; images=[]; loio=[]; results=[]
    for i,d in data.items():
        if d['status']!='ok':
            for band in IDS: images.append(dict(image=i,group=d['group'],condition=band,status=d['status'],valid_locations=d['valid'],normal_count=0,normal_median=None,normal_std=None,normal_p99=None))
            continue
        for band in IDS:
            scores=d['scores'][band]; normals=scores[d['normal']&np.isfinite(scores)]; ns=prev.summary_stats(normals); images.append(dict(image=i,group=d['group'],condition=band,status='ok',valid_locations=d['valid'],normal_count=ns['count'],normal_median=ns['median'],normal_std=ns['std'],normal_p99=ns['p99']))
            for e in [e for e in d['entries'] if e['label']=='局所変形']:
                dv=scores[e['center']&np.isfinite(scores)]; ds=prev.defect_stats(dv,ns); reason='normal_count_below_100' if ns['count']<prev.MIN_NORMAL else 'normal_std_too_small' if ns['std'] is None or ns['std']<prev.MIN_STD else 'no_defect_centers' if not len(dv) else ''
                metric.append(dict(image=i,group=d['group'],instance=e['index'],condition=band,defect_count=ds['count'],defect_median=ds['median'],defect_max=ds['max'],defect_p95=ds['p95'],normal_count=ns['count'],normal_median=ns['median'],normal_std=ns['std'],normal_p99=ns['p99'],S=ds['S'],S_invalid_reason=reason))
    for group in ('grid','no_grid'):
        members={i:d for i,d in data.items() if d['group']==group and d['status']=='ok'}
        for band in IDS:
            for i,d in members.items():
                train=[v['scores'][band][v['normal']&np.isfinite(v['scores'][band])] for j,v in members.items() if j!=i]; threshold=prev.weighted_threshold(train); score=d['scores'][band]; nv=score[d['normal']&np.isfinite(score)]; entries=[e for e in d['entries'] if e['label']=='局所変形']
                for e in entries:
                    dv=score[e['center']&np.isfinite(score)]; evaluated=threshold is not None and len(dv)>0
                    loio.append(dict(image=i,group=group,instance=e['index'],condition=band,threshold=threshold,normal_exceed_rate=float(np.mean(nv>threshold)) if len(nv) and threshold is not None else None,defect_count=len(dv),defect_exceed_rate=float(np.mean(dv>threshold)) if evaluated else None,defect_max_minus_threshold=float(dv.max()-threshold) if evaluated else None,detected=bool(np.any(dv>threshold)) if evaluated else None))
                if not entries: loio.append(dict(image=i,group=group,instance=None,condition=band,threshold=threshold,normal_exceed_rate=float(np.mean(nv>threshold)) if len(nv) and threshold is not None else None,defect_count=0,defect_exceed_rate=None,defect_max_minus_threshold=None,detected=None))
    for group in ('grid','no_grid'):
        for band in IDS:
            rows=[z for z in loio if z['group']==group and z['condition']==band]; defects=[z for z in rows if z['detected'] is not None]; nv={z['image']:z['normal_exceed_rate'] for z in rows if z['normal_exceed_rate'] is not None}; ss=[z['S'] for z in metric if z['group']==group and z['condition']==band and z['S'] is not None]
            results.append(dict(group=group,condition=band,detected=sum(z['detected'] for z in defects),evaluable=len(defects),missed=sum(not z['detected'] for z in defects),normal_exceed_image_mean=float(np.mean(list(nv.values()))) if nv else None,normal_exceed_image_min=min(nv.values()) if nv else None,normal_exceed_image_max=max(nv.values()) if nv else None,S_median=float(np.median(ss)) if ss else None))
    save(summary/'band_metrics.csv',results); save(summary/'detection_by_band.csv',results); save(summary/'defect_instance_metrics.csv',metric); save(summary/'loio_metrics.csv',loio); save(summary/'image_metrics.csv',images)
    prev.safe_csv(summary/'selected_references.csv',[dict(image=i,condition=band,path=f'../images/image{i}/scores_and_selections.npz',encoding='selection_A..F: bit 0-7 = NW,N,NE,W,E,SW,S,SE; coordinate = center + 128*direction') for i,d in data.items() if d['status']=='ok' for band in IDS])
    # A separate evaluation with the original saved K4 metrics confirms LOIO equality.
    import csv
    with (KC/'summary'/'loio_metrics.csv').open(encoding='utf-8-sig') as f: prior=list(csv.DictReader(f))
    for z in loio:
        if z['condition']!='A': continue
        q=[x for x in prior if int(x['image'])==z['image'] and x['K']=='4' and x['domain']=='all' and x['instance']==('' if z['instance'] is None else str(z['instance']))]
        if q and z['threshold'] is not None: assert abs(float(q[0]['threshold'])-z['threshold'])<1e-5
    plot_results(summary,results,loio,metric)
    detail(root,data,metric,loio)
    report(summary,results,loio,metric,mode)

def plot_results(summary,results,loio,metric):
    figs=summary/'figures'; q=[z for z in results if z['group']=='grid']; fig,ax=plt.subplots(figsize=(8,4)); ax.bar(IDS,[z['detected'] for z in q]); ax.set(ylabel='LOIO detected / 15',xlabel='Frequency condition'); ax2=ax.twinx(); ax2.plot(IDS,[z['normal_exceed_image_mean']*100 for z in q],color='red',marker='o'); ax2.set_ylabel('Mean normal exceedance (%)'); fig.savefig(figs/'detection_by_band.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    for i in sorted({z['image'] for z in loio if z['group']=='grid'}):
        v=[next(z for z in loio if z['group']=='grid' and z['condition']==band and z['image']==i)['normal_exceed_rate'] for band in IDS]; ax.plot(IDS,np.array(v)*100,alpha=.7,marker='.',label=f'image{i}')
    ax.set(ylabel='Normal LOIO exceedance (%)',xlabel='Frequency condition'); ax.legend(ncol=4,fontsize=7); fig.savefig(figs/'normal_exceed_by_image.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    keys=sorted({(z['image'],z['instance']) for z in metric if z['group']=='grid'})
    for kind in ('S','detected'):
        matrix=np.full((len(keys),6),np.nan)
        for y,(i,instance) in enumerate(keys):
            for x,band in enumerate(IDS):
                source=metric if kind=='S' else loio; z=next(z for z in source if z['group']=='grid' and z['image']==i and z['instance']==instance and z['condition']==band); val=z[kind]; matrix[y,x]=float(val) if val is not None else np.nan
        fig,ax=plt.subplots(figsize=(8,8)); im=ax.imshow(matrix,aspect='auto',cmap='coolwarm' if kind=='S' else 'Blues'); ax.set(xticks=range(6),xticklabels=IDS,yticks=range(len(keys)),yticklabels=[f'image{i}:{j}' for i,j in keys],xlabel='Frequency condition'); fig.colorbar(im,ax=ax,label=kind); fig.savefig(figs/f'{kind}_heatmap.png',dpi=130,bbox_inches='tight'); plt.close(fig)

def detail(root,data,metric,loio):
    for i in (5,15,17):
        d=data[i]; r=d['r']; img=d['img']; h,w=img.shape; mask=next(m for name,m in d['labels'] if name=='局所変形'); fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
        for normalized in (False,True):
            if normalized: fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
            for ax,band in zip(axes.flat,IDS):
                score=d['scores'][band]; threshold=next(z['threshold'] for z in loio if z['image']==i and z['condition']==band); mm=next(z for z in metric if z['image']==i and z['condition']==band); shown=score-threshold if normalized else score
                im=ax.imshow(img,cmap='gray'); im=ax.imshow(shown,extent=old.extent(r,16),alpha=.65,cmap='coolwarm' if normalized else 'inferno',vmin=-1 if normalized else None,vmax=1 if normalized else None); ax.contour(mask,levels=[.5],colors='cyan'); ax.set_title(f"{band}: max {mm['defect_max']:.2f}, threshold {threshold:.2f}"); fig.colorbar(im,ax=ax,shrink=.6)
            fig.suptitle('Score minus own LOIO threshold (shared +/- 1 dB)' if normalized else 'Individual display scales (not thresholds)'); fig.savefig(root/'images'/f'image{i}'/('threshold_difference.png' if normalized else 'band_maps.png'),dpi=120); plt.close(fig)
        entry=next(e for e in d['entries'] if e['label']=='局所変形'); yy,xx=np.nonzero(entry['center']&np.isfinite(d['scores']['A'])); k=np.argmax(d['scores']['A'][yy,xx]); y,x=int(yy[k]),int(xx[k]); target=r['spectra'][y,x]; index=y*len(r['xs'])+x; chosen=int(d['selected']['A'][y,x]); selected=np.unpackbits(np.array([chosen],np.uint8),bitorder='little').astype(bool); ref=np.median(r['spectra'].reshape(-1,len(target))[d['idx'][index,selected]],axis=0); freq=d['centers']; fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True,constrained_layout=True); axes[0].plot(freq,target,label='target'); axes[0].plot(freq,ref,label='A reference'); axes[1].plot(freq,target-ref,label='target - reference');
        for ax in axes:
            for lo,hi in ((.01,.05),(.05,.10),(.10,.30),(.30,.50)): ax.axvspan(lo,hi,alpha=.07,color='blue' if lo in (.01,.10) else 'orange')
            ax.legend(); ax.set_ylabel('dB')
        axes[1].set_xlabel('cycles/pixel'); fig.suptitle(f'image{i} representative A max ({r["xs"][x]}, {r["ys"][y]})'); fig.savefig(root/'images'/f'image{i}'/'representative_spectrum.png',dpi=130); plt.close(fig)

def report(summary,results,loio,metric,mode):
    lines=['# 周波数帯別の局所変形検出','','FFT窓256、stride16、参照間隔128、K=4。FFT前処理・正規化・放射スペクトルは既存結果を使用し、帯域切り出し後の再正規化なし。選択したビン数でRMSを計算。A=0.01–0.50、B=0.05–0.50、C=0.01–0.05、D=0.05–0.10、E=0.10–0.30、F=0.30–0.50 cycles/pixel、左閉右開。元の最終有効ビンはA・B・Fに含める。ビンの中心と数はband_definitions.csv。', '参照固定' if mode=='fixed_references' else '各帯域で参照スペクトル間距離を計算して選択。Aは固定参照と同一。', '帯域別・画像を除外した画像均等正常99%閾値、格子あり/なしは別集計。正常は全ラベルと非重複の窓。傷中心のいずれかが閾値超ならインスタンスを検出。画像内の重複する窓は独立ではない。image18は候補3個以上がなく未評価。', '', '## LOIO結果']
    for group in ('grid','no_grid'):
        lines.append(f'### {group}')
        for z in results:
            if z['group']==group: lines.append(f"- {z['condition']}: 検出 {z['detected']}/{z['evaluable']}、正常超過率画像平均 {z['normal_exceed_image_mean']:.2%}（画像範囲 {z['normal_exceed_image_min']:.2%}〜{z['normal_exceed_image_max']:.2%}）、傷S中央値 {z['S_median']:.3f}。")
    lines+=['','## image5・15・17']
    for i in (5,15,17):
        for band in IDS:
            z=next(z for z in metric if z['image']==i and z['condition']==band); l=next(z for z in loio if z['image']==i and z['condition']==band); lines.append(f"- image{i} {band}: 傷中央値 {z['defect_median']:.3f}、最大 {z['defect_max']:.3f}、正常中央値 {z['normal_median']:.3f}、正常標準偏差 {z['normal_std']:.3f}、S {z['S']:.3f}、LOIO閾値 {l['threshold']:.3f}、検出 {l['detected']}。")
    lines+=['','## 制約','帯域を小さくするとRMSの分母のビン数も変わるため、スコアの大小は全帯域への二乗寄与率とは異なる。帯域・方法をこの画像群の結果で選ぶ場合、そのLOIOは選択を含めた独立評価ではない。代表点スペクトルは全帯域の傷中心最大点で選択しており、他帯域の最大点とは限らない。窓単位正常超過率は画像単位誤検出率ではない。']
    (summary/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    settings=json.loads((PRIOR/'settings.json').read_text(encoding='utf-8')); assert (settings['window'],settings['stride'],settings['reference_spacing'],settings['frequency'],settings['bin_width'])==(256,16,128,[.01,.5],.01)
    prev.N=256; prev.STRIDE=16; out=base.unique_dir(OUT); fixed=analyze('fixed_references',out); specific=analyze('band_specific_references',out)
    comp=out/'overall_comparison'; (comp/'comparison_figures').mkdir(parents=True); import csv
    def read(p):
        with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
    a=read(out/'fixed_references'/'summary'/'band_metrics.csv'); b=read(out/'band_specific_references'/'summary'/'band_metrics.csv'); agreement=read(out/'band_specific_references'/'summary'/'reference_agreement.csv'); rows=[]
    for x in a:
        y=next(v for v in b if v['group']==x['group'] and v['condition']==x['condition']); ag=[float(v['exact_selection_agreement']) for v in agreement if v['group']==x['group'] and v['condition']==x['condition']]
        rows.append(dict(group=x['group'],condition=x['condition'],fixed_detected=x['detected'],reselected_detected=y['detected'],evaluable=x['evaluable'],fixed_normal_exceed=x['normal_exceed_image_mean'],reselected_normal_exceed=y['normal_exceed_image_mean'],fixed_S_median=x['S_median'],reselected_S_median=y['S_median'],selection_agreement_image_mean=float(np.mean(ag))))
    save(comp/'fixed_vs_reselected.csv',rows); fig,ax=plt.subplots(figsize=(8,4)); q=[z for z in rows if z['group']=='grid']; x=np.arange(6); ax.bar(x-.18,[int(z['fixed_detected']) for z in q],width=.36,label='fixed'); ax.bar(x+.18,[int(z['reselected_detected']) for z in q],width=.36,label='reselected'); ax.set(xticks=x,xticklabels=IDS,ylabel='LOIO detected / 15'); ax.legend(); fig.savefig(comp/'comparison_figures'/'detection_comparison.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    lines=['# 周波数帯と参照選択の比較','','実験①は全帯域で選んだ参照を固定し帯域のみ変更。実験②は各帯域で選び直した。どちらも帯域別にLOIO正常99%閾値を計算し、FFTや正規化は変更しない。選択一致率は地点単位の選択集合の完全一致率を画像ごとに集計。画像単位の閾値超過率ではなく正常窓単位。','','## 格子あり比較']
    for z in q: lines.append(f"- {z['condition']}: 固定 {z['fixed_detected']}/{z['evaluable']}、再選択 {z['reselected_detected']}/{z['evaluable']}、正常超過率 {float(z['fixed_normal_exceed']):.2%}→{float(z['reselected_normal_exceed']):.2%}、選択集合一致率 {z['selection_agreement_image_mean']:.1%}。")
    lines+=['','## 結論と制約','帯域による差は傷と正常の双方、画像別の誤検出率を併せて確認する。正常で小さく傷で大きい帯域が全傷に普遍的かは今回の小標本では判断できない。今回の結果から帯域を選ぶ場合、それ自体が評価画像の傷ラベルを利用した探索であり、独立した未使用画像で検証が必要。参照再選択は帯域の変更と選択の変更を同時に含むため、実験①と分けて解釈する。']
    (comp/'final_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('Output:',out,flush=True)

if __name__=='__main__': main()
