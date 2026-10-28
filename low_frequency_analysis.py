#!/usr/bin/env python3
"""Inspect low-frequency detection changes, bin ablation, spatial false positives and reference choices."""
import csv,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import label, distance_transform_edt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import evaluate_all_radial_fft as prev
import multiscale_fft_experiment as base
import stride_grid_overlap_experiment as old
import reference_selection_analysis as ra

ROOT=base.ROOT; FREQ=ROOT/'output_frequency_band_analysis'; FFT=ROOT/'output_all_images_radial_fft'; KD=ROOT/'output_k4_defect_analysis_20261026_222547'; OUT=ROOT/'output_low_frequency_analysis'; CONDITIONS=['all4','omit1','omit2','omit3','omit4']
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def write(p,rows): prev.safe_csv(p,rows)
def nr(v): return float(v) if v not in ('',None) else np.nan
def yes(v): return v=='True'
def occupancy(mask,x,y): return float(mask[y-128:y+128,x-128:x+128].mean())

def main():
    out=base.unique_dir(OUT)
    for p in ('detection_comparison','frequency_bins','anomaly_maps','reference_selection','reference_selection/selected_reference_figures','summary'): (out/p).mkdir(parents=True,exist_ok=True)
    fixed=read(FREQ/'fixed_references'/'summary'/'loio_metrics.csv'); chosen=read(FREQ/'band_specific_references'/'summary'/'loio_metrics.csv'); fm=read(FREQ/'fixed_references'/'summary'/'defect_instance_metrics.csv'); fm_reselected=read(FREQ/'band_specific_references'/'summary'/'defect_instance_metrics.csv'); all_defects=read(KD/'summary'/'all_defect_metrics.csv'); agree=read(FREQ/'band_specific_references'/'summary'/'reference_agreement.csv'); bins=read(FREQ/'fixed_references'/'summary'/'band_definitions.csv'); prior_times=read(FREQ/'band_specific_references'/'summary'/'processing_times.csv'); prev.N=256; prev.STRIDE=16
    lowbins=[z for z in bins if z['condition']=='C']; frequencies=np.array([float(z['frequency_center']) for z in lowbins]); assert len(frequencies)==4
    write(out/'frequency_bins'/'low_frequency_bins.csv',[dict(bin=i+1,frequency=f,approx_period_px=1/f,approx_period_note='bin center reciprocal; not directly observed grid period') for i,f in enumerate(frequencies)])
    candidates={}; summaries={}; changes=[]; fp=[]; comps=[]; perbin=[]; refdetail=[]; rep=[]; stats=[]; processing=[]
    for i in range(1,22):
        start=time.perf_counter(); folder=out/'anomaly_maps'/f'image{i}'; folder.mkdir(); group='no_grid' if i in prev.NO_GRID else ('small_reference' if i==18 else 'grid'); image=np.asarray(Image.open(base.DATA/f'image{i}.png').convert('L')); labels=prev.annotation(base.DATA/f'image{i}.json',image.shape)
        fpath=FREQ/'fixed_references'/'images'/f'image{i}'/'scores_and_selections.npz'; cpath=FREQ/'band_specific_references'/'images'/f'image{i}'/'scores_and_selections.npz'
        if not fpath.exists() or not labels:
            (folder/'status.txt').write_text('No valid location with at least 3 candidates or no label.\n',encoding='utf-8'); continue
        with np.load(fpath) as z: a={k:z[k].copy() for k in z.files}
        with np.load(cpath) as z: c={k:z[k].copy() for k in z.files}
        r=old.load_result(FFT/f'image{i}'/'result.npz'); normal,entries=prev.classification(labels,r); idx,valid,sp,ng=__import__('frequency_band_analysis').candidates(r); n=len(sp); ny,nx=r['scores'].shape; fixedbits=np.unpackbits(a['selection_C'].reshape(-1,1),axis=1,bitorder='little').astype(bool); cbits=np.unpackbits(c['selection_C'].reshape(-1,1),axis=1,bitorder='little').astype(bool)
        low_idx=np.flatnonzero((np.flatnonzero(r['bin_valid'])+.5)*.01<.05); assert len(low_idx)==4
        lo_ng=ng[:,:,low_idx]
        with np.errstate(all='ignore'):
            import warnings
            with warnings.catch_warnings(): warnings.simplefilter('ignore',RuntimeWarning); refs=np.nanmedian(np.where(fixedbits[:,:,None],lo_ng,np.nan),axis=1)
        diff=sp[:,low_idx]-refs; squared=diff**2; validflat=(valid.sum(1)>=3); bands={}
        for j,name in enumerate(CONDITIONS):
            select=np.ones(4,bool)
            if j: select[j-1]=False
            v=np.sqrt(np.mean(squared[:,select],axis=1)); v[~validflat]=np.nan; bands[name]=v.reshape(ny,nx)
        assert np.allclose(bands['all4'],a['score_C'],rtol=1e-6,atol=1e-6,equal_nan=True)
        candidates[i]=dict(group=group,normal=normal,entries=entries,bands=bands)
        masks=ra.make_masks(labels,image.shape); all_defmask=masks['局所変形']; distance=distance_transform_edt(~all_defmask) if all_defmask.any() else None; xx,yy=np.meshgrid(r['xs'],r['ys']); edge=np.minimum.reduce([xx-128,yy-128,image.shape[1]-xx-128,image.shape[0]-yy-128]);
        for condition,score,loio in [('A',a['score_A'],fixed),('B',a['score_C'],fixed),('C',c['score_C'],chosen)]:
            band='A' if condition=='A' else 'C'; row=next(z for z in loio if int(z['image'])==i and z['condition']==band); threshold=nr(row['threshold']); nmask=normal&np.isfinite(score); above=nmask&(score>threshold); components,count=label(above,np.ones((3,3))); far=(distance[yy,xx]>=256) if distance is not None else np.ones_like(above); edgezone=edge<128
            comp_sizes=np.bincount(components.ravel())[1:]; ys,xs=np.nonzero(above); finite_norm=int(nmask.sum()); entry=dict(image=i,group=group,condition=condition,threshold=threshold,normal_count=finite_norm,normal_above=int(above.sum()),normal_exceed_fraction=float(above.sum()/finite_norm) if finite_norm else None,image_has_normal_exceed=bool(above.any()),component_count=int(count),largest_component_grid_cells=int(comp_sizes.max()) if len(comp_sizes) else 0,normal_exceed_far_256px=int(np.sum(above&far)),normal_exceed_edge_128px=int(np.sum(above&edgezone)))
            fp.append(entry)
            for component,size in enumerate(comp_sizes,1):
                ys,xs=np.nonzero(components==component); comps.append(dict(image=i,group=group,condition=condition,component=component,grid_locations=int(size),grid_cell_footprint_px2=int(size*256),min_x=int(r['xs'][xs].min()),max_x=int(r['xs'][xs].max()),min_y=int(r['ys'][ys].min()),max_y=int(r['ys'][ys].max())))
        # Whole-image A/B visualization: raw shared scale and threshold-relative significance.
        fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True); axes[0,0].imshow(image,cmap='gray',vmin=0,vmax=255); axes[0,0].set_title('Original / annotations')
        for nm,m in labels:
            if nm=='局所変形' and m.any(): axes[0,0].contour(m,levels=[.5],colors='cyan',linewidths=.8)
        rates=[]
        for col,band in enumerate(('A','C'),1):
            score=a['score_'+band]; threshold=nr(next(z['threshold'] for z in fixed if int(z['image'])==i and z['condition']==band)); rates.append((score,threshold));
        pool=np.concatenate([v[np.isfinite(v)] for v,t in rates]); low,high=np.percentile(pool,[1,99])
        for col,(score,threshold) in enumerate(rates,1):
            im=axes[0,col].imshow(score,extent=old.extent(r,16),cmap='inferno',vmin=low,vmax=high); axes[0,col].set_title(('All-band' if col==1 else 'Low-band')+' shared raw scale'); fig.colorbar(im,ax=axes[0,col],shrink=.7)
            axes[1,col].imshow(score>threshold,extent=old.extent(r,16),cmap='gray',vmin=0,vmax=1); axes[1,col].set_title(f'Above LOIO threshold {threshold:.2f}')
        axes[1,0].imshow(image,cmap='gray'); axes[1,0].set_title('Difference: score / threshold');
        # Two threshold-relative maps are independently exported to avoid conflating thresholds with raw intensity.
        fig.savefig(folder/'maps_and_exceedances.png',dpi=105); plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
        for ax,(score,threshold),name in zip(axes,rates,['all','low']):
            im=ax.imshow(score/threshold,extent=old.extent(r,16),cmap='coolwarm',vmin=0,vmax=2); ax.set_title(f'{name}: score/LOIO threshold'); fig.colorbar(im,ax=ax,shrink=.7)
        fig.savefig(folder/'threshold_relative_maps.png',dpi=110); plt.close(fig)
        for e in [e for e in entries if e['label']=='局所変形']:
            instance=e['index']; mask=labels[instance][1]; yydef,xxdef=np.nonzero(mask); m=next(z for z in all_defects if int(z['image'])==i and z['instance']==str(instance)); lA=next(z for z in fixed if int(z['image'])==i and z['instance']==str(instance) and z['condition']=='A'); lB=next(z for z in fixed if int(z['image'])==i and z['instance']==str(instance) and z['condition']=='C'); lC=next(z for z in chosen if int(z['image'])==i and z['instance']==str(instance) and z['condition']=='C');
            cls='both' if yes(lA['detected']) and yes(lB['detected']) else 'low_only' if yes(lB['detected']) else 'all_only' if yes(lA['detected']) else 'neither'; record=dict(image=i,group=group,instance=instance,area_px=int(m['area_px']),centroid_x=nr(m['centroid_x']),centroid_y=nr(m['centroid_y']),category=cls)
            for nm,l in [('all',lA),('low_fixed',lB),('low_reselected',lC)]:
                score=a['score_A'] if nm=='all' else a['score_C'] if nm=='low_fixed' else c['score_C']; mscore=score[e['center']&np.isfinite(score)]; threshold=nr(l['threshold']); met=next((v for v in (fm_reselected if nm=='low_reselected' else fm) if int(v['image'])==i and v['instance']==str(instance) and v['condition']==('A' if nm=='all' else 'C')),None); record.update({f'{nm}_detected':yes(l['detected']),f'{nm}_maximum':float(mscore.max()),f'{nm}_threshold':threshold,f'{nm}_max_minus_threshold':float(mscore.max()-threshold),f'{nm}_exceed_count':int(np.sum(mscore>threshold)),f'{nm}_center_count':len(mscore),f'{nm}_exceed_fraction':float(np.mean(mscore>threshold)),f'{nm}_S':nr(met['S']) if met else None})
            changes.append(record)
            # Each maximum is selected independently, then both scores are evaluated at those coordinates.
            positions={}
            for title,score,selectmask in [('all_max',a['score_A'],e['center']),('low_max',a['score_C'],e['center']),('normal_all_max',a['score_A'],normal),('normal_low_max',a['score_C'],normal)]:
                ys,xs=np.nonzero(selectmask&np.isfinite(score)); j=np.argmax(score[ys,xs]); positions[title]=(int(ys[j]),int(xs[j]))
            fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
            for ax,(title,(iy,ix)) in zip(axes.flat,positions.items()):
                row=iy*nx+ix; X=int(r['xs'][ix]); Y=int(r['ys'][iy]); use=fixedbits[row]; tar=sp[row]; reference=np.median(ng[row,use],axis=0); freqs=(np.flatnonzero(r['bin_valid'])+.5)*.01; dd=tar-reference
                for q in ng[row,use]: ax.plot(freqs,q,color='gray',alpha=.35,lw=.7)
                ax.plot(freqs,tar,color='red',label='target'); ax.plot(freqs,reference,color='black',ls='--',label='median'); ax.axvspan(.01,.05,color='blue',alpha=.1); ax.set(title=f'{title} ({X},{Y}): A {a["score_A"][iy,ix]:.2f}, C {a["score_C"][iy,ix]:.2f}',xlabel='cycles/pixel',ylabel='dB'); ax.legend(fontsize=6)
                for j,f in enumerate(freqs): rep.append(dict(image=i,group=group,instance=instance,position=title,x=X,y=Y,frequency=float(f),target_db=float(tar[j]),median_db=float(reference[j]),difference_db=float(dd[j]),difference_squared=float(dd[j]**2),score_all=float(a['score_A'][iy,ix]),score_low=float(a['score_C'][iy,ix]),threshold_all=nr(lA['threshold']),threshold_low=nr(lB['threshold']),reference_ids=' '.join(str(j+1) for j in np.flatnonzero(use))))
                # Save label-contamination details at low-frequency maximum; label data are never used for selection.
                if title=='low_max':
                    for j,(dx,dy) in enumerate(ra.OFF):
                        if not valid[row,j]: continue
                        xp,yp=X+dx*128,Y+dy*128; spec=ng[row,j]; other=ng[row,valid[row]]; rms=np.sqrt(np.mean((other-spec)**2,axis=1));
                        refdetail.append(dict(image=i,group=group,instance=instance,position=title,target_x=X,target_y=Y,reference_id=j+1,reference_x=xp,reference_y=yp,fixed_selected=bool(fixedbits[row,j]),low_reselected=bool(cbits[row,j]),background_fraction=occupancy(masks['background'],xp,yp),tag_fraction=occupancy(masks['tag'],xp,yp),deformation_fraction=occupancy(masks['局所変形'],xp,yp),other_damage_fraction=occupancy(masks['other'],xp,yp),mean_pair_rms_all=float(rms.mean()),target_rms_all=float(np.sqrt(np.mean((spec-tar)**2)))))
            fig.suptitle(f'image{i} defect {instance}: target, four references, median'); fig.savefig(folder/f'defect_{instance}_representatives.png',dpi=120); plt.close(fig)
            # Four 1-bin absolute differences across all normal / defect-center points.
            norm_idx=np.flatnonzero(normal.ravel()&validflat); defect_idx=np.flatnonzero(e['center'].ravel()&validflat)
            for j,f in enumerate(frequencies):
                nvals=np.abs(diff[norm_idx,j]); dvals=np.abs(diff[defect_idx,j]); ns=prev.summary_stats(nvals); ds=prev.defect_stats(dvals,ns); perbin.append(dict(image=i,group=group,instance=instance,frequency=f,period_center_px=1/f,normal_median=ns['median'],normal_std=ns['std'],defect_median=ds['median'],defect_p95=ds['p95'],S=ds['S']))
        processing.append(dict(image=i,group=group,seconds=time.perf_counter()-start)); print('image',i,'completed',flush=True)
    write(out/'detection_comparison'/'per_defect_detection.csv',changes); write(out/'detection_comparison'/'detection_changes.csv',changes); write(out/'summary'/'defect_metrics.csv',changes); write(out/'summary'/'image_false_positive_metrics.csv',fp); write(out/'summary'/'normal_exceed_components.csv',comps); write(out/'summary'/'representative_spectra.csv',rep); write(out/'frequency_bins'/'per_bin_features.csv',perbin); write(out/'reference_selection'/'reference_selection_details.csv',refdetail); write(out/'reference_selection'/'reference_agreement.csv',agree); write(out/'summary'/'processing_times.csv',processing)
    evaluate_ablation(out,candidates,changes,fp,agree,prior_times,fixed,chosen,perbin,refdetail)

def evaluate_ablation(out,data,changes,fp,agree,timing,fixed,chosen,perbin,refdetail):
    loio=[]; metrics=[]; instances=[]
    for group in ('grid','no_grid'):
        members={i:d for i,d in data.items() if d['group']==group}
        for condition in CONDITIONS:
            for i,d in members.items():
                train=[v['bands'][condition][v['normal']&np.isfinite(v['bands'][condition])] for j,v in members.items() if i!=j]; threshold=prev.weighted_threshold(train); score=d['bands'][condition]; nv=score[d['normal']&np.isfinite(score)]; ns=prev.summary_stats(nv); ent=[e for e in d['entries'] if e['label']=='局所変形']
                if not ent: loio.append(dict(image=i,group=group,instance=None,condition=condition,threshold=threshold,normal_exceed_rate=float(np.mean(nv>threshold)),detected=None,defect_exceed_rate=None,S=None)); continue
                for e in ent:
                    dv=score[e['center']&np.isfinite(score)]; ds=prev.defect_stats(dv,ns); v=dict(image=i,group=group,instance=e['index'],condition=condition,threshold=threshold,normal_exceed_rate=float(np.mean(nv>threshold)),detected=bool(np.any(dv>threshold)),defect_exceed_rate=float(np.mean(dv>threshold)),S=ds['S']); loio.append(v); instances.append(v)
            q=[z for z in loio if z['group']==group and z['condition']==condition]; defects=[z for z in q if z['detected'] is not None]; nrates={z['image']:z['normal_exceed_rate'] for z in q}; ss=[z['S'] for z in defects if z['S'] is not None]; metrics.append(dict(group=group,condition=condition,detected=sum(z['detected'] for z in defects),evaluable=len(defects),normal_exceed_image_mean=float(np.mean(list(nrates.values()))),S_median=float(np.median(ss))))
    assert sum(z['all_detected'] for z in changes if z['group']=='grid')==7
    assert sum(z['low_fixed_detected'] for z in changes if z['group']=='grid')==11
    for z in loio:
        if z['condition']=='all4' and z['instance'] is not None:
            prior=next(v for v in fixed if int(v['image'])==z['image'] and v['condition']=='C' and v['instance']==str(z['instance']))
            assert abs(z['threshold']-float(prior['threshold']))<1e-5 and z['detected']==yes(prior['detected'])
    write(out/'summary'/'loio_metrics.csv',loio); write(out/'frequency_bins'/'leave_one_bin_out_metrics.csv',metrics); write(out/'frequency_bins'/'leave_one_bin_out_instances.csv',instances)
    fig,ax=plt.subplots(figsize=(8,4)); q=[v for v in metrics if v['group']=='grid']; ax.bar(CONDITIONS,[v['detected'] for v in q]); ax.set(ylabel='LOIO detected / 15',xlabel='Low-frequency bins'); fig.savefig(out/'frequency_bins'/'bin_importance.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    grid=[v for v in changes if v['group']=='grid']; fig,ax=plt.subplots(figsize=(7,8)); mat=np.array([[int(v['all_detected']),int(v['low_fixed_detected'])] for v in grid]); ax.imshow(mat,aspect='auto',cmap='Blues',vmin=0,vmax=1); ax.set(xticks=[0,1],xticklabels=['all','low fixed'],yticks=range(len(grid)),yticklabels=[f"image{v['image']}:{v['instance']}" for v in grid]); fig.savefig(out/'detection_comparison'/'detection_heatmap.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    methods=[]
    for condition in ('A','B','C'):
        x=[v for v in fp if v['condition']==condition and v['group']=='grid']; defects=[v for v in grid if v[{'A':'all_detected','B':'low_fixed_detected','C':'low_reselected_detected'}[condition]]];
        if condition=='C':
            sm=read(FREQ/'band_specific_references'/'summary'/'band_metrics.csv'); S=nr(next(v['S_median'] for v in sm if v['group']=='grid' and v['condition']=='C'))
        else: sm=read(FREQ/'fixed_references'/'summary'/'band_metrics.csv'); S=nr(next(v['S_median'] for v in sm if v['group']=='grid' and v['condition']==('A' if condition=='A' else 'C')))
        methods.append(dict(condition=condition,detected=len(defects),evaluable=len(grid),normal_exceed_image_mean=float(np.mean([v['normal_exceed_fraction'] for v in x])),images_with_normal_exceed=sum(v['image_has_normal_exceed'] for v in x),image_count=len(x),S_median=S))
    write(out/'reference_selection'/'fixed_vs_reselected_metrics.csv',methods)
    lines=['# 低周波FFTの検出変化と参照選択','','256 px、stride16、K=4。A=全帯域で選択し全帯域評価、B=全帯域で選択し低周波0.01〜0.05のみ評価、C=低周波で参照を再選択し低周波評価。既存特徴量を再利用、各条件・各群のLOIO閾値を独立に再計算。AとBの集計は7/15・11/15を再現。帯域で再正規化しない。画像単位誤検出は「ラベル非重複の正常窓が1点以上閾値を超過した画像数」であり、完全正常画像に対する画像単位誤検出率ではない。画像18は候補不足・ラベルなしのため未評価。','','## 各傷の検出状態']
    for v in grid: lines.append(f"- image{v['image']} 傷{v['instance']}: 全帯域 {v['all_detected']}、低周波 {v['low_fixed_detected']}、分類 {v['category']}。全帯域 max−閾値 {v['all_max_minus_threshold']:+.3f} dB、低周波 {v['low_fixed_max_minus_threshold']:+.3f} dB。")
    for name in ('both','low_only','all_only','neither'): lines.append(f"- {name}: {sum(v['category']==name for v in grid)}傷。")
    lines+=['','## 低周波4ビンの除外実験']
    for v in metrics:
        if v['group']=='grid': lines.append(f"- {v['condition']}: {v['detected']}/{v['evaluable']}検出、正常超過率画像平均 {v['normal_exceed_image_mean']:.2%}、傷S中央値 {v['S_median']:.3f}。")
    lines+=['','## 参照選択と正常窓超過']
    for v in methods: lines.append(f"- {v['condition']}: {v['detected']}/{v['evaluable']}検出、正常窓超過率画像平均 {v['normal_exceed_image_mean']:.2%}、正常超過窓を含む画像 {v['images_with_normal_exceed']}/{v['image_count']}、傷S中央値 {v['S_median']:.3f}。")
    vals=[float(v['exact_selection_agreement']) for v in agree if v['condition']=='C' and v['group']=='grid']; lines.append(f'- BとCの全有効地点における参照集合の完全一致率の画像平均 {np.mean(vals):.1%}。')
    lines+=['','## image5・15・17']
    for v in changes:
        if v['image'] in (5,15,17): lines.append(f"- image{v['image']} 傷{v['instance']}: A最大 {v['all_maximum']:.3f}/閾値 {v['all_threshold']:.3f}、B最大 {v['low_fixed_maximum']:.3f}/閾値 {v['low_fixed_threshold']:.3f}。A超過 {v['all_exceed_count']}/{v['all_center_count']}、B超過 {v['low_fixed_exceed_count']}/{v['low_fixed_center_count']}。")
    lines+=['','## 解釈と限界','比較するRMSは帯域別のビン数で平均するため、全帯域と低周波の数値の大小だけで正常が安定したと断定できない。傷の最大と学習側閾値の差、正常窓分布と画像別超過率を併せて確認する。各低周波ビンの代表周期はビン中心の逆数であり、格子の実測周期や変形の物理的原因を示さない。','連結成分はストライド16 pxの正常窓中心の8近傍連結で、面積換算は格子セル数×256 px²。重複するFFT窓・実際の傷の面積とは異なる。傷からの距離は中心とアノテーション画素の最短距離、端からの距離はFFT対象窓と画像端の最短距離。','全帯域最大点と低周波最大点をそれぞれ選び、両点で両スコアを比較した。代表点はスコアに基づく選択のため独立検証ではない。参照の占有率は選択の後に正解ラベルを使って計測し、参照決定には使用しない。低周波4ビンとビン除外はこの画像群を見た探索であり、最高の結果を未使用画像で確認する。']
    (out/'summary'/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('Output:',out,flush=True)

if __name__=='__main__': main()
