"""Explain the spectral mechanism behind the fixed global grid-notch experiment.

Uses the unmodified local FFT implementation in analysis_fft_normalization_128.py
and the fixed per-image top-2 hard-radius-2 notch from the prior analysis.
"""
from __future__ import annotations
import csv, json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

import analysis_fft_normalization_128 as det
import fft_grid_notch_removal_analysis as notch

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'data20260903'/'data20260903'
PRIOR=ROOT/'analysis_full_image_fft_peaks'; OUT=ROOT/'analysis_fft_notch_spectral_mechanism'
IMAGES=list(range(2,18)); BANDS=((.02,.05),(.05,.10),(.10,.20),(.20,.50)); EPS=1e-12

def write_csv(path, rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)

def refs(spec, step):
    return np.median(np.stack([spec[step+dy*step:len(spec)-step+dy*step,step+dx*step:spec.shape[1]-step+dx*step]
        for dy,dx in det.OFFSETS]),axis=0)

def stats(a):
    a=np.asarray(a); return {'mean':float(a.mean()),'median':float(np.median(a)),'std':float(a.std()),'p95':float(np.quantile(a,.95)),'p99':float(np.quantile(a,.99))}

def band_fields(delta, target, reference, freq, prefix):
    total=float(np.sum(delta**2)); d={}
    for lo,hi in BANDS:
        m=(freq>=lo)&(freq<(hi if hi<.5 else hi+1e-9)); tag=f'{lo:.2f}_{hi:.2f}'
        d[f'{prefix}_target_mean_{tag}']=float(target[m].mean()); d[f'{prefix}_reference_mean_{tag}']=float(reference[m].mean())
        d[f'{prefix}_rms_{tag}']=float(np.sqrt(np.mean(delta[m]**2)));d[f'{prefix}_contribution_{tag}']=float(np.sum(delta[m]**2)/total) if total else math.nan
    return d

def rep_plot(cases, freq, out):
    fig,ax=plt.subplots(len(cases),3,figsize=(16,4.2*len(cases)),squeeze=False,constrained_layout=True)
    for row,c in zip(ax,cases):
        title=f"image{c['image']} #{c['defect_number']} {c['defect_type']} ({c['case']})"
        for cond,color in [('before','tab:gray'),('after','tab:blue')]:
            row[0].plot(freq,c[f'{cond}_raw_target'],color=color,lw=1.8,label=f'{cond}: defect')
            row[0].plot(freq,c[f'{cond}_raw_reference'],color=color,lw=1,ls='--',label=f'{cond}: ref median')
            row[1].plot(freq,c[f'{cond}_norm_target'],color=color,lw=1.8,label=f'{cond}: defect')
            row[1].plot(freq,c[f'{cond}_norm_reference'],color=color,lw=1,ls='--',label=f'{cond}: ref median')
            row[2].plot(freq,c[f'{cond}_norm_delta'],color=color,lw=1.8,label=cond)
        for a,t in zip(row,('Raw radial spectra (dB)','Normalised radial spectra (dB)','Normalised defect − ref (dB)')):
            a.set(title=t,xlabel='Spatial frequency (cycles/pixel)',ylabel='Power / difference (dB)');a.axhline(0,color='black',lw=.6);a.legend(fontsize=7)
            for lo,hi in BANDS:a.axvspan(lo,hi,color='orange',alpha=.06)
        row[0].text(.02,.96,title,transform=row[0].transAxes,va='top',fontsize=9)
    fig.savefig(out,dpi=150);plt.close(fig)

def distributions_plot(all_scores,out):
    fig,ax=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for name,color in [('before','tab:gray'),('after','tab:blue')]:
        n=np.concatenate(all_scores[name]['normal']);d=np.concatenate(all_scores[name]['defect'])
        bins=np.linspace(0,max(np.quantile(n,.997),np.quantile(d,.997)),65)
        ax[0].hist(n,bins=bins,density=True,histtype='step',lw=2,color=color,label=f'{name} normal')
        ax[0].hist(d,bins=bins,density=True,histtype='step',lw=2,ls='--',color=color,label=f'{name} defect')
        ax[1].hist(n,bins=bins,density=True,histtype='step',lw=2,color=color,label=f'{name} normal')
    ax[0].set(title='Strict-8 local FFT scores',xlabel='RMS spectral difference (dB)',ylabel='density');ax[1].set(title='Normal score distribution only',xlabel='RMS spectral difference (dB)',ylabel='density')
    for a in ax:a.legend(fontsize=8)
    fig.savefig(out,dpi=150);plt.close(fig)

def band_plot(rows,normal_band,out):
    types=sorted(set(r['defect_type'] for r in rows)); labels=[f'{a:.2f}-{b:.2f}' for a,b in BANDS]
    fig,ax=plt.subplots(1,2,figsize=(15,5),constrained_layout=True)
    x=np.arange(len(labels));w=.35
    for j,cond in enumerate(('before','after')):
        vals=[];nvals=[]
        for lo,hi in BANDS:
            key=f'{cond}_rms_{lo:.2f}_{hi:.2f}'; vals.append(float(np.median([r[key] for r in rows])))
            nvals.append(normal_band[cond][f'{lo:.2f}_{hi:.2f}']['median'])
        ax[0].bar(x+(j-.5)*w,vals,w,label=f'defect vs ref: {cond}')
        ax[1].bar(x+(j-.5)*w,nvals,w,label=f'normal vs ref: {cond}')
    ax[0].set(title='Defect-reference RMS by band (all instances)',xticks=x,xticklabels=labels,ylabel='RMS difference (dB)',xlabel='frequency band (cycles/pixel)')
    ax[1].set(title='Normal-reference RMS by band (same target grid)',xticks=x,xticklabels=labels,ylabel='RMS difference (dB)',xlabel='frequency band (cycles/pixel)')
    for a in ax:a.legend(fontsize=8)
    fig.savefig(out,dpi=150);plt.close(fig)

def relation_plot(rows,out):
    x=np.array([r['after_rms_0.02_0.05']-r['before_rms_0.02_0.05'] for r in rows]);y=np.array([r['after_score']-r['before_score'] for r in rows])
    fig,ax=plt.subplots(figsize=(7,6),constrained_layout=True)
    for typ in sorted(set(r['defect_type'] for r in rows)):
        q=[r['defect_type']==typ for r in rows];ax.scatter(x[q],y[q],label=typ,s=45)
    corr=float(np.corrcoef(x,y)[0,1]) if len(x)>1 else math.nan
    ax.axhline(0,color='black',lw=.7);ax.axvline(0,color='black',lw=.7);ax.set(title=f'Instance score vs 0.02–0.05 RMS change (r={corr:.3f})',xlabel='band RMS change after − before (dB)',ylabel='instance score change (dB)');ax.legend(fontsize=8)
    fig.savefig(out,dpi=150);plt.close(fig)
    return corr

def representative_roi_plot(reps,top,fam,out):
    fig,ax=plt.subplots(len(reps),3,figsize=(12,4*len(reps)),squeeze=False,constrained_layout=True)
    for row,r in zip(ax,reps):
        g=notch.load_gray(r['image']);peaks=notch.peak_sets(r['image'],top,fam)['top2'];f,*_=notch.apply_notch(g,peaks,'hard',2)
        x,y=r['target_x'],r['target_y'];half=96;x0=max(0,x-half);x1=min(g.shape[1],x+half);y0=max(0,y-half);y1=min(g.shape[0],y+half);sl=np.s_[y0:y1,x0:x1]
        lim=np.percentile(g,[1,99]);dl=np.percentile(abs(f-g),99.5)
        for a,img,title,cmap in zip(row,(g,f,f-g),('before','after: top2 hard r2','after − before'),('gray','gray','coolwarm')):
            if cmap=='gray':a.imshow(img[sl],cmap=cmap,vmin=lim[0],vmax=lim[1])
            else:a.imshow(img[sl],cmap=cmap,vmin=-dl,vmax=dl)
            a.axhline(y-y0,color='lime',lw=.7);a.axvline(x-x0,color='lime',lw=.7);a.set_title(f"{r['case']}: {title}");a.set_xticks([]);a.set_yticks([])
    fig.savefig(out,dpi=160);plt.close(fig)

def grid_metric_image15(top, family):
    g=notch.load_gray(15);sets=notch.peak_sets(15,top,family);rows=sets['top2'];d,e,grid,_=notch.mask_from_annotations(15,g.shape)
    after,shifted,ash,_=notch.apply_notch(g,rows,'hard',2);comp=notch.retained_grid(shifted,rows,2)
    # Component RMS directly measures the removed periodic image, unlike the old 20px highpass statistic.
    metrics=[]
    for name,m in [('grid_points',grid),('valid_nonannotation',~(d|e))]:
        for a,label in [(g,'original'),(after,'notched'),(comp,'removed_grid_component')]:
            metrics.append({'region':name,'image_component':label,'mean':float(a[m].mean()),'std':float(a[m].std()),'rms':float(np.sqrt(np.mean(a[m]**2)))})
    # Individual two-axis components document both grid directions.
    for i,r in enumerate(rows,1):
        one=notch.retained_grid(shifted,[r],2);metrics.append({'region':'grid_points','image_component':f'axis_{i}_component','mean':float(one[grid].mean()),'std':float(one[grid].std()),'rms':float(np.sqrt(np.mean(one[grid]**2)))})
    selected_mask=1-notch.notch(g.shape,rows,'hard',2)
    pbefore=np.sum(abs(shifted)**2*selected_mask)
    pafter=np.sum(abs(ash)**2*selected_mask)
    metrics.append({'region':'selected_fft_peaks','image_component':'power','mean':float(pbefore),'std':float(pafter),'rms':100*(1-pafter/max(pbefore,EPS))})
    lim=np.percentile(g,[1,99]);dl=np.percentile(abs(comp),99.5);fig,ax=plt.subplots(1,4,figsize=(16,4),constrained_layout=True)
    for a,img,title,cmap in zip(ax,(g,comp,after,g-after),('original','inverse FFT of removed peak neighbourhoods','notched image','difference = removed component'),('gray','coolwarm','gray','coolwarm')):
        if cmap=='gray':a.imshow(img[::3,::3],cmap=cmap,vmin=lim[0],vmax=lim[1])
        else:a.imshow(img[::3,::3],cmap=cmap,vmin=-dl,vmax=dl)
        a.set_title(title);a.set_xticks([]);a.set_yticks([])
    fig.savefig(OUT/'image15_grid_component_spatial_validation.png',dpi=150);plt.close(fig)
    return metrics

def main():
    OUT.mkdir(exist_ok=True);top=notch.read_csv(PRIOR/'fft_peak_pairs_top20.csv');fam=notch.read_csv(PRIOR/'grid_family_candidates.csv')
    setup=det.radial_setup();freq=setup['centers'];step=det.SPACING//det.STRIDE
    all_scores={k:{'normal':[],'defect':[]} for k in ('before','after')}; instances=[]; band_rows=[]; normal_band={k:{f'{a:.2f}_{b:.2f}':[] for a,b in BANDS} for k in ('before','after')}; case_data=[]
    for n in IMAGES:
        print(f'image{n}',flush=True);gray=notch.load_gray(n);defs,exc,_,defects=notch.mask_from_annotations(n,gray.shape); peaks=notch.peak_sets(n,top,fam)['top2'];filtered,*_=notch.apply_notch(gray,peaks,'hard',2)
        # same window-centre grid, same annotation masks, before and after.
        rr={}
        for cond,img in [('before',gray),('after',filtered)]: rr[cond]=det.spectra_for_image(img,setup,workers=4,batch=64)
        sx=rr['before']['xs'][step:-step];sy=rr['before']['ys'][step:-step];union=defs
        target=det.rect_any(det.integral(union),sx,sy,16);excluded=det.rect_any(det.integral(exc),sx,sy,16);normal=(~target)&(~excluded)
        spec_ref={}
        for cond in ('before','after'):
            raw,norm=rr[cond]['raw'],rr[cond]['norm'];rawref=refs(raw,step);normref=refs(norm,step)
            rawt=raw[step:-step,step:-step];normt=norm[step:-step,step:-step];rawdiff=rawt-rawref;normdiff=normt-normref
            score=np.sqrt(np.mean(normdiff**2,axis=-1));spec_ref[cond]=(rawt,rawref,normt,normref,rawdiff,normdiff,score)
            all_scores[cond]['normal'].append(score[normal]);all_scores[cond]['defect'].append(score[target&~excluded])
            for lo,hi in BANDS:
                m=(freq>=lo)&(freq<(hi if hi<.5 else hi+1e-9));normal_band[cond][f'{lo:.2f}_{hi:.2f}'].extend(np.sqrt(np.mean(normdiff[...,m]**2,axis=-1))[normal].tolist())
        # Fixed representative coordinate: highest BEFORE normalised score in each defect's unchanged candidate set.
        for j,a in enumerate(defects,1):
            cand=det.rect_any(det.integral(a['mask']),sx,sy,16)&~excluded;yy,xx=np.nonzero(cand)
            if not len(yy):continue
            q=int(np.argmax(spec_ref['before'][6][yy,xx]));iy,ix=int(yy[q]),int(xx[q]);r={'image':n,'defect_number':j,'defect_type':a['label'],'target_x':int(sx[ix]),'target_y':int(sy[iy]),'candidate_windows':int(len(yy))}
            for cond in ('before','after'):
                rt,rf,nt,nf,rd,nd,sc=spec_ref[cond];r[f'{cond}_score']=float(sc[iy,ix]);r.update(band_fields(nd[iy,ix],nt[iy,ix],nf[iy,ix],freq,cond))
                case_data.append({'key':(n,j), 'condition':cond,'raw_target':rt[iy,ix],'raw_reference':rf[iy,ix],'norm_target':nt[iy,ix],'norm_reference':nf[iy,ix],'norm_delta':nd[iy,ix]})
            instances.append(r)
            for lo,hi in BANDS: band_rows.append({'image':n,'defect_number':j,'defect_type':a['label'],'band_low':lo,'band_high':hi,'before_rms':r[f'before_rms_{lo:.2f}_{hi:.2f}'],'after_rms':r[f'after_rms_{lo:.2f}_{hi:.2f}'],'rms_change':r[f'after_rms_{lo:.2f}_{hi:.2f}']-r[f'before_rms_{lo:.2f}_{hi:.2f}'],'before_contribution':r[f'before_contribution_{lo:.2f}_{hi:.2f}'],'after_contribution':r[f'after_contribution_{lo:.2f}_{hi:.2f}']})
    # Global fixed normal grids and per-instance thresholds.
    normal_stats=[];perf=[]
    for cond in ('before','after'):
        n=np.concatenate(all_scores[cond]['normal']);d=np.concatenate(all_scores[cond]['defect']);normal_stats.append({'condition':cond,'evaluation_unit':'normal_window',**stats(n)})
        x={'condition':cond,'evaluation_unit':'window','normal_windows':len(n),'defect_windows':len(d),'roc_auc':float(roc_auc_score(np.r_[np.zeros(len(n)),np.ones(len(d))],np.r_[n,d]) )}
        for fpr in (.01,.05):
            th=float(np.quantile(n,1-fpr,method='higher'));x[f'threshold_fpr_{int(fpr*100)}pct']=th;x[f'actual_fpr_{int(fpr*100)}pct']=float(np.mean(n>th));x[f'window_tpr_fpr_{int(fpr*100)}pct']=float(np.mean(d>th))
            for r in instances:r[f'{cond}_hit_fpr_{int(fpr*100)}pct']=bool(r[f'{cond}_score']>th)
        perf.append(x)
    normal_band_stats={c:{k:stats(v) for k,v in d.items()} for c,d in normal_band.items()}
    nbandrows=[{'condition':c,'band':k,**v} for c,d in normal_band_stats.items() for k,v in d.items()]
    typ=[]
    for cond in ('before','after'):
        for t in sorted(set(r['defect_type'] for r in instances)):
            q=[r for r in instances if r['defect_type']==t];typ.append({'condition':cond,'defect_type':t,'instances':len(q),'median_score':float(np.median([r[f'{cond}_score'] for r in q])),'detection_rate_fpr_1pct':float(np.mean([r[f'{cond}_hit_fpr_1pct'] for r in q])),'detection_rate_fpr_5pct':float(np.mean([r[f'{cond}_hit_fpr_5pct'] for r in q]))})
    # Representative cases: largest improvement, near unchanged, largest worsening; all retain same centre.
    delta=np.array([r['after_score']-r['before_score'] for r in instances]);choice=[int(np.argmax(delta)),int(np.argmin(abs(delta))),int(np.argmin(delta))]; reps=[]
    for label,i in zip(('largest score increase','near unchanged','largest score decrease'),choice):
        r=instances[i];d={'case':label,**r}
        for c in case_data:
            if c['key']==(r['image'],r['defect_number']):
                for k,v in c.items():
                    if k not in ('key','condition'):d[f'{c["condition"]}_{k}']=v
        reps.append(d)
    write_csv(OUT/'defect_fixed_window_spectral_metrics.csv',instances);write_csv(OUT/'defect_band_change_long.csv',band_rows);write_csv(OUT/'normal_window_score_statistics.csv',normal_stats);write_csv(OUT/'normal_window_band_rms_statistics.csv',nbandrows);write_csv(OUT/'local_fft_performance_reproduced.csv',perf);write_csv(OUT/'defect_type_performance.csv',typ);write_csv(OUT/'representative_cases.csv',[{k:v for k,v in r.items() if not isinstance(v,np.ndarray)} for r in reps]);write_csv(OUT/'image15_grid_component_metrics.csv',grid_metric_image15(top,fam))
    rep_plot(reps,freq,OUT/'representative_fixed_window_spectra.png');representative_roi_plot(reps,top,fam,OUT/'representative_fixed_window_rois.png');distributions_plot(all_scores,OUT/'normal_and_defect_score_distributions.png');band_plot(instances,normal_band_stats,OUT/'band_rms_defect_vs_normal.png');corr=relation_plot(instances,OUT/'score_change_vs_lowband_change.png')
    (OUT/'settings.json').write_text(json.dumps({'notch':'same prior top2 conjugate pairs; hard radius 2','local_fft':'imported unchanged from analysis_fft_normalization_128.py; 128px Hann mean removal, stride16, strict8 reference, total-power normalisation, radial bins .02-.50','fixed_windows':'all before/after defect and normal comparisons use identical strict-8 target centres; representative defect coordinate is baseline normalised-score maximum and is not reselected after notch'},ensure_ascii=False,indent=2),encoding='utf-8')
    report(perf,normal_stats,normal_band_stats,typ,instances,corr)

def report(perf,nstat,nband,typ,inst,corr):
    p={r['condition']:r for r in perf};s={r['condition']:r for r in nstat};b=p['before'];a=p['after'];low0=nband['before']['0.02_0.05'];low1=nband['after']['0.02_0.05']
    lines=['# 格子ノッチで局所FFT検出が変化した理由', '', '## 結論', '',
      f'- 同一の窓中心・同一の8参照窓で比較した結果、ROC-AUCは **{b["roc_auc"]:.4f} → {a["roc_auc"]:.4f}**、FPR 1% TPRは **{b["window_tpr_fpr_1pct"]:.4f} → {a["window_tpr_fpr_1pct"]:.4f}**、FPR 5% TPRは **{b["window_tpr_fpr_5pct"]:.4f} → {a["window_tpr_fpr_5pct"]:.4f}**で再現した。',
      f'- 正常windowスコアの標準偏差は **{s["before"]["std"]:.4f} → {s["after"]["std"]:.4f}**、99%閾値は **{s["before"]["p99"]:.4f} → {s["after"]["p99"]:.4f}**。正常同士のばらつきが小さくなったかは、この二つの値と帯域別CSVで判断する。',
      f'- 0.02–0.05 cycles/pixelの正常–参照RMS中央値は **{low0["median"]:.4f} → {low1["median"]:.4f}**。傷個体の同帯域RMS変化と総スコア変化の相関は **r={corr:.3f}**であり、低周波帯だけで全ての性能変化を説明できるかを過大解釈してはならない。',
      '- 仮説A（正常ばらつき低下）と仮説B（傷差増大）は、`normal_window_*` と `defect_*` を並べて評価した。低FPR改善は、正常スコアの尾部・傷スコアの双方の変化で起こり得るため、片方だけを根拠にしない。',
      '- 以前の `grid_modulation_reduction_pct` は20px高域残差の標準偏差で、約180pxの格子以外の粒状成分にも強く反応した。本解析では選択ピーク近傍を逆FFTした空間成分そのものを `image15_grid_component_spatial_validation.png` とCSVで示す。', '',
      '## 傷種別', '']
    for r in typ: lines.append(f'- {r["condition"]} / {r["defect_type"]}: n={r["instances"]}, median={r["median_score"]:.3f}, TPR@1%={r["detection_rate_fpr_1pct"]:.3f}, TPR@5%={r["detection_rate_fpr_5pct"]:.3f}')
    lines += ['', '## 出力', '', '- `representative_fixed_window_spectra.png`: 固定傷窓・固定8参照窓のraw/normalised spectraと差。', '- `representative_fixed_window_rois.png`: 同じ固定窓中心の傷画像（前後・差）。', '- `band_rms_defect_vs_normal.png`: 傷–参照と正常–参照を分離した帯域比較。', '- `normal_and_defect_score_distributions.png`: 正常・傷score分布。', '- `score_change_vs_lowband_change.png`: 個体ごとの低帯域変化とscore変化。', '- `image15_grid_component_spatial_validation.png`: 除去したピーク近傍の逆FFT成分（旧指標の代替）。', '', '## 限界', '', '固定ノッチ条件は前回のimage15視覚比較に基づく探索的条件であり、完全な独立テストではない。前処理採用時は画像単位leave-one-image-outでピーク集合と半径を固定して追試する。']
    (OUT/'report_ja.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__': main()
