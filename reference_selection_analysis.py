#!/usr/bin/env python3
"""Evaluate fixed reference selection rules using saved radial FFT features."""
import itertools, json, time, warnings
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import evaluate_all_radial_fft as prev
import multiscale_fft_experiment as base
import stride_grid_overlap_experiment as old

ROOT=base.ROOT; PRIOR=ROOT/'output_all_images_radial_fft'; OUT=ROOT/'output_reference_selection_analysis'
OFF=[(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)] # (dx,dy), stable order
METHODS=('A_all','B_lower5','C_low_background','D_consistent5')

def overlap(mask,r):
    ii=np.pad(mask.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1); y=r['ys'].astype(int)-128; x=r['xs'].astype(int)-128
    return (ii[np.ix_(y+256,x+256)]-ii[np.ix_(y,x+256)]-ii[np.ix_(y+256,x)]+ii[np.ix_(y,x)])/65536

def make_masks(labels,shape):
    result={k:np.zeros(shape,bool) for k in ('background','tag','局所変形','other')}
    for name,mask in labels: result[name if name in result else 'other'] |= mask
    result['unlabeled']=~np.logical_or.reduce(list(result.values()))
    return result

def arrays(r,occ):
    ny,nx=r['scores'].shape; n=ny*nx; flat=r['spectra'].reshape(n,-1); yy,xx=np.indices((ny,nx)); inds=np.full((n,8),-1,int)
    for j,(dx,dy) in enumerate(OFF):
        y=yy+dy*8; x=xx+dx*8; ok=(y>=0)&(y<ny)&(x>=0)&(x<nx); inds[:,j]=np.where(ok,y*nx+x,-1).ravel()
    valid=inds>=0; ng=flat[np.maximum(inds,0)]; bg=occ['background'].ravel()[np.maximum(inds,0)]; bg=np.where(valid,bg,np.inf)
    return inds,valid,ng,bg

def compute(r,occ):
    start=time.perf_counter(); inds,valid,ng,bg=arrays(r,occ); n=len(inds); counts=valid.sum(1); target=r['spectra'].reshape(n,-1)
    t=time.perf_counter();
    # Mean pairwise RMS to all available references defines medoid; select it and
    # four references nearest to it. Equal distances resolve by OFF order.
    delta=ng[:,:,None,:]-ng[:,None,:,:]; dist=np.sqrt(np.mean(delta*delta,axis=-1)); ok=valid[:,:,None]&valid[:,None,:]
    mean=np.where(valid,np.sum(np.where(ok,dist,0),axis=2)/np.maximum(counts[:,None],1),np.inf)
    medoid=np.argmin(mean,axis=1); distance=dist[np.arange(n),medoid]; distance=np.where(valid,distance,np.inf)
    order=np.argsort(distance,axis=1,kind='stable'); ranks=np.argsort(order,axis=1,kind='stable'); select_d=valid&(ranks<5)
    similarity=time.perf_counter()-t; t=time.perf_counter()
    order_bg=np.argsort(bg,axis=1,kind='stable'); ranks_bg=np.argsort(order_bg,axis=1,kind='stable')
    sel={'A_all':valid,'B_lower5':valid & np.array([dy>=0 for dx,dy in OFF]),'C_low_background':valid&(ranks_bg<5),'D_consistent5':select_d}
    scores={}; refs={}; count={}; method_times={}
    for m,selected in sel.items():
        method_start=time.perf_counter()
        count[m]=selected.sum(1); buf=np.where(selected[:,:,None],ng,np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning); ref=np.nanmedian(buf,axis=1)
        s=np.sqrt(np.mean((target-ref)**2,axis=1)); s[count[m]<3]=np.nan
        scores[m]=s.reshape(r['scores'].shape); refs[m]=ref; count[m]=count[m].reshape(r['scores'].shape)
        method_times['median_score_'+m+'_seconds']=time.perf_counter()-method_start
    elapsed=time.perf_counter()-t
    return scores,count,sel,refs,inds,ng,{'similarity_seconds':similarity,'selection_median_score_seconds':elapsed,'total_seconds':time.perf_counter()-start,**method_times}

def run():
    out=base.unique_dir(OUT); summary=out/'summary'; summary.mkdir(); prev.N=256; prev.STRIDE=16; data={}; allmetrics=[]; timing=[]; features=[]; selections=[]; instances=[]
    for number in range(2,18):
        start=time.perf_counter(); image=base.DATA/f'image{number}.png'; img=np.asarray(Image.open(image).convert('L')); labels=prev.annotation(image.with_suffix('.json'),img.shape); r=old.load_result(PRIOR/f'image{number}'/'result.npz')
        normal,entries=prev.classification(labels,r); masks=make_masks(labels,img.shape); occ={key:overlap(mask,r) for key,mask in masks.items()}; scores,count,sel,refs,inds,ng,t=compute(r,occ)
        assert np.allclose(scores['A_all'],r['scores'],rtol=1e-5,atol=1e-5,equal_nan=True)
        common=np.logical_and.reduce([np.isfinite(scores[m]) for m in METHODS]); normal_values={}
        for m in METHODS:
            for domain,commonmask in [('all',np.ones_like(common)),('common',common)]:
                score=scores[m]; nv=score[normal&commonmask&np.isfinite(score)]; ns=prev.summary_stats(nv); normal_values[(m,domain)]=nv
                for ent in [e for e in entries if e['label']=='局所変形']:
                    dv=score[ent['center']&commonmask&np.isfinite(score)]; ds=prev.defect_stats(dv,ns)
                    allmetrics.append(dict(image=number,method=m,domain=domain,instance=ent['index'],normal_count=ns['count'],normal_median=ns['median'],normal_std=ns['std'],normal_p99=ns['p99'],normal_p99_exceed_fraction=float(np.mean(nv>ns['p99'])) if len(nv) else None,defect_count=ds['count'],defect_median=ds['median'],defect_max=ds['max'],S=ds['S'],defect_above_normal_p99=ds['normal_p99_exceed_rate']))
            if number in (15,17):
                for k in range(3,9): instances.append(dict(image=number,method=m,reference_count=k,points=int(np.sum((count[m]==k)&np.isfinite(scores[m])))))
        t.update(image=number,wall_including_io_and_masks=time.perf_counter()-start,fft_recomputed=False); timing.append(t)
        data[number]=dict(r=r,img=img,labels=labels,masks=masks,occ=occ,normal=normal,entries=entries,scores=scores,count=count,sel=sel,refs=refs,inds=inds,ng=ng,normal_values=normal_values)
        print('image',number,'processed',round(t['wall_including_io_and_masks'],2),'s',flush=True)
    prev.safe_csv(summary/'all_image_metrics.csv',allmetrics); prev.safe_csv(summary/'method_metrics.csv',[v for v in allmetrics if v['image'] in (15,17)]); prev.safe_csv(summary/'reference_counts.csv',instances); prev.safe_csv(summary/'processing_times.csv',timing)
    loio=[]
    for m in ('A_all','B_lower5','D_consistent5'):
        for num,d in data.items():
            threshold=prev.weighted_threshold([v['normal_values'][(m,'common')] for i,v in data.items() if i!=num]); nv=d['normal_values'][(m,'common')]; common=np.logical_and.reduce([np.isfinite(d['scores'][x]) for x in METHODS]); score=d['scores'][m]
            for ent in [e for e in d['entries'] if e['label']=='局所変形']:
                dv=score[ent['center']&common&np.isfinite(score)]; loio.append(dict(image=num,method=m,instance=ent['index'],threshold=threshold,normal_exceed_fraction=float(np.mean(nv>threshold)),defect_max=float(dv.max()) if len(dv) else None,detected=bool(np.any(dv>threshold)),domain='common'))
            if not any(e['label']=='局所変形' for e in d['entries']): loio.append(dict(image=num,method=m,instance=None,threshold=threshold,normal_exceed_fraction=float(np.mean(nv>threshold)),defect_max=None,detected=None,domain='common'))
    prev.safe_csv(summary/'loio_metrics.csv',loio)
    for num in (15,17):
        d=data[num]; r=d['r']; entry=next(e for e in d['entries'] if e['label']=='局所変形'); yy,xx=np.nonzero(entry['center']&np.isfinite(d['scores']['A_all'])); k=np.argmax(d['scores']['A_all'][yy,xx]); y,x=int(yy[k]),int(xx[k]); index=y*len(r['xs'])+x; X,Y=int(r['xs'][x]),int(r['ys'][y]); folder=out/f'image{num}'; folder.mkdir(); valid=d['inds'][index]>=0; target=r['spectra'][y,x]; freq=(np.arange(len(r['bin_valid']))+.5)*.01; freq=freq[r['bin_valid']]
        fig,ax=plt.subplots(figsize=(9,7)); ax.imshow(d['img'],cmap='gray',vmin=0,vmax=255); ax.contour(d['masks']['局所変形'],levels=[.5],colors='cyan',linewidths=.8); ax.add_patch(Rectangle((X-128,Y-128),256,256,fill=False,edgecolor='red',lw=2))
        for j in np.flatnonzero(valid):
            dx,dy=OFF[j]; px,py=X+dx*128,Y+dy*128; ax.add_patch(Rectangle((px-128,py-128),256,256,fill=False,edgecolor='yellow',alpha=.7)); ax.text(px,py,str(j+1),color='yellow')
            ci=d['inds'][index,j]; cy,cx=divmod(ci,len(r['xs'])); v=d['ng'][index,j]; others=d['ng'][index,valid]; pair=np.sqrt(np.mean((others-v)**2,axis=1)); cosine=(others@v)/(np.linalg.norm(others,axis=1)*np.linalg.norm(v)+1e-12); use=np.flatnonzero(valid)!=j
            row=dict(image=num,id=j+1,x=px,y=py,target_rms=float(np.sqrt(np.mean((target-v)**2))),other_rms_mean=float(np.mean(pair[use])),other_cosine_mean=float(np.mean(cosine[use])),target_overlap_fraction=((256-abs(dx*128))*(256-abs(dy*128)))/65536)
            for label,o in d['occ'].items(): row[label+'_fraction']=float(o[cy,cx])
            features.append(row)
            for method in METHODS: selections.append(dict(image=num,method=method,id=j+1,x=px,y=py,selected=bool(d['sel'][method][index,j])))
        ax.set(xlim=(max(0,X-270),min(d['img'].shape[1],X+270)),ylim=(min(d['img'].shape[0],Y+270),max(0,Y-270)),title=f'image{num} representative target red, references yellow'); fig.savefig(folder/'references.png',dpi=130,bbox_inches='tight'); plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,5)); cmap=plt.get_cmap('viridis');
        for j in np.flatnonzero(valid):
            ci=d['inds'][index,j]; cy,cx=divmod(ci,len(r['xs'])); ax.plot(freq,d['ng'][index,j],color=cmap(d['occ']['background'][cy,cx]),lw=1,label=f'ref {j+1}')
        ax.plot(freq,target,color='red',lw=2,label='target')
        for m,style in zip(METHODS,['--','-.',':','-']): ax.plot(freq,d['refs'][m][index],lw=1.5,ls=style,label=m)
        ax.set(xlabel='cycles/pixel',ylabel='dB',title='References colored by background fraction (0–1)'); ax.legend(ncol=2,fontsize=7); fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap=cmap),ax=ax,label='Background fraction'); fig.savefig(folder/'reference_spectra.png',dpi=130,bbox_inches='tight'); plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,4)); vals=d['ng'][index,valid]; ax.plot(freq,np.std(vals,axis=0)); ax.set(xlabel='cycles/pixel',ylabel='Reference standard deviation (dB)',title='Per-frequency reference variation'); fig.savefig(folder/'reference_variation.png',dpi=130,bbox_inches='tight'); plt.close(fig)
        for shared in (True,False):
            fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True); pool=np.concatenate([v[np.isfinite(v)] for v in d['scores'].values()]); vmin,vmax=np.percentile(pool,[1,99]);
            for ax,m in zip(axes.flat,METHODS):
                im=ax.imshow(d['scores'][m],extent=old.extent(r,16),cmap='inferno',vmin=vmin if shared else None,vmax=vmax if shared else None); ax.contour(d['masks']['局所変形'],levels=[.5],colors='cyan',origin='upper',extent=[0,d['img'].shape[1],d['img'].shape[0],0]); ax.set_title(m); fig.colorbar(im,ax=ax,shrink=.7)
            fig.suptitle('Shared display scale, not detection threshold' if shared else 'Individual display scales, not thresholds'); fig.savefig(folder/('anomaly_maps.png' if shared else 'anomaly_maps_individual.png'),dpi=120); plt.close(fig)
        if num==17:
            assert valid.sum()==8; rows=[]
            for n,combo in enumerate(itertools.combinations(range(8),5),1):
                vals=d['ng'][index,list(combo)]; median=np.median(vals,axis=0); rms=float(np.sqrt(np.mean((target-median)**2))); positions=[divmod(d['inds'][index,j],len(r['xs'])) for j in combo]
                row={'combination':n,'ids':' '.join(str(j+1) for j in combo),'coordinates':';'.join(f'{X+OFF[j][0]*128},{Y+OFF[j][1]*128}' for j in combo),'rms':rms,'background_mean':float(np.mean([d['occ']['background'][p] for p in positions])),'defect_mean':float(np.mean([d['occ']['局所変形'][p] for p in positions])),'target_mean_rms':float(np.mean(np.sqrt(np.mean((vals-target)**2,axis=1)))),'reference_pairwise_rms':float(np.mean([np.sqrt(np.mean((vals[a]-vals[b])**2)) for a,b in itertools.combinations(range(5),2)])),'lower5':combo==(3,4,5,6,7)}
                rows.append(row)
            prev.safe_csv(folder/'all_56_combinations.csv',rows); freqrows=[]
            for z in rows:
                combo=[int(i)-1 for i in z['ids'].split()]; med=np.median(d['ng'][index,combo],axis=0)
                freqrows.extend(dict(combination=z['combination'],frequency=float(f),median_db=float(v)) for f,v in zip(freq,med))
            prev.safe_csv(folder/'combination_spectra.csv',freqrows)
            for name,key in [('combination_rms','combination'),('background_vs_rms','background_mean'),('defect_overlap_vs_rms','defect_mean')]:
                fig,ax=plt.subplots(figsize=(7,5)); ax.scatter([z[key] for z in rows],[z['rms'] for z in rows],s=25,alpha=.65); z=next(z for z in rows if z['lower5']); ax.scatter(z[key],z['rms'],c='red',s=65,label='Lower 5'); ax.axhline(d['scores']['A_all'][y,x],ls='--',color='black',label='All 8'); ax.set(xlabel=key,ylabel='Target vs median RMS (dB)'); ax.legend(); fig.savefig(folder/f'{name}.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    prev.safe_csv(summary/'reference_features.csv',features); prev.safe_csv(summary/'selected_references.csv',selections)
    fig,ax=plt.subplots(figsize=(8,4));
    for num in (15,17):
        v=[next(q for q in allmetrics if q['image']==num and q['method']==m and q['domain']=='common') for m in METHODS]; ax.plot(METHODS,[z['S'] for z in v],marker='o',label=f'image{num}')
    ax.set(ylabel='S (common points)'); ax.legend(); fig.savefig(summary/'separation_scores.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    report(out,data,allmetrics,loio,timing)

def report(out,data,metrics,loio,timing):
    import csv
    summary=out/'summary'; rows=list(csv.DictReader((out/'image17'/'all_56_combinations.csv').open(encoding='utf-8-sig'))); vals=np.array([float(z['rms']) for z in rows]); bg=np.array([float(z['background_mean']) for z in rows]); defect=np.array([float(z['defect_mean']) for z in rows]);
    lines=['# 参照選択の検証','','既存の放射FFT特徴量を再利用。image2〜17、256 px、stride16、間隔128、周辺3×3、0.01〜0.50 cycles/pixel、ビン0.01。56組合せはimage17の従来法における傷中心最大点での探索結果。上側除外・背景低5・一貫性5の比較は参照位置と数が同時に変化し、因果分離を保証しない。','','A=利用可能な最大8個、B=上側3候補を除外、C=背景占有率の低い最大5個（正解ラベル使用、診断専用）、D=周辺参照間のRMS距離の平均が最小のmedoidとその最近傍4個。Dは対象スペクトルと正解ラベルを使わない。候補5個未満では全候補を使用、3個未満は無効。距離・背景の同値は左上から右下の固定順に選択。類似した多数派が正常とは限らず、距離が大きくても強制選択するため安全な自動フォールバック閾値は検証していない。','','対象窓が全ラベルと重ならない点を正常と仮定。共通領域はA〜Dがすべて有効な座標。傷中心・正常領域とも同一画像・同一方式のS、画像内正常99%点を計算。正常誤検出率には画像内正常99%点超と、LOIO閾値超を区別して使用。','','Cは評価画像の正解ラベルを使う探索診断のみ。LOIOにはA・B・Dを固定し、他の画像の正常点を画像均等重みとした99%点を方式ごとに算出した。評価画像ラベルは閾値と参照選択には使用せず、評価と共通有効座標の定義には使用する。Dの規則は事前固定であるが、今回2画像を調べた後の探索であり、独立した外部検証ではない。image5は既存条件選択に使われている。','','## 56組合せ（image17代表点）',f'5参照のRMS範囲 {vals.min():.4f}〜{vals.max():.4f} dB、中央値 {np.median(vals):.4f}。背景占有率とRMSのPearson相関 {np.corrcoef(bg,vals)[0,1]:.3f}、傷占有率との相関 {np.corrcoef(defect,vals)[0,1]:.3f}。56組は参照を共有する従属観測であり、相関は因果を示さない。最大の組合せは傷スコアを見て選ぶ探索的上限で、未知画像での性能ではない。','','## 画像ごとの結果（共通地点）']
    refrows=list(csv.DictReader((summary/'reference_features.csv').open(encoding='utf-8-sig'))); sr=list(csv.DictReader((summary/'selected_references.csv').open(encoding='utf-8-sig')))
    f17=[z for z in refrows if z['image']=='17']; low=sorted(f17,key=lambda z:(float(z['background_fraction']),int(z['id'])))[:5]; high=sorted(f17,key=lambda z:(-float(z['background_fraction']),int(z['id'])))[:5]
    ids=lambda group:' '.join(sorted((z['id'] for z in group),key=int))
    for label,group in [('背景低5',low),('背景高5',high)]:
        matching=next(z for z in rows if z['ids']==ids(group)); lines.insert(-2,f"- {label}: 参照 {matching['ids']}、RMS {float(matching['rms']):.4f} dB。")
    for num in (15,17):
        c={z['id'] for z in sr if z['image']==str(num) and z['method']=='C_low_background' and z['selected']=='True'}; d={z['id'] for z in sr if z['image']==str(num) and z['method']=='D_consistent5' and z['selected']=='True'}
        lines.insert(-2,f'- image{num}代表点：Cの参照 {sorted(c)}、Dの参照 {sorted(d)}、共通 {len(c&d)}/{len(c|d)}（和集合基準）。')
    for num in (15,17):
        for m in METHODS:
            z=next(z for z in metrics if z['image']==num and z['method']==m and z['domain']=='common'); lines.append(f"- image{num} {m}: 傷中央値 {z['defect_median']:.3f}, 最大 {z['defect_max']:.3f}, 正常中央値 {z['normal_median']:.3f}, 正常標準偏差 {z['normal_std']:.3f}, S {z['S']:.3f}, 正常99%点超の傷中心 {z['defect_above_normal_p99']:.1%}。")
    lines+=['','## LOIO（共通地点）']
    for m in ('A_all','B_lower5','D_consistent5'):
        q=[z for z in loio if z['method']==m]; defects=[z for z in q if z['detected'] is not None]; per_image={z['image']:z['normal_exceed_fraction'] for z in q}; lines.append(f"- {m}: 検出 {sum(z['detected'] for z in defects)}/{len(defects)}、正常超過率の画像平均 {np.mean(list(per_image.values())):.2%}。")
    baseline={(z['image'],z['instance']):z['detected'] for z in loio if z['method']=='A_all' and z['detected'] is not None}
    for m in ('B_lower5','D_consistent5'):
        compare={(z['image'],z['instance']):z['detected'] for z in loio if z['method']==m and z['detected'] is not None}
        gained=[f'{i}:{k}' for (i,k),v in compare.items() if v and not baseline[(i,k)]]; lost=[f'{i}:{k}' for (i,k),v in compare.items() if not v and baseline[(i,k)]]
        lines.append(f"- {m} の従来法からの新規検出（画像:ラベル番号） {', '.join(gained) or 'なし'}、新規見逃し {', '.join(lost) or 'なし'}。")
    lines+=['','## 処理時間（特徴量計算済み）',f"16画像での平均：従来方式の中央値・RMS {np.mean([t['median_score_A_all_seconds'] for t in timing]):.3f} 秒、一貫性距離の計算 {np.mean([t['similarity_seconds'] for t in timing]):.3f} 秒、Dの中央値・RMS {np.mean([t['median_score_D_consistent5_seconds'] for t in timing]):.3f} 秒。候補配列作成等を含む全方式合計 {np.mean([t['total_seconds'] for t in timing]):.3f} 秒。FFT、読込、図保存を含まず、同一プロセスで計測した参考値。"]
    lines+=['','## 限界・次の検証','参照数を固定しても組合せにより代表点のRMSが変われば、単に8個という数だけでは説明できない。一方、背景・傷・参照同士の距離は互いに関連し、56組合せは独立ではないため、どれが主因かは確定できない。背景なしと類似スペクトルの一致は代表点のselected_references.csvを比較する。Bは上側を除く固定規則であり画像全体に一般化できる保証はない。','今回の結果からラベルなしのDが正常格子を見つけたと断定できない。参照同士の多数派にも傷が混入する。採用する方式やフォールバックは別の画像で事前に決め、独立データで確認する。計算時間はFFT特徴量読み込み後の選択・中央値・RMSの測定を含む。selection_median_score_secondsはA〜D合計であり、D単独の追加時間ではない。']
    (summary/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('Output:',out,flush=True)

if __name__=='__main__': run()
