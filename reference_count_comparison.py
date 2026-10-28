#!/usr/bin/env python3
"""Compare K=1..8 medoid-neighbor reference selection with edge-aware LOIO."""
import json, time, warnings
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
import reference_selection_analysis as prior

ROOT=base.ROOT; PRIOR=ROOT/'output_all_images_radial_fft'; OUT=ROOT/'output_reference_count_comparison'; OFF=prior.OFF; KS=range(1,9)

def compute(r):
    t0=time.perf_counter(); shape=r['scores'].shape; ny,nx=shape; spec=r['spectra'].reshape(-1,r['spectra'].shape[-1]); n=len(spec); yy,xx=np.indices(shape); idx=np.full((n,8),-1,np.int32)
    for j,(dx,dy) in enumerate(OFF):
        y=yy+dy*8; x=xx+dx*8; good=(y>=0)&(y<ny)&(x>=0)&(x<nx); idx[:,j]=np.where(good,y*nx+x,-1).ravel()
    good=idx>=0; candidates=good.sum(1); ng=spec[np.maximum(idx,0)]; init=time.perf_counter()-t0
    t=time.perf_counter(); delta=ng[:,:,None,:]-ng[:,None,:,:]; dist=np.sqrt(np.mean(delta*delta,axis=-1)); rms_time=time.perf_counter()-t; del delta
    t=time.perf_counter(); pairvalid=good[:,:,None]&good[:,None,:]; means=np.where(good,np.sum(np.where(pairvalid,dist,0),axis=2)/np.maximum(candidates[:,None],1),np.inf); med=np.argmin(means,axis=1); medtime=time.perf_counter()-t
    t=time.perf_counter(); distance=np.where(good,dist[np.arange(n),med],np.inf); order=np.argsort(distance,axis=1,kind='stable'); rank=np.argsort(order,axis=1,kind='stable'); sorttime=time.perf_counter()-t
    scores={}; chosen={}; times=[]
    for k in KS:
        t=time.perf_counter(); use=good&(rank<k); selection_time=time.perf_counter()-t
        t=time.perf_counter();
        with warnings.catch_warnings(): warnings.simplefilter('ignore',RuntimeWarning); ref=np.nanmedian(np.where(use[:,:,None],ng,np.nan),axis=1)
        median_time=time.perf_counter()-t; t=time.perf_counter(); s=np.sqrt(np.mean((spec-ref)**2,axis=1)); s[candidates<3]=np.nan; scoretime=time.perf_counter()-t
        scores[k]=s.reshape(shape); chosen[k]=np.packbits(use,axis=1,bitorder='little').reshape(shape); times.append(dict(K=k,setup_seconds=init,distance_seconds=rms_time,medoid_seconds=medtime,ordering_seconds=sorttime,select_seconds=selection_time,median_seconds=median_time,score_seconds=scoretime,total_if_separate_seconds=init+rms_time+medtime+sorttime+selection_time+median_time+scoretime))
    return scores,chosen,candidates.reshape(shape),idx,ng,times

def csvwrite(path,rows): prev.safe_csv(path,rows)

def representative_fig(folder,num,k,r,image,mask,scores,selected,idx,ng,iy,ix):
    Y=int(r['ys'][iy]); X=int(r['xs'][ix]); index=iy*len(r['xs'])+ix; use=np.unpackbits(np.array([selected[k][iy,ix]],dtype=np.uint8),bitorder='little').astype(bool); target=r['spectra'][iy,ix]; ref=np.median(ng[index,use],axis=0); freq=(np.arange(len(r['bin_valid']))+.5)*.01; freq=freq[r['bin_valid']]
    fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True); axes[0].imshow(image,cmap='gray',vmin=0,vmax=255); axes[0].imshow(scores[k],extent=old.extent(r,16),alpha=.55,cmap='inferno'); axes[0].contour(mask,levels=[.5],colors='cyan'); axes[0].set(xlim=(max(0,X-300),min(image.shape[1],X+300)),ylim=(min(image.shape[0],Y+300),max(0,Y-300)),title='Overlay (individual color scale)')
    axes[1].imshow(image,cmap='gray',vmin=0,vmax=255); axes[1].contour(mask,levels=[.5],colors='cyan'); axes[1].add_patch(Rectangle((X-128,Y-128),256,256,fill=False,color='red',lw=2))
    for j in np.flatnonzero(use):
        dx,dy=OFF[j]; axes[1].add_patch(Rectangle((X+dx*128-128,Y+dy*128-128),256,256,fill=False,color='yellow',alpha=.75)); axes[1].text(X+dx*128,Y+dy*128,str(j+1),color='yellow')
        axes[2].plot(freq,ng[index,j],alpha=.45,lw=.8)
    axes[1].set(xlim=(max(0,X-270),min(image.shape[1],X+270)),ylim=(min(image.shape[0],Y+270),max(0,Y-270)),title=f'K={k}, actual {use.sum()}, red target')
    axes[2].plot(freq,target,color='red',lw=2,label='target'); axes[2].plot(freq,ref,color='black',ls='--',lw=2,label='median'); axes[2].set(xlabel='cycles/pixel',ylabel='dB',title=f'RMS {scores[k][iy,ix]:.3f} dB'); axes[2].legend()
    folder.mkdir(parents=True,exist_ok=True); fig.savefig(folder/'detail.png',dpi=120); plt.close(fig)

def main():
    settings=json.loads((PRIOR/'settings.json').read_text(encoding='utf-8')); assert (settings['window'],settings['stride'],settings['reference_spacing'],settings['frequency'],settings['bin_width'])==(256,16,128,[.01,.5],.01)
    out=base.unique_dir(OUT); summary=out/'summary'; summary.mkdir(); prev.N=256; prev.STRIDE=16; all_data={}; metrics=[]; timing=[]; actual=[]; edge_checks=[]
    for number in range(1,22):
        path=base.DATA/f'image{number}.png'; img=np.asarray(Image.open(path).convert('L')); labels=prev.annotation(path.with_suffix('.json'),img.shape)
        source=PRIOR/f'image{number}'/'result.npz'
        if source.exists(): r=old.load_result(source)
        else:
            rr=prev.analyze(img,4,64,False); r={key:rr[key] for key in ('xs','ys','scores','spectra','bin_counts','bin_valid')}
        scores,selected,candidates,idx,ng,times=compute(r); normal,entries=prev.classification(labels,r); ny,nx=r['scores'].shape; xx,yy=np.meshgrid(r['xs'],r['ys']); edge=np.minimum.reduce([xx-128,yy-128,img.shape[1]-xx-128,img.shape[0]-yy-128]); valid=candidates>=3
        if normal is None: normal=np.zeros((ny,nx),bool)
        # All K values operate on the same candidate-valid coordinates.
        for k in KS: assert np.array_equal(np.isfinite(scores[k]),valid)
        for m in range(3,9):
            mask=candidates==m
            for k in range(m,9): assert np.allclose(scores[k][mask],scores[m][mask],rtol=1e-6,atol=1e-6,equal_nan=True)
        if number in (15,17):
            expected=(896,224,5) if number==15 else (656,336,8); x,y,M=expected; xi=int(np.flatnonzero(r['xs']==x)[0]); yi=int(np.flatnonzero(r['ys']==y)[0]); assert candidates[yi,xi]==M
            edge_checks.append(dict(image=number,x=x,y=y,candidates=M,actual_by_K=' '.join(str(min(k,M)) for k in KS),pass_check=True))
        group='no_grid' if number in prev.NO_GRID else ('small_reference' if number==18 else 'grid'); domains={'all':valid,'interior8':candidates==8}; samples={}
        folder=(out/f'image{number}') if number in (15,17) else (out/'other_images'/f'image{number}'); folder.mkdir(parents=True,exist_ok=True)
        # Selected bits follow OFF order; centers + 128*OFF recover every reference coordinate.
        np.savez_compressed(folder/'locations.npz',x_centers=r['xs'],y_centers=r['ys'],candidate_count=candidates.astype(np.uint8),edge_distance_px=edge.astype(np.int16),**{f'score_k{k}':scores[k] for k in KS},**{f'selected_bits_k{k}':selected[k] for k in KS})
        for k in KS:
            for M in range(3,9): actual.append(dict(image=number,group=group,K=k,candidates=M,actual_used=min(k,M),locations=int(np.sum(candidates==M))))
            for domain,dm in domains.items():
                nv=scores[k][normal&dm&valid]; ns=prev.summary_stats(nv); samples[(k,domain)]=nv
                for ent in [e for e in entries if e['label']=='局所変形']:
                    dv=scores[k][ent['center']&dm&valid]; ds=prev.defect_stats(dv,ns)
                    metrics.append(dict(image=number,group=group,K=k,domain=domain,instance=ent['index'],normal_count=ns['count'],normal_median=ns['median'],normal_std=ns['std'],normal_p99=ns['p99'],normal_p99_exceed_rate=float(np.mean(nv>ns['p99'])) if len(nv) else None,defect_count=ds['count'],defect_median=ds['median'],defect_max=ds['max'],defect_p95=ds['p95'],S=ds['S'],defect_normal_p99_exceed_rate=ds['normal_p99_exceed_rate']))
        for t in times: timing.append(dict(image=number,group=group,**t))
        all_data[number]=dict(group=group,scores=scores,normal=normal,entries=entries,samples=samples,domains=domains,valid=valid)
        if number in (15,17):
            ent=next(e for e in entries if e['label']=='局所変形'); yy,xx=np.nonzero(ent['center']&valid); i=np.argmax(scores[5][yy,xx]); iy,ix=int(yy[i]),int(xx[i]); mask=next(m for name,m in labels if name=='局所変形')
            for k in KS: representative_fig(folder/f'k{k}',number,k,r,img,mask,scores,selected,idx,ng,iy,ix)
            fig,axes=plt.subplots(2,4,figsize=(17,8),constrained_layout=True); values=np.concatenate([v[np.isfinite(v)] for v in scores.values()]); low,high=np.percentile(values,[1,99])
            for ax,k in zip(axes.flat,KS):
                im=ax.imshow(scores[k],extent=old.extent(r,16),vmin=low,vmax=high,cmap='inferno'); ax.contour(mask,levels=[.5],colors='cyan',origin='upper',extent=[0,img.shape[1],img.shape[0],0]); ax.set_title(f'K={k}'); fig.colorbar(im,ax=ax,shrink=.6)
            fig.suptitle('Shared display range; not detection thresholds'); fig.savefig(folder/'comparison.png',dpi=120); plt.close(fig)
        print(f'image{number}: {group}, valid={int(valid.sum())}',flush=True)
    csvwrite(summary/'defect_instance_metrics.csv',metrics); csvwrite(summary/'actual_reference_counts.csv',actual); csvwrite(summary/'processing_times.csv',timing); csvwrite(summary/'edge_checks.csv',edge_checks)
    loio=[]
    for group in ('grid','no_grid'):
        members={i:d for i,d in all_data.items() if d['group']==group}
        for domain in ('all','interior8'):
            for k in KS:
                for i,d in members.items():
                    threshold=prev.weighted_threshold([v['samples'][(k,domain)] for j,v in members.items() if j!=i]); normal=d['samples'][(k,domain)]
                    for ent in [e for e in d['entries'] if e['label']=='局所変形']:
                        dv=d['scores'][k][ent['center']&d['domains'][domain]&d['valid']]; usable=bool(len(dv) and threshold is not None)
                        loio.append(dict(image=i,group=group,K=k,domain=domain,instance=ent['index'],threshold=threshold,normal_exceed_rate=float(np.mean(normal>threshold)) if len(normal) and threshold is not None else None,defect_count=len(dv),defect_exceed_rate=float(np.mean(dv>threshold)) if usable else None,detected=bool(np.any(dv>threshold)) if usable else None))
                    if not any(e['label']=='局所変形' for e in d['entries']): loio.append(dict(image=i,group=group,K=k,domain=domain,instance=None,threshold=threshold,normal_exceed_rate=float(np.mean(normal>threshold)) if len(normal) and threshold is not None else None,defect_count=0,defect_exceed_rate=None,detected=None))
    csvwrite(summary/'loio_metrics.csv',loio)
    aggregate=[]
    for group in ('grid','no_grid'):
        for domain in ('all','interior8'):
            for k in KS:
                rows=[z for z in loio if z['group']==group and z['domain']==domain and z['K']==k]; usable=[z for z in rows if z['detected'] is not None]; normal_by_image={z['image']:z['normal_exceed_rate'] for z in rows if z['normal_exceed_rate'] is not None}; ms=[z['S'] for z in metrics if z['group']==group and z['domain']==domain and z['K']==k and z['S'] is not None]
                aggregate.append(dict(group=group,domain=domain,K=k,detected=sum(z['detected'] for z in usable),evaluable=len(usable),non_evaluable=sum(z['instance'] is not None and z['detected'] is None for z in rows),normal_exceed_image_mean=float(np.mean(list(normal_by_image.values()))) if normal_by_image else None,S_median=float(np.median(ms)) if ms else None))
    csvwrite(summary/'reference_count_metrics.csv',aggregate)
    for name,key,ylabel in [('detection_vs_reference_count','detected','Detected instances'),('false_positive_vs_reference_count','normal_exceed_image_mean','Normal exceed rate'),('separation_vs_reference_count','S_median','Median S across defects')]:
        fig,ax=plt.subplots(figsize=(8,5))
        for domain in ('all','interior8'):
            q=[a for a in aggregate if a['group']=='grid' and a['domain']==domain]; ax.plot(list(KS),[z[key] for z in q],marker='o',label=domain)
        ax.set(xlabel='Requested references K',ylabel=ylabel,xticks=list(KS)); ax.legend(); fig.savefig(summary/f'{name}.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,7)); defects=sorted(set((z['image'],z['instance']) for z in metrics if z['group']=='grid')); matrix=np.full((len(defects),8),np.nan)
    for i,(num,instance) in enumerate(defects):
        for k in KS:
            z=next(v for v in metrics if v['image']==num and v['instance']==instance and v['group']=='grid' and v['K']==k and v['domain']=='all'); matrix[i,k-1]=z['S'] if z['S'] is not None else np.nan
    im=ax.imshow(matrix,aspect='auto',cmap='coolwarm'); ax.set(xticks=range(8),xticklabels=list(KS),yticks=range(len(defects)),yticklabels=[f'image{i}:{j}' for i,j in defects],xlabel='K',ylabel='Defect instance'); fig.colorbar(im,ax=ax,label='S'); fig.savefig(summary/'defect_reference_heatmap.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6))
    for i,(num,instance) in enumerate(defects): ax.plot(list(KS),matrix[i],marker='.',label=f'image{num}:{instance}')
    ax.set(xlabel='K',ylabel='S (all valid locations)',xticks=list(KS)); ax.legend(ncol=3,fontsize=7); fig.savefig(summary/'individual_separation_vs_reference_count.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
    for ax,k in zip(axes.flat,KS):
        q=[z for z in actual if z['K']==k and z['group']=='grid']; ax.bar(range(1,9),[sum(z['locations'] for z in q if z['actual_used']==i) for i in range(1,9)]); ax.set_title(f'K={k}'); ax.set(xlabel='Actual reference count',ylabel='Locations')
    fig.savefig(summary/'actual_reference_distribution.png',dpi=120); plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
    for ax,k in zip(axes.flat,KS):
        mat=np.zeros((21,8))
        for z in actual:
            if z['K']==k: mat[z['image']-1,z['actual_used']-1]+=z['locations']
        ax.imshow(mat,aspect='auto'); ax.set(title=f'K={k}',xlabel='Actual references',ylabel='Image',xticks=range(8),xticklabels=range(1,9),yticks=[0,4,9,14,17,20],yticklabels=[1,5,10,15,18,21])
    fig.savefig(summary/'actual_reference_distribution_by_image.png',dpi=130); plt.close(fig)
    lines=['# 参照数と画像端の比較','','既存放射FFT特徴量を再利用し、image1〜21を解析。窓256 px・stride16・参照間隔128 px。D方式は全有効候補のスペクトル間RMS距離から平均距離最小のmedoidを選び、同点は左上から右下の固定順、medoidから近い順に最大K個を選んで周波数別中央値を取る。K=1はmedoid単独。候補数Mが3以上の地点だけを全K共通で評価し、実使用数=min(K,M)。対象と参照は完全に画像内にある窓のみで、パディングは使用しない。','','全有効地点と、8候補が利用可能な地点を別評価。LOIOの閾値はK・領域・照明グループ別に、評価画像以外の正常窓の画像均等重み99%点として求める。正常窓は全アノテーションと重ならない窓。傷中心に1点以上の超過があるインスタンスを検出とし、8候補領域に傷中心がない場合は未評価として分母から除外する。無格子1,19,20,21は別のLOIO。image18は小画像の参考集計で、独立群の閾値は作らない。','','前回の5参照LOIOの共通領域とは異なり、今回は全有効と8候補の2領域で閾値を再計算している。Kはこの実験で比較しているため、最大検出のKを選んだ成績は独立した最終評価ではない。各FFT窓は重複する。','','## 画像端の検証','image15 (896,224) は候補5、image17 (656,336) は候補8。各候補数Mの地点でK>=Mの全結果が一致すること、全Kで候補3以上の同じ地点が有効なことをassertで確認。image15の最上端 y=128 の対象窓も画像内に収まり、対応する参照が3個以上あれば計算する。','','## 格子画像のLOIO結果']
    for domain in ('all','interior8'):
        lines.append(f'### {domain}')
        for z in [v for v in aggregate if v['group']=='grid' and v['domain']==domain]: lines.append(f"- K={z['K']}: 検出 {z['detected']}/{z['evaluable']}、未評価 {z['non_evaluable']}、正常超過率の画像平均 {z['normal_exceed_image_mean']:.2%}、傷Sの中央値 {z['S_median']:.3f}。")
    lines+=['','## 無格子画像（別LOIO、全有効地点）']
    for z in [v for v in aggregate if v['group']=='no_grid' and v['domain']=='all']: lines.append(f"- K={z['K']}: 検出 {z['detected']}/{z['evaluable']}、正常超過率の画像平均 {z['normal_exceed_image_mean']:.2%}、傷Sの中央値 {z['S_median']:.3f}。")
    lines+=['','## image15・17と計算時間']
    for num in (15,17):
        q=[z for z in metrics if z['image']==num and z['domain']=='all']; lines.append(f"- image{num}の各KのS: "+', '.join(f"{z['K']}={z['S']:.3f}" for z in q)+'。')
    for k in (1,4,5,8):
        q=[z for z in timing if z['group']=='grid' and z['K']==k]; lines.append(f"- K={k}のFFT計算後の単独実行相当時間の画像平均 {np.mean([z['total_if_separate_seconds'] for z in q]):.3f}秒。")
    lines.append('image18は候補数3以上の窓が0点で、ラベルもないため性能評価はできない。')
    lines+=['','## 注意','K=8は有効候補全数の中央値と一致する。medoid近傍のKと、ランダムKや任意のKは異なる。K=1対K>1の比較は中央値と選択数の影響を見るが、各Kで選ばれる集合も変わるため、それぞれの因果効果を完全には分離しない。正常誤検出率は窓単位であり、実製品の画像単位誤検出率ではない。画像ごとに参照数を最適化した結果や自動選択の性能は測定していない。','全地点の候補数・K別異常度・選択方向・画像端距離は各locations.npzに保存。selected_bits_kNのbit 0〜7は左上、上、右上、左、右、左下、下、右下に対応。中心座標に方向×128 pxを加えると選択参照座標を復元できる。','処理時間は既存FFT特徴量の読込後を対象とする。距離・medoid・並べ替えは全Kで共用し、total_if_separate_secondsは各Kを単独実行した場合の構成要素の合算である。']
    (summary/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('Output:',out,flush=True)

if __name__=='__main__': main()
