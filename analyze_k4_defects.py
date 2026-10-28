#!/usr/bin/env python3
"""Per-instance diagnostic review of saved K=4,5,8 radial FFT results."""
import csv, json
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
import reference_selection_analysis as ra

PRIOR=base.ROOT/'output_all_images_radial_fft'; KC=base.ROOT/'output_reference_count_comparison'; OUT=base.ROOT/'output_k4_defect_analysis'; KS=(4,5,8); OFF=ra.OFF

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def num(v): return float(v) if v not in ('',None) else np.nan
def flag(v): return {'True':True,'False':False}.get(v)
def save(path,rows): prev.safe_csv(path,rows)
def proportion(mask,x,y): return float(mask[y-128:y+128,x-128:x+128].sum()/65536)

def main():
    out=base.unique_dir(OUT); summary=out/'summary'; figs=summary/'figures'; figs.mkdir(parents=True); (out/'images').mkdir()
    prev.N=256; prev.STRIDE=16
    ms=read(KC/'summary'/'defect_instance_metrics.csv'); ls=read(KC/'summary'/'loio_metrics.csv'); rows=[]; comparisons=[]; contamination=[]; contributions=[]; images=[]; timing=[]
    for numimg in range(1,22):
        source=base.DATA/f'image{numimg}.png'; image=np.asarray(Image.open(source).convert('L')); labels=prev.annotation(source.with_suffix('.json'),image.shape); folder=out/'images'/f'image{numimg}'; folder.mkdir()
        group='no_grid' if numimg in prev.NO_GRID else ('small_reference' if numimg==18 else 'grid')
        img_rows=[z for z in ls if int(z['image'])==numimg and z['K']=='4' and z['domain']=='all']
        scorepath=(KC/f'image{numimg}'/'locations.npz') if numimg in (15,17) else (KC/'other_images'/f'image{numimg}'/'locations.npz')
        with np.load(scorepath) as z: loc={k:z[k].copy() for k in z.files}
        valid=int(np.sum(loc['candidate_count']>=3));
        images.append(dict(image=numimg,group=group,width=image.shape[1],height=image.shape[0],labels_available=labels is not None,defect_instances=sum(name=='局所変形' for name,_ in (labels or [])),valid_locations=valid,loio_threshold=num(img_rows[0]['threshold']) if img_rows else None,normal_loio_exceed_rate=num(img_rows[0]['normal_exceed_rate']) if img_rows else None,status='ok' if valid and labels else ('no labels' if valid else 'no location with 3 references')))
        if not valid or not labels:
            (folder/'status.txt').write_text('No evaluable locations with >=3 reference candidates or no annotations.\n',encoding='utf-8'); continue
        defect_labels=[(j,mask) for j,(label,mask) in enumerate(labels) if label=='局所変形']
        if not defect_labels: (folder/'status.txt').write_text('No local deformation annotation.\n',encoding='utf-8'); continue
        r=old.load_result(PRIOR/f'image{numimg}'/'result.npz'); full_masks=ra.make_masks(labels,image.shape); normal,entries=prev.classification(labels,r); xxgrid,yygrid=np.meshgrid(r['xs'],r['ys']); h,w=image.shape
        for instance,mask in defect_labels:
            entry=next(e for e in entries if e['index']==instance); yy,xx=np.nonzero(mask); cx=float(xx.mean()); cy=float(yy.mean()); area=int(mask.sum()); bw=int(xx.max()-xx.min()+1); bh=int(yy.max()-yy.min()+1); long=max(bw,bh); short=min(bw,bh)
            dent=entry['center']&np.isfinite(loc['score_k4']); iy,ix=np.nonzero(dent)
            if not len(iy):
                rows.append(dict(image=numimg,group=group,instance=instance,status='no defect center with >=3 candidates',area_px=area,width_px=bw,height_px=bh)); continue
            kmax=int(np.argmax(loc['score_k4'][iy,ix])); y,x=int(iy[kmax]),int(ix[kmax]); X,Y=int(r['xs'][x]),int(r['ys'][y]); bits={k:np.unpackbits(np.array([loc[f'selected_bits_k{k}'][y,x]],np.uint8),bitorder='little').astype(bool) for k in KS}; offsets=[]
            for j,(dx,dy) in enumerate(OFF):
                xp,yp=X+dx*128,Y+dy*128
                if not (128<=xp<=w-128 and 128<=yp<=h-128): continue
                xidx=np.flatnonzero(r['xs']==xp); yidx=np.flatnonzero(r['ys']==yp)
                if len(xidx)==0 or len(yidx)==0: continue
                offsets.append((j,xp,yp,int(yidx[0]),int(xidx[0])))
            target=r['spectra'][y,x]; sums={}; diffinfo={}; freqindex=np.flatnonzero(r['bin_valid']); freq=(freqindex+.5)*.01
            other=np.logical_or.reduce([m for j,(_,m) in enumerate(labels) if j!=instance]) if len(labels)>1 else np.zeros(image.shape,bool)
            bytype={'background':full_masks['background'],'tag':full_masks['tag'],'current_defect':mask,'other_annotations':other,'any_deformation':full_masks['局所変形'],'other_damage':full_masks['other']}
            target_oc={name:proportion(m,X,Y) for name,m in bytype.items()}
            for j,xp,yp,ry,rx in offsets:
                record=dict(image=numimg,group=group,instance=instance,reference_id=j+1,x=xp,y=yp,selected_k4=bool(bits[4][j]),selected_k5=bool(bits[5][j]),selected_k8=bool(bits[8][j]),target_spectrum_rms=float(np.sqrt(np.mean((target-r['spectra'][ry,rx])**2))))
                for name,m in bytype.items(): record[name+'_fraction']=proportion(m,xp,yp)
                contamination.append(record)
            for k in KS:
                spectra=np.stack([r['spectra'][ry,rx] for j,xp,yp,ry,rx in offsets if bits[k][j]]); ref=np.median(spectra,axis=0); diff=target-ref; sq=diff**2; total=float(sq.sum()); rms=float(np.sqrt(np.mean(sq)))
                assert np.isclose(rms,loc[f'score_k{k}'][y,x],rtol=1e-5,atol=1e-5)
                diffinfo[k]=(spectra,ref,diff,sq)
                # freqindex 50 corresponds to the exact 0.50 cycles/pixel endpoint.
                for name,band in [('0.01–0.05',(freqindex>=1)&(freqindex<5)),('0.05–0.10',(freqindex>=5)&(freqindex<10)),('0.10–0.20',(freqindex>=10)&(freqindex<20)),('0.20–0.30',(freqindex>=20)&(freqindex<30)),('0.30–0.50',(freqindex>=30)&(freqindex<=50))]:
                    frac=float(sq[band].sum()/total) if total else np.nan; contributions.append(dict(image=numimg,group=group,instance=instance,K=k,representative_x=X,representative_y=Y,band=name,contribution=frac,bins=int(band.sum())))
            met={k:next(z for z in ms if int(z['image'])==numimg and z['instance']==str(instance) and z['K']==str(k) and z['domain']=='all') for k in KS}; lm={k:next(z for z in ls if int(z['image'])==numimg and z['instance']==str(instance) and z['K']==str(k) and z['domain']=='all') for k in KS}
            selected_rows=[z for z in contamination if z['image']==numimg and z['instance']==instance and z['selected_k4']]; rbg=float(np.mean([z['background_fraction'] for z in selected_rows])); rdef=float(np.mean([z['current_defect_fraction'] for z in selected_rows])); rother=float(np.mean([z['other_annotations_fraction'] for z in selected_rows])); refnum=len(selected_rows)
            record=dict(image=numimg,group=group,instance=instance,status='evaluated',area_px=area,width_px=bw,height_px=bh,long_px=long,short_px=short,aspect_ratio=long/short,centroid_x=cx,centroid_y=cy,bbox_edge_distance_px=int(min(xx.min(),yy.min(),w-1-xx.max(),h-1-yy.max())),center_count=len(iy),representative_x=X,representative_y=Y,representative_edge_distance_px=int(loc['edge_distance_px'][y,x]),candidates=int(loc['candidate_count'][y,x]),actual_selected_k4=refnum,grid_brightness_class='undetermined',target_current_defect_fraction=target_oc['current_defect'],target_background_fraction=target_oc['background'],target_tag_fraction=target_oc['tag'],target_other_annotation_fraction=target_oc['other_annotations'],ref_background_mean=rbg,ref_current_defect_mean=rdef,ref_other_annotations_mean=rother)
            for k in KS:
                record.update({f'k{k}_defect_median':num(met[k]['defect_median']),f'k{k}_defect_max':num(met[k]['defect_max']),f'k{k}_normal_median':num(met[k]['normal_median']),f'k{k}_normal_std':num(met[k]['normal_std']),f'k{k}_S':num(met[k]['S']),f'k{k}_threshold':num(lm[k]['threshold']),f'k{k}_max_minus_threshold':num(met[k]['defect_max'])-num(lm[k]['threshold']),f'k{k}_exceed_rate':num(lm[k]['defect_exceed_rate']),f'k{k}_detected':flag(lm[k]['detected']),f'k{k}_representative_score':float(loc[f'score_k{k}'][y,x])})
            rows.append(record); detected=[str(k) for k in KS if record[f'k{k}_detected']]; comparisons.append(dict(image=numimg,group=group,instance=instance,detected_k4=record['k4_detected'],detected_k5=record['k5_detected'],detected_k8=record['k8_detected'],classification=','.join(detected) if detected else 'none',k4_vs_k8=('gained' if record['k4_detected'] and not record['k8_detected'] else 'lost' if not record['k4_detected'] and record['k8_detected'] else 'same'),excluded_k4_reference_ids=','.join(str(j+1) for j,_,_,_,_ in offsets if bits[8][j] and not bits[4][j])))
            sub=folder/f'defect_{instance}'; sub.mkdir();
            spectral_rows=[]
            for b,f in enumerate(freq):
                z={'frequency_bin_index':int(freqindex[b]),'frequency_bin_center':float(f),'target_db':float(target[b])}
                for k in KS:
                    _,ref,diff,sq=diffinfo[k]; z[f'k{k}_median_db']=float(ref[b]); z[f'k{k}_difference_db']=float(diff[b]); z[f'k{k}_squared_contribution']=float(sq[b]/sq.sum()) if sq.sum() else None
                for j,_,_,ry,rx in offsets: z[f'reference_{j+1}_db']=float(r['spectra'][ry,rx,b])
                spectral_rows.append(z)
            save(sub/'spectra.csv',spectral_rows)
            fig,axes=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
            for ax in axes: ax.imshow(image,cmap='gray',vmin=0,vmax=255); ax.contour(mask,levels=[.5],colors='cyan'); ax.add_patch(Rectangle((X-128,Y-128),256,256,fill=False,edgecolor='red',lw=2)); ax.set(xlim=(max(0,X-300),min(w,X+300)),ylim=(min(h,Y+300),max(0,Y-300)))
            for j,xp,yp,_,_ in offsets:
                if bits[4][j]: axes[1].add_patch(Rectangle((xp-128,yp-128),256,256,fill=False,edgecolor='yellow',lw=1.3)); axes[1].text(xp,yp,str(j+1),color='yellow')
            axes[0].set_title('Target window (red)'); axes[1].set_title('K=4 selected references (yellow)'); fig.savefig(sub/'target_and_references.png',dpi=125); plt.close(fig)
            fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True); vals=np.concatenate([loc[f'score_k{k}'][np.isfinite(loc[f'score_k{k}'])] for k in KS]); vmin,vmax=np.percentile(vals,[1,99])
            for ax,k in zip(axes,KS):
                im=ax.imshow(image,cmap='gray'); im=ax.imshow(loc[f'score_k{k}'],extent=old.extent(r,16),cmap='inferno',alpha=.6,vmin=vmin,vmax=vmax); ax.contour(mask,levels=[.5],colors='cyan'); ax.set(xlim=(max(0,X-300),min(w,X+300)),ylim=(min(h,Y+300),max(0,Y-300)),title=f'K={k}'); fig.colorbar(im,ax=ax,shrink=.7)
            fig.suptitle('Shared display range, not threshold'); fig.savefig(sub/'score_maps.png',dpi=125); plt.close(fig)
            fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
            for k,color in zip(KS,['tab:blue','tab:orange','tab:green']):
                spectra,ref,diff,sq=diffinfo[k]; axes[0,0].plot(freq,ref,color=color,label=f'K={k} median'); axes[0,1].plot(freq,diff,color=color,label=f'K={k}'); q=[v for v in contributions if v['image']==numimg and v['instance']==instance and v['K']==k]; axes[1,1].plot(range(5),[v['contribution'] for v in q],marker='o',color=color,label=f'K={k}')
            axes[0,0].plot(freq,target,color='red',lw=2,label='target');
            for spectrum in diffinfo[4][0]: axes[1,0].plot(freq,spectrum,alpha=.65)
            axes[1,0].plot(freq,target,color='red',lw=2,label='target'); axes[1,0].plot(freq,diffinfo[4][1],color='black',ls='--',label='median'); axes[0,0].set_title('Target and medians'); axes[0,1].set_title('Target minus median (dB)'); axes[1,0].set_title('K=4 selected spectra'); axes[1,1].set_title('Squared-RMS contribution'); axes[1,1].set_xticks(range(5),['.01-.05','.05-.10','.10-.20','.20-.30','.30-.50'],rotation=20)
            for ax in axes.flat: ax.legend(fontsize=7)
            fig.savefig(sub/'spectra_and_contributions.png',dpi=125); plt.close(fig)
        print(f'image{numimg}: {len(defect_labels)} local deformations',flush=True)
    save(summary/'all_defect_metrics.csv',rows); save(summary/'reference_contamination.csv',contamination); save(summary/'frequency_contributions.csv',contributions); save(summary/'k4_k5_k8_comparison.csv',comparisons); save(summary/'image_summary.csv',images)
    grid=[z for z in rows if z.get('status')=='evaluated' and z['group']=='grid']; variables=['area_px','long_px','short_px','aspect_ratio','bbox_edge_distance_px','target_current_defect_fraction','ref_background_mean','ref_current_defect_mean','actual_selected_k4','k4_S']
    stats=[]
    for var in variables:
        for detected in (True,False):
            vals=np.array([z[var] for z in grid if z['k4_detected']==detected],float); stats.append(dict(feature=var,detected=detected,n=len(vals),median=float(np.median(vals)) if len(vals) else None,minimum=float(vals.min()) if len(vals) else None,maximum=float(vals.max()) if len(vals) else None))
    save(summary/'detected_vs_missed.csv',stats)
    for key,name,label in [('area_px','size_vs_S','Area (px²)'),('ref_background_mean','reference_background_vs_S','Mean selected reference background fraction')]:
        fig,ax=plt.subplots(figsize=(7,5));
        for detected,color in [(True,'tab:blue'),(False,'tab:red')]:
            v=[z for z in grid if z['k4_detected']==detected]; ax.scatter([z[key] for z in v],[z['k4_S'] for z in v],label='detected' if detected else 'missed',color=color)
            for z in v: ax.annotate(str(z['image']),(z[key],z['k4_S']),fontsize=7)
        if key=='area_px': ax.set_xscale('log')
        ax.set(xlabel=label,ylabel='K=4 S'); ax.legend(); fig.savefig(figs/f'{name}.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    for ax,key in zip(axes.flat,['area_px','long_px','aspect_ratio','target_current_defect_fraction','ref_background_mean','ref_current_defect_mean']):
        v=[[z[key] for z in grid if z['k4_detected']==state] for state in (True,False)]; ax.boxplot(v,labels=['detected','missed']); ax.set_title(key)
    fig.savefig(figs/'detected_vs_missed.png',dpi=130); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,8)); matrix=np.array([[int(z[f'k{k}_detected']) for k in KS] for z in grid]); ax.imshow(matrix,aspect='auto',cmap='Blues',vmin=0,vmax=1); ax.set(xticks=range(3),xticklabels=KS,yticks=range(len(grid)),yticklabels=[f"image{z['image']}:{z['instance']}" for z in grid],xlabel='K',ylabel='Defect'); fig.savefig(figs/'loio_detection_heatmap.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4)); v=[z for z in images if z['normal_loio_exceed_rate'] is not None]; ax.bar([z['image'] for z in v],[z['normal_loio_exceed_rate'] for z in v],color=['tab:blue' if z['group']=='grid' else 'tab:orange' for z in v]); ax.set(xlabel='Image',ylabel='K=4 LOIO normal exceed rate',xticks=range(1,22)); fig.savefig(figs/'normal_exceed_by_image.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    write_report(summary,rows,comparisons,contamination,contributions,images,stats)
    print('Output:',out,flush=True)

def write_report(summary,rows,comparisons,contamination,contributions,images,stats):
    grid=[z for z in rows if z.get('status')=='evaluated' and z['group']=='grid']; on=[z for z in grid if z['k4_detected']]; off=[z for z in grid if not z['k4_detected']]
    lines=['# K=4 局所変形の検出・見逃し分析','','256 px窓、stride16、参照間隔128 px、最大8候補、medoidと最近傍のK=4・5・8。元の放射FFT特徴量・保存済み選択結果・全有効地点LOIOを使用。候補3未満は無効。Kごとの閾値は画像を除いた正常画像から設定し、格子あり・なしを分けた。K=4は同じ画像群の先行比較から選んだ探索条件であり、独立した汎化性能ではない。','','代表窓は各傷のK=4傷中心スコア最大点で、他のKも同じ座標の対象・参照スペクトルを比較。検出の成否はKごとに傷中心全地点の最大値で判定するため、代表点のスコアと別Kの傷中心最大値は一致しない場合がある。代表点選択により傷の平均的な状態を表すとは限らない。参照窓や画像範囲は強く重複する。明部・暗部・格子境界の信頼できる分類は実施せず全件未判定。','','各周波数帯の寄与は対象と基準のdB差の二乗和を全ビンの二乗和で割った値。0.50 cycles/pixelの終点成分を最後の帯に含め、最後のビン中心表示が0.505になることがある。','','## 検出・見逃し',f'格子あり：検出 {len(on)}/{len(grid)}、見逃し {len(off)}/{len(grid)}。']
    for z in grid: lines.append(f"- image{z['image']} 傷{z['instance']}: {'検出' if z['k4_detected'] else '見逃し'}、面積 {z['area_px']} px²、長短辺 {z['long_px']}/{z['short_px']} px、S {z['k4_S']:.2f}、最大 {z['k4_defect_max']:.3f}、LOIO閾値 {z['k4_threshold']:.3f}、差 {z['k4_max_minus_threshold']:+.3f} dB。")
    lines+=['','## 群別の中央値']
    for key in ('area_px','long_px','short_px','aspect_ratio','bbox_edge_distance_px','target_current_defect_fraction','ref_background_mean','ref_current_defect_mean','actual_selected_k4','k4_S'):
        a=next(z for z in stats if z['feature']==key and z['detected']); b=next(z for z in stats if z['feature']==key and not z['detected']); lines.append(f"- {key}: 検出群 {a['median']:.4g}（範囲 {a['minimum']:.4g}〜{a['maximum']:.4g}）、見逃し群 {b['median']:.4g}（範囲 {b['minimum']:.4g}〜{b['maximum']:.4g}）。")
    lines+=['','## 周波数帯別寄与の群比較']
    for band in ('0.01–0.05','0.05–0.10','0.10–0.20','0.20–0.30','0.30–0.50'):
        for state in (True,False):
            ids={(z['image'],z['instance']) for z in grid if z['k4_detected']==state}; values=[v['contribution'] for v in contributions if v['K']==4 and v['band']==band and (v['image'],v['instance']) in ids]
            lines.append(f"- {band} cycles/pixel、{'検出' if state else '見逃し'}群: 寄与率の中央値 {np.median(values):.1%}。")
    lines+=['','## K=4とK=8の違い']
    for c in comparisons:
        if c['group']=='grid' and c['k4_vs_k8']!='same':
            refs=[r for r in contamination if r['image']==c['image'] and r['instance']==c['instance'] and r['selected_k8'] and not r['selected_k4']]
            bg=np.mean([r['background_fraction'] for r in refs]); defect=np.mean([r['current_defect_fraction'] for r in refs]); lines.append(f"- image{c['image']} 傷{c['instance']}: {c['k4_vs_k8']}、代表点でK=4から除外した参照 {c['excluded_k4_reference_ids']}、その平均背景占有率 {bg:.1%}、対象傷占有率 {defect:.1%}。これらは当該参照が原因であることを示す対照実験ではない。")
    lines+=['','## image5・15・17の詳細']
    for z in rows:
        if z.get('status')=='evaluated' and z['image'] in (5,15,17):
            band=[v for v in contributions if v['image']==z['image'] and v['instance']==z['instance'] and v['K']==4]; peak=max(band,key=lambda v:v['contribution']); lines.append(f"- image{z['image']} 傷{z['instance']}: K4検出 {z['k4_detected']}、K5 {z['k5_detected']}、K8 {z['k8_detected']}。代表点 ({z['representative_x']},{z['representative_y']})、参照{z['actual_selected_k4']}個、対象傷占有 {z['target_current_defect_fraction']:.1%}、選択参照の平均背景 {z['ref_background_mean']:.1%}・対象傷 {z['ref_current_defect_mean']:.1%}。寄与最大帯 {peak['band']} cycles/pixel ({peak['contribution']:.1%})。")
    no=[z for z in rows if z.get('status')=='evaluated' and z['group']=='no_grid']; lines+=['','## 無格子画像とimage18',f'無格子の局所変形は {len(no)} 個、K=4検出 {sum(z["k4_detected"] for z in no)} 個。格子ありとは別のLOIO閾値。image18は3候補以上の地点が0、ラベルなしのため未評価。','','## 解釈と今後','小標本の群中央値・代表窓の違いは記述的な観察であり、格子との位置や混入が検出成否の原因だと断定しない。LOIO最大値は近接窓を多数走査した結果であり、窓を独立サンプルとみなさない。大きいSは検出を保証せず、異常度の絶対値と他画像から決めた閾値の関係を併せて見る。','K=4で得意・不得意な傷の傾向は各表と図から仮説として確認し、対象窓と参照窓の混入、格子方向を保持した特徴、異なる検査画像での閾値安定性を事前に決めた条件で比較する。現在のデータだけで自動参照選択や傷の物理的原因を確定しない。']
    (summary/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__': main()
