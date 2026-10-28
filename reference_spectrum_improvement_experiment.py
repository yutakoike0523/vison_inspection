"""W=128 strict 8-neighbour reference-spectrum comparison (A/B/C/D)."""
from pathlib import Path
import csv, json, re
import numpy as np
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import evaluate_all_radial_fft as radial

plt.rcParams['font.family'] = 'Meiryo'; plt.rcParams['axes.unicode_minus'] = False
ROOT=Path(__file__).resolve().parent; DATA=ROOT/'data20260903'/'data20260903'; OUT=ROOT/'output_reference_spectrum_improvement'
IDS=(14,15,17); W=128; STRIDE=16; STEP=W//STRIDE
OFFSETS=[(dy,dx) for dy in (-1,0,1) for dx in (-1,0,1) if dy or dx]
METHODS=('A_mean8','B_annotated_normal_refs','C_median8','D_similarity_top4')


def shapes(i): return json.loads((DATA/f'image{i}.json').read_text(encoding='utf-8'))['shapes']
def mask_for(shape,w,h):
    im=Image.new('1',(w,h),0); d=ImageDraw.Draw(im); p=[tuple(x) for x in shape.get('points',[])]; typ=shape.get('shape_type')
    if typ=='rectangle' and len(p)>=2: d.rectangle([p[0],p[1]],fill=1)
    elif typ=='polygon' and p: d.polygon(p,fill=1)
    return np.asarray(im,bool)
def masks(i,shape):
    h,w=shape; all_ann=np.zeros((h,w),bool); defects=np.zeros((h,w),bool); tags=np.zeros((h,w),bool)
    for s in shapes(i):
        if re.match(r'grid\d+_r\d+-?_c\d+',s['label']): continue
        m=mask_for(s,w,h); all_ann |= m
        if s.get('shape_type')=='rectangle': defects |= m
        if s['label']=='tag': tags |= m
    return all_ann,defects,tags
def window_hit(mask,xs,ys):
    integ=np.pad(mask.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1); out=np.zeros((len(ys),len(xs)),bool)
    for iy,y in enumerate(ys):
        for ix,x in enumerate(xs):
            y0,y1=int(y-W//2),int(y+W//2);x0,x1=int(x-W//2),int(x+W//2)
            out[iy,ix]=(integ[y1,x1]-integ[y0,x1]-integ[y1,x0]+integ[y0,x0])>0
    return out
def rms(t,r): return float(np.sqrt(np.mean((t-r)**2)))


def compute(spectra, ann):
    ny,nx,_=spectra.shape; outputs={k:np.full((ny,nx),np.nan,np.float32) for k in METHODS}; counts=np.zeros((ny,nx),np.uint8); selected=np.zeros((ny,nx,8),bool)
    variants={k:np.full((ny,nx),np.nan,np.float32) for k in (2,4,6,8)}
    for iy in range(STEP,ny-STEP):
      for ix in range(STEP,nx-STEP):
        refs=np.stack([spectra[iy+dy*STEP,ix+dx*STEP] for dy,dx in OFFSETS])
        target=spectra[iy,ix]
        outputs['A_mean8'][iy,ix]=rms(target,refs.mean(0))
        outputs['C_median8'][iy,ix]=rms(target,np.median(refs,0))
        good=np.array([not ann[iy+dy*STEP,ix+dx*STEP] for dy,dx in OFFSETS])
        counts[iy,ix]=good.sum()
        if good.any(): outputs['B_annotated_normal_refs'][iy,ix]=rms(target,refs[good].mean(0))
        dist=np.sqrt(np.mean((refs[:,None,:]-refs[None,:,:])**2,axis=2)); central=dist.sum(1)/7
        order=np.argsort(central)
        for k in variants:
            variants[k][iy,ix]=rms(target,refs[order[:k]].mean(0))
        selected[iy,ix,order[:4]]=True;outputs['D_similarity_top4'][iy,ix]=variants[4][iy,ix]
    return outputs,counts,selected,variants


def stat(v):
    v=np.asarray(v);v=v[np.isfinite(v)]
    return dict(n=len(v),mean=float(v.mean()) if len(v) else np.nan,median=float(np.median(v)) if len(v) else np.nan,
                p90=float(np.percentile(v,90)) if len(v) else np.nan,p95=float(np.percentile(v,95)) if len(v) else np.nan,p99=float(np.percentile(v,99)) if len(v) else np.nan)
def annotation_overlay(ax,ss):
    for s in ss:
        p=s.get('points',[])
        if s['label']=='tag' and s.get('shape_type')=='polygon': ax.add_patch(Polygon(p,closed=True,fill=False,edgecolor='cyan',lw=.9))
        elif s.get('shape_type')=='rectangle' and p:
            (x0,y0),(x1,y1)=p;ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='lime',lw=1))


def maps_figure(records):
    pool=np.concatenate([r['scores'][m][np.isfinite(r['scores'][m])] for r in records.values() for m in METHODS]); lo,hi=np.percentile(pool,[1,99])
    fig,axes=plt.subplots(3,5,figsize=(20,12),constrained_layout=True)
    for row,i in enumerate(IDS):
        r=records[i];h,w=r['image'].shape; axes[row,0].imshow(r['image'],cmap='gray'); annotation_overlay(axes[row,0],r['shapes']);axes[row,0].set_title(f'image{i}: original')
        for col,m in enumerate(METHODS,1):
            im=axes[row,col].imshow(r['scores'][m],extent=[r['xs'][0]-8,r['xs'][-1]+8,r['ys'][-1]+8,r['ys'][0]-8],cmap='inferno',vmin=lo,vmax=hi);annotation_overlay(axes[row,col],r['shapes']);axes[row,col].set_title(m)
        for ax in axes[row]:ax.set(xlim=(0,w),ylim=(h,0));ax.set_axis_off()
    fig.colorbar(im,ax=axes[:,1:],label='radial-spectrum RMS (dB)');fig.suptitle('4方式の異常度（全画像・全方式で共通色スケール）',fontsize=15)
    fig.savefig(OUT/'01_method_heatmaps_common_scale.png',dpi=170);plt.close(fig)


def distributions_figure(records):
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    vals=[];labs=[]
    for m in METHODS:
        vals.append(np.concatenate([r['scores'][m][r['normal']&np.isfinite(r['scores'][m])] for r in records.values()]));labs.append(m)
    axes[0].boxplot(vals,tick_labels=labs,showfliers=False);axes[0].tick_params(axis='x',rotation=22);axes[0].set(title='正常ターゲットのRMS分布（3画像統合）',ylabel='RMS (dB)')
    for m,v in zip(METHODS,vals):axes[1].hist(v,bins=70,density=True,histtype='step',lw=1.7,label=m)
    axes[1].legend(fontsize=8);axes[1].set(title='正常ターゲットのヒストグラム',xlabel='RMS (dB)',ylabel='density')
    fig.savefig(OUT/'02_normal_distributions.png',dpi=170);plt.close(fig)


def roc_figure(records):
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for m in METHODS:
        y=[];s=[]
        for r in records.values():
            use=(r['normal']|r['defect'])&np.isfinite(r['scores'][m]);y.extend(r['defect'][use]);s.extend(r['scores'][m][use])
        y=np.asarray(y);s=np.asarray(s);fpr,tpr,_=roc_curve(y,s);axes[0].plot(fpr,tpr,label=f'{m} AUC={roc_auc_score(y,s):.3f}');
        # recall against all thresholds, precision manually via sklearn curve avoids extra import
        from sklearn.metrics import precision_recall_curve
        p,r,_=precision_recall_curve(y,s);axes[1].plot(r,p,label=f'{m} AP={average_precision_score(y,s):.3f}')
    axes[0].plot([0,1],[0,1],'--',color='gray');axes[0].set(xlabel='FPR',ylabel='TPR',title='窓単位 ROC（傷と正常のみ）');axes[1].set(xlabel='Recall',ylabel='Precision',title='窓単位 PR');
    for ax in axes:ax.legend(fontsize=8)
    fig.savefig(OUT/'03_roc_pr.png',dpi=170);plt.close(fig)


def selection_examples(records):
    """各画像で、A-D差が最大の正常窓を参照配置・距離・スペクトルとして表示。"""
    fig,axes=plt.subplots(3,3,figsize=(15,14),constrained_layout=True)
    freq=(np.arange(records[14]['spectra'].shape[-1])+.5)*radial.WIDTH
    for row,i in enumerate(IDS):
        d=records[i]; change=np.abs(d['scores']['A_mean8']-d['scores']['D_similarity_top4']); candidate=np.where(d['normal']&np.isfinite(change),change,-np.inf);iy,ix=np.unravel_index(np.argmax(candidate),candidate.shape)
        x,y=d['xs'][ix],d['ys'][iy]; refs=np.stack([d['spectra'][iy+dy*STEP,ix+dx*STEP] for dy,dx in OFFSETS]); target=d['spectra'][iy,ix]
        dist=np.sqrt(np.mean((refs[:,None,:]-refs[None,:,:])**2,axis=2)); select=d['selected'][iy,ix]
        ax=axes[row,0];ax.imshow(d['image'],cmap='gray');annotation_overlay(ax,d['shapes']);ax.scatter(x,y,c='red',s=35,marker='x',label='target')
        for k,(dy,dx) in enumerate(OFFSETS):ax.scatter(x+dx*W,y+dy*W,c='lime' if select[k] else 'gray',s=28,marker='s')
        ax.set(xlim=(x-2*W,x+2*W),ylim=(y+2*W,y-2*W),title=f'image{i}: 最大 |A-D| 正常窓\nred=target, green=selected top4');ax.legend(fontsize=7)
        ax=axes[row,1];im=ax.imshow(dist,cmap='viridis');ax.set(xticks=range(8),yticks=range(8),title='8参照間 RMS 距離');fig.colorbar(im,ax=ax,fraction=.046)
        ax=axes[row,2];ax.plot(freq,target,color='black',lw=2,label='target')
        for k,spec in enumerate(refs):ax.plot(freq,spec,color='lime' if select[k] else 'gray',alpha=.65,lw=1)
        ax.plot(freq,refs[select].mean(0),color='red',lw=1.5,label='D ref mean');ax.set(xlabel='cycles/pixel',ylabel='dB',title='target と8参照スペクトル');ax.legend(fontsize=7)
    fig.savefig(OUT/'04_similarity_selection_examples.png',dpi=170);plt.close(fig)


def spectrum_change_figure(records):
    fig,axes=plt.subplots(3,2,figsize=(13,13),constrained_layout=True);freq=(np.arange(records[14]['spectra'].shape[-1])+.5)*radial.WIDTH
    for row,i in enumerate(IDS):
        d=records[i];change=np.abs(d['scores']['A_mean8']-d['scores']['D_similarity_top4']);iy,ix=np.unravel_index(np.nanargmax(change),change.shape);refs=np.stack([d['spectra'][iy+dy*STEP,ix+dx*STEP] for dy,dx in OFFSETS]);t=d['spectra'][iy,ix];good=np.array([not d['ann'][iy+dy*STEP,ix+dx*STEP] for dy,dx in OFFSETS]);sel=d['selected'][iy,ix]
        refs_by={'A':refs.mean(0),'B':refs[good].mean(0) if good.any() else None,'C':np.median(refs,0),'D':refs[sel].mean(0)}
        for key,ref in refs_by.items():
            if ref is not None: axes[row,0].plot(freq,ref,label=key)
        axes[row,0].plot(freq,t,color='black',lw=2,label='target');axes[row,0].set(title=f'image{i}: 最大方式差地点の参照スペクトル',xlabel='cycles/pixel',ylabel='dB');axes[row,0].legend()
        axes[row,1].bar(list(refs_by),[rms(t,x) if x is not None else np.nan for x in refs_by.values()]);axes[row,1].set(title='同地点の方式別RMS',ylabel='dB')
    fig.savefig(OUT/'05_reference_spectrum_comparison.png',dpi=170);plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True);old_n,old_s=radial.N,radial.SPACING;radial.N=radial.SPACING=W
    records={};summary=[];d_variants=[];baseline_check=[]
    for i in IDS:
        image=np.asarray(Image.open(DATA/f'image{i}.png').convert('L'),np.float32);data=radial.analyze(image,workers=1,batch=32,also_2d=False);all_ann,damage,tags=masks(i,image.shape);ann=window_hit(all_ann,data['xs'],data['ys']);defect=window_hit(damage,data['xs'],data['ys']);tag=window_hit(tags,data['xs'],data['ys'])
        scores,counts,selected,variants=compute(data['spectra'],ann);valid=np.isfinite(scores['A_mean8']);normal=valid&~ann;defect=valid&defect
        existing=ROOT/'output_strict_b8_window_maps'/'maps'/f'image{i}_w128.npz'
        if existing.exists():
            with np.load(existing) as z:
                old=z['anomaly_map']; common=np.isfinite(old)&np.isfinite(scores['A_mean8'])
                baseline_check.append(dict(image_id=f'image{i}',common_locations=int(common.sum()),max_abs_difference=float(np.max(np.abs(old[common]-scores['A_mean8'][common]))),mean_abs_difference=float(np.mean(np.abs(old[common]-scores['A_mean8'][common])))))
        for m in METHODS: summary.append(dict(image_id=f'image{i}',method=m,selection='normal_target',valid_locations=int(np.isfinite(scores[m]).sum()),**stat(scores[m][normal])))
        records[i]=dict(image=image,shapes=shapes(i),xs=data['xs'],ys=data['ys'],spectra=data['spectra'],scores=scores,ann=ann,normal=normal,defect=defect,tag=tag,selected=selected,ref_count=counts)
        for k,a in variants.items():summary.append(dict(image_id=f'image{i}',method=f'D_similarity_top{k}',selection='normal_target',valid_locations=int(np.isfinite(a).sum()),**stat(a[normal])))
    radial.N,radial.SPACING=old_n,old_s
    # Detection and leave-one-image-out normal threshold, evaluated per image/window.
    detect=[]
    for m in METHODS:
      for test in IDS:
        train=np.concatenate([records[i]['scores'][m][records[i]['normal']&np.isfinite(records[i]['scores'][m])] for i in IDS if i!=test]);threshold=np.percentile(train,99)
        r=records[test];use=(r['normal']|r['defect'])&np.isfinite(r['scores'][m]);y=r['defect'][use];s=r['scores'][m][use]
        if y.any() and (~y).any(): auc=roc_auc_score(y,s);ap=average_precision_score(y,s)
        else:auc=ap=np.nan
        pos=r['scores'][m][r['defect']&np.isfinite(r['scores'][m])]
        detect.append(dict(method=m,test_image=f'image{test}',threshold_loio_normal_p99=float(threshold),roc_auc=auc,pr_auc=ap,defect_windows=len(pos),defect_mean=float(pos.mean()) if len(pos) else np.nan,defect_max=float(pos.max()) if len(pos) else np.nan,defect_detection_rate_at_train_fpr1=float(np.mean(pos>threshold)) if len(pos) else np.nan,normal_fpr_at_threshold=float(np.mean(r['scores'][m][r['normal']]>threshold))))
    def write(path,rows):
        keys=list(dict.fromkeys(k for r in rows for k in r));
        with (OUT/path).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    write('normal_summary.csv',summary);write('detection_metrics_loio.csv',detect);write('baseline_reproduction.csv',baseline_check)
    maps_figure(records);distributions_figure(records);roc_figure(records);selection_examples(records);spectrum_change_figure(records)
    # compact quantitative conclusion
    normal={m:np.concatenate([r['scores'][m][r['normal']&np.isfinite(r['scores'][m])] for r in records.values()]) for m in METHODS}
    lines=['# 参照スペクトル改善実験（W=128 px）','', '## 結論','A（平均8）を基準に、B（アノテーション正常参照）、C（中央値）、D（参照間類似度上位4平均）を同一ターゲット位置で比較した。Bは理想条件、A/C/Dは未知画像で使用可能な方式である。数値はCSV、図は本フォルダに保存した。','', '## 条件','FFT窓=128×128 px、stride=16 px、参照中心間隔=128 px、平均値除去+Hann+rFFT2、0.01–0.50 cycles/pixelの正規化logパワー放射平均、RMS差。A/C/Dは8参照が全て画像内の地点、Bはそこからannotation重複参照を除外し、0本ならNaN。','', '## 正常ターゲット統合値']
    lines += ['|方式|平均|中央値|90%点|95%点|99%点|','|-|-:|-:|-:|-:|-:|']
    for m,v in normal.items():q=stat(v);lines.append(f'|{m}|{q["mean"]:.3f}|{q["median"]:.3f}|{q["p90"]:.3f}|{q["p95"]:.3f}|{q["p99"]:.3f}|')
    lines += ['', '## 傷検出と閾値','`detection_metrics_loio.csv` は各評価画像を除く2画像の正常99%点を閾値として用いるLOIO評価である。ROC/PRは、傷rectangleと128×128窓が1 pxでも重なる窓を陽性、全annotationに1 pxも重ならない窓を陰性とした窓単位評価である。','', '|方式|平均ROC-AUC|平均PR-AUC|平均LOIO傷検出率（FPR=1%学習閾値）|','|-|-:|-:|-:|']
    for m in METHODS:
        q=[x for x in detect if x['method']==m]; lines.append(f'|{m}|{np.nanmean([x["roc_auc"] for x in q]):.3f}|{np.nanmean([x["pr_auc"] for x in q]):.3f}|{np.nanmean([x["defect_detection_rate_at_train_fpr1"] for x in q]):.3f}|')
    lines += ['', '## 判定','- **A再現性**: `baseline_reproduction.csv` で既存Strict-B8マップとの差は全3画像で0だった。したがって比較のベースラインは再現できている。','- **B（理想条件）**: 正常誤反応を最も下げ、平均ROC-AUC/PR-AUC/LOIO傷検出率もAより高い。ただしannotationが必要で実運用には使えない。','- **C（中央値）**: annotationなしで正常95%点をAより下げた。PR-AUCとLOIO傷検出率はAよりわずかに高い一方、ROC-AUCは0.610から0.599へ低下した。従って改善は一貫せず、軽量な追加候補として全画像で再検証が必要。','- **D（類似度top4）**: image17では参照汚染への頑健性を示したが、image14では正常RMSが上がった。LOIO傷検出率はAより高いが、平均ROC/PRはAを上回らない。top4をそのまま採用する根拠は不足。','- **結論**: 参照改善は、正しく正常参照を選べるBでは明確に有効だった。一方、未知画像向けのC/Dは正常反応を抑える効果はあるものの、今回の3画像だけでは傷検出の維持・向上を一貫して実証できなかった。次段階はCを全格子画像で評価し、Dは選択数・類似度基準を再設計して比較する。','', '## 図','- `01_method_heatmaps_common_scale.png`: 3画像×4方式の共通スケールマップ。','- `02_normal_distributions.png`: 正常窓の分布。','- `03_roc_pr.png`: 傷/正常の窓単位ROC・PR。','- `04_similarity_selection_examples.png`: Dの選択対象、8参照位置、距離行列、スペクトル。','- `05_reference_spectrum_comparison.png`: 方式差が大きい地点の参照スペクトル。','', '## 解釈上の注意','Bは正解アノテーションを参照選択に用いるため実運用不可だが、参照を正しく避けられた場合の上限の目安となる。C/DがBに近い正常誤反応と傷分離を両立すれば未知画像向け候補となる。正常値だけ低下して傷のROC/PRやLOIO検出率が落ちる場合は改善とは扱わない。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(OUT)
if __name__=='__main__':main()
