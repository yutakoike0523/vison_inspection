#!/usr/bin/env python3
"""Reference-layout and annotation-assisted reference comparison (exploratory)."""
import csv
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import evaluate_all_radial_fft as prev
import multiscale_fft_experiment as base
import stride_grid_overlap_experiment as old

OUT=base.ROOT/'output_reference_selection'
N=256; STRIDE=16; SPACING=128
LAYOUTS={'neighbor8':[(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)],
         'neighbor5':[(-1,0),(1,0),(-1,1),(0,1),(1,1)],
         'bottom3':[(-1,1),(0,1),(1,1)],'sides_bottom3':[(-1,0),(1,0),(0,1)]}

def csvout(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    out=base.unique_dir(OUT); thresholds=old.prior_thresholds(); metrics=[]; refs=[]
    for num in (15,17):
        im=np.asarray(Image.open(base.DATA/f'image{num}.png').convert('L')); labels=prev.annotation(base.DATA/f'image{num}.json',im.shape)
        r=old.load_result(old.OUT/f'image{num}'/'stride_16'/'result.npz'); prev.N=N; prev.STRIDE=STRIDE
        normal,entries=prev.classification(labels,r); defect=next(e for e in entries if e['label']=='局所変形')['center']
        y,x=np.nonzero(defect & np.isfinite(r['scores'])); k=np.argmax(r['scores'][y,x]); ry,rx=int(y[k]),int(x[k])
        xs,ys=r['xs'],r['ys']; spec=r['spectra']; ny,nx=spec.shape[:2]; Y,X=np.meshgrid(ys,xs,indexing='ij'); coords=np.column_stack((X.ravel(),Y.ravel()))
        candidate=np.flatnonzero(normal.ravel()); tree=cKDTree(coords[candidate]); _,near=tree.query(coords,k=min(1024,len(candidate)),workers=1)
        # Fixed distant candidates: at least 512 px from the annotated defect pixels; greedily spread by 256 px.
        mask=next(m for name,m in labels if name=='局所変形'); ydef,xdef=np.nonzero(mask); x0,x1=xdef.min(),xdef.max(); y0,y1=ydef.min(),ydef.max()
        points=coords[candidate]; dx=np.maximum(np.maximum(x0-points[:,0],points[:,0]-x1),0); dy=np.maximum(np.maximum(y0-points[:,1],points[:,1]-y1),0)
        eligible=candidate[np.hypot(dx,dy)>=512]; chosen=[]
        for c in eligible:
            if all(np.hypot(*(coords[c]-coords[p]))>=256 for p in chosen): chosen.append(int(c))
            if len(chosen)==32: break
        print(f'image{num}: normal candidates {len(candidate)}, distant candidates {len(chosen)}',flush=True)
        if len(chosen)<8: raise ValueError('Insufficient distant normal candidates')
        methods=list(LAYOUTS)+['local_clean8','distant_clean8']; maps={m:np.full((ny,nx),np.nan,np.float32) for m in methods}; representative={}
        for iy in range(ny):
            for ix in range(nx):
                target=spec[iy,ix]; row=iy*nx+ix
                for method in methods:
                    if method in LAYOUTS:
                        idx=[((iy+dy*SPACING//STRIDE)*nx+ix+dx*SPACING//STRIDE) for dx,dy in LAYOUTS[method] if 0<=iy+dy*SPACING//STRIDE<ny and 0<=ix+dx*SPACING//STRIDE<nx]
                        if len(idx)<3: continue
                    else:
                        pool=candidate[np.atleast_1d(near[row])] if method=='local_clean8' else chosen
                        # Reject any candidate whose FFT window intersects the target window.
                        idx=[int(j) for j in pool if abs(int(coords[j,0])-int(coords[row,0]))>=N or abs(int(coords[j,1])-int(coords[row,1]))>=N][:8]
                        if method=='local_clean8' and len(idx)<8 and len(candidate)>1024:
                            _,all_near=tree.query(coords[row],k=len(candidate)); pool=candidate[np.atleast_1d(all_near)]
                            idx=[int(j) for j in pool if abs(int(coords[j,0])-int(coords[row,0]))>=N or abs(int(coords[j,1])-int(coords[row,1]))>=N][:8]
                        if len(idx)<8: continue
                    ref=np.median(spec.reshape(-1,spec.shape[-1])[idx],axis=0); score=float(np.sqrt(np.mean((target-ref)**2))); maps[method][iy,ix]=score
                    if iy==ry and ix==rx: representative[method]=(ref,idx,score)
        freq=(np.arange(len(r['bin_valid']))+.5)*.01; freq=freq[r['bin_valid']]; folder=out/f'image{num}'; folder.mkdir()
        np.savez_compressed(folder/'score_maps.npz',x_centers=xs,y_centers=ys,**maps)
        for method in methods:
            scores=maps[method]; n=prev.summary_stats(scores[normal]); d=prev.defect_stats(scores[defect & np.isfinite(scores)],n); yy,xx=np.nonzero(defect & np.isfinite(scores)); kk=np.argmax(scores[yy,xx]); ref,idx,rep=representative[method]
            metrics.append({'image':num,'method':method,'normal_count':n['count'],'normal_median':n['median'],'normal_std':n['std'],'normal_old_threshold_exceed_rate':float(np.mean(scores[normal & np.isfinite(scores)]>thresholds[num])),'defect_count':d['count'],'defect_median':d['median'],'defect_max':d['max'],'defect_max_x':int(xs[xx[kk]]),'defect_max_y':int(ys[yy[kk]]),'S':d['S'],'old_neighbor8_loio_threshold':thresholds[num],'above_old_threshold':bool(d['max']>thresholds[num]),'representative_x':int(xs[rx]),'representative_y':int(ys[ry]),'representative_score':rep,'reference_count':len(idx)})
            fig,ax=plt.subplots(figsize=(9,5));
            for j in idx: ax.plot(freq,spec.reshape(-1,spec.shape[-1])[j],lw=.8,alpha=.5)
            ax.plot(freq,ref,'k--',lw=2,label='Median'); ax.plot(freq,spec[ry,rx],color='red',lw=2,label='Target'); ax.set(title=f'image{num} {method}, score {rep:.3f} dB',xlabel='Frequency (cycles/pixel)',ylabel='Normalized radial power (dB)'); ax.legend(); fig.savefig(folder/f'{method}_spectra.png',dpi=130,bbox_inches='tight'); plt.close(fig)
            for j in idx:
                cx,cy=map(int,coords[j]); overlap='; '.join(f'{name}:{int(mask[cy-N//2:cy+N//2,cx-N//2:cx+N//2].sum())}' for name,mask in labels if mask[cy-N//2:cy+N//2,cx-N//2:cx+N//2].any())
                refs.append({'image':num,'method':method,'target_x':int(xs[rx]),'target_y':int(ys[ry]),'reference_x':cx,'reference_y':cy,'annotation_overlap_pixels':overlap})
            with (folder/f'{method}_representative.csv').open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.writer(f); w.writerow(['frequency','target_db','median_db','difference_db','squared_difference','rms_contribution']+[f'ref_{coords[j,0]}_{coords[j,1]}' for j in idx]); delta=spec[ry,rx]-ref; sq=delta**2; w.writerows(zip(freq,spec[ry,rx],ref,delta,sq,sq/sq.sum(),*[spec.reshape(-1,spec.shape[-1])[j] for j in idx]))
        # Validate our eight-neighbor implementation against the original map.
        a=maps['neighbor8']; b=r['scores']; assert np.allclose(a,b,rtol=1e-5,atol=1e-5,equal_nan=True)
        fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
        for ax,m in zip(axes.flat,methods):
            v=maps[m]; imshow=ax.imshow(v,extent=old.extent(r,STRIDE),cmap='inferno',interpolation='nearest'); ax.contour(mask,levels=[.5],colors='cyan',origin='upper',extent=[0,im.shape[1],im.shape[0],0]); ax.set_title(m); fig.colorbar(imshow,ax=ax,shrink=.7)
        fig.suptitle('Each map uses its own display range; not a threshold'); fig.savefig(folder/'score_maps.png',dpi=130); plt.close(fig)
    csvout(out/'metrics.csv',metrics); csvout(out/'representative_references.csv',refs)
    lines=['# 参照配置・正常参照の探索実験','','image15・17の全画像、256 px窓、stride 16 px、参照間隔128 px。既存の放射平均スペクトルを使用。代表点は従来8参照方式で傷中心の異常度が最大の点。FFT処理・RMS式は不変。周辺配置は境界で欠ける参照を除き最低3点とする。正常スコアも同じ方法で算出し、窓が全ラベルと重ならない地点を正常とした。','','local_clean8はラベルを含まない窓から、対象窓と重ならない最も近い8窓。distant_clean8は傷の外接矩形から512 px以上離れたラベルなし窓を256 px以上離して走査順に最大32点選び、そのうち対象窓と重ならない先頭8点。対象ごとに候補を除外するため完全固定テンプレートではない。両手法は評価画像の正解ラベルを参照に使用する探索的な診断で、未知画像で実行可能な方法の性能ではない。傷から離れた場所でも参照同士や周辺参照と画像構造が異なる可能性がある。','','既存8参照方式のLOIO閾値は他の方法では較正されていないため、超過の有無は参考値に限る。方法ごとの学習画像を使ったLOIO閾値・独立評価は実施していない。図の各マップは個別の色範囲であり、表示色は判定閾値ではない。','','## 結果']
    for z in metrics: lines.append(f"- image{z['image']} {z['method']}: 傷中心中央値 {z['defect_median']:.3f}、最大 {z['defect_max']:.3f}、S {z['S']:.3f}、従来LOIO閾値 {z['old_neighbor8_loio_threshold']:.3f} 超 {z['above_old_threshold']}、正常窓の従来閾値超 {z['normal_old_threshold_exceed_rate']:.2%}、同一代表点 {z['representative_score']:.3f}（参照 {z['reference_count']}）。")
    lines+=['','## 限界','正常ラベルのない部分を正常と仮定。ラベルで参照から異常を除外できるという前提を用いており、推論時には同じ情報を利用できない。Sと異常度は同一画像内で計算した記述統計で、参照配置・参照数・距離・窓の重なりは同時に変化する。これらだけから検出性能の因果関係や未知画像の性能を断定できない。周波数別寄与と参照座標は各CSV、スペクトルは各PNGに保存。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('Output:',out,flush=True)

if __name__=='__main__': main()
