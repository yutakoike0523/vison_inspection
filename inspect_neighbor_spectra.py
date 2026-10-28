#!/usr/bin/env python3
"""Inspect the individual reference spectra at the maximum defect-center score."""
import csv
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import evaluate_all_radial_fft as prev
import multiscale_fft_experiment as base
import stride_grid_overlap_experiment as stride

OUT=base.ROOT/'output_neighbor_spectra_inspection'
N=256
STEP=4
SPACING=128

def run():
    output=base.unique_dir(OUT)
    summary=[]
    for number in (15,17):
        path=base.DATA/f'image{number}.png'
        img=np.asarray(Image.open(path).convert('L'))
        labels=prev.annotation(path.with_suffix('.json'),img.shape)
        masks=[mask for name,mask in labels if name=='局所変形']
        if len(masks)!=1: raise ValueError('Expected one local-deformation annotation')
        r=stride.load_result(stride.OUT/f'image{number}'/'stride_4'/'result.npz')
        center=base.selections(masks[0],r,N)[0] & np.isfinite(r['scores'])
        yy,xx=np.nonzero(center); k=np.argmax(r['scores'][yy,xx]); y,x=int(yy[k]),int(xx[k])
        target=r['spectra'][y,x]; X=int(r['xs'][x]); Y=int(r['ys'][y]); freq=(np.arange(len(r['bin_valid']))+.5)*.01; freq=freq[r['bin_valid']]
        neighbors=[]
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if not (dx or dy): continue
                iy=y+dy*(SPACING//STEP); ix=x+dx*(SPACING//STEP)
                if not (0<=iy<len(r['ys']) and 0<=ix<len(r['xs'])): continue
                px,py=int(r['xs'][ix]),int(r['ys'][iy]); x0,y0=px-N//2,py-N//2
                overlaps={name:int(mask[y0:y0+N,x0:x0+N].sum()) for name,mask in labels}
                neighbors.append(dict(id=f'{len(neighbors)+1}',x=px,y=py,dx=dx,dy=dy,spectrum=r['spectra'][iy,ix],overlaps=overlaps))
        values=np.stack([n['spectrum'] for n in neighbors]); median=np.median(values,axis=0)
        recomputed=float(np.sqrt(np.mean((target-median)**2))); saved=float(r['scores'][y,x])
        fig,(ax,ax2)=plt.subplots(1,2,figsize=(16,6),constrained_layout=True)
        colors=plt.cm.tab10(np.linspace(0,.8,len(neighbors)))
        for n,col in zip(neighbors,colors):
            ax.plot(freq,n['spectrum'],color=col,alpha=.7,lw=1,label=f"{n['id']}: ({n['x']},{n['y']})")
        ax.plot(freq,median,'k--',lw=2.3,label='Reference median'); ax.plot(freq,target,color='red',lw=2.2,label='Target')
        ax.set(xlabel='Radial frequency (cycles/pixel)',ylabel='Normalized radial power (dB)',title=f'image{number}: target ({X},{Y}), {len(neighbors)} references'); ax.legend(fontsize=7,ncol=2)
        ax2.imshow(img,cmap='gray',vmin=0,vmax=255,extent=[0,img.shape[1],img.shape[0],0]); ax2.contour(masks[0],levels=[.5],colors='cyan',origin='upper',extent=[0,img.shape[1],img.shape[0],0],linewidths=1)
        ax2.add_patch(Rectangle((X-N//2,Y-N//2),N,N,fill=False,edgecolor='red',linewidth=2)); ax2.text(X,Y,'T',color='red',weight='bold')
        for n,col in zip(neighbors,colors):
            ax2.add_patch(Rectangle((n['x']-N//2,n['y']-N//2),N,N,fill=False,edgecolor=col,linewidth=1.5)); ax2.text(n['x'],n['y'],n['id'],color=col,weight='bold',fontsize=11)
        ax2.set(xlim=(max(0,X-270),min(img.shape[1],X+270)),ylim=(min(img.shape[0],Y+270),max(0,Y-270)),title='Target and reference windows (cyan: deformation)')
        fig.savefig(output/f'image{number}_spectra_and_windows.png',dpi=160); plt.close(fig)
        with (output/f'image{number}_spectra.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f); writer.writerow(['frequency_cycles_per_pixel','target_db','median_db']+[f"reference_{n['id']}_db" for n in neighbors]); writer.writerows(zip(freq,target,median,*[n['spectrum'] for n in neighbors]))
        with (output/f'image{number}_references.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f); writer.writerow(['id','x','y','rms_to_median_db','annotation_overlap_pixels']);
            for n in neighbors: writer.writerow([n['id'],n['x'],n['y'],float(np.sqrt(np.mean((n['spectrum']-median)**2))),'; '.join(f'{key}:{value}' for key,value in n['overlaps'].items() if value)])
        outside=float(np.mean((target<values.min(axis=0)) | (target>values.max(axis=0))))
        summary.append((number,X,Y,len(neighbors),saved,recomputed,outside,neighbors))
    lines=['# 参照スペクトルの個別確認','','256 px窓、stride 4 px、参照間隔128 px。各画像で局所変形ラベル内の中心点のうち異常度最大の点を使用。既存の放射平均スペクトルを再利用し、周辺を成分ごとに中央値化する前の全参照を表示。窓が画像外に出る参照は存在しないため除く。スペクトルの縦軸は従来の正規化放射パワーのdB値。','','## 結果']
    for number,x,y,count,score,check,outside,neighbors in summary:
        lines.append(f'- image{number}: 対象 ({x}, {y})、参照 {count}本、異常度 {score:.4f} dB（再計算 {check:.4f} dB）。対象が参照の最小〜最大の範囲外にある周波数ビンの割合 {outside:.1%}。')
        for n in neighbors:
            overlap=', '.join(f'{name} {pixels}px' for name,pixels in n['overlaps'].items() if pixels) or 'なし'
            lines.append(f"  - 参照{n['id']} ({n['x']}, {n['y']}): 中央値との差RMS {np.sqrt(np.mean((n['spectrum']-np.median(np.stack([v['spectrum'] for v in neighbors]),axis=0))**2)):.3f} dB、ラベルとの重複: {overlap}。")
    lines+=['','## 解釈上の注意','参照の最小〜最大範囲は正常域や信頼区間ではない。参照の中にも傷・札・背景や未ラベルの構造が含まれ得る。窓が重なるため互いに独立でもない。対象が範囲外のビン数や参照の中央値からの距離だけでは、正常基準としての妥当性や検出性能の因果関係は確定できない。各参照の領域とスペクトルを図・CSVで確認し、除外や別の参照選択方法は別実験で評価する。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(output)

if __name__=='__main__': run()
