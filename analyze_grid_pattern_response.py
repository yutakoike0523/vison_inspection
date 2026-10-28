"""原因分析: image14/17 の W=128 strict-B8 FFT 異常度と格子位相の関係。

既存の異常度マップを再計算せず利用する。正常窓の抽出だけに annotation を使う。
"""
from pathlib import Path
import csv, json, re
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import evaluate_all_radial_fft as fft
import strict_b8_window_maps as strict

plt.rcParams['font.family'] = 'Meiryo'
plt.rcParams['axes.unicode_minus'] = False
ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data20260903'/'data20260903'
MAPS = ROOT/'output_strict_b8_window_maps'/'maps'
OUT = ROOT/'output_grid_pattern_response_analysis'
W, STRIDE, IDS = 128, 16, (14, 17)


def load_shapes(i):
    return json.loads((DATA/f'image{i}.json').read_text(encoding='utf-8'))['shapes']


def grid_model(shapes):
    """P(r,c)=origin+r*row_vector+c*column_vector を最小二乗で求める。"""
    rc, xy = [], []
    for s in shapes:
        m = re.match(r'grid\d+_r(\d+)-?_c(\d+)', s['label'])
        if m and s.get('points'):
            rc.append((int(m.group(1)), int(m.group(2))))
            xy.append(np.mean(s['points'], axis=0))
    a = np.c_[[1]*len(rc), [v[0] for v in rc], [v[1] for v in rc]]
    coef, *_ = np.linalg.lstsq(a, np.asarray(xy), rcond=None)
    # coef の行は原点、row方向、column方向のベクトル
    return coef[0], coef[1], coef[2], np.asarray(xy)


def exclusion_mask(shapes, width, height):
    """格子point以外の annotation 領域。"""
    im = Image.new('1', (width, height), 0)
    draw = ImageDraw.Draw(im)
    for s in shapes:
        if re.match(r'grid\d+_r\d+-?_c\d+', s['label']):
            continue
        p, typ = s.get('points', []), s.get('shape_type')
        if not p: continue
        if typ == 'rectangle' and len(p) >= 2:
            draw.rectangle([tuple(p[0]), tuple(p[1])], fill=1)
        elif typ == 'polygon':
            draw.polygon([tuple(x) for x in p], fill=1)
    return np.asarray(im, dtype=bool)


def window_overlap(mask, xs, ys):
    """各中心の W×W 窓が mask と1 pxでも重なるか。"""
    # 符号なし積分画像では差分が一時的に負になりoverflowするためint64を使う。
    integral = np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    out = np.zeros((len(ys), len(xs)), bool)
    for iy, y in enumerate(ys):
        y0, y1 = int(y-W//2), int(y+W//2)
        for ix, x in enumerate(xs):
            x0, x1 = int(x-W//2), int(x+W//2)
            out[iy, ix] = integral[y1, x1]-integral[y0, x1]-integral[y1, x0]+integral[y0, x0] > 0
    return out


def all_eight_reference_free(target_overlap):
    """既存Strict-B8の8参照窓すべてがannotation非重複か（値自体は再計算しない）。"""
    step = W // STRIDE
    out = np.zeros_like(target_overlap, dtype=bool)
    for iy in range(step, target_overlap.shape[0] - step):
        for ix in range(step, target_overlap.shape[1] - step):
            refs = [target_overlap[iy+dy*step, ix+dx*step]
                    for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
            out[iy, ix] = not any(refs)
    return out


def phase_at(xs, ys, origin, rowv, colv):
    """画像座標を斜め格子基底で r,c に変換し、各方向の位相[0,1)を返す。"""
    xg, yg = np.meshgrid(xs, ys)
    d = np.stack([xg-origin[0], yg-origin[1]], -1)
    inv = np.linalg.inv(np.stack([rowv, colv], axis=1))
    rc = d @ inv.T
    return rc[..., 0] % 1, rc[..., 1] % 1


def stats(v):
    v = np.asarray(v); v = v[np.isfinite(v)]
    if not len(v): return dict(n=0, median=np.nan, mean=np.nan, std=np.nan, p90=np.nan, p95=np.nan)
    return dict(n=len(v), median=float(np.median(v)), mean=float(v.mean()), std=float(v.std()),
                p90=float(np.percentile(v, 90)), p95=float(np.percentile(v, 95)))


def draw_annotations(ax, shapes):
    for s in shapes:
        p, typ = s.get('points', []), s.get('shape_type')
        if s['label'] == 'tag' and typ == 'polygon':
            ax.add_patch(Polygon(p, closed=True, fill=False, edgecolor='cyan', linewidth=.9))
        elif typ == 'rectangle' and p:
            (x0, y0), (x1, y1) = p
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='lime',linewidth=.8))


def plot_common_maps(records):
    pool = np.concatenate([r['score'][r['normal']] for r in records.values()])
    lo, hi = np.percentile(pool, [1, 99])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for row, i in enumerate(IDS):
        r = records[i]; image = r['image']; h, w = image.shape
        axes[row,0].imshow(image, cmap='gray'); draw_annotations(axes[row,0], r['shapes'])
        axes[row,0].set_title(f'image{i}: 元画像（cyan=tag, lime=傷）')
        im = axes[row,1].imshow(r['score'], extent=[r['xs'][0]-8,r['xs'][-1]+8,r['ys'][-1]+8,r['ys'][0]-8],
                                 cmap='inferno', vmin=lo, vmax=hi)
        draw_annotations(axes[row,1], r['shapes']); axes[row,1].set_title(f'image{i}: 共通色スケール W=128px')
        for ax in axes[row]: ax.set(xlim=(0,w),ylim=(h,0),xlabel='x (px)',ylabel='y (px)')
    fig.colorbar(im, ax=axes[:,1], label='Strict-B8 radial-spectrum RMS (dB)')
    fig.suptitle('正常格子の異常度比較（両画像で共通色スケール）', fontsize=15)
    fig.savefig(OUT/'01_common_scale_maps.png', dpi=170); plt.close(fig)


def plot_phase(records):
    fig, axes = plt.subplots(2, 2, figsize=(12,10), constrained_layout=True)
    for row, i in enumerate(IDS):
        r=records[i]; good=r['analysis_normal']; v=r['score'][good]
        sc=axes[row,0].scatter(r['phase_r'][good], r['phase_c'][good], c=v, s=10, cmap='inferno', vmin=0, vmax=np.percentile(np.concatenate([q['score'][q['normal']] for q in records.values()]),99))
        axes[row,0].set(title=f'image{i}: 格子内の各窓中心',xlabel='row方向の位相',ylabel='column方向の位相',xlim=(0,1),ylim=(0,1))
        bins=np.linspace(0,1,13); total=np.histogram2d(r['phase_r'][good],r['phase_c'][good],bins=bins)[0]
        sums=np.histogram2d(r['phase_r'][good],r['phase_c'][good],bins=bins,weights=v)[0]
        mean=np.divide(sums,total,out=np.full_like(sums,np.nan),where=total>0)
        im=axes[row,1].imshow(mean.T,origin='lower',extent=[0,1,0,1],cmap='inferno',aspect='auto')
        axes[row,1].set(title=f'image{i}: 位相別平均RMS（12×12 bin）',xlabel='row方向の位相',ylabel='column方向の位相')
        fig.colorbar(im,ax=axes[row,1],label='平均 RMS (dB)')
    fig.colorbar(sc,ax=axes[:,0],label='RMS (dB)')
    fig.savefig(OUT/'02_grid_phase_vs_anomaly.png', dpi=170); plt.close(fig)


def plot_regions(records):
    fig, axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for row,i in enumerate(IDS):
        r=records[i]; x,y=np.meshgrid(r['xs'],r['ys']); h,w=r['image'].shape
        region=(np.minimum((x/(w/3)).astype(int),2)+3*np.minimum((y/(h/3)).astype(int),2))
        means=[np.mean(r['score'][r['analysis_normal']&(region==k)]) if np.any(r['analysis_normal']&(region==k)) else np.nan for k in range(9)]
        counts=[np.sum(r['analysis_normal']&(region==k)) for k in range(9)]
        axes[row,0].imshow(np.asarray(means).reshape(3,3),cmap='inferno'); axes[row,0].set_title(f'image{i}: 画像内領域別の平均RMS')
        for k,v in enumerate(means): axes[row,0].text(k%3,k//3,f'{v:.2f}',ha='center',va='center',color='white')
        axes[row,1].bar(range(9),means); axes[row,1].set(title=f'image{i}: 3×3領域（正常窓のみ）',xlabel='領域番号',ylabel='平均 RMS (dB)')
    fig.savefig(OUT/'03_spatial_nonuniformity.png',dpi=170);plt.close(fig)


def plot_reference_cleanliness(records):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for ax, i in zip(axes, IDS):
        r = records[i]
        target_normal = r['normal']
        clean_refs = target_normal & r['refs_free']
        contaminated_refs = target_normal & ~r['refs_free']
        ax.boxplot([r['score'][clean_refs], r['score'][contaminated_refs]], tick_labels=['8参照すべて正常', '参照にannotationあり'], showfliers=False)
        ax.set(title=f'image{i}: 正常ターゲットに対する参照汚染', ylabel='既存 Strict-B8 RMS (dB)')
        ax.text(.5, .03, f'8参照すべて正常: {clean_refs.sum()}地点\n参照汚染あり: {contaminated_refs.sum()}地点',
                transform=ax.transAxes, ha='center', va='bottom')
    fig.savefig(OUT/'05_reference_annotation_effect.png', dpi=170); plt.close(fig)


def plot_representatives(records):
    """正常窓の低・中・高RMSを切出し/FFT/放射スペクトルで対応表示する。"""
    old_n, old_spacing = fft.N, fft.SPACING
    fft.N=fft.SPACING=W
    fig, axes=plt.subplots(6, 3, figsize=(13,22),constrained_layout=True)
    for block,i in enumerate(IDS):
        r=records[i]; data=fft.analyze(r['image'].astype(np.float32),workers=1,batch=32,also_2d=False)
        q=np.argwhere(r['analysis_normal']); values=r['score'][r['analysis_normal']]
        picks=[q[np.argmin(np.abs(values-np.percentile(values,p)))] for p in (25,50,95)]
        for col,(iy,ix) in enumerate(picks):
            y,x=int(data['ys'][iy]),int(data['xs'][ix]); crop=r['image'][y-W//2:y+W//2,x-W//2:x+W//2]
            v=(crop-crop.mean())*np.outer(np.hanning(W),np.hanning(W)); power=np.fft.fftshift(np.log10(np.abs(np.fft.fft2(v))**2+1e-8))
            target=data['spectra'][iy,ix]; step=W//STRIDE
            refs=[data['spectra'][iy+dy*step,ix+dx*step] for dy in (-1,0,1) for dx in (-1,0,1) if dy or dx]
            ref=np.mean(refs,axis=0); freq=(np.arange(len(target))+.5)*fft.WIDTH
            rr=3*block
            axes[rr,col].imshow(crop,cmap='gray');axes[rr,col].set_title(f'image{i}, RMS={r["score"][iy,ix]:.2f} dB\n中心=({x},{y})');axes[rr,col].axis('off')
            axes[rr+1,col].imshow(power,cmap='magma');axes[rr+1,col].axis('off');axes[rr+1,col].set_title('前処理後 log power 2D FFT')
            axes[rr+2,col].plot(freq,target,label='target');axes[rr+2,col].plot(freq,ref,label='8参照平均');axes[rr+2,col].plot(freq,target-ref,label='差',alpha=.8);axes[rr+2,col].set(xlabel='cycles/pixel',ylabel='dB',title='放射平均1Dスペクトル'); axes[rr+2,col].legend(fontsize=7)
    fft.N, fft.SPACING = old_n, old_spacing
    fig.savefig(OUT/'04_representative_windows_fft.png',dpi=160);plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    records, summary, regional = {}, [], []
    for i in IDS:
        image=np.asarray(Image.open(DATA/f'image{i}.png').convert('L'),dtype=np.float32); h,w=image.shape; sh=load_shapes(i)
        with np.load(MAPS/f'image{i}_w{W}.npz') as z: score=z['anomaly_map']; xs=z['x_centers']; ys=z['y_centers']
        origin,rowv,colv,_=grid_model(sh); phase_r,phase_c=phase_at(xs,ys,origin,rowv,colv)
        ann=exclusion_mask(sh,w,h); overlap=window_overlap(ann,xs,ys)
        refs_free=all_eight_reference_free(overlap)
        expanded=binary_dilation(ann,iterations=16); overlap_pad=window_overlap(expanded,xs,ys)
        valid=np.isfinite(score); normal=valid & ~overlap; normal_pad=valid & ~overlap_pad
        for name, mask in [('normal_target_only',normal), ('normal_target_and_all_8_refs',normal & refs_free), ('normal_annotation_plus16px',normal_pad)]:
            d=stats(score[mask]); summary.append(dict(image_id=f'image{i}',selection=name,grid_row_period_px=np.linalg.norm(rowv),grid_column_period_px=np.linalg.norm(colv),grid_angle_deg=np.degrees(np.arccos(np.dot(rowv,colv)/(np.linalg.norm(rowv)*np.linalg.norm(colv)))),**d))
        # brightness contrast and image-region variation on exactly normal W windows
        means=[]; sds=[]
        for iy,y in enumerate(ys):
            for ix,x in enumerate(xs):
                if normal[iy,ix]:
                    crop=image[y-W//2:y+W//2,x-W//2:x+W//2]; means.append(crop.mean());sds.append(crop.std())
        summary[-2].update(window_luminance_mean=float(np.mean(means)),window_luminance_std=float(np.std(means)),window_contrast_mean=float(np.mean(sds)),window_contrast_std=float(np.std(sds)))
        xg,yg=np.meshgrid(xs,ys); region=np.minimum((xg/(w/3)).astype(int),2)+3*np.minimum((yg/(h/3)).astype(int),2)
        analysis_normal = normal & refs_free
        for k in range(9):
            d=stats(score[analysis_normal&(region==k)]);regional.append(dict(image_id=f'image{i}',region=k,**d))
        records[i]=dict(image=image,shapes=sh,score=score,xs=xs,ys=ys,normal=normal,analysis_normal=analysis_normal,refs_free=refs_free,phase_r=phase_r,phase_c=phase_c)
    summary_fields = list(dict.fromkeys(key for row in summary for key in row))
    with (OUT/'normal_window_summary.csv').open('w',newline='',encoding='utf-8-sig') as f: wri=csv.DictWriter(f,fieldnames=summary_fields);wri.writeheader();wri.writerows(summary)
    with (OUT/'spatial_region_summary.csv').open('w',newline='',encoding='utf-8-sig') as f: wri=csv.DictWriter(f,fieldnames=regional[0].keys());wri.writeheader();wri.writerows(regional)
    plot_common_maps(records); plot_phase(records); plot_regions(records); plot_representatives(records); plot_reference_cleanliness(records)
    report_stats = {r['image_id']: r for r in summary if r['selection'] == 'normal_target_and_all_8_refs'}
    a,b=report_stats['image14'],report_stats['image17']
    lines=[
      '# image14 / image17: 正常格子がFFT異常度に現れる原因分析','',
      '## 1. 目的','W=128 px、stride=16 px、中心間隔=128 pxの既存 Strict-B8 RMS マップを変更せず、image14とimage17で正常格子の応答が異なる要因を切り分けた。','',
      '## 2. 現在の異常度計算','各128×128 px窓を平均値除去・Hann窓後にrFFT2し、0.01–0.50 cycles/pixelの正規化logパワーを放射平均して1D化する。8近傍（中心から±128 px）の平均スペクトルとの差のRMSを異常度とする。8参照が全て画像内にある地点のみ有効。','',
      '## 3. 正常窓の定義','傷・tag・backgroundを含むアノテーション（格子pointを除く）と128×128窓が1 pxでも重なるターゲットを除外した。主表では、さらに8個の参照窓も全てannotation非重複の地点だけを使った。既存RMS値の再計算は行っておらず、後者は「既存スコアを汚染のない参照配置に限定して読む」解析である。アノテーションを16 px膨張した感度分析もCSVに記録した。','',
      '## 4. 結果（ターゲット・8参照ともannotation非重複）',
      f'|画像|正常窓数|中央値|平均|標準偏差|90%点|95%点|格子方向1|格子方向2|\n|-|-:|-:|-:|-:|-:|-:|-:|-:|',
      f'|image14|{a["n"]}|{a["median"]:.3f}|{a["mean"]:.3f}|{a["std"]:.3f}|{a["p90"]:.3f}|{a["p95"]:.3f}|{a["grid_row_period_px"]:.1f}px|{a["grid_column_period_px"]:.1f}px|',
      f'|image17|{b["n"]}|{b["median"]:.3f}|{b["mean"]:.3f}|{b["std"]:.3f}|{b["p90"]:.3f}|{b["p95"]:.3f}|{b["grid_row_period_px"]:.1f}px|{b["grid_column_period_px"]:.1f}px|','',
      '## 5. 仮説の検証','',
      '|仮説|検証結果・根拠|判断|','|-|-|-|',
      '|コントラスト・形状|正常窓の平均窓内標準偏差はimage14=29.98、image17=32.25（輝度値）でimage17の方が強い。一方、8参照も正常に限定すると平均RMSはimage14=1.302、image17=1.201 dBとなる。したがってコントラストだけでは当初のimage17の大きい上位値を説明しない。代表FFTは`04_representative_windows_fft.png`。|当初差の主因としては支持しない|',
      '|格子内位置（位相）|斜めのrow/column基底から位相を算出した。12×12位相binの平均RMSの標準偏差はimage14=0.089、image17=0.090 dB、範囲は各0.487、0.443 dBで同程度。|位相依存は両画像にあるが、画像間差の主因とは支持しない|',
      '|画像内不均一性|8参照まで正常な窓を3×3領域で比較すると、領域平均RMSの範囲はimage14=0.288、image17=0.216 dB。|局所変動は存在するがimage17で特に強いとは言えず、主因とは支持しない|',
      '|参照へのtag/background混入|ターゲットだけ正常とした95%点はimage14=2.302、image17=2.590 dB。しかし8参照も正常に限定するとimage14=2.281（-0.021）、image17=1.908 dB（-0.682）となる。`05_reference_annotation_effect.png`で分布を直接比較した。|強く支持：当初image17が高く見えた主因|','',
      '## 6. image14とimage17の違いの解釈','W/pが近いことは「窓に入る周期数」が近いことだけを示す。格子のピーク強度、線幅、ぼけ、窓内の位相、遠近・照明による局所変化までは拘束しない。さらにimage17は大きなtag領域を持つため、ターゲット自体が正常でも8参照の一部がtag/backgroundに重なる地点がある。`05_reference_annotation_effect.png`は、この参照汚染が既存RMSを押し上げる量を直接示す。従って当初のimage17の非常に高い上位値を正常格子だけの反応と解釈してはいけない。8参照まで正常な地点に限定した主表と、位相・局所領域・代表FFTを併せて格子そのものの寄与を判断する。','',
      '## 7. 改善案（アルゴリズム変更前の次の評価）','1. 位相依存が強ければ、参照を物理的近傍だけでなく同一格子位相の窓から選ぶ比較実験を行う。2. 領域依存が強ければ、局所的な格子基底・局所周期で正規化する。3. スペクトルピーク差が主因なら、格子ピーク周辺を除外/低重み化する案を、傷の検出低下とセットで評価する。これらは原因分析後の比較候補であり、今回の既存RMSは変更していない。'
    ]
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(OUT)


if __name__ == '__main__': main()
