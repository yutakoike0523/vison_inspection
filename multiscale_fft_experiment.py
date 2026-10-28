#!/usr/bin/env python3
"""Exact neighboring-spectrum experiment and FFT worker benchmark."""
import argparse
import csv
import json
import os
import platform
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import psutil
from PIL import Image, ImageDraw
from scipy import fft
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data20260903' / 'data20260903'
OUT = ROOT / 'output_multiscale_fft'
STRIDE, SPACING, BATCH, EPS = 16, 128, 64, np.float32(1e-12)
FMIN, FMAX = .02, .50

for font in ['Yu Gothic', 'Meiryo', 'Noto Sans CJK JP']:
    if font in {f.name for f in fm.fontManager.ttflist}:
        plt.rcParams['font.family'] = font
        break
plt.rcParams['axes.unicode_minus'] = False


def unique_dir(path):
    if path.exists():
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = path.with_name(path.name + '_' + stamp)
        i = 1
        while path.exists():
            path = path.with_name(path.name + f'_{i}')
            i += 1
    path.mkdir(parents=True)
    return path


def csv_write(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8-sig')
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def analyze(img, size, workers, batch=BATCH):
    """Same operations and float32 power normalization as original; disk-backed spectra."""
    begin = time.perf_counter()
    h, w = img.shape
    ys = np.arange(0, h-size+1, STRIDE); xs = np.arange(0, w-size+1, STRIDE)
    if not len(xs) or not len(ys):
        raise ValueError('Image is smaller than the FFT window')
    fy = np.fft.fftshift(np.fft.fftfreq(size)).astype(np.float32)
    fx = np.fft.rfftfreq(size).astype(np.float32)
    gy, gx = np.meshgrid(fy, fx, indexing='ij')
    mask = (np.sqrt(gx**2+gy**2) >= FMIN) & (np.sqrt(gx**2+gy**2) <= FMAX)
    hann = np.hanning(size).astype(np.float32); hann = np.outer(hann, hann).astype(np.float32)
    fft_time = 0.0
    fft_sample = None
    with tempfile.TemporaryDirectory(prefix='neighbor_fft_') as temp:
        spectra = np.memmap(Path(temp)/'spectra.dat', dtype=np.float32, mode='w+',
                            shape=(len(ys),len(xs),size,size//2+1))
        start = time.perf_counter()
        for iy, y in enumerate(ys):
            windows = np.moveaxis(sliding_window_view(img[y:y+size], size, axis=1), 1, 0)[::STRIDE]
            for x in range(0, len(xs), batch):
                v = np.asarray(windows[x:x+batch], dtype=np.float32).copy()
                v -= v.mean(axis=(-2,-1), keepdims=True); v *= hann[None]
                t = time.perf_counter(); s = fft.rfft2(v, axes=(-2,-1), workers=workers); fft_time += time.perf_counter()-t
                if fft_sample is None: fft_sample = s[0].copy()
                s = fft.fftshift(s, axes=(-2,)); power = (np.abs(s)**2).astype(np.float32)
                total = np.maximum(power[:,mask].sum(axis=1,keepdims=True), EPS)
                power /= total[:,:,None]
                spectra[iy,x:x+batch] = (10.0*np.log10(power+EPS)).astype(np.float32)
        feature_time = time.perf_counter()-start
        start = time.perf_counter()
        scores = np.full((len(ys),len(xs)), np.nan, np.float32)
        step = SPACING//STRIDE
        offsets = [(dy*step,dx*step) for dy in (-1,0,1) for dx in (-1,0,1) if dy or dx]
        best = -np.inf; info = None
        # Per target: no global neighbor stack; same NumPy median and RMS as source.
        for iy in range(len(ys)):
            for ix in range(len(xs)):
                neighbors = [spectra[iy+dy,ix+dx] for dy,dx in offsets
                             if 0 <= iy+dy < len(ys) and 0 <= ix+dx < len(xs)]
                if not neighbors: continue
                ref = np.median(np.stack(neighbors),axis=0)
                diff = spectra[iy,ix] - ref
                score = np.sqrt(np.mean(diff[mask]**2)); scores[iy,ix] = score
                if score > best:
                    best = score
                    info = (iy,ix,spectra[iy,ix].copy(),ref.copy(),diff.copy(),len(neighbors))
        compare_time = time.perf_counter()-start
        del neighbors, spectra
    total_time = time.perf_counter()-begin
    return dict(scores=scores, xs=xs+size//2, ys=ys+size//2, info=info, fx=fx, fy=fy, fft_sample=fft_sample,
                fft_seconds=fft_time, feature_seconds=feature_time, compare_seconds=compare_time,
                total_seconds=total_time, peak_memory_bytes=psutil.Process().memory_info().peak_wset if os.name == 'nt' else None)


def shape_mask(shape, width, height):
    canvas = Image.new('1',(width,height)); draw = ImageDraw.Draw(canvas)
    points = [tuple(map(float,p)) for p in shape['points']]
    typ = shape.get('shape_type') or 'polygon'
    if typ == 'rectangle':
        (x1,y1),(x2,y2)=points; draw.rectangle((min(x1,x2),min(y1,y2),max(x1,x2),max(y1,y2)), fill=1)
    elif typ == 'polygon': draw.polygon(points, fill=1)
    elif typ == 'circle':
        (x,y),(a,b)=points; r=((a-x)**2+(b-y)**2)**.5; draw.ellipse((x-r,y-r,x+r,y+r),fill=1)
    elif typ == 'point': draw.ellipse((points[0][0]-5,points[0][1]-5,points[0][0]+5,points[0][1]+5),fill=1)
    elif typ in ('line','linestrip'): draw.line(points, fill=1, width=10)
    else: raise ValueError(f'Unsupported Labelme shape type: {typ}')
    return np.asarray(canvas,dtype=bool)


def load_labels(path, shape):
    if not path.exists(): return None
    data = json.loads(path.read_text(encoding='utf-8'))
    h,w=shape
    if data.get('imageWidth') != w or data.get('imageHeight') != h:
        raise ValueError('Labelme image dimensions do not match input image')
    return [(s['label'],shape_mask(s,w,h)) for s in data['shapes']]


def selections(mask, result, size):
    xs,ys=result['xs'],result['ys']; yy,xx=np.meshgrid(ys,xs,indexing='ij')
    center=mask[yy,xx]
    integ=np.pad(mask.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
    y=yy-size//2; x=xx-size//2
    overlap=(integ[y+size,x+size]-integ[y,x+size]-integ[y+size,x]+integ[y,x])>0
    return center,overlap


def summarize(values):
    values=np.asarray(values); values=values[np.isfinite(values)]
    if not len(values): return (float('nan'),)*3
    return float(np.max(values)),float(np.median(values)),float(np.percentile(values,95))


def extent(r):
    return [r['xs'][0]-STRIDE/2,r['xs'][-1]+STRIDE/2,r['ys'][-1]+STRIDE/2,r['ys'][0]-STRIDE/2]


def plot_single(img, r, folder):
    a=r['scores']; ext=extent(r)
    for overlay,name in [(False,'anomaly_map.png'),(True,'anomaly_overlay.png')]:
        fig,ax=plt.subplots(figsize=(10,7))
        if overlay: ax.imshow(img,cmap='gray',extent=[0,img.shape[1],img.shape[0],0])
        im=ax.imshow(a,cmap='inferno',extent=ext,interpolation='nearest',alpha=.55 if overlay else 1)
        ax.set(xlabel='x (px)',ylabel='y (px)',xlim=(0,img.shape[1]),ylim=(img.shape[0],0))
        fig.colorbar(im,ax=ax,label='RMS difference (dB)'); fig.savefig(folder/name,dpi=150,bbox_inches='tight'); plt.close(fig)
    iy,ix,target,ref,diff,count=r['info']; fig,axes=plt.subplots(1,3,figsize=(15,4))
    common=np.concatenate([target.ravel(),ref.ravel()]); lo,hi=np.percentile(common,[5,99.5])
    ex=[r['fx'][0],r['fx'][-1],r['fy'][0],r['fy'][-1]]
    for ax,data,title in zip(axes,[target,ref,np.abs(diff)],['Target',f'Reference median (n={count})','Absolute difference']):
        im=ax.imshow(data,origin='lower',extent=ex,aspect='auto',vmin=lo if title!='Absolute difference' else 0,vmax=hi if title!='Absolute difference' else None)
        ax.set(title=title,xlabel='fx (cycles/pixel)',ylabel='fy (cycles/pixel)'); fig.colorbar(im,ax=ax)
    fig.suptitle(f"Maximum at ({r['xs'][ix]}, {r['ys'][iy]})"); fig.savefig(folder/'spectrum_comparison.png',dpi=150,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(); ax.hist(a[np.isfinite(a)],bins=60); ax.set(xlabel='RMS difference (dB)',ylabel='Number of windows'); fig.savefig(folder/'histogram.png',dpi=150,bbox_inches='tight'); plt.close(fig)


def plot_comparison(results,img,folder,normalized):
    xmin=max(r['xs'][0] for r in results.values()); xmax=min(r['xs'][-1] for r in results.values())
    ymin=max(r['ys'][0] for r in results.values()); ymax=min(r['ys'][-1] for r in results.values())
    arrays=[]
    for r in results.values():
        a=r['scores']; arrays.append(a[np.ix_((r['ys']>=ymin)&(r['ys']<=ymax),(r['xs']>=xmin)&(r['xs']<=xmax))])
    if normalized:
        arrays=[(a-np.nanmedian(a))/np.nanstd(a) if np.nanstd(a)>0 else a*0 for a in arrays]
    lo=min(np.nanmin(a) for a in arrays); hi=max(np.nanpercentile(a,99) for a in arrays)
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    for ax,(size,r),a in zip(axes.flat,results.items(),arrays):
        im=ax.imshow(a,extent=[xmin-8,xmax+8,ymax+8,ymin-8],cmap='inferno',vmin=lo,vmax=hi,interpolation='nearest')
        ax.set(title=f'{size} px',xlabel='x (px)',ylabel='y (px)')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='(score - common median) / common std' if normalized else 'RMS difference (dB)',shrink=.8)
    fig.suptitle('Common center region; shared color scale (upper limit: largest 99th percentile)')
    fig.savefig(folder/('normalized_maps_comparison.png' if normalized else 'anomaly_maps_comparison.png'),dpi=160); plt.close(fig)


def multiscale(args,img,labels):
    out=unique_dir(args.output/args.image.stem); comp=out/'comparison'; comp.mkdir()
    results={}; metrics=[]; defects=[]
    all_label = np.logical_or.reduce([m for _,m in labels]) if labels else np.zeros(img.shape,bool)
    for size in args.windows:
        print(f'Window {size}: analyzing...',flush=True); r=analyze(img,size,args.workers,args.batch_size); results[size]=r
        folder=out/f'window_{size}'; folder.mkdir(); plot_single(img,r,folder)
        a=r['scores']; iy,ix=r['info'][:2]; _,normal_excluded=selections(all_label,r,size); normal=a[~normal_excluded & np.isfinite(a)]
        row=dict(window=size,frequency_resolution=1/size,count=int(np.isfinite(a).sum()),mean=float(np.nanmean(a)),median=float(np.nanmedian(a)),std=float(np.nanstd(a)),maximum=float(np.nanmax(a)),max_x=int(r['xs'][ix]),max_y=int(r['ys'][iy]),p95=float(np.nanpercentile(a,95)),p99=float(np.nanpercentile(a,99)),normal_count=len(normal),normal_median=float(np.median(normal)) if len(normal) and labels else float('nan'),normal_std=float(np.std(normal)) if len(normal) and labels else float('nan'),fft_seconds=r['fft_seconds'],feature_seconds=r['feature_seconds'],comparison_seconds=r['compare_seconds'],analysis_seconds=r['total_seconds'],process_peak_memory_bytes=r['peak_memory_bytes'])
        row['normal_p95'] = float(np.percentile(normal,95)) if len(normal) and labels else float('nan')
        row['normal_p99'] = float(np.percentile(normal,99)) if len(normal) and labels else float('nan')
        metrics.append(row)
        for index,(label,mask) in enumerate(labels or []):
            if label != '局所変形': continue
            center,overlap=selections(mask,r,size)
            for mode,selected in [('center',center),('overlap',overlap)]:
                vals=a[selected & np.isfinite(a)]; maximum,median,p95=summarize(vals)
                yy,xx=np.nonzero(selected & np.isfinite(a)); pos=int(np.argmax(vals)) if len(vals) else None
                sd=np.std(normal) if len(normal) else 0
                defects.append(dict(window=size,annotation_index=index,label=label,selection=mode,count=len(vals),maximum=maximum,median=median,p95=p95,max_x=int(r['xs'][xx[pos]]) if pos is not None else '',max_y=int(r['ys'][yy[pos]]) if pos is not None else '',normal_median=float(np.median(normal)) if len(normal) else float('nan'),normal_std=float(sd) if len(normal) else float('nan'),separation=(median-np.median(normal))/sd if sd>0 and len(vals) else float('nan')))
        np.savez_compressed(folder/'result.npz',anomaly_map=a,x_centers=r['xs'],y_centers=r['ys'],window_size=size,stride=STRIDE,reference_spacing=SPACING,freq_min=FMIN,freq_max=FMAX)
        print(f'Window {size}: {r["total_seconds"]:.1f}s',flush=True)
    csv_write(comp/'metrics.csv',metrics); csv_write(comp/'defect_metrics.csv',defects)
    plot_comparison(results,img,comp,False); plot_comparison(results,img,comp,True)
    lines=['# 局所FFT窓サイズ比較',f'画像: {args.image.name}（{img.shape[1]}×{img.shape[0]} px）', '', '## 条件',f'窓: {args.windows} px、stride {STRIDE} px、参照3×3（中央除外、最大8）、間隔 {SPACING} px、周波数 {FMIN}–{FMAX} cycles/pixel。平滑化・マスク除外なし。', '周波数分解能（cycles/pixel）: '+', '.join(f'{m["window"]}: {m["frequency_resolution"]:.6f}' for m in metrics), '比較図は全窓に共通の中心座標範囲。正規化図は窓ごとに共通範囲の (値−中央値)/標準偏差。色の上限は各図の99パーセンタイル最大値であり、傷の判定閾値ではない。', '正常表面は未ラベル領域を仮定し、窓がいずれかのアノテーションに重なる場合は除外。中心・重なりの各傷評価は正常窓とは別に計算。', '', '## 結果']
    for m in metrics:
        lines.append(f"- {m['window']} px: 全域中央値 {m['median']:.3f} dB、95% {m['p95']:.3f} dB、最大 {m['maximum']:.3f} dB ({m['max_x']}, {m['max_y']})、正常中央値 {m['normal_median']:.3f} dB、正常標準偏差 {m['normal_std']:.3f} dB。解析 {m['analysis_seconds']:.1f} 秒。")
    lines += ['', '## 局所変形']
    if defects:
        for d in defects: lines.append(f"- {d['window']} px / {d['selection']}: 窓数 {d['count']}、中央値 {d['median']:.3f}、95% {d['p95']:.3f}、最大 {d['maximum']:.3f} dB ({d['max_x']}, {d['max_y']})、S={d['separation']:.3f}。")
        center=[d for d in defects if d['selection']=='center' and np.isfinite(d['separation'])]
        if center:
            best=max(center,key=lambda d:d['separation']); lines.append(f"中心基準のS最大は {best['window']} px（{best['separation']:.3f}）。1画像・1箇所であり、汎化性能やサイズ別の最適条件は判断できない。")
            for d in center:
                m=next(item for item in metrics if item['window']==d['window'])
                lines.append(f"- {d['window']} px: 正常95% {m['normal_p95']:.3f}、正常99% {m['normal_p99']:.3f} dB。局所変形の中心窓の最大値はそれぞれとの差 {d['maximum']-m['normal_p95']:+.3f}、{d['maximum']-m['normal_p99']:+.3f} dB。")
    else: lines.append('局所変形のアノテーションがないため検出精度は評価しない。')
    lines += ['', '## 考察と今後', '窓サイズごとに正常標準偏差と傷領域の中央値・上位値・最大位置を参照。正常標準偏差には撮影条件や領域の違いも含まれ、スペクトル自体の安定性を単独で証明しない。局所変形が1箇所の場合、大きさ別の優劣は判断できない。', '参照間隔128 pxに対し192・256 pxでは対象窓と隣接参照窓が重複し、同じ変形が双方に入ると差が弱まる可能性がある。周波数刻みも窓ごとに変わるため、窓サイズだけでなく評価周波数ビンも変わる。', '複数画像・複数の大きさの変形で再評価し、中心と重なりの結果を併せて確認してからマルチスケール統合を検討する。', '最大メモリはWindowsのプロセス生涯ピーク作業セットで、窓ごとにリセットされない。画像読込・図保存を除き、解析時間には特徴抽出、周辺比較、memmap作成・削除を含む。']
    (comp/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Output:',out,flush=True)
    return out


def benchmark(args,img):
    out=unique_dir(args.output/'benchmark'); logical=psutil.cpu_count(logical=True) or 1
    workers=sorted(set([n for n in [1,2,4,8,16,logical] if n<=logical]))
    cpu=platform.processor()
    if os.name=='nt':
        try:
            cpu=subprocess.check_output(['powershell','-NoProfile','-Command','(Get-CimInstance Win32_Processor).Name'],text=True,encoding='utf-8',errors='replace').strip()
        except Exception: pass
    info=dict(model=cpu,physical_cores=psutil.cpu_count(logical=False),logical_cpus=logical,platform=platform.platform(),batch_size=args.batch_size,windows=args.windows,workers=workers,repeats=args.repeats)
    (out/'cpu_information.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]; consistency=[]
    for size in args.windows:
        print(f'Benchmark warmup: window {size}',flush=True); analyze(img,size,1,args.batch_size)
        baseline=None; baseline_fft=None
        for n in workers:
            for trial in range(1,args.repeats+1):
                print(f'Benchmark window {size}, workers {n}, trial {trial}/{args.repeats}',flush=True)
                r=analyze(img,size,n,args.batch_size)
                if baseline is None: baseline=r['scores'].copy(); baseline_fft=r['fft_sample'].copy()
                error=float(np.nanmax(np.abs(r['scores']-baseline)))
                consistent=bool(np.allclose(r['scores'],baseline,rtol=1e-5,atol=1e-5,equal_nan=True))
                fft_error=float(np.max(np.abs(r['fft_sample']-baseline_fft)))
                fft_match=bool(np.allclose(r['fft_sample'],baseline_fft,rtol=1e-5,atol=1e-5))
                consistency.append(consistent and fft_match)
                rows.append(dict(window=size,stride=STRIDE,workers=n,trial=trial,fft_seconds=r['fft_seconds'],feature_seconds=r['feature_seconds'],comparison_seconds=r['compare_seconds'],analysis_seconds=r['total_seconds'],process_peak_memory_bytes=r['peak_memory_bytes'],max_score_abs_difference=error,maps_match=consistent,fft_sample_max_abs_difference=fft_error,fft_sample_match=fft_match))
                csv_write(out/'benchmark_results.csv',rows)
    def avg(size,n,key): return np.mean([r[key] for r in rows if r['window']==size and r['workers']==n])
    def std(size,n,key): return np.std([r[key] for r in rows if r['window']==size and r['workers']==n])
    size=128 if 128 in args.windows else args.windows[0]
    fig,ax=plt.subplots()
    for key,label in [('fft_seconds','FFT rfft2'),('analysis_seconds','Total analysis')]:
        ax.errorbar(workers,[avg(size,n,key) for n in workers],yerr=[std(size,n,key) for n in workers],marker='o',capsize=3,label=label)
    ax.set(xlabel='FFT workers',ylabel='Time (seconds)',title=f'Window {size} px'); ax.legend(); fig.savefig(out/'processing_time.png',dpi=150,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots()
    for key,label in [('fft_seconds','FFT rfft2'),('analysis_seconds','Total analysis')]:
        ax.plot(workers,[avg(size,1,key)/avg(size,n,key) for n in workers],marker='o',label=label)
    ax.plot(workers,workers,'--',label='Ideal linear'); ax.set(xlabel='FFT workers',ylabel='Speedup vs 1 worker'); ax.legend(); fig.savefig(out/'speedup.png',dpi=150,bbox_inches='tight'); plt.close(fig)
    if len(args.windows)>1:
        fig,ax=plt.subplots()
        for size2 in args.windows: ax.errorbar(workers,[avg(size2,n,'fft_seconds') for n in workers],yerr=[std(size2,n,'fft_seconds') for n in workers],marker='o',capsize=3,label=f'{size2} px')
        ax.set(xlabel='FFT workers',ylabel='FFT time (seconds)'); ax.legend(); fig.savefig(out/'window_size_benchmark.png',dpi=150,bbox_inches='tight'); plt.close(fig)
    best=min(workers,key=lambda n:avg(size,n,'analysis_seconds')); fastest_fft=min(workers,key=lambda n:avg(size,n,'fft_seconds'))
    tolerance=std(size,best,'analysis_seconds')
    recommended=min(n for n in workers if avg(size,n,'analysis_seconds') <= avg(size,best,'analysis_seconds')+tolerance)
    lines=['# FFTワーカー数の実測',f"CPU: {cpu}、物理 {info['physical_cores']} コア、論理 {logical} CPU。FFT workersはSciPyが使用するワーカー数で、物理コア数と同義ではない。",f'対象: {args.image.name}、窓 {size} px、stride {STRIDE}、周辺3×3、間隔 {SPACING} px、batch {args.batch_size}、周波数 {FMIN}–{FMAX}。',f'各条件 {args.repeats} 回、窓ごとに1回ウォームアップ（集計外）。同一プロセス内で順に計測。', 'FFT時間はrfft2呼び出し時間の合計。特徴量時間は前処理・FFT・正規化・memmap書き込みを含む。解析全体は画像読込・PNG保存を除き、一時ファイル作成・削除を含む。メモリ値はプロセス生涯ピークで各試行の独立ピークではない。', '', '## 結果']
    for n in workers:
        lines.append(f"- {n} workers: FFT {avg(size,n,'fft_seconds'):.3f} ± {std(size,n,'fft_seconds'):.3f} 秒、全体 {avg(size,n,'analysis_seconds'):.3f} ± {std(size,n,'analysis_seconds'):.3f} 秒、全体高速化 {avg(size,1,'analysis_seconds')/avg(size,n,'analysis_seconds'):.2f} 倍。")
    lines += ['', f'最短の平均FFT時間は {fastest_fft} workers、最短の平均解析時間は {best} workers。差が標準偏差と同程度なら優劣は不確かで、少ないワーカーを選ぶ余地がある。',f'全試行の異常度マップと先頭窓の複素FFTの一致（rtol=atol=1e-5）: {all(consistency)}。各試行の最大絶対差はCSV参照。複素FFT全窓の比較は行っていない。', 'FFT以外の周辺中央値計算が支配的な場合、FFTの短縮は全体へ比例して反映されない。メモリ帯域やCPU使用率は直接測定していないため、制約要因は特定できない。', '128 px以外の窓サイズを計測していない場合、窓サイズ別の並列効果は判断できない。']
    lines.append(f'暫定推奨は {recommended} workers。最短の平均解析時間に、その条件の標準偏差を加えた時間以内に入る最小ワーカー数という便宜的基準で選定した。CPU使用率は直接測定していない。')
    (out/'benchmark_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('Benchmark output:',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,default=DATA/'image5.png')
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=BATCH)
    p.add_argument('--windows',type=int,nargs='+',default=None)
    p.add_argument('--benchmark',action='store_true')
    p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    args.windows=args.windows or ([128] if args.benchmark else [64,128,192,256])
    if args.workers<1 or args.batch_size<1 or args.repeats<3 or any(n<2 or n%2 for n in args.windows) or len(set(args.windows))!=len(args.windows): p.error('positive workers and batch size, at least 3 repeats, and unique even windows >= 2 required')
    img=np.asarray(Image.open(args.image).convert('L'),dtype=np.float32)
    if args.benchmark: benchmark(args,img)
    else: multiscale(args,img,load_labels(args.image.with_suffix('.json'),img.shape))


if __name__=='__main__': main()
