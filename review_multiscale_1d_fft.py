"""Summarize completed multiscale results; optionally redraw maps from CSV."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from compare_multiscale_1d_fft import ROOT,SIZES,METRICS,map_comparisons,overlap_map,read_gray,write_csv


def read_rows(path):
    rows=list(csv.DictReader(path.open(encoding='utf-8-sig')))
    for row in rows:
        for k,v in row.items():
            if k=='filename': continue
            if not v: row[k]=None
            elif v in ('True','False'): row[k]=v=='True'
            else:
                try: row[k]=float(v)
                except ValueError: continue
        for k in ('x_start','y_start','block_size','stride','block_row','block_col'):
            row[k]=int(row[k])
    return rows


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'outputs/multiscale_1d_fft')
    p.add_argument('--redraw',action='store_true')
    args=p.parse_args(argv); out=args.output
    report=json.loads((out/'run_summary.json').read_text())
    visual_path=out/'visual_review.json'
    visual=json.loads(visual_path.read_text(encoding='utf-8')) if visual_path.exists() else {}
    metrics=[]; validation=[]
    for entry in report['images']:
        stem=Path(entry['filename']).stem; dest=out/stem
        gray=read_gray(Path(report.get('input_dir',ROOT/'data20260903/data20260903'))/entry['filename'])
        rows={n:read_rows(dest/f'size_{n}/features.csv') if entry['counts'][str(n)] else [] for n in SIZES}
        if args.redraw: map_comparisons(gray,rows,dest,report['band'])
        notes=[f'# {stem}：256 / 128 観察メモ','',
               '傷・正常の正解ラベルは未確定。以下は特徴量の差であり、傷判定ではない。',
               '帯域は0.01〜0.04 cycles/pixelの仮設定。128は帯域内4 bins、256は8 binsで、推定の粗さが異なる。','',
               '## 256でも変化が見えるか',
               '256の帯域エネルギー割合の分布を記録する。位置は比較マップで確認する。']
        for d in ('x','y'):
            key=f'{d}_hann_band_energy_ratio'
            maps=[overlap_map(gray.shape,rows[n],key)[0] for n in SIZES]
            common=np.isfinite(maps[0]) & np.isfinite(maps[1])
            diff=np.where(common,np.abs(maps[1]-maps[0]),np.nan)
            values=np.array([r[key] for r in rows[256] if r[key] is not None])
            q=np.percentile(values,[10,50,90]) if values.size else [np.nan]*3
            if common.any():
                y,x=np.unravel_index(np.nanargmax(diff),diff.shape)
                mean=float(np.nanmean(diff)); maximum=float(diff[y,x])
                note=f'{d.upper()}：256の10/50/90百分位={q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}。サイズ間の平均絶対差={mean:.3f}、最大差={maximum:.3f}（例 x={x}, y={y}）。'
            else: x=y=None; mean=maximum=None; note=f'{d.upper()}：比較可能な画素なし。'
            notes.append('- '+note)
            metrics.append(dict(filename=entry['filename'],direction=d,mean_absolute_ratio_difference=mean,
                maximum_ratio_difference=maximum,example_x=x,example_y=y))
        notes.extend(['','## 128で初めて目立つか',
            '上記の差が大きい位置を重点的に目視確認する。サイズ間の差には、窓の範囲・周波数分解能・平均による打ち消しの違いも含まれる。「128だけで傷を検出」とはまだ判定しない。','',
            '## 背景・境界にも反応するか',
            '元画像を重ねた振幅マップと集中度マップを照合する。低い集中度を背景として除外しない。背景と傷の両方が低くなる可能性があり、確定ラベルなしの自動除外は行わない。'])
        if stem in visual:
            notes.extend(['','## 比較図の目視所見',visual[stem]])
        (dest/'observations.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')
        index=dest/'index.html'
        t=index.read_text(encoding='utf-8')
        t+='<h2>Review</h2><a href="observations.md">Observations</a>'
        for f in sorted(dest.glob('representative_*.png')):
            if f.name not in t: t+=f'<p><a href="{f.name}">{f.name}</a></p>'
        index.write_text(t,encoding='utf-8')
        for n in SIZES:
            expected=max(0,(gray.shape[0]-n)//(n//2)+1)*max(0,(gray.shape[1]-n)//(n//2)+1)
            assert len(rows[n])==expected==entry['counts'][str(n)]
            if expected:
                a=np.load(dest/f'size_{n}/profiles_spectra.npz')
                assert len(a['positions'])==expected
                assert np.isfinite(a['x_hann_amplitude']).all()
                assert np.array_equal(a['positions'],[[r[k] for k in ('block_row','block_col','x_start','y_start')] for r in rows[n]])
                count=sum((dest/f'size_{n}'/f).read_text(encoding='utf-8').count('<svg ') for f in entry['atlas'][str(n)])
                assert count==expected
            validation.append(dict(filename=entry['filename'],size=n,blocks=expected,status='passed'))
        print(f'Reviewed {stem}',flush=True)
    write_csv(out/'scale_differences.csv',metrics); write_csv(out/'validation.csv',validation)
    summary=['# 256・128ブロックの1次元FFT比較','',
        f'画像数：{len(report["images"])}。ブロック数：{report["total_blocks"]}。処理エラー：{len(report["errors"])}。',
        '', '## 最初に開くもの',
        '- `index.html`：全画像への入口。画像ごとの比較マップ、全ブロックスペクトル一覧にリンク。',
        '- `overview_01.png`〜：全画像の帯域エネルギー割合の比較縮小一覧。',
        '- `image15/representative_r4_c8_zoom.png`：傷・異物候補の同一中心256/128比較。',
        '- `image15/representative_r4_c6_zoom.png`：近傍格子候補の同一中心比較。',
        '- `all_features_256.csv` / `all_features_128.csv`：全ブロックの値と位置。',
        '- `all_representative_features.csv`：同一中心切り出しの値。通常の走査ブロックとは別の追加解析。',
        '- 各画像の `observations.md`：サイズ間の数値差と確認すべき事項。',
        '', '## 読み方と条件',
        '元画像はリサイズせず、256/stride128 と128/stride64で解析。Hannを主比較とし、plainもCSV・NPZ・全ブロックHTML一覧に保存。',
        '比較図は上段X・下段Y、左256・右128。対応する左右のカラースケールは同一。画像間の振幅スケールは同一ではない。',
        '重複部分は有限な特徴量の平均。灰色は特徴量未定義、赤色はブロックによる解析範囲外。ブロック境界図とCSVで位置を確認できる。',
        f'探索帯域は{report["band"]} cycles/pixel。格子帯域の確定値ではない。全帯域最大ピークは勾配・境界も拾う。帯域内ピーク振幅の図も併記した。',
        'ピーク±1 binの集中率はCSVに残すが、128/256で幅が異なるため主比較には使わない。共通帯域でもbin数・窓の広がり・投影平均の打ち消しは異なる。',
        '周波数マップも重複平均なので、画素値は単一ブロックの実測ピーク周波数とは限らない。生のピーク周波数はCSVを参照。',
        'スペクトル一覧はHTML内に全グラフをSVGとして埋め込み、外部通信なしで閲覧可能。青=X、橙=Y、実線=Hann、破線=plain。',
        '', '## 同じ場所での実測値',
        '|画像|代表位置（元128/stride128の行列）|サイズ|X帯域比|Y帯域比|Xピーク振幅|Yピーク振幅|',
        '|---|---|---:|---:|---:|---:|---:|']
    repfile=out/'all_representative_features.csv'
    if repfile.exists():
        for r in csv.DictReader(repfile.open(encoding='utf-8-sig')):
            summary.append('|'+ '|'.join([r['filename'],r['representative'],r['block_size']]+[f'{float(r[k]):.3f}' for k in ('x_hann_band_energy_ratio','y_hann_band_energy_ratio','x_hann_peak_amplitude','y_hann_peak_amplitude')])+'|')
    summary.extend(['', '## 次の判断',
        '傷の確定ラベルがないため検出率・見逃し率は算出していない。大ブロックの値だけを根拠に小ブロック解析を省略する方式は、現時点では採用しない。',
        'image15の既存候補では、傷候補と近傍格子のY帯域比の大小関係が256と128で逆転した。大きいブロックで低集中度だけを拾うルールは、小ブロックで見える差を拾える保証がない。',
        '次は元画像で正常・傷・背景の位置を確認し、そのラベルを固定して両サイズの見逃しを比較する。単にサイズで値が変わることを傷の証拠にはしない。',
        '', '## 再実行',
        '```powershell', 'python compare_multiscale_1d_fft.py',
        'python -m unittest test_compare_multiscale_1d_fft.py', '```',
        '既定の出力先が存在すれば日時付きの新規フォルダを作り、以前の実験は上書きしない。`--images image7 image15`で限定、`--band LOW HIGH`で探索帯域を変更可能。',
        '各画像の構造・全ブロック数・NPZ/CSVの座標対応・全スペクトルの収録数を検証し、`validation.csv`に記録した。'])
    (out/'README.md').write_text('\n'.join(summary)+'\n',encoding='utf-8')
    files=sorted(out.glob('image*/compare_band_energy_ratio.png'),key=lambda f:int(f.parent.name[5:]))
    for start in range(0,len(files),6):
        fig,axes=plt.subplots(3,2,figsize=(16,18))
        for ax in axes.flat: ax.axis('off')
        for ax,f in zip(axes.flat,files[start:start+6]):
            ax.imshow(cv2.cvtColor(cv2.imread(str(f)),cv2.COLOR_BGR2RGB)); ax.set_title(f.parent.name)
        fig.tight_layout(); fig.savefig(out/f'overview_{start//6+1:02d}.png',dpi=100); plt.close(fig)
    print('All output structure and block count checks passed',flush=True)


if __name__=='__main__': main()
