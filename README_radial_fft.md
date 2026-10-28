# 2D FFTと放射方向平均の比較実験

`numpy`、`scipy`、`matplotlib`、`Pillow`、`psutil`が必要です。`multiscale_fft_experiment.py` のLabelme処理等を利用します。

```powershell
# image5、64/128/192/256 px、半径ビン幅 0.01 cycles/pixel
python radial_fft_experiment.py

# 画像や設定を変更
python radial_fft_experiment.py --image data20260903/data20260903/image15.png --bin-width 0.005 --workers 4 --batch-size 64
```

入力画像と同じ場所の同名Labelme JSONを読み込みます。画像の分析ではマスクをかけず、正常の集計において全ラベルに窓が重なる場合を除外します。局所変形の中心と窓全体の重なりを別々に集計します。アノテーションがない場合、検出性能は評価しません。

0.01 cycles/pixelのビンを全窓で共通に使い、周波数範囲内の成分のみ集約します。フルFFT換算で4成分に満たないビンは欠損として評価から外し、補間しません。`comparison/bin_counts.csv` に各ビンの成分数を記録します。半径ビン境界の表記は0からの固定幅で、グラフではビン中心にプロットします。各範囲のパワー正規化は元コードの片側FFTの総パワーを維持し、放射平均は負のfx側の対称成分を正しく重み付けした線形パワーを使います。従ってフルFFTの総和で正規化する定義とは異なります。

解析結果は `output_radial_fft/画像名/` に保存し、同名の出力がある場合は日時を付けたディレクトリを新規作成します。画像別のNPZには2D・各周波数範囲の放射異常度、生の座標、ビン数と設定を含みます。`comparison/settings.json` とCSVにも条件・測定値を記録します。スペクトルの一時ファイルには十分なディスク容量が必要です。

FFT結果は2方式および4周波数範囲で共用します。時間計測の範囲とメモリ値の制約はレポートに記載しています。比較図の色範囲は表示設定であり、傷判定の閾値ではありません。
