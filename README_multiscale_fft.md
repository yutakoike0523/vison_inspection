# 局所FFT窓サイズ・FFTワーカー比較

必要なPythonパッケージ: `numpy`, `scipy`, `matplotlib`, `Pillow`, `psutil`。

```powershell
# image5 の64・128・192・256 px比較
python multiscale_fft_experiment.py

# 別画像とFFTワーカー数を指定
python multiscale_fft_experiment.py --image data20260903/data20260903/image15.png --workers 4

# 窓128 pxのFFTワーカー数比較（ウォームアップ1回、各条件3回）
python multiscale_fft_experiment.py --benchmark

# 追加実験：複数窓サイズのワーカー数比較（計算負荷が大きい）
python multiscale_fft_experiment.py --benchmark --windows 64 128 192 256
```

画像と同じ場所・同じ名前のLabelme JSONを読み込みます。対応形状はpolygon、rectangle、circle、point、line、linestripです。pointとlineのマスク幅はLabelmeの既定値に合わせています。画像の寸法がJSONと異なる場合や未対応形状はエラーにします。`局所変形`以外のラベルも正常領域から除外します。正常評価は窓全体が全ラベルと重ならない窓を対象とし、未ラベル領域を正常と仮定します。傷は中心が含まれる場合と窓が1画素以上重なる場合を別に集計します。

周辺参照はマスクせず、画像端では存在する参照のみを使用します。平均除去、ハン窓、rfft2、パワー正規化、対数変換、周波数範囲、中央値およびRMSは元コードと同じです。参照間隔は128 px、strideは16 pxです。比較画像は全窓の共通中心座標範囲を表示し、正規化は各窓の共通範囲の中央値と標準偏差を使用します。色範囲は表示用で判定閾値ではありません。

出力は `output_multiscale_fft/image5` と `output_multiscale_fft/benchmark`。既存ディレクトリがある場合は日時付きの別ディレクトリに保存します。FFTスペクトルは一時memmapファイルに保存し、処理後に削除します。十分な一時ディスク容量が必要です。記録する最大メモリはWindowsのプロセス生涯ピーク作業セットで、各試行の独立したピークではありません。ベンチマークは同じプロセスで連続実行し、画像読込と画像出力は計測に含めません。
