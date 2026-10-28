# 全画像の放射方向平均FFT評価

必要: `numpy`, `scipy`, `matplotlib`, `Pillow`, `psutil`。同じフォルダーに `multiscale_fft_experiment.py` と `radial_fft_experiment.py` が必要です。

```powershell
python evaluate_all_radial_fft.py

# FFTワーカー数とバッチサイズを指定
python evaluate_all_radial_fft.py --workers 4 --batch-size 64

# 同一範囲の従来2D方式も計算（より時間とディスク容量が必要）
python evaluate_all_radial_fft.py --two-d --output output_all_images_radial_fft_2d
```

条件は256 px、stride 16 px、参照3×3（中央除く）、128 px間隔、0.01〜0.50 cycles/pixel、半径ビン幅0.01。image5実験と同じ片側パワー正規化と対称重みを使用します。参照が3点未満の窓は評価不能とし、全窓が該当する画像は除外します。正常領域は全アノテーションと窓が重ならない領域です。正常100窓未満または標準偏差が1e-6未満の場合は分離スコアを計算しません。Labelme JSONがない場合、傷がないとは見なさず評価不能とします。

結果は `output_all_images_radial_fft` に画像単位で保存されます。同じ設定で再実行すると完了した画像を飛ばし、未処理の画像だけを処理します。中断した未完了の作業は `imageN_partial...` として残り、再実行では別の作業フォルダーが作成されます。既存の画像結果を上書きしません。集計は実行ごとに `summary`、`summary_日時` の順に新規作成します。再開時に設定や入力フォルダーを変更する場合は、別の `--output` を指定してください。

閾値評価は格子あり・なしを別々に行うLeave-One-Image-Outです。各評価画像を除いた同グループの正常点を画像ごとに同じ合計重みで統合し、その99パーセンタイルをfold共通の閾値とします。foldごとに閾値は異なり、事前に固定された単一の閾値の検証ではありません。中心窓・重なり窓を別々に評価し、傷インスタンスは対応窓のいずれかが閾値を超えると検出とします。image5は条件選択に使われているため、完全な独立評価ではありません。

時間は画像読込・図表出力を除く解析時間です。各画像は別プロセスで実行し、Windowsでは解析終了時までのプロセスピーク作業セットを記録します。プロセス起動やライブラリのメモリも含み、方式別に分解できません。
