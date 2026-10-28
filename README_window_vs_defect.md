# 局所変形サイズとFFT窓サイズ

`window_vs_defect_experiment.py` は既存の `evaluate_all_radial_fft.py`、`radial_fft_experiment.py`、`multiscale_fft_experiment.py` を使用します。必要パッケージは NumPy、SciPy、Matplotlib、Pillow、psutil です。

```powershell
python window_vs_defect_experiment.py

# 周辺参照間隔の追加検証を省略
python window_vs_defect_experiment.py --no-extra-spacing

# 計算済みデータを再集計（新しい summary フォルダーに保存）
python window_vs_defect_experiment.py --summary-only
```

対象は image1〜21。窓64・128・192・256 px、stride 16、周辺3×3、128 px間隔、0.01〜0.50 cycles/pixel、放射ビン幅0.01。FFTワーカー4、バッチ64です。前回の256 px結果は設定と入力フォルダーが一致するときだけ再利用します。`settings.json` が異なる場合は別の `--output` を指定してください。

出力は `output_window_size_vs_defect_size`。画像別NPZ・図と、`summary` 以下のCSV・図・レポートを保存します。完了済み画像と窓はスキップし、集計は別名で保存します。ビンの成分数と使用有無は `frequency_bins.csv` に保存します。`image18` は小画像として条件ごとに参照数を確認し、他の画像の閾値学習には含めません。

正常は全ラベルとFFT窓が重ならない領域で、未ラベルを正常と仮定します。正常100窓未満、正常標準偏差1e-6未満、傷の評価窓4未満ではSを評価不能とします。各窓での全領域と、すべての窓で評価可能な共通中心座標範囲を区別します。LOIOは同じ画像群の学習画像を均等に重み付けした正常99%点をfold単位で使用します。

追加実験では、192・256 pxのうち同じデータ内でLOIO検出数が多い窓を探索的に選び、既存のFFT特徴量を再利用して参照間隔192・256 pxを試します。参照位置も変わるため、差を窓重複だけの効果とは解釈できません。
