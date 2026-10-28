# 低周波FFTの要因分析

`python low_frequency_analysis.py` を実行する。既存の `output_frequency_band_analysis`、`output_all_images_radial_fft`、`output_k4_defect_analysis_20261026_222547` の結果を再利用する。新しい日時付き出力を作り、従来結果は変更しない。全帯域7/15、低周波固定11/15、全4ビンの再計算一致を検証する。

既定出力 `output_low_frequency_analysis` に対しては、`python low_frequency_supplement.py`、`python low_frequency_summary_supplement.py`、`python low_frequency_complete_metrics.py` を順に実行すると、参照別スペクトル、参照配置図、空間分布、処理時間、参照再選択のSを補完できる。補助スクリプトは既定出力を対象とするため、日時付き出力を使用する場合はスクリプトの出力先を合わせる。

`summary/report.md` に結果と条件、`detection_comparison` に傷単位の変化、`frequency_bins` にビン別差と1ビン除外結果、`anomaly_maps/imageN` に全帯域・低周波のマップと代表スペクトルを保存する。`reference_selection` には正解ラベルを選択後に照合した参照窓の混入率を保存する。

画像単位の正常超過は、その画像のラベルに重ならない正常窓が1点以上閾値を超えることとして集計している。連結成分は解析中心の8近傍であり、実際の傷の面積ではない。今回の帯域やビン選択は探索であり、未使用画像で検証する。
