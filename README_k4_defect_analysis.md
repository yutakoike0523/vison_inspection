# K=4 局所変形の診断

`python analyze_k4_defects.py` を実行する。既存の `output_all_images_radial_fft` にあるFFT特徴量と `output_reference_count_comparison` のK別スコア・選択方向・LOIO結果を再利用する。既存出力があるときは日時付きフォルダに保存する。

`summary/report.md` が集計と解釈、`all_defect_metrics.csv` が傷別の形状・汚染・検出結果、`reference_contamination.csv` が各参照窓、`frequency_contributions.csv` が帯域別の寄与。各 `images/imageN/defect_M` 内に対象・参照の配置、異常度マップ、スペクトル、周波数ごとの `spectra.csv` を保存する。

代表窓はK=4の傷中心異常度が最大の点。K=5・8もその同じ窓のスペクトルを表示するが、各Kの検出はそれぞれ傷中心全地点から判定する。格子位置は未判定。K=4を同じデータで選んでいるため、実験結果は独立した性能検証ではない。
