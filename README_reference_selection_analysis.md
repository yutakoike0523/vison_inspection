# 周辺参照の選択方法の比較

`python reference_selection_analysis.py` を実行する。既存の `output_all_images_radial_fft/image2`〜`image17` の `result.npz` と、既存の画像・同名Labelme JSONを使用する。FFT特徴量は再計算しない。出力先が存在するときは日時付きの新しいフォルダを作成する。

`summary/report.md` に条件と結果、`summary/loio_metrics.csv` に画像別の方式別LOIO結果、`summary/all_image_metrics.csv` に画像内の傷・正常統計、`summary/processing_times.csv` に特徴量計算後の処理時間を保存する。image17の56組合せと各中央値スペクトルは `image17` のCSVに保存する。

方式Cは正解の背景ラベルを参照選択に使用する原因分析用。方式Dは参照間RMSで代表参照とその近傍を選ぶラベルなしの固定規則。LOIO閾値は方式ごとに評価画像以外の正常点から設定し、画像内99%点や従来方式の閾値と区別する。独立した最終性能検証ではない。
