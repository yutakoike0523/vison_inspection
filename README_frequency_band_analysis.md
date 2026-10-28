# 放射FFTの周波数帯比較

`python frequency_band_analysis.py` を実行する。保存済みの `output_all_images_radial_fft` の特徴量と `output_reference_count_comparison` のK=4参照選択を利用する。既存の出力先は上書きせず、日時付きフォルダを作成する。

`fixed_references` は全帯域で選んだ参照を固定してRMSに用いる帯域だけ変更する。`band_specific_references` は各帯域で参照を選び直す。各方式・帯域のLOIOは別々に閾値を作成する。`overall_comparison/final_report.md` と各 `summary` に結果を保存する。選択集合は各画像のNPZのビット配列として保存され、方向と座標の復元方法は `selected_references.csv` に記載する。

画像5・15・17の正常窓と傷中心の分布図、および詳細比較表は、それぞれ `frequency_band_distributions.py` と `frequency_band_supplement.py` で追加作成できる。これらの補助スクリプトの対象は既定の `output_frequency_band_analysis` である。

周波数帯はビン中心を用いて半開区間で選択し、元の最後の有効ビンは全帯域・低周波除外・高周波条件に含める。帯域切り出し後の再正規化は行わない。今回の帯域選択は探索であり、独立した検証画像が必要。
