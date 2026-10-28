# FFT ストライドと格子・変形位置の比較

`python stride_grid_overlap_experiment.py` を実行すると、image15 と image17 の全画像を FFT 窓 256 px、参照間隔 128 px、ストライド 16・8・4 px で解析する。既定のデータディレクトリは既存の `multiscale_fft_experiment.py` と同じ。`--data` と `--output` で変更できる。

出力済みの計算結果は再利用する。図や集計だけ再作成するには `python stride_grid_overlap_experiment.py --summary-only` を使う。条件が異なる既存の出力先は使用できない。出力先は既定で `output_stride_grid_overlap`、全体の集計とレポートは `summary`、画像ごとの図は `image15/comparison` と `image17/comparison` に保存する。

共通座標での数値一致確認に加え、各ストライドの全座標と共通 16 px 座標で傷中心・窓の重なりを別に評価する。既存の画像別 LOIO 正常 99 パーセンタイル閾値を全ストライドに適用し、ストライド別の再学習は行わない。画像内の未ラベル領域を正常とみなす仮定や、FFT 窓と参照窓の重複による制約はレポートに記載する。計算時間には画像読込や図保存は含まない。
