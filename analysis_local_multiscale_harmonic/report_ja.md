# 局所マルチスケールFFTによる格子高調波ノッチ

## 結論

- OLAのノッチなし最大絶対復元誤差（image15）は 2.274e-13 であり、窓処理と合成の基礎復元を確認した。
- 高調波は512/256/128 pxの実ピーク一致（512 px prominence ≥4 dB かつ短い窓の一方 ≥3 dB）を満たす場合だけ局所ノッチに加えた。未確認の予測整数倍は除去していない。
- image15で確認された高調波種: 2g1, 2g1+g2, 2g2, 3g1, 3g2, g1+2g2, g1+g2, g1-g2。
- 検出性能の優劣は下表CSVで比較する。局所適応ノッチは高調波を多く除去するほど良いとは限らず、傷帯域の損失を含めて判断する。

## 性能（固定窓集合）

- none: AUC=0.6843, TPR@1%=0.0361, TPR@5%=0.1878
- global_base: AUC=0.6908, TPR@1%=0.0722, TPR@5%=0.2261
- local_base: AUC=0.6922, TPR@1%=0.0641, TPR@5%=0.2231
- local_all: AUC=0.6845, TPR@1%=0.0788, TPR@5%=0.2261

## 出力

- `multiscale_peak_observations.csv`: 512/256/128pxのピーク位置・prominence・一致判定。
- `harmonic_confirmation_summary.csv`: 画像別・高調波別の確認率。
- `image15_local_harmonic_comparison.png`: 基本のみ／高調波込み／逆FFT成分／FFT。
- `local_ola_reconstruction_metrics.csv`: OLA復元精度とノッチ幅（cycles/pixel）。
- `local_harmonic_performance.csv`, `local_harmonic_type_performance.csv`: 固定窓集合の性能。

## 限界

512px窓には約3周期しか含まれず、局所周波数推定には分解能限界がある。局所ピークが不安定な位置での過剰ノッチを避けるため、確認条件は保守的にした。
