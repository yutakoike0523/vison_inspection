# 元画像を128・256画素ブロックに分ける局所FFT実験

格子の周期性が局所的に変化するかを確認する実験です。傷の合否判定や確率推定は行いません。テープ、外枠、金属、照明の違いも高得点になり得ます。

**現在の既定の色付けはR（`peak_energy_ratio`）です。** ピーク周辺のエネルギーを、DC円外のFFT全体のエネルギーで割った値をそのまま表示します。低いRが赤、高いRが青で、全画像・128/256の両条件とも0～1に固定します。画像ごとの自動伸張や正常基準との差し引きはしません。

Rは各ブロックの強いピークへの集中度です。高いRが正常格子であることを保証するわけではありません。既定のDC半径5、ピーク周辺半径2、最大8候補点の条件は従来のままです。候補が不足するブロックでは8点未満になる場合があります。

## 実行

```powershell
python -m pip install numpy matplotlib opencv-python
python local_fft_block_experiment.py
```

同じフォルダの `compare_fft_center_peaks.py` から、画像読み込み、平均値除去、Hann窓、FFT、中心拡大、対称ピーク検出を再利用します。この2ファイルを同じ場所に置いてください。局所FFT用に局所最大の近傍サイズを指定できるようにしましたが、既存の全画像FFTの既定動作は変えていません。

既定の入力は `data20260903/data20260903` 直下の元画像、出力は `outputs/local_fft_peak_energy_ratio` です。ファイル名順で処理し、trimingサブフォルダは読みません。画像をリサイズせず、128×128・256×256の両方で実行します。端の不足ブロックは除外します。PNG/JPG/JPEG/BMP/TIF/TIFFに対応し、8/16ビットの輝度を保持します。

特定の画像だけ試す場合:

```powershell
python local_fft_block_experiment.py --filenames image7.png image8.png image11.png image12.png image18.png
```

50%オーバーラップ:

```powershell
python local_fft_block_experiment.py --block-sizes 128 256 --strides 64 128 --output-dir outputs/local_fft_overlap
```

256画素だけなら `--block-sizes 256 --strides 128` です。オーバーラップした画素のヒートマップ値は、その画素を含むブロックの異常度の平均です。除外した端やスコア計算不可の画素はNaNで保持し、ヒートマップを塗りません。

## 保存内容

各 `outputs/local_fft_peak_energy_ratio/元ファイル名/block_128_stride_128/` または `block_256_stride_256/` に保存します。

| ファイル | 内容 |
|---|---|
| `block_grid.png` | 元画像＋境界＋0始まりの行・列番号 |
| `peak_energy_ratio_heatmap.png` | Rのヒートマップ。赤＝低い、青＝高い |
| `peak_energy_ratio_overlay.png` | 元画像＋半透明のRヒートマップ＋境界 |
| `features.csv` | 全ブロックの特徴量、スコアと各加算項 |
| `top_blocks.csv` | Rが低い順のブロックの座標・特徴量・順位 |
| `top_blocks_fft_comparison.png` | R最大の比較用1ブロックとRが低い順の6ブロックの、元画像・FFT全体・中心拡大・DCマスク＋候補 |
| `peak_energy_ratio_map.npy` | 元画像と同じ縦横画素数のRマップ。未評価画素はNaN |
| `report.json` | 正常基準の仮定、設定、重み、被覆範囲、上位候補 |

画像のフォルダには `元ファイル名_block_size_comparison.png` も保存します。128・256のヒートマップを上段、半透明重ね合わせを下段に並べます。Rの色範囲は常に0～1です。重複領域はRの平均で表示します。

DC円外のエネルギーがゼロの場合、Rは未定義なので着色と順位付けから除外します。CSVの `peak_energy_ratio_defined=False` で識別してください。この場合だけ、互換性のため残している数値0をRの観測値として解釈しないでください。分母が正で採用ピークがない場合のR=0は、赤で表示します。

全体の `all_block_features.csv` で全画像・全サイズの値を比較できます。`report.json` は全体レポート、`reference_blocks_template.json` は手動基準の入力用ひな形です。

`--top-k 10` で上位件数、`--show-values`（`--show-scores` も可）で重ね合わせ図へのR表示を指定できます。R表示時は `--score-vmax` を受け付けず、0～1を維持します。

従来の複合異常度の表示も `python local_fft_block_experiment.py --color-metric anomaly_score` で使用できます。その場合は従来の `outputs/local_fft_block_experiment` に保存し、`anomaly_heatmap.png` 等のファイル名になります。`--score-vmax` や正常基準はこの表示に適用されます。CSVにはどちらの表示でもRと従来スコアの両方を保存します。

## FFTと特徴量

各ブロックで、float化 → ブロック平均を引く → `np.outer(np.hanning(H), np.hanning(W))` を掛ける → `fft2` → `fftshift` を実行します。振幅は `abs(F)`、表示は `log1p(abs(F))`、エネルギーは **`abs(F)**2`** を使います。DC除外や表示マスクで、FFT配列そのものは書き換えません。

中心拡大の既定半径はブロック幅の1/4（128なら32ビン、256なら64ビン）です。DC除外半径は5ビン。局所最大は3×3近傍、DC外のlog振幅の95パーセンタイル以上を候補とし、近接候補を間引いて最大8点＝4対称ペアを採用します。

- **主ピーク**：最も強いペアの片方のx/y、距離、軸方向角度、log振幅。角度はx右・y下の座標で0～180度に統一し、180度違う共役点を同じ向きとして扱います。距離はFFTビン単位、実周期は `block_size / distance` [画素]です。
- **強度**：CSVには生の振幅とlog振幅も保存。スコアには `main_peak_magnitude / sum(Hann窓)` を使い、ブロックの面積による単純な増幅を補正します。照明・コントラストによる影響は残ります。
- **対称性**：対になるピークも検出条件を満たした場合に採用します。実画像はノイズも共役対称なので、対称性だけでは正常と判断できません。
- **Peak energy ratio**：採用した全ピークの半径2ビンの円の和集合内のエネルギー ÷ FFT全体のDC円外のエネルギー。円が重なっても二重加算しません。分子・分母ともDC円内を除きます。
- **Spectral spread**：DC円外の各周波数点から最も近い採用ピークまでの距離について、パワー重み付きRMSを計算し、ブロック幅で割った値です。単位はcycles/pixelです。小さいほどピーク近傍に集中し、大きいほど周辺・遠方に広がる定義です。ピーク数や位置にも依存するため、単独で傷を判定しません。

各ブロックの `peak_candidates_json` 列に全候補の座標・強度・ペア番号を保存します。CSVをPythonで読む場合、`json.loads(row['peak_candidates_json'])` で候補リストに戻せます。`energy_fraction_outside_dc` には、マスク後に残る全FFTエネルギーの割合を保存します。

設定例:

```powershell
python local_fft_block_experiment.py --dc-mask-radius 5 --peak-radius 2 --max-peaks 8 --peak-percentile 95
```

## 正常基準と異常度

この節は `--color-metric anomaly_score` で使う従来の複合異常度について説明します。**Rの色付けや順位は、正常基準・重み・複合異常度を参照しません。** 以下の手動基準の実行例で複合異常度を表示したい場合は、`--color-metric anomaly_score` を付けてください。

**方式A（既定）** は画像・サイズごとの全ブロックの特徴量中央値を基準と仮定します。角度は180度周期の円周上の中央値を使います。これは実際の正常保証ではなく、格子以外が多い画像では基準が偏ります。

**方式B** は確認した正常ブロックを指定します。`block_grid.png` の **0始まりのrow,col** を使います。座標はブロックサイズやstrideを変えると変わります。

```powershell
# 座標は操作例。元画像を見て正常と確認した座標に置き換えてください。
python local_fft_block_experiment.py --filenames image7.png --block-sizes 128 --reference-mode manual --reference-blocks "2,3;2,4" --output-dir outputs/local_fft_manual
```

複数画像・サイズごとに指定するには、出力された `reference_blocks_template.json` を別名でコピーして編集します。

```json
{
  "image7.png": {
    "128": [[2, 3], [2, 4]],
    "256": [[1, 2], [2, 2]]
  }
}
```

```powershell
python local_fft_block_experiment.py --filenames image7.png --reference-mode manual --reference-json my_reference_blocks.json --output-dir outputs/local_fft_manual
```

空または範囲外の座標はエラーにし、中央値方式へ黙って切り替えません。コード先頭の `REFERENCE_BLOCKS = [(row, col), ...]` でも指定できます（`--reference-mode manual` が必要）。

異常度の各項は次のように無次元化し、0～1に収めて重み付き和にします。基準をr、対象をbとします。

| 加算項 | 計算 |
|---|---|
| `peak_position_error` | `abs(distance_b - distance_r) / max(distance_r, 1)` |
| `peak_angle_error` | 180度周期での角度差 / 90度 |
| `peak_strength_drop` | `max(0, 1 - strength_b / strength_r)` |
| `energy_ratio_drop` | `max(0, 1 - ratio_b / ratio_r)` |
| `spectral_spread_increase` | `max(0, (spread_b - spread_r) / max(spread_r, 1 / block_size))` |

既定は全項の重みが1なのでスコア範囲は0～5です。重みはコード先頭の `WEIGHTS`、または位置・角度・強度・エネルギー比・広がりの順で指定します。

```powershell
python local_fft_block_experiment.py --weights 1 0.5 1 2 1
```

ピークなしのブロックは、使用可能な正常基準があれば欠落のペナルティを与えます。基準自体に使用可能なピークがない場合はスコアを欠測値にし、0点＝正常とは扱いません。主ピーク位置・角度・spreadの中央値は欠測を除いて計算し、強度・energy ratioのゼロ値は中央値計算に含めます。

比較図の先頭は基準群の中でスコアが最も低い実ブロックです。特徴量の中央値そのものの画像ではなく、目視比較用の代表です。

## 解釈上のポイント

- **ブロックに分ける理由**：画像全体では格子のエネルギーに埋もれる小さな変化を、位置付きで調べられるからです。
- **局所FFTで変化を捉える可能性**：傷や異物が周期構造を乱すと、ピークの強度・位置・向き・集中度が変わり、周辺周波数へエネルギーが広がることがあるためです。ただし必ず変化するとは限りません。
- **正常と異常の予想**：正常な格子では特定のピーク付近にエネルギーが集中しやすく、乱れた領域では低下・拡散が起きるという仮説を検証します。テープや外枠にも反応するので、上位ブロック画像と特徴量を合わせて見てください。
- **128と256の比較理由**：128は位置を細かく分けられる一方、含む周期数が少なくFFTピークが不安定になりがちです。256は周期推定が安定しやすい一方、小さな傷の寄与が薄まります。Hann窓でブロック端の傷も弱まるため、次の比較として50%オーバーラップが使えます。

**DCマスクの半径に注意**：周期60画素の成分は128ブロックで約2.1ビン、256ブロックで約4.3ビンです。半径5では基本周期まで隠します。既定値5は指定条件に合わせていますが、`--dc-mask-radius 1` や `2` の別出力と比べ、マスクなしFFTや候補位置を確認してください。同じビン半径でもブロックサイズによって除く物理周波数帯が違います。

```powershell
python local_fft_block_experiment.py --filenames image7.png image11.png --dc-mask-radius 1 --output-dir outputs/local_fft_dc1
```

格子の2方向が同程度に強いと主ピークが入れ替わり、角度差が大きく出る場合があります。`--weights 1 0 1 1 1` で角度項を外して比較できます。高度な格子方向の対応付けは今回実装していません。

image18は428×321画素なので、256では完全なブロックが1個だけです。従来の複合異常度の方式Aでは自分自身が基準になって0点になりますが、Rは基準不要で、この1ブロック自体の集中度を表示できます。画像端に残る領域は除外されています。

## 検証

```powershell
python -m unittest test_local_fft_block_experiment test_compare_fft_center_peaks -v
```

既知周期の検出、入力不変、エネルギー比、乱れを加えた合成ブロックの順位、中央値／手動基準、角度の180度折り返し、欠落ピーク、基準不成立、端除外、オーバーラップ平均を確認します。合成例で反応することは、実画像の傷検出性能を保証するものではありません。
