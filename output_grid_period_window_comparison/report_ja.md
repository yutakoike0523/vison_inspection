# 格子周期に応じた局所FFT窓サイズの比較

## 1. 実験目的と仮説

格子周期Pの1/2/3倍窓と固定128 px窓を、同じ局所放射FFT定義で比較した。仮説は、複数周期を含む窓が正常格子のばらつきを安定化し、傷との分離を改善する一方、大窓では局所傷が希釈される、というものである。

## 2. 設定と再現方法

平均除去、2D Hann、SciPy `rfft2`（backward scaling）、非正規化power=|F|²、共役重み付き放射平均、10log10、8参照dBスペクトルの算術平均、対象との差のRMSを既存実装から踏襲した。FFTサイズ=W、ゼロパディングなし、ノッチ除去なし、stride=16 px、放射ビン幅=0.01 cycles/pixel。

参照距離は固定48 pxと round16(0.375W)、帯域は0.020–0.500と0.005–0.500を比較した。8窓が全て画像内にある中心だけを有効とした。スコアは窓中心の16×16 pxセルに対応付け、セルが傷矩形に1 pxでも重なれば傷、傷でなくtag/backgroundに重なれば除外、それ以外を正常表面とした。傷と除外の重複は傷を優先した。

FPR 1%/5%閾値は条件ごとに他画像の正常セルを画像等重みで集約したLOIO分位点であり、評価画像のラベルで画像別最適化していない。これは比較用閾値で、未知画像へそのまま適用する運用閾値ではない。

実行コマンドは `py -3.10 grid_period_window_comparison.py --output output_grid_period_window_comparison`、処理時間は7.1分。詳細は `settings.json` と `processing_time.csv`。

## 3. 全体結果（共通評価領域、d=48、0.020–0.500）

|窓|ROC-AUC|PR-AUC|傷検出率@FPR1%|評価傷/対象外|
|---|---:|---:|---:|---:|
|fixed128|0.6677|0.0182|0.0%|0/22 (14対象外)|
|period_1x|0.5121|0.0119|13.6%|3/22 (14対象外)|
|period_2x|0.4785|0.0122|13.6%|3/22 (14対象外)|
|period_3x|0.4816|0.0117|9.1%|2/22 (14対象外)|

PR-AUC最大は **fixed128**、ROC-AUC最大は **fixed128**、FPR 1%での傷単位検出率最大は **period_1x**。異なる指標が一致しない場合は、正常誤検出と傷単位検出を併記して判断する。

参考として、条件ごとに使える全有効領域（母数が異なる）も示す。

|窓|ROC-AUC|PR-AUC|傷検出率@FPR1%|評価傷/対象外|
|---|---:|---:|---:|---:|
|fixed128|0.6546|0.0211|3.2%|1/31 (5対象外)|
|period_1x|0.4589|0.0119|5.9%|2/34 (2対象外)|
|period_2x|0.4224|0.0114|9.4%|3/32 (4対象外)|
|period_3x|0.4334|0.0115|0.0%|0/28 (8対象外)|

## 最重要4問への回答

- **Q1:** 共通領域の分離性能では固定128 pxが最良だった。周期倍率内では従来帯域のPR-AUC最大は **period_2x** (0.0122)。ただしFPR 1%傷検出は1倍と2倍が各3/22で同率であり、単一指標で万能な倍率はなかった。
- **Q2:** 共通ではない。画像別PR-AUC勝者（共通領域が成立した16画像）は fixed128=3画像, period_1x=4画像, period_2x=3画像, period_3x=6画像 で分散した。
- **Q3:** 種類差があった。共通領域FPR 1%では点状は1/2/3倍が各2/8、局所変形は1倍・2倍が各1/12、線状は全条件0/2だった。面状異常は指定画像範囲にアノテーションがなく評価不能。全有効領域では線状のみ固定128が1/3を検出した。
- **Q4:** 低周波追加で、周期倍率内のPR-AUC最良は **period_2x→period_1x** (0.0122→0.0127)へ変わった。一方、全4窓では固定128が引き続き最良で、低周波追加によりPR-AUCは0.0182→0.0165へ低下した。

## 4. 各画像の最適窓（共通領域、PR-AUC）

- image2: period_3x (PR-AUC=0.0800, W=400 px, W/P=2.99)
- image3: period_3x (PR-AUC=0.0165, W=272 px, W/P=3.07)
- image4: period_2x (PR-AUC=0.0214, W=176 px, W/P=1.94)
- image5: fixed128 (PR-AUC=0.0190, W=128 px, W/P=0.94)
- image6: period_3x (PR-AUC=0.0373, W=400 px, W/P=3.03)
- image7: period_2x (PR-AUC=0.0417, W=112 px, W/P=1.90)
- image8: period_2x (PR-AUC=0.0275, W=112 px, W/P=1.87)
- image9: period_3x (PR-AUC=0.0149, W=272 px, W/P=3.01)
- image10: period_3x (PR-AUC=0.0405, W=416 px, W/P=3.05)
- image11: fixed128 (PR-AUC=0.0222, W=128 px, W/P=1.87)
- image12: period_3x (PR-AUC=0.0384, W=272 px, W/P=2.98)
- image13: period_1x (PR-AUC=0.0505, W=176 px, W/P=0.98)
- image14: period_1x (PR-AUC=0.0194, W=144 px, W/P=0.99)
- image15: period_1x (PR-AUC=0.0581, W=176 px, W/P=0.98)
- image16: fixed128 (PR-AUC=0.2346, W=128 px, W/P=0.66)
- image17: period_1x (PR-AUC=0.0383, W=144 px, W/P=0.99)

## 5. 傷種類別

|種類|最良窓（検出率@FPR1%）|検出率|評価件数|
|---|---|---:|---:|
|area|fixed128|NA|0|
|line|fixed128|0.0%|2|
|local_deformation|period_1x|8.3%|12|
|point|period_1x|25.0%|8|

## 6. 改善・悪化例

窓間で生スコアを直接比較せず、各条件のLOIO FPR 1%閾値に対する最大スコア比を図の候補抽出に用いた。性能判断はAUC・固定FPR指標を優先する。

- 改善候補 image8 defect2: period_1x で閾値比がfixed128比 +0.568。`defect_zoom_comparison/`参照。
- 改善候補 image7 defect2: period_2x で閾値比がfixed128比 +0.547。`defect_zoom_comparison/`参照。
- 改善候補 image8 defect3: period_1x で閾値比がfixed128比 +0.439。`defect_zoom_comparison/`参照。
- 改善候補 image7 defect1: period_2x で閾値比がfixed128比 +0.386。`defect_zoom_comparison/`参照。
- 改善候補 image8 defect1: period_1x で閾値比がfixed128比 +0.382。`defect_zoom_comparison/`参照。
- 悪化候補 image15 defect2: period_3x で閾値比がfixed128比 -0.812。`defect_zoom_comparison/`参照。
- 悪化候補 image13 defect3: period_3x で閾値比がfixed128比 -0.777。`defect_zoom_comparison/`参照。
- 悪化候補 image17 defect2: period_3x で閾値比がfixed128比 -0.698。`defect_zoom_comparison/`参照。
- 悪化候補 image16 defect1: period_3x で閾値比がfixed128比 -0.671。`defect_zoom_comparison/`参照。
- 悪化候補 image2 defect1: period_3x で閾値比がfixed128比 -0.538。`defect_zoom_comparison/`参照。

## 7. 参照距離の影響

- fixed48: PR-AUC最大 fixed128 (0.0182)。
- proportional: PR-AUC最大 fixed128 (0.0182)。

固定128は両距離がともに48 pxなので同値。比例距離にすると、2倍窓のPR-AUCは0.0122→0.0171、3倍窓は0.0117→0.0144へ改善した。大窓でd=48を固定した際の高い重なり率が性能を下げていた可能性がある。両者の差は窓サイズだけでなく参照の局所性の影響を表す。

## 8. 低周波帯域の影響

- conventional: PR-AUC最大 fixed128 (0.0182)。各条件の実ビン数はCSVに記録し、0ビン条件は評価から除外した。
- low_inclusive: PR-AUC最大 fixed128 (0.0165)。各条件の実ビン数はCSVに記録し、0ビン条件は評価から除外した。

## 9. 考察・次の実験

最適倍率の共通性は、全体最良だけでなく画像別勝者と種類別結果の分散で判断する。次段階では、今回の上位倍率を固定したまま独立画像で運用閾値を校正し、格子位相に対する窓中心のずれ、放射ビン幅、参照集約（平均/中央値）を一因子ずつ検証する。

## データ上の制約

- image2〜image18に実在する傷アノテーションは36件であり、依頼文の54件とは一致しない。54件へ水増しせず、36件を母数とした。
- image18は428×321 pxで、大窓や共通領域が成立しない条件がある。対象外件数を失敗と分離した。
- backgroundアノテーションがない画像でも、tagと傷以外の未除外領域を既存評価と同様に正常候補とした。
- 窓サイズ間でdBスコアの絶対値は直接比較せず、AUC、固定FPR、傷単位検出を主指標とした。

## 出力一覧

指定CSV、比較画像、傷ズーム、指標図を各ディレクトリへ保存した。`valid_region_summary.csv`には全条件の有効セル数・面積と評価可能傷数を記録した。

`defect_zoom_comparison/` は傷アノテーション枠を描かない同一領域・共通スケールの拡大図。`defect_fft_comparison/` は各傷・4窓について、d=48 px固定、0.02–0.50 cycles/pixelで、対象/8参照窓配置、対象2D log-power、8参照平均2D dBスペクトル、符号付き2D差、放射平均1Dスペクトル、1D差を示す。代表対象は傷と重なる有効セルの最大スコア位置で、可視化選択にのみアノテーションを使用した。