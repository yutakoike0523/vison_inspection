# 元画像のFFT前処理比較

Hann適用後のFFTを中央拡大し、DCマスクと対称ピーク候補を確認する追加処理は、[中央拡大・ピーク候補の説明](README_fft_center_peaks.md) を参照してください。実行コマンドは `python compare_fft_center_peaks.py` です。

```powershell
python -m pip install numpy matplotlib opencv-python
python compare_fft_preprocessing.py
```

独立して実行できるスクリプトです。既定では `data20260903/data20260903` 直下の元画像21枚を処理し、`outputs/fft_preprocessing_comparison` に保存します。サブフォルダは検索しないため、既存のトリミング画像は対象外です。グレースケール化以外のリサイズ・トリミングは行いません。8/16ビット入力の輝度スケールを保持します。

```powershell
python compare_fft_preprocessing.py --input-dir "画像フォルダ" --output-dir "保存フォルダ"
```

対象拡張子は png, jpg, jpeg, bmp, tif, tiff（大文字も可）です。ファイル名の辞書順（image1, image10, …, image2, …）で読み込みます。複数ページTIFFは先頭画像を使用します。読み込み等に失敗した画像を記録し、残りの画像を続けて処理します。

比較図は左が元画像、右が上から次の3条件です。ファイル名・画像サイズを上部に表示します。

1. `Plain FFT`: グレースケール画像をそのままFFT。
2. `Mean-subtracted FFT`: 全体の平均値を引いてFFT。
3. `Mean-subtracted + Hann-windowed FFT`: 平均値除去後、`np.outer(np.hanning(H), np.hanning(W))` を掛けてFFT。

表示はすべて `np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(data))))` です。3条件を合わせた値の1～99.8パーセンタイルを共通の色範囲にします。上限を超える成分は白く飽和しますが、計算値は変えません。画像ごとに表示範囲は変わります。範囲全体を使う場合は次のように指定します。

```powershell
python compare_fft_preprocessing.py --lower-percentile 0 --upper-percentile 100 --output-dir outputs/fft_preprocessing_full_range
```

出力:

- `元ファイル名_fft_comparison.png`: 3条件の比較図。
- `statistics.csv`: 入力と前処理後のmean/std、DC振幅、DC振幅/画素数、色範囲。
- `report.json`: 処理順、成功・失敗ファイルとエラー内容。

確認するポイント:

- **Plainとの比較**: 中央DC、軸方向の十字、格子の対称なピークについて、位置・幅・周囲への広がりを見る。
- **平均値除去**: DCがほぼゼロになる。窓なしではDC以外は原理的に変わらないため、十字全体や照明むらが消える処理ではない。共通スケールではPlainとほぼ同じに見えるのが自然。
- **Hann窓**: 画像端の不連続によるスペクトル漏れが弱まることを確認する。内部の明暗境界由来の成分は残り得る。ピークは幅が広がり振幅も下がるため、暗くなったことだけを改善とみなさない。端の傷も弱まる。

平均値除去後に窓を掛けると、窓による重み付けで平均が再びゼロでなくなる場合があります。指定した処理順を守るため、窓を掛けた後の平均値再除去や振幅補正は行っていません。
