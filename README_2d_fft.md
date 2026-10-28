# 2次元FFTの表示

`analyze_2d_fft.py` は元画像と周波数空間の画像を並べて表示します。
OpenCV / NumPy / Matplotlib を使用します。

正常輝点のマスク・低周波パワー積分・しきい値判定は
`fft_residual_detector.py` を使用します。手順は [検出器の説明](README_fft_residual_detector.md) を参照してください。

## 実行

PowerShellでこのフォルダに移動して実行します。

```powershell
python -m pip install opencv-python numpy matplotlib
python analyze_2d_fft.py "data20260903/data20260903/image2.png"
```

画像番号で指定する場合（image2、ハン窓・平均値減算をon）:

```powershell
# トリミング済み画像
python analyze_2d_fft.py --image 2 --triming --hann --remove-mean
# トリミング前の画像
python analyze_2d_fft.py --image 2 --hann --remove-mean
# 画面を開かず保存
python analyze_2d_fft.py --image 2 --triming --hann --remove-mean --save outputs/fft_viewer/image2_triming_fft_hann_mean.png --no-show
```

`--image` の数字を変えると別の画像を選べます。
スクリプトと同じフォルダにある `data20260903/data20260903` 内を参照し、
`--triming` を付けると `triming/imageN_triming.png` を読み込みます。
`--trimmed` も同じ意味で使えます。存在しない画像はエラーになります。
画像パスと `--image` は同時に指定できません。
`--hann` と `--remove-mean` は指定するとon、省略するとoffです。

正常領域と傷領域を同じ縦・横画素数で切り出し、比較する例:

```powershell
python analyze_2d_fft.py normal.png scratch.png --remove-mean --hann --save comparison.png
```

`normal.png` と `scratch.png` は利用者が用意する切り出し画像です。
画像を指定した順に上から並べ、ファイルとの対応を端末にも表示します。
元画像は0～255、FFT画像は比較対象全体で共通の表示スケールにします。
撮影倍率・照明・切り出しサイズを揃えて比較してください。
サイズが違う画像はエラーとし、自動リサイズは行いません。

画面を開かずPNGだけ保存する例:

```powershell
python analyze_2d_fft.py "data20260903/data20260903/image2.png" --save outputs/fft_viewer/image2_fft.png --no-show
```

Pythonからは `from analyze_2d_fft import analyze_2d_fft` として、
`analyze_2d_fft('your_image.png')` で呼び出せます。

## 計算と観察のポイント

- 初期設定は提示コードと同じ `log(1 + |FFT|)`（対数振幅スペクトル）です。
  パワースペクトルは `|FFT|**2` で、振幅とは区別します。
- 中央はゼロ周波数（DC）です。平均輝度の影響を弱める場合は `--remove-mean` を使います。
- 規則的な模様は離散的なピークとして現れることがあります。
  線状の傷はその向きに垂直な方向への広がりを生むことがありますが、必ず見えるとは限りません。
- 切り出し境界、照明むら、格子の傾きもスペクトルに影響します。
  明るい筋や広がりだけで傷と断定することはできません。
- `--hann` は境界の影響を弱めますが、端にある傷の寄与も弱め、ピークを広げます。
  両方の画像に同じ設定を適用し、窓なしの結果とも比較してください。
- 画像は8ビットのグレースケールとして読み込みます。
  16ビット画像の元の階調を保持する解析には対応していません。
- 比較図内では表示スケールが共通ですが、別々の実行間ではFFTの上限が変わります。
  明るさを比べる画像は同じコマンドで指定してください。

付属の `outputs/fft_viewer/image2_fft.png` はimage2全体の実行例です。
ブロック1・5・6・7の位置指定や正常／傷の判定は行っていません。
