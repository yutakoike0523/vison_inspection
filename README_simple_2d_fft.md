# 全画像をそのまま2次元FFT

## マウスで周波数成分を選ぶ

```powershell
python interactive_2d_fft.py
python interactive_2d_fft.py --image 15 --trimmed
```

元画像とFFTの窓、選択した成分を表示する窓の2つが開きます。

- FFT上にカーソルを置くと、その周波数成分を空間画像（縞模様）として別窓に表示します。
- 左クリックで成分を固定します。別の場所をクリックすると、固定した成分が加算されます。
- 別窓は上から「カーソルの成分」「固定した合計」「固定した合計＋カーソルの成分」です。
- 対称な周波数は1組として選択します。同じ組を再クリックしても重複加算しません。
- `Undo` / 右クリック / `U` で直前の固定を取り消し、`Clear` / `C` で全解除します。
- ドロップダウンで全37枚を選べます。`Original / Trimmed` で対応画像に切り替えます。切り替えると選択はリセットされます。
- FFTのツールバーで拡大してから、ズームモードを解除すると細かい点を選べます。

1画素の周波数ビンとその共役成分を、元画像の振幅・位相のまま逆変換します。逆FFTと等価な式で計算しています。中央のDCを選ぶと平均輝度が加算されます。負の値もあるため、別窓は青＝負、白＝0、赤＝正で表示し、3枚で共通の自動スケールを使います。表示スケールは変化しますが、加算する数値は正規化しません。

## 全画像の静止画を保存する

```powershell
python batch_simple_2d_fft.py
```

依存パッケージ: `numpy matplotlib opencv-python pillow`。

`data20260903/data20260903/image*.png` と、その `triming` フォルダの既存画像をすべて読み込みます。トリミング画像がない番号も、元画像は処理します。新たな切り出しは行いません。

処理の本体は次の2行です。入力を8ビットのグレースケールにする以外、平均値除去、窓関数、フィルタ、分割、リサイズなどは行いません。

```python
fft = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))
spectrum = np.log1p(np.abs(fft))
```

結果は `outputs/simple_2d_fft/` に保存します。

- `comparisons/imageN_comparison.png`: 上段が元画像とFFT、下段がトリミング画像とFFT。トリミング画像がない番号は上段のみ。
- `overview_original.png`: 元画像のFFT一覧。
- `overview_trimmed.png`: トリミング画像のFFT一覧。
- `original/`、`trimmed/`: 入力と同じ画素数のFFT表示PNG。
- `report.json`: 入力ファイル、画像サイズ、処理数、表示スケール。

表示するのは自然対数による振幅 `log(1 + |F|)` です。中央がゼロ周波数（DC）です。全画像で同じ明るさスケールを使用します。FFTの振幅は正規化していないので、画像の画素数や平均輝度によっても変わります。表示PNGは可視化用で、生の複素FFT値ではありません。一覧や比較図だけ表示時に縮小されます。

保存先や入力フォルダを変える場合:

```powershell
python batch_simple_2d_fft.py --input-dir data20260903/data20260903 --output-dir outputs/simple_2d_fft
```
