"""Display grayscale images and their log-amplitude 2-D FFT spectra.

Examples:
    python analyze_2d_fft.py --image 2 --triming --hann --remove-mean
    python analyze_2d_fft.py image.png
    python analyze_2d_fft.py normal.png scratch.png --remove-mean --hann
    python analyze_2d_fft.py image.png --save fft.png --no-show
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def read_grayscale(image_path):
    """Read Unicode paths on Windows, including Japanese filenames."""
    path = Path(image_path)
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if data.size else None
    if img is None:
        raise ValueError(f"画像を読み込めません: {path}")
    if min(img.shape) < 3:
        raise ValueError("縦・横ともに3画素以上の画像を指定してください。")
    return img


def log_amplitude_spectrum(img, *, remove_mean=False, hann=False):
    """Return log(1 + abs(fftshift(fft2(img)))). This is not power."""
    data = img.astype(np.float64)
    if remove_mean:
        data -= data.mean()
    if hann:
        # 切り出し境界での不連続を弱める（画像端の傷も弱まる）。
        data *= np.outer(np.hanning(data.shape[0]), np.hanning(data.shape[1]))
    shifted = np.fft.fftshift(np.fft.fft2(data))
    return np.log1p(np.abs(shifted))


def analyze_2d_fft(image_path, *, compare_paths=(), remove_mean=False,
                   hann=False, save_path=None, show=True):
    """Display one image or compare equally sized crops using shared scales.

    Returns the Matplotlib figure. Input is decoded as 8-bit grayscale.
    When show=False, the caller can close the figure after use.
    """
    import matplotlib.pyplot as plt

    paths = [Path(image_path), *(Path(p) for p in compare_paths)]
    images = [read_grayscale(p) for p in paths]
    if any(img.shape != images[0].shape for img in images):
        raise ValueError("比較画像は縦・横の画素数を揃えて切り出してください。")
    spectra = [log_amplitude_spectrum(img, remove_mean=remove_mean, hann=hann)
               for img in images]
    # 画像ごとの自動調整を避け、すべて同じ明るさスケールで表示する。
    upper = max(max(float(s.max()) for s in spectra), 1e-12)
    fig, axes = plt.subplots(len(images), 2, figsize=(11, 4 * len(images)),
                             squeeze=False, layout="constrained")
    for row, (img, spectrum) in enumerate(zip(images, spectra)):
        axes[row, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title(f"Original image {row + 1} ({img.shape[1]} x {img.shape[0]})")
        axes[row, 0].axis("off")
        plotted = axes[row, 1].imshow(spectrum, cmap="gray", vmin=0,
                                      vmax=upper, interpolation="nearest")
        axes[row, 1].set_title(f"2D FFT: log(1 + |F|), image {row + 1}")
        axes[row, 1].axis("off")
    fig.colorbar(plotted, ax=list(axes[:, 1]), label="Log amplitude", shrink=0.8)
    fig.suptitle(f"Mean subtraction: {'on' if remove_mean else 'off'} | "
                 f"Hann window: {'on' if hann else 'off'}")
    for index, path in enumerate(paths, start=1):
        print(f"Image {index}: {path}")
    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160)
        print(f"Saved: {output.resolve()}")
    if show:
        plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description="画像と2次元FFTの対数振幅を表示します。")
    parser.add_argument("images", nargs="*", help="画像パス。複数指定時は同じ画素数で比較")
    parser.add_argument("--image", type=int, metavar="N", help="画像番号（例: 2 → image2.png）")
    parser.add_argument("--triming", "--trimmed", action="store_true",
                        help="--imageで指定した番号のトリミング済み画像を使用")
    parser.add_argument("--remove-mean", action="store_true", help="FFT前に平均輝度を引く")
    parser.add_argument("--hann", action="store_true", help="FFT前に2次元ハン窓を適用")
    parser.add_argument("--save", type=Path, help="比較図の保存先（例: fft.png）")
    parser.add_argument("--no-show", action="store_true", help="画面表示を省略（--saveと併用）")
    args = parser.parse_args()
    if args.image is not None:
        if args.images:
            parser.error("画像パスと --image は同時に指定できません。")
        if args.image < 1:
            parser.error("--image には1以上の画像番号を指定してください。")
        data_dir = Path(__file__).resolve().parent / "data20260903" / "data20260903"
        if args.triming:
            args.images = [data_dir / "triming" / f"image{args.image}_triming.png"]
        else:
            args.images = [data_dir / f"image{args.image}.png"]
    elif args.triming:
        parser.error("--triming は --image と一緒に指定してください。")
    elif not args.images:
        parser.error("画像パス、または --image N を指定してください。")
    if args.no_show and args.save is None:
        parser.error("--no-show を指定する場合は --save も指定してください。")
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    try:
        fig = analyze_2d_fft(args.images[0], compare_paths=args.images[1:],
                             remove_mean=args.remove_mean, hann=args.hann,
                             save_path=args.save, show=not args.no_show)
    except (OSError, ValueError, cv2.error) as exc:
        parser.exit(1, f"エラー: {exc}\n")
    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__ == "__main__":
    main()
