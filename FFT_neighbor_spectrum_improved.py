#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
局所的な2次元FFTパワースペクトルを、周辺ブロックの中央値スペクトルと比較する。

複素FFTの位相は比較せず、平行移動の影響を受けにくいパワーを比較する。
各ブロックの平均輝度除去は、直流成分を抑えるために残す。
"""

from pathlib import Path
import argparse
import os
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import fft
from scipy.ndimage import gaussian_filter
from numpy.lib.stride_tricks import sliding_window_view

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_PATH = SCRIPT_DIR / "data20260903" / "data20260903" / "image15.png"
OUTPUT_DIR = SCRIPT_DIR / "output_neighbor_spectrum"

# 1回のFFTに使う局所画像の大きさ
WINDOW_SIZE = 128

# 判定位置の間隔。stride=2は非常に重いため、新コードでは32 pxとする。
ANALYSIS_STRIDE = 32
BATCH_SIZE = 64
FFT_WORKERS = -1

# 周辺基準：64 px間隔で5×5地点、中央を除く最大24ブロック。
REFERENCE_SPACING_PIXELS = 64
REFERENCE_RADIUS_STEPS = 2

# DC近傍を除外し、この周波数範囲でスペクトル差を評価する。
FREQ_MIN = 0.02
FREQ_MAX = 0.50
EPSILON = np.float32(1e-12)
SMOOTH_SIGMA = 0.0
MIN_NEIGHBORS = 6


def setup_japanese_font():
    candidates_by_os = {
        "Windows": ["Yu Gothic", "Meiryo", "MS Gothic"],
        "Darwin": ["Hiragino Sans", "Hiragino Kaku Gothic ProN", "YuGothic"],
        "Linux": ["Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic"],
    }
    installed = {font.name for font in fm.fontManager.ttflist}
    for name in candidates_by_os.get(platform.system(), []):
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def make_frequency_grid(window_size):
    """rfft2に対応する周波数座標と評価マスクを作る。"""
    fy = np.fft.fftshift(np.fft.fftfreq(window_size)).astype(np.float32)
    fx = np.fft.rfftfreq(window_size).astype(np.float32)
    fy_grid, fx_grid = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(fx_grid**2 + fy_grid**2)
    mask = (radius >= FREQ_MIN) & (radius <= FREQ_MAX)
    return fy, fx, mask


def calc_log_power_batch(patches, hann_2d, frequency_mask):
    """
    局所画像群を正規化ログパワースペクトルに変換する。

    総パワーで正規化することで、単純なコントラスト差の影響を抑える。
    """
    values = np.asarray(patches, dtype=np.float32).copy()
    values -= values.mean(axis=(-2, -1), keepdims=True)
    values *= hann_2d[None, :, :]

    spectrum = fft.rfft2(values, axes=(-2, -1), workers=FFT_WORKERS)
    spectrum = fft.fftshift(spectrum, axes=(-2,))
    power = (np.abs(spectrum) ** 2).astype(np.float32)

    total_power = power[:, frequency_mask].sum(axis=1, keepdims=True)
    total_power = np.maximum(total_power, EPSILON)
    power /= total_power[:, :, None]

    log_power = (10.0 * np.log10(power + EPSILON)).astype(np.float32)
    if SMOOTH_SIGMA > 0:
        # Smooth frequency axes only; fy wraps, fx uses the half-spectrum edge.
        log_power = gaussian_filter(log_power, sigma=(0, SMOOTH_SIGMA, SMOOTH_SIGMA),
                                    mode=("nearest", "wrap", "reflect"))
    return log_power


def calc_all_spectra(img_gray, hann_2d, frequency_mask):
    """画像全体を走査し、各判定位置の2次元ログパワーを作る。"""
    height, width = img_gray.shape
    y0_list = np.arange(0, height - WINDOW_SIZE + 1, ANALYSIS_STRIDE, dtype=int)
    x0_list = np.arange(0, width - WINDOW_SIZE + 1, ANALYSIS_STRIDE, dtype=int)

    spectra = np.empty(
        (len(y0_list), len(x0_list), WINDOW_SIZE, WINDOW_SIZE // 2 + 1),
        dtype=np.float32,
    )

    iterator = y0_list
    if tqdm is not None:
        iterator = tqdm(y0_list, desc="2次元FFT")

    for iy, y0 in enumerate(iterator):
        band = img_gray[y0 : y0 + WINDOW_SIZE, :]
        windows = sliding_window_view(band, window_shape=WINDOW_SIZE, axis=1)
        windows = np.moveaxis(windows, 1, 0)[::ANALYSIS_STRIDE]

        for start in range(0, len(x0_list), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(x0_list))
            spectra[iy, start:end] = calc_log_power_batch(
                windows[start:end], hann_2d, frequency_mask
            )

    x_centers = x0_list + WINDOW_SIZE / 2
    y_centers = y0_list + WINDOW_SIZE / 2
    return spectra, x_centers, y_centers


def make_reference_offsets():
    """5×5の参照位置（中央を除く）を特徴マップ上のオフセットで返す。"""
    if REFERENCE_SPACING_PIXELS % ANALYSIS_STRIDE != 0:
        raise ValueError("REFERENCE_SPACING_PIXELSはANALYSIS_STRIDEの倍数にしてください。")

    index_step = REFERENCE_SPACING_PIXELS // ANALYSIS_STRIDE
    offsets = []
    for gy in range(-REFERENCE_RADIUS_STEPS, REFERENCE_RADIUS_STEPS + 1):
        for gx in range(-REFERENCE_RADIUS_STEPS, REFERENCE_RADIUS_STEPS + 1):
            if gy == 0 and gx == 0:
                continue
            offsets.append((gy * index_step, gx * index_step))
    return offsets


def calc_neighbor_anomaly(spectra, frequency_mask, valid_windows):
    """
    周辺スペクトルの中央値を基準に、RMS dB差を計算する。

    画像端では画像内にある参照ブロックだけを使う。
    """
    ny, nx = spectra.shape[:2]
    offsets = make_reference_offsets()
    anomaly_map = np.full((ny, nx), np.nan, dtype=np.float32)

    neighbor_counts = np.zeros((ny, nx), dtype=np.int16)
    max_score = -np.inf
    max_info = None
    positions = range(ny)
    if tqdm is not None:
        positions = tqdm(positions, desc="周辺スペクトル比較")

    for iy in positions:
        for ix in range(nx):
            if not valid_windows[iy, ix]:
                continue
            neighbors = []
            for dy, dx in offsets:
                jy, jx = iy + dy, ix + dx
                if 0 <= jy < ny and 0 <= jx < nx and valid_windows[jy, jx]:
                    neighbors.append(spectra[jy, jx])

            neighbor_counts[iy, ix] = len(neighbors)
            if len(neighbors) < MIN_NEIGHBORS:
                continue

            reference = np.median(np.stack(neighbors, axis=0), axis=0)
            difference = spectra[iy, ix] - reference
            score = np.sqrt(np.mean(difference[frequency_mask] ** 2))
            anomaly_map[iy, ix] = score

            if score > max_score:
                max_score = float(score)
                max_info = {
                    "iy": iy,
                    "ix": ix,
                    "target": spectra[iy, ix].copy(),
                    "reference": reference.astype(np.float32, copy=False),
                    "difference": difference.astype(np.float32, copy=False),
                    "neighbor_count": len(neighbors),
                }

    return anomaly_map, max_info, neighbor_counts


def save_anomaly_map(anomaly_map, x_centers, y_centers):
    path = OUTPUT_DIR / "neighbor_spectrum_anomaly_map.png"
    fig, ax = plt.subplots(figsize=(12, 8))
    image = ax.imshow(
        anomaly_map,
        origin="upper",
        extent=[x_centers[0], x_centers[-1], y_centers[-1], y_centers[0]],
        aspect="equal",
        cmap="inferno",
    )
    fig.colorbar(image, ax=ax, label="周辺基準との2次元スペクトル差［RMS dB］")
    ax.set(xlabel="x［pixel］", ylabel="y［pixel］", title="周辺ブロック基準のスペクトル異常度")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_anomaly_overlay(img_gray, anomaly_map, x_centers, y_centers):
    path = OUTPUT_DIR / "neighbor_spectrum_anomaly_overlay.png"
    height, width = img_gray.shape
    finite = anomaly_map[np.isfinite(anomaly_map)]
    vmax = float(np.percentile(finite, 99))
    vmin = float(np.percentile(finite, 50))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_gray, cmap="gray", origin="upper", extent=[0, width, height, 0])
    overlay = ax.imshow(
        anomaly_map,
        cmap="inferno",
        alpha=0.55,
        origin="upper",
        extent=[x_centers[0], x_centers[-1], y_centers[-1], y_centers[0]],
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    fig.colorbar(overlay, ax=ax, label="異常度［RMS dB］")
    ax.set(xlabel="x［pixel］", ylabel="y［pixel］", title="原画像とスペクトル異常度の重ね合わせ")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_max_spectrum_comparison(max_info, fy, fx, x_centers, y_centers):
    path = OUTPUT_DIR / "max_anomaly_spectrum_comparison.png"
    target = max_info["target"]
    reference = max_info["reference"]
    absolute_difference = np.abs(max_info["difference"])

    common = np.concatenate([target.ravel(), reference.ravel()])
    vmin, vmax = np.percentile(common, [5, 99.5])
    diff_vmax = np.percentile(absolute_difference, 99.5)
    extent = [float(fx[0]), float(fx[-1]), float(fy[0]), float(fy[-1])]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    im0 = axes[0].imshow(target, origin="lower", extent=extent, aspect="auto", vmin=vmin, vmax=vmax)
    axes[0].set_title("判定対象")
    axes[1].imshow(reference, origin="lower", extent=extent, aspect="auto", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"周辺の中央値（{max_info['neighbor_count']}ブロック）")
    im2 = axes[2].imshow(
        absolute_difference,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="magma",
        vmin=0,
        vmax=diff_vmax,
    )
    axes[2].set_title("絶対差")

    for ax in axes:
        ax.set_xlabel("fx［cycles/pixel］")
        ax.set_ylabel("fy［cycles/pixel］")
    fig.colorbar(im0, ax=axes[:2], label="正規化ログパワー［dB］", shrink=0.85)
    fig.colorbar(im2, ax=axes[2], label="スペクトル差［dB］", shrink=0.85)

    ix, iy = max_info["ix"], max_info["iy"]
    fig.suptitle(f"最大異常位置 x={x_centers[ix]:.1f}, y={y_centers[iy]:.1f}")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def window_validity(image_shape, mask_paths):
    """Reject a window when any pixel overlaps a nonzero exclusion mask."""
    height, width = image_shape
    excluded = np.zeros((height, width), dtype=bool)
    for path in mask_paths:
        mask = np.asarray(Image.open(path).convert("L"))
        if mask.shape != image_shape:
            raise ValueError(f"Mask size {mask.shape} does not match image {image_shape}: {path}")
        excluded |= mask > 0
    integral = np.pad(excluded.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    ys = np.arange(0, height - WINDOW_SIZE + 1, ANALYSIS_STRIDE)
    xs = np.arange(0, width - WINDOW_SIZE + 1, ANALYSIS_STRIDE)
    y, x = np.meshgrid(ys, xs, indexing="ij")
    sums = (integral[y + WINDOW_SIZE, x + WINDOW_SIZE] - integral[y, x + WINDOW_SIZE]
            - integral[y + WINDOW_SIZE, x] + integral[y, x])
    return sums == 0


def main():
    global IMAGE_PATH, OUTPUT_DIR, FREQ_MIN, FREQ_MAX, SMOOTH_SIGMA, MIN_NEIGHBORS
    global ANALYSIS_STRIDE, BATCH_SIZE, REFERENCE_RADIUS_STEPS
    parser = argparse.ArgumentParser(description="Compare local FFT spectra with neighboring windows")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--image", type=Path, help="Process a single image")
    mode.add_argument("--all", action="store_true", help="Process every image in the input directory")
    parser.add_argument("--image-dir", type=Path, default=SCRIPT_DIR / "data20260903" / "data20260903",
                        help="Input directory used when --image is omitted")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--exclude-mask", type=Path, action="append", default=[],
                        help="Exclude nonzero mask pixels (repeatable; same mask for every image)")
    parser.add_argument("--freq-min", type=float, default=FREQ_MIN)
    parser.add_argument("--freq-max", type=float, default=FREQ_MAX)
    parser.add_argument("--smooth-sigma", type=float, default=0.0,
                        help="Frequency smoothing sigma in FFT bins")
    parser.add_argument("--min-neighbors", type=int, default=MIN_NEIGHBORS)
    parser.add_argument("--stride", type=int, help="Analysis stride in pixels (must divide 64)")
    parser.add_argument("--batch-size", type=int, help="FFT batch size")
    parser.add_argument("--neighborhood", type=int, help="Odd side length of the square reference grid")
    args = parser.parse_args()
    if args.image is None and not args.all:
        print("処理する画像を選択してください。")
        print("1: フォルダー内の全画像")
        print("2: 1つの画像")
        while True:
            choice = input("選択 [1/2]: ").strip()
            if choice == "1":
                args.all = True
                break
            if choice == "2":
                number = input("画像番号（例: 15）: ").strip()
                if not number.isdecimal() or int(number) < 1:
                    print("1以上の画像番号を入力してください。")
                    continue
                args.image = args.image_dir / f"image{int(number)}.png"
                break
            print("1 または 2 を入力してください。")
    def choose_integer(value, prompt, default, valid, explanation):
        if value is not None:
            if not valid(value):
                parser.error(explanation)
            return value
        while True:
            raw = input(f"{prompt} [{default}]: ").strip()
            try:
                chosen = int(raw) if raw else default
                if valid(chosen):
                    return chosen
            except ValueError:
                pass
            print(explanation)

    ANALYSIS_STRIDE = choose_integer(args.stride, "ストライド（px）", 32,
        lambda n: n > 0 and REFERENCE_SPACING_PIXELS % n == 0,
        "ストライドは64を割り切れる正の整数にしてください。")
    BATCH_SIZE = choose_integer(args.batch_size, "バッチサイズ", 64,
        lambda n: n > 0, "バッチサイズは正の整数にしてください。")
    neighborhood = choose_integer(args.neighborhood, "周辺ブロックの縦横数（例: 5 で 5×5）", 5,
        lambda n: n >= 3 and n % 2 == 1,
        "周辺ブロックの縦横数は3以上の奇数にしてください。")
    REFERENCE_RADIUS_STEPS = neighborhood // 2
    FREQ_MIN, FREQ_MAX = args.freq_min, args.freq_max
    SMOOTH_SIGMA, MIN_NEIGHBORS = args.smooth_sigma, args.min_neighbors
    if not (0 <= FREQ_MIN < FREQ_MAX <= 0.5):
        parser.error("Require 0 <= freq-min < freq-max <= 0.5")
    if SMOOTH_SIGMA < 0 or not 1 <= MIN_NEIGHBORS <= neighborhood ** 2 - 1:
        parser.error("Require smooth-sigma >= 0 and min-neighbors between 1 and neighborhood² - 1")
    setup_japanese_font()
    if args.image is not None:
        images = [args.image]
    else:
        if not args.image_dir.is_dir():
            parser.error(f"Input directory does not exist: {args.image_dir}")
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        images = sorted((p for p in args.image_dir.iterdir()
                         if p.is_file() and p.suffix.lower() in extensions),
                        key=lambda p: p.name.lower())
        if not images:
            parser.error(f"No supported images in {args.image_dir}")
    failures = []
    stems = [p.stem.lower() for p in images]
    for index, image_path in enumerate(images, 1):
        IMAGE_PATH = image_path
        folder = image_path.name if stems.count(image_path.stem.lower()) > 1 else image_path.stem
        OUTPUT_DIR = args.output / folder if args.image is None else args.output
        print(f"\n[{index}/{len(images)}] {image_path}", flush=True)
        try:
            process_image(args.exclude_mask)
        except Exception as exc:
            failures.append((image_path, exc))
            print(f"Error processing {image_path}: {exc}", flush=True)
    print(f"Completed: {len(images) - len(failures)}/{len(images)}")
    if failures:
        raise SystemExit(1)


def process_image(exclude_masks):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"入力画像が見つかりません: {IMAGE_PATH}")

    img_gray = np.asarray(Image.open(IMAGE_PATH).convert("L"), dtype=np.float32)
    height, width = img_gray.shape
    if height < WINDOW_SIZE or width < WINDOW_SIZE:
        raise ValueError("入力画像がWINDOW_SIZEより小さいです。")

    hann_1d = np.hanning(WINDOW_SIZE).astype(np.float32)
    hann_2d = np.outer(hann_1d, hann_1d).astype(np.float32)
    fy, fx, frequency_mask = make_frequency_grid(WINDOW_SIZE)

    print("入力:", IMAGE_PATH)
    print("画像サイズ:", width, "x", height)
    print("解析stride:", ANALYSIS_STRIDE, "pixel")
    side = 2 * REFERENCE_RADIUS_STEPS + 1
    print("バッチサイズ:", BATCH_SIZE)
    print(f"周辺参照: {side}×{side}（中央を除き最大{side * side - 1}ブロック） /", REFERENCE_SPACING_PIXELS, "pixel間隔")

    spectra, x_centers, y_centers = calc_all_spectra(
        img_gray, hann_2d, frequency_mask
    )
    print("スペクトル配列:", spectra.shape, f"({spectra.nbytes / 1024**2:.1f} MiB)")

    valid_windows = window_validity(img_gray.shape, exclude_masks)
    anomaly_map, max_info, neighbor_counts = calc_neighbor_anomaly(spectra, frequency_mask, valid_windows)
    if max_info is None:
        raise RuntimeError("周辺基準を作成できませんでした。")

    paths = [
        save_anomaly_map(anomaly_map, x_centers, y_centers),
        save_anomaly_overlay(img_gray, anomaly_map, x_centers, y_centers),
        save_max_spectrum_comparison(max_info, fy, fx, x_centers, y_centers),
    ]

    npz_path = OUTPUT_DIR / "neighbor_spectrum_result.npz"
    np.savez_compressed(
        npz_path,
        anomaly_map=anomaly_map,
        x_centers=x_centers,
        y_centers=y_centers,
        window_size=np.int32(WINDOW_SIZE),
        analysis_stride=np.int32(ANALYSIS_STRIDE),
        batch_size=np.int32(BATCH_SIZE),
        reference_spacing_pixels=np.int32(REFERENCE_SPACING_PIXELS),
        reference_radius_steps=np.int32(REFERENCE_RADIUS_STEPS),
        freq_min=np.float32(FREQ_MIN),
        freq_max=np.float32(FREQ_MAX),
        valid_windows=valid_windows,
        neighbor_counts=neighbor_counts,
        min_neighbors=np.int32(MIN_NEIGHBORS),
        smooth_sigma=np.float32(SMOOTH_SIGMA),
        exclusion_masks=np.asarray([str(p) for p in exclude_masks]),
    )

    print("出力先:", OUTPUT_DIR)
    for path in paths:
        print(" -", path.name)
    print(" -", npz_path.name)


if __name__ == "__main__":
    main()
