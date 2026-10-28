#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fft_radial_volume.py

画像を n x n の窓で走査し，各位置で2次元FFTを計算して，
半径方向（radial）のスペクトルプロファイルを保存するスクリプト．

主な機能
--------
1. compute
   - 画像を n x n の正方領域で走査
   - stride 指定可能
   - multiprocessing による並列化
   - 各窓の2次元FFT power spectrumから radial profile を計算
   - profile長 = n // 2 + 1
   - NPZ形式で保存

2. view
   - 保存した NPZ を開く
   - 任意の周波数bin f の (x,y) マップを表示
   - 原画像へカラーマップで重畳
   - alpha，cmap，vmin，vmax 等を指定可能

注意
----
ここで f は「radial frequency bin」です．
f = 0, 1, ..., n//2

窓中心の座標は，
    x_center = x_start + (n - 1) / 2
    y_center = y_start + (n - 1) / 2
です．

偶数 n の場合，中心は半画素位置になります．
"""

import argparse
import os
import sys
from pathlib import Path
from multiprocessing import Pool, shared_memory, cpu_count

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# multiprocessing 用グローバル変数
# ============================================================

_G_IMAGE = None
_G_SHM = None
_G_SHAPE = None
_G_DTYPE = None

_G_N = None
_G_STRIDE = None
_G_X_STARTS = None
_G_RADIAL_INDEX = None
_G_RADIAL_COUNT = None
_G_WINDOW = None
_G_USE_DB = None
_G_EPS = None


def load_grayscale_image(filename):
    """
    画像をグレースケールfloat32で読み込む．
    """
    img = Image.open(filename)
    img = img.convert("F")
    arr = np.asarray(img, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError("2次元グレースケール画像として読み込めませんでした．")

    return arr


def make_window(n, window_name):
    """
    2次元窓関数を作成する．
    """
    if window_name == "none":
        return np.ones((n, n), dtype=np.float32)

    if window_name == "hann":
        w = np.hanning(n).astype(np.float32)
        return np.outer(w, w).astype(np.float32)

    if window_name == "hamming":
        w = np.hamming(n).astype(np.float32)
        return np.outer(w, w).astype(np.float32)

    if window_name == "blackman":
        w = np.blackman(n).astype(np.float32)
        return np.outer(w, w).astype(np.float32)

    raise ValueError(f"未知のwindowです: {window_name}")


def make_radial_lookup(n):
    """
    fftshift後の n x n 周波数平面について，
    各画素がどのradial binに入るかをあらかじめ計算する．

    採用するbinは 0 ～ n//2 のみ．
    角の領域は半径がn//2を超えるため除外する．

    Returns
    -------
    radial_index_valid : 1次元int配列
        有効画素だけのradial bin番号．
    valid_mask_flat : 1次元bool配列
        flattenしたFFT平面に対する有効画素mask．
    radial_count : 各binに含まれる画素数
    """
    fy = np.fft.fftshift(np.fft.fftfreq(n) * n)
    fx = np.fft.fftshift(np.fft.fftfreq(n) * n)

    xx, yy = np.meshgrid(fx, fy)

    rr = np.sqrt(xx * xx + yy * yy)

    # 最近傍整数半径へ割り当て
    radial_index = np.rint(rr).astype(np.int32)

    max_bin = n // 2
    valid = radial_index <= max_bin

    idx_valid = radial_index.ravel()[valid.ravel()]

    count = np.bincount(idx_valid, minlength=max_bin + 1).astype(np.float64)

    return idx_valid, valid.ravel(), count


def radial_profile_from_patch(patch):
    """
    1個の n x n patch からradial profileを計算する．
    """
    global _G_WINDOW, _G_RADIAL_INDEX, _G_RADIAL_COUNT
    global _G_USE_DB, _G_EPS, _G_VALID_MASK_FLAT

    # DC差の影響を減らしたい場合は，ここで平均値を引く
    # patch = patch - patch.mean()

    patch = patch * _G_WINDOW

    fft = np.fft.fft2(patch)
    power = np.abs(np.fft.fftshift(fft)) ** 2

    values = power.ravel()[_G_VALID_MASK_FLAT]

    sums = np.bincount(
        _G_RADIAL_INDEX,
        weights=values,
        minlength=len(_G_RADIAL_COUNT)
    )

    profile = sums / _G_RADIAL_COUNT

    if _G_USE_DB:
        profile = 10.0 * np.log10(profile + _G_EPS)

    return profile.astype(np.float32, copy=False)


def _worker_init(
    shm_name,
    shape,
    dtype_str,
    n,
    stride,
    x_starts,
    radial_index,
    valid_mask_flat,
    radial_count,
    window,
    use_db,
    eps,
):
    """
    各workerの初期化．
    元画像はshared_memory経由で共有する．
    """
    global _G_IMAGE, _G_SHM, _G_SHAPE, _G_DTYPE
    global _G_N, _G_STRIDE, _G_X_STARTS
    global _G_RADIAL_INDEX, _G_RADIAL_COUNT, _G_VALID_MASK_FLAT
    global _G_WINDOW, _G_USE_DB, _G_EPS

    _G_SHM = shared_memory.SharedMemory(name=shm_name)
    _G_SHAPE = tuple(shape)
    _G_DTYPE = np.dtype(dtype_str)

    _G_IMAGE = np.ndarray(
        _G_SHAPE,
        dtype=_G_DTYPE,
        buffer=_G_SHM.buf
    )

    _G_N = int(n)
    _G_STRIDE = int(stride)
    _G_X_STARTS = np.asarray(x_starts, dtype=np.int32)

    _G_RADIAL_INDEX = np.asarray(radial_index, dtype=np.int32)
    _G_VALID_MASK_FLAT = np.asarray(valid_mask_flat, dtype=bool)
    _G_RADIAL_COUNT = np.asarray(radial_count, dtype=np.float64)

    _G_WINDOW = np.asarray(window, dtype=np.float32)
    _G_USE_DB = bool(use_db)
    _G_EPS = float(eps)


def _process_y_index(iy):
    """
    出力の1行分をworkerで計算する．

    Returns
    -------
    iy : int
    row_profiles : shape = (Nx, F)
    """
    y0 = iy * _G_STRIDE

    nx = len(_G_X_STARTS)
    nf = _G_N // 2 + 1

    row = np.empty((nx, nf), dtype=np.float32)

    for ix, x0 in enumerate(_G_X_STARTS):
        patch = _G_IMAGE[y0:y0 + _G_N, x0:x0 + _G_N]
        row[ix, :] = radial_profile_from_patch(patch)

    return iy, row


def compute_volume(args):
    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    image = load_grayscale_image(image_path)

    h, w = image.shape
    n = args.n
    stride = args.stride

    if n <= 1:
        raise ValueError("n は2以上にしてください．")

    if stride <= 0:
        raise ValueError("stride は1以上にしてください．")

    if n > h or n > w:
        raise ValueError(
            f"window n={n} が画像サイズ {w} x {h} より大きいです．"
        )

    x_starts = np.arange(0, w - n + 1, stride, dtype=np.int32)
    y_starts = np.arange(0, h - n + 1, stride, dtype=np.int32)

    nx = len(x_starts)
    ny = len(y_starts)
    nf = n // 2 + 1

    print("========================================")
    print("FFT radial volume computation")
    print("========================================")
    print(f"image       : {image_path}")
    print(f"image size  : {w} x {h}")
    print(f"window      : {n} x {n}")
    print(f"stride      : {stride}")
    print(f"x positions : {nx}")
    print(f"y positions : {ny}")
    print(f"freq bins   : {nf}")
    print(f"workers     : {args.workers}")
    print(f"window func : {args.window}")
    print(f"dB          : {args.db}")
    print()

    # 出力データサイズ概算
    size_bytes = ny * nx * nf * np.dtype(np.float32).itemsize
    print(
        "output size : "
        f"{size_bytes / (1024 ** 2):.1f} MiB "
        f"({size_bytes / (1024 ** 3):.2f} GiB)"
    )
    print()

    # radial lookup
    radial_index, valid_mask_flat, radial_count = make_radial_lookup(n)
    window = make_window(n, args.window)

    # shared memoryへ画像をコピー
    shm = shared_memory.SharedMemory(create=True, size=image.nbytes)

    try:
        shm_array = np.ndarray(
            image.shape,
            dtype=image.dtype,
            buffer=shm.buf
        )
        shm_array[:] = image

        # 結果配列
        volume = np.empty((ny, nx, nf), dtype=np.float32)

        workers = args.workers
        if workers <= 0:
            workers = max(1, cpu_count() - 1)

        workers = min(workers, ny)

        chunksize = args.chunksize
        if chunksize <= 0:
            chunksize = max(1, ny // (workers * 8))

        print(f"actual workers : {workers}")
        print(f"chunksize      : {chunksize}")
        print()

        with Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(
                shm.name,
                image.shape,
                image.dtype.str,
                n,
                stride,
                x_starts,
                radial_index,
                valid_mask_flat,
                radial_count,
                window,
                args.db,
                args.eps,
            )
        ) as pool:

            done = 0

            for iy, row in pool.imap_unordered(
                _process_y_index,
                range(ny),
                chunksize=chunksize
            ):
                volume[iy, :, :] = row
                done += 1

                if done == 1 or done % max(1, ny // 100) == 0 or done == ny:
                    percent = 100.0 * done / ny
                    print(
                        f"\rprogress : {done}/{ny} "
                        f"({percent:6.2f}%)",
                        end="",
                        flush=True
                    )

        print()
        print()

        # 窓中心の画像座標
        x_centers = x_starts.astype(np.float32) + (n - 1) / 2.0
        y_centers = y_starts.astype(np.float32) + (n - 1) / 2.0

        # 周波数bin
        freq_bins = np.arange(nf, dtype=np.int32)

        output = Path(args.output)

        if output.suffix.lower() != ".npz":
            output = output.with_suffix(".npz")

        save_kwargs = dict(
            radial=volume,
            image=image.astype(np.float32),
            x_starts=x_starts,
            y_starts=y_starts,
            x_centers=x_centers,
            y_centers=y_centers,
            freq_bins=freq_bins,
            n=np.int32(n),
            stride=np.int32(stride),
            source_image=np.array(str(image_path)),
            window=np.array(args.window),
            db=np.bool_(args.db),
        )

        if args.compressed:
            np.savez_compressed(output, **save_kwargs)
        else:
            np.savez(output, **save_kwargs)

        print(f"saved : {output}")
        print(f"radial shape : {volume.shape}")
        print()
        print(
            "radial[y_index, x_index, f] が，"
            "その位置における周波数bin f のスペクトルです．"
        )

    finally:
        shm.close()
        shm.unlink()


def percentile_range(data, low=1.0, high=99.0):
    finite = data[np.isfinite(data)]

    if finite.size == 0:
        return None, None

    return (
        float(np.percentile(finite, low)),
        float(np.percentile(finite, high))
    )


def view_volume(args):
    npz_path = Path(args.npz)

    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path, allow_pickle=False)

    radial = data["radial"]
    image = data["image"]
    x_centers = data["x_centers"]
    y_centers = data["y_centers"]

    n = int(data["n"])
    stride = int(data["stride"])
    is_db = bool(data["db"])

    f = args.f

    if f < 0 or f >= radial.shape[2]:
        raise ValueError(
            f"f={f} は範囲外です．"
            f"指定可能範囲は 0 ～ {radial.shape[2] - 1} です．"
        )

    freq_map = radial[:, :, f]

    if args.vmin is None or args.vmax is None:
        auto_vmin, auto_vmax = percentile_range(
            freq_map,
            args.pmin,
            args.pmax
        )
    else:
        auto_vmin = args.vmin
        auto_vmax = args.vmax

    vmin = args.vmin if args.vmin is not None else auto_vmin
    vmax = args.vmax if args.vmax is not None else auto_vmax

    h, w = image.shape

    # 各スペクトル値を窓中心座標へ対応させて表示
    #
    # imshowではextentの境界を与える必要があるため，
    # pixel centerから半stride分だけ外側へ広げる．
    if len(x_centers) > 1:
        dx = float(x_centers[1] - x_centers[0])
    else:
        dx = float(stride)

    if len(y_centers) > 1:
        dy = float(y_centers[1] - y_centers[0])
    else:
        dy = float(stride)

    extent = [
        float(x_centers[0] - dx / 2.0),
        float(x_centers[-1] + dx / 2.0),
        float(y_centers[-1] + dy / 2.0),
        float(y_centers[0] - dy / 2.0),
    ]

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.imshow(
        image,
        cmap="gray",
        origin="upper",
        extent=[-0.5, w - 0.5, h - 0.5, -0.5],
        interpolation="nearest"
    )

    overlay = ax.imshow(
        freq_map,
        cmap=args.cmap,
        alpha=args.alpha,
        origin="upper",
        extent=extent,
        interpolation=args.interpolation,
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(overlay, ax=ax)

    if is_db:
        cbar.set_label("Radial FFT power [dB]")
    else:
        cbar.set_label("Radial FFT power")

    ax.set_title(
        f"Radial FFT map : f={f}, "
        f"window={n}x{n}, stride={stride}"
    )
    ax.set_xlabel("x [pixel]")
    ax.set_ylabel("y [pixel]")

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)

    plt.tight_layout()

    if args.save is not None:
        save_path = Path(args.save)
        fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
        print(f"saved figure : {save_path}")

    if not args.no_show:
        plt.show()


def info_volume(args):
    npz_path = Path(args.npz)

    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path, allow_pickle=False)

    radial = data["radial"]
    image = data["image"]

    print("========================================")
    print("NPZ information")
    print("========================================")
    print(f"file          : {npz_path}")
    print(f"image shape   : {image.shape}")
    print(f"radial shape  : {radial.shape}")
    print(f"window n      : {int(data['n'])}")
    print(f"stride        : {int(data['stride'])}")
    print(f"frequency bin : 0 .. {radial.shape[2] - 1}")
    print(f"window func   : {str(data['window'])}")
    print(f"dB            : {bool(data['db'])}")
    print(f"source image  : {str(data['source_image'])}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "画像を局所FFTし，radial spectrum volumeをNPZ保存し，"
            "任意frequency binを画像へ重畳表示する．"
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    # --------------------------------------------------------
    # compute
    # --------------------------------------------------------
    p_compute = subparsers.add_parser(
        "compute",
        help="画像全体を走査してradial FFT volumeを計算する"
    )

    p_compute.add_argument(
        "image",
        help="入力画像ファイル"
    )

    p_compute.add_argument(
        "-o",
        "--output",
        default="radial_fft_volume.npz",
        help="出力NPZファイル名"
    )

    p_compute.add_argument(
        "-n",
        type=int,
        default=64,
        help="FFT窓サイズ n x n"
    )

    p_compute.add_argument(
        "-s",
        "--stride",
        type=int,
        default=1,
        help="走査stride"
    )

    p_compute.add_argument(
        "-w",
        "--workers",
        type=int,
        default=0,
        help=(
            "worker数．0の場合は CPU数-1 を自動設定"
        )
    )

    p_compute.add_argument(
        "--chunksize",
        type=int,
        default=0,
        help="multiprocessingのchunksize．0なら自動設定"
    )

    p_compute.add_argument(
        "--window",
        choices=["none", "hann", "hamming", "blackman"],
        default="hann",
        help="2次元FFT前の窓関数"
    )

    p_compute.add_argument(
        "--db",
        action="store_true",
        help="radial powerをdBへ変換して保存する"
    )

    p_compute.add_argument(
        "--eps",
        type=float,
        default=1.0e-12,
        help="dB変換時のlog(0)回避値"
    )

    p_compute.add_argument(
        "--compressed",
        action="store_true",
        help="np.savez_compressedを使用する"
    )

    p_compute.set_defaults(func=compute_volume)

    # --------------------------------------------------------
    # view
    # --------------------------------------------------------
    p_view = subparsers.add_parser(
        "view",
        help="NPZ中の指定frequency binを原画像へ重畳表示する"
    )

    p_view.add_argument(
        "npz",
        help="computeで作成したNPZファイル"
    )

    p_view.add_argument(
        "-f",
        type=int,
        required=True,
        help="表示するradial frequency bin"
    )

    p_view.add_argument(
        "--cmap",
        default="turbo",
        help="Matplotlib colormap名"
    )

    p_view.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="重畳画像の透明度"
    )

    p_view.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="LUT下限値"
    )

    p_view.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="LUT上限値"
    )

    p_view.add_argument(
        "--pmin",
        type=float,
        default=1.0,
        help="vmin未指定時に使用するpercentile"
    )

    p_view.add_argument(
        "--pmax",
        type=float,
        default=99.0,
        help="vmax未指定時に使用するpercentile"
    )

    p_view.add_argument(
        "--interpolation",
        choices=[
            "nearest",
            "bilinear",
            "bicubic"
        ],
        default="nearest",
        help="スペクトルマップ表示時の補間"
    )

    p_view.add_argument(
        "--save",
        default=None,
        help="表示画像を保存するファイル名"
    )

    p_view.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="保存時のdpi"
    )

    p_view.add_argument(
        "--no-show",
        action="store_true",
        help="画面表示しない"
    )

    p_view.set_defaults(func=view_volume)

    # --------------------------------------------------------
    # info
    # --------------------------------------------------------
    p_info = subparsers.add_parser(
        "info",
        help="NPZの情報を表示する"
    )

    p_info.add_argument(
        "npz",
        help="NPZファイル"
    )

    p_info.set_defaults(func=info_volume)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
