"""Compare plain FFT, mean subtraction, and mean subtraction + 2D Hann.

Run: python compare_fft_preprocessing.py
Requires: python -m pip install numpy matplotlib opencv-python
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CONDITIONS = ("Plain FFT", "Mean-subtracted FFT",
              "Mean-subtracted + Hann-windowed FFT")


def read_gray(path):
    """Unicode-safe grayscale loading; preserve 8/16-bit intensity scale."""
    encoded = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH) if encoded.size else None
    if gray is None:
        raise ValueError("Cannot decode image")
    if min(gray.shape) < 3:
        raise ValueError("Image must be at least 3 x 3 pixels for a Hann window")
    if not np.isfinite(gray).all():
        raise ValueError("Image contains non-finite values")
    return gray


def compute_conditions(gray):
    image = gray.astype(np.float64)
    # 平均値除去: 一定の明るさに対応するDC成分を除く。
    # DC以外の周波数成分や、画像内の明暗境界そのものは除去しない。
    centered = image - image.mean()
    height, width = image.shape
    wy = np.hanning(height)
    wx = np.hanning(width)
    # Hann窓: 画像端を滑らかにゼロへ近づけ、端の不連続による漏れを弱める。
    # 端の特徴も弱まり、ピーク幅・振幅も変わる。窓の振幅補正はしない。
    window = np.outer(wy, wx)
    spectra, statistics = [], []
    for label, data in zip(CONDITIONS, (image, centered, centered * window)):
        transformed = np.fft.fftshift(np.fft.fft2(data))
        spectra.append(np.log1p(np.abs(transformed)))
        statistics.append({
            "condition": label,
            "processed_mean": float(data.mean()),
            "processed_std": float(data.std()),
            "dc_abs": float(abs(transformed[height // 2, width // 2])),
            "dc_abs_per_pixel": float(abs(transformed[height // 2, width // 2]) / data.size),
        })
    return spectra, statistics


def shared_limits(spectra, lower_percentile=1.0, upper_percentile=99.8):
    # 表示だけを調整する。3条件をまとめた分布から同じ上下限を決める。
    # FFTの値自体を変更したり、DCをマスクしたりはしない。
    pooled = np.concatenate([s.ravel() for s in spectra])
    low, high = np.percentile(pooled, [lower_percentile, upper_percentile])
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def save_comparison(gray, spectra, statistics, path, destination, limits, percentiles):
    height, width = gray.shape
    fig = plt.figure(figsize=(14, 13), layout="constrained")
    try:
        grid = fig.add_gridspec(3, 2, width_ratios=[1, 1.1])
        original_ax = fig.add_subplot(grid[:, 0])
        original_ax.imshow(gray, cmap="gray", vmin=float(gray.min()),
                           vmax=max(float(gray.max()), float(gray.min()) + 1))
        original_ax.set_title("Original grayscale image\nNo resize / no crop", fontsize=13)
        original_ax.set_xticks([])
        original_ax.set_yticks([])
        original_ax.set_xlabel(
            f"Input mean: {gray.mean():.4f} | std: {gray.std():.4f}\n"
            f"FFT display: log(1 + |F|), shared range {limits[0]:.3f} to {limits[1]:.3f}\n"
            f"Pooled percentiles: {percentiles[0]:g}% to {percentiles[1]:g}%\n"
            "Values outside display range are saturated; FFT values are unchanged.\n"
            "DC is at center. Hann amplitude is not compensated.",
            fontsize=10, labelpad=16)
        fft_axes = []
        for row, (spectrum, stats) in enumerate(zip(spectra, statistics)):
            ax = fig.add_subplot(grid[row, 1])
            plotted = ax.imshow(spectrum, cmap="gray", vmin=limits[0], vmax=limits[1],
                                interpolation="nearest")
            ax.set_title(f"{stats['condition']}\n"
                         f"|DC|={stats['dc_abs']:.4g} | mean={stats['processed_mean']:.4g} | "
                         f"std={stats['processed_std']:.4g}", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            fft_axes.append(ax)
        fig.colorbar(plotted, ax=fft_axes, label="log(1 + |F|) - shared scale",
                     shrink=.8, extend="both")
        fig.suptitle(f"{path.name} | {width} x {height} pixels | {gray.dtype}", fontsize=16)
        fig.savefig(destination, dpi=150)
    finally:
        plt.close(fig)


def run_batch(input_dir, output_dir, lower_percentile=1.0, upper_percentile=99.8):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if input_dir == output_dir:
        raise ValueError("Input and output folders must be different")
    # 非再帰: trimingサブフォルダは読み込まず、指定フォルダ直下の元画像だけを対象にする。
    paths = sorted((p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS),
                   key=lambda p: p.name)
    if not paths:
        raise ValueError(f"No supported images in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, successes, errors = [], [], []
    for index, path in enumerate(paths, start=1):
        try:
            gray = read_gray(path)
            spectra, stats = compute_conditions(gray)
            limits = shared_limits(spectra, lower_percentile, upper_percentile)
            # 拡張子も含め、同名のPNG/JPEGなどで出力が衝突しないようにする。
            destination = output_dir / f"{path.name}_fft_comparison.png"
            save_comparison(gray, spectra, stats, path, destination, limits,
                            (lower_percentile, upper_percentile))
            for item in stats:
                rows.append({"filename": path.name, "width": gray.shape[1], "height": gray.shape[0],
                             "input_dtype": str(gray.dtype), "input_mean": float(gray.mean()),
                             "input_std": float(gray.std()), **item,
                             "display_vmin": limits[0], "display_vmax": limits[1]})
            successes.append({"input": str(path), "output": str(destination)})
            print(f"[{index}/{len(paths)}] OK: {path.name} | "
                  + " / ".join(f"DC={s['dc_abs']:.4g}" for s in stats), flush=True)
        except Exception as exc:
            errors.append({"input": str(path), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{index}/{len(paths)}] ERROR: {path.name}: {exc}", flush=True)
    # 成功が0件でもCSVヘッダーとエラーレポートを保存する。
    fields = list(rows[0]) if rows else ["filename", "condition", "dc_abs"]
    with (output_dir / "statistics.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {"input_directory": str(input_dir), "recursive": False,
              "condition_order": CONDITIONS, "display_percentiles": [lower_percentile, upper_percentile],
              "display_scale": "Shared within each figure; may differ between source images",
              "success_count": len(successes), "error_count": len(errors),
              "successes": successes, "errors": errors}
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done: {len(successes)} succeeded, {len(errors)} failed -> {output_dir}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data20260903" / "data20260903")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "fft_preprocessing_comparison")
    parser.add_argument("--lower-percentile", type=float, default=1.0)
    parser.add_argument("--upper-percentile", type=float, default=99.8)
    args = parser.parse_args()
    if not 0 <= args.lower_percentile < args.upper_percentile <= 100:
        parser.error("Require 0 <= lower-percentile < upper-percentile <= 100")
    try:
        report = run_batch(args.input_dir, args.output_dir, args.lower_percentile, args.upper_percentile)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    # エラーがあっても全画像を処理してから終了コードで通知する。
    return 1 if report["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
