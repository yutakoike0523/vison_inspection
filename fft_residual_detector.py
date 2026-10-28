"""Detect excess low-frequency power outside a fixed normal-peak mask.

fit: learn a mask from normal images and a threshold from separate normal images.
predict: apply the saved model without relearning peaks from the inspected image.
All images must have the same dimensions, scale, orientation and exposure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from analyze_2d_fft import read_grayscale


def normalized_power(gray):
    """Power bins sum to window-weighted variance, in grayscale units squared."""
    data = np.asarray(gray, dtype=np.float64)
    if data.ndim != 2 or min(data.shape) < 16 or not np.isfinite(data).all():
        raise ValueError("有限値の2次元画像（縦横16画素以上）が必要です。")
    window = np.outer(np.hanning(data.shape[0]), np.hanning(data.shape[1]))
    # 窓を掛けた後のDCもゼロになるよう、窓の重み付き平均を引く。
    centered = data - np.sum(data * window) / window.sum()
    fft = np.fft.fftshift(np.fft.fft2(centered * window))
    return np.abs(fft) ** 2 / (data.size * np.sum(window ** 2))


def frequency_grid(shape):
    fy = np.fft.fftshift(np.fft.fftfreq(shape[0]))
    fx = np.fft.fftshift(np.fft.fftfreq(shape[1]))
    xx, yy = np.meshgrid(fx, fy)
    return xx, yy, np.hypot(xx, yy)


def build_mask(reference, *, low=0.08, dc=0.005, peak_radius=0.004,
               peak_relative=0.01, peak_prominence=4.0):
    """Find normal spectral peaks, then exclude circular neighborhoods.

    Coordinates and radii use cycles/pixel, not display pixels. A peak must
    exceed both a relative amplitude floor (in power) and its local average.
    """
    values = (low, dc, peak_radius, peak_relative, peak_prominence)
    if not all(np.isfinite(values)):
        raise ValueError("周波数・ピーク設定には有限値を指定してください。")
    if not 0 <= dc < low <= 0.5 or peak_radius <= 0:
        raise ValueError("0 <= dc < low <= 0.5、peak-radius > 0 が必要です。")
    if not 0 < peak_relative <= 1 or peak_prominence <= 1:
        raise ValueError("0 < peak-relative <= 1、peak-prominence > 1 が必要です。")
    xx, yy, radius = frequency_grid(reference.shape)
    # DC周辺のハン窓主ローブも除外する。
    effective_dc = max(dc, 2.0 / min(reference.shape))
    band = (radius > effective_dc) & (radius <= low)
    if not band.any():
        raise ValueError("低周波帯が空です。画像サイズまたは --low を大きくしてください。")
    # 低周波帯の境界に重なる輝点も候補に含める。
    radius_used = max(peak_radius, 2.0 / min(reference.shape))
    search = (radius > effective_dc) & (radius <= low + radius_used)
    local_max = cv2.dilate(reference, np.ones((5, 5), np.uint8))
    local_mean = cv2.blur(reference, (15, 15), borderType=cv2.BORDER_REFLECT)
    floor = max(float(reference[search].max()) * peak_relative, 1e-15)
    candidates = (search & (reference >= local_max) & (reference >= floor)
                  & (reference >= peak_prominence * np.maximum(local_mean, 1e-15)))
    ys, xs = np.nonzero(candidates)
    order = np.argsort(reference[ys, xs])[::-1]
    peak_mask = np.zeros(reference.shape, dtype=bool)
    peaks = []
    # 強いピークから採用し、同じピーク近傍の重複選択を抑制する。
    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])
        if peak_mask[y, x]:
            continue
        fx, fy = float(xx[y, x]), float(yy[y, x])
        peaks.append([fx, fy])
        # 実数画像の共役対称なピークを両側とも無視する。
        for sign in (-1, 1):
            peak_mask |= (xx - sign * fx) ** 2 + (yy - sign * fy) ** 2 <= radius_used ** 2
    keep = band & ~peak_mask
    coverage = float(np.sum(band & peak_mask) / band.sum())
    if not keep.any() or coverage > 0.8:
        raise ValueError("ピークマスクが低周波帯の80%以上を覆います。設定を見直してください。")
    return keep, peak_mask, {"low": low, "dc_requested": dc, "dc_effective": effective_dc,
                             "peak_radius_requested": peak_radius,
                             "peak_radius_effective": radius_used,
                             "peak_relative": peak_relative, "peak_prominence": peak_prominence,
                             "peak_centers_cycles_per_pixel": peaks,
                             "masked_band_fraction": coverage}


def residual_score(power, keep):
    if power.shape != keep.shape:
        raise ValueError(f"画像サイズがモデルと異なります: {power.shape} / {keep.shape}。同じサイズで切り出してください。")
    return float(power[keep].sum())


def manual_peak_mask(shape, peaks, *, radius_bins=2.0, dc_bins=2.0, low=0.025):
    """Fixed mask from user-specified fftshift array coordinates (x, y)."""
    if not all(np.isfinite([radius_bins, dc_bins, low])) or radius_bins <= 0 or dc_bins < 0 or not 0 < low <= 0.5:
        raise ValueError("マスク半径>0、DC半径>=0、0<low<=0.5の有限値を指定してください。")
    h, w = shape
    yy, xx = np.indices(shape)
    cx, cy = w // 2, h // 2
    _, _, frequency_radius = frequency_grid(shape)
    band = (frequency_radius <= low) & ((xx-cx)**2 + (yy-cy)**2 > dc_bins**2)
    peak_mask = np.zeros(shape, dtype=bool)
    centers = set()
    for x, y in peaks:
        if not 0 <= x < w or not 0 <= y < h:
            raise ValueError(f"FFT座標が画像外です: {(x, y)} / {shape}")
        centers.add((x, y))
        centers.add(((2*cx-x) % w, (2*cy-y) % h))
    for x, y in centers:
        # Circular distances also handle conjugate pairs at Nyquist edges.
        dx, dy = np.abs(xx-x), np.abs(yy-y)
        dx, dy = np.minimum(dx, w-dx), np.minimum(dy, h-dy)
        peak_mask |= dx**2 + dy**2 <= radius_bins**2
    keep = band & ~peak_mask
    if not keep.any() or np.sum(keep) < 0.2 * np.sum(band):
        raise ValueError("積分できる低周波領域が不足しています。半径・帯域を見直してください。")
    return keep, peak_mask, {"low": low, "mask_mode": "user_coordinates",
                             "shape": list(shape), "dc_radius_bins": dc_bins,
                             "peak_radius_bins": radius_bins,
                             "requested_peaks_xy": [list(p) for p in peaks],
                             "masked_centers_xy": [list(p) for p in sorted(centers)],
                             "masked_band_fraction": float(np.sum(band & peak_mask) / band.sum())}


def load_power_images(paths, shape=None):
    powers = []
    for path in paths:
        power = normalized_power(read_grayscale(path))
        if shape is None:
            shape = power.shape
        if power.shape != shape:
            raise ValueError(f"画像サイズを揃えてください: {path}: {power.shape} / {shape}")
        powers.append(power)
    return powers


def save_diagnostic(gray, power, reference, keep, peak_mask, metadata, output,
                    score=None, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xx, yy, _ = frequency_grid(power.shape)
    low = metadata["low"]
    zoom = (np.abs(xx) <= low * 1.2) & (np.abs(yy) <= low * 1.2)
    ys, xs = np.nonzero(zoom)
    cut = np.s_[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    dx, dy = 1 / power.shape[1], 1 / power.shape[0]
    extent = [xx[cut].min() - dx / 2, xx[cut].max() + dx / 2,
              yy[cut].max() + dy / 2, yy[cut].min() - dy / 2]
    log_power = np.log1p(power)
    log_reference = np.log1p(reference)
    vmax = max(float(log_power[cut].max()), float(log_reference[cut].max()), 1e-12)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    axes[0, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Input image")
    axes[0, 0].axis("off")
    ref_title = "Mask source: log(1 + power)" if metadata.get("mask_mode") == "user_coordinates" else "Normal reference: log(1 + power)"
    items = [(log_reference, ref_title, "gray", vmax),
             (log_power, "Input: log(1 + power)", "gray", vmax),
             (peak_mask, "Ignored normal peaks (white)", "gray", 1),
             (keep, "Integrated frequency bins (white)", "gray", 1),
             (np.where(keep, log_power, 0), "Residual: log(1 + power)", "magma", vmax)]
    for ax, (array, title, cmap, upper) in zip(axes.flat[1:], items):
        ax.imshow(array[cut], cmap=cmap, vmin=0, vmax=upper, extent=extent,
                  interpolation="nearest")
        ax.set(title=title, xlabel="fx [cycles/pixel]", ylabel="fy [cycles/pixel]")
    if metadata.get("threshold") is None:
        title = f"UNCALIBRATED | residual power = {score:.6g} | no defect verdict"
    elif score is None:
        title = f"Normal model | threshold = {metadata['threshold']:.6g} | inspect peak mask before use"
    else:
        verdict = "ANOMALY CANDIDATE" if score > metadata["threshold"] else "BELOW THRESHOLD"
        title = f"{verdict} | residual power = {score:.6g} | threshold = {metadata['threshold']:.6g}"
    fig.suptitle(title)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def fit(args):
    if Path(args.model).suffix.lower() != ".npz":
        raise ValueError("--model は拡張子 .npz で指定してください。")
    normal_paths = [Path(p).resolve() for p in args.normal]
    calibration_paths = [Path(p).resolve() for p in args.calibration]
    if set(normal_paths) & set(calibration_paths):
        raise ValueError("マスク作成用としきい値校正用には別の正常画像を指定してください。")
    if len(set(calibration_paths)) < 3:
        raise ValueError("しきい値校正用には異なる正常画像を3枚以上指定してください。")
    if not np.isfinite(args.margin) or args.margin <= 0:
        raise ValueError("--margin は0より大きい有限値にしてください。")
    normal_powers = load_power_images(normal_paths)
    reference = np.mean(normal_powers, axis=0)
    keep, peak_mask, metadata = build_mask(reference, low=args.low, dc=args.dc,
                                         peak_radius=args.peak_radius,
                                         peak_relative=args.peak_relative,
                                         peak_prominence=args.peak_prominence)
    calibration = load_power_images(calibration_paths, reference.shape)
    scores = [residual_score(p, keep) for p in calibration]
    threshold = max(max(scores) * (1 + args.margin), 1e-12)
    metadata.update({"version": 1, "shape": list(reference.shape),
                     "score_units": "8-bit grayscale squared",
                     "preprocessing": "Hann-weighted mean subtraction, Hann window, normalized power",
                     "threshold": threshold, "threshold_method": "max(calibration_scores) * (1 + margin)",
                     "margin": args.margin, "calibration_scores": scores,
                     "normal_paths": [str(p) for p in normal_paths],
                     "calibration_paths": [str(p) for p in calibration_paths]})
    model = Path(args.model)
    model.parent.mkdir(parents=True, exist_ok=True)
    with model.open("wb") as stream:
        np.savez_compressed(stream, reference=reference, keep=keep,
                            peak_mask=peak_mask, metadata=json.dumps(metadata))
    model.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    save_diagnostic(read_grayscale(normal_paths[0]), normal_powers[0], reference, keep,
                    peak_mask, metadata, model.with_suffix(".png"))
    print(f"Model: {model.resolve()}\nThreshold: {threshold:.6g}")
    print(f"Peak pairs: {len(metadata['peak_centers_cycles_per_pixel'])}, masked band: {metadata['masked_band_fraction']:.1%}")
    if not metadata["peak_centers_cycles_per_pixel"]:
        print("注意: 輝点を検出できませんでした。モデル図とピーク設定を確認してください。")


def selected_image(args):
    if args.image is not None:
        if args.input or args.image < 1:
            raise ValueError("--image は1以上で、画像パスと同時に指定しないでください。")
        data_dir = Path(__file__).resolve().parent / "data20260903" / "data20260903"
        path = (data_dir / "triming" / f"image{args.image}_triming.png" if args.triming
                else data_dir / f"image{args.image}.png")
    else:
        if not args.input or args.triming:
            raise ValueError("画像パス、または --image N [--triming] を指定してください。")
        path = Path(args.input)
    gray = read_grayscale(path)
    if args.roi:
        x, y, w, h = args.roi
        if min(x, y) < 0 or min(w, h) < 16 or x + w > gray.shape[1] or y + h > gray.shape[0]:
            raise ValueError("--roi は画像内の x y 幅 高さ（幅・高さ16以上）で指定してください。")
        gray = gray[y:y+h, x:x+w]
    return path, gray


def write_prediction(args, path, gray, reference, keep, peak_mask, metadata, model_path):
    power = normalized_power(gray)
    score = residual_score(power, keep)
    threshold = metadata.get("threshold")
    verdict = score > threshold if threshold is not None else None
    result = {"input": str(path.resolve()), "roi_xywh": args.roi,
              "model": str(Path(model_path).resolve()), "score": score,
              "threshold": threshold, "anomaly_candidate": verdict,
              "score_over_threshold": score / threshold if threshold is not None else None,
              "score_units": metadata["score_units"]}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    save_diagnostic(gray, power, reference, keep, peak_mask, metadata,
                    out / "diagnostic.png", score=score, show=args.show)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("判定: 未校正（スコアのみ）" if verdict is None else
          ("判定: 異常候補（しきい値超過）" if verdict else "判定: しきい値以下"))


def predict(args):
    path, gray = selected_image(args)
    with np.load(args.model, allow_pickle=False) as model:
        reference, keep, peak_mask = model["reference"], model["keep"], model["peak_mask"]
        metadata = json.loads(str(model["metadata"].item()))
    write_prediction(args, path, gray, reference, keep, peak_mask, metadata, args.model)


def measure(args):
    path, gray = selected_image(args)
    try:
        peaks = [tuple(map(int, p.split(","))) for p in args.peaks]
        if any(len(p) != 2 for p in peaks):
            raise ValueError()
    except ValueError:
        raise ValueError("--peaks は x,y の形式で指定してください（例: 560,709）。") from None
    keep, peak_mask, metadata = manual_peak_mask(gray.shape, peaks,
                                                radius_bins=args.radius_bins,
                                                dc_bins=args.dc_bins, low=args.low)
    power = normalized_power(gray)
    threshold = args.threshold
    if threshold is not None and (not np.isfinite(threshold) or threshold <= 0):
        raise ValueError("--threshold は0より大きい有限値にしてください。")
    if args.calibration and threshold is not None:
        raise ValueError("--calibration と --threshold は同時に指定できません。")
    if not np.isfinite(args.margin) or args.margin <= 0:
        raise ValueError("--margin は0より大きい有限値にしてください。")
    if args.calibration:
        paths = [Path(p).resolve() for p in args.calibration]
        if len(set(paths)) < 3 or path.resolve() in paths:
            raise ValueError("検査画像とは別の正常校正画像を3枚以上指定してください。")
        scores = [residual_score(p, keep) for p in load_power_images(paths, gray.shape)]
        threshold = max(max(scores) * (1 + args.margin), 1e-12)
        metadata.update({"calibration_paths": [str(p) for p in paths],
                         "calibration_scores": scores, "margin": args.margin})
    metadata.update({"version": 1, "threshold": threshold,
                     "threshold_method": "normal_calibration_max_with_margin" if args.calibration
                     else ("user_supplied" if threshold is not None else "uncalibrated"),
                     "score_units": "8-bit grayscale squared",
                     "mask_source": str(path.resolve()), "roi_xywh": args.roi})
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "model.npz"
    with model_path.open("wb") as stream:
        np.savez_compressed(stream, reference=power, keep=keep, peak_mask=peak_mask,
                            metadata=json.dumps(metadata))
    (out / "model.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_prediction(args, path, gray, power, keep, peak_mask, metadata, model_path)


def main():
    parser = argparse.ArgumentParser(description="正常輝点を除外した低周波パワーによる異常候補検出")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("fit", help="別々の正常画像群でマスク作成・しきい値校正")
    train.add_argument("--normal", nargs="+", required=True, help="マスク用正常画像")
    train.add_argument("--calibration", nargs="+", required=True, help="校正用正常画像（3枚以上）")
    train.add_argument("--model", default="outputs/fft_detector/model.npz")
    train.add_argument("--low", type=float, default=0.08, help="低周波帯上限 [cycles/pixel]")
    train.add_argument("--dc", type=float, default=0.005, help="中心除外半径 [cycles/pixel]")
    train.add_argument("--peak-radius", type=float, default=0.004, help="輝点マスク半径 [cycles/pixel]")
    train.add_argument("--peak-relative", type=float, default=0.01, help="最大正常ピークに対する下限比率（パワー）")
    train.add_argument("--peak-prominence", type=float, default=4.0, help="局所平均パワーに対するピーク比率")
    train.add_argument("--margin", type=float, default=0.1, help="校正最大値に加える比率（0.1=10%%）")
    train.set_defaults(run=fit)
    test = sub.add_parser("predict", help="保存モデルで1画像・1領域を判定")
    test.add_argument("input", nargs="?", help="検査画像パス")
    test.add_argument("--image", type=int, help="画像番号")
    test.add_argument("--triming", "--trimmed", action="store_true")
    test.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    test.add_argument("--model", default="outputs/fft_detector/model.npz")
    test.add_argument("--output", default="outputs/fft_detector/prediction")
    test.add_argument("--show", action="store_true", help="診断図を画面にも表示")
    test.set_defaults(run=predict)
    manual = sub.add_parser("measure", help="指定したFFT輝点を除外して積分。しきい値未指定ならスコアのみ")
    manual.add_argument("input", nargs="?")
    manual.add_argument("--image", type=int)
    manual.add_argument("--triming", "--trimmed", action="store_true")
    manual.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    manual.add_argument("--peaks", nargs="+", required=True, help="fftshift後の配列座標 x,y を複数指定")
    manual.add_argument("--radius-bins", type=float, default=2, help="輝点除外の半径（FFT配列の画素数）")
    manual.add_argument("--dc-bins", type=float, default=2, help="DC除外の半径（FFT配列の画素数）")
    manual.add_argument("--low", type=float, default=0.025, help="積分帯上限 [cycles/pixel]")
    manual.add_argument("--threshold", type=float, help="同一条件で決めた判定しきい値")
    manual.add_argument("--calibration", nargs="+", help="しきい値を校正する別の正常画像3枚以上")
    manual.add_argument("--margin", type=float, default=0.1)
    manual.add_argument("--output", default="outputs/fft_detector/manual")
    manual.add_argument("--show", action="store_true")
    manual.set_defaults(run=measure)
    args = parser.parse_args()
    try:
        args.run(args)
    except (ValueError, OSError, KeyError, cv2.error) as exc:
        parser.exit(1, f"エラー: {exc}\n")


if __name__ == "__main__":
    main()
