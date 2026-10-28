"""Full-image Hann FFT, center zoom, DC display mask, and simple paired peaks.

python compare_fft_center_peaks.py
Requires: numpy, matplotlib, opencv-python. No other local modules required.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parent
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ZOOM_RADIUS = 150
DC_MASK_RADIUS = 10
MAX_PEAKS = 20  # Individual points, i.e. at most ten conjugate pairs.
FOCUS_IMAGES = {"image7", "image8", "image11", "image12", "image18"}


def read_gray(path):
    encoded = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH) if encoded.size else None
    if gray is None:
        raise ValueError("Cannot decode image")
    if min(gray.shape) < 3 or not np.isfinite(gray).all():
        raise ValueError("Require a finite grayscale image at least 3 x 3 pixels")
    return gray


def hann_fft(gray):
    data = gray.astype(np.float64)
    # 一定の平均輝度（DC）を除く。明暗境界自体を除去する処理ではない。
    data -= data.mean()
    height, width = data.shape
    wy = np.hanning(height)
    wx = np.hanning(width)
    # 画像端で滑らかにゼロにし、端の不連続によるスペクトル漏れを弱める。
    window = np.outer(wy, wx)
    shifted = np.fft.fftshift(np.fft.fft2(data * window))
    return shifted, np.log1p(np.abs(shifted))


def center_zoom(spectrum, radius, dc_radius):
    height, width = spectrum.shape
    cy, cx = height // 2, width // 2
    x0, x1 = max(0, cx - radius), min(width, cx + radius)
    y0, y1 = max(0, cy - radius), min(height, cy + radius)
    # 切り出すのはFFTの表示領域だけ。入力画像は全画素をFFTに使用する。
    zoom = spectrum[y0:y1, x0:x1].copy()
    yy, xx = np.mgrid[y0:y1, x0:x1]
    outside_dc = (xx - cx) ** 2 + (yy - cy) ** 2 > dc_radius ** 2
    bounds = (x0, x1, y0, y1)
    extent = (x0 - cx - .5, x1 - cx - .5, y1 - cy - .5, y0 - cy - .5)
    return zoom, outside_dc, bounds, extent


def display_limits(values, lower=1., upper=99.8):
    if values.size == 0:
        return 0., 1.
    low, high = np.percentile(values, [lower, upper])
    return float(low), float(high if high > low else low + 1.)


def detect_peaks(zoom, outside_dc, bounds, image_shape, max_peaks=MAX_PEAKS,
                 peak_percentile=99., min_distance=7, local_max_size=7):
    """Local maxima above a percentile threshold, selected in conjugate pairs.

    Detection uses un-clipped log amplitudes; DC is excluded by a boolean mask.
    Both points must be inside the zoom and pass the same local maximum test.
    This is a visual aid, not a lattice classifier or a significance test.
    """
    values = zoom[outside_dc]
    if not values.size or float(values.max()) <= 1e-10:
        return [], None
    threshold = float(np.percentile(values, peak_percentile))
    # Exclude DC for candidate eligibility, not by replacing it before dilation:
    # this avoids manufacturing a local maximum on the edge of the DC mask.
    local_max = cv2.dilate(zoom, np.ones((local_max_size, local_max_size), dtype=np.uint8))
    eligible = outside_dc & (zoom >= threshold) & (zoom >= local_max - 1e-10) & (zoom > 1e-10)
    height, width = image_shape
    cy, cx = height // 2, width // 2
    x0, x1, y0, y1 = bounds
    candidates = {(int(y + y0), int(x + x0)) for y, x in np.argwhere(eligible)}
    ordered = sorted(candidates, key=lambda p: (-float(zoom[p[0] - y0, p[1] - x0]), p))
    peaks, selected, visited = [], [], set()
    for y, x in ordered:
        if len(peaks) + 2 > max_peaks:
            break
        partner = ((2 * cy - y) % height, (2 * cx - x) % width)
        key = tuple(sorted(((y, x), partner)))
        if key in visited:
            continue
        visited.add(key)
        if partner == (y, x) or partner not in candidates:
            continue
        # Do not count multiple nearby maxima as separate peaks of one lobe.
        if any(np.hypot(py - sy, px - sx) < min_distance
               for py, px in key for sy, sx in selected):
            continue
        pair_id = len(peaks) // 2 + 1
        for side, (py, px) in zip(("A", "B"), key):
            dx, dy = px - cx, py - cy
            fx, fy = dx / width, dy / height
            frequency = float(np.hypot(fx, fy))
            peaks.append({
                "peak_id": len(peaks) + 1, "pair_id": pair_id, "label": f"{pair_id}{side}",
                "x_offset": dx, "y_offset": dy,
                "distance_from_center": float(np.hypot(dx, dy)),
                "angle_deg": float(np.degrees(np.arctan2(dy, dx))),
                "log_amplitude": float(zoom[py - y0, px - x0]),
                "fx_cycles_per_pixel": fx, "fy_cycles_per_pixel": fy,
                "frequency_angle_deg": float(np.degrees(np.arctan2(fy, fx))),
                "period_pixels": 1. / frequency if frequency else None,
            })
            selected.append((py, px))
    return peaks, threshold


def annotate_peaks(ax, peaks):
    # Hollow circles and offset numbers leave peak centers visible.
    ax.scatter([p["x_offset"] for p in peaks], [p["y_offset"] for p in peaks],
               s=90, facecolors="none", edgecolors="#ff4040", linewidths=1)
    for peak in peaks:
        ax.annotate(peak["label"], (peak["x_offset"], peak["y_offset"]),
                    xytext=(5, 5), textcoords="offset points", color="#ff4040", fontsize=8)


def save_comparison(gray, spectrum, zoom, outside_dc, bounds, extent, record, destination):
    full_limits = record["full_display_limits"]
    zoom_limits = record["zoom_display_limits"]
    # 可視化用コピーのみをマスク。複素FFT・log振幅・ピーク計算値は変更しない。
    masked = zoom.copy()
    masked[~outside_dc] = zoom_limits[0]
    fig, axes = plt.subplots(2, 3, figsize=(17, 11), layout="constrained")
    try:
        axes[0, 0].imshow(gray, cmap="gray")
        axes[0, 0].set_title("Original grayscale image\nNo resize / no crop")
        axes[0, 0].axis("off")
        height, width = gray.shape
        cy, cx = height // 2, width // 2
        full_extent = (-cx - .5, width - cx - .5, height - cy - .5, -cy - .5)
        full = axes[0, 1].imshow(spectrum, cmap="gray", extent=full_extent,
                                  vmin=full_limits[0], vmax=full_limits[1], interpolation="nearest")
        axes[0, 1].set_title("Mean-subtracted + Hann-windowed FFT\nFull spectrum (yellow = zoom area)")
        x0, x1, y0, y1 = bounds
        axes[0, 1].add_patch(Rectangle((x0 - cx - .5, y0 - cy - .5), x1 - x0, y1 - y0,
                                     fill=False, edgecolor="#e6ba33", linewidth=1))
        fig.colorbar(full, ax=axes[0, 1], shrink=.7, label="log(1 + |F|), full", extend="both")
        zoom_axes = [axes[0, 2], axes[1, 0], axes[1, 1]]
        titles = [f"FFT center zoom (requested +/-{record['zoom_radius']} bins)",
                  f"FFT center zoom + DC mask (r={record['dc_mask_radius']:g} bins)",
                  f"Peak candidates: {len(record['peaks'])} points / {len(record['peaks']) // 2} pairs"]
        for ax, data, title in zip(zoom_axes, (zoom, masked, masked), titles):
            plotted = ax.imshow(data, cmap="gray", extent=extent, interpolation="nearest",
                                vmin=zoom_limits[0], vmax=zoom_limits[1])
            ax.set_title(title, fontsize=11)
        for ax in zoom_axes[1:]:
            ax.add_patch(Circle((0, 0), record["dc_mask_radius"], fill=False,
                                edgecolor="#35afcc", linewidth=.7, linestyle="--"))
        annotate_peaks(axes[1, 1], record["peaks"])
        fig.colorbar(plotted, ax=zoom_axes, shrink=.7, label="log(1 + |F|), shared zoom scale", extend="both")
        for ax in [axes[0, 1], *zoom_axes]:
            ax.set(xlabel="x offset from DC [FFT bins]", ylabel="y offset from DC [FFT bins]")
        axes[1, 2].axis("off")
        lines = ["Selected conjugate pairs", "Pair   A offset (x, y)   Period [px]  log(1+|F|)",
                 "---------------------------------------------"]
        for p in record["peaks"][::2]:
            lines.append(f"{p['pair_id']:>3}   ({p['x_offset']:>4},{p['y_offset']:>4})"
                         f"       {p['period_pixels']:>7.2f}      {p['log_amplitude']:.2f}")
        if not record["peaks"]:
            lines.append("No pairs passed these settings.")
        lines += ["", "Period = 1 / hypot(dx / width, dy / height)",
                  f"Peak threshold: percentile {record['peak_percentile']:g}",
                  f"Min. peak spacing: {record['min_peak_distance']} FFT bins", "",
                  "Full and zoom views use separate display ranges.",
                  "All three zoom views share the SAME range.",
                  "DC disk excluded from zoom scale and detection.",
                  "Only display values are clipped / masked.", "",
                  "Pairs are candidates, not confirmed lattice peaks.",
                  "Real images have conjugate symmetry even for",
                  "noise, tape boundaries and other structures."]
        axes[1, 2].text(0, .98, "\n".join(lines), va="top", family="monospace", fontsize=9)
        fig.suptitle(f"{record['filename']} | {width} x {height} pixels | mean subtraction + 2D Hann",
                     fontsize=16)
        fig.savefig(destination, dpi=150)
    finally:
        plt.close(fig)
    return masked


def save_focus_overview(focus, destination):
    if not focus:
        return
    by_stem = {Path(item[0]["filename"]).stem: item for item in focus}
    focus = [by_stem[name] for name in ("image7", "image8", "image11", "image12", "image18")
             if name in by_stem]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout="constrained")
    try:
        for ax in axes.flat:
            ax.axis("off")
        for ax, (record, masked, extent) in zip(axes.flat, focus):
            low, high = record["zoom_display_limits"]
            ax.imshow(masked, cmap="gray", vmin=low, vmax=high, extent=extent, interpolation="nearest")
            annotate_peaks(ax, record["peaks"])
            ax.axis("on")
            ax.set_title(f"{record['filename']} | {record['width']} x {record['height']}\n"
                         f"{record['detected_peak_count']} points / {record['detected_peak_count'] // 2} pairs")
            ax.set(xlabel="x offset [FFT bins]", ylabel="y offset [FFT bins]")
        settings = focus[0][0]
        axes.flat[-1].text(.05, .8, "Full original images; no crop / resize\n"
                           "Mean subtraction + 2D Hann + FFT\n\n"
                           f"Zoom: +/-{settings['zoom_radius']} bins; "
                           f"DC mask: r={settings['dc_mask_radius']:g} bins\n\n"
                           "Each image has its own display scale.\n"
                           "A/B labels identify conjugate pairs.\n"
                           "Symmetry alone does not identify a lattice.", fontsize=12, va="top")
        fig.suptitle("FFT center peaks: selected five images", fontsize=16)
        fig.savefig(destination, dpi=150)
    finally:
        plt.close(fig)


def run_batch(input_dir, output_dir, zoom_radius=ZOOM_RADIUS, dc_mask_radius=DC_MASK_RADIUS,
              max_peaks=MAX_PEAKS, peak_percentile=99., min_peak_distance=7,
              lower_percentile=1., upper_percentile=99.8):
    source, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if source == output:
        raise ValueError("Input and output folders must be different")
    paths = sorted((p for p in source.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS),
                   key=lambda p: p.name)
    if not paths:
        raise ValueError(f"No supported images in {source}")
    output.mkdir(parents=True, exist_ok=True)
    records, errors, focus = [], [], []
    for index, path in enumerate(paths, start=1):
        try:
            gray = read_gray(path)
            transformed, spectrum = hann_fft(gray)
            zoom, outside_dc, bounds, extent = center_zoom(spectrum, zoom_radius, dc_mask_radius)
            peaks, threshold = detect_peaks(zoom, outside_dc, bounds, gray.shape, max_peaks,
                                            peak_percentile, min_peak_distance)
            record = {"filename": path.name, "width": gray.shape[1], "height": gray.shape[0],
                      "zoom_radius": zoom_radius, "dc_mask_radius": dc_mask_radius,
                      "zoom_bounds_xyxy_exclusive": [bounds[0], bounds[2], bounds[1], bounds[3]],
                      "zoom_width": zoom.shape[1], "zoom_height": zoom.shape[0],
                      "detected_peak_count": len(peaks), "detected_pair_count": len(peaks) // 2,
                      "max_peaks": max_peaks, "peak_percentile": peak_percentile,
                      "min_peak_distance": min_peak_distance, "peak_threshold_log_amplitude": threshold,
                      "dc_abs": float(abs(transformed[gray.shape[0] // 2, gray.shape[1] // 2])),
                      "full_display_limits": display_limits(spectrum, lower_percentile, upper_percentile),
                      "zoom_display_limits": display_limits(zoom[outside_dc], lower_percentile, upper_percentile),
                      "display_percentiles": [lower_percentile, upper_percentile],
                      "angle_convention": "degrees, atan2(y_offset,x_offset); x right, y down",
                      "peaks": peaks}
            destination = output / f"{path.name}_fft_center_comparison.png"
            masked = save_comparison(gray, spectrum, zoom, outside_dc, bounds, extent, record, destination)
            (output / f"{path.name}_peaks.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            records.append(record)
            if path.stem in FOCUS_IMAGES:
                focus.append((record, masked, extent))
            print(f"[{index}/{len(paths)}] {path.name}: {len(peaks)} peaks / {len(peaks) // 2} pairs", flush=True)
        except Exception as exc:
            errors.append({"filename": path.name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{index}/{len(paths)}] ERROR: {path.name}: {exc}", flush=True)
    summary_fields = ["filename", "width", "height", "zoom_radius", "dc_mask_radius", "zoom_width",
                      "zoom_height", "detected_peak_count", "detected_pair_count", "peak_threshold_log_amplitude"]
    with (output / "statistics.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    peak_fields = ["filename", "peak_id", "pair_id", "label", "x_offset", "y_offset",
                   "distance_from_center", "angle_deg", "log_amplitude", "fx_cycles_per_pixel",
                   "fy_cycles_per_pixel", "frequency_angle_deg", "period_pixels"]
    with (output / "peaks.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=peak_fields)
        writer.writeheader()
        for record in records:
            writer.writerows({"filename": record["filename"], **peak} for peak in record["peaks"])
    report = {"input_directory": str(source), "recursive": False, "success_count": len(records),
              "error_count": len(errors), "errors": errors, "images": records,
              "note": "Peaks are visual candidates; conjugate symmetry is not evidence of lattice origin alone."}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_focus_overview(focus, output / "focus_images_peaks.png")
    print(f"Done: {len(records)} succeeded, {len(errors)} failed -> {output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data20260903" / "data20260903")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "fft_center_peaks")
    parser.add_argument("--zoom-radius", type=int, default=ZOOM_RADIUS)
    parser.add_argument("--dc-mask-radius", type=float, default=DC_MASK_RADIUS)
    parser.add_argument("--max-peaks", type=int, default=MAX_PEAKS)
    parser.add_argument("--peak-percentile", type=float, default=99.)
    parser.add_argument("--min-peak-distance", type=int, default=7)
    parser.add_argument("--lower-percentile", type=float, default=1.)
    parser.add_argument("--upper-percentile", type=float, default=99.8)
    args = parser.parse_args()
    if args.zoom_radius < 1 or not 0 <= args.dc_mask_radius < args.zoom_radius:
        parser.error("Require zoom-radius >= 1 and 0 <= dc-mask-radius < zoom-radius")
    if args.max_peaks < 2 or args.min_peak_distance < 1 or not 0 <= args.peak_percentile <= 100:
        parser.error("Require max-peaks >= 2, min-peak-distance >= 1, and peak-percentile in [0,100]")
    if not 0 <= args.lower_percentile < args.upper_percentile <= 100:
        parser.error("Require 0 <= lower-percentile < upper-percentile <= 100")
    try:
        report = run_batch(**vars(args))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return int(report["error_count"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
