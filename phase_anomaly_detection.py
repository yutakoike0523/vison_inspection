"""Independent Fourier phase/amplitude experiment; no spatial grid-point detection.

Run ``python phase_anomaly_detection.py --help``. Coordinates are always (x, y).
Frequency peak options are signed FFT-bin offsets from the shifted DC location.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import scipy
from scipy.ndimage import distance_transform_edt, gaussian_filter, maximum_filter
import skimage
from skimage.restoration import unwrap_phase
from PIL import Image

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    min_frequency: float = 0.0025
    max_frequency: float = 0.08
    min_angle_deg: float = 35.0
    candidate_count: int = 30
    band_sigma: float = 2.0  # equivalent bins at min(height, width)
    phase_sigma: float = 100.0  # spatial pixels
    amplitude_sigma: float = 100.0
    min_window: float = 0.15
    min_amplitude_fraction: float = 0.15
    phase_weight: float = 0.5
    amplitude_weight: float = 0.5
    score_display_max: float = 6.0  # robust standard deviations, same for all overlays


def find_image(root: Path = ROOT) -> Path:
    """Prefer the same unannotated crop used by existing image15 experiments."""
    for name in ("image15_triming.png", "image15.png"):
        paths = sorted(p for p in root.rglob(name)
                       if "outputs" not in p.parts and "__MACOSX" not in p.parts)
        if paths:
            if len(paths) > 1:
                raise ValueError(f"Multiple {name} files; select --image explicitly: {paths}")
            return paths[0]
    raise FileNotFoundError("image15 image not found; use --image PATH")


def validate_config(c: Config):
    if not 0 < c.min_frequency < c.max_frequency < 0.5:
        raise ValueError("Require 0 < min-frequency < max-frequency < 0.5")
    if not 0 < c.min_angle_deg <= 90 or c.candidate_count < 2:
        raise ValueError("min-angle-deg must be in (0,90], candidate-count >= 2")
    if min(c.band_sigma, c.phase_sigma, c.amplitude_sigma, c.score_display_max) <= 0:
        raise ValueError("Bandwidth, smoothing sigmas and display maximum must be positive")
    if not 0 < c.min_window < 1 or not 0 <= c.min_amplitude_fraction < 1:
        raise ValueError("Invalid window/amplitude confidence threshold")
    if min(c.phase_weight, c.amplitude_weight) < 0 or c.phase_weight + c.amplitude_weight <= 0:
        raise ValueError("Score weights must be nonnegative with positive sum")
    if not all(np.isfinite(v) for v in asdict(c).values()):
        raise ValueError("All configuration values must be finite")


def frequency_grid(shape):
    h, w = shape
    return np.meshgrid(np.fft.fftshift(np.fft.fftfreq(w)),
                       np.fft.fftshift(np.fft.fftfreq(h)))


def peak_record(magnitude, dx, dy):
    h, w = magnitude.shape
    x, y = w // 2 + int(dx), h // 2 + int(dy)
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError(f"Peak {(dx, dy)} outside FFT")
    fx, fy = dx / w, dy / h
    if fx == 0 and fy == 0 or abs(fx) >= .5 or abs(fy) >= .5:
        raise ValueError("DC and Nyquist peaks cannot define an analytic sideband")
    sx, sy = (2 * (w // 2) - x) % w, (2 * (h // 2) - y) % h
    a, b = float(magnitude[y, x]), float(magnitude[sy, sx])
    return dict(x=x, y=y, dx=int(dx), dy=int(dy), fx=fx, fy=fy,
                radius=float(np.hypot(dx, dy)),
                frequency_radius=float(np.hypot(fx, fy)),
                period_px=float(1 / np.hypot(fx, fy)),
                angle_deg=float(np.degrees(np.arctan2(fy, fx)) % 180),
                magnitude=a, symmetric_x=sx, symmetric_y=sy,
                symmetric_magnitude=b, symmetry_ratio=min(a, b) / max(a, b, 1e-30))


def angle_distance(p, q):
    d = abs(p["angle_deg"] - q["angle_deg"]) % 180
    return min(d, 180 - d)


def detect_peaks(magnitude, c, manual=None):
    h, w = magnitude.shape
    fx, fy = frequency_grid(magnitude.shape)
    radius = np.hypot(fx, fy)
    half = (fy > 0) | ((fy == 0) & (fx > 0))
    band = (radius >= c.min_frequency) & (radius <= c.max_frequency)
    local = magnitude == maximum_filter(magnitude, size=5, mode="wrap")
    yy, xx = np.nonzero(local & half & band)
    order = np.argsort(magnitude[yy, xx])[::-1]
    candidates = []
    for i in order:
        p = peak_record(magnitude, xx[i] - w // 2, yy[i] - h // 2)
        if p["symmetry_ratio"] < .95:
            continue
        # Local contrast is diagnostic; conjugate symmetry alone is not proof of a grid.
        y, x = p["y"], p["x"]
        nearby = magnitude[max(0, y-5):y+6, max(0, x-5):x+6]
        p["local_prominence"] = p["magnitude"] / max(float(np.median(nearby)), 1e-12)
        candidates.append(p)
        if len(candidates) >= c.candidate_count:
            break
    if manual is not None:
        selected = [peak_record(magnitude, *p) for p in manual]
    else:
        pairs = [(p, q) for p, q in itertools.combinations(candidates, 2)
                 if angle_distance(p, q) >= c.min_angle_deg]
        if not pairs:
            raise ValueError("No two separated spectral peaks; adjust search range or --peak1/--peak2")
        # Joint selection prevents choosing two peaks on the same harmonic ray.
        selected = list(max(pairs, key=lambda pq: np.sqrt(pq[0]["magnitude"] * pq[1]["magnitude"])
                            * np.sin(np.deg2rad(angle_distance(*pq)))))
    if angle_distance(*selected) < c.min_angle_deg:
        raise ValueError("Selected peaks are too nearly parallel (including conjugate pairs)")
    for i, p in enumerate(selected, 1):
        p["direction"] = i
    return candidates, selected


def sideband_mask(shape, p, selected, sigma_bins):
    """Disjoint single-sideband Gaussian discs, tapered to zero before DC/other carriers."""
    fx, fy = frequency_grid(shape)
    center = np.array([p["fx"], p["fy"]])
    competitors = [-center]
    for q in selected:
        if q["direction"] != p["direction"]:
            competitors.extend([np.array([q["fx"], q["fy"]]), -np.array([q["fx"], q["fy"]])])
    separation = min(float(np.linalg.norm(center - v)) for v in competitors)
    sigma = sigma_bins / min(shape)
    cutoff = min(3 * sigma, .9 * np.linalg.norm(center), .48 * separation,
                 .49 - max(abs(center)))
    if cutoff <= 0:
        raise ValueError("Peak too close to Nyquist")
    d = np.hypot(fx - center[0], fy - center[1])
    taper = .5 * (1 + np.cos(np.pi * np.clip((d / cutoff - .8) / .2, 0, 1)))
    mask = np.exp(-.5 * (d / sigma) ** 2) * taper
    mask[d >= cutoff] = 0
    return mask, dict(sigma_cycles_per_pixel=float(sigma), cutoff_cycles_per_pixel=float(cutoff),
                      cutoff_equivalent_bins=float(cutoff * min(shape)),
                      spatial_gaussian_sigma_px=float(1 / (2 * np.pi * sigma)),
                      support_limited_by_separation=bool(cutoff < 3 * sigma - 1e-12))


def weighted_smooth(values, valid, sigma):
    weight = gaussian_filter(valid.astype(float), sigma, mode="reflect")
    numerator = gaussian_filter(np.where(valid, values, 0), sigma, mode="reflect")
    return np.divide(numerator, weight, out=np.zeros_like(numerator), where=weight > 1e-8)


def phase_background(phase, valid, sigma):
    """Remove affine carrier/ramp before smoothing to avoid artificial boundary residuals."""
    h, w = phase.shape
    yy, xx = np.mgrid[-1:1:complex(h), -1:1:complex(w)]
    # Subsample only the least-squares solve, never the image or the FFT.
    stride = max(1, int(np.sqrt(phase.size / 60000)))
    v = valid[::stride, ::stride]
    design = np.column_stack([np.ones(v.sum()), xx[::stride, ::stride][v], yy[::stride, ::stride][v]])
    coef, *_ = np.linalg.lstsq(design, phase[::stride, ::stride][v], rcond=None)
    plane = coef[0] + coef[1] * xx + coef[2] * yy
    return plane + weighted_smooth(phase - plane, valid, sigma)


def robust_normalize(score, valid):
    values = score[valid]
    median = float(np.median(values))
    mad = float(np.median(abs(values - median)))
    scale = max(1.4826 * mad, 1e-8)
    normalized = np.where(valid, np.maximum((score - median) / scale, 0), np.nan)
    return normalized, dict(median=median, mad=mad, scale=scale, scale_floor=1e-8)


def stats(values, valid=None):
    v = values[valid] if valid is not None else values[np.isfinite(values)]
    v = v[np.isfinite(v)]
    if not v.size:
        return dict(count=0, min=None, max=None, median=None, mean=None, p95=None)
    return dict(count=int(v.size), min=float(v.min()), max=float(v.max()),
                median=float(np.median(v)), mean=float(v.mean()), p95=float(np.percentile(v, 95)))


def boundary_diagnostics(result):
    diagnostics = {}
    for key in ("normalized_phase_score", "normalized_amplitude_score"):
        values = result[key]
        valid = np.isfinite(values)
        distance = distance_transform_edt(valid)
        y, x = np.unravel_index(np.nanargmax(values), values.shape)
        top = valid & (values >= np.nanpercentile(values, 99))
        diagnostics[key] = dict(max_xy=[int(x), int(y)], max=float(values[y, x]),
            max_distance_to_validity_edge_px=float(distance[y, x]),
            top_1pct_within_50px_of_validity_edge_fraction=float(np.mean(distance[top] < 50)),
            valid_area_within_50px_edge_fraction=float(np.mean(distance[valid] < 50)))
    return diagnostics


def analyze(gray, c, manual=None, roi_mask=None, *, full_area=False, use_hann=True):
    validate_config(c)
    gray = np.asarray(gray, dtype=float)
    if gray.ndim != 2 or min(gray.shape) < 32 or not np.isfinite(gray).all():
        raise ValueError("Expected a finite grayscale image at least 32 x 32")
    h, w = gray.shape
    roi = np.ones(gray.shape, bool) if roi_mask is None else np.asarray(roi_mask, bool)
    if roi.shape != gray.shape or roi.sum() < 1024:
        raise ValueError("ROI must match input dimensions and include >=1024 pixels")
    hann = np.outer(np.hanning(h), np.hanning(w)) if use_hann else np.ones(gray.shape)
    if roi_mask is not None and not full_area:
        # Feather a geometric ROI; do not threshold the intensity/grid pattern.
        padded = np.pad(roi, 1)
        distance = distance_transform_edt(padded)[1:-1, 1:-1]
        hann *= np.sin(.5 * np.pi * np.clip(distance / 32, 0, 1)) ** 2
    mean = float(gray.mean() if full_area else gray[roi].mean())
    preprocessed = (gray - mean) * hann
    spectrum = np.fft.fftshift(np.fft.fft2(preprocessed))
    magnitude = abs(spectrum)
    if magnitude.max() < 1e-8:
        raise ValueError("Image has no measurable contrast")
    candidates, peaks = detect_peaks(magnitude, c, manual)
    base_valid = np.ones(gray.shape, bool) if full_area else (hann >= c.min_window) & roi
    directions = []
    yy, xx = np.indices(gray.shape)
    for p in peaks:
        mask, band = sideband_mask(gray.shape, p, peaks, c.band_sigma)
        filtered = spectrum * mask
        complex_image = np.fft.ifft2(np.fft.ifftshift(filtered))
        amplitude = abs(complex_image)
        # Filter the window by the corresponding baseband kernel. This reduces
        # the Hann envelope without treating edge values as reliable measurements.
        lowpass = np.roll(mask, (-p["dy"], -p["dx"]), axis=(0, 1))
        envelope = (np.fft.ifft2(np.fft.fft2(hann) * np.fft.ifftshift(lowpass)).real
                    if use_hann else np.ones(gray.shape))
        amp_valid = (np.ones(gray.shape, bool) if full_area
                     else base_valid & (envelope >= c.min_window))
        corrected = amplitude / np.maximum(envelope, c.min_window)
        if amp_valid.sum() < 100:
            raise ValueError("Too little reliable window support")
        threshold = max(.2, float(np.median(corrected[amp_valid])) * c.min_amplitude_fraction)
        phase_valid = amp_valid if full_area else amp_valid & (corrected >= threshold)
        if phase_valid.sum() < 100:
            raise ValueError("Sideband amplitude too weak for phase recovery")
        carrier = 2 * np.pi * (p["fx"] * xx + p["fy"] * yy)
        wrapped = np.angle(complex_image)
        demodulated = np.angle(complex_image * np.exp(-1j * carrier))
        unwrapped = unwrap_phase(np.ma.array(demodulated, mask=~phase_valid), rng=0).filled(np.nan)
        background = phase_background(unwrapped, phase_valid, c.phase_sigma)
        residual = np.where(phase_valid, unwrapped - background, np.nan)
        amp_smooth = weighted_smooth(corrected, amp_valid, c.amplitude_sigma)
        amp_residual = np.where(amp_valid, abs(corrected - amp_smooth), np.nan)
        relative_amp_residual = amp_residual / np.maximum(amp_smooth, .2)
        # Adjacent recovered-phase jumps flag possible unwrap defects / weak carriers.
        jumps = []
        for axis in (0, 1):
            diff = np.diff(unwrapped, axis=axis)
            finite = np.isfinite(diff)
            jumps.append(float(np.mean(abs(diff[finite]) > np.pi)) if finite.any() else None)
        directions.append(dict(mask=mask, filtered=filtered, complex_image=complex_image,
                               amplitude=amplitude, amplitude_corrected=corrected,
                               envelope=envelope, wrapped=wrapped, demodulated=demodulated,
                               unwrapped=unwrapped + carrier, smooth=background + carrier,
                               residual=residual, amplitude_smooth=amp_smooth,
                               amplitude_residual=amp_residual, relative_amplitude_residual=relative_amp_residual,
                               phase_valid=phase_valid, amplitude_valid=amp_valid,
                               diagnostics=dict(**band, amplitude_confidence_threshold=threshold,
                                                phase_valid_fraction=float(phase_valid.mean()),
                                                adjacent_demodulated_jumps_over_pi_fraction_yx=jumps)))
    d1, d2 = directions
    phase_valid = d1["phase_valid"] & d2["phase_valid"]
    amplitude_valid = d1["amplitude_valid"] & d2["amplitude_valid"]
    if phase_valid.sum() < 100:
        raise ValueError("Insufficient shared support between phase directions")
    phase_score = np.where(phase_valid, np.hypot(d1["residual"], d2["residual"]), np.nan)
    amp_score = np.where(amplitude_valid, np.hypot(d1["relative_amplitude_residual"],
                                                d2["relative_amplitude_residual"]), np.nan)
    amp_absolute = np.where(amplitude_valid, np.hypot(d1["amplitude_residual"],
                                                   d2["amplitude_residual"]), np.nan)
    phase_z, phase_norm = robust_normalize(phase_score, phase_valid)
    amp_z, amp_norm = robust_normalize(amp_score, amplitude_valid)
    total = np.where(phase_valid, c.phase_weight * phase_z + c.amplitude_weight * amp_z, np.nan)
    return dict(gray=gray, mean=mean, hann=hann, preprocessed=preprocessed, spectrum=spectrum,
                full_area=bool(full_area), use_hann=bool(use_hann),
                candidates=candidates, peaks=peaks, directions=directions,
                phase_valid=phase_valid, amplitude_valid=amplitude_valid,
                phase_score=phase_score, amplitude_score=amp_score, amplitude_score_absolute=amp_absolute,
                normalized_phase_score=phase_z, normalized_amplitude_score=amp_z, total_score=total,
                normalization=dict(phase=phase_norm, amplitude=amp_norm))


class Writer:
    def __init__(self, output):
        self.output = output
        self.scales = {}

    def map(self, name, data, cmap="viridis", limits=None):
        p = self.output / name
        p.parent.mkdir(parents=True, exist_ok=True)
        finite = data[np.isfinite(data)]
        if limits is None:
            limits = np.percentile(finite, [1, 99]) if finite.size else (0, 1)
        lo, hi = map(float, limits)
        if hi <= lo:
            hi = lo + 1e-8
        color = plt.get_cmap(cmap).copy()
        color.set_bad("#707070")
        plt.imsave(p, data, cmap=color, vmin=lo, vmax=hi)
        self.scales[name] = dict(min=lo, max=hi, colormap=cmap, invalid_color="#707070")

    def overlay(self, name, gray, score, vmax):
        valid = np.isfinite(score)
        amount = np.clip(np.nan_to_num(score) / vmax, 0, 1)
        base = np.repeat((gray / 255)[..., None], 3, axis=2)
        color = plt.get_cmap("inferno")(amount)[..., :3]
        alpha = (.7 * amount * valid)[..., None]
        result = base * (1 - alpha) + color * alpha
        p = self.output / name
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.uint8(np.clip(result * 255, 0, 255))).save(p)
        self.scales[name] = dict(min=0, max=vmax, colormap="inferno", invalid="original grayscale")


def region_comparison(result, source_name, crop, source_shape):
    """Existing R171 and eight controls are fixed review ROIs, never used in analysis."""
    if source_name != "image15_triming.png":
        return []
    h, w = source_shape
    ox, oy, _, _ = crop
    rows = []
    for region_id in [171, 154, 155, 156, 170, 172, 186, 187, 188]:
        row, col = divmod(region_id - 1, 16)
        box = [col * w // 16, row * h // 16, (col+1) * w // 16, (row+1) * h // 16]
        x0, y0, x1, y1 = box
        ih, iw = result["gray"].shape
        rx0, ry0, rx1, ry1 = max(x0-ox, 0), max(y0-oy, 0), min(x1-ox, iw), min(y1-oy, ih)
        if rx1 <= rx0 or ry1 <= ry0:
            continue
        sl = np.s_[ry0:ry1, rx0:rx1]
        entry = dict(region_id=region_id, role="candidate" if region_id == 171 else "unverified_control",
                     source_box_xyxy=box, analysis_box_xyxy=[rx0, ry0, rx1, ry1],
                     phase_valid_fraction=float(result["phase_valid"][sl].mean()))
        for key in ("phase_score", "amplitude_score", "normalized_phase_score", "normalized_amplitude_score", "total_score"):
            entry[key] = stats(result[key][sl])
        rows.append(entry)
    return rows


def save_results(r, c, output, source, crop, source_shape, mask_path, manual):
    output.mkdir(parents=True, exist_ok=True)
    wr = Writer(output)
    wr.map("01_input/original_gray.png", r["gray"], "gray", (0, 255))
    wr.map("01_input/preprocessed.png", r["preprocessed"], "gray")
    wr.map("01_input/hann_window.png", r["hann"], "gray", (0, 1))
    magnitude = abs(r["spectrum"])
    wr.map("02_fft/fft_magnitude_spectrum.png", magnitude, "gray", (0, magnitude.max()))
    wr.map("02_fft/fft_log_spectrum.png", np.log1p(magnitude), "magma")
    h, w = r["gray"].shape
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    for panel, ax in enumerate(axes):
        ax.imshow(np.log1p(magnitude), cmap="magma")
        ax.plot(w//2, h//2, "w+", ms=10)
        ax.annotate("DC", (w//2, h//2), xytext=(5, -12), textcoords="offset points", color="white")
        for i, p in enumerate(r["candidates"], 1):
            ax.plot([p["x"], p["symmetric_x"]], [p["y"], p["symmetric_y"]], "co", ms=3, fillstyle="none")
            if panel == 1 and i <= 8:
                ax.text(p["x"]+1, p["y"], str(i), color="cyan", fontsize=7)
        for p, d, color in zip(r["peaks"], r["directions"], ["lime", "yellow"]):
            ax.plot(p["x"], p["y"], "+", color=color, ms=12)
            if panel == 1:
                offset = (-100, -30) if p["direction"] == 1 else (15, -30)
                ax.annotate(f"D{p['direction']} ({p['dx']},{p['dy']})", (p["x"], p["y"]),
                            xytext=offset, textcoords="offset points", color=color,
                            arrowprops=dict(arrowstyle="-", color=color))
            # Frequency discs appear elliptical on unequal FFT axes; show the actual support contour.
            ax.contour(d["mask"], levels=[.01], colors=[color], linewidths=.8)
    axes[0].set_title("FFT log magnitude / conjugate candidates")
    zoom = max(25, int(max(p["radius"] for p in r["peaks"]) * 2))
    axes[1].set(xlim=(w//2-zoom, w//2+zoom), ylim=(h//2+zoom, h//2-zoom), title="DC and selected single sidebands")
    for ax in axes:
        ax.set(xlabel="shifted FFT x [bin]", ylabel="shifted FFT y [bin]")
    (output / "03_peak_detection").mkdir(exist_ok=True)
    fig.savefig(output / "03_peak_detection/fft_detected_peaks.png", dpi=160)
    fig.savefig(output / "02_fft/fft_center_marked.png", dpi=160)
    plt.close(fig)
    with (output / "03_peak_detection/peak_candidates.csv").open("w", newline="", encoding="utf-8-sig") as f:
        if r["candidates"]:
            fields = sorted(set().union(*(p.keys() for p in r["candidates"])))
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(r["candidates"])
    arrays = {k: r[k] for k in ("gray", "hann", "preprocessed", "spectrum", "phase_valid", "amplitude_valid",
              "phase_score", "amplitude_score", "amplitude_score_absolute", "normalized_phase_score",
              "normalized_amplitude_score", "total_score")}
    for i, d in enumerate(r["directions"], 1):
        wr.map(f"04_bandpass/direction{i}_mask.png", d["mask"], "gray", (0, 1))
        wr.map(f"04_bandpass/direction{i}_filtered_fft.png", np.log1p(abs(d["filtered"])), "magma", (0, np.log1p(magnitude.max())))
        wr.map(f"04_bandpass/direction{i}_mask_zoom.png", d["mask"][h//2-zoom:h//2+zoom+1, w//2-zoom:w//2+zoom+1], "gray", (0, 1))
        wr.map(f"05_inverse_fft/direction{i}_reconstructed.png", 2*d["complex_image"].real, "gray")
        wr.map(f"06_phase/phase{i}_wrapped.png", d["wrapped"], "twilight", (-np.pi, np.pi))
        wr.map(f"06_phase/phase{i}_demodulated_wrapped.png", d["demodulated"], "twilight", (-np.pi, np.pi))
        wr.map(f"06_phase/phase{i}_unwrapped.png", d["unwrapped"])
        wr.map(f"06_phase/phase{i}_valid.png", d["phase_valid"].astype(float), "gray", (0, 1))
        wr.map(f"07_phase_residual/phase{i}_smooth.png", np.where(d["phase_valid"], d["smooth"], np.nan))
        limit = max(float(np.nanpercentile(abs(d["residual"]), 99)), 1e-6)
        wr.map(f"07_phase_residual/phase{i}_residual.png", d["residual"], "coolwarm", (-limit, limit))
        for key, filename in [("amplitude", f"amplitude{i}"), ("amplitude_corrected", f"amplitude{i}_window_corrected"),
                              ("amplitude_smooth", f"amplitude{i}_smooth"), ("amplitude_residual", f"amplitude{i}_residual")]:
            wr.map(f"08_amplitude/{filename}.png", np.where(d["amplitude_valid"], d[key], np.nan) if key != "amplitude" else d[key])
        for key, value in d.items():
            if isinstance(value, np.ndarray):
                arrays[f"direction{i}_{key}"] = value
    for name, folder in [("phase", "07_phase_residual"), ("amplitude", "08_amplitude")]:
        wr.map(f"{folder}/{name}_score.png", r[f"{name}_score"], "inferno", (0, float(np.nanpercentile(r[f"{name}_score"], 99))))
        for dest in [folder, "09_anomaly_score"]:
            wr.overlay(f"{dest}/{name}_score_overlay.png", r["gray"], r[f"normalized_{name}_score"], c.score_display_max)
    wr.overlay("09_anomaly_score/total_score_overlay.png", r["gray"], r["total_score"], c.score_display_max)
    wr.map("09_anomaly_score/phase_valid.png", r["phase_valid"].astype(float), "gray", (0, 1))
    wr.map("09_anomaly_score/amplitude_valid.png", r["amplitude_valid"].astype(float), "gray", (0, 1))
    wr.map("09_anomaly_score/total_score.png", r["total_score"], "inferno", (0, c.score_display_max))
    # Full numerical maps accompany display-clipped PNGs (invalid scores remain NaN).
    np.savez_compressed(output / "analysis_arrays.npz", **{k: v.astype(np.complex64 if np.iscomplexobj(v) else np.float32)
                                                        if v.dtype != bool else v for k, v in arrays.items()})
    regions = region_comparison(r, source.name, crop, source_shape)
    fig, axes = plt.subplots(2, 3, figsize=(14, 11), layout="constrained")
    for ax, title, values in zip(axes.ravel(), ["Input grayscale", "Phase validity (white = usable)", "Amplitude validity",
                                               "Phase anomaly (robust z)", "Amplitude anomaly (robust z)", "Combined anomaly"],
                                [r["gray"], r["phase_valid"], r["amplitude_valid"], r["normalized_phase_score"], r["normalized_amplitude_score"], r["total_score"]]):
        if "anomaly" in title:
            ax.imshow(r["gray"], cmap="gray", vmin=0, vmax=255)
            alpha = .7 * np.clip(np.nan_to_num(values) / c.score_display_max, 0, 1)
            im = ax.imshow(np.ma.masked_invalid(values), cmap="inferno", vmin=0, vmax=c.score_display_max, alpha=alpha)
            fig.colorbar(im, ax=ax, shrink=.65, label="positive robust z")
        else:
            ax.imshow(values, cmap="gray")
        for region in regions:
            if region["region_id"] == 171:
                x0, y0, x1, y1 = region["analysis_box_xyxy"]
                ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="cyan", linewidth=1))
                ax.text(x1+5, y0, "R171", color="cyan", fontsize=8)
        ax.set_title(title)
        ax.set_axis_off()
    fig.savefig(output / "09_anomaly_score/summary.png", dpi=150)
    plt.close(fig)
    if any(row["region_id"] == 171 for row in regions):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout="constrained")
        target = next(x for x in regions if x["region_id"] == 171)
        x0, y0, x1, y1 = target["analysis_box_xyxy"]
        for ax, key, title in zip(axes, ["gray", "normalized_phase_score", "normalized_amplitude_score"], ["R171 and fixed neighbor regions", "Phase robust z", "Amplitude robust z"]):
            ax.imshow(r["gray"], cmap="gray", vmin=0, vmax=255)
            if key != "gray":
                alpha = .7 * np.clip(np.nan_to_num(r[key]) / c.score_display_max, 0, 1)
                im = ax.imshow(np.ma.masked_invalid(r[key]), cmap="inferno", vmin=0, vmax=c.score_display_max, alpha=alpha)
                fig.colorbar(im, ax=ax, shrink=.75)
            for region in regions:
                a, b, e, f = region["analysis_box_xyxy"]
                color = "cyan" if region["region_id"] == 171 else "lime"
                ax.add_patch(Rectangle((a, b), e-a, f-b, fill=False, edgecolor=color, linewidth=1))
                ax.text(a+2, b+12, str(region["region_id"]), color=color, fontsize=8)
            ax.set(xlim=(max(0, x0-150), min(w, x1+150)), ylim=(min(h, y1+150), max(0, y0-150)), title=title)
        fig.savefig(output / "09_anomaly_score/candidate_comparison.png", dpi=160)
        plt.close(fig)
    edge_review = boundary_diagnostics(r)
    (output / "boundary_diagnostics.json").write_text(json.dumps(edge_review, indent=2), encoding="utf-8")
    report = dict(image=str(source.resolve()), image_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  source_shape_hw=list(source_shape), analysis_shape_hw=[h, w], crop_xyxy=crop,
                  coordinate_system="input image pixels; crop origin explicitly recorded; peak x/y are shifted FFT indices",
                  roi_mask=str(mask_path) if mask_path else None, mean_intensity_removed=r["mean"],
                  config=asdict(c), peak_selection="manual" if manual else "automatic",
                  fft_peaks=r["peaks"], peak_candidates=r["candidates"],
                  direction_separation_deg=angle_distance(*r["peaks"]),
                  bandpass_radius=[d["diagnostics"]["cutoff_equivalent_bins"] for d in r["directions"]],
                  bandpass_sigma_equivalent_bins=c.band_sigma, phase_smoothing_sigma=c.phase_sigma,
                  direction_diagnostics=[d["diagnostics"] for d in r["directions"]],
                  phase_valid_fraction=float(r["phase_valid"].mean()), amplitude_valid_fraction=float(r["amplitude_valid"].mean()),
                  phase_score=stats(r["phase_score"]), amplitude_score=stats(r["amplitude_score"]),
                  total_score=stats(r["total_score"]), normalization=r["normalization"], region_comparison=regions,
                  boundary_diagnostics=edge_review,
                  display_scales=wr.scales,
                  versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                skimage=skimage.__version__, matplotlib=matplotlib.__version__),
                  limitations=["Single image; R171 and controls are unverified visual candidates, not ground truth.",
                               "Conjugate symmetry is inherent to any real image, not evidence of a grid by itself.",
                               "Fixed sidebands can miss local carrier drift or mix harmonics; inspect FFT and reconstructions.",
                               "Finite bandwidth limits spatial resolution; small scratches are blurred.",
                               "Low-amplitude/edge phase is excluded, not declared normal; amplitude remains separately visible.",
                               "Unwrapping may fail near phase singularities or disconnected valid areas.",
                               "Reflection, dirt and illumination can produce high scores; no depth or defect classification."])
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--image", type=Path, help="Default: auto-discover image15 crop, then original")
    p.add_argument("--output", type=Path, default=ROOT / "output_phase_analysis")
    p.add_argument("--roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), help="Exclusive crop bounds in source pixels")
    p.add_argument("--roi-mask", type=Path, help="Optional geometric mask with same dimensions as source image")
    p.add_argument("--peak1", type=int, nargs=2, metavar=("DX", "DY"))
    p.add_argument("--peak2", type=int, nargs=2, metavar=("DX", "DY"))
    defaults = Config()
    for field, value in asdict(defaults).items():
        p.add_argument("--" + field.replace("_", "-"), type=type(value), default=value)
    return p.parse_args()


def main():
    args = parse_args()
    if (args.peak1 is None) != (args.peak2 is None):
        raise ValueError("Supply both --peak1 and --peak2, or neither")
    source = args.image or find_image()
    gray = np.asarray(Image.open(source).convert("L"), float)
    source_shape = gray.shape
    h, w = gray.shape
    crop = list(args.roi or (0, 0, w, h))
    x0, y0, x1, y1 = crop
    if not 0 <= x0 < x1 <= w or not 0 <= y0 < y1 <= h:
        raise ValueError("ROI lies outside input image")
    roi_mask = None
    if args.roi_mask:
        roi_mask = np.asarray(Image.open(args.roi_mask).convert("L")) > 0
        if roi_mask.shape != gray.shape:
            raise ValueError("ROI mask must have the source image dimensions")
        roi_mask = roi_mask[y0:y1, x0:x1]
    gray = gray[y0:y1, x0:x1]
    c = Config(**{key: getattr(args, key) for key in asdict(Config())})
    manual = [args.peak1, args.peak2] if args.peak1 is not None else None
    print(f"Input: {source}; crop={crop}; shape={gray.shape}", flush=True)
    r = analyze(gray, c, manual, roi_mask)
    print("Peaks: " + json.dumps(r["peaks"]), flush=True)
    report = save_results(r, c, args.output, source, crop, source_shape, args.roi_mask, manual)
    print(json.dumps({"output": str(args.output.resolve()), "phase_valid_fraction": report["phase_valid_fraction"],
                      "phase_score": report["phase_score"], "amplitude_score": report["amplitude_score"]}, indent=2))


if __name__ == "__main__":
    main()
