#!/usr/bin/env python3
"""Compare image15 local radial-FFT maps before and after local grid removal.

The two columns differ only by in-band total-power normalisation.  Display
limits are derived from auxiliary normal regions (p05--p99) and are shared by
all three processing conditions within each column.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft

import fft_grid_notch_removal_analysis as grid
import local_multiscale_grid_harmonic_analysis as local
import radial_fft_experiment as radial
import window_size_comparison_unnormalized as labels


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_image15_grid_removal_fft_comparison"
IMAGE_ID = 15
WINDOW = 128
STRIDE = 16
SPACING = 48
FREQ = (0.01, 0.50)
BIN_WIDTH = 0.01
EPS = np.float32(1e-12)
OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1)
                for dx in (-1, 0, 1) if dy or dx)


def radial_setup() -> dict:
    """Use the repository's conjugate-weighted radial-bin definition."""
    _, _, _, specs = radial.grids(WINDOW, BIN_WIDTH, [FREQ])
    return specs[FREQ]


def spectra(image: np.ndarray, workers: int = 4, batch: int = 64) -> dict:
    """Calculate raw and total-power-normalised radial log-power spectra."""
    h, w = image.shape
    half = WINDOW // 2
    xs = np.arange(half, w - half + 1, STRIDE, dtype=np.int32)
    ys = np.arange(half, h - half + 1, STRIDE, dtype=np.int32)
    setup = radial_setup()
    valid = setup["valid"]
    raw = np.empty((len(ys), len(xs), int(valid.sum())), np.float32)
    norm = np.empty_like(raw)
    hann1 = np.hanning(WINDOW).astype(np.float32)
    hann = np.outer(hann1, hann1).astype(np.float32)

    for iy, y in enumerate(ys):
        row = np.lib.stride_tricks.sliding_window_view(
            image[y-half:y+half], WINDOW, axis=1)
        row = np.moveaxis(row, 1, 0)[::STRIDE]
        for start in range(0, len(xs), batch):
            values = np.asarray(row[start:start+batch], np.float32).copy()
            values -= values.mean(axis=(-2, -1), keepdims=True)
            values *= hann
            transformed = fft.rfft2(values, axes=(-2, -1), workers=workers)
            power = np.abs(fft.fftshift(transformed, axes=(-2,)))**2
            power = power.astype(np.float32)
            sums = power.reshape(len(power), -1) @ setup["matrix"]
            mean_power = sums[:, valid] / setup["counts"][valid][None, :]
            total = np.maximum(power[:, setup["mask"]].sum(axis=1), EPS)
            raw[iy, start:start+len(values)] = 10*np.log10(mean_power + EPS)
            norm[iy, start:start+len(values)] = 10*np.log10(
                mean_power / total[:, None] + EPS)
    return {"xs": xs, "ys": ys, "raw": raw, "norm": norm}


def strict_mean8_score(spec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RMS from the mean spectrum of exactly eight references at 48 px."""
    step = SPACING // STRIDE
    target = spec[step:-step, step:-step]
    refs = np.stack([
        spec[step + dy*step:len(spec)-step + dy*step,
             step + dx*step:spec.shape[1]-step + dx*step]
        for dy, dx in OFFSETS
    ])
    reference = refs.mean(axis=0)
    score = np.sqrt(np.mean((target-reference)**2, axis=-1)).astype(np.float32)
    return score, refs, reference


def extent(xs: np.ndarray, ys: np.ndarray) -> list[float]:
    return [xs[0]-STRIDE/2, xs[-1]+STRIDE/2,
            ys[-1]+STRIDE/2, ys[0]-STRIDE/2]


def normal_mask(xs: np.ndarray, ys: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, dict]:
    """Match the existing auxiliary-scale definition (16 px evaluation area)."""
    masks = labels.masks_for(IMAGE_ID, shape, xs, ys, grid.DATA)
    return masks["normal"], masks


def limits(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    lo, hi = np.percentile(values, [5, 99])
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # Reproduce the same local notch conditions used by the earlier comparison.
    top = grid.read_csv(local.PRIOR / "fft_peak_pairs_top20.csv")
    family = grid.read_csv(local.PRIOR / "grid_family_candidates.csv")
    original, windows, _ = local.analyze_image(IMAGE_ID, top, family, detailed=True)
    fundamentals, _ = local.ola(original, windows, "local_base")
    confirmed, _ = local.ola(original, windows, "local_all")
    images = {
        "No grid removal": original.astype(np.float32),
        "Local fundamentals": fundamentals.astype(np.float32),
        "Fundamental + confirmed harmonics": confirmed.astype(np.float32),
    }

    maps: dict[str, dict[str, np.ndarray]] = {}
    target_xs = target_ys = None
    for condition, image in images.items():
        result = spectra(image)
        step = SPACING // STRIDE
        target_xs = result["xs"][step:-step]
        target_ys = result["ys"][step:-step]
        raw_score, _, _ = strict_mean8_score(result["raw"])
        norm_score, _, _ = strict_mean8_score(result["norm"])
        maps[condition] = {"raw": raw_score, "norm": norm_score}

    assert target_xs is not None and target_ys is not None
    normal, annotation_masks = normal_mask(target_xs, target_ys, original.shape)
    # The scale is anchored to the original image's auxiliary normal region,
    # then held fixed across all grid-removal conditions.
    scales = {
        key: limits(maps["No grid removal"][key][normal])
        for key in ("norm", "raw")
    }

    fig, axes = plt.subplots(4, 2, figsize=(15, 19), constrained_layout=True)
    column_info = (("norm", "Normalised"), ("raw", "Unnormalised"))
    for col, (key, title) in enumerate(column_info):
        ax = axes[0, col]
        ax.imshow(original, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{title}: original image")
        ax.set_axis_off()

        lo, hi = scales[key]
        heatmaps = []
        for row, (condition, background) in enumerate(images.items(), start=1):
            ax = axes[row, col]
            ax.imshow(background, cmap="gray", vmin=0, vmax=255)
            heat = ax.imshow(
                maps[condition][key], extent=extent(target_xs, target_ys),
                cmap="inferno", vmin=lo, vmax=hi, alpha=.68,
                interpolation="nearest")
            heatmaps.append(heat)
            if annotation_masks["defect_pixel"].any():
                ax.contour(annotation_masks["defect_pixel"].astype(np.uint8),
                           [.5], colors="#38ff5a", linewidths=.8)
            if annotation_masks["exclusion_pixel"].any():
                ax.contour(annotation_masks["exclusion_pixel"].astype(np.uint8),
                           [.5], colors="#00e5ff", linewidths=.7)
            ax.set(xlim=(0, original.shape[1]), ylim=(original.shape[0], 0))
            ax.set_title(f"{condition}\nshared normal p05-p99: {lo:.3f}-{hi:.3f} dB")
            ax.set_axis_off()
        fig.colorbar(heatmaps[-1], ax=axes[1:, col], label="RMS (dB)",
                     shrink=.72, pad=.015)

    fig.suptitle(
        "image15: local radial-FFT comparison before/after grid removal\n"
        "W=128, stride=16, 8-reference mean at 48 px, 0.01-0.50 cycles/pixel",
        fontsize=15)
    output = OUT / "image15_normalised_unnormalised_shared_normal_scale.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)

    rows = []
    for key, display in column_info:
        lo, hi = scales[key]
        for condition in images:
            score = maps[condition][key]
            vals = score[np.isfinite(score)]
            normal_vals = score[normal & np.isfinite(score)]
            rows.append({
                "spectrum": display,
                "condition": condition,
                "display_p05": lo,
                "display_p99": hi,
                "display_scale_source": "original auxiliary normal region",
                "score_median": float(np.median(vals)),
                "score_p99": float(np.percentile(vals, 99)),
                "normal_median": float(np.median(normal_vals)),
                "normal_p99": float(np.percentile(normal_vals, 99)),
            })
    write_csv(OUT / "score_summary.csv", rows)
    np.savez_compressed(
        OUT / "image15_score_maps.npz", xs=target_xs, ys=target_ys,
        normal_mask=normal,
        **{f"{kind}_{key}": value[key]
           for kind, value in zip(("none", "fundamentals", "harmonics"), maps.values())
           for key in ("norm", "raw")})
    settings = {
        "image": IMAGE_ID, "window_px": WINDOW, "stride_px": STRIDE,
        "reference_distance_px": SPACING, "reference_count": 8,
        "reference_aggregation": "per-frequency arithmetic mean",
        "frequency_range_cycles_per_pixel": list(FREQ),
        "radial_bin_width_cycles_per_pixel": BIN_WIDTH,
        "normalisation": "divide 2D power by total power in 0.01-0.50 before radial averaging",
        "score": "RMS(target radial dB spectrum - mean of 8 reference radial dB spectra)",
        "display_scale": "original auxiliary normal-region p05-p99, shared across three conditions per column",
    }
    (OUT / "settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
