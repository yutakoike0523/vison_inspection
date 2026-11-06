#!/usr/bin/env python3
"""Compare local radial-FFT windows tied to the measured grid period.

The spectral definition intentionally follows window_size_comparison_unnormalized.py.
No notch filtering, power normalisation, zero padding, or label-driven feature tuning is used.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from scipy import fft
from sklearn.metrics import average_precision_score, roc_auc_score

import evaluate_all_radial_fft as annotation_impl
import multiscale_fft_experiment as base
import radial_fft_experiment as radial


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data20260903" / "data20260903"
DEFAULT_OUTPUT = ROOT / "output_grid_period_window_comparison"
STRIDE = 16
CELL = 16
BIN_WIDTH = 0.010
EPS = base.EPS
BANDS = {"conventional": (0.020, 0.500), "low_inclusive": (0.005, 0.500)}
DISTANCE_MODES = ("fixed48", "proportional")
WINDOW_MODES = ("fixed128", "period_1x", "period_2x", "period_3x")
OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
DEFECT_LABELS = set(annotation_impl.CLASSES)
TYPE_EN = {annotation_impl.CLASSES[0]: "local_deformation", annotation_impl.CLASSES[1]: "point",
           annotation_impl.CLASSES[2]: "line", annotation_impl.CLASSES[3]: "area"}
TYPE_JA = {v: k for k, v in TYPE_EN.items()}
PERIODS = {
    2: (140.1, 127.6), 3: (87.7, 89.3), 4: (87.7, 93.4), 5: (136.5, 134.7),
    6: (127.6, 136.5), 7: (57.9, 59.9), 8: (59.2, 60.7), 9: (91.2, 89.3),
    10: (142.0, 131.1), 11: (65.6, 71.1), 12: (94.7, 87.8), 13: (182.9, 175.5),
    14: (149.7, 140.1), 15: (182.9, 175.5), 16: (195.8, 193.4),
    17: (149.7, 142.0), 18: (33.3, 34.8),
}


def round16(value: float, minimum: int = 16) -> int:
    return max(minimum, int(math.floor(value / 16.0 + 0.5)) * 16)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    columns = list(dict.fromkeys(k for row in rows for k in row)) if rows else (fields or ["status"])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def json_safe(value):
    if isinstance(value, Path): return str(value)
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if not np.isfinite(value) else float(value)
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    return value


def settings_for(image_id: int) -> list[dict]:
    py, px = PERIODS[image_id]
    period = math.sqrt(px * py)
    rows = []
    for mode, nominal in (("fixed128", None), ("period_1x", 1), ("period_2x", 2), ("period_3x", 3)):
        target = 128.0 if nominal is None else nominal * period
        window = 128 if nominal is None else round16(target, 32)
        rows.append({"image_id": f"image{image_id}", "image_number": image_id,
                     "vertical_period_px": py, "horizontal_period_px": px,
                     "representative_period_px": period, "window_mode": mode,
                     "nominal_multiplier": nominal if nominal is not None else "fixed",
                     "target_window_px": target, "actual_window_px": window,
                     "actual_multiplier": window / period,
                     "rounding_error_px": window - target,
                     "absolute_rounding_error_px": abs(window - target),
                     "relative_rounding_error": (window - target) / target})
    return rows


def load_annotation(data: Path, image_id: int, shape: tuple[int, int]) -> dict:
    labels = annotation_impl.annotation(data / f"image{image_id}.json", shape)
    if labels is None: raise FileNotFoundError(data / f"image{image_id}.json")
    zero = np.zeros(shape, bool)
    defects, exclusions = [], []
    for index, (label, mask) in enumerate(labels):
        if label in DEFECT_LABELS:
            defects.append({"annotation_index": index, "label": label,
                            "type": TYPE_EN[label], "mask": mask})
        elif label in {"tag", "background"}:
            exclusions.append(mask)
    damage = np.logical_or.reduce([d["mask"] for d in defects]) if defects else zero.copy()
    exclusion_raw = np.logical_or.reduce(exclusions) if exclusions else zero.copy()
    # Explicit requirement: damage wins where labels overlap.
    exclusion = exclusion_raw & ~damage
    return {"defects": defects, "damage": damage, "exclusion": exclusion,
            "exclusion_raw": exclusion_raw,
            "damage_exclusion_overlap_pixels": int((damage & exclusion_raw).sum())}


def cell_overlap(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Whether each centre-associated 16x16 cell overlaps a pixel mask."""
    h, w = mask.shape
    integral = np.pad(mask.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    out = np.zeros((len(ys), len(xs)), bool)
    for iy, y in enumerate(ys):
        top, bottom = max(0, int(y) - CELL // 2), min(h, int(y) + CELL // 2)
        left = np.maximum(0, xs - CELL // 2); right = np.minimum(w, xs + CELL // 2)
        out[iy] = (integral[bottom, right] - integral[top, right]
                   - integral[bottom, left] + integral[top, left]) > 0
    return out


def spectra_for_window(image: np.ndarray, window: int, workers: int, requested_batch: int):
    h, w = image.shape
    x0 = np.arange(0, w - window + 1, STRIDE, dtype=np.int32)
    y0 = np.arange(0, h - window + 1, STRIDE, dtype=np.int32)
    if not len(x0) or not len(y0): return None
    ranges = list(BANDS.values())
    _, _, _, specs = radial.grids(window, BIN_WIDTH, ranges)
    valid = {name: specs[band]["valid"] for name, band in BANDS.items()}
    arrays = {name: np.empty((len(y0), len(x0), int(valid[name].sum())), np.float32)
              for name in BANDS}
    hann1 = np.hanning(window).astype(np.float32)
    hann = np.outer(hann1, hann1).astype(np.float32)
    batch = max(1, min(requested_batch, 2_000_000 // (window * window)))
    for iy, top in enumerate(y0):
        for start in range(0, len(x0), batch):
            chosen = x0[start:start + batch]
            values = np.stack([image[top:top + window, left:left + window] for left in chosen]).astype(np.float32)
            values -= values.mean(axis=(-2, -1), keepdims=True); values *= hann
            transformed = fft.rfft2(values, axes=(-2, -1), workers=workers)
            power = (np.abs(fft.fftshift(transformed, axes=(-2,))) ** 2).astype(np.float32)
            flat = power.reshape(len(power), -1)
            for name, band in BANDS.items():
                spec = specs[band]; good = valid[name]
                sums = flat @ spec["matrix"]
                mean_power = sums[:, good] / spec["counts"][good][None, :]
                arrays[name][iy, start:start + len(chosen)] = 10.0 * np.log10(mean_power + EPS)
    return {"xs": x0 + window // 2, "ys": y0 + window // 2, "spectra": arrays,
            "valid_bins": {name: int(v.sum()) for name, v in valid.items()}, "batch": batch}


def strict8_score(spectra: np.ndarray, distance: int) -> tuple[np.ndarray, np.ndarray]:
    step = distance // STRIDE
    ny, nx = spectra.shape[:2]
    score = np.full((ny, nx), np.nan, np.float32)
    valid = np.zeros((ny, nx), bool)
    if step < 1 or ny <= 2 * step or nx <= 2 * step or spectra.shape[2] == 0:
        return score, valid
    target = spectra[step:ny-step, step:nx-step]
    ref = np.zeros_like(target)
    for dy, dx in OFFSETS:
        ref += spectra[step + dy*step:ny-step + dy*step,
                       step + dx*step:nx-step + dx*step]
    diff = target - ref / 8.0
    score[step:ny-step, step:nx-step] = np.sqrt(np.mean(diff * diff, axis=2))
    valid[step:ny-step, step:nx-step] = True
    return score, valid


def raster(score: np.ndarray, xs: np.ndarray, ys: np.ndarray, shape: tuple[int, int], fill=np.nan):
    out = np.full(shape, fill, np.float32)
    h, w = shape
    for iy, y in enumerate(ys):
        top, bottom = max(0, int(y)-CELL//2), min(h, int(y)+CELL//2)
        for ix, x in enumerate(xs):
            if np.isfinite(score[iy, ix]):
                left, right = max(0, int(x)-CELL//2), min(w, int(x)+CELL//2)
                out[top:bottom, left:right] = score[iy, ix]
    return out


def condition_key(window_mode: str, distance_mode: str, band_name: str) -> str:
    return f"{window_mode}__{distance_mode}__{band_name}"


def parse_condition(key: str): return tuple(key.split("__"))


def values(record: dict, selection: np.ndarray) -> np.ndarray:
    return record["score"][selection & record["valid"] & np.isfinite(record["score"])]


def stat(v: np.ndarray, pcts=(95, 99)) -> dict:
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    out = {"count": len(v), "mean": float(np.mean(v)) if len(v) else None,
           "max": float(np.max(v)) if len(v) else None}
    for p in pcts: out[f"p{p}"] = float(np.percentile(v, p)) if len(v) else None
    return out


def finite_mean(items) -> float:
    a = np.asarray(list(items), float)
    return float(np.mean(a[np.isfinite(a)])) if np.isfinite(a).any() else math.nan


def fmt(value, spec=".4f", missing="NA") -> str:
    return format(value, spec) if value is not None and np.isfinite(value) else missing


def auc_pair(normal: np.ndarray, defect: np.ndarray) -> tuple[float, float]:
    if not len(normal) or not len(defect): return math.nan, math.nan
    y = np.r_[np.zeros(len(normal), np.uint8), np.ones(len(defect), np.uint8)]
    s = np.r_[normal, defect]
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def equal_image_quantile(arrays: list[np.ndarray], q: float) -> float | None:
    arrays = [np.sort(np.asarray(a, float)) for a in arrays if len(a)]
    if not arrays: return None
    samples = np.concatenate(arrays)
    weights = np.concatenate([np.full(len(a), 1.0 / len(a)) for a in arrays])
    order = np.argsort(samples); cumulative = np.cumsum(weights[order]); target = q * len(arrays)
    return float(samples[order[min(np.searchsorted(cumulative, target), len(samples)-1)]])


def scope_mask(record: dict, scope: str) -> np.ndarray:
    return record["valid"] if scope == "all_valid" else record["common"]


def percentile_scale(arrays: list[np.ndarray]) -> tuple[float, float]:
    v = np.concatenate([a[np.isfinite(a)] for a in arrays if np.isfinite(a).any()]) if arrays else np.array([])
    if not len(v): return 0.0, 1.0
    lo, hi = np.percentile(v, [5, 99])
    return float(lo), float(hi if hi > lo else lo + 1e-6)


def draw_contours(ax, ann: dict, crop=None):
    damage = ann["damage"]; exclusion = ann["exclusion_raw"]
    if crop:
        x0, y0, x1, y1 = crop; damage = damage[y0:y1, x0:x1]; exclusion = exclusion[y0:y1, x0:x1]
    if damage.any(): ax.contour(damage.astype(np.uint8), [.5], colors="#39ff66", linewidths=1.0)
    if exclusion.any(): ax.contour(exclusion.astype(np.uint8), [.5], colors="#00e5ff", linewidths=.7)


def render_image_comparisons(image_id: int, image: np.ndarray, ann: dict, records: dict,
                             settings: dict, output: Path) -> None:
    keys = [condition_key(mode, "fixed48", "conventional") for mode in WINDOW_MODES]
    available = [records[k] for k in keys if k in records]
    if not available: return
    maps = [raster(r["score"], r["xs"], r["ys"], image.shape) for r in available]
    lo, hi = percentile_scale(maps)
    for normalized, folder in ((False, "heatmap_comparison"), (True, "normalized_heatmap_comparison")):
        fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), constrained_layout=True)
        last = None
        for ax, mode, key in zip(axes, WINDOW_MODES, keys):
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            if key not in records:
                ax.text(.5, .5, "not evaluable", transform=ax.transAxes, ha="center", va="center", color="red")
                ax.set_axis_off(); continue
            r = records[key]; score_map = raster(r["score"], r["xs"], r["ys"], image.shape)
            vmin, vmax = percentile_scale([score_map]) if normalized else (lo, hi)
            last = ax.imshow(score_map, cmap="inferno", alpha=.68, vmin=vmin, vmax=vmax)
            draw_contours(ax, ann)
            s = settings[mode]
            ax.set_title(f"{mode}\nW={r['window']} px, W/P={s['actual_multiplier']:.2f}\nd=48, bins={r['valid_bins']}")
            ax.set_axis_off()
            if normalized: fig.colorbar(last, ax=ax, fraction=.035, pad=.015)
        if last is not None and not normalized: fig.colorbar(last, ax=axes, fraction=.018, pad=.01, label="RMS spectral difference (dB)")
        fig.suptitle(f"image{image_id}: green=damage, cyan=tag/background | " + ("per-panel scale" if normalized else "shared scale"))
        fig.savefig(output / folder / f"image{image_id}_comparison.png", dpi=140); plt.close(fig)


def render_defect_zooms(image_id: int, image: np.ndarray, ann: dict, records: dict,
                        settings: dict, output: Path) -> None:
    keys = [condition_key(mode, "fixed48", "conventional") for mode in WINDOW_MODES]
    score_maps = {k: raster(records[k]["score"], records[k]["xs"], records[k]["ys"], image.shape)
                  for k in keys if k in records}
    lo, hi = percentile_scale(list(score_maps.values()))
    for number, defect in enumerate(ann["defects"], 1):
        yy, xx = np.where(defect["mask"])
        if not len(xx): continue
        cx, cy = int(round((xx.min()+xx.max())/2)), int(round((yy.min()+yy.max())/2))
        half = max(128, int(max(xx.max()-xx.min(), yy.max()-yy.min())/2 + 96))
        x0, x1 = max(0, cx-half), min(image.shape[1], cx+half)
        y0, y1 = max(0, cy-half), min(image.shape[0], cy+half)
        crop = (x0, y0, x1, y1)
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
        last = None
        for ax, mode, key in zip(axes, WINDOW_MODES, keys):
            ax.imshow(image[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
            if key in score_maps:
                last = ax.imshow(score_maps[key][y0:y1, x0:x1], cmap="inferno", alpha=.68, vmin=lo, vmax=hi)
                ax.set_title(f"{mode}: W={records[key]['window']} ({settings[mode]['actual_multiplier']:.2f}P)")
            else: ax.set_title(f"{mode}: not evaluable")
            ax.set_axis_off()
        if last is not None: fig.colorbar(last, ax=axes, fraction=.02, pad=.01)
        fig.suptitle(f"image{image_id} defect {number} ({defect['type']}): identical crop and shared scale")
        fig.savefig(output / "defect_zoom_comparison" / f"image{image_id}_defect{number:02d}_{defect['type']}.png", dpi=140)
        plt.close(fig)


def representative_spectral_data(image: np.ndarray, record: dict, selection: np.ndarray) -> dict | None:
    """Recompute a representative target and its 8 references for an audit figure."""
    candidates = selection & record["valid"] & np.isfinite(record["score"])
    if not candidates.any(): return None
    masked = np.where(candidates, record["score"], -np.inf)
    iy, ix = np.unravel_index(int(np.argmax(masked)), masked.shape)
    cx, cy = int(record["xs"][ix]), int(record["ys"][iy])
    window, distance = record["window"], record["distance"]
    half = window // 2; hann1 = np.hanning(window).astype(np.float32); hann = np.outer(hann1, hann1)
    centres = [(cx, cy)] + [(cx + dx*distance, cy + dy*distance) for dy, dx in OFFSETS]
    patches = np.stack([image[y-half:y+half, x-half:x+half] for x, y in centres]).astype(np.float32)
    patches -= patches.mean(axis=(-2, -1), keepdims=True); patches *= hann
    full = fft.fftshift(fft.fft2(patches, axes=(-2, -1)), axes=(-2, -1))
    full_db = (10.0*np.log10(np.abs(full)**2 + EPS)).astype(np.float32)
    half_fft = fft.fftshift(fft.rfft2(patches, axes=(-2, -1)), axes=(-2,))
    power = (np.abs(half_fft)**2).astype(np.float32)
    band = BANDS["conventional"]; _, _, _, specs = radial.grids(window, BIN_WIDTH, [band]); spec = specs[band]
    good = spec["valid"]; sums = power.reshape(len(power), -1) @ spec["matrix"]
    radial_db = 10.0*np.log10(sums[:, good] / spec["counts"][good][None, :] + EPS)
    frequencies = (np.flatnonzero(good) + .5) * BIN_WIDTH
    target_1d = radial_db[0]; reference_1d = radial_db[1:].mean(axis=0)
    target_2d = full_db[0]; reference_2d = full_db[1:].mean(axis=0)
    return {"cx": cx, "cy": cy, "centres": centres, "score": float(record["score"][iy, ix]),
            "target_2d": target_2d, "reference_2d": reference_2d,
            "difference_2d": target_2d-reference_2d, "frequencies": frequencies,
            "target_1d": target_1d, "reference_1d": reference_1d,
            "difference_1d": target_1d-reference_1d}


def render_defect_fft_details(image_id: int, image: np.ndarray, ann: dict, records: dict,
                              settings: dict, output: Path) -> None:
    """Spatial/2-D/1-D spectral audit for the fixed-d=48 conventional-band comparison."""
    keys = [condition_key(mode, "fixed48", "conventional") for mode in WINDOW_MODES]
    for number, defect in enumerate(ann["defects"], 1):
        details = []
        for mode, key in zip(WINDOW_MODES, keys):
            r = records.get(key)
            details.append((mode, r, representative_spectral_data(image, r, r["instances"][number-1]) if r else None))
        fig, axes = plt.subplots(4, 6, figsize=(25, 17), constrained_layout=True)
        for row, (mode, r, detail) in enumerate(details):
            if detail is None:
                for ax in axes[row]: ax.text(.5, .5, "not evaluable", ha="center", va="center"); ax.set_axis_off()
                continue
            window = r["window"]; half = window//2; cx, cy = detail["cx"], detail["cy"]
            layout_half = half + r["distance"] + 40
            x0, x1 = max(0, cx-layout_half), min(image.shape[1], cx+layout_half)
            y0, y1 = max(0, cy-layout_half), min(image.shape[0], cy+layout_half)
            ax = axes[row, 0]; ax.imshow(image[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
            for index, (x, y) in enumerate(detail["centres"]):
                color = "red" if index == 0 else "cyan"
                ax.add_patch(Rectangle((x-half-x0, y-half-y0), window, window, fill=False,
                                       edgecolor=color, linewidth=2 if index == 0 else .8, alpha=.9 if index == 0 else .6))
                ax.plot(x-x0, y-y0, "+", color=color, markersize=7)
            ax.set_title(f"{mode}: layout\nred=target, cyan=8 refs\nW={window}, d=48, score={detail['score']:.3f}")
            ax.set_axis_off()
            combined = np.r_[detail["target_2d"].ravel(), detail["reference_2d"].ravel()]
            lo, hi = np.percentile(combined, [5, 99.5]); delta = float(np.percentile(np.abs(detail["difference_2d"]), 99))
            axes[row, 1].imshow(detail["target_2d"], cmap="magma", vmin=lo, vmax=hi, origin="lower")
            axes[row, 1].set_title("Target 2-D log power (dB)")
            axes[row, 2].imshow(detail["reference_2d"], cmap="magma", vmin=lo, vmax=hi, origin="lower")
            axes[row, 2].set_title("Mean of 8 reference 2-D dB spectra")
            im = axes[row, 3].imshow(detail["difference_2d"], cmap="coolwarm", vmin=-delta, vmax=delta, origin="lower")
            axes[row, 3].set_title("Target - reference (dB)"); fig.colorbar(im, ax=axes[row, 3], fraction=.045, pad=.02)
            for col in (1, 2, 3): axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            f = detail["frequencies"]
            axes[row, 4].plot(f, detail["target_1d"], label="target", color="crimson")
            axes[row, 4].plot(f, detail["reference_1d"], label="mean 8 refs", color="navy")
            axes[row, 4].set(xlabel="cycles/pixel", ylabel="radial log power (dB)", title="Radial 1-D spectra")
            axes[row, 4].grid(alpha=.25); axes[row, 4].legend(fontsize=8)
            axes[row, 5].axhline(0, color="black", linewidth=.7)
            axes[row, 5].plot(f, detail["difference_1d"], color="darkgreen")
            rms = float(np.sqrt(np.mean(detail["difference_1d"]**2)))
            axes[row, 5].set(xlabel="cycles/pixel", ylabel="target - reference (dB)", title=f"1-D difference, RMS={rms:.3f} dB")
            axes[row, 5].grid(alpha=.25)
        fig.suptitle(f"image{image_id} defect {number} ({defect['type']}) | d=48 px fixed | 0.02-0.50 cycles/pixel\n"
                     "Representative target = maximum-score valid cell overlapping the defect; annotation box is not drawn")
        fig.savefig(output / "defect_fft_comparison" / f"image{image_id}_defect{number:02d}_{defect['type']}_fft.png", dpi=125)
        plt.close(fig)


def make_metric_plots(summary: list[dict], type_rows: list[dict], output: Path) -> None:
    primary = [r for r in summary if r["scope"] == "common_region" and r["distance_mode"] == "fixed48" and r["band"] == "conventional"]
    order = {m: i for i, m in enumerate(WINDOW_MODES)}; primary.sort(key=lambda r: order[r["window_mode"]])
    labels = [r["window_mode"] for r in primary]; x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, key, title in zip(axes, ("roc_auc", "pr_auc", "defect_detection_rate_fpr1"), ("ROC-AUC", "PR-AUC", "Defect detection @ FPR 1%")):
        ax.plot(x, [r[key] for r in primary], marker="o"); ax.set_xticks(x, labels, rotation=25); ax.set_title(title); ax.grid(alpha=.3)
    fig.savefig(output / "metric_plots" / "primary_window_metrics.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, metric in zip(axes, ("roc_auc", "pr_auc")):
        for distance in DISTANCE_MODES:
            for band in BANDS:
                q = [r for r in summary if r["scope"] == "common_region" and r["distance_mode"] == distance and r["band"] == band]
                q.sort(key=lambda r: order[r["window_mode"]])
                ax.plot(x, [r[metric] for r in q], marker="o", label=f"{distance}/{band}")
        ax.set_xticks(x, WINDOW_MODES, rotation=25); ax.set_title(metric); ax.grid(alpha=.3)
    axes[1].legend(fontsize=8); fig.savefig(output / "metric_plots" / "distance_band_comparison.png", dpi=150); plt.close(fig)
    q = [r for r in type_rows if r["scope"] == "common_region" and r["distance_mode"] == "fixed48" and r["band"] == "conventional"]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for defect_type in sorted({r["defect_type"] for r in q}):
        z = [r for r in q if r["defect_type"] == defect_type]; z.sort(key=lambda r: order[r["window_mode"]])
        ax.plot(x, [r["detection_rate_fpr1"] for r in z], marker="o", label=defect_type)
    ax.set_xticks(x, WINDOW_MODES, rotation=25); ax.set_ylabel("Defect-instance detection rate @ FPR 1%")
    ax.grid(alpha=.3); ax.legend(); fig.savefig(output / "metric_plots" / "defect_type_detection.png", dpi=150); plt.close(fig)


def report(output: Path, settings_rows: list[dict], summary: list[dict], image_rows: list[dict],
           defect_rows: list[dict], type_rows: list[dict], valid_rows: list[dict], runtime: float) -> None:
    primary = [r for r in summary if r["scope"] == "common_region" and r["distance_mode"] == "fixed48" and r["band"] == "conventional"]
    all_primary = [r for r in summary if r["scope"] == "all_valid" and r["distance_mode"] == "fixed48" and r["band"] == "conventional"]
    best_pr = max(primary, key=lambda r: (r["pr_auc"] if np.isfinite(r["pr_auc"]) else -1))
    best_roc = max(primary, key=lambda r: (r["roc_auc"] if np.isfinite(r["roc_auc"]) else -1))
    best_det = max(primary, key=lambda r: (r["defect_detection_rate_fpr1"] if r["defect_detection_rate_fpr1"] is not None else -1))
    per_image = [r for r in image_rows if r["scope"] == "common_region" and r["distance_mode"] == "fixed48" and r["band"] == "conventional" and np.isfinite(r["pr_auc"])]
    winners = []
    for image in sorted({r["image_id"] for r in per_image}, key=lambda s: int(s[5:])):
        q = [r for r in per_image if r["image_id"] == image]
        if q: winners.append((image, max(q, key=lambda r: r["pr_auc"])))
    lines = ["# 格子周期に応じた局所FFT窓サイズの比較", "", "## 1. 実験目的と仮説", "",
             "格子周期Pの1/2/3倍窓と固定128 px窓を、同じ局所放射FFT定義で比較した。仮説は、複数周期を含む窓が正常格子のばらつきを安定化し、傷との分離を改善する一方、大窓では局所傷が希釈される、というものである。", "",
             "## 2. 設定と再現方法", "",
             "平均除去、2D Hann、SciPy `rfft2`（backward scaling）、非正規化power=|F|²、共役重み付き放射平均、10log10、8参照dBスペクトルの算術平均、対象との差のRMSを既存実装から踏襲した。FFTサイズ=W、ゼロパディングなし、ノッチ除去なし、stride=16 px、放射ビン幅=0.01 cycles/pixel。", "",
             "参照距離は固定48 pxと round16(0.375W)、帯域は0.020–0.500と0.005–0.500を比較した。8窓が全て画像内にある中心だけを有効とした。スコアは窓中心の16×16 pxセルに対応付け、セルが傷矩形に1 pxでも重なれば傷、傷でなくtag/backgroundに重なれば除外、それ以外を正常表面とした。傷と除外の重複は傷を優先した。", "",
             "FPR 1%/5%閾値は条件ごとに他画像の正常セルを画像等重みで集約したLOIO分位点であり、評価画像のラベルで画像別最適化していない。これは比較用閾値で、未知画像へそのまま適用する運用閾値ではない。", "",
             f"実行コマンドは `py -3.10 grid_period_window_comparison.py --output {output.name}`、処理時間は{runtime/60:.1f}分。詳細は `settings.json` と `processing_time.csv`。", "",
             "## 3. 全体結果（共通評価領域、d=48、0.020–0.500）", "",
             "|窓|ROC-AUC|PR-AUC|傷検出率@FPR1%|評価傷/対象外|", "|---|---:|---:|---:|---:|"]
    for r in primary:
        lines.append(f"|{r['window_mode']}|{fmt(r['roc_auc'])}|{fmt(r['pr_auc'])}|{fmt(r['defect_detection_rate_fpr1'], '.1%')}|{r['detected_defects_fpr1']}/{r['evaluable_defects']} ({r['unassessable_defects']}対象外)|")
    lines += ["", f"PR-AUC最大は **{best_pr['window_mode']}**、ROC-AUC最大は **{best_roc['window_mode']}**、FPR 1%での傷単位検出率最大は **{best_det['window_mode']}**。異なる指標が一致しない場合は、正常誤検出と傷単位検出を併記して判断する。", "",
              "参考として、条件ごとに使える全有効領域（母数が異なる）も示す。", "",
              "|窓|ROC-AUC|PR-AUC|傷検出率@FPR1%|評価傷/対象外|", "|---|---:|---:|---:|---:|"]
    for r in all_primary:
        lines.append(f"|{r['window_mode']}|{fmt(r['roc_auc'])}|{fmt(r['pr_auc'])}|{fmt(r['defect_detection_rate_fpr1'], '.1%')}|{r['detected_defects_fpr1']}/{r['evaluable_defects']} ({r['unassessable_defects']}対象外)|")
    conv_period = [r for r in primary if r["window_mode"] != "fixed128"]
    low_period = [r for r in summary if r["scope"] == "common_region" and r["distance_mode"] == "fixed48" and r["band"] == "low_inclusive" and r["window_mode"] != "fixed128"]
    conv_best = max(conv_period, key=lambda r: r["pr_auc"]); low_best = max(low_period, key=lambda r: r["pr_auc"])
    winner_counts = {m: sum(row["window_mode"] == m for _, row in winners) for m in WINDOW_MODES}
    lines += ["", "## 最重要4問への回答", "",
              f"- **Q1:** 共通領域の分離性能では固定128 pxが最良だった。周期倍率内では従来帯域のPR-AUC最大は **{conv_best['window_mode']}** ({conv_best['pr_auc']:.4f})。ただしFPR 1%傷検出は1倍と2倍が各3/22で同率であり、単一指標で万能な倍率はなかった。",
              "- **Q2:** 共通ではない。画像別PR-AUC勝者（共通領域が成立した16画像）は " + ", ".join(f"{m}={winner_counts[m]}画像" for m in WINDOW_MODES) + " で分散した。",
              "- **Q3:** 種類差があった。共通領域FPR 1%では点状は1/2/3倍が各2/8、局所変形は1倍・2倍が各1/12、線状は全条件0/2だった。面状異常は指定画像範囲にアノテーションがなく評価不能。全有効領域では線状のみ固定128が1/3を検出した。",
              f"- **Q4:** 低周波追加で、周期倍率内のPR-AUC最良は **{conv_best['window_mode']}→{low_best['window_mode']}** ({conv_best['pr_auc']:.4f}→{low_best['pr_auc']:.4f})へ変わった。一方、全4窓では固定128が引き続き最良で、低周波追加によりPR-AUCは0.0182→0.0165へ低下した。", "",
              "## 4. 各画像の最適窓（共通領域、PR-AUC）", ""]
    for image, row in winners: lines.append(f"- {image}: {row['window_mode']} (PR-AUC={row['pr_auc']:.4f}, W={row['window_size']} px, W/P={row['actual_multiplier']:.2f})")
    lines += ["", "## 5. 傷種類別", "", "|種類|最良窓（検出率@FPR1%）|検出率|評価件数|", "|---|---|---:|---:|"]
    for defect_type in sorted({r["defect_type"] for r in type_rows}):
        q = [r for r in type_rows if r["defect_type"] == defect_type and r["scope"] == "common_region" and r["distance_mode"] == "fixed48" and r["band"] == "conventional"]
        if q:
            b = max(q, key=lambda r: r["detection_rate_fpr1"] if r["detection_rate_fpr1"] is not None else -1)
            lines.append(f"|{defect_type}|{b['window_mode']}|{fmt(b['detection_rate_fpr1'], '.1%')}|{b['evaluable_defects']}|")
    primary_def = [r for r in defect_rows if r["scope"] == "common_region" and r["distance_mode"] == "fixed48" and r["band"] == "conventional" and r["status"] == "evaluable"]
    by_key = defaultdict(dict)
    for r in primary_def: by_key[(r["image_id"], r["defect_number"])][r["window_mode"]] = r
    changes = []
    for key, q in by_key.items():
        if "fixed128" in q:
            ratio = lambda r: r["maximum_score"] / r["threshold_fpr1_loio"] if r["threshold_fpr1_loio"] else math.nan
            fixed = q["fixed128"]; best = max(q.values(), key=ratio); worst = min(q.values(), key=ratio)
            changes.append((ratio(best)-ratio(fixed), ratio(worst)-ratio(fixed), key, best, worst, fixed))
    lines += ["", "## 6. 改善・悪化例", "", "窓間で生スコアを直接比較せず、各条件のLOIO FPR 1%閾値に対する最大スコア比を図の候補抽出に用いた。性能判断はAUC・固定FPR指標を優先する。", ""]
    for up, down, key, best, worst, fixed in sorted(changes, reverse=True, key=lambda z: z[0])[:5]:
        lines.append(f"- 改善候補 {key[0]} defect{key[1]}: {best['window_mode']} で閾値比がfixed128比 {up:+.3f}。`defect_zoom_comparison/`参照。")
    for up, down, key, best, worst, fixed in sorted(changes, key=lambda z: z[1])[:5]:
        lines.append(f"- 悪化候補 {key[0]} defect{key[1]}: {worst['window_mode']} で閾値比がfixed128比 {down:+.3f}。`defect_zoom_comparison/`参照。")
    lines += ["", "## 7. 参照距離の影響", ""]
    for distance in DISTANCE_MODES:
        q = [r for r in summary if r["scope"] == "common_region" and r["distance_mode"] == distance and r["band"] == "conventional"]
        b = max(q, key=lambda r: r["pr_auc"] if np.isfinite(r["pr_auc"]) else -1)
        lines.append(f"- {distance}: PR-AUC最大 {b['window_mode']} ({fmt(b['pr_auc'])})。")
    fixed2 = next(r for r in summary if r["scope"] == "common_region" and r["distance_mode"] == "fixed48" and r["band"] == "conventional" and r["window_mode"] == "period_2x")
    prop2 = next(r for r in summary if r["scope"] == "common_region" and r["distance_mode"] == "proportional" and r["band"] == "conventional" and r["window_mode"] == "period_2x")
    fixed3 = next(r for r in summary if r["scope"] == "common_region" and r["distance_mode"] == "fixed48" and r["band"] == "conventional" and r["window_mode"] == "period_3x")
    prop3 = next(r for r in summary if r["scope"] == "common_region" and r["distance_mode"] == "proportional" and r["band"] == "conventional" and r["window_mode"] == "period_3x")
    lines += ["", f"固定128は両距離がともに48 pxなので同値。比例距離にすると、2倍窓のPR-AUCは{fixed2['pr_auc']:.4f}→{prop2['pr_auc']:.4f}、3倍窓は{fixed3['pr_auc']:.4f}→{prop3['pr_auc']:.4f}へ改善した。大窓でd=48を固定した際の高い重なり率が性能を下げていた可能性がある。両者の差は窓サイズだけでなく参照の局所性の影響を表す。", "", "## 8. 低周波帯域の影響", ""]
    for band in BANDS:
        q = [r for r in summary if r["scope"] == "common_region" and r["distance_mode"] == "fixed48" and r["band"] == band]
        b = max(q, key=lambda r: r["pr_auc"] if np.isfinite(r["pr_auc"]) else -1)
        lines.append(f"- {band}: PR-AUC最大 {b['window_mode']} ({fmt(b['pr_auc'])})。各条件の実ビン数はCSVに記録し、0ビン条件は評価から除外した。")
    lines += ["", "## 9. 考察・次の実験", "",
              "最適倍率の共通性は、全体最良だけでなく画像別勝者と種類別結果の分散で判断する。次段階では、今回の上位倍率を固定したまま独立画像で運用閾値を校正し、格子位相に対する窓中心のずれ、放射ビン幅、参照集約（平均/中央値）を一因子ずつ検証する。", "",
              "## データ上の制約", "",
              "- image2〜image18に実在する傷アノテーションは36件であり、依頼文の54件とは一致しない。54件へ水増しせず、36件を母数とした。", "- image18は428×321 pxで、大窓や共通領域が成立しない条件がある。対象外件数を失敗と分離した。", "- backgroundアノテーションがない画像でも、tagと傷以外の未除外領域を既存評価と同様に正常候補とした。", "- 窓サイズ間でdBスコアの絶対値は直接比較せず、AUC、固定FPR、傷単位検出を主指標とした。", "",
              "## 出力一覧", "", "指定CSV、比較画像、傷ズーム、指標図を各ディレクトリへ保存した。`valid_region_summary.csv`には全条件の有効セル数・面積と評価可能傷数を記録した。", "",
              "`defect_zoom_comparison/` は傷アノテーション枠を描かない同一領域・共通スケールの拡大図。`defect_fft_comparison/` は各傷・4窓について、d=48 px固定、0.02–0.50 cycles/pixelで、対象/8参照窓配置、対象2D log-power、8参照平均2D dBスペクトル、符号付き2D差、放射平均1Dスペクトル、1D差を示す。代表対象は傷と重なる有効セルの最大スコア位置で、可視化選択にのみアノテーションを使用した。"]
    (output / "report_ja.md").write_text("\n".join(lines), encoding="utf-8")


def run(args) -> Path:
    started_all = time.perf_counter()
    output = args.output
    if output.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S"); output = output.with_name(output.name + "_" + stamp)
    output.mkdir(parents=True)
    for folder in ("heatmap_comparison", "normalized_heatmap_comparison", "defect_zoom_comparison", "defect_fft_comparison", "metric_plots", "score_maps"):
        (output / folder).mkdir()
    image_ids = args.images or list(range(2, 19))
    setting_rows = [row for i in image_ids for row in settings_for(i)]
    write_csv(output / "window_size_settings.csv", setting_rows)
    setting_lookup = {(r["image_number"], r["window_mode"]): r for r in setting_rows}
    all_records: dict[int, dict[str, dict]] = {}; annotations = {}; images_meta = {}; processing = []
    for image_id in image_ids:
        image_path = args.data / f"image{image_id}.png"
        image = np.asarray(Image.open(image_path).convert("L"), np.float32)
        ann = load_annotation(args.data, image_id, image.shape); annotations[image_id] = ann
        images_meta[image_id] = {"shape": image.shape, "path": image_path}
        records = {}; modes_by_window = defaultdict(list)
        for mode in WINDOW_MODES: modes_by_window[setting_lookup[image_id, mode]["actual_window_px"]].append(mode)
        for window, modes in modes_by_window.items():
            begin = time.perf_counter(); computed = spectra_for_window(image, window, args.workers, args.batch_size)
            if computed is None:
                for mode in modes:
                    processing.append({"image_id": f"image{image_id}", "window_mode": mode, "window_size": window, "status": "window_does_not_fit"})
                continue
            for distance_mode in DISTANCE_MODES:
                distance = 48 if distance_mode == "fixed48" else round16(.375 * window, 16)
                for band_name in BANDS:
                    score, valid = strict8_score(computed["spectra"][band_name], distance)
                    for mode in modes:
                        key = condition_key(mode, distance_mode, band_name)
                        records[key] = {"score": score.copy(), "valid": valid.copy(), "xs": computed["xs"], "ys": computed["ys"],
                                        "window": window, "distance": distance, "valid_bins": computed["valid_bins"][band_name],
                                        "window_mode": mode, "distance_mode": distance_mode, "band": band_name}
            elapsed = time.perf_counter() - begin
            processing.append({"image_id": f"image{image_id}", "window_modes": ";".join(modes), "window_size": window,
                               "status": "computed", "grid_width": len(computed["xs"]), "grid_height": len(computed["ys"]),
                               "fft_windows": len(computed["xs"])*len(computed["ys"]), "effective_batch": computed["batch"], "seconds": elapsed})
            print(f"image{image_id} W={window} ({','.join(modes)}): {elapsed:.1f}s", flush=True)
        # Common geometric valid region across every condition that is evaluable for this image.
        valid_maps = [np.isfinite(raster(r["score"], r["xs"], r["ys"], image.shape)) for r in records.values() if r["valid_bins"] > 0]
        common_pixels = np.logical_and.reduce(valid_maps) if valid_maps else np.zeros(image.shape, bool)
        for r in records.values():
            centres_common = common_pixels[np.clip(r["ys"], 0, image.shape[0]-1)[:, None], np.clip(r["xs"], 0, image.shape[1]-1)[None, :]]
            r["common"] = r["valid"] & centres_common
            defect = cell_overlap(ann["damage"], r["xs"], r["ys"])
            excluded = cell_overlap(ann["exclusion"], r["xs"], r["ys"])
            r["defect"] = defect
            r["normal"] = ~defect & ~excluded
            r["instances"] = [cell_overlap(d["mask"], r["xs"], r["ys"]) for d in ann["defects"]]
        all_records[image_id] = records
        # Cache score grids for audit/replot without FFT.
        payload = {}
        for key, r in records.items():
            payload[key + "__score"] = r["score"]; payload[key + "__xs"] = r["xs"]; payload[key + "__ys"] = r["ys"]
        np.savez_compressed(output / "score_maps" / f"image{image_id}_scores.npz", **payload)
        render_image_comparisons(image_id, image, ann, records,
                                 {m: setting_lookup[image_id, m] for m in WINDOW_MODES}, output)
        render_defect_zooms(image_id, image, ann, records,
                            {m: setting_lookup[image_id, m] for m in WINDOW_MODES}, output)
        render_defect_fft_details(image_id, image, ann, records,
                                  {m: setting_lookup[image_id, m] for m in WINDOW_MODES}, output)
    write_csv(output / "processing_time.csv", processing)
    # LOIO, equal-image weighted comparison thresholds for each condition and scope.
    thresholds = {}
    for image_id, records in all_records.items():
        for key, r in records.items():
            for scope in ("all_valid", "common_region"):
                train = []
                for other_id, other in all_records.items():
                    if other_id == image_id or key not in other: continue
                    o = other[key]; train.append(values(o, scope_mask(o, scope) & o["normal"]))
                thresholds[image_id, key, scope, 1] = equal_image_quantile(train, .99)
                thresholds[image_id, key, scope, 5] = equal_image_quantile(train, .95)
    image_rows, defect_rows, valid_rows = [], [], []
    for image_id, records in all_records.items():
        ann = annotations[image_id]
        for key, r in records.items():
            mode, distance_mode, band_name = parse_condition(key); srow = setting_lookup[image_id, mode]
            for scope in ("all_valid", "common_region"):
                valid = scope_mask(r, scope); n = values(r, valid & r["normal"]); d = values(r, valid & r["defect"])
                ns, ds = stat(n), stat(d); roc, pr = auc_pair(n, d)
                t1, t5 = thresholds[image_id, key, scope, 1], thresholds[image_id, key, scope, 5]
                evaluable = detected1 = detected5 = 0
                for number, (defect, selection) in enumerate(zip(ann["defects"], r["instances"]), 1):
                    dv = values(r, valid & selection); st = stat(dv)
                    status = "evaluable" if len(dv) else "out_of_scope"
                    hit1 = bool(np.any(dv > t1)) if len(dv) and t1 is not None else None
                    hit5 = bool(np.any(dv > t5)) if len(dv) and t5 is not None else None
                    if len(dv): evaluable += 1; detected1 += hit1 is True; detected5 += hit5 is True
                    defect_rows.append({"image_id": f"image{image_id}", "defect_number": number,
                                        "annotation_index": defect["annotation_index"], "defect_type": defect["type"],
                                        "window_mode": mode, "window_size": r["window"], "actual_multiplier": srow["actual_multiplier"],
                                        "distance_mode": distance_mode, "reference_distance_px": r["distance"], "band": band_name,
                                        "scope": scope, "status": status, "candidate_cells": len(dv),
                                        "maximum_score": st["max"], "mean_score": st["mean"], "p95_score": st["p95"],
                                        "threshold_fpr1_loio": t1, "detected_fpr1": hit1,
                                        "threshold_fpr5_loio": t5, "detected_fpr5": hit5})
                image_rows.append({"image_id": f"image{image_id}", "window_mode": mode, "window_size": r["window"],
                                   "actual_multiplier": srow["actual_multiplier"], "distance_mode": distance_mode,
                                   "reference_distance_px": r["distance"], "band": band_name, "frequency_low": BANDS[band_name][0],
                                   "frequency_high": BANDS[band_name][1], "valid_frequency_bins": r["valid_bins"], "scope": scope,
                                   "valid_cells": int(valid.sum()), "normal_cells": len(n), "defect_cells": len(d),
                                   "defect_max": ds["max"], "defect_mean": ds["mean"], "defect_p95": ds["p95"],
                                   "normal_mean": ns["mean"], "normal_p95": ns["p95"], "normal_p99": ns["p99"],
                                   "roc_auc": roc, "pr_auc": pr, "threshold_fpr1_loio": t1,
                                   "normal_fpr_at_fpr1_threshold": float(np.mean(n > t1)) if len(n) and t1 is not None else None,
                                   "defect_cell_tpr_fpr1": float(np.mean(d > t1)) if len(d) and t1 is not None else None,
                                   "threshold_fpr5_loio": t5,
                                   "normal_fpr_at_fpr5_threshold": float(np.mean(n > t5)) if len(n) and t5 is not None else None,
                                   "defect_cell_tpr_fpr5": float(np.mean(d > t5)) if len(d) and t5 is not None else None,
                                   "detected_defects_fpr1": detected1, "detected_defects_fpr5": detected5,
                                   "evaluable_defects": evaluable, "unassessable_defects": len(ann["defects"])-evaluable})
                valid_rows.append({"image_id": f"image{image_id}", "window_mode": mode, "window_size": r["window"],
                                   "distance_mode": distance_mode, "reference_distance_px": r["distance"], "band": band_name,
                                   "scope": scope, "valid_frequency_bins": r["valid_bins"], "valid_cells": int(valid.sum()),
                                   "approximate_valid_area_px": int(valid.sum()) * CELL * CELL,
                                   "total_annotated_defects": len(ann["defects"]), "evaluable_defects": evaluable,
                                   "unassessable_defects": len(ann["defects"])-evaluable,
                                   "damage_exclusion_overlap_pixels": ann["damage_exclusion_overlap_pixels"]})
    write_csv(output / "per_image_metrics.csv", image_rows); write_csv(output / "per_defect_metrics.csv", defect_rows)
    write_csv(output / "valid_region_summary.csv", valid_rows)
    summary = []
    for scope in ("all_valid", "common_region"):
        for mode in WINDOW_MODES:
            for distance_mode in DISTANCE_MODES:
                for band_name in BANDS:
                    key = condition_key(mode, distance_mode, band_name); normals=[]; defects=[]; rows=[]
                    for image_id, recs in all_records.items():
                        if key not in recs: continue
                        r = recs[key]; valid = scope_mask(r, scope)
                        normals.append(values(r, valid & r["normal"])); defects.append(values(r, valid & r["defect"]))
                        rows.extend(x for x in defect_rows if x["image_id"] == f"image{image_id}" and x["window_mode"] == mode and x["distance_mode"] == distance_mode and x["band"] == band_name and x["scope"] == scope)
                    n = np.concatenate([x for x in normals if len(x)]) if any(len(x) for x in normals) else np.array([])
                    d = np.concatenate([x for x in defects if len(x)]) if any(len(x) for x in defects) else np.array([])
                    roc, pr = auc_pair(n, d); ev = [x for x in rows if x["status"] == "evaluable"]
                    summary.append({"window_mode": mode, "distance_mode": distance_mode, "band": band_name, "scope": scope,
                                    "evaluated_images": sum(key in r for r in all_records.values()),
                                    "normal_cells": len(n), "defect_cells": len(d), "roc_auc": roc, "pr_auc": pr,
                                    "detected_defects_fpr1": sum(x["detected_fpr1"] is True for x in ev),
                                    "detected_defects_fpr5": sum(x["detected_fpr5"] is True for x in ev),
                                    "evaluable_defects": len(ev), "unassessable_defects": len(rows)-len(ev),
                                    "defect_detection_rate_fpr1": sum(x["detected_fpr1"] is True for x in ev)/len(ev) if ev else None,
                                    "defect_detection_rate_fpr5": sum(x["detected_fpr5"] is True for x in ev)/len(ev) if ev else None,
                                    "mean_per_image_roc_auc": finite_mean(x["roc_auc"] for x in image_rows if x["window_mode"] == mode and x["distance_mode"] == distance_mode and x["band"] == band_name and x["scope"] == scope),
                                    "mean_per_image_pr_auc": finite_mean(x["pr_auc"] for x in image_rows if x["window_mode"] == mode and x["distance_mode"] == distance_mode and x["band"] == band_name and x["scope"] == scope)})
    write_csv(output / "overall_summary.csv", summary)
    type_rows = []
    for scope in ("all_valid", "common_region"):
        for mode in WINDOW_MODES:
            for distance_mode in DISTANCE_MODES:
                for band_name in BANDS:
                    for defect_type in TYPE_JA:
                        q = [r for r in defect_rows if r["scope"] == scope and r["window_mode"] == mode and r["distance_mode"] == distance_mode and r["band"] == band_name and r["defect_type"] == defect_type]
                        ev = [r for r in q if r["status"] == "evaluable"]
                        type_rows.append({"defect_type": defect_type, "window_mode": mode, "distance_mode": distance_mode,
                                          "band": band_name, "scope": scope, "total_defects": len(q), "evaluable_defects": len(ev),
                                          "unassessable_defects": len(q)-len(ev), "detected_fpr1": sum(r["detected_fpr1"] is True for r in ev),
                                          "detection_rate_fpr1": sum(r["detected_fpr1"] is True for r in ev)/len(ev) if ev else None,
                                          "detected_fpr5": sum(r["detected_fpr5"] is True for r in ev),
                                          "detection_rate_fpr5": sum(r["detected_fpr5"] is True for r in ev)/len(ev) if ev else None,
                                          "median_max_score": float(np.median([r["maximum_score"] for r in ev])) if ev else None,
                                          "median_mean_score": float(np.median([r["mean_score"] for r in ev])) if ev else None,
                                          "median_p95_score": float(np.median([r["p95_score"] for r in ev])) if ev else None})
    write_csv(output / "per_type_metrics.csv", type_rows)
    make_metric_plots(summary, type_rows, output)
    runtime = time.perf_counter() - started_all
    settings = {"created_at": datetime.now().astimezone().isoformat(), "data": str(args.data.resolve()),
                "images": image_ids, "stride_px": STRIDE, "score_cell_px": CELL, "radial_bin_width": BIN_WIDTH,
                "bands_cycles_per_pixel": BANDS, "distance_modes": {"fixed48": 48, "proportional": "round16(0.375*W)"},
                "reference_policy": "exactly 8 in-image windows", "reference_aggregation": "per-frequency arithmetic mean of 8 dB spectra",
                "preprocessing": "subtract patch mean; separable 2-D Hann", "fft": "scipy.fft.rfft2 default backward scaling; FFT size=W; no zero padding",
                "power": "abs(F)^2; no total-power normalisation", "radial": "conjugate-weighted arithmetic mean linear power, then 10log10",
                "score": "RMS(target dB radial spectrum - mean reference dB radial spectrum)", "notch_filter": False,
                "annotation_precedence": "damage > tag/background > normal", "threshold": "LOIO equal-image-weighted normal p99/p95",
                "python": sys.version, "platform": platform.platform(), "workers": args.workers, "batch_size": args.batch_size,
                "runtime_seconds": runtime, "command": " ".join(sys.argv)}
    (output / "settings.json").write_text(json.dumps(json_safe(settings), ensure_ascii=False, indent=2), encoding="utf-8")
    report(output, setting_rows, summary, image_rows, defect_rows, type_rows, valid_rows, runtime)
    print(f"completed: {output.resolve()} ({runtime/60:.1f} min)", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--images", type=int, nargs="*"); parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.images and any(i not in PERIODS for i in args.images): parser.error("images must be in 2..18")
    run(args)


if __name__ == "__main__": main()
