"""Estimate projected-grid spacing from a histogram band and inspect local 2-D FFTs.

Example:
  python grid_period_fft.py data20260903/data20260903/image15.png --group-index 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from PIL import Image
from scipy.ndimage import (binary_erosion, convolve, find_objects, gaussian_filter,
                           label, map_coordinates, maximum_filter)

from colorize_three_histogram_groups import COLORS, analyze, tint


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs" / "grid_period_fft"
WINDOW_CHOICES = np.array([64, 96, 128, 160, 192, 224, 256, 288,
                           320, 384, 512, 640, 768, 896, 1024, 1152,
                           1280, 1536], dtype=int)


def histogram_groups(gray: np.ndarray):
    counts, smooth, ranges, peaks, valleys = analyze(gray)
    return counts, smooth, ranges, peaks, valleys


def autocorrelation_2d(source: np.ndarray):
    """Windowed, overlap-normalized linear autocorrelation with zero at center."""
    data = np.asarray(source, dtype=np.float64)
    h, w = data.shape
    if min(h, w) < 32:
        raise ValueError("Autocorrelation ROI must be at least 32x32 pixels")
    window = np.outer(np.hanning(h), np.hanning(w))
    mean = np.sum(data * window) / max(window.sum(), 1e-12)
    signal = (data - mean) * window
    pad = (2 * h, 2 * w)

    def corr(a):
        f = np.fft.rfft2(a, s=pad)
        return np.fft.fftshift(np.fft.irfft2(f * np.conj(f), s=pad))

    raw_corr = corr(signal)
    overlap = corr(window)
    ac = raw_corr / np.maximum(overlap, overlap.max() * 1e-7)
    cy, cx = h, w
    ac /= max(float(ac[cy, cx]), 1e-12)
    return ac, (cx, cy)


def _sample_near(ac, cx, cy, dx, dy, radius=2):
    x, y = cx + int(round(dx)), cy + int(round(dy))
    patch = ac[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1]
    return float(np.max(patch)) if patch.size else float("-inf")


def estimate_spacing(mask: np.ndarray, min_strength: float = 0.045,
                     min_distance: float = 24.0, min_harmonics: int = 2):
    """Find credible symmetric off-center autocorrelation peaks.

    Symmetric +/- lags are represented once. ``chosen`` remains the nearest
    credible lag for compatibility; use ``select_grid_basis`` to choose two
    independent grid translations when available.
    """
    ac, (cx, cy) = autocorrelation_2d(mask.astype(float))
    h, w = mask.shape
    yy, xx = np.indices(ac.shape)
    dx, dy = xx - cx, yy - cy
    distance = np.hypot(dx, dy)
    max_radius = min(h, w) * 0.45
    # Suppress the broad zero-lag peak; only search a bounded neighborhood.
    eligible = (distance >= min_distance) & (distance <= max_radius)
    maxima = (ac == maximum_filter(ac, size=5, mode="constant", cval=-np.inf)) & eligible
    ys, xs = np.where(maxima)
    candidates = []
    for py, px in zip(ys, xs):
        vx, vy = int(px - cx), int(py - cy)
        # A symmetric pair (+v, -v) is one translation, not two candidates.
        if vy < 0 or (vy == 0 and vx < 0):
            continue
        d = float(np.hypot(vx, vy))
        value = float(ac[py, px])
        opposite = _sample_near(ac, cx, cy, -vx, -vy, 2)
        symmetry = min(value, opposite) / max(value, opposite, 1e-12)
        if value < min_strength or opposite < min_strength * 0.75 or symmetry < 0.55:
            continue
        # Check whether peaks repeat at integer multiples of this lag.
        harmonic_values = [_sample_near(ac, cx, cy, k * vx, k * vy, 2)
                           for k in (2, 3) if np.hypot(k * vx, k * vy) <= max_radius]
        harmonic_support = sum(v >= max(0.025, value * 0.28) for v in harmonic_values)
        required_harmonics = min(min_harmonics, len(harmonic_values))
        if required_harmonics == 0 or harmonic_support < required_harmonics:
            continue
        # Rank by peak strength, symmetric support, and repeated-lag evidence;
        # distance breaks ties in favor of the closest credible peak.
        score = value * (0.65 + 0.35 * symmetry) * (1 + 0.12 * harmonic_support)
        candidates.append({"dx": vx, "dy": vy, "distance": d,
                           "angle_deg": float(np.degrees(np.arctan2(vy, vx))),
                           "strength": value, "opposite_strength": opposite,
                           "symmetry": float(symmetry), "harmonic_support": int(harmonic_support),
                           "score": float(score)})
    if not candidates:
        return ac, (cx, cy), None, []
    # Among robust peaks, prefer the nearest vector; for near-equal lags use
    # strength/support to avoid selecting a weak sidelobe.
    candidates.sort(key=lambda p: (round(p["distance"], 1), -p["score"]))
    chosen = candidates[0]
    return ac, (cx, cy), chosen, candidates[:60]


def select_grid_basis(candidates, min_sine_angle: float = 0.45,
                      length_ratio_range=(0.45, 2.2)):
    """Select two short independent translation vectors from autocorrelation peaks."""
    if not candidates:
        return []
    first = candidates[0]
    a = np.array([first["dx"], first["dy"]], dtype=float)
    a_norm = np.linalg.norm(a)
    if a_norm <= 1e-9:
        return []
    for point in candidates[1:]:
        b = np.array([point["dx"], point["dy"]], dtype=float)
        b_norm = np.linalg.norm(b)
        if b_norm <= 1e-9:
            continue
        sine = abs(a[0] * b[1] - a[1] * b[0]) / (a_norm * b_norm)
        ratio = b_norm / a_norm
        if sine >= min_sine_angle and length_ratio_range[0] <= ratio <= length_ratio_range[1]:
            return [first, point]
    return [first]


def reciprocal_basis_from_lags(basis):
    """Convert two spatial translations to reciprocal vectors in cycles/pixel."""
    if len(basis) < 2:
        return []
    a = np.array([basis[0]["dx"], basis[0]["dy"]], dtype=float)
    b = np.array([basis[1]["dx"], basis[1]["dy"]], dtype=float)
    direct = np.column_stack([a, b])
    if abs(float(np.linalg.det(direct))) < 1e-9:
        return []
    reciprocal = np.linalg.inv(direct).T
    return [reciprocal[:, 0].tolist(), reciprocal[:, 1].tolist()]


def detect_white_centers(mask, offset_xy=(0, 0), min_area=8, max_area=3000,
                         min_circularity=0.18):
    """Measure connected bright-mask components and retain round compact blobs."""
    structure = np.ones((3, 3), dtype=bool)
    labels, count = label(np.asarray(mask, dtype=bool), structure=structure)
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    yy_all, xx_all = np.indices(labels.shape)
    sum_x = np.bincount(labels.ravel(), weights=xx_all.ravel(), minlength=count + 1)
    sum_y = np.bincount(labels.ravel(), weights=yy_all.ravel(), minlength=count + 1)
    slices = find_objects(labels, max_label=count)
    ox, oy = offset_xy
    candidates, all_components = [], []
    for component_id in range(1, count + 1):
        area = int(areas[component_id])
        if not area:
            continue
        component_slice = slices[component_id - 1]
        y0, y1 = component_slice[0].start, component_slice[0].stop
        x0, x1 = component_slice[1].start, component_slice[1].stop
        region = labels[y0:y1, x0:x1] == component_id
        perimeter = int(np.count_nonzero(region & ~binary_erosion(region, structure=structure, border_value=0)))
        circularity = float(4 * np.pi * area / max(perimeter * perimeter, 1))
        item = {"component_id": component_id, "area_px": area,
                "center_xy": [float(sum_x[component_id] / area + ox),
                              float(sum_y[component_id] / area + oy)],
                "bbox_xywh": [x0 + ox, y0 + oy, x1 - x0, y1 - y0],
                "circularity": circularity,
                "equivalent_diameter_px": float(2 * np.sqrt(area / np.pi))}
        all_components.append(item)
        if min_area <= area <= max_area and circularity >= min_circularity:
            candidates.append(item)
    return candidates, all_components


def optimize_grid_phase(centers_xy, basis, roi_origin_xy=(0, 0), coarse_step=.04):
    """Fit origin phase over one lattice cell, testing both checker parities."""
    if len(basis) < 2 or len(centers_xy) < 1:
        return None
    a = np.array([basis[0]["dx"], basis[0]["dy"]], dtype=float)
    b = np.array([basis[1]["dx"], basis[1]["dy"]], dtype=float)
    matrix = np.column_stack([a, b])
    if abs(float(np.linalg.det(matrix))) < 1e-8:
        return None
    centers = np.asarray(centers_xy, dtype=float) - np.asarray(roi_origin_xy, dtype=float)
    inv = np.linalg.inv(matrix)
    offsets = np.array([(di, dj) for di in range(-2, 3) for dj in range(-2, 3)], dtype=int)

    def score(alpha, beta, parity):
        origin = alpha * a + beta * b
        uv = (centers - origin) @ inv.T
        base = np.rint(uv).astype(int)
        ij = base[:, None, :] + offsets[None, :, :]
        valid = ((ij[:, :, 0] + ij[:, :, 1]) % 2) == parity
        points = origin + ij[:, :, 0, None] * a + ij[:, :, 1, None] * b
        all_distances = np.linalg.norm(centers[:, None, :] - points, axis=2)
        all_distances[~valid] = np.inf
        nearest_ids = np.argmin(all_distances, axis=1)
        distances = all_distances[np.arange(len(centers)), nearest_ids]
        nearest = points[np.arange(len(centers)), nearest_ids]
        # Huber loss limits the influence of non-grid circular components.
        delta = 12.0
        loss = np.where(distances <= delta, .5 * distances**2,
                        delta * (distances - .5 * delta))
        return float(np.mean(loss)), distances, nearest, origin

    best = None
    for parity in (0, 1):
        step = coarse_step
        parity_best = None
        for _ in range(3):
            span = np.arange(-step, step + 1e-9, step / 4)
            alphas = (np.arange(0, 1, step) if parity_best is None else
                      (parity_best["alpha"] + span) % 1)
            betas = (np.arange(0, 1, step) if parity_best is None else
                     (parity_best["beta"] + span) % 1)
            refined = None
            for alpha in alphas:
                for beta in betas:
                    value, distances, nearest, origin = score(float(alpha), float(beta), parity)
                    if refined is None or value < refined["loss"]:
                        refined = {"loss": value, "alpha": float(alpha), "beta": float(beta),
                                   "parity": parity, "distances": distances,
                                   "nearest": nearest, "origin_local": origin}
            parity_best = refined
            if best is None or parity_best["loss"] < best["loss"]:
                best = parity_best
            step /= 4
    best["origin_xy"] = (best["origin_local"] + np.asarray(roi_origin_xy)).tolist()
    return best


def optimize_grid_phase_by_local_contrast(gray, basis, coarse_step=.05):
    """Optimize lattice translation using center-versus-ring image contrast.

    The autocorrelation basis is fixed. Only the phase coefficients alpha and
    beta in t = alpha*a + beta*b are searched. For each lattice site, a
    circular center mean minus an annular mean is sampled from a precomputed
    local-contrast image. The parity with the strongest positive contrast is
    selected as the white-side sublattice.
    """
    if len(basis) < 2:
        return None
    a = np.array([basis[0]["dx"], basis[0]["dy"]], dtype=float)
    b = np.array([basis[1]["dx"], basis[1]["dy"]], dtype=float)
    matrix = np.column_stack([a, b])
    if abs(float(np.linalg.det(matrix))) < 1e-8:
        return None

    image = np.asarray(gray, dtype=np.float32)
    h, w = image.shape
    pitch = max(8.0, min(float(np.linalg.norm(a)), float(np.linalg.norm(b))))
    inner_radius = max(2, int(round(pitch * .12)))
    ring_inner = max(inner_radius + 1, int(round(pitch * .20)))
    ring_outer = max(ring_inner + 1, int(round(pitch * .36)))
    yy, xx = np.mgrid[-ring_outer:ring_outer + 1, -ring_outer:ring_outer + 1]
    radius2 = xx * xx + yy * yy
    center_kernel = (radius2 <= inner_radius ** 2).astype(np.float32)
    ring_kernel = ((radius2 >= ring_inner ** 2) & (radius2 <= ring_outer ** 2)).astype(np.float32)
    center_kernel /= max(float(center_kernel.sum()), 1.0)
    ring_kernel /= max(float(ring_kernel.sum()), 1.0)
    contrast_map = convolve(image, center_kernel, mode="reflect") - convolve(image, ring_kernel, mode="reflect")

    # Include enough lattice indices to cover the image for any phase in [0,1)^2.
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=float)
    uv = corners @ np.linalg.inv(matrix).T
    imin, jmin = np.floor(uv.min(axis=0)).astype(int) - 2
    imax, jmax = np.ceil(uv.max(axis=0)).astype(int) + 2
    ii, jj = np.meshgrid(np.arange(imin, imax + 1), np.arange(jmin, jmax + 1), indexing="ij")
    ij = np.column_stack([ii.ravel(), jj.ravel()])
    parity_ids = ((ij[:, 0] + ij[:, 1]) % 2).astype(int)
    margin = ring_outer + 1

    def evaluate(alpha, beta):
        origin = alpha * a + beta * b
        points = origin + ij[:, 0, None] * a + ij[:, 1, None] * b
        valid = ((points[:, 0] >= margin) & (points[:, 0] < w - margin) &
                 (points[:, 1] >= margin) & (points[:, 1] < h - margin))
        values = map_coordinates(contrast_map, [points[:, 1], points[:, 0]],
                                 order=1, mode="nearest", prefilter=False)
        scores, counts = [], []
        for parity in (0, 1):
            selected = valid & (parity_ids == parity)
            counts.append(int(selected.sum()))
            scores.append(float(np.mean(values[selected])) if selected.any() else float("-inf"))
        return scores, counts, points, valid, origin

    initial_scores, initial_counts, _, _, _ = evaluate(0.0, 0.0)
    best = None
    step = coarse_step
    alpha_values = np.arange(0.0, 1.0, coarse_step)
    beta_values = np.arange(0.0, 1.0, coarse_step)
    coarse_score_map = np.full((len(beta_values), len(alpha_values)), np.nan, dtype=float)
    for level in range(3):
        if level:
            offsets = np.arange(-5, 6) * step
            alpha_values = (best["alpha"] + offsets) % 1.0
            beta_values = (best["beta"] + offsets) % 1.0
        level_best = best
        for ai, alpha in enumerate(alpha_values):
            for bi, beta in enumerate(beta_values):
                scores, counts, points, valid, origin = evaluate(float(alpha), float(beta))
                parity = int(np.argmax(scores))
                score = scores[parity]
                if level == 0:
                    coarse_score_map[bi, ai] = score
                if level_best is None or score > level_best["score"]:
                    level_best = {"alpha": float(alpha), "beta": float(beta), "score": score,
                                  "parity": parity, "parity_scores": scores,
                                  "counts": counts, "points": points, "valid": valid,
                                  "origin_local": origin}
        best = level_best
        step /= 5.0

    best["origin_xy"] = best["origin_local"].tolist()
    best["initial_parity_scores"] = initial_scores
    best["initial_parity_counts"] = initial_counts
    best["inner_radius_px"] = inner_radius
    best["ring_inner_radius_px"] = ring_inner
    best["ring_outer_radius_px"] = ring_outer
    best["improvement_over_initial"] = float(best["score"] - max(initial_scores))
    best["contrast_map"] = contrast_map
    best["search_score_map"] = coarse_score_map
    return best


def save_local_contrast_phase_figures(gray, basis, phase, out, offset_xy=(0, 0)):
    """Save source, lattice overlays, and a visual summary of phase search."""
    image = np.asarray(gray)
    h, w = image.shape
    a = np.array([basis[0]["dx"], basis[0]["dy"]], dtype=float)
    b = np.array([basis[1]["dx"], basis[1]["dy"]], dtype=float)
    points = phase["points"]
    valid = phase["valid"]
    # Recreate lattice integer indices for coloring the two checker sublattices.
    matrix = np.column_stack([a, b])
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=float)
    uv = corners @ np.linalg.inv(matrix).T
    imin, jmin = np.floor(uv.min(axis=0)).astype(int) - 2
    imax, jmax = np.ceil(uv.max(axis=0)).astype(int) + 2
    ii, jj = np.meshgrid(np.arange(imin, imax + 1), np.arange(jmin, jmax + 1), indexing="ij")
    parity_ids = ((ii.ravel() + jj.ravel()) % 2).astype(int)
    origin_local = np.asarray(phase["origin_xy"], dtype=float)
    fitted_local = origin_local + ii.ravel()[:, None] * a + jj.ravel()[:, None] * b
    margin = phase["ring_outer_radius_px"] + 1
    inside = ((fitted_local[:, 0] >= margin) & (fitted_local[:, 0] < w - margin) &
              (fitted_local[:, 1] >= margin) & (fitted_local[:, 1] < h - margin))
    ox, oy = offset_xy
    origin = origin_local + np.array([ox, oy])
    fitted = fitted_local + np.array([ox, oy])
    white = fitted[inside & (parity_ids == phase["parity"])]
    black = fitted[inside & (parity_ids != phase["parity"])]

    Image.fromarray(image).save(out / "00_original_grayscale.png")

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    if len(white): ax.scatter(white[:, 0], white[:, 1], marker="+", c="lime", s=45, label="selected white-side centers")
    if len(black): ax.scatter(black[:, 0], black[:, 1], marker="x", c="deepskyblue", s=25, label="other checker side")
    ax.scatter([origin[0]], [origin[1]], marker="*", c="yellow", edgecolors="black", s=170, label="phase origin")
    ax.quiver(*origin, *a, angles="xy", scale_units="xy", scale=1, color="orange", label="a")
    ax.quiver(*origin, *b, angles="xy", scale_units="xy", scale=1, color="cyan", label="b")
    ax.set(xlim=(0, w), ylim=(h, 0), title="Estimated grid centers over original grayscale")
    ax.legend(loc="upper right")
    ax.text(.015, .015,
            f"a=({a[0]:.1f},{a[1]:.1f}) px   b=({b[0]:.1f},{b[1]:.1f}) px\n"
            f"score before={max(phase['initial_parity_scores']):.3f}   after={phase['score']:.3f}   "
            f"white parity={(phase['parity'])}", transform=ax.transAxes, color="white",
            bbox={"facecolor": "black", "alpha": .72, "pad": 4})
    fig.savefig(out / "07_optimal_grid_overlay.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    if len(white): ax.scatter(white[:, 0], white[:, 1], marker="+", c="#00ff50", s=48, label="white-side sublattice")
    if len(black): ax.scatter(black[:, 0], black[:, 1], marker="x", c="#28aaff", s=28, label="black-side sublattice")
    ax.set(xlim=(0, w), ylim=(h, 0), title="Checker sublattices: parity-colored predicted centers")
    ax.legend(loc="upper right")
    fig.savefig(out / "08_checkerboard_grid_overlay.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    scores = np.asarray(phase["search_score_map"], dtype=float)
    im = ax.imshow(scores, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="viridis")
    ax.scatter([phase["alpha"]], [phase["beta"]], c="red", marker="*", s=150,
               edgecolors="white", label="selected phase")
    ax.set(xlabel="alpha in t = alpha*a + beta*b", ylabel="beta", title="Best checker-parity local-contrast score over phase")
    ax.legend()
    fig.colorbar(im, ax=ax, label="Mean center-minus-ring contrast [gray levels]")
    fig.savefig(out / "09_phase_score_map.png", dpi=160)
    plt.close(fig)

    details = {k: v for k, v in phase.items()
               if k not in {"contrast_map", "points", "valid", "search_score_map", "origin_local"}}
    details["grid_vector_a"] = a.tolist()
    details["grid_vector_b"] = b.tolist()
    details["grid_origin_xy"] = phase["origin_xy"]
    details["number_of_predicted_centers"] = int(np.count_nonzero(phase["valid"]))
    details["white_side_center_count"] = int(len(white))
    details["black_side_center_count"] = int(len(black))
    (out / "phase_optimization.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")


def save_phase_figures(gray, mask, candidates, basis, phase, roi_xywh, out):
    """Save candidate, ideal-grid, overlay, and alignment-error visualizations."""
    rx, ry, rw, rh = roi_xywh
    a = np.array([basis[0]["dx"], basis[0]["dy"]], dtype=float)
    b = np.array([basis[1]["dx"], basis[1]["dy"]], dtype=float)
    origin = np.asarray(phase["origin_xy"])
    white, black = [], []
    corners = np.array([[rx, ry], [rx+rw, ry], [rx, ry+rh], [rx+rw, ry+rh]])
    uv = (corners - origin) @ np.linalg.inv(np.column_stack([a, b])).T
    imin, jmin = np.floor(uv.min(axis=0)).astype(int) - 2
    imax, jmax = np.ceil(uv.max(axis=0)).astype(int) + 2
    for i in range(imin, imax + 1):
        for j in range(jmin, jmax + 1):
            point = origin + i*a + j*b
            if rx <= point[0] < rx+rw and ry <= point[1] < ry+rh:
                (white if (i+j) % 2 == phase["parity"] else black).append(point)
    white = np.asarray(white).reshape(-1, 2)
    black = np.asarray(black).reshape(-1, 2)
    centers = np.array([c["center_xy"] for c in candidates], dtype=float).reshape(-1, 2)

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    for c in candidates:
        x, y, w, h = c["bbox_xywh"]
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=.8))
    if len(centers): ax.scatter(centers[:,0], centers[:,1], c="red", s=22, label="white-component centers")
    ax.set(xlim=(rx, rx+rw), ylim=(ry+rh, ry)); ax.legend(loc="upper right")
    ax.set_title(f"1. Detected white candidates: {len(candidates)}")
    fig.savefig(out / "06_white_center_candidates.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    if len(white): ax.scatter(white[:,0], white[:,1], marker="+", c="lime", s=48, label="white-side grid")
    if len(black): ax.scatter(black[:,0], black[:,1], c="dodgerblue", s=10, label="other checker side")
    ax.scatter([origin[0]], [origin[1]], marker="*", c="yellow", edgecolors="black", s=200, label="origin")
    ax.quiver(*origin, *a, angles="xy", scale_units="xy", scale=1, color="orange", label="a")
    ax.quiver(*origin, *b, angles="xy", scale_units="xy", scale=1, color="cyan", label="b")
    ax.set(xlim=(rx,rx+rw), ylim=(ry+rh,ry), aspect="equal", title="2. Estimated checker grid")
    ax.legend(loc="upper right"); fig.savefig(out / "07_ideal_grid.png", dpi=160); plt.close(fig)

    for filename, draw_errors in (("08_grid_overlay.png", False), ("09_alignment_errors.png", True)):
        fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
        ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
        if len(white): ax.scatter(white[:,0], white[:,1], marker="+", c="lime", s=48, label="white-side grid")
        if len(black): ax.scatter(black[:,0], black[:,1], c="dodgerblue", s=10, label="other checker side")
        if len(centers):
            ax.scatter(centers[:,0], centers[:,1], c="red", s=20, label="white centers")
            if draw_errors:
                for c, p in zip(centers, phase["nearest"] + np.asarray([rx, ry])):
                    ax.plot([c[0],p[0]],[c[1],p[1]], color="yellow", linewidth=.7)
        ax.scatter([origin[0]], [origin[1]], marker="*", c="yellow", edgecolors="black", s=200, label="origin")
        ax.set(xlim=(rx,rx+rw), ylim=(ry+rh,ry), title=("4. Center-to-grid alignment errors" if draw_errors else "3. Original + estimated grid"))
        if draw_errors:
            ax.text(.02,.02, f"mean={np.mean(phase['distances']):.2f}px  median={np.median(phase['distances']):.2f}px", transform=ax.transAxes, color="white", bbox={"facecolor":"black","alpha":.7})
        ax.legend(loc="upper right"); fig.savefig(out / filename, dpi=160); plt.close(fig)


def choose_window(spacing: float, periods_per_window: float, max_size: int | None = None):
    raw = spacing * periods_per_window
    choices = WINDOW_CHOICES[WINDOW_CHOICES <= max_size] if max_size else WINDOW_CHOICES
    if not len(choices):
        choices = np.array([max_size], dtype=int)
    selected = int(choices[np.argmin(np.abs(choices - raw))])
    smaller = int(choices[np.argmin(np.abs(choices - selected * 0.5))])
    larger = int(choices[np.argmin(np.abs(choices - selected * 1.5))])
    return raw, selected, sorted(set([smaller, selected, larger]))


def crop_square(image: np.ndarray, center_xy: tuple[int, int], size: int):
    h, w = image.shape
    cx, cy = center_xy
    size = min(size, h, w)
    x0 = int(np.clip(cx - size // 2, 0, w - size))
    y0 = int(np.clip(cy - size // 2, 0, h - size))
    return image[y0:y0+size, x0:x0+size], (x0, y0)


def fft_products(gray_roi: np.ndarray):
    source = gray_roi.astype(np.float64)
    dc_removed = source - source.mean()
    hann = np.outer(np.hanning(source.shape[0]), np.hanning(source.shape[1]))
    windowed = dc_removed * hann
    spectrum = np.fft.fftshift(np.fft.fft2(windowed))
    magnitude = np.abs(spectrum)
    log_magnitude = np.log1p(magnitude)
    frequencies = np.fft.fftshift(np.fft.fftfreq(source.shape[0]))
    return windowed, magnitude, log_magnitude, frequencies


def save_summary_figure(gray, colorized, candidate_mask, out, roi_xywh):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original grayscale")
    axes[1].imshow(colorized)
    axes[1].set_title("Histogram peak-band classification")
    axes[2].imshow(candidate_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Grid candidate binary mask")
    x, y, w, h = roi_xywh
    for ax in (axes[0], axes[1], axes[2]):
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#ffdf00", linewidth=2))
    axes[0].text(8, 24, "yellow box = analysis ROI", color="#ffdf00", fontsize=9,
                 bbox={"facecolor": "black", "alpha": .65, "pad": 3})
    for ax in axes:
        ax.axis("off")
    fig.savefig(out / "01_input_classification_mask.png", dpi=150)
    plt.close(fig)


def save_autocorrelation(ac, center, estimate, candidates, basis, out,
                         top_n=12, name="02_autocorrelation.png"):
    cx, cy = center
    h, w = ac.shape
    fig, ax = plt.subplots(figsize=(10, 8.5), constrained_layout=True)
    im = ax.imshow(ac, cmap="coolwarm", origin="lower",
                   extent=[-cx, w-cx-1, -cy, h-cy-1], vmin=-0.25, vmax=1)
    ax.scatter([0], [0], color="yellow", s=90, marker="x", linewidths=2, label="zero-lag center")
    shown = candidates[:top_n]
    for rank, point in enumerate(shown, start=1):
        dx, dy = point["dx"], point["dy"]
        ax.scatter([dx, -dx], [dy, -dy], s=28, facecolors="none",
                   edgecolors="white", linewidths=.9, alpha=.9)
        # Keep all candidate markers, but label only the strongest few so
        # nearby peaks do not turn the autocorrelation plot into a text blob.
        if rank <= min(6, top_n):
            ax.text(dx + 5, dy + 5, f"{rank}: {point['distance']:.1f}",
                    color="white", fontsize=7,
                    bbox={"facecolor": "black", "alpha": .55, "pad": 1.5})

    basis_colors = ["#ffe600", "#00e5ff"]
    basis_labels = []
    for i, point in enumerate(basis[:2]):
        dx, dy = point["dx"], point["dy"]
        color = basis_colors[i]
        ax.plot([0, dx], [0, dy], color=color, linewidth=2.3)
        ax.scatter([dx, -dx], [dy, -dy], color=color, edgecolor="black", s=82, zorder=5)
        basis_labels.append(f"basis {i+1}: ({dx:+d},{dy:+d}) px, |v|={point['distance']:.1f}px")
    if basis_labels:
        ax.text(.02, .98, "\n".join(basis_labels), transform=ax.transAxes,
                va="top", ha="left", fontsize=8.5, color="black",
                bbox={"facecolor": "white", "alpha": .82, "pad": 5})
    if estimate:
        title_extra = f"; nearest credible lag = {estimate['distance']:.1f} px"
    else:
        title_extra = ""
    ax.set(xlabel="Horizontal shift dx [px]", ylabel="Vertical shift dy [px]",
           title=f"2-D autocorrelation: periodic-peak candidates{title_extra}")
    if shown:
        max_candidate = max(max(abs(p["dx"]), abs(p["dy"])) for p in shown)
        view_limit = min(max(cx, cy), max(160, max_candidate * 1.35))
    else:
        view_limit = max(160, min(cx, cy) * 0.23)
    ax.set_xlim(-view_limit, view_limit)
    ax.set_ylim(-view_limit, view_limit)
    fig.colorbar(im, ax=ax, label="Normalized autocorrelation")
    fig.savefig(out / name, dpi=160)
    plt.close(fig)


def save_fft_window(gray, candidate_mask, center_xy, size, representative_period,
                    basis, reciprocal_basis, out):
    roi, top_left = crop_square(gray, center_xy, size)
    mask_roi, _ = crop_square(candidate_mask.astype(float), center_xy, size)
    windowed, magnitude, log_mag, freq = fft_products(roi)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.8), constrained_layout=True)
    axes[0].imshow(roi, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[0].set_title(f"Original ROI ({size}px)")
    axes[1].imshow(mask_roi, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("Candidate mask ROI")
    lim = max(float(np.max(np.abs(windowed))), 1e-9)
    axes[2].imshow(windowed, cmap="gray", vmin=-lim, vmax=lim, interpolation="nearest")
    axes[2].set_title("Mean-subtracted + Hann")
    axes[3].imshow(magnitude, cmap="magma", origin="lower")
    axes[3].set_title("FFT magnitude")
    extent = [freq[0], freq[-1], freq[0], freq[-1]]
    axes[4].imshow(log_mag, cmap="magma", origin="lower", extent=extent)
    reciprocal_norms = []
    reciprocal_colors = ["#ffffff", "#40ffff"]
    for i, vector in enumerate(reciprocal_basis[:2]):
        gx, gy = vector
        reciprocal_norms.append(float(np.hypot(gx, gy)))
        axes[4].scatter([gx, -gx], [gy, -gy], marker="x", s=62,
                        color=reciprocal_colors[i], linewidths=1.6,
                        label=f"reciprocal basis {i+1}")
    f_reference = 1.0 / representative_period
    axes[4].add_patch(Circle((0, 0), f_reference, fill=False, color="#ffe066",
                             linestyle="--", linewidth=1.2,
                             label=f"rough 1/median(|a|,|b|)={f_reference:.4f}"))
    max_f = max(reciprocal_norms + [f_reference])
    lim_f = min(.25, max(.02, max_f * 3.0))
    axes[4].set_xlim(-lim_f, lim_f)
    axes[4].set_ylim(-lim_f, lim_f)
    axes[4].set_title("fftshift log(1+|FFT|)")
    axes[4].set_xlabel("fx [cycles/pixel]")
    axes[4].set_ylabel("fy [cycles/pixel]")
    axes[4].legend(fontsize=7, loc="upper right")
    for ax in axes[:4]:
        ax.axis("off")
    filename = out / f"fft_window_{size}px.png"
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    # Save the required intermediate images individually as well as in the panel.
    Image.fromarray(roi).save(out / f"roi_{size}px.png")
    Image.fromarray((mask_roi * 255).astype(np.uint8)).save(out / f"candidate_mask_roi_{size}px.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].imshow(windowed, cmap="gray")
    axes[0].set_title("Mean-subtracted grayscale x Hann")
    axes[1].imshow(log_mag, cmap="magma", origin="lower")
    axes[1].set_title("fftshift log(1 + magnitude)")
    for ax in axes:
        ax.axis("off")
    fig.savefig(out / f"hann_and_logfft_{size}px.png", dpi=150)
    plt.close(fig)
    mask_windowed, mask_magnitude, mask_log_mag, _ = fft_products(mask_roi)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].imshow(mask_roi, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("Binary candidate mask")
    axes[1].imshow(mask_windowed, cmap="gray")
    axes[1].set_title("Mask x Hann")
    axes[2].imshow(mask_log_mag, cmap="magma", origin="lower")
    axes[2].set_title("Mask FFT: fftshift log(1 + magnitude)")
    for ax in axes:
        ax.axis("off")
    fig.savefig(out / f"mask_fft_{size}px.png", dpi=150)
    plt.close(fig)
    return {"window_px": size, "top_left_xy": list(top_left),
            "roi_center_xy": [top_left[0] + size//2, top_left[1] + size//2],
            "representative_period_px": float(representative_period),
            "reciprocal_basis_cycles_per_pixel": reciprocal_basis,
            "magnitude_max": float(magnitude.max()), "file": filename.name}


def compare_local_regions(mask, out, labels):
    h, w = mask.shape
    size = max(96, int(min(h, w) * 0.80))
    size = min(size, h, w)
    centers = {
        "top_left": (int(w * .27), int(h * .27)),
        "center": (w // 2, h // 2),
        "bottom_right": (int(w * .73), int(h * .73)),
    }
    reports = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, (label, center) in zip(axes, centers.items()):
        region, top = crop_square(mask.astype(float), center, size)
        ac, ac_center, estimate, _ = estimate_spacing(region > .5)
        ax.imshow(region, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        spacing = estimate["distance"] if estimate else None
        reports[label] = {"center_xy": list(center), "top_left_xy": list(top),
                          "roi_size_px": size, "estimated_grid_spacing_px": spacing,
                          "peak": estimate}
        ax.set_title(f"{label}: {spacing:.1f} px" if spacing else f"{label}: no stable peak")
        ax.axis("off")
    fig.savefig(out / "04_local_region_masks.png", dpi=150)
    plt.close(fig)
    (out / "local_periods.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--group-index", type=int, default=2,
                        help="1-based ascending brightness peak band to use as mask (default: 2)")
    parser.add_argument("--periods-per-window", type=float, default=6.0,
                        help="Desired number of grid periods in each FFT window")
    parser.add_argument("--fft-center", type=int, nargs=2, metavar=("X", "Y"), default=None,
                        help="FFT ROI center in source-image pixels; default is image center")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"), default=None,
                        help="Optional analysis ROI in source pixels: x y width height")
    parser.add_argument("--autocorr-source", choices=("mask", "gray"), default="mask",
                        help="Use candidate binary mask (default) or detrended grayscale")
    parser.add_argument("--min-peak-distance", type=float, default=24.0,
                        help="Ignore shorter autocorrelation lags (default: 24 px)")
    parser.add_argument("--min-peak-strength", type=float, default=0.045,
                        help="Minimum normalized autocorrelation strength for a periodic peak")
    parser.add_argument("--top-autocorr-peaks", type=int, default=12,
                        help="Number of credible autocorrelation peak pairs to label")
    parser.add_argument("--basis-min-sine", type=float, default=0.45,
                        help="Minimum sine of angle between selected grid translation vectors")
    parser.add_argument("--white-min-area", type=int, default=8,
                        help="Minimum connected-component area for white center candidates")
    parser.add_argument("--white-max-area", type=int, default=3000,
                        help="Maximum connected-component area; rejects large labels/background")
    parser.add_argument("--white-min-circularity", type=float, default=0.18,
                        help="Minimum component circularity 4*pi*area/perimeter^2")
    parser.add_argument("--center-group-index", type=int, default=None,
                        help="Brightness band for white-center detection; defaults to --group-index")
    args = parser.parse_args()
    source = args.image.resolve()
    if not source.exists():
        parser.error(f"Image not found: {source}")
    out = args.output.resolve() if args.output else OUTPUT_ROOT / source.stem
    out.mkdir(parents=True, exist_ok=True)

    gray = np.asarray(Image.open(source).convert("L"), dtype=np.uint8)
    counts, smooth, ranges, peaks, valleys = histogram_groups(gray)
    # For a one-mode trimmed histogram, use its sole band even when the
    # backward-compatible default group index (2) was requested.
    group_index = 1 if len(ranges) == 1 and args.group_index == 2 else args.group_index
    if not 1 <= group_index <= len(ranges):
        parser.error(f"group-index must be between 1 and {len(ranges)} for this image")
    colorized, group_masks = tint(gray, ranges)
    candidate_mask = group_masks[group_index - 1]
    if min(gray.shape) < 96:
        parser.error("Input image must be at least 96 pixels in both dimensions")
    Image.fromarray(gray).save(out / "original_grayscale.png")
    Image.fromarray(colorized).save(out / "histogram_classification.png")
    Image.fromarray(candidate_mask.astype(np.uint8) * 255).save(out / "grid_candidate_mask.png")
    if args.roi:
        rx, ry, rw, rh = args.roi
        if rx < 0 or ry < 0 or rw < 96 or rh < 96 or rx + rw > gray.shape[1] or ry + rh > gray.shape[0]:
            parser.error("--roi must be at least 96x96 and fit inside the image")
    else:
        rx, ry, rw, rh = 0, 0, gray.shape[1], gray.shape[0]
    save_summary_figure(gray, colorized, candidate_mask, out, (rx, ry, rw, rh))
    mask_roi = candidate_mask[ry:ry+rh, rx:rx+rw]
    gray_roi_for_ac = gray[ry:ry+rh, rx:rx+rw].astype(float)
    ac_source = (gray_roi_for_ac - gaussian_filter(gray_roi_for_ac, sigma=max(4, min(rw, rh) * .025))
                 if args.autocorr_source == "gray" else mask_roi)
    ac, ac_center, peak, candidates = estimate_spacing(
        ac_source, args.min_peak_strength, args.min_peak_distance, min_harmonics=2)
    basis = select_grid_basis(candidates, min_sine_angle=args.basis_min_sine)
    reciprocal_basis = reciprocal_basis_from_lags(basis)
    phase = optimize_grid_phase_by_local_contrast(gray_roi_for_ac, basis)
    if phase is not None:
        save_local_contrast_phase_figures(gray, basis, phase, out, (rx, ry))
    save_autocorrelation(ac, ac_center, peak, candidates, basis, out,
                         top_n=max(1, args.top_autocorr_peaks))
    region_reports = compare_local_regions(mask_roi, out, [])
    basis_lengths = [float(point["distance"]) for point in basis]
    representative_period = (float(np.median(basis_lengths)) if basis_lengths
                             else (float(peak["distance"]) if peak else None))
    report = {
        "source": str(source), "image_size_wh": [gray.shape[1], gray.shape[0]],
        "histogram_peak_levels": peaks, "histogram_peak_ranges": ranges,
        "selected_group_index_1_based": group_index,
        "requested_group_index_1_based": args.group_index,
        "selected_brightness_range": ranges[group_index - 1],
        "candidate_mask_fraction": float(candidate_mask.mean()),
        "analysis_roi_xywh": [rx, ry, rw, rh], "autocorrelation_source": args.autocorr_source,
        "periods_per_window_N": args.periods_per_window,
        "minimum_autocorrelation_peak_strength": args.min_peak_strength,
        "minimum_autocorrelation_lag_px": args.min_peak_distance,
        "autocorrelation_peak_candidates": candidates,
        "selected_grid_translation_basis": basis,
        "grid_phase_alignment": None if phase is None else {
            "method": "maximize mean center-disk minus surrounding-annulus grayscale contrast over lattice phase and checker parity",
            "grid_vector_a": [basis[0]["dx"], basis[0]["dy"]],
            "grid_vector_b": [basis[1]["dx"], basis[1]["dy"]],
            "grid_origin": (np.asarray(phase["origin_xy"]) + np.asarray([rx, ry])).tolist(),
            "selected_checker_parity": phase["parity"],
            "phase_score_before": max(phase["initial_parity_scores"]),
            "phase_score_before_by_parity": phase["initial_parity_scores"],
            "phase_score_after": phase["score"],
            "phase_score_after_by_parity": phase["parity_scores"],
            "phase_score_improvement": phase["improvement_over_initial"],
            "number_of_predicted_centers": int(np.count_nonzero(phase["valid"])),
            "phase_coefficients_alpha_beta": [phase["alpha"], phase["beta"]],
            "inner_disk_radius_px": phase["inner_radius_px"],
            "outer_ring_radii_px": [phase["ring_inner_radius_px"], phase["ring_outer_radius_px"]],
        },
        "reciprocal_basis_cycles_per_pixel": reciprocal_basis,
        "representative_translation_period_px": representative_period,
        "local_region_estimates": region_reports,
        "status": "period_peak_not_found" if peak is None else "needs_visual_review",
        "notes": ["Histogram peak band is used to build the binary candidate mask.",
                  "Autocorrelation can contain several valid grid translations; the nearest lag is not automatically one cell size.",
                  "Two independent short translation vectors are selected when possible.",
                  "Grid phase is fitted by maximizing local grayscale contrast (center disk minus outer ring); both checker parities are tested.",
                  "FFT uses original grayscale ROI (mean removed + Hann); binary-mask FFT is saved only for comparison."],
    }
    if peak is not None and representative_period is not None:
        raw_window, selected_window, comparison_windows = choose_window(
            representative_period, args.periods_per_window, min(gray.shape))
        center = tuple(args.fft_center) if args.fft_center else (rx + rw // 2, ry + rh // 2)
        fft_reports = [save_fft_window(gray, candidate_mask, center, size,
                                       representative_period, basis, reciprocal_basis, out)
                       for size in comparison_windows]
        report.update({
            # This representative translation length is not guaranteed to be
            # one checker-cell width for every projected lattice.
            "estimated_grid_spacing_px": representative_period,
            "selected_autocorrelation_peak": peak,
            "recommended_window_raw_px": raw_window,
            "selected_fft_window_px": selected_window,
            "window_selection_rule": "median selected translation-basis length x N, then nearest practical WINDOW_CHOICES value",
            "comparison_fft_windows_px": comparison_windows,
            "fft_center_xy": list(center), "fft_outputs": fft_reports,
            "status": "needs_visual_review",
        })
        fig, axes = plt.subplots(2, len(fft_reports), figsize=(6 * len(fft_reports), 9), constrained_layout=True)
        if len(fft_reports) == 1:
            axes = np.asarray(axes).reshape(2, 1)
        reciprocal_norms = [float(np.hypot(g[0], g[1])) for g in reciprocal_basis]
        rough_frequency = 1.0 / representative_period
        plot_limit = max(.02, min(.25, max([rough_frequency] + reciprocal_norms) * 3.0))
        for col, item in enumerate(fft_reports):
            n = item["window_px"]
            roi, _ = crop_square(gray, center, n)
            mask_patch, _ = crop_square(candidate_mask.astype(float), center, n)
            for row, patch in enumerate((roi.astype(float), mask_patch)):
                source_patch = patch - patch.mean()
                hann = np.outer(np.hanning(n), np.hanning(n))
                spectrum = np.fft.fftshift(np.fft.fft2(source_patch * hann))
                magnitude = np.abs(spectrum)
                db = 20 * np.log10(np.maximum(magnitude / max(magnitude.max(), 1e-12), 1e-6))
                f = np.fft.fftshift(np.fft.fftfreq(n))
                extent = [f[0], f[-1], f[0], f[-1]]
                ax = axes[row, col]
                ax.imshow(db, cmap="magma", origin="lower", extent=extent, vmin=-50, vmax=0)
                ax.add_patch(Circle((0, 0), rough_frequency, fill=False, color="#ffe066",
                                    linestyle="--", linewidth=1.1))
                reciprocal_colors = ["white", "#40ffff"]
                for i, vector in enumerate(reciprocal_basis[:2]):
                    gx, gy = vector
                    ax.scatter([gx, -gx], [gy, -gy], marker="x",
                               color=reciprocal_colors[i], s=50, linewidths=1.4)
                ax.set(xlim=(-plot_limit, plot_limit), ylim=(-plot_limit, plot_limit),
                       xlabel="fx [cycles/pixel]", ylabel="fy [cycles/pixel]")
                if row == 0:
                    ax.set_title(f"Grayscale {n}x{n}")
                else:
                    ax.set_title(f"Candidate mask {n}x{n}")
        fig.suptitle("FFT window comparison; dashed ring is a rough scalar reference, crosses are reciprocal-basis predictions")
        fig.savefig(out / "05_fft_size_comparison.png", dpi=160)
        plt.close(fig)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
