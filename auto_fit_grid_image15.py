"""Fit an autocorrelation-derived checker lattice to trimmed image15.

This is a visualization/validation experiment, not a defect detector. The
autocorrelation vectors remain fixed; only lattice phase and checker parity are
optimized. Outputs go under outputs/trimmed_grid_period_analysis/⑦_自動格子復元.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import convolve, gaussian_filter

from grid_period_fft import estimate_spacing, histogram_groups, select_grid_basis


ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = ROOT / "data20260903/data20260903/triming/image15_triming.png"
DEFAULT_OUT = ROOT / "outputs/trimmed_grid_period_analysis/⑦_自動格子復元/image15"


def lattice_indices(shape, a, b, margin=0):
    h, w = shape
    matrix = np.column_stack((a, b))
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=float)
    uv = corners @ np.linalg.inv(matrix).T
    lo = np.floor(uv.min(axis=0)).astype(int) - 2
    hi = np.ceil(uv.max(axis=0)).astype(int) + 2
    ii, jj = np.meshgrid(np.arange(lo[0], hi[0] + 1),
                         np.arange(lo[1], hi[1] + 1), indexing="ij")
    return np.column_stack((ii.ravel(), jj.ravel()))


def phase_search(gray, a, b, coarse=.04):
    h, w = gray.shape
    pitch = min(np.linalg.norm(a), np.linalg.norm(b))
    r0 = max(2, int(round(pitch * .12)))
    r1, r2 = max(r0 + 1, int(round(pitch * .20))), max(r0 + 2, int(round(pitch * .36)))
    yy, xx = np.mgrid[-r2:r2 + 1, -r2:r2 + 1]
    d2 = xx * xx + yy * yy
    disk = (d2 <= r0 * r0).astype(np.float32)
    ring = ((d2 >= r1 * r1) & (d2 <= r2 * r2)).astype(np.float32)
    disk /= disk.sum()
    ring /= ring.sum()
    contrast = convolve(gray.astype(np.float32), disk, mode="reflect") - convolve(
        gray.astype(np.float32), ring, mode="reflect")
    ij = lattice_indices(gray.shape, a, b)
    parity = ((ij[:, 0] + ij[:, 1]) & 1)
    margin = r2 + 1

    def evaluate(alpha, beta):
        origin = alpha * a + beta * b
        points = origin + ij[:, 0, None] * a + ij[:, 1, None] * b
        valid = ((points[:, 0] >= margin) & (points[:, 0] < w - margin) &
                 (points[:, 1] >= margin) & (points[:, 1] < h - margin))
        vals = contrast[np.clip(np.rint(points[:, 1]).astype(int), 0, h - 1),
                        np.clip(np.rint(points[:, 0]).astype(int), 0, w - 1)]
        means = [float(np.mean(vals[valid & (parity == p)])) for p in (0, 1)]
        # The sign selects which sublattice is brighter; magnitude measures
        # checker separation. Divide by robust image contrast for scale stability.
        score = abs(means[0] - means[1]) / (1.4826 * np.median(np.abs(contrast - np.median(contrast))) + 1e-6)
        return score, means, origin, points, valid

    before = evaluate(0.0, 0.0)
    best = None
    score_map = None
    step = coarse
    for level in range(3):
        if level == 0:
            av = np.arange(0, 1, step)
            bv = np.arange(0, 1, step)
            score_map = np.zeros((len(bv), len(av)), dtype=float)
        else:
            offsets = np.arange(-5, 6) * step
            av = (best["alpha"] + offsets) % 1
            bv = (best["beta"] + offsets) % 1
        for ai, alpha in enumerate(av):
            for bi, beta in enumerate(bv):
                res = evaluate(float(alpha), float(beta))
                if level == 0:
                    score_map[bi, ai] = res[0]
                if best is None or res[0] > best["score"]:
                    best = {"alpha": float(alpha), "beta": float(beta), "score": res[0],
                            "means": res[1], "origin": res[2], "points": res[3],
                            "valid": res[4]}
        step /= 5
    best["parity"] = int(np.argmax(best["means"]))
    best["before_score"] = before[0]
    best["before_means"] = before[1]
    best["score_map"] = score_map
    best["indices"] = ij
    best["parity_ids"] = parity
    best["contrast_map"] = contrast
    best["radii"] = [r0, r1, r2]
    return best


def save_lattice_image(gray, a, b, origin, indices, parity_ids, valid, parity, dest, mode):
    h, w = gray.shape
    pts = origin + indices[:, 0, None] * a + indices[:, 1, None] * b
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    # Draw each cell as a parallelogram centered on its lattice site.
    for k in np.flatnonzero(inside):
        x, y = pts[k]
        corners = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
        poly = np.array([x, y]) + corners[:, :1] * a + corners[:, 1:] * b
        color = "#00ff66" if parity_ids[k] == parity else "#20baff"
        ax.plot(*np.vstack((poly, poly[0])).T, color=color, lw=.65, alpha=.82)
        if mode == "indexed" and x > 25 and y > 25 and x < w - 25 and y < h - 25:
            ax.text(x, y, f"{indices[k,0]},{indices[k,1]}", color="white", fontsize=6,
                    ha="center", va="center", bbox={"facecolor": "black", "alpha": .55, "pad": 1})
    ax.scatter(pts[inside, 0], pts[inside, 1], s=12, c=np.where(parity_ids[inside] == parity, "lime", "deepskyblue"),
               marker="+")
    ax.scatter([origin[0]], [origin[1]], marker="*", s=150, c="yellow", edgecolors="black", zorder=5)
    ax.quiver(origin[0], origin[1], a[0], a[1], angles="xy", scale_units="xy", scale=1,
              color="orange", width=.004)
    ax.quiver(origin[0], origin[1], b[0], b[1], angles="xy", scale_units="xy", scale=1,
              color="cyan", width=.004)
    ax.set(xlim=(0, w), ylim=(h, 0), title={"initial": "① Initial lattice (origin=(0,0))",
        "optimized": "② Optimized origin and checker parity",
        "indexed": "③ Cell IDs (i,j), selected checker side in green"}[mode])
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    fig.savefig(dest, dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--group-index", type=int, default=2)
    ap.add_argument("--coarse-step", type=float, default=.04)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gray = np.asarray(Image.open(args.image).convert("L"), dtype=np.uint8)
    counts, smooth, ranges, peaks, valleys = histogram_groups(gray)
    group = 1 if len(ranges) == 1 and args.group_index == 2 else args.group_index
    if not 1 <= group <= len(ranges):
        raise SystemExit(f"group-index={group} outside detected bands: {ranges}")
    masks = np.array([(gray >= lo) & (gray <= hi) for lo, hi in ranges])
    mask = masks[group - 1]
    ac, center, peak, candidates = estimate_spacing(mask, min_strength=.045,
                                                    min_distance=24, min_harmonics=2)
    basis = select_grid_basis(candidates, min_sine_angle=.45)
    if len(basis) < 2:
        raise SystemExit("自己相関から独立な2本の格子ベクトルを選べませんでした。")
    a = np.array([basis[0]["dx"], basis[0]["dy"]], dtype=float)
    b = np.array([basis[1]["dx"], basis[1]["dy"]], dtype=float)
    phase = phase_search(gray, a, b, args.coarse_step)
    ij, pids, valid = phase["indices"], phase["parity_ids"], phase["valid"]
    save_lattice_image(gray, a, b, np.zeros(2), ij, pids, valid, phase["parity"], args.output / "01_initial_grid.png", "initial")
    save_lattice_image(gray, a, b, phase["origin"], ij, pids, valid, phase["parity"], args.output / "02_optimized_grid.png", "optimized")
    save_lattice_image(gray, a, b, phase["origin"], ij, pids, valid, phase["parity"], args.output / "03_indexed_cells.png", "indexed")
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.imshow(phase["score_map"], origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="viridis")
    ax.scatter([phase["alpha"]], [phase["beta"]], c="red", marker="*", s=130, edgecolors="white")
    ax.set(xlabel="alpha", ylabel="beta", title="Checker-contrast score across one lattice period")
    fig.savefig(args.output / "04_phase_score_map.png", dpi=160)
    plt.close(fig)
    Image.fromarray(gray).save(args.output / "00_original_grayscale.png")
    best_means = phase["means"]
    report = {
        "source": str(args.image.resolve()), "image_size_wh": [gray.shape[1], gray.shape[0]],
        "histogram_peak_levels": peaks, "brightness_bands": ranges, "selected_band_index_1based": group,
        "selected_band": ranges[group - 1], "autocorrelation_basis_a_xy_px": a.tolist(),
        "autocorrelation_basis_b_xy_px": b.tolist(), "basis_lengths_px": [float(np.linalg.norm(a)), float(np.linalg.norm(b))],
        "origin_initial_xy_px": [0, 0], "origin_optimized_xy_px": phase["origin"].tolist(),
        "phase_coefficients_alpha_beta": [phase["alpha"], phase["beta"]],
        "checker_parity_selected_as_brighter": phase["parity"],
        "checker_local_contrast_means_by_parity": [float(x) for x in best_means],
        "score_before": float(phase["before_score"]), "score_after": float(phase["score"]),
        "score_improvement_factor": float(phase["score"] / max(phase["before_score"], 1e-9)),
        "number_of_lattice_centers_in_image": int(valid.sum()), "local_disk_ring_radii_px": phase["radii"],
        "status": "visual_review_required",
        "limitations": ["a,b are fixed autocorrelation translation candidates; no curvature correction is applied.",
                        "checker contrast is a simple local center-minus-ring grayscale measure.",
                        "visual fit must be reviewed; selected bright parity is not ground-truth labeling."]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
