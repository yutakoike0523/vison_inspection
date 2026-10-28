"""Select grid vectors by angular bands and radial autocorrelation profiles.

The legacy local-peak selector remains available for side-by-side comparison.
This experiment intentionally reports confidence and keeps ambiguous results
for review instead of forcing a successful lattice fit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.signal import find_peaks

from grid_period_fft import (autocorrelation_2d, estimate_spacing,
                             histogram_groups, select_grid_basis)
from auto_fit_grid_image15 import phase_search, save_lattice_image


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "data20260903/data20260903/triming"
OUTPUT_ROOT = ROOT / "outputs/⑫_自己相関方向追跡"

# All values can be changed from CLI; these defaults are shared by image15/10.
ANGLE_STEP_DEG = 1.0
ANGULAR_BAND_HALF_WIDTH_PX = 4
MIN_RADIUS_PX = 24.0
MIN_BASIS_ANGLE_SEPARATION_DEG = 30.0
RADIAL_SMOOTH_SIGMA_PX = 1.1
MIN_PEAK_PROMINENCE = 0.008
MIN_DIRECTION_SYMMETRY = 0.55


def radial_profiles(ac, center, theta_deg, min_radius, max_radius, band_half_width):
    cx, cy = center
    radii = np.arange(float(min_radius), float(max_radius) + .5, 1.0)
    offsets = np.arange(-band_half_width, band_half_width + 1, dtype=float)
    theta = np.deg2rad(theta_deg)
    ux, uy = np.cos(theta), np.sin(theta)
    nx, ny = -uy, ux
    plus = np.empty((len(offsets), len(radii)), dtype=np.float32)
    minus = np.empty_like(plus)
    for k, off in enumerate(offsets):
        xp, yp = cx + radii * ux + off * nx, cy + radii * uy + off * ny
        xm, ym = cx - radii * ux + off * nx, cy - radii * uy + off * ny
        plus[k] = map_coordinates(ac, [yp, xp], order=1, mode="nearest", prefilter=False)
        minus[k] = map_coordinates(ac, [ym, xm], order=1, mode="nearest", prefilter=False)
    return radii, np.median(plus, axis=0), np.median(minus, axis=0)


def paired_radial_peaks(radii, positive, negative, prominence=MIN_PEAK_PROMINENCE):
    p = gaussian_filter1d(positive.astype(float), RADIAL_SMOOTH_SIGMA_PX)
    n = gaussian_filter1d(negative.astype(float), RADIAL_SMOOTH_SIGMA_PX)
    pp, props = find_peaks(p, prominence=prominence, distance=5, width=1)
    nn, nprops = find_peaks(n, prominence=prominence, distance=5, width=1)
    candidates = []
    used = set()
    for k, idx in enumerate(pp):
        if len(nn) == 0:
            continue
        j = int(np.argmin(np.abs(nn - idx)))
        other = int(nn[j])
        if abs(other - idx) > 4:
            continue
        used.add(j)
        pval, nval = float(p[idx]), float(n[other])
        sym = float(min(pval, nval) / max(abs(pval), abs(nval), 1e-9))
        prom = float(min(props["prominences"][k], nprops["prominences"][j]))
        r = float((radii[idx] + radii[other]) / 2)
        candidates.append({"radius_px": r, "positive_peak": pval, "negative_peak": nval,
                           "symmetry": sym, "prominence": prom,
                           "width_px": float((props["widths"][k] + nprops["widths"][j]) / 2),
                           "score": float(prom * sym)})
    return p, n, sorted(candidates, key=lambda x: x["radius_px"])


def angle_scan(ac, center, angle_step=ANGLE_STEP_DEG,
               half_width=ANGULAR_BAND_HALF_WIDTH_PX, min_radius=MIN_RADIUS_PX):
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    max_radius = min(cx, cy) * .88
    angles = np.arange(0.0, 180.0, angle_step)
    scores, symmetries, profiles = [], [], {}
    for theta in angles:
        radii, pos, neg = radial_profiles(ac, (cx, cy), theta, min_radius,
                                          max_radius, half_width)
        ps, ns, peaks = paired_radial_peaks(radii, pos, neg)
        credible = [p for p in peaks if p["symmetry"] >= MIN_DIRECTION_SYMMETRY]
        # Use several symmetric peaks along the ray as evidence for a direction.
        score = float(sum(p["score"] for p in sorted(credible,
                            key=lambda q: q["score"], reverse=True)[:3]))
        symmetry = float(np.mean([p["symmetry"] for p in credible[:3]])) if credible else 0.0
        scores.append(score)
        symmetries.append(symmetry)
        profiles[float(theta)] = {"radii": radii, "positive": ps, "negative": ns,
                                  "peaks": peaks}
    scores = np.asarray(scores)
    # Smooth the angular score across wraparound (direction is modulo 180°).
    smoothed = gaussian_filter1d(np.r_[scores, scores, scores], sigma=max(.7, 2 / angle_step))[len(scores):2*len(scores)]
    return angles, scores, smoothed, np.asarray(symmetries), profiles


def angle_dist_undirected(a, b):
    d = abs(float(a) - float(b)) % 180.0
    return min(d, 180.0 - d)


def choose_direction_vectors(angles, raw_scores, scores, symmetries, profiles,
                             min_separation=MIN_BASIS_ANGLE_SEPARATION_DEG):
    ranked = np.argsort(scores)[::-1]
    chosen = []
    for idx in ranked:
        theta = float(angles[idx])
        if any(angle_dist_undirected(theta, x["theta_deg"]) < min_separation for x in chosen):
            continue
        profile = profiles[theta]
        plausible = [p for p in profile["peaks"]
                     if p["prominence"] >= MIN_PEAK_PROMINENCE
                     and p["symmetry"] >= MIN_DIRECTION_SYMMETRY]
        if not plausible:
            continue
        # Nearest credible peak avoids automatically jumping to a harmonic.
        peak = min(plausible, key=lambda p: p["radius_px"])
        later_peaks = [p for p in plausible if p["radius_px"] > peak["radius_px"] + 1]
        second_peak = later_peaks[0] if later_peaks else None
        rad = np.deg2rad(theta)
        vec = [float(peak["radius_px"] * np.cos(rad)),
               float(peak["radius_px"] * np.sin(rad))]
        neighbors = [scores[k] for k in range(len(scores))
                     if angle_dist_undirected(theta, angles[k]) >= 5
                     and angle_dist_undirected(theta, angles[k]) <= 15]
        local_baseline = float(np.mean(neighbors)) if neighbors else 0.0
        angle_prom = float(scores[idx] - local_baseline)
        angle_quality = float(np.clip(angle_prom / max(.5 * scores[idx], 1e-9), 0, 1))
        peak_quality = float(np.clip(peak["prominence"] / .20, 0, 1))
        width_quality = float(1 / (1 + peak["width_px"] / max(peak["radius_px"], 1)))
        second_gap = (float(abs(peak["prominence"] - second_peak["prominence"])
                            / max(peak["prominence"], second_peak["prominence"], 1e-9))
                      if second_peak else 1.0)
        chosen.append({"theta_deg": theta, "direction_score": float(scores[idx]),
                       "raw_direction_score": float(raw_scores[idx]),
                       "angle_prominence": angle_prom, "direction_symmetry": float(symmetries[idx]),
                       "radius_peak": peak, "vector_xy_px": vec,
                       "vector_length_px": float(np.linalg.norm(vec)),
                       "second_radial_peak": second_peak,
                       "first_second_peak_prominence_gap": second_gap,
                       "confidence_components": {"angle_prominence_quality": angle_quality,
                                                  "positive_negative_symmetry": peak["symmetry"],
                                                  "peak_prominence_quality": peak_quality,
                                                  "radial_peak_width_quality": width_quality,
                                                  "first_second_radial_peak_gap": second_gap},
                       "radial_profile": profile})
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        return chosen, "unreliable"
    sep = angle_dist_undirected(chosen[0]["theta_deg"], chosen[1]["theta_deg"])
    independence = float(np.clip(np.sin(np.deg2rad(sep)), 0, 1))
    for d in chosen:
        c = d["confidence_components"]
        c["basis_independence"] = independence
        d["confidence"] = float(.35*c["angle_prominence_quality"]
                                + .20*c["positive_negative_symmetry"]
                                + .15*c["peak_prominence_quality"]
                                + .10*c["radial_peak_width_quality"]
                                + .10*c["first_second_radial_peak_gap"]
                                + .10*independence)
    min_conf = min(x["confidence"] for x in chosen)
    status = "ok" if min_conf >= .75 and sep >= 45 else "needs_review"
    if min_conf < .20:
        status = "unreliable"
    chosen[0]["basis_angle_separation_deg"] = sep
    chosen[1]["basis_angle_separation_deg"] = sep
    return chosen, status


def save_outputs(gray, ac, ac_center, old_basis, new_dirs, angles, raw_score,
                 score, symmetry, profiles, phase, out):
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gray).save(out / "00_original_grayscale.png")
    cy, cx = ac_center
    extent = [-(cx), ac.shape[1]-cx-1, ac.shape[0]-cy-1, -(cy)]
    # Original autocorrelation with legacy and directional vectors.
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    lim = min(ac.shape) * .40
    ax.imshow(ac, origin="lower", extent=extent, cmap="coolwarm", vmin=-.2, vmax=1)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="dx [px]", ylabel="dy [px]",
           title="Autocorrelation: legacy peak vectors and angular-profile vectors")
    for k, item in enumerate(old_basis[:2]):
        ax.arrow(0, 0, item["dx"], item["dy"], color=["yellow", "cyan"][k], width=1.3,
                 length_includes_head=True, label=f"legacy {k+1}")
    for k, item in enumerate(new_dirs[:2]):
        vx, vy = item["vector_xy_px"]
        ax.arrow(0, 0, vx, vy, color=["lime", "magenta"][k], width=1.3,
                 length_includes_head=True, label=f"angular {k+1}")
    ax.legend(loc="upper right")
    fig.savefig(out / "01_autocorrelation_original.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for k, ax in enumerate(axes):
        ax.imshow(ac, origin="lower", extent=extent, cmap="coolwarm", vmin=-.2, vmax=1)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="dx [px]", ylabel="dy [px]")
        if k == 0:
            ax.set_title("Legacy: local peak enumeration")
            vecs = [[x["dx"], x["dy"]] for x in old_basis[:2]]
        else:
            ax.set_title("New: angular band + radial peak")
            vecs = [x["vector_xy_px"] for x in new_dirs[:2]]
        for q, vec in enumerate(vecs):
            ax.arrow(0, 0, vec[0], vec[1], color=["yellow", "cyan"][q], width=1.5,
                     length_includes_head=True)
            ax.text(vec[0], vec[1], f" {np.linalg.norm(vec):.1f}px", color="white",
                    fontsize=8, bbox={"facecolor": "black", "alpha": .55, "pad": 2})
    fig.savefig(out / "06_old_new_basis_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.plot(angles, raw_score, color="0.7", lw=.8, label="raw score")
    ax.plot(angles, score, color="navy", lw=1.5, label="smoothed direction score")
    for k, d in enumerate(new_dirs[:2]):
        ax.axvline(d["theta_deg"], color=["limegreen", "magenta"][k], ls="--",
                   label=f"basis {k+1}: {d['theta_deg']:.1f}°")
    ax.set(xlabel="angle [deg] (undirected, 0–180)", ylabel="direction score",
           title="Angle vs direction score")
    ax.legend(ncol=4)
    fig.savefig(out / "02_angle_score.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.imshow(ac, origin="lower", extent=extent, cmap="coolwarm", vmin=-.2, vmax=1)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="dx [px]", ylabel="dy [px]",
           title=f"Selected angular bands (±{ANGULAR_BAND_HALF_WIDTH_PX}px)")
    for k, d in enumerate(new_dirs[:2]):
        theta = np.deg2rad(d["theta_deg"])
        ux, uy = np.cos(theta), np.sin(theta)
        nx, ny = -uy, ux
        r = lim * .9
        color = ["lime", "magenta"][k]
        ax.plot([-r*ux, r*ux], [-r*uy, r*uy], color=color, lw=2,
                label=f"{d['theta_deg']:.1f}° ±{ANGULAR_BAND_HALF_WIDTH_PX}px")
        for side in (-1, 1):
            offx, offy = side * ANGULAR_BAND_HALF_WIDTH_PX * nx, side * ANGULAR_BAND_HALF_WIDTH_PX * ny
            ax.plot([-r*ux + offx, r*ux + offx], [-r*uy + offy, r*uy + offy],
                    color=color, lw=.8, alpha=.8)
        vx, vy = d["vector_xy_px"]
        ax.scatter([vx, -vx], [vy, -vy], s=60, facecolors="none", edgecolors=color)
    ax.legend()
    fig.savefig(out / "03_direction_bands.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for k, (ax, d) in enumerate(zip(axes, new_dirs[:2])):
        prof = d["radial_profile"]
        r = prof["radii"]
        ax.plot(r, prof["positive"], label="+ direction")
        ax.plot(r, prof["negative"], label="− direction", alpha=.8)
        peak = d["radius_peak"]
        ax.axvline(peak["radius_px"], color="red", ls="--",
                   label=f"credible r={peak['radius_px']:.1f}px")
        ax.set(xlabel="radius r [px]", ylabel="normalized autocorrelation",
               title=f"basis{k+1}: θ={d['theta_deg']:.1f}°, symmetry={peak['symmetry']:.2f}")
        ax.legend()
    fig.savefig(out / "04_radial_profiles.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    colors = ["#2774c8", "#36a269"]
    for k, d in enumerate(new_dirs[:2]):
        pr = d["radial_profile"]
        axes[k].plot(pr["radii"], pr["positive"], color=colors[k], label="+θ")
        axes[k].plot(pr["radii"], pr["negative"], color="gray", label="θ+180°")
        axes[k].axvline(d["radius_peak"]["radius_px"], color="red", ls="--")
        axes[k].set_title(f"basis{k+1}: θ={d['theta_deg']:.1f}°, r={d['radius_peak']['radius_px']:.1f}px")
        axes[k].set_xlabel("distance [px]")
        axes[k].set_ylabel("autocorrelation")
        axes[k].legend()
    fig.savefig(out / "05_radial_profile_peaks.png", dpi=160)
    plt.close(fig)

    # New vectors before/after origin fitting and indexed cells.
    vecs = [d["vector_xy_px"] for d in new_dirs]
    a, b = np.asarray(vecs[0]), np.asarray(vecs[1])
    ids, parity, valid = phase["indices"], phase["parity_ids"], phase["valid"]
    save_lattice_image(gray, a, b, np.zeros(2), ids, parity, valid, phase["parity"],
                       out / "07_grid_overlay_before_origin_fit.png", "initial")
    save_lattice_image(gray, a, b, phase["origin"], ids, parity, valid, phase["parity"],
                       out / "08_grid_overlay_after_origin_fit.png", "optimized")
    save_lattice_image(gray, a, b, phase["origin"], ids, parity, valid, phase["parity"],
                       out / "09_indexed_cells.png", "indexed")


def analyze_image(image_path, args):
    gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
    _, _, ranges, peaks, _ = histogram_groups(gray)
    group = 1 if len(ranges) == 1 and args.group_index == 2 else args.group_index
    if not 1 <= group <= len(ranges):
        raise ValueError(f"{image_path.name}: requested band {group} but bands={ranges}")
    lo, hi = ranges[group - 1]
    mask = (gray >= lo) & (gray <= hi)
    ac, center = autocorrelation_2d(mask.astype(float))
    old_ac, old_center, old_peak, old_candidates = estimate_spacing(mask, min_strength=.045,
                                                                     min_distance=24, min_harmonics=2)
    old_basis = select_grid_basis(old_candidates, min_sine_angle=.45)
    angles, raw, smoothed, symmetry, profiles = angle_scan(
        ac, center, args.angle_step, args.band_half_width, args.min_radius)
    directions, status = choose_direction_vectors(angles, raw, smoothed, symmetry,
                                                   profiles, args.min_basis_separation)
    if len(directions) == 2:
        a, b = [np.asarray(d["vector_xy_px"]) for d in directions]
        phase = phase_search(gray, a, b, args.phase_step)
        phase_summary = {"origin_xy": phase["origin"].tolist(), "alpha": phase["alpha"],
                         "beta": phase["beta"], "checker_parity": phase["parity"],
                         "score_before": float(phase["before_score"]),
                         "score_after": float(phase["score"]),
                         "score_improvement_factor": float(phase["score"] / max(phase["before_score"], 1e-9)),
                         "number_of_grid_centers": int(phase["valid"].sum())}
    else:
        phase = None
        phase_summary = None
    out = args.output_root / image_path.stem.replace("_triming", "")
    out.mkdir(parents=True, exist_ok=True)
    # Angle plot and diagnostics even when the estimate is unreliable.
    if phase is not None:
        save_outputs(gray, ac, center, old_basis, directions, angles, raw,
                     smoothed, symmetry, profiles, phase, out)
    else:
        Image.fromarray(gray).save(out / "00_original_grayscale.png")
        fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
        ax.plot(angles, smoothed)
        ax.set(xlabel="angle [deg]", ylabel="direction score", title="Direction scores (no reliable pair)")
        fig.savefig(out / "02_angle_score.png", dpi=150)
        plt.close(fig)
    # Ensure vector diagnostics exist separately from the other figures.
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.plot(angles, raw, color="0.72", lw=.8, label="raw")
    ax.plot(angles, smoothed, color="navy", label="smoothed")
    for k, d in enumerate(directions[:2]):
        ax.axvline(d["theta_deg"], color=["green", "magenta"][k], ls="--",
                   label=f"selected {d['theta_deg']:.1f} deg")
    ax.set(xlabel="angle [deg]", ylabel="direction score", title="Angle vs direction score")
    ax.legend()
    fig.savefig(out / "02_angle_score.png", dpi=160)
    plt.close(fig)

    old_a = [float(old_basis[0]["dx"]), float(old_basis[0]["dy"])] if len(old_basis) > 0 else None
    old_b = [float(old_basis[1]["dx"]), float(old_basis[1]["dy"])] if len(old_basis) > 1 else None
    new_a = directions[0]["vector_xy_px"] if len(directions) > 0 else None
    new_b = directions[1]["vector_xy_px"] if len(directions) > 1 else None
    basis_angle = None
    if new_a and new_b:
        u, v = np.asarray(new_a), np.asarray(new_b)
        basis_angle = float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (np.linalg.norm(u)*np.linalg.norm(v)), -1, 1))))
    confidence = float(min((d["confidence"] for d in directions), default=0.0))
    report = {"image_name": image_path.name, "image_size_wh": [gray.shape[1], gray.shape[0]],
              "selected_brightness_band": [int(lo), int(hi)], "angle_step_deg": args.angle_step,
              "band_half_width_px": args.band_half_width, "min_radius_px": args.min_radius,
              "min_basis_angle_separation_deg": args.min_basis_separation,
              "old_basis_a": old_a, "old_basis_b": old_b,
              "new_basis_a": new_a, "new_basis_b": new_b,
              "basis_1": [{k: v for k, v in d.items() if k != "radial_profile"} for d in directions[:1]],
              "basis_2": [{k: v for k, v in d.items() if k != "radial_profile"} for d in directions[1:2]],
              "basis_angle_deg": basis_angle, "checker_phase_fit": phase_summary,
              "overall_confidence": confidence, "status": status,
              "old_candidate_peaks_count": len(old_candidates),
              "histogram_peak_levels": peaks, "autocorrelation_grid_center_xy": list(center),
              "notes": ["Old method: enumerate local autocorrelation peaks, then select independent vectors.",
                        "New method: scan angular bands first, score symmetric radial profiles, then select the first credible radial peak.",
                        "The angular-profile method does not force 90-degree basis separation.",
                        "Phase fitting uses the existing center-minus-ring score; no curvature correction is applied."]}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", nargs="*", default=["image15", "image10"],
                    help="Trimmed image stems to process (default: image15 image10)")
    ap.add_argument("--group-index", type=int, default=2)
    ap.add_argument("--angle-step", type=float, default=ANGLE_STEP_DEG)
    ap.add_argument("--band-half-width", type=int, default=ANGULAR_BAND_HALF_WIDTH_PX)
    ap.add_argument("--min-radius", type=float, default=MIN_RADIUS_PX)
    ap.add_argument("--min-basis-separation", type=float, default=MIN_BASIS_ANGLE_SEPARATION_DEG)
    ap.add_argument("--phase-step", type=float, default=.04)
    ap.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = ap.parse_args()
    reports = []
    for stem in args.images:
        path = INPUT_DIR / f"{stem}_triming.png"
        if not path.exists():
            raise SystemExit(f"Input not found: {path}")
        report = analyze_image(path, args)
        reports.append(report)
        print(f"{stem}: status={report['status']}, old a,b={report['old_basis_a']},{report['old_basis_b']}; "
              f"new a,b={report['new_basis_a']},{report['new_basis_b']}; confidence={report['overall_confidence']:.3f}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "all_images_summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
