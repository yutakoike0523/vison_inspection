"""Full-image 2-D FFT peak-pair analysis for the grid-light dataset.

This is deliberately standalone: it only creates analysis_full_image_fft_peaks
and never modifies prior analysis directories or source images.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data20260903" / "data20260903"
OUT = ROOT / "analysis_full_image_fft_peaks"
IMAGES = list(range(2, 19))  # image1, 19, 20, 21 are explicitly grid-free.
DC_EXCLUSION_RADIUS_BINS = 4.0
TOP_PAIRS = 20
RECON_NEIGHBORHOOD_RADIUS = 2
EPS = 1e-12


def load_gray(path: Path) -> np.ndarray:
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(path)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    return a.astype(np.float64)


def fft_data(gray: np.ndarray, hann: bool) -> dict:
    h, w = gray.shape
    centered = gray - gray.mean()
    if hann:
        win = np.outer(np.hanning(h), np.hanning(w))
        work = centered * win
    else:
        win = np.ones_like(centered)
        work = centered
    coeff = np.fft.fft2(work)
    shifted = np.fft.fftshift(coeff)
    power = np.abs(shifted) ** 2
    power_db = 10 * np.log10(power + EPS)
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    return {"coeff": coeff, "shifted": shifted, "power": power, "db": power_db,
            "fx": fx, "fy": fy, "work": work, "window": win}


def find_peak_pairs(data: dict, image_number: int, window_name: str, n: int = TOP_PAIRS) -> list[dict]:
    """Return local maxima in one frequency half plane, ranked as symmetric pairs."""
    db, power, fx_axis, fy_axis = data["db"], data["power"], data["fx"], data["fy"]
    h, w = db.shape
    cy, cx = h // 2, w // 2
    local = db == maximum_filter(db, size=5, mode="nearest")
    yy, xx = np.nonzero(local)
    rows = []
    for y, x in zip(yy, xx):
        fy, fx = float(fy_axis[y]), float(fx_axis[x])
        radius_bin = math.hypot(y - cy, x - cx)
        # Unique representative of a conjugate pair: upper half-plane, then right half-axis.
        if radius_bin <= DC_EXCLUSION_RADIUS_BINS or not (fy > 0 or (abs(fy) < 1e-15 and fx > 0)):
            continue
        sy, sx = (-y) % h, (-x) % w
        p = float((power[y, x] + power[sy, sx]) / 2.0)
        dist = math.hypot(fx, fy)
        if dist <= 0:
            continue
        rows.append({
            "image": image_number, "window": window_name, "y": int(y), "x": int(x),
            "sym_y": int(sy), "sym_x": int(sx), "fx": fx, "fy": fy,
            "frequency_distance_cpp": dist, "period_px": 1.0 / dist,
            "direction_deg": math.degrees(math.atan2(fy, fx)),
            "radius_bins": radius_bin, "pair_power": p,
            "pair_power_db": 10 * math.log10(p + EPS),
        })
    rows.sort(key=lambda r: r["pair_power"], reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank_all"] = rank
    return rows[:n], rows


def family_candidates(all_peaks: list[dict]) -> tuple[list[dict], list[dict]]:
    """Select up to two non-parallel low-frequency bases and their collinear harmonics.

    This is evidence-guided, not a claim that all selected peaks are grid light:
    the score rewards repeated integer multiples within the observed local peaks.
    """
    # Start from the strongest two independent, lattice-scale peaks.  This makes
    # the selection inspectable and avoids calling an axis-aligned boundary peak
    # a "fundamental" merely because many unrelated peaks lie on its direction.
    candidates = [r for r in all_peaks[:40] if 45 <= r["period_px"] <= 260]
    chosen = []
    for seed in candidates:
        if not chosen:
            chosen.append(dict(seed))
            continue
        dot = abs(seed["fx"] * chosen[0]["fx"] + seed["fy"] * chosen[0]["fy"])
        denom = seed["frequency_distance_cpp"] * chosen[0]["frequency_distance_cpp"]
        # Treat a conjugate direction as identical; require a genuinely second direction.
        if denom and dot / denom < math.cos(math.radians(25)):
            chosen.append(dict(seed))
            break
    for base in chosen:
        base["family_score"] = base["pair_power_db"]
    selected = []
    for base_index, base in enumerate(chosen, 1):
        vx, vy = base["fx"], base["fy"]
        norm = math.hypot(vx, vy)
        for p in all_peaks[:400]:
            mult = (p["fx"] * vx + p["fy"] * vy) / (norm * norm)
            cross = abs(p["fx"] * vy - p["fy"] * vx) / norm
            nearest = int(round(mult))
            if 1 <= nearest <= 6 and cross <= 1.5 / 1536 and abs(mult - nearest) <= 0.16:
                q = dict(p)
                q["base_index"] = base_index
                q["harmonic_order"] = nearest
                selected.append(q)
    # unique frequency pairs, retain only first attribution
    unique = {}
    for p in selected:
        unique.setdefault((p["y"], p["x"]), p)
    return chosen, list(unique.values())


def make_mask(shape: tuple[int, int], pairs: list[dict], radius: int) -> np.ndarray:
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    Y, X = np.ogrid[:h, :w]
    for p in pairs:
        for y, x in ((p["y"], p["x"]), (p["sym_y"], p["sym_x"])):
            mask |= (Y - y) ** 2 + (X - x) ** 2 <= radius ** 2
    return mask


def reconstruct(data: dict, pairs: list[dict], radius: int) -> np.ndarray:
    mask_shifted = make_mask(data["shifted"].shape, pairs, radius)
    keep = np.fft.ifftshift(data["shifted"] * mask_shifted)
    return np.fft.ifft2(keep).real


def plot_overview(path: Path, gray: np.ndarray, data: dict, top: list[dict], family: list[dict], title: str) -> None:
    conditions = [("Top 1 pair", top[:1]), ("Top 3 pairs", top[:3]),
                  ("Top 5 pairs", top[:5]), ("Top 10 pairs", top[:10]),
                  ("Grid-family candidates", family)]
    images = [reconstruct(data, p, RECON_NEIGHBORHOOD_RADIUS) for _, p in conditions]
    absmax = max(float(np.max(np.abs(a))) for a in images) or 1.0
    # Plot a 1/3 preview only. All FFTs/inverse FFTs above remain full resolution;
    # downsampling here keeps the 68 diagnostic figures practical to generate/view.
    preview = lambda a: a[::3, ::3]
    fig, axes = plt.subplots(4, 4, figsize=(16, 15), constrained_layout=True)
    ax = axes.ravel()
    ax[0].imshow(preview(gray), cmap="gray")
    ax[0].set_title("Original grayscale")
    lo, hi = np.percentile(data["db"], [5, 99.8])
    ax[1].imshow(preview(data["db"]), cmap="magma", vmin=lo, vmax=hi)
    ax[1].set_title("Power spectrum (dB)")
    ax[2].imshow(preview(data["db"]), cmap="magma", vmin=lo, vmax=hi)
    for i, p in enumerate(top, 1):
        ax[2].plot([p["x"] / 3, p["sym_x"] / 3], [p["y"] / 3, p["sym_y"] / 3], "co", ms=3)
        ax[2].text(p["x"] / 3 + 3, p["y"] / 3 + 3, str(i), color="cyan", fontsize=7)
    ax[2].set_title("Top 20 conjugate peak pairs")
    ax[3].axis("off")
    ax[3].text(0.03, 0.94, "Inverse FFT: retained complex coefficients\n(radius = 2 FFT bins)", va="top", fontsize=11)
    for k, ((name, _), im) in enumerate(zip(conditions, images), 4):
        ax[k].imshow(preview(im), cmap="gray")
        ax[k].set_title(name + " — grayscale")
    for k, ((name, _), im) in enumerate(zip(conditions, images), 9):
        ax[k].imshow(preview(im), cmap="coolwarm", vmin=-absmax, vmax=absmax)
        ax[k].set_title(name + " — signed")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title, fontsize=15)
    fig.savefig(path, dpi=85)
    plt.close(fig)


def plot_radius_comparison(path: Path, data: dict, top: list[dict], title: str) -> None:
    radii = [0, 1, 2, 3]
    images = [reconstruct(data, top[:5], r) for r in radii]
    lim = max(float(np.max(np.abs(a))) for a in images) or 1.0
    preview = lambda a: a[::3, ::3]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
    for col, (r, im) in enumerate(zip(radii, images)):
        axes[0, col].imshow(preview(im), cmap="gray")
        axes[0, col].set_title(f"Top 5 pairs, radius {r}: grayscale")
        axes[1, col].imshow(preview(im), cmap="coolwarm", vmin=-lim, vmax=lim)
        axes[1, col].set_title(f"Top 5 pairs, radius {r}: signed")
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=85)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = ["image", "window", "rank_all", "y", "x", "sym_y", "sym_x", "fx", "fy",
            "frequency_distance_cpp", "period_px", "direction_deg", "radius_bins",
            "pair_power", "pair_power_db"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "per_image").mkdir(exist_ok=True)
    all_rows, family_rows, summary = [], [], []
    for image_number in IMAGES:
        gray = load_gray(SOURCE / f"image{image_number}.png")
        for hann, name in ((False, "none"), (True, "hann")):
            data = fft_data(gray, hann)
            top, all_peaks = find_peak_pairs(data, image_number, name)
            bases, family = family_candidates(all_peaks)
            all_rows.extend(top)
            for b in bases:
                family_rows.append({"image": image_number, "window": name, "kind": "base", **b})
            for p in family:
                family_rows.append({"image": image_number, "window": name, "kind": "harmonic", **p})
            plot_overview(OUT / "per_image" / f"image{image_number:02d}_{name}_overview.png",
                          gray, data, top, family, f"image{image_number} — full-image FFT ({name} window)")
            plot_radius_comparison(OUT / "per_image" / f"image{image_number:02d}_{name}_radius_comparison.png",
                                   data, top, f"image{image_number} — top 5 peak pairs; neighborhood comparison ({name})")
            summary.append({
                "image": image_number, "window": name, "top_pair_period_px": top[0]["period_px"],
                "top_pair_direction_deg": top[0]["direction_deg"],
                "top_pair_power_db": top[0]["pair_power_db"],
                "grid_base_count": len(bases),
                "grid_base_1_period_px": bases[0]["period_px"] if bases else "",
                "grid_base_1_direction_deg": bases[0]["direction_deg"] if bases else "",
                "grid_base_2_period_px": bases[1]["period_px"] if len(bases) > 1 else "",
                "grid_base_2_direction_deg": bases[1]["direction_deg"] if len(bases) > 1 else "",
                "grid_family_pair_count": len(family),
            })
            print(f"image{image_number} {name}: top period={top[0]['period_px']:.2f}px, family={len(family)}")
    write_csv(OUT / "fft_peak_pairs_top20.csv", all_rows)
    fields = ["image", "window", "kind", "base_index", "harmonic_order", "rank_all", "fx", "fy",
              "frequency_distance_cpp", "period_px", "direction_deg", "pair_power", "pair_power_db"]
    with (OUT / "grid_family_candidates.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(family_rows)
    with (OUT / "per_image_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    (OUT / "README.txt").write_text(
        "Input: full images image2-image18. Preprocessing: grayscale, mean removal, then FFT.\n"
        "Two variants were generated: no window and 2-D Hann window. DC exclusion: radius <= 4 FFT bins.\n"
        "Peak rows are conjugate-symmetric pairs; fx/fy units are cycles per pixel.\n"
        "Reconstructions preserve the original complex FFT coefficients (including phase).\n"
        "Hann reconstructions are intentionally not divided by the window: division near the zero-valued edge is unstable;\n"
        "they therefore display the expected tapered amplitude. Main visual interpretation uses the no-window reconstruction.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
