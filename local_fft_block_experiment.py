"""Local FFT experiment on full original images; compare 128 / 256 px blocks.

Run: python local_fft_block_experiment.py
Uses FFT helpers from compare_fft_center_peaks.py in this same folder.
Dependencies: numpy, opencv-python, matplotlib (no machine learning).
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Rectangle
import numpy as np

from compare_fft_center_peaks import (
    ROOT, EXTENSIONS, read_gray, hann_fft, center_zoom, display_limits, detect_peaks,
)


BLOCK_SIZES = [128, 256]
DC_MASK_RADIUS = 5
DEFAULT_COLOR_METRIC = "peak_energy_ratio"
REFERENCE_BLOCKS = []  # Zero-based (row, col), used with --reference-mode manual.
WEIGHTS = {"peak_position_error": 1., "peak_angle_error": 1., "peak_strength_drop": 1.,
           "energy_ratio_drop": 1., "spectral_spread_increase": 1.}


def block_positions(shape, size, stride):
    for row, y in enumerate(range(0, shape[0] - size + 1, stride)):
        for col, x in enumerate(range(0, shape[1] - size + 1, stride)):
            yield row, col, x, y


def block_features(block, dc_radius=DC_MASK_RADIUS, zoom_radius=None, max_peaks=8,
                   peak_percentile=95., peak_radius=2.):
    size = block.shape[0]
    transformed, log_spectrum = hann_fft(block)
    magnitude = np.abs(transformed)
    power = magnitude ** 2  # Energy uses |F|^2, never the logarithmic display.
    radius = min(zoom_radius if zoom_radius is not None else size // 4, size // 2)
    zoom, valid_zoom, bounds, extent = center_zoom(log_spectrum, radius, dc_radius)
    peaks, threshold = detect_peaks(zoom, valid_zoom, bounds, block.shape, max_peaks,
                                    peak_percentile, min_distance=3, local_max_size=3)
    yy, xx = np.mgrid[:size, :size]
    dx, dy = xx - size // 2, yy - size // 2
    valid = dx ** 2 + dy ** 2 > dc_radius ** 2
    total = float(power[valid].sum())
    all_energy = float(power.sum())
    # Union of peak disks: overlapping disks must not count energy twice.
    nearest_squared = np.full(block.shape, np.inf)
    for peak in peaks:
        distance_squared = (dx - peak["x_offset"]) ** 2 + (dy - peak["y_offset"]) ** 2
        np.minimum(nearest_squared, distance_squared, out=nearest_squared)
    peak_zone = valid & (nearest_squared <= peak_radius ** 2)
    peak_energy = float(power[peak_zone].sum())
    # RMS distance to the nearest selected peak, weighted by all non-DC power.
    # Unit = cycles/pixel; this includes diffuse energy outside the peak disks.
    spread = (float(np.sqrt(np.sum(power[valid] * nearest_squared[valid]) / total) / size)
              if peaks and total > 1e-20 else None)
    main = peaks[0] if peaks else None
    main_mag = (float(magnitude[size // 2 + main["y_offset"], size // 2 + main["x_offset"]])
                if main else 0.)
    window_sum = float(np.hanning(size).sum() ** 2)
    features = {
        "block_mean": float(block.mean()), "block_std": float(block.std()),
        "zoom_radius": radius, "dc_mask_radius": dc_radius,
        "peak_neighborhood_radius": peak_radius, "detected_peak_count": len(peaks),
        "main_peak_x": main["x_offset"] if main else None,
        "main_peak_y": main["y_offset"] if main else None,
        "main_peak_distance": main["distance_from_center"] if main else None,
        "main_peak_angle": main["angle_deg"] % 180 if main else None,
        "main_peak_log_amplitude": main["log_amplitude"] if main else None,
        "main_peak_magnitude": main_mag,
        "main_peak_frequency": main["distance_from_center"] / size if main else None,
        "main_peak_period_pixels": main["period_pixels"] if main else None,
        # Dividing by window sum avoids simple block-area scaling in strength.
        "peak_strength_normalized": main_mag / max(window_sum, 1e-20),
        "symmetric_peak_found": bool(main),
        "peak_energy_ratio": peak_energy / total if total > 1e-20 else 0.,
        "peak_energy_ratio_defined": total > 1e-20,
        "spectral_spread": spread,
        "spectral_spread_bins": spread * size if spread is not None else None,
        "total_energy_excluding_dc": total,
        "energy_fraction_outside_dc": total / all_energy if all_energy > 1e-20 else 0.,
        "peak_threshold_log_amplitude": threshold,
        "peak_candidates_json": json.dumps(peaks, separators=(",", ":")),
        "feature_status": "ok" if main else "no_peak_candidates",
    }
    return features, {"spectrum": log_spectrum, "zoom": zoom, "valid_zoom": valid_zoom,
                      "extent": extent, "peaks": peaks}


def axial_distance(a, b):
    """Directions are modulo 180 degrees: +f and -f describe one orientation."""
    delta = np.abs(np.asarray(a) - b) % 180
    return np.minimum(delta, 180 - delta)


def axial_median(angles):
    values = np.asarray(angles, dtype=float) % 180
    if not len(values):
        return None
    cost = np.sum(axial_distance(values[:, None], values[None, :]), axis=1)
    return float(values[np.argmin(cost)])


def build_reference(records, mode="median", reference_blocks=None):
    if mode == "manual":
        requested = {tuple(pair) for pair in (reference_blocks or [])}
        available = {(r["block_row"], r["block_col"]) for r in records}
        if not requested or requested - available:
            raise ValueError(f"Manual reference blocks are empty or outside the grid: {sorted(requested - available)}")
        selected = [r for r in records if (r["block_row"], r["block_col"]) in requested]
    else:
        selected = records
    good = [r for r in selected if r["feature_status"] == "ok"]
    fields = ["main_peak_distance", "peak_strength_normalized", "peak_energy_ratio", "spectral_spread"]
    ref = {"mode": mode, "reference_count": len(selected), "reference_peak_count": len(good),
           "reference_blocks": [[r["block_row"], r["block_col"]] for r in selected],
           "main_peak_angle": axial_median([r["main_peak_angle"] for r in good])}
    for field in fields:
        values = [r[field] for r in selected if r[field] is not None]
        ref[field] = float(np.median(values)) if values else None
    ref["valid"] = bool(good and ref["peak_strength_normalized"] > 1e-12
                        and ref["peak_energy_ratio"] > 1e-12)
    ref["angle_median_absolute_deviation_deg"] = (
        float(np.median(axial_distance([r["main_peak_angle"] for r in good], ref["main_peak_angle"])))
        if good else None)
    ref["note"] = ("Coordinate-wise feature medians; angles use an axial circular median. "
                   "This reference is an assumption, not a verified normal class.")
    if not ref["valid"]:
        ref["warning"] = "No usable median peak reference; anomaly scores are unavailable, not zero."
    elif len(selected) == 1:
        ref["warning"] = "Only one reference block: it is compared with itself and cannot validate normality."
    elif len(good) < len(selected) / 2:
        ref["warning"] = "Most reference blocks have no selected peaks; review DC radius and reference selection."
    return ref


def score_records(records, reference, weights):
    for record in records:
        if not reference["valid"]:
            record.update({name: None for name in weights})
            record["anomaly_score"] = None
            record["score_status"] = "invalid_reference"
            continue
        present = record["feature_status"] == "ok"
        ref_distance = reference["main_peak_distance"]
        ref_strength = reference["peak_strength_normalized"]
        ref_ratio = reference["peak_energy_ratio"]
        ref_spread = reference["spectral_spread"]
        components = {
            "peak_position_error": abs(record["main_peak_distance"] - ref_distance) / max(ref_distance, 1.) if present else 1.,
            "peak_angle_error": float(axial_distance(record["main_peak_angle"], reference["main_peak_angle"])) / 90 if present else 1.,
            "peak_strength_drop": max(0., 1 - record["peak_strength_normalized"] / ref_strength),
            "energy_ratio_drop": max(0., 1 - record["peak_energy_ratio"] / ref_ratio),
            "spectral_spread_increase": max(0., (record["spectral_spread"] - ref_spread) /
                                             max(ref_spread, 1. / record["block_size"])) if present else 1.,
        }
        # Dimensionless 0..1 terms keep one large deviation from dominating.
        record.update({name: float(np.clip(value, 0., 1.)) for name, value in components.items()})
        record["anomaly_score"] = float(sum(weights[name] * record[name] for name in weights))
        record["score_status"] = "ok" if present else "missing_peaks_penalty"


def analyze_blocks(gray, filename, size, stride, mode="median", reference_blocks=None,
                   weights=None, **feature_options):
    records = []
    for row, col, x, y in block_positions(gray.shape, size, stride):
        features, _ = block_features(gray[y:y + size, x:x + size], **feature_options)
        records.append({"filename": filename, "block_row": row, "block_col": col,
                        "x_start": x, "y_start": y, "block_size": size, "stride": stride, **features})
    if not records:
        raise ValueError(f"Image {gray.shape[1]}x{gray.shape[0]} has no complete {size}x{size} blocks")
    reference = build_reference(records, mode, reference_blocks)
    score_records(records, reference, weights or WEIGHTS)
    return records, reference


def metric_value(record, metric):
    # A zero denominator makes R undefined, not a low-concentration observation.
    if metric == "peak_energy_ratio" and not record["peak_energy_ratio_defined"]:
        return None
    value = record[metric]
    return value if value is not None and np.isfinite(value) else None


def metric_style(metric):
    if metric == "peak_energy_ratio":
        return "RdYlBu", "R: peak energy / non-DC energy (red=low, blue=high)", "neither"
    return "inferno", "Relative anomaly score (not a defect probability)", "max"


def rank_blocks(records, metric):
    valid = [r for r in records if metric_value(r, metric) is not None]
    return sorted(valid, key=lambda r: metric_value(r, metric), reverse=metric == "anomaly_score")


def rasterize_scores(shape, records, metric="anomaly_score"):
    """Overlapping blocks contribute their mean metric; uncovered pixels are NaN."""
    sums, counts = np.zeros(shape, dtype=float), np.zeros(shape, dtype=np.int32)
    for record in records:
        value = metric_value(record, metric)
        if value is None:
            continue
        x, y, size = record["x_start"], record["y_start"], record["block_size"]
        sums[y:y + size, x:x + size] += value
        counts[y:y + size, x:x + size] += 1
    heatmap = np.full(shape, np.nan)
    np.divide(sums, counts, out=heatmap, where=counts > 0)
    return heatmap


def draw_boundaries(ax, records, labels=False, scores=False, metric="anomaly_score"):
    for record in records:
        x, y, size = record["x_start"], record["y_start"], record["block_size"]
        ax.add_patch(Rectangle((x - .5, y - .5), size, size, fill=False,
                               edgecolor="white", linewidth=.4, alpha=.65))
        if labels or scores:
            text = f"{record['block_row']},{record['block_col']}" if labels else ""
            if scores:
                score = metric_value(record, metric)
                text += f"\n{score:.2f}" if score is not None else "\nN/A"
            ax.text(x + 3, y + 5, text, color="white", fontsize=5, va="top",
                    bbox={"facecolor": "black", "alpha": .4, "pad": .5, "edgecolor": "none"})


def render_map(gray, records, heatmap, kind, destination, vmax, show_scores=False, metric="anomaly_score"):
    cmap, color_label, extend = metric_style(metric)
    fig, ax = plt.subplots(figsize=(11, 8), layout="constrained")
    try:
        if kind != "heatmap":
            ax.imshow(gray, cmap="gray")
        if kind != "grid":
            ax.set_facecolor("#777777")
            plotted = ax.imshow(np.ma.masked_invalid(heatmap), cmap=cmap, vmin=0, vmax=vmax,
                                alpha=.48 if kind == "overlay" else 1., interpolation="nearest")
            fig.colorbar(ScalarMappable(norm=Normalize(0, vmax), cmap=cmap), ax=ax,
                         label=color_label, extend=extend)
        if kind != "heatmap":
            draw_boundaries(ax, records, labels=kind == "grid", scores=show_scores and kind == "overlay", metric=metric)
        ax.set(xlim=(-.5, gray.shape[1] - .5), ylim=(gray.shape[0] - .5, -.5))
        ax.set_title(f"{records[0]['filename']} | {records[0]['block_size']} px / stride {records[0]['stride']} | {metric}: {kind}\n"
                     "Uncovered / unscored pixels have no heat color; grid labels are zero-based row,col")
        ax.axis("off")
        fig.savefig(destination, dpi=170)
    finally:
        plt.close(fig)


def write_csv(path, records):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def save_block_comparison(gray, records, reference, feature_options, top_k, destination, metric="anomaly_score"):
    ranked = rank_blocks(records, metric)
    if not ranked:
        return []
    if metric == "peak_energy_ratio":
        chosen = [("Highest R (comparison)", ranked[-1])]
        chosen += [(f"Low R rank {i + 1}", r) for i, r in enumerate(ranked[:top_k])]
    else:
        ref_keys = {tuple(p) for p in reference["reference_blocks"]}
        representative = min((r for r in records if (r["block_row"], r["block_col"]) in ref_keys),
                             key=lambda r: r["anomaly_score"])
        chosen = [("Reference representative", representative)] + [(f"Rank {i + 1}", r) for i, r in enumerate(ranked[:top_k])]
    details = []
    for label, record in chosen:
        x, y, size = record["x_start"], record["y_start"], record["block_size"]
        block = gray[y:y + size, x:x + size]
        _, fft = block_features(block, **feature_options)
        details.append((label, record, block, fft))
    pooled = np.concatenate([fft["zoom"][fft["valid_zoom"]] for _, _, _, fft in details])
    limits = display_limits(pooled)
    fig, axes = plt.subplots(len(details), 4, figsize=(15, 3 * len(details)), squeeze=False, layout="constrained")
    try:
        for row, (label, rec, block, fft) in enumerate(details):
            masked = fft["zoom"].copy()
            masked[~fft["valid_zoom"]] = limits[0]
            axes[row, 0].imshow(block, cmap="gray", vmin=float(gray.min()), vmax=max(float(gray.max()), float(gray.min()) + 1))
            value_label = "R" if metric == "peak_energy_ratio" else "score"
            axes[row, 0].set_title(f"{label}: r{rec['block_row']},c{rec['block_col']}\n"
                                   f"{value_label}={metric_value(rec, metric):.3f}", fontsize=10)
            for ax, data, title in zip(axes[row, 1:], (fft["spectrum"], fft["zoom"], masked),
                                       ("Hann FFT (full)", "FFT center zoom", "DC mask + peak candidates")):
                ax.imshow(data, cmap="gray", vmin=limits[0], vmax=limits[1], interpolation="nearest",
                          extent=fft["extent"] if ax != axes[row, 1] else None)
                ax.set_title(title, fontsize=10)
            axes[row, 3].add_patch(Circle((0, 0), rec["dc_mask_radius"], fill=False, edgecolor="cyan", linewidth=.6))
            axes[row, 3].scatter([p["x_offset"] for p in fft["peaks"]], [p["y_offset"] for p in fft["peaks"]],
                                 s=55, facecolors="none", edgecolors="red", linewidths=.7)
            spread = f"{rec['spectral_spread']:.4f}" if rec["spectral_spread"] is not None else "N/A"
            axes[row, 0].set_xlabel(f"Energy ratio={rec['peak_energy_ratio']:.3f} | spread={spread}", fontsize=9)
            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])
        ordering = "lowest R first (no normal-reference subtraction)" if metric == "peak_energy_ratio" else f"reference={reference['mode']}"
        fig.suptitle(f"{records[0]['filename']} | block {records[0]['block_size']} | "
                     f"{ordering}\nAll FFT panels share log-amplitude scale {limits[0]:.2f}..{limits[1]:.2f}", fontsize=14)
        fig.savefig(destination, dpi=140)
    finally:
        plt.close(fig)
    return ranked[:top_k]


def save_size_comparison(gray, results, destination, vmax, metric="anomaly_score"):
    cmap, color_label, extend = metric_style(metric)
    fig, axes = plt.subplots(2, len(results), figsize=(7 * len(results), 10), squeeze=False, layout="constrained")
    try:
        for col, result in enumerate(results):
            for row in (0, 1):
                ax = axes[row, col]
                ax.set_facecolor("#777777")
                if row == 1:
                    ax.imshow(gray, cmap="gray")
                plotted = ax.imshow(np.ma.masked_invalid(result["heatmap"]), cmap=cmap, vmin=0, vmax=vmax,
                                    alpha=.48 if row else 1, interpolation="nearest")
                ax.set_title(f"{result['size']} x {result['size']} | stride {result['stride']}\n"
                             + ("Overlay" if row else ("Peak energy ratio R" if metric == "peak_energy_ratio" else "Anomaly heatmap")))
                ax.axis("off")
        fig.colorbar(ScalarMappable(norm=Normalize(0, vmax), cmap=cmap), ax=list(axes.flat),
                     label=color_label, extend=extend, shrink=.85)
        first = results[0]["records"][0]
        fig.suptitle(f"{first['filename']} | {metric} | DC radius={first['dc_mask_radius']:g} bins | "
                     f"peak radius={first['peak_neighborhood_radius']:g} bins\n"
                     "Shared scale across block sizes; uncovered / undefined pixels excluded")
        fig.savefig(destination, dpi=150)
    finally:
        plt.close(fig)


def run_batch(args):
    metric = args.color_metric
    source, output = args.input_dir.resolve(), args.output_dir.resolve()
    if source == output:
        raise ValueError("Input and output directories must differ")
    paths = sorted((p for p in source.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS), key=lambda p: p.name)
    if args.filenames:
        missing = set(args.filenames) - {p.name for p in paths}
        if missing:
            raise ValueError(f"Files not found: {sorted(missing)}")
        paths = [p for p in paths if p.name in args.filenames]
    if not paths:
        raise ValueError("No supported input images")
    output.mkdir(parents=True, exist_ok=True)
    manual = json.loads(args.reference_json.read_text(encoding="utf-8-sig")) if args.reference_json else {}
    coordinates = REFERENCE_BLOCKS
    if args.reference_blocks:
        coordinates = [tuple(map(int, item.split(","))) for item in args.reference_blocks.split(";")]
    weights = dict(zip(WEIGHTS, args.weights))
    feature_options = {"dc_radius": args.dc_mask_radius, "zoom_radius": args.zoom_radius,
                       "max_peaks": args.max_peaks, "peak_percentile": args.peak_percentile,
                       "peak_radius": args.peak_radius}
    strides = args.strides or args.block_sizes
    if len(strides) == 1:
        strides = strides * len(args.block_sizes)
    report = {"input_directory": str(source), "color_metric": metric,
              "color_map": metric_style(metric)[0], "ranking_order": "ascending" if metric == "peak_energy_ratio" else "descending",
              "reference_mode": args.reference_mode, "weights": weights,
              "feature_options": feature_options, "images": [], "errors": [],
              "note": "Exploratory ranking, not a defect classifier. DC mask can suppress lattice fundamentals."}
    reference_template = {}
    all_features = []
    for index, path in enumerate(paths, start=1):
        try:
            gray = read_gray(path)
        except Exception as exc:
            report["errors"].append({"filename": path.name, "error": str(exc)})
            continue
        image_output = output / path.name
        image_output.mkdir(exist_ok=True)
        results = []
        reference_template[path.name] = {}
        for size, stride in zip(args.block_sizes, strides):
            try:
                refs = manual.get(path.name, {}).get(str(size), coordinates)
                records, reference = analyze_blocks(gray, path.name, size, stride, args.reference_mode,
                                                     refs, weights, **feature_options)
                results.append({"size": size, "stride": stride, "records": records, "reference": reference,
                                "heatmap": rasterize_scores(gray.shape, records, metric)})
                reference_template[path.name][str(size)] = []
                print(f"[{index}/{len(paths)}] {path.name}: block {size}, {len(records)} blocks; "
                      f"peaks {sum(r['feature_status'] == 'ok' for r in records)}/{len(records)}", flush=True)
            except Exception as exc:
                report["errors"].append({"filename": path.name, "block_size": size, "error": str(exc)})
                print(f"ERROR: {path.name} block {size}: {exc}", flush=True)
        if not results:
            continue
        scores = [r["anomaly_score"] for result in results for r in result["records"] if r["anomaly_score"] is not None]
        vmax = 1. if metric == "peak_energy_ratio" else (args.score_vmax or (max(float(np.percentile(scores, 99)), .1) if scores else 1.))
        for result in results:
            records, reference = result["records"], result["reference"]
            directory = image_output / f"block_{result['size']}_stride_{result['stride']}"
            directory.mkdir(exist_ok=True)
            write_csv(directory / "features.csv", records)
            # Raw numerical map preserves NaNs for excluded pixels and overlap averaging.
            prefix = "peak_energy_ratio" if metric == "peak_energy_ratio" else "anomaly"
            np.save(directory / ("peak_energy_ratio_map.npy" if metric == "peak_energy_ratio" else "anomaly_score_map.npy"), result["heatmap"])
            for kind, filename in [("grid", "block_grid.png"), ("heatmap", f"{prefix}_heatmap.png"),
                                   ("overlay", f"{prefix}_overlay.png")]:
                render_map(gray, records, result["heatmap"], kind, directory / filename, vmax, args.show_scores, metric)
            top = save_block_comparison(gray, records, reference, feature_options, args.top_k,
                                        directory / "top_blocks_fft_comparison.png", metric)
            with (directory / "top_blocks.csv").open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["rank", *records[0]])
                writer.writeheader()
                writer.writerows({"rank": i + 1, **r} for i, r in enumerate(top))
            summary = {"filename": path.name, "width": gray.shape[1], "height": gray.shape[0],
                       "block_size": result["size"], "stride": result["stride"], "block_count": len(records),
                       "reference": reference, "weights": weights, "feature_options": feature_options,
                       "color_metric": metric, "color_map": metric_style(metric)[0], "display_limits": [0, vmax],
                       "ranking_order": report["ranking_order"],
                       "displayed_block_count": sum(metric_value(r, metric) is not None for r in records),
                       "display_score_vmax": vmax, "score_count": sum(r["anomaly_score"] is not None for r in records),
                       "covered_width": max(r["x_start"] + r["block_size"] for r in records),
                       "covered_height": max(r["y_start"] + r["block_size"] for r in records),
                       "top_blocks": [{"row": r["block_row"], "col": r["block_col"], "score": r["anomaly_score"],
                                       "ranking_value": metric_value(r, metric), "peak_energy_ratio": r["peak_energy_ratio"]} for r in top]}
            (directory / "report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            report["images"].append(summary)
            all_features.extend(records)
        save_size_comparison(gray, results, image_output / f"{path.name}_block_size_comparison.png", vmax, metric)
    if all_features:
        write_csv(output / "all_block_features.csv", all_features)
    (output / "reference_blocks_template.json").write_text(json.dumps(reference_template, indent=2), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done: {len(report['images'])} image/size results, {len(report['errors'])} errors -> {output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data20260903" / "data20260903")
    parser.add_argument("--output-dir", type=Path, help="Default depends on color metric")
    parser.add_argument("--color-metric", choices=["peak_energy_ratio", "anomaly_score"], default=DEFAULT_COLOR_METRIC)
    parser.add_argument("--filenames", nargs="+")
    parser.add_argument("--block-sizes", type=int, nargs="+", default=BLOCK_SIZES)
    parser.add_argument("--strides", type=int, nargs="+", help="Default: each block size (no overlap)")
    parser.add_argument("--dc-mask-radius", type=float, default=DC_MASK_RADIUS)
    parser.add_argument("--zoom-radius", type=int, help="Default: block_size / 4")
    parser.add_argument("--max-peaks", type=int, default=8)
    parser.add_argument("--peak-percentile", type=float, default=95.)
    parser.add_argument("--peak-radius", type=float, default=2.)
    parser.add_argument("--reference-mode", choices=["median", "manual"], default="median")
    parser.add_argument("--reference-blocks", help='Zero-based coordinates, e.g. "2,3;2,4"')
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--weights", type=float, nargs=5, default=list(WEIGHTS.values()),
                        metavar=("POSITION", "ANGLE", "STRENGTH", "ENERGY", "SPREAD"))
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--show-scores", "--show-values", action="store_true", help="Print selected color metric on overlay")
    parser.add_argument("--score-vmax", type=float, help="Display upper limit only; no classification threshold")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = ROOT / "outputs" / ("local_fft_peak_energy_ratio" if args.color_metric == "peak_energy_ratio" else "local_fft_block_experiment")
    if args.color_metric == "peak_energy_ratio" and args.score_vmax is not None:
        parser.error("R always uses the fixed range 0..1; score-vmax applies only to anomaly_score")
    if any(s < 16 for s in args.block_sizes) or len(set(args.block_sizes)) != len(args.block_sizes):
        parser.error("Block sizes must be distinct integers >= 16")
    strides = args.strides or args.block_sizes
    if len(strides) not in (1, len(args.block_sizes)):
        parser.error("Specify one stride or one stride per block size")
    if len(strides) == 1:
        strides = strides * len(args.block_sizes)
    if any(not 1 <= stride <= size for size, stride in zip(args.block_sizes, strides)):
        parser.error("Require 1 <= stride <= block size")
    zooms = [min(args.zoom_radius if args.zoom_radius is not None else s // 4, s // 2) for s in args.block_sizes]
    if not 0 <= args.dc_mask_radius < min(zooms) or not np.isfinite(args.dc_mask_radius):
        parser.error("DC radius must be >= 0 and smaller than each zoom radius")
    if not 0 <= args.peak_percentile <= 100 or args.max_peaks < 2 or args.peak_radius <= 0 or not np.isfinite(args.peak_radius):
        parser.error("Invalid peak detector settings")
    if args.top_k < 1 or any(w < 0 or not np.isfinite(w) for w in args.weights) or not sum(args.weights) > 0:
        parser.error("Require top-k >= 1 and finite nonnegative weights, with at least one positive")
    if args.score_vmax is not None and (args.score_vmax <= 0 or not np.isfinite(args.score_vmax)):
        parser.error("score-vmax must be finite and positive")
    if args.reference_mode != "manual" and (args.reference_blocks or args.reference_json):
        parser.error("Use --reference-mode manual with reference block settings")
    if args.reference_blocks and args.reference_json:
        parser.error("Use either reference-blocks or reference-json")
    try:
        report = run_batch(args)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
