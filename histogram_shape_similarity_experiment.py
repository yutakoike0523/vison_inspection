"""Label-free block surface-candidate extraction from histogram shape.

This is an additive experiment. It does not alter the existing Otsu/FFT tools.
Run: python histogram_shape_similarity_experiment.py
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import wasserstein_distance

from background_surface_hist_experiment import (
    DEFAULT_INPUT, ROOT, bounds, extract_blocks, image_number, save_figure, write_csv,
)


DEFAULT_OUT = ROOT / "outputs" / "histogram_shape_similarity_experiment"
COLORS = matplotlib.colormaps.get_cmap("tab20").resampled(20)
CLASS_COLORS = {"surface_candidate": "#ff8c00", "tag_background_attribute": "#2b78e4",
                "tag_or_damage_attribute": "#d62728", "other_unlabeled": "#2b78e4", "hold": "#f1c232"}
THRESHOLDS = (0.80, 0.85, 0.90)


def normalize_without_255(hists: np.ndarray):
    nonwhite = hists[:, :255].sum(axis=1)
    safe = np.maximum(nonwhite, 1e-12)
    return hists[:, :255] / safe[:, None], nonwhite


def intersection_matrix(hists: np.ndarray) -> np.ndarray:
    result = np.minimum(hists[:, None, :], hists[None, :, :]).sum(axis=2)
    np.fill_diagonal(result, 1.0)
    return result


def wasserstein_matrix(hists: np.ndarray) -> np.ndarray:
    bins = np.arange(hists.shape[1], dtype=float)
    result = np.zeros((len(hists), len(hists)), dtype=float)
    for i in range(len(hists)):
        for j in range(i):
            result[i, j] = result[j, i] = wasserstein_distance(bins, bins, hists[i], hists[j])
    return result


def draw_matrix(matrix, ids, title, path, cmap, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    step = max(1, len(ids) // 12)
    ax.set(title=title, xlabel="block", ylabel="block")
    ax.set_xticks(range(0, len(ids), step), ids[::step], rotation=90, fontsize=7)
    ax.set_yticks(range(0, len(ids), step), ids[::step], fontsize=7)
    fig.colorbar(im, ax=ax)
    save_figure(fig, path)


def component_edges(indices, features, rows, cols):
    selected = {(features[i]["row"], features[i]["col"]) for i in indices}
    edges = sum((r + 1, c) in selected for r, c in selected) + sum((r, c + 1) in selected for r, c in selected)
    # A connected compact group scores one; isolated blocks score zero.
    return float(min(1.0, edges / max(1, len(selected) - 1))) if len(selected) > 1 else 0.0


def group_summary(groups, similarity, full_hists, no255_hists, features, rows, cols):
    summaries = []
    for group in sorted(set(groups)):
        indices = np.flatnonzero(groups == group)
        pair = similarity[np.ix_(indices, indices)]
        values = pair[np.triu_indices(len(indices), 1)] if len(indices) > 1 else np.array([1.0])
        representative = no255_hists[indices].mean(axis=0)
        rep_total = representative.sum()
        if rep_total > 1e-12:
            representative /= rep_total
        rep_sim = np.minimum(no255_hists, representative).sum(axis=1)
        white = full_hists[indices, 255] if full_hists.shape[1] > 255 else np.zeros(len(indices))
        summaries.append({"cluster": int(group), "block_count": len(indices),
                          "block_ids": ",".join(features[i]["block_id"] for i in indices),
                          "mean_brightness": float(np.mean([features[i]["mean"] for i in indices])),
                          "mean_iqr": float(np.mean([features[i]["iqr"] for i in indices])),
                          "mean_std": float(np.mean([features[i]["std"] for i in indices])),
                          "mean_white_fraction": float(np.mean(white)),
                          "mean_nonwhite_fraction": float(np.mean(1-white)),
                          "within_similarity_mean": float(values.mean()),
                          "within_similarity_min": float(values.min()),
                          "representative_similarity_mean": float(rep_sim[indices].mean()),
                          "spatial_compactness": component_edges(indices, features, rows, cols),
                          "indices": indices, "representative": representative, "rep_similarity": rep_sim})
    return summaries


def choose_threshold(similarity, hists, features, rows, cols):
    """Try documented fixed candidates; select compact, internally similar groups.

    Complete linkage is used specifically to prevent chaining dissimilar blocks
    into one group. The selected threshold is a diagnostic choice, not learned.
    """
    dist = np.clip(1 - similarity, 0, 1)
    tree = linkage(squareform(dist, checks=False), method="complete") if len(similarity) > 1 else np.empty((0, 4))
    trials = []
    for threshold in THRESHOLDS:
        group = fcluster(tree, t=1-threshold, criterion="distance") if len(similarity) > 1 else np.ones(1, dtype=int)
        summary = group_summary(group, similarity, hists, hists, features, rows, cols)
        supported = [x for x in summary if x["block_count"] >= 3]
        coverage = sum(x["block_count"] for x in supported) / len(features)
        # Favor support by multi-block groups, then fewer fragmented groups.
        score = coverage - .04 * len(summary)
        trials.append({"threshold": threshold, "groups": group, "summary": summary, "score": score,
                       "group_count": len(summary), "supported_blocks": int(sum(x["block_count"] for x in supported))})
    # Prefer the most discriminating candidate only after maximizing support.
    return max(trials, key=lambda x: (x["score"], x["threshold"])), trials


def select_surface_group(summary):
    """Choose only a credible compact, shape-consistent group, otherwise hold."""
    candidates = [x for x in summary if x["block_count"] >= 3 and x["within_similarity_min"] >= .80]
    if not candidates:
        return None, "No cluster has at least three blocks with minimum internal similarity >= 0.80."
    width_values = np.array([x["mean_iqr"] for x in candidates])
    width_scale = max(float(width_values.max() - width_values.min()), 1.0)
    for item in candidates:
        # This uses shape consistency, spatial continuity, and distribution width;
        # it deliberately has no term for group size alone.
        item["surface_candidate_score"] = (item["within_similarity_mean"] + .12 * item["spatial_compactness"] +
                                           .08 * ((item["mean_iqr"] - width_values.min()) / width_scale))
    ranked = sorted(candidates, key=lambda x: x["surface_candidate_score"], reverse=True)
    if len(ranked) > 1 and ranked[0]["surface_candidate_score"] - ranked[1]["surface_candidate_score"] < .025:
        return None, "Top two shape-consistent groups have too similar a selection score."
    selected = ranked[0]
    reason = (f"cluster {selected['cluster']}: n={selected['block_count']}, "
              f"within mean/min={selected['within_similarity_mean']:.3f}/{selected['within_similarity_min']:.3f}, "
              f"IQR={selected['mean_iqr']:.2f}, compactness={selected['spatial_compactness']:.2f}")
    return selected, reason


def classify_from_representative(features, representative, member_similarity, all_hists):
    similarity = np.minimum(all_hists, representative).sum(axis=1)
    # The threshold is derived from the selected group itself. A separate hold
    # band avoids treating every nonmatching block as definite background.
    surface_threshold = float(np.percentile(member_similarity, 10) - .015)
    background_threshold = max(0.0, surface_threshold - .08)
    labels = np.full(len(features), "hold", dtype=object)
    labels[similarity >= surface_threshold] = "surface"
    labels[similarity < background_threshold] = "background"
    for row, score, label in zip(features, similarity, labels):
        row["surface_representative_similarity"] = float(score)
        row["surface_similarity_threshold"] = surface_threshold
        row["background_similarity_threshold"] = background_threshold
        row["shape_classification"] = label
    return similarity, surface_threshold, background_threshold


def apply_annotation_exclusion(features):
    """Create display classes from the requested annotation-coverage rule.

    This does not enter clustering or representative selection: the automatic
    histogram result remains in ``automatic_shape_classification``.
    """
    for row in features:
        row["automatic_shape_classification"] = row["shape_classification"]
        # Damage takes precedence even for a small overlap. Tag/background
        # requires a majority of the block; every remaining block is surface.
        if row.get("damage_fraction", 0.0) > 0.0:
            row["display_classification"] = "tag_or_damage_attribute"
        elif row.get("tag_fraction", 0.0) >= .50 or row.get("background_fraction", 0.0) >= .50:
            row["display_classification"] = "tag_background_attribute"
        else:
            row["display_classification"] = "surface_candidate"
        if row.get("damage_fraction", 0.0) > 0.0:
            row["display_classification_any_tag_background"] = "tag_or_damage_attribute"
        elif row.get("tag_fraction", 0.0) > 0.0 or row.get("background_fraction", 0.0) > 0.0:
            row["display_classification_any_tag_background"] = "tag_background_attribute"
        else:
            row["display_classification_any_tag_background"] = "surface_candidate"


def cluster_overlay(gray, features, groups, path):
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True); ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    for row, group in zip(features, groups):
        color = COLORS((int(group)-1) % 20)
        ax.add_patch(plt.Rectangle((row["x0"], row["y0"]), row["x1"]-row["x0"], row["y1"]-row["y0"], fill=False, lw=2, edgecolor=color))
        ax.text(row["x0"]+2, row["y0"]+13, f"G{group}", color=color, fontsize=7, bbox={"facecolor":"black", "alpha":.6,"pad":1})
    ax.set(title="Histogram-shape clusters (numbers have no semantic label)"); ax.axis("off"); save_figure(fig, path)


def classification_overlay(gray, features, path, field="display_classification", title=None):
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True); ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    for row in features:
        label = row[field]; color = CLASS_COLORS[label]
        ax.add_patch(plt.Rectangle((row["x0"], row["y0"]), row["x1"]-row["x0"], row["y1"]-row["y0"], fill=False, lw=2, edgecolor=color))
        ax.text(row["x0"]+2, row["y0"]+13, row["block_id"], color=color, fontsize=6, bbox={"facecolor":"black", "alpha":.55,"pad":1})
    ax.set(title=title or "Orange=surface; blue=tag/background coverage >=50%; red=any damage overlap"); ax.axis("off"); save_figure(fig, path)


def surface_vs_other_histograms(features, no255_hists, image_name, path, field="display_classification", rule_label="tag/background >=50%"):
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    order = ("surface_candidate", "tag_background_attribute", "tag_or_damage_attribute", "other_unlabeled", "hold")
    labels = {"surface_candidate": "surface candidate", "tag_background_attribute": "tag / background attribute",
              "tag_or_damage_attribute": "tag / damage / mixed attribute", "other_unlabeled": "other unlabeled", "hold": "hold"}
    for category in order:
        indices = [i for i, row in enumerate(features) if row[field] == category]
        if indices:
            # Every visible curve is one block; no average curve is drawn.
            for first, index in enumerate(indices):
                ax.plot(no255_hists[index], color=CLASS_COLORS[category], lw=.8,
                        label=f"{labels[category]} (n={len(indices)})" if first == 0 else None)
    ax.set(title=f"{image_name}: individual block histograms ({rule_label}; 0-254)", xlabel="brightness", ylabel="probability")
    ax.legend(fontsize=7); ax.grid(alpha=.25); save_figure(fig, path)


def hist_figure(summary, path):
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for item in summary:
        ax.plot(item["representative"], color=COLORS((item["cluster"]-1)%20),
                label=f"G{item['cluster']} n={item['block_count']} minS={item['within_similarity_min']:.2f}")
    ax.set(title="Cluster representative histograms (0-254, renormalized)", xlabel="brightness", ylabel="probability")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.25); save_figure(fig, path)


def surface_similarity_heatmap(scores, rows, cols, path):
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    matrix = scores.reshape(rows, cols); im=ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    for r in range(rows):
        for c in range(cols): ax.text(c,r,f"{matrix[r,c]:.2f}",ha="center",va="center",fontsize=7,color="white")
    ax.set(title="Similarity to selected surface representative", xlabel="column", ylabel="row", xticks=range(cols), yticks=range(rows)); fig.colorbar(im, ax=ax)
    save_figure(fig, path)


def render_report(out, image_name, trial, trials, selected, reason, features, white_fraction):
    labels = Counter(row["shape_classification"] for row in features)
    display_labels = Counter(row["display_classification"] for row in features)
    lines = [f"# Histogram shape comparison: {image_name}", "", "## Scope", "This run is label-free. The whole-image histogram was not used as a surface reference.", "", "## Method comparison", "| similarity threshold | cluster count | blocks in clusters n>=3 | selection score |", "|---:|---:|---:|---:|"]
    lines += [f"| {x['threshold']:.2f} | {x['group_count']} | {x['supported_blocks']} | {x['score']:.3f} |" for x in trials]
    lines += ["", f"Selected complete-linkage threshold: `{trial['threshold']:.2f}`. Complete linkage is used to avoid chaining dissimilar blocks.", "", "## Surface-candidate decision", reason]
    if selected is None:
        lines += ["No representative was selected; every block is held. This is a result, not an error."]
    else:
        highest = sorted(features, key=lambda row: row["surface_representative_similarity"], reverse=True)[:8]
        lowest = sorted(features, key=lambda row: row["surface_representative_similarity"])[:8]
        lines += [f"Automatic counts before annotation display: `{dict(labels)}`.", f"Display counts: `{dict(display_labels)}`. Any damage overlap is red; tag/background coverage >=50% is blue; every other block is orange surface. This display rule is separate from the automatic histogram decision.", "", "Highest similarity blocks: " + ", ".join(f"{r['block_id']}={r['surface_representative_similarity']:.3f}" for r in highest), "", "Lowest similarity blocks: " + ", ".join(f"{r['block_id']}={r['surface_representative_similarity']:.3f}" for r in lowest), "", f"White (255) fraction: min={white_fraction.min():.3f}, median={np.median(white_fraction):.3f}, max={white_fraction.max():.3f}. It is recorded separately and never treated as an automatic background label."]
    lines += ["", "## Interpretation", "Full-range intersection, 0-254 intersection, 255 fraction difference, and Wasserstein distance are exported separately. Their agreement or disagreement must be inspected before combining them; no fixed weighted composite is claimed."]
    (out / "comparison_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def analyze_one(path, args):
    gray, features, full_hists = extract_blocks(path, args.rows, args.cols)
    ids = [row["block_id"] for row in features]
    no255, nonwhite = normalize_without_255(full_hists)
    for row, fraction in zip(features, nonwhite):
        row["nonwhite_distribution_reliable"] = bool(fraction >= .05)
    sim_full, sim_no255 = intersection_matrix(full_hists), intersection_matrix(no255)
    wasserstein = wasserstein_matrix(full_hists)
    out = args.output_dir / path.stem; out.mkdir(parents=True, exist_ok=True)
    draw_matrix(sim_full, ids, "Histogram intersection: brightness 0-255", out / "similarity_full.png", "magma", 0, 1)
    draw_matrix(sim_no255, ids, "Histogram intersection: brightness 0-254 only", out / "similarity_without255.png", "magma", 0, 1)
    draw_matrix(wasserstein, ids, "Wasserstein distance: brightness 0-255", out / "wasserstein_distance.png", "viridis")
    trial, trials = choose_threshold(sim_no255, no255, features, args.rows, args.cols)
    summary = group_summary(trial["groups"], sim_no255, full_hists, no255, features, args.rows, args.cols)
    selected, reason = select_surface_group(summary)
    for item in summary:
        item["selected_surface_group"] = bool(selected and item["cluster"] == selected["cluster"])
    cluster_overlay(gray, features, trial["groups"], out / "cluster_overlay.png")
    hist_figure(summary, out / "cluster_histograms.png")
    if selected:
        scores, _, _ = classify_from_representative(features, selected["representative"], selected["rep_similarity"][selected["indices"]], no255)
    else:
        scores = np.zeros(len(features))
        for row in features:
            row.update(surface_representative_similarity="", surface_similarity_threshold="", background_similarity_threshold="", shape_classification="hold")
    apply_annotation_exclusion(features)
    surface_similarity_heatmap(scores, args.rows, args.cols, out / "surface_similarity.png")
    classification_overlay(gray, features, out / "classification_overlay.png")
    surface_vs_other_histograms(features, no255, path.stem, out / "surface_vs_other_histograms.png")
    classification_overlay(
        gray, features, out / "classification_overlay_any_tag_background.png",
        field="display_classification_any_tag_background",
        title="Orange=surface; blue=any tag/background overlap; red=any damage overlap",
    )
    surface_vs_other_histograms(
        features, no255, path.stem, out / "surface_vs_other_histograms_any_tag_background.png",
        field="display_classification_any_tag_background", rule_label="any tag/background overlap",
    )
    # Pairwise table preserves the three independent shape measurements.
    pair_rows=[]
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            pair_rows.append({"block_a":ids[i],"block_b":ids[j],"intersection_full":float(sim_full[i,j]),"intersection_without255":float(sim_no255[i,j]),"white_fraction_difference":float(abs(full_hists[i,255]-full_hists[j,255])),"wasserstein_distance":float(wasserstein[i,j])})
    write_csv(out / "pairwise_shape_comparison.csv", pair_rows)
    clean_summary = [{k:v for k,v in item.items() if k not in {"indices","representative","rep_similarity"}} for item in summary]
    write_csv(out / "cluster_summary.csv", clean_summary)
    result=[]
    for index,row in enumerate(features):
        result.append({k:v for k,v in row.items() if k not in {"hist"}} | {"white_fraction":float(full_hists[index,255]),"nonwhite_fraction":float(nonwhite[index]),"cluster":int(trial["groups"][index]),"intersection_full_to_cluster_rep":float(np.minimum(full_hists[index], full_hists[trial["groups"]==trial["groups"][index]].mean(axis=0)).sum())})
    write_csv(out / "classification_results.csv", result)
    render_report(out, path.stem, trial, trials, selected, reason, features, full_hists[:,255])
    return {"image":path.stem,"selected":None if selected is None else int(selected["cluster"]),"reason":reason,"counts":dict(Counter(row["display_classification"] for row in features)),"threshold":trial["threshold"]}


def make_gallery(output_dir: Path, filenames: tuple[str, str]):
    """Create compact all-image views without changing the per-image plots."""
    for filename in filenames:
        sources = sorted(output_dir.glob(f"image*/{filename}"), key=lambda p: image_number(p.parent / (p.parent.name + ".png")))
        if not sources:
            continue
        thumb_w, thumb_h, cols = 360, 190, 3
        rows = int(np.ceil(len(sources) / cols))
        canvas = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "white")
        for index, source in enumerate(sources):
            with Image.open(source) as graph:
                graph = graph.convert("RGB"); graph.thumbnail((thumb_w, thumb_h - 20), Image.Resampling.LANCZOS)
                x, y = (index % cols) * thumb_w, (index // cols) * thumb_h + 20
                canvas.paste(graph, (x + (thumb_w - graph.width) // 2, y))
            ImageDraw.Draw(canvas).text((x + 6, (index // cols) * thumb_h + 3), source.parent.name, fill="black")
        canvas.save(output_dir / f"all_images_{filename}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",type=Path,default=DEFAULT_INPUT); parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--rows",type=int,default=7); parser.add_argument("--cols",type=int,default=7); parser.add_argument("--images",default="")
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    wanted=set(filter(None,args.images.split(","))); paths=sorted(args.input_dir.glob("image[0-9]*.png"),key=image_number); paths=[p for p in paths if not wanted or p.stem in wanted]
    prior = {}
    summary_path = args.output_dir / "experiment_summary.csv"
    if summary_path.exists():
        with summary_path.open(encoding="utf-8-sig", newline="") as stream:
            prior = {row["image"]: row for row in csv.DictReader(stream)}
    summaries=[analyze_one(path,args) for path in paths]
    prior.update({row["image"]: row for row in summaries})
    all_summaries = [prior[key] for key in sorted(prior, key=lambda name: image_number(Path(name + ".png")))]
    write_csv(summary_path, all_summaries)
    lines=["# Histogram-shape extraction: all-image summary", "", "This summary is label-free; candidate names are hypotheses, not confirmed normal labels.", "", "| image | selected cluster | threshold | surface/background/hold |", "|---|---:|---:|---|"]
    lines += [f"| {row['image']} | {row['selected']} | {float(row['threshold']):.2f} | {row['counts']} |" for row in all_summaries]
    (args.output_dir / "overall_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    make_gallery(args.output_dir, ("cluster_histograms.png", "surface_vs_other_histograms.png",
                                   "surface_vs_other_histograms_any_tag_background.png",
                                   "classification_overlay_any_tag_background.png"))
    print(f"completed {len(summaries)} images; summary contains {len(all_summaries)} images in {args.output_dir}")


if __name__ == "__main__": main()
