"""Background-versus-surface experiment using only luminance histograms.

Examples:
  python background_surface_hist_experiment.py --analyze
  python background_surface_hist_experiment.py --label image1.png
  python background_surface_hist_experiment.py --analyze --train-images image1,image2 --test-images image3,image4

The label tool saves block labels separately from the source Labelme JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

import fft_mask_ground_truth_evaluation as ground_truth
from block_spectral_viewer import DEFAULT_INPUT, image_number, read_gray


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "outputs" / "background_surface_hist_experiment"
LABEL_ROOT = ROOT / "outputs" / "block_background_surface_labels"
LIGHT_OFF = {"image1", "image19", "image20", "image21"}
VALID_LABELS = ("background", "normal", "mixed", "damage", "unlabeled")
LABEL_COLORS = {"normal": "#29a34a", "background": "#2b78e4", "mixed": "#f1c232",
                "damage": "#d62728", "unlabeled": "#888888", "unassigned": "#f1c232"}
AUTO_COLORS = {"surface": "#29a34a", "background": "#2b78e4", "hold": "#f1c232"}


def bounds(height: int, width: int, row: int, col: int, rows: int, cols: int):
    ys = np.linspace(0, height, rows + 1, dtype=int)
    xs = np.linspace(0, width, cols + 1, dtype=int)
    return int(xs[col]), int(ys[row]), int(xs[col + 1]), int(ys[row + 1])


def histogram(gray: np.ndarray) -> np.ndarray:
    return np.bincount(gray.astype(np.uint8).ravel(), minlength=256).astype(float) / gray.size


def intersection(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.minimum(a, b).sum())


def label_path(image_name: str) -> Path:
    return LABEL_ROOT / f"{Path(image_name).stem}_7x7_labels.json"


def load_labels(image_name: str, rows: int, cols: int) -> dict[str, str]:
    path = label_path(image_name)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("rows") != rows or data.get("cols") != cols:
        return {}
    return {str(k): v for k, v in data.get("labels", {}).items() if v in VALID_LABELS}


def auto_block_label(record: dict, x0: int, y0: int, x1: int, y1: int) -> str:
    """Conservative conversion from existing polygon annotations to block labels."""
    masks = record["masks"]
    area = (x1 - x0) * (y1 - y0)
    fractions = {name: float(mask[y0:y1, x0:x1].sum()) / area for name, mask in masks.items()}
    # Only a dominant annotation defines the semantic attribute of the block.
    if fractions["damage"] >= .80:
        return "damage"
    if fractions["tag"] >= .80 or fractions["background"] >= .80:
        return "background"
    if fractions["surface"] >= .80:
        return "normal"
    if any(value > 0 for value in fractions.values()):
        return "mixed"
    return "unlabeled"


def annotation_fractions(record: dict | None, x0: int, y0: int, x1: int, y1: int) -> dict[str, float]:
    if record is None:
        return {"tag_fraction": 0.0, "background_fraction": 0.0, "damage_fraction": 0.0}
    area = (x1 - x0) * (y1 - y0)
    return {f"{name}_fraction": float(mask[y0:y1, x0:x1].sum()) / area
            for name, mask in record["masks"].items() if name in {"tag", "background", "damage"}}


def extract_blocks(image_path: Path, rows: int, cols: int):
    gray = read_gray(image_path)
    record = None
    json_path = image_path.with_suffix(".json")
    if json_path.exists():
        record = ground_truth.load_record(json_path)
    manual = load_labels(image_path.name, rows, cols)
    features, hists = [], []
    for row in range(rows):
        for col in range(cols):
            x0, y0, x1, y1 = bounds(*gray.shape, row, col, rows, cols)
            block = gray[y0:y1, x0:x1]
            key = f"B{row + 1},{col + 1}"
            hist = histogram(block)
            auto = auto_block_label(record, x0, y0, x1, y1) if record else "unlabeled"
            label = manual.get(key, auto)
            fractions = annotation_fractions(record, x0, y0, x1, y1)
            q1, q3 = np.percentile(block, [25, 75])
            features.append({"block_id": key, "row": row + 1, "col": col + 1,
                             "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                             "pixel_count": block.size, "mean": float(block.mean()),
                             "std": float(block.std()), "hist_peak_probability": float(hist.max()),
                             "iqr": float(q3 - q1), "label_source": "manual" if key in manual else "annotation",
                             "ground_truth_label": label, **fractions})
            hists.append(hist)
    return gray, features, np.vstack(hists)


def write_csv(path: Path, rows: list[dict]):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def save_figure(fig, path: Path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def heatmap(values, rows, cols, title, path, fmt=".1f"):
    matrix = np.asarray(values).reshape(rows, cols)
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    im = ax.imshow(matrix, cmap="viridis")
    for r in range(rows):
        for c in range(cols): ax.text(c, r, format(matrix[r, c], fmt), ha="center", va="center", fontsize=7, color="white")
    ax.set(title=title, xlabel="column", ylabel="row", xticks=range(cols), yticks=range(rows))
    fig.colorbar(im, ax=ax)
    save_figure(fig, path)


def similarity_components(sim: np.ndarray, threshold: float) -> list[int]:
    n, group, next_id = len(sim), [-1] * len(sim), 0
    for start in range(n):
        if group[start] >= 0: continue
        stack, group[start] = [start], next_id
        while stack:
            i = stack.pop()
            for j in np.flatnonzero(sim[i] >= threshold):
                if group[j] < 0: group[j] = next_id; stack.append(int(j))
        next_id += 1
    return group


def best_threshold(scores, labels, positive_if_high=True):
    scores, labels = np.asarray(scores), np.asarray(labels)
    candidates = np.unique(scores)
    if len(candidates) == 0: return None
    best = None
    for threshold in candidates:
        pred = scores >= threshold if positive_if_high else scores < threshold
        tp = int(np.sum(pred & labels)); fp = int(np.sum(pred & ~labels)); fn = int(np.sum(~pred & labels))
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        item = (f1, float(threshold))
        if best is None or item > best: best = item
    return best[1]


def otsu_1d(values: np.ndarray) -> tuple[float, float]:
    """Otsu threshold and explained-variance ratio for arbitrary scalar data."""
    values = np.asarray(values, dtype=float)
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return lo, 0.0
    counts, edges = np.histogram(values, bins=256, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2
    weight0 = np.cumsum(counts).astype(float)
    weight1 = len(values) - weight0
    sum0 = np.cumsum(counts * centers)
    total = sum0[-1]
    mean0 = np.divide(sum0, weight0, out=np.zeros_like(sum0), where=weight0 > 0)
    mean1 = np.divide(total - sum0, weight1, out=np.zeros_like(sum0), where=weight1 > 0)
    between = weight0 * weight1 * (mean0 - mean1) ** 2
    index = int(np.argmax(between))
    total_variance = float(np.var(values) * len(values) ** 2)
    separation = float(between[index] / total_variance) if total_variance else 0.0
    return float(centers[index]), separation


def automatic_classification(features: list[dict], hists: np.ndarray) -> dict:
    """Label-free Otsu and histogram-cluster candidates for one image only."""
    std = np.asarray([row["std"] for row in features], dtype=float)
    threshold, separation = otsu_1d(std)
    # Keep blocks very close to the data-derived boundary as an explicit hold.
    margin = max(.02 * (std.max() - std.min()), .25)
    for row in features:
        delta = row["std"] - threshold
        row["std_otsu_threshold"] = threshold
        row["std_otsu_separation"] = separation
        row["auto_std"] = "hold" if abs(delta) <= margin else ("surface" if delta > 0 else "background")

    similarity = np.minimum(hists[:, None, :], hists[None, :, :]).sum(axis=2)
    np.fill_diagonal(similarity, 1.0)
    distance = np.clip(1.0 - similarity, 0.0, 1.0)
    tree = linkage(squareform(distance, checks=False), method="average") if len(features) > 1 else np.empty((0, 4))
    groups2 = fcluster(tree, t=2, criterion="maxclust") if len(features) > 1 else np.ones(1, dtype=int)
    groups3 = fcluster(tree, t=min(3, len(features)), criterion="maxclust") if len(features) > 1 else np.ones(1, dtype=int)
    widths = {int(group): float(np.mean(std[groups2 == group])) for group in np.unique(groups2)}
    narrow, broad = min(widths, key=widths.get), max(widths, key=widths.get)
    width_gap = widths[broad] - widths[narrow]
    # A 15%-of-observed-range gap is deliberately conservative: do not give
    # physical names to clusters that differ too little in this observation.
    cluster_clear = width_gap >= max(.15 * (std.max() - std.min()), .5)
    for index, row in enumerate(features):
        row["hist_cluster2"] = int(groups2[index])
        row["hist_cluster3"] = int(groups3[index])
        row["hist_cluster_width"] = widths[int(groups2[index])]
        row["auto_hist"] = ("surface" if groups2[index] == broad else "background") if cluster_clear else "hold"
        row["auto_consensus"] = row["auto_std"] if row["auto_std"] == row["auto_hist"] else "hold"
    eligible = [row for row in features if row["auto_std"] != "hold" and row["auto_hist"] != "hold"]
    agreement = float(np.mean([row["auto_std"] == row["auto_hist"] for row in eligible])) if eligible else 0.0
    return {"similarity": similarity, "distance": distance, "otsu_threshold": threshold,
            "otsu_separation": separation, "otsu_margin": margin, "cluster_widths": widths,
            "cluster_clear": cluster_clear, "cluster_width_gap": width_gap, "agreement": agreement}


def train_model(records):
    train = [row for record in records for row in record["features"] if row["ground_truth_label"] in ("normal", "background")]
    normal = [row for row in train if row["ground_truth_label"] == "normal"]
    background = [row for row in train if row["ground_truth_label"] == "background"]
    if not normal or not background:
        return None
    normal_hist = np.mean([row["hist"] for row in normal], axis=0)
    for row in train: row["train_similarity"] = intersection(row["hist"], normal_hist)
    labels = [row["ground_truth_label"] == "normal" for row in train]
    return {"std_threshold": best_threshold([r["std"] for r in train], labels),
            "similarity_threshold": best_threshold([r["train_similarity"] for r in train], labels),
            "normal_histogram": normal_hist.tolist(), "train_rows": len(train),
            "normal_rows": len(normal), "background_rows": len(background)}


def classify(features, model):
    if model is None:
        for row in features:
            row.update(pred_std="unassigned", pred_similarity="unassigned", pred_combined="unassigned", normal_similarity="")
        return
    normal_hist = np.asarray(model["normal_histogram"])
    for row in features:
        similarity = intersection(row["hist"], normal_hist)
        std_pred = "normal" if row["std"] >= model["std_threshold"] else "background"
        sim_pred = "normal" if similarity >= model["similarity_threshold"] else "background"
        # A disagreement is intentionally retained rather than forced into either class.
        combined = std_pred if std_pred == sim_pred else "mixed"
        row.update(pred_std=std_pred, pred_similarity=sim_pred, pred_combined=combined,
                   normal_similarity=similarity)


def metrics(features):
    eligible = [r for r in features if r["ground_truth_label"] in ("normal", "background")]
    if not eligible or all(r["pred_combined"] == "unassigned" for r in eligible):
        return [{"metric": "evaluation", "value": "not_available_without_independent_train_and_test_labels"}]
    y = np.array([r["ground_truth_label"] == "normal" for r in eligible])
    p = np.array([r["pred_combined"] == "normal" for r in eligible])
    held = np.array([r["pred_combined"] == "mixed" for r in eligible])
    tp, tn = int(np.sum(p & y)), int(np.sum(~p & ~y & ~held))
    fp, fn = int(np.sum(p & ~y)), int(np.sum(~p & y & ~held))
    evaluated = max(1, int((~held).sum()))
    return [{"metric": key, "value": value} for key, value in {
        "eligible_blocks": len(eligible), "held_out_blocks": int(held.sum()), "hold_rate": float(held.mean()),
        "accuracy": (tp + tn) / evaluated, "normal_precision": tp / max(1, tp + fp),
        "normal_recall": tp / max(1, tp + fn), "normal_f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn}.items()]


def render_overlay(gray, features, field, title, path):
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    for row in features:
        color = AUTO_COLORS[row[field]]
        ax.add_patch(plt.Rectangle((row["x0"], row["y0"]), row["x1"]-row["x0"], row["y1"]-row["y0"], fill=False, lw=2, edgecolor=color))
        ax.text(row["x0"]+2, row["y0"]+13, row["block_id"], color=color, fontsize=6, bbox={"facecolor":"black", "alpha":.55, "pad":1})
    ax.set(title=title); ax.axis("off"); save_figure(fig, path)


def render_outputs(image_name, gray, features, hists, rows, cols, out, auto):
    labels = [r["ground_truth_label"] for r in features]
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    for label in sorted(set(labels)):
        indices = [i for i, value in enumerate(labels) if value == label]
        ax.scatter([features[i]["mean"] for i in indices], [features[i]["std"] for i in indices],
                   label=label, c=LABEL_COLORS.get(label, "#888888"), s=42)
    ax.set(title=f"{image_name}: mean vs standard deviation", xlabel="mean brightness", ylabel="std brightness")
    ax.legend(fontsize=8); ax.grid(alpha=.25); save_figure(fig, out / "feature_scatter.png")
    heatmap([r["std"] for r in features], rows, cols, "Std brightness per block", out / "std_heatmap.png")
    heatmap([r["mean"] for r in features], rows, cols, "Mean brightness per block", out / "mean_heatmap.png")
    sim = auto["similarity"]
    ids = [r["block_id"] for r in features]
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(sim, vmin=0, vmax=1, cmap="magma")
    ax.set(title=f"{image_name}: histogram intersection", xlabel="block", ylabel="block")
    step = max(1, len(ids) // 12); ax.set_xticks(range(0, len(ids), step), ids[::step], rotation=90, fontsize=7); ax.set_yticks(range(0, len(ids), step), ids[::step], fontsize=7)
    fig.colorbar(im, ax=ax, label="intersection similarity"); save_figure(fig, out / "similarity_heatmap.png")
    write_csv(out / "histogram_similarity.csv", [{"block_a": ids[i], "block_b": ids[j], "similarity": float(sim[i, j])} for i in range(len(ids)) for j in range(len(ids))])
    component_threshold = float(np.quantile(sim[np.triu_indices_from(sim, 1)], .80)) if len(ids) > 1 else 1
    components = similarity_components(sim, component_threshold)
    for row, component in zip(features, components): row["similarity_component"] = component; row["component_threshold"] = component_threshold
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    full = histogram(gray)
    axes[0].plot(full, color="black", label="full image")
    for label in ("normal", "background"):
        pick = [i for i, value in enumerate(labels) if value == label]
        if pick: axes[0].plot(np.mean(hists[pick], axis=0), color=LABEL_COLORS[label], label=f"{label} mean")
    axes[0].set(title="Full image and labeled-class histograms", xlabel="brightness", ylabel="probability"); axes[0].legend(); axes[0].grid(alpha=.25)
    grouped = {"normal-normal": [], "background-background": [], "normal-background": []}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            pair = {labels[i], labels[j]}
            key = "normal-normal" if pair == {"normal"} else "background-background" if pair == {"background"} else "normal-background" if pair == {"normal", "background"} else None
            if key: grouped[key].append(sim[i, j])
    axes[1].boxplot([value or [np.nan] for value in grouped.values()], tick_labels=list(grouped)); axes[1].set(title="Labeled similarity distributions", ylabel="intersection similarity", ylim=(0, 1)); axes[1].tick_params(axis="x", rotation=20)
    save_figure(fig, out / "histogram_comparison.png")
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.hist([row["std"] for row in features], bins=min(14, len(features)), color="#777777", edgecolor="white")
    ax.axvline(auto["otsu_threshold"], color="#d62728", lw=2, label=f"Otsu={auto['otsu_threshold']:.2f}")
    ax.axvspan(auto["otsu_threshold"]-auto["otsu_margin"], auto["otsu_threshold"]+auto["otsu_margin"], color="#f1c232", alpha=.3, label="hold band")
    ax.set(title="Std distribution and automatic threshold", xlabel="block brightness std", ylabel="block count"); ax.legend(); save_figure(fig, out / "std_distribution.png")
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for group in sorted(set(row["hist_cluster2"] for row in features)):
        selected = [i for i, row in enumerate(features) if row["hist_cluster2"] == group]
        width = np.mean([features[i]["std"] for i in selected])
        ax.plot(np.mean(hists[selected], axis=0), label=f"cluster {group}, n={len(selected)}, mean std={width:.2f}")
    ax.set(title="Histogram-cluster representative distributions", xlabel="brightness", ylabel="probability"); ax.legend(fontsize=8); ax.grid(alpha=.25); save_figure(fig, out / "cluster_histograms.png")
    render_overlay(gray, features, "auto_std", "Std/Otsu: green=surface candidate, blue=background candidate, yellow=hold", out / "std_classification_overlay.png")
    render_overlay(gray, features, "auto_hist", "Histogram clusters: green=surface candidate, blue=background candidate, yellow=hold", out / "histogram_classification_overlay.png")
    render_overlay(gray, features, "auto_consensus", "Consensus: green=surface candidate, blue=background candidate, yellow=hold", out / "classification_overlay.png")


def report(image_name, features, model, auto, out):
    counts = Counter(row["ground_truth_label"] for row in features)
    lines = [f"# Background and normal surface experiment: {image_name}", "", "## Objective", "Separate background from normal surface using brightness-distribution statistics only.", "", "## Evidence", f"- block labels: `{dict(counts)}`", f"- std range: {min(r['std'] for r in features):.3f} to {max(r['std'] for r in features):.3f}", f"- mean range: {min(r['mean'] for r in features):.3f} to {max(r['mean'] for r in features):.3f}"]
    auto_counts = Counter(row["auto_consensus"] for row in features)
    lines += ["", "## Label-free automatic classification", f"- Otsu std threshold: {auto['otsu_threshold']:.4f}; separation ratio: {auto['otsu_separation']:.3f}; hold band: +/- {auto['otsu_margin']:.3f}.", f"- Histogram 2-cluster mean std: {auto['cluster_widths']}; width gap: {auto['cluster_width_gap']:.3f}; mapping clear: {auto['cluster_clear']}.", f"- Otsu versus histogram agreement (non-hold blocks): {auto['agreement']:.3f}.", f"- Consensus counts: `{dict(auto_counts)}`."]
    if model is not None: lines += ["", "## Optional supervised validation", f"Independent training model available: std >= {model['std_threshold']:.4f}; histogram intersection >= {model['similarity_threshold']:.4f}."]
    lines += ["", "## Interpretation", "Green means a surface candidate, not confirmed defect-free normal surface. Yellow means that the statistical evidence is ambiguous or the two simple methods disagree. Existing annotations are used only for later validation, never for automatic classification.", "", "## Files", "- classification_results.csv: automatic Otsu, histogram-cluster, and consensus results", "- histogram_similarity.csv: all pairwise intersections", "- std_distribution.png and cluster_histograms.png", "- std_classification_overlay.png and histogram_classification_overlay.png", "- classification_overlay.png: consensus", "", "## Caveat", "A two-cluster partition is an image-local hypothesis. The full-image histogram is not used as a surface reference."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(args):
    paths = sorted(args.input_dir.glob("image[0-9]*.png"), key=image_number)
    selected = set(args.images.split(",")) if args.images else None
    paths = [p for p in paths if selected is None or p.stem in selected]
    raw = []
    for path in paths:
        gray, features, hists = extract_blocks(path, args.rows, args.cols)
        for feature, hist in zip(features, hists): feature["hist"] = hist
        raw.append({"path": path, "gray": gray, "features": features, "hists": hists})
    train_ids = set(filter(None, args.train_images.split(",")))
    test_ids = set(filter(None, args.test_images.split(",")))
    if bool(train_ids) != bool(test_ids) or train_ids & test_ids:
        raise ValueError("Specify disjoint --train-images and --test-images together, or neither.")
    model = train_model([record for record in raw if record["path"].stem in train_ids]) if train_ids else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if model: (args.output_dir / "trained_rules.json").write_text(json.dumps({k:v for k,v in model.items() if k != "normal_histogram"}, indent=2), encoding="utf-8")
    for record in raw:
        apply_model = model if (not test_ids or record["path"].stem in test_ids) else None
        auto = automatic_classification(record["features"], record["hists"])
        classify(record["features"], apply_model)
        out = args.output_dir / record["path"].stem; out.mkdir(parents=True, exist_ok=True)
        render_outputs(record["path"].stem, record["gray"], record["features"], record["hists"], args.rows, args.cols, out, auto)
        saved = [{k:v for k,v in row.items() if k != "hist"} for row in record["features"]]
        write_csv(out / "block_features.csv", saved)
        write_csv(out / "classification_results.csv", saved)
        write_csv(out / "classification_metrics.csv", metrics(record["features"]))
        report(record["path"].stem, record["features"], apply_model, auto, out)
        print(f"saved {out}")


def label_image(args):
    # A small dedicated interactive tool: labels are written separately from the source data.
    matplotlib.use("TkAgg", force=True)
    import matplotlib.pyplot as interactive
    path = args.label.resolve(); gray = read_gray(path); labels = load_labels(path.name, args.rows, args.cols); current = [None]
    fig, ax = interactive.subplots(figsize=(11, 8)); ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    def redraw():
        ax.clear(); ax.imshow(gray, cmap="gray", vmin=0, vmax=255); h,w=gray.shape
        for r in range(args.rows + 1): ax.axhline(r*h/args.rows, color="white", lw=.7)
        for c in range(args.cols + 1): ax.axvline(c*w/args.cols, color="white", lw=.7)
        for r in range(args.rows):
            for c in range(args.cols):
                key=f"B{r+1},{c+1}"; x0,y0,x1,y1=bounds(h,w,r,c,args.rows,args.cols); value=labels.get(key,"unlabeled")
                ax.add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,lw=2,edgecolor=LABEL_COLORS[value]))
                ax.text(x0+3,y0+14,key+" "+value[0].upper(),color=LABEL_COLORS[value],fontsize=7,bbox={"facecolor":"black","alpha":.6,"pad":1})
        ax.set_title("click block; N normal, B background, M mixed, D damage, U unlabeled, S save"); ax.axis("off"); fig.canvas.draw_idle()
    def click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None: return
        current[0]=(min(args.rows-1,int(event.ydata)*args.rows//gray.shape[0]),min(args.cols-1,int(event.xdata)*args.cols//gray.shape[1]))
    def key(event):
        if event.key == "s":
            target=label_path(path.name); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps({"image":path.name,"rows":args.rows,"cols":args.cols,"labels":labels},indent=2),encoding="utf-8"); print(f"saved {target}"); return
        if current[0] is None or event.key not in {"n","b","m","d","u"}: return
        value={"n":"normal","b":"background","m":"mixed","d":"damage","u":"unlabeled"}[event.key]; r,c=current[0]; labels[f"B{r+1},{c+1}"]=value; redraw()
    fig.canvas.mpl_connect("button_press_event",click); fig.canvas.mpl_connect("key_press_event",key); redraw(); interactive.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze", action="store_true"); parser.add_argument("--label", type=Path)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rows", type=int, default=7); parser.add_argument("--cols", type=int, default=7)
    parser.add_argument("--images", default=""); parser.add_argument("--train-images", default=""); parser.add_argument("--test-images", default="")
    args=parser.parse_args()
    if args.label: label_image(args)
    elif args.analyze: analyze(args)
    else: parser.error("choose --analyze or --label IMAGE")


if __name__ == "__main__": main()
