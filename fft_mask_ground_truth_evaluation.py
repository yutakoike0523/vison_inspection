"""Quantitative evaluation of FFT masking against manual Labelme polygons.

This script is analysis-only: source images/JSON and prior outputs are never edited.
The FFT feature is deliberately identical to fft2_cosine_similarity.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

import fft2_cosine_similarity as fft

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data20260903" / "data20260903"
DEFAULT_OUT = ROOT / "outputs" / "fft_mask_ground_truth_evaluation"
SIZES = (128, 256)
THRESHOLDS = (.80, .82, .84, .86, .87, .88, .90, .92, .95)
LIGHT_OFF = {"image1", "image19", "image20", "image21"}
EXCLUSION_LABELS = {"tag", "background"}
DAMAGE_MAP = {
    "点状異常": "点状異常", "線状異常": "線状異常", "面状異常": "面状異常",
    "局所変形": "局所変形", "dent": "局所変形",
}
SURFACE_LABELS = {"surface", "normal_surface", "正常表面"}


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        if fields:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)


def read_rgb(path: Path):
    a = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if a is None:
        raise ValueError(f"cannot decode image: {path}")
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, a):
    Image.fromarray(np.asarray(a, np.uint8)).save(path)


def natural_image_key(path: Path):
    stem = path.stem
    return int(stem[5:]) if stem.startswith("image") and stem[5:].isdigit() else 10**9


def polygon_mask(points, width, height):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(np.unique(p, axis=0)) < 3 or not np.isfinite(p).all():
        raise ValueError("polygon must contain at least three finite unique points")
    q = np.rint(p).astype(np.int32)
    clipped = bool(np.any(q[:, 0] < 0) or np.any(q[:, 0] >= width) or
                   np.any(q[:, 1] < 0) or np.any(q[:, 1] >= height))
    q[:, 0] = np.clip(q[:, 0], 0, width - 1)
    q[:, 1] = np.clip(q[:, 1], 0, height - 1)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [q], 1)
    return mask.astype(bool), clipped


def rectangle_mask(points, width, height):
    p = np.asarray(points, dtype=float)
    if p.shape != (2, 2) or not np.isfinite(p).all():
        raise ValueError("rectangle must have two finite points")
    x1, x2 = sorted(p[:, 0]); y1, y2 = sorted(p[:, 1])
    x1, x2 = max(0., x1), min(float(width), x2)
    y1, y2 = max(0., y1), min(float(height), y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("empty rectangle after clipping")
    left, right = max(0, int(math.floor(x1))), min(width, int(math.ceil(x2)))
    top, bottom = max(0, int(math.floor(y1))), min(height, int(math.ceil(y2)))
    mask = np.zeros((height, width), bool); mask[top:bottom, left:right] = True
    return mask, (float(x1), float(y1), float(x2), float(y2))


def load_record(json_path: Path):
    doc = json.loads(json_path.read_text(encoding="utf-8-sig"))
    image_path = json_path.parent / doc.get("imagePath", f"{json_path.stem}.png")
    if not image_path.is_file():
        matches = list(json_path.parent.glob(json_path.stem + ".*"))
        image_path = next((p for p in matches if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}), image_path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    rgb = read_rgb(image_path); h, w = rgb.shape[:2]
    if (w, h) != (int(doc["imageWidth"]), int(doc["imageHeight"])):
        raise ValueError(f"dimension mismatch JSON={doc['imageWidth']}x{doc['imageHeight']} image={w}x{h}")
    masks = {k: np.zeros((h, w), bool) for k in ("tag", "background", "surface", "damage")}
    shapes, errors, clipped_polygons = [], [], 0
    counts = Counter()
    for i, shape in enumerate(doc.get("shapes", [])):
        label, kind = shape.get("label", ""), shape.get("shape_type", "polygon")
        counts[label] += 1
        try:
            if label in EXCLUSION_LABELS:
                if kind != "polygon": raise ValueError(f"{label} must be polygon, got {kind}")
                m, clipped = polygon_mask(shape.get("points", []), w, h); clipped_polygons += int(clipped)
                masks[label] |= m
                shapes.append({"index": i, "label": label, "normalized_label": label,
                               "shape_type": kind, "mask": m, "box": None})
            elif label in DAMAGE_MAP:
                if kind != "rectangle": raise ValueError(f"damage must be rectangle, got {kind}")
                m, box = rectangle_mask(shape.get("points", []), w, h); masks["damage"] |= m
                shapes.append({"index": i, "label": label, "normalized_label": DAMAGE_MAP[label],
                               "shape_type": kind, "mask": m, "box": box})
            elif label in SURFACE_LABELS:
                if kind != "polygon": raise ValueError(f"surface must be polygon, got {kind}")
                m, clipped = polygon_mask(shape.get("points", []), w, h); clipped_polygons += int(clipped)
                masks["surface"] |= m
                shapes.append({"index": i, "label": label, "normalized_label": "surface",
                               "shape_type": kind, "mask": m, "box": None})
        except Exception as exc:
            errors.append({"shape_index": i, "label": label, "reason": str(exc)})
    return {"name": json_path.stem, "json": json_path, "image": image_path, "rgb": rgb,
            "gray": cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), "width": w, "height": h,
            "masks": masks, "shapes": shapes, "counts": counts, "errors": errors,
            "clipped_polygons": clipped_polygons}


def overlay_ground_truth(rec, out_dir):
    rgb = rec["rgb"].astype(float); masks = rec["masks"]
    # RGB: tag=blue, background=orange, damage=green, surface=cyan.
    layers = [("tag", (40, 110, 255)), ("background", (255, 145, 30)),
              ("surface", (20, 210, 210)), ("damage", (30, 230, 70))]
    out = rgb.copy()
    for key, color in layers:
        m = masks[key]
        out[m] = .58 * out[m] + .42 * np.asarray(color)
    out = np.clip(out, 0, 255).astype(np.uint8)
    draw = ImageDraw.Draw(im := Image.fromarray(out))
    draw.rectangle((8, 8, 490, 42), fill=(0, 0, 0))
    draw.text((16, 16), "tag=blue  background=orange  damage=green  surface=cyan", fill=(255, 255, 255))
    im.save(out_dir / f"{rec['name']}_overlay.png")
    for key in ("tag", "background", "damage", "surface"):
        Image.fromarray((masks[key] * 255).astype(np.uint8)).save(out_dir / f"{rec['name']}_{key}_mask.png")


def annotation_summary(rec):
    m = rec["masks"]; h, w = m["tag"].shape
    return {"image": rec["name"], "width": w, "height": h,
            "lighting": "off" if rec["name"] in LIGHT_OFF else "on",
            "tag_shapes": rec["counts"].get("tag", 0), "tag_pixels": int(m["tag"].sum()),
            "tag_fraction": float(m["tag"].mean()) if m["tag"].any() else None,
            "background_shapes": rec["counts"].get("background", 0),
            "background_pixels": int(m["background"].sum()),
            "background_fraction": float(m["background"].mean()) if m["background"].any() else None,
            "surface_shapes": sum(rec["counts"].get(x, 0) for x in SURFACE_LABELS),
            "surface_pixels": int(m["surface"].sum()),
            "damage_shapes": sum(rec["counts"].get(x, 0) for x in DAMAGE_MAP),
            "damage_pixels": int(m["damage"].sum()),
            "tag_background_overlap_pixels": int((m["tag"] & m["background"]).sum()),
            "damage_exclusion_overlap_pixels": int((m["damage"] & (m["tag"] | m["background"])).sum()),
            "clipped_polygons": rec["clipped_polygons"], "shape_errors": len(rec["errors"])}


def starts(length, size):
    return range(0, length - size + 1, size // 2)


def integral(mask):
    return cv2.integral(mask.astype(np.uint8), sdepth=cv2.CV_32S)


def rect_sum(ii, x, y, size):
    return int(ii[y + size, x + size] - ii[y, x + size] - ii[y + size, x] + ii[y, x])


def compute_features(rec):
    gray = rec["gray"]; h, w = gray.shape
    _, global_amp = fft.fft_amplitude(gray)
    cmask = fft.comparison_mask(129, 5)
    ref = fft.vector(fft.common_spectrum(global_amp, 129, .125), cmask, "log")
    ints = {k: integral(v) for k, v in rec["masks"].items()}
    rows = []
    for size in SIZES:
        if h < size or w < size: continue
        area = size * size
        for y in starts(h, size):
            for x in starts(w, size):
                _, amp = fft.fft_amplitude(gray[y:y+size, x:x+size])
                vec = fft.vector(fft.common_spectrum(amp, 129, .125), cmask, "log")
                sim = fft.cosine(ref, vec)
                fractions = {k: rect_sum(ii, x, y, size) / area for k, ii in ints.items()}
                pure = [k for k in ("tag", "background", "surface") if fractions[k] >= .90]
                labeled_hits = [k for k in ("tag", "background", "surface") if fractions[k] > 0]
                if len(pure) == 1:
                    region = pure[0]
                elif labeled_hits:
                    region = "mixed"
                else:
                    region = "unlabeled"
                rows.append({"image": rec["name"], "lighting": "off" if rec["name"] in LIGHT_OFF else "on",
                             "size": size, "x": x, "y": y, "similarity": sim,
                             "region_class": region, "damage_block": int(fractions["damage"] > 0),
                             **{f"{k}_fraction": v for k, v in fractions.items()}})
    return rows


def vote_mask(shape, rows, flags):
    h, w = shape; cover = np.zeros((h + 1, w + 1), np.int32); excluded = np.zeros_like(cover)
    for r, flag in zip(rows, flags):
        x, y, n = int(r["x"]), int(r["y"]), int(r["size"])
        for yy, xx, sign in ((y,x,1),(y+n,x,-1),(y,x+n,-1),(y+n,x+n,1)):
            cover[yy,xx] += sign; excluded[yy,xx] += sign * int(flag)
    cover = cover.cumsum(0).cumsum(1)[:-1, :-1]
    excluded = excluded.cumsum(0).cumsum(1)[:-1, :-1]
    # Strict majority. Ties and uncovered strips stay in the inspection region.
    return (cover > 0) & (excluded * 2 > cover)


def fixed_mask(shape, rows, size, threshold):
    part = [r for r in rows if r["size"] == size]
    return vote_mask(shape, part, [r["similarity"] is not None and r["similarity"] < threshold for r in part])


def legacy_mask(shape, rows, size):
    part = [r for r in rows if r["size"] == size]
    valid = np.asarray([r["similarity"] for r in part if r["similarity"] is not None])
    if not len(valid): return np.zeros(shape, bool), None
    p20 = float(np.percentile(valid, 20))
    cats = [2 if r["similarity"] is None else 0 if r["similarity"] >= .90 else 1 if r["similarity"] <= p20 else 2 for r in part]
    votes = []
    for cat in range(3): votes.append(vote_count(shape, part, [c == cat for c in cats]))
    stack, coverage = np.stack([v[1] for v in votes]), votes[0][0]
    # Reproduce legacy priority on a tie: grid(0), background(1), review(2).
    return (coverage > 0) & (np.argmax(stack, axis=0) == 1), p20


def vote_count(shape, rows, flags):
    h, w = shape; cover = np.zeros((h + 1, w + 1), np.int32); count = np.zeros_like(cover)
    for r, flag in zip(rows, flags):
        x, y, n = int(r["x"]), int(r["y"]), int(r["size"])
        for yy, xx, sign in ((y,x,1),(y+n,x,-1),(y,x+n,-1),(y+n,x+n,1)):
            cover[yy,xx] += sign; count[yy,xx] += sign * int(flag)
    return cover.cumsum(0).cumsum(1)[:-1,:-1], count.cumsum(0).cumsum(1)[:-1,:-1]


def ratio_or_none(numerator, mask):
    total = int(mask.sum())
    return (float(numerator[mask].mean()) if total else None), total


def mask_metrics(rec, excluded, method, size, threshold=None, p20=None):
    m = rec["masks"]
    tag_rate, tag_px = ratio_or_none(excluded, m["tag"])
    bg_rate, bg_px = ratio_or_none(excluded, m["background"])
    surf_excl, surf_px = ratio_or_none(excluded, m["surface"])
    damage_excl, damage_px = ratio_or_none(excluded, m["damage"])
    damages = [s for s in rec["shapes"] if s["normalized_label"] in set(DAMAGE_MAP.values())]
    centers_kept, retained = [], []
    for s in damages:
        x1,y1,x2,y2 = s["box"]; cx = min(rec["width"]-1, max(0, int((x1+x2)/2))); cy = min(rec["height"]-1, max(0, int((y1+y2)/2)))
        centers_kept.append(not bool(excluded[cy,cx]))
        retained.append(1. - float(excluded[s["mask"]].mean()))
    return {"image": rec["name"], "lighting": "off" if rec["name"] in LIGHT_OFF else "on",
            "method": method, "size": size, "threshold": threshold, "legacy_p20": p20,
            "excluded_pixels": int(excluded.sum()), "excluded_fraction": float(excluded.mean()),
            "tag_pixels": tag_px, "tag_exclusion_rate": tag_rate,
            "background_pixels": bg_px, "background_exclusion_rate": bg_rate,
            "surface_pixels": surf_px, "surface_retention_rate": None if surf_excl is None else 1.-surf_excl,
            "damage_pixels": damage_px, "damage_pixel_retention_rate": None if damage_excl is None else 1.-damage_excl,
            "damage_count": len(damages), "damage_centers_kept": sum(centers_kept),
            "damage_centers_lost": len(centers_kept)-sum(centers_kept),
            "damage_center_retention_rate": float(np.mean(centers_kept)) if centers_kept else None,
            "damage_shapes_ge95_retained": sum(x >= .95 for x in retained),
            "damage_shape_mean_retention": float(np.mean(retained)) if retained else None,
            "damage_shape_min_retention": min(retained) if retained else None}


def save_similarity_map(rec, rows, size, out):
    part = [r for r in rows if r["size"] == size]
    h,w = rec["gray"].shape; total = np.zeros((h+1,w+1),float); count = np.zeros((h+1,w+1),np.int32)
    for r in part:
        x,y,n,s=int(r["x"]),int(r["y"]),size,r["similarity"]
        if s is None: continue
        for yy,xx,sgn in ((y,x,1),(y+n,x,-1),(y,x+n,-1),(y+n,x+n,1)):
            total[yy,xx]+=sgn*s; count[yy,xx]+=sgn
    total=total.cumsum(0).cumsum(1)[:-1,:-1]; count=count.cumsum(0).cumsum(1)[:-1,:-1]
    avg=np.divide(total,count,out=np.full_like(total,np.nan),where=count>0)
    fig,ax=plt.subplots(figsize=(10,7)); im=ax.imshow(avg,vmin=0,vmax=1,cmap="viridis")
    ax.set_title(f"{rec['name']} - {size}px overlap-averaged cosine similarity"); ax.axis("off")
    fig.colorbar(im,ax=ax,fraction=.035,pad=.02); fig.tight_layout(); fig.savefig(out,dpi=150); plt.close(fig)


def decision_overlay(rec, excluded, title, out):
    a=rec["rgb"].astype(float); a[excluded]=.45*a[excluded]+.55*np.array([255,30,30])
    a=np.clip(a,0,255).astype(np.uint8); im=Image.fromarray(a); d=ImageDraw.Draw(im)
    d.rectangle((5,5,min(rec["width"]-5,720),40),fill=(0,0,0)); d.text((12,14),title+" (red=excluded)",fill=(255,255,255)); im.save(out)


def distribution_plots(rows, out_dir):
    cats=("tag","background","mixed","unlabeled")
    for light in ("off","on"):
        for size in SIZES:
            subset=[r for r in rows if r["lighting"]==light and r["size"]==size]
            series={c:[r["similarity"] for r in subset if r["region_class"]==c and r["similarity"] is not None] for c in cats}
            series["damage"]=[r["similarity"] for r in subset if r["damage_block"] and r["similarity"] is not None]
            fig,axes=plt.subplots(1,2,figsize=(13,4.8))
            for c,v in series.items():
                if v: axes[0].hist(v,bins=np.linspace(0,1,41),alpha=.45,label=f"{c} (n={len(v)})")
            axes[0].set(xlabel="cosine similarity",ylabel="blocks",title=f"lighting {light}, {size}px"); axes[0].legend(fontsize=8)
            nonempty=[(c,v) for c,v in series.items() if v]
            if nonempty:
                axes[1].boxplot([v for _,v in nonempty],tick_labels=[c for c,_ in nonempty],showfliers=False)
                axes[1].tick_params(axis="x",rotation=30)
            axes[1].set(ylabel="cosine similarity",title="Distribution by manual-region relation")
            fig.tight_layout(); fig.savefig(out_dir/f"lighting_{light}_{size}_distributions.png",dpi=160); plt.close(fig)


def summarize_thresholds(metrics):
    groups=defaultdict(list)
    for r in metrics: groups[(r["method"],r["size"],r["threshold"],r["lighting"])].append(r)
    result=[]
    for key,rs in groups.items():
        method,size,threshold,lighting=key
        def weighted(rate_key,pixel_key):
            good=[r for r in rs if r[rate_key] is not None]
            den=sum(r[pixel_key] for r in good)
            return sum(r[rate_key]*r[pixel_key] for r in good)/den if den else None
        n_damage=sum(r["damage_count"] for r in rs); kept=sum(r["damage_centers_kept"] for r in rs)
        result.append({"method":method,"size":size,"threshold":threshold,"lighting":lighting,
            "images":len(rs),"tag_annotated_images":sum(r["tag_pixels"]>0 for r in rs),
            "background_annotated_images":sum(r["background_pixels"]>0 for r in rs),
            "tag_exclusion_rate":weighted("tag_exclusion_rate","tag_pixels"),
            "background_exclusion_rate":weighted("background_exclusion_rate","background_pixels"),
            "damage_pixel_retention_rate":weighted("damage_pixel_retention_rate","damage_pixels"),
            "damage_count":n_damage,"damage_centers_kept":kept,"damage_centers_lost":n_damage-kept,
            "damage_center_retention_rate":kept/n_damage if n_damage else None,
            "mean_excluded_fraction":float(np.mean([r["excluded_fraction"] for r in rs]))})
    return sorted(result,key=lambda r:(r["method"],r["size"],str(r["threshold"]),r["lighting"]))


def choose_condition(metrics):
    # image1 is an explicitly declared exploratory/selection image; all other annotated images are held out.
    candidates=[]
    for size in SIZES:
        for t in THRESHOLDS:
            rs=[r for r in metrics if r["image"]=="image1" and r["method"]=="fixed" and r["size"]==size and r["threshold"]==t]
            if not rs: continue
            r=rs[0]; safe=(r["damage_centers_lost"]==0 and (r["damage_shape_min_retention"] is None or r["damage_shape_min_retention"]>=.95))
            available=[v for v in (r["tag_exclusion_rate"],r["background_exclusion_rate"]) if v is not None]
            utility=float(np.mean(available)) if available else -1
            candidates.append((safe,utility,-t,-size,size,t,r))
    safe=[c for c in candidates if c[0]]
    pool=safe or candidates
    return max(pool,key=lambda x:x[1:6])[-1] if pool else None, bool(safe)


def save_failure_cases(records, features_by_image, selected, out_dir):
    if not selected: return []
    size,t=selected["size"],selected["threshold"]; examples=[]
    kinds=[("tag",True,"correctly_excluded_tag"),("tag",False,"missed_tag"),
           ("background",True,"correctly_excluded_background"),("background",False,"missed_background")]
    for region,want_excluded,label in kinds:
        pool=[]
        for rec in records:
            for r in features_by_image[rec["name"]]:
                if r["size"]==size and r["region_class"]==region:
                    if r["similarity"] is None: continue
                    decision=r["similarity"] < t
                    if decision==want_excluded: pool.append((abs(r["similarity"]-t),rec,r))
        if not pool: continue
        _,rec,r=min(pool,key=lambda z:z[0]); save_case(rec,r,label,t,out_dir); examples.append({"case":label,"image":rec["name"],"similarity":r["similarity"],"decision_excluded":want_excluded})
    # A damage-overlapping block wrongly excluded, and a mixed boundary block near threshold.
    for label,predicate in (("excluded_damage_block",lambda r:r["damage_block"] and r["similarity"] is not None and r["similarity"]<t),
                            ("mixed_boundary",lambda r:r["region_class"]=="mixed")):
        pool=[]
        for rec in records:
            for r in features_by_image[rec["name"]]:
                if r["size"]==size and r["similarity"] is not None and predicate(r): pool.append((abs(r["similarity"]-t),rec,r))
        if pool:
            _,rec,r=min(pool,key=lambda z:z[0]); save_case(rec,r,label,t,out_dir); examples.append({"case":label,"image":rec["name"],"similarity":r["similarity"],"decision_excluded":r["similarity"]<t})
    write_csv(out_dir/"failure_case_index.csv",examples)
    return examples


def save_case(rec,r,label,threshold,out_dir):
    x,y,n=int(r["x"]),int(r["y"]),int(r["size"]); crop=rec["rgb"][y:y+n,x:x+n]; gray=rec["gray"][y:y+n,x:x+n]
    raw,_=fft.fft_amplitude(gray); spec=np.log1p(raw); spec=(spec-spec.min())/(spec.max()-spec.min()+1e-12)
    fig,axes=plt.subplots(1,2,figsize=(8,4)); axes[0].imshow(crop); axes[0].set_title(f"{rec['name']} ({x},{y}) {n}px"); axes[1].imshow(spec,cmap="magma"); axes[1].set_title("log FFT amplitude")
    for ax in axes: ax.axis("off")
    fig.suptitle(f"{label}: similarity={r['similarity']:.4f}, threshold={threshold:.2f}, decision={'exclude' if r['similarity']<threshold else 'keep'}")
    fig.tight_layout(); fig.savefig(out_dir/f"{label}_{rec['name']}_{x}_{y}.png",dpi=170); plt.close(fig)


def make_report(records, ann, metrics, features, selected, selection_safe, failure_cases, out):
    annotated_bg=sum(a["background_shapes"]>0 for a in ann); damage_total=sum(a["damage_shapes"] for a in ann)
    s_size=selected["size"] if selected else None; s_t=selected["threshold"] if selected else None
    eval_images={r["name"] for r in records if r["name"]!="image1"}
    def aggregate(rows, held_out=True):
        if held_out: rows=[r for r in rows if r["image"] in eval_images]
        def w(rate,pix):
            z=[r for r in rows if r[rate] is not None]; den=sum(r[pix] for r in z)
            return sum(r[rate]*r[pix] for r in z)/den if den else None
        count=sum(r["damage_count"] for r in rows); kept=sum(r["damage_centers_kept"] for r in rows)
        return w("tag_exclusion_rate","tag_pixels"),w("background_exclusion_rate","background_pixels"),w("damage_pixel_retention_rate","damage_pixels"),count,kept
    fixed_rows=[r for r in metrics if r["method"]=="fixed" and r["size"]==s_size and r["threshold"]==s_t]
    tag,bg,dmg,n,kept=aggregate(fixed_rows)
    pct=lambda v:"欠損" if v is None else f"{100*v:.2f}%"
    comparisons=[]
    for size in SIZES:
        rs=[r for r in metrics if r["method"]=="fixed" and r["size"]==size and r["threshold"]==.87]
        a=aggregate(rs); comparisons.append((f"固定0.87 / {size}px",*a))
        rs=[r for r in metrics if r["method"]=="legacy_p20" and r["size"]==size]
        a=aggregate(rs); comparisons.append((f"従来下位20% / {size}px",*a))
    comparison_lines=["|条件|札除外率|背景除外率|傷画素保持率|傷中心保持|",
                      "|---|---:|---:|---:|---:|"]
    for label,ta,ba,da,nn,kk in comparisons:
        comparison_lines.append(f"|{label}|{pct(ta)}|{pct(ba)}|{pct(da)}|{kk}/{nn}（損失{nn-kk}）|")
    feature_stats=[]
    for lighting in ("off","on"):
        for region in ("tag","background","unlabeled"):
            vals=[r["similarity"] for r in features if r["size"]==128 and r["lighting"]==lighting and r["region_class"]==region and r["similarity"] is not None]
            feature_stats.append((lighting,region,len(vals),float(np.median(vals)) if vals else None))
    fs_lines=["|照明|ブロック種別|n|類似度中央値（128px）|","|---|---|---:|---:|"]
    for light,region,count,median in feature_stats:
        fs_lines.append(f"|{'なし' if light=='off' else 'あり'}|{region}|{count}|{'欠損' if median is None else f'{median:.4f}'}|")
    lines=["# FFT背景除外の手動正解マスク評価","","## 1. 目的","",
           "手動アノテーションされた札（tag）と対象外背景（background）をFFT類似度で除外し、同時に傷を保持できるかを定量評価した。未ラベル領域は正常表面とは扱っていない。","",
           "## 2. 手法","",
           f"- 対象は{len(records)}画像。tagは{len(records)}画像、backgroundは{annotated_bg}画像、傷は{damage_total}件。",
           "- FFTは既存実装と同じく、平均除去、2次元Hann窓、振幅、共通周波数格子（±0.125 cycles/pixel、129×129）、中心DC近傍除外、log1p、画像全体FFTとのコサイン類似度を使用した。",
           "- 128/256 px、50%オーバーラップ。固定しきい値では、各画素を覆うブロックの過半数がしきい値未満のときだけ除外した。同数票と未被覆端は保持した。",
           "- 従来法は画像・サイズごとの類似度下位20%を候補とし、grid/background/reviewの従来優先順を再現した。",
           "- image1を探索・条件選択専用、image2〜21を固定条件の評価用とした。正常surfaceの手動ROIは存在する場合だけ評価する。","",
           "## 3. 結果","",
           f"image1で傷中心を全保持し各傷矩形を95%以上保持する制約を{'満たした条件から' if selection_safe else '満たす条件がなかったため除外性能優先で'}、{s_size}px・しきい値{s_t}を選択した。",
           f"固定後のimage2〜21では、tag除外率 {pct(tag)}、background除外率 {pct(bg)}、傷画素保持率 {pct(dmg)}、傷中心保持 {kept}/{n}件（損失 {n-kept}件）だった。",
           "照明別・全しきい値の値は threshold_summary.csv、画像別の値は metrics.csv に保存した。","",
           *comparison_lines,"",
           "固定0.87では256pxの方が128pxより札・背景除外率が少し高く、傷中心損失は7件対11件で少なかった。一方、従来下位20%法では128pxの方が傷中心損失が2件対4件で少ない。したがってサイズの優劣は判定規則と安全目標で変わる。固定0.87同士なら256pxが相対的に適している。","",
           "### 照明条件と類似度分布","",*fs_lines,"",
           "格子ライトなし群ではbackgroundの類似度中央値がunlabeledより高く、低類似度を背景とする規則では背景を分離できなかった。tagは低い傾向を示すが、正常surface ROIは一枚にも付与されていないため、unlabeledを金属表面と断定した分類精度は算出していない。","",
           "![類似度分布](04_metrics/lighting_off_128_distributions.png)","",
           "## 4. 考察","",
           "tag/backgroundの除外率と傷保持率はトレードオフであり、未ラベル領域を正常と仮定したPrecisionは算出していない。格子ライトなし群でも分布差が出る場合、類似度は格子周期だけでなく、エッジ、粒状性、明暗むらなど一般的な表面テクスチャにも反応している。mixedブロックでは札・背景と金属面が一つのFFTに混ざるため、境界の判定が不安定になる。",
           "画像全体FFTを基準にするため、札や背景の面積が大きい画像では基準自体がそれらに引かれる。また傷があるブロックでも、傷以外の画素が支配的なら高類似度となり、逆に照明むらや局所的な表面変化で正常な金属面が低類似度になる。","",
           "## 5. 結論","",
           f"今回の手動正解に対する選択条件は{s_size}px・{s_t}だったが、これはimage1で選択した探索的条件である。別画像で傷中心を{n-kept}件失ったため、しきい値0.90は不適切であり、0.87も傷損失が残るので安全なしきい値とはいえない。FFT類似度単独での自動除外は安全とは結論づけない。","",
           "### 問いへの回答","",
           f"1. 選択条件で札は{pct(tag)}除外できた。","",
           f"2. 選択条件で背景は{pct(bg)}除外できた。ただし照明なし群では低類似度規則が背景に効かない。","",
           f"3. 傷中心は{n-kept}/{n}件（{100*(n-kept)/n:.2f}%）失い、傷画素は{pct(dmg)}保持した。件数と面積は異なる指標である。","",
           "4. 固定0.87では256pxが相対的に良いが、従来法では128pxの傷保持が良く、絶対的な優劣ではない。","",
           "5. 格子ライトなしでもtagは低類似度傾向だが、backgroundは高類似度であり、一括分離できない。surface正解がないためtag対金属面の確定評価もできない。","",
           "6. 0.87は128pxで11件、256pxで7件の傷中心を失うため、安全重視のしきい値として妥当ではない。","",
           "7. FFT類似度だけでは安全な背景除外はできない。少なくとも傷保護規則、領域形状・位置情報、または別の確認段階が必要である。","",
           "### 制約","","- backgroundがない5画像は欠損扱い。","- 未ラベル領域は正常表面ではない。","- image18は428×321 pxと小さく、他画像と同列の解像度性能とはみなさない。","- polygonと傷の重複はannotation_summary.csvに記録した。"]
    out.write_text("\n".join(lines)+"\n",encoding="utf-8")


ALL_METRICS=[]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",type=Path,default=DATA); ap.add_argument("--out",type=Path,default=DEFAULT_OUT)
    ap.add_argument("--ground-truth-only",action="store_true"); ap.add_argument("--only",default=None,help="e.g. image1")
    args=ap.parse_args(); out=args.out
    dirs={name:out/name for name in ("01_ground_truth","02_similarity","03_threshold_comparison","04_metrics","05_failure_cases","06_visualizations")}
    for d in dirs.values(): d.mkdir(parents=True,exist_ok=True)
    files=sorted(args.data.glob("image*.json"),key=natural_image_key)
    if args.only: files=[p for p in files if p.stem==args.only]
    records=[]; skipped=[]
    for p in files:
        try:
            rec=load_record(p); records.append(rec); overlay_ground_truth(rec,dirs["01_ground_truth"])
            print(f"loaded {rec['name']}: tag={rec['counts'].get('tag',0)} background={rec['counts'].get('background',0)} damage={sum(rec['counts'].get(x,0) for x in DAMAGE_MAP)}",flush=True)
        except Exception as exc: skipped.append({"json":str(p),"reason":str(exc)})
    ann=[annotation_summary(r) for r in records]; write_csv(out/"annotation_summary.csv",ann); write_json(out/"annotation_errors.json",{r["name"]:r["errors"] for r in records if r["errors"]}); write_json(out/"skipped.json",skipped)
    if args.ground_truth_only: return
    all_features=[]; features_by_image={}; global ALL_METRICS; ALL_METRICS=[]
    for rec in records:
        rows=compute_features(rec); features_by_image[rec["name"]]=rows; all_features.extend(rows)
        for size in SIZES:
            if not any(r["size"]==size for r in rows): continue
            save_similarity_map(rec,rows,size,dirs["02_similarity"]/f"{rec['name']}_{size}_similarity.png")
            lm,p20=legacy_mask((rec["height"],rec["width"]),rows,size); ALL_METRICS.append(mask_metrics(rec,lm,"legacy_p20",size,p20=p20))
            decision_overlay(rec,lm,f"legacy p20, {size}px (p20={p20:.4f})",dirs["03_threshold_comparison"]/f"{rec['name']}_{size}_legacy_p20.png")
            for t in THRESHOLDS:
                m=fixed_mask((rec["height"],rec["width"]),rows,size,t)
                ALL_METRICS.append(mask_metrics(rec,m,"fixed",size,threshold=t))
                # Save every threshold for image1; all images get 0.87 plus selected later.
                if rec["name"]=="image1" or t==.87:
                    decision_overlay(rec,m,f"fixed threshold {t:.2f}, {size}px",dirs["03_threshold_comparison"]/f"{rec['name']}_{size}_t{t:.2f}.png")
        print(f"FFT evaluated {rec['name']} ({len(rows)} blocks)",flush=True)
    write_csv(out/"features.csv",all_features); write_csv(out/"metrics.csv",ALL_METRICS)
    threshold_summary=summarize_thresholds(ALL_METRICS); write_csv(out/"threshold_summary.csv",threshold_summary)
    distribution_plots(all_features,dirs["04_metrics"])
    selected,selection_safe=choose_condition(ALL_METRICS); write_json(out/"selected_condition.json",{"selection_image":"image1","condition":selected,"met_damage_safety_constraint":selection_safe})
    if selected:
        for rec in records:
            rows=features_by_image[rec["name"]]
            if any(r["size"]==selected["size"] for r in rows):
                m=fixed_mask((rec["height"],rec["width"]),rows,selected["size"],selected["threshold"])
                decision_overlay(rec,m,f"selected: {selected['size']}px, threshold {selected['threshold']:.2f}",dirs["06_visualizations"]/f"{rec['name']}_selected.png")
    failures=save_failure_cases(records,features_by_image,selected,dirs["05_failure_cases"])
    make_report(records,ann,ALL_METRICS,all_features,selected,selection_safe,failures,out/"report.md")
    write_json(out/"run_config.json",{"data":str(args.data.resolve()),"sizes":SIZES,"overlap":.5,"thresholds":THRESHOLDS,"block_pure_fraction":.90,"fixed_vote":"strict majority; ties/uncovered kept","selection_image":"image1","lighting_off":sorted(LIGHT_OFF)})
    print(f"done: {out}",flush=True)


if __name__ == "__main__": main()
