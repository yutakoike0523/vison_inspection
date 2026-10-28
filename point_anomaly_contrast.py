"""Reproducible intensity-only point anomaly experiment (no FFT or learned model)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
PILOT = {"image16", "image18", "image20"}
LABEL = "点状異常"
FILTERS = {"loose": (2, 2500, 2, 120, 8.0), "moderate": (6, 1200, 2, 80, 4.0)}
THRESHOLDS = (8, 16, 24, 40)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def csv_out(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(path, a):
    Image.fromarray(a).save(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(annotation, out):
    data = json.loads(annotation.read_text(encoding="utf-8-sig"))
    records, skipped, manifest = [], [], []
    for entry in data["images"]:
        name = Path(entry["imagePath"]).stem
        if name not in {f"image{i}" for i in range(1, 22)}:
            continue
        candidates = [annotation.parent / entry["imagePath"],
                      ROOT / "data20260903/data20260903" / entry["imagePath"]]
        path = next((p.resolve() for p in candidates if p.is_file()), None)
        if path is None:
            skipped.append({"image": name, "reason": "image file missing", "paths": str(candidates)})
            continue
        rgb = read_image(path)
        h, w = rgb.shape[:2]
        if (w, h) != (entry["imageWidth"], entry["imageHeight"]):
            skipped.append({"image": name, "reason": "JSON/image dimension mismatch; no implicit rescaling", "paths": str(path)})
            continue
        boxes = []
        for shape in entry["shapes"]:
            if shape["label"] != LABEL:
                continue
            if shape["shape_type"] != "rectangle" or len(shape["points"]) != 2:
                raise ValueError(f"Unsupported point annotation: {name}")
            p = np.asarray(shape["points"], float)
            box = [float(p[:, 0].min()), float(p[:, 1].min()),
                   float(p[:, 0].max()), float(p[:, 1].max())]
            if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
                raise ValueError(f"Invalid annotation bounds: {name} {box}")
            boxes.append(box)
        records.append({"name": name, "path": path, "boxes": boxes, "width": w, "height": h})
        manifest.append({"image": name, "path": str(path), "width": w, "height": h,
                         "point_count": len(boxes), "sha256": digest(path)})
    records.sort(key=lambda x: int(x["name"][5:]))
    write_json(out / "01_original/source_manifest.json", manifest)
    write_json(out / "skipped.json", skipped)
    return records, manifest


def transformations(gray):
    yield "original", "baseline", gray, 8
    for lo, hi in [(1, 99), (2, 98)]:
        low, high = np.percentile(gray, [lo, hi])
        a = np.clip((gray.astype(np.float32) - low) * 255 / max(high - low, 1), 0, 255).round().astype(np.uint8)
        yield f"stretch_{lo}_{hi}", "contrast", a, 8
    for clip in [1.0, 2.0, 3.0]:
        for tile in [8, 16]:
            a = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(gray)
            yield f"clahe_c{int(clip)}_t{tile}", "clahe", a, 8
    for sigma in [3, 8, 16]:
        yield f"background_s{sigma}", "background_diff", gray, sigma


def residual(a, sigma):
    f = a.astype(np.float32)
    return cv2.GaussianBlur(f, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT_101) - f


def components(score, threshold, polarity):
    response = {"dark": score, "bright": -score, "both": np.abs(score)}[polarity]
    count, labels, stats, centroids = cv2.connectedComponentsWithStats((response >= threshold).astype(np.uint8), connectivity=8)
    peaks = np.zeros(count, dtype=np.float32)
    np.maximum.at(peaks, labels.ravel(), response.ravel())
    result = []
    for k in range(1, count):
        x, y, w, h, area = map(int, stats[k])
        result.append({"x1": x, "y1": y, "x2": x + w, "y2": y + h,
                       "width": w, "height": h, "area": area,
                       "cx": x + w / 2, "cy": y + h / 2,
                       "score": float(peaks[k]), "aspect": max(w / h, h / w)})
    return result


def filter_components(candidates, profile):
    amin, amax, dmin, dmax, aspect = FILTERS[profile]
    return [d for d in candidates if amin <= d["area"] <= amax
            and dmin <= d["width"] <= dmax and dmin <= d["height"] <= dmax and d["aspect"] <= aspect]


def match(detections, boxes, criterion):
    """Maximum-cardinality one-to-one assignment, IoU as secondary preference."""
    n, m = len(boxes), len(detections)
    if not n or not m:
        return {}, {}
    ious = np.zeros((n, m))
    valid = np.zeros((n, m), dtype=bool)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        for j, d in enumerate(detections):
            inter = max(0, min(x2, d["x2"]) - max(x1, d["x1"])) * max(0, min(y2, d["y2"]) - max(y1, d["y1"]))
            ious[i, j] = inter / ((x2-x1)*(y2-y1) + d["width"]*d["height"] - inter)
            valid[i, j] = (x1 <= d["cx"] < x2 and y1 <= d["cy"] < y2) if criterion == "center" else ious[i, j] >= .1
    reward = np.where(valid, n + 1 + ious, 0)
    rows, cols = linear_sum_assignment(-np.concatenate([reward, np.zeros((n, n))], axis=1))
    assignment, overlap = {}, {}
    for i, j in zip(rows, cols):
        if j < m and valid[i, j]:
            assignment[int(j)] = int(i)
            overlap[int(j)] = float(ious[i, j])
    return assignment, overlap


def counts(detections, boxes, criterion):
    assignment, _ = match(detections, boxes, criterion)
    return len(assignment), len(detections) - len(assignment), len(boxes) - len(assignment)


def summarize(tp, fp, fn, n):
    p = tp / (tp + fp) if tp + fp else 0.
    r = tp / (tp + fn) if tp + fn else 0.
    return {"images": n, "gt": tp+fn, "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": 2*p*r/(p+r) if p+r else 0., "fp_per_image": fp/n if n else 0.}


def pilot_sweep(records, out):
    totals = {}
    for rec in records:
        if rec["name"] not in PILOT:
            continue
        print("Pilot sweep:", rec["name"], flush=True)
        gray = cv2.cvtColor(read_image(rec["path"]), cv2.COLOR_RGB2GRAY)
        for variant, family, a, sigma in transformations(gray):
            score = residual(a, sigma)
            for threshold in THRESHOLDS:
                for polarity in ["dark", "bright", "both"]:
                    raw = components(score, threshold, polarity)
                    for profile in FILTERS:
                        key = (family, variant, sigma, threshold, polarity, profile)
                        ds = filter_components(raw, profile)
                        c = np.array(counts(ds, rec["boxes"], "center") + (1,), int)
                        totals[key] = totals.get(key, np.zeros(4, int)) + c
    rows = []
    for key, c in totals.items():
        row = dict(zip(["family", "variant", "sigma", "threshold", "polarity", "filter"], key))
        row.update(summarize(*map(int, c)))
        rows.append(row)
    csv_out(out / "pilot_parameter_sweep.csv", rows)
    selected = {}
    for family in ["baseline", "contrast", "clahe", "background_diff"]:
        subset = [r for r in rows if r["family"] == family]
        selected[family] = max(subset, key=lambda r: (r["f1"], r["recall"], -r["fp"]))
    write_json(out / "selected_parameters.json", selected)
    return selected


def montage(arrays, titles, width=640, nearest=False):
    panels = []
    for a, title in zip(arrays, titles):
        p = Image.fromarray(a).convert("RGB")
        height = round(p.height * width / p.width)
        p = p.resize((width, height), Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height+36), "#202020")
        canvas.paste(p, (0, 36))
        ImageDraw.Draw(canvas).text((8, 10), title, fill="white")
        panels.append(canvas)
    result = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), "#202020")
    x = 0
    for p in panels:
        result.paste(p, (x, 0))
        x += p.width
    return result


def draw_boxes(rgb, boxes, detections=()):
    a = Image.fromarray(rgb).convert("RGB")
    d = ImageDraw.Draw(a)
    for candidate in detections:
        d.rectangle([candidate[k] for k in ["x1", "y1", "x2", "y2"]], outline=(255, 0, 255), width=2)
    for i, b in enumerate(boxes):
        d.rectangle(b, outline=(0, 255, 0), width=2)
        d.text((b[0], max(0, b[1]-12)), f"GT{i+1}", fill=(0, 255, 0))
    return np.asarray(a)


def contrast_stats(a, box):
    h, w = a.shape
    x1, y1 = np.floor(box[:2]).astype(int)
    x2, y2 = np.ceil(box[2:]).astype(int)
    pad = max(8, round(max(x2-x1, y2-y1)/2))
    left, top, right, bottom = max(0,x1-pad), max(0,y1-pad), min(w,x2+pad), min(h,y2+pad)
    patch = a[top:bottom, left:right].astype(float)
    mask = np.ones(patch.shape, bool)
    mask[y1-top:y2-top, x1-left:x2-left] = False
    ring = patch[mask]
    inside = a[y1:y2, x1:x2].astype(float)
    bg = float(np.median(ring))
    noise = float(np.std(ring))
    dark = bg-float(np.percentile(inside,10))
    bright = float(np.percentile(inside,90))-bg
    return {"ring_median": bg, "ring_std": noise, "inside_mean": float(inside.mean()),
            "dark_delta_p10": dark, "bright_delta_p90": bright,
            "mean_delta": bg-float(inside.mean()), "tail_cnr": max(dark, bright)/max(noise,1.)}


def evaluate(records, selected, out):
    metric_rows, detection_rows, local_rows, crop_rows = [], [], [], []
    per_image = []
    fp_examples = {f: [] for f in selected}
    for rec in sorted(records, key=lambda r: (r['name'] not in PILOT, int(r['name'][5:]))):
        name, boxes = rec["name"], rec["boxes"]
        print("Evaluate and visualize:", name, flush=True)
        rgb = read_image(rec["path"])
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        variants = {v: (family,a,sigma) for v,family,a,sigma in transformations(gray)}
        displays = {"baseline": rgb}
        titles = ["Original RGB (unchanged pixels)"]
        for variant, (family,a,sigma) in variants.items():
            if family == "baseline":
                continue
            folder = {"contrast":"02_contrast", "clahe":"03_clahe", "background_diff":"04_background_diff"}[family]
            if family == "background_diff":
                score = residual(a,sigma)
                np.save(out / folder / f"{name}_{variant}_signed.npy", score)
                # Fixed gain for all images, zero residual = 128; dark defects are white.
                displayed = np.clip(128+4*score,0,255).round().astype(np.uint8)
                save_image(out / folder / f"{name}_{variant}_dark.png", np.clip(4*score,0,255).round().astype(np.uint8))
                save_image(out / folder / f"{name}_{variant}_bright.png", np.clip(-4*score,0,255).round().astype(np.uint8))
            else:
                displayed = a
            save_image(out / folder / f"{name}_{variant}.png", displayed)
            if variant == selected[family]["variant"]:
                displays[family] = displayed
        for family, config in selected.items():
            _, a, sigma = variants[config["variant"]]
            score = residual(a, sigma)
            ds = filter_components(components(score, config["threshold"], config["polarity"]), config["filter"])
            center, overlaps = match(ds, boxes, "center")
            iou_match, _ = match(ds, boxes, "iou")
            for criterion in ["center", "iou"]:
                tp, fp, fn = counts(ds, boxes, criterion)
                per_image.append({"image": name, "split": "pilot" if name in PILOT else "remaining",
                                  "family": family, "criterion": criterion, **summarize(tp,fp,fn,1)})
            for i,d in enumerate(ds):
                detection_rows.append({"image": name, "family": family, "variant": config["variant"],
                    "candidate": i+1, "polarity": config["polarity"], **d,
                    "center_gt": center[i]+1 if i in center else "", "center_match_iou": overlaps.get(i,""),
                    "iou_gt": iou_match[i]+1 if i in iou_match else ""})
            save_image(out / "06_detection" / f"{name}_{family}.png", draw_boxes(rgb, boxes, ds))
            unmatched = sorted([(i,d) for i,d in enumerate(ds) if i not in center], key=lambda t:-t[1]["score"])
            if name in PILOT:
                for i,d in unmatched[:4]:
                    x,y = int(d["cx"]),int(d["cy"])
                    x1,y1,x2,y2 = max(0,x-48),max(0,y-48),min(gray.shape[1],x+48),min(gray.shape[0],y+48)
                    crop = draw_boxes(rgb, [], [d])[y1:y2,x1:x2]
                    fp_examples[family].append((crop,f"{name} ({x},{y}) s={d['score']:.1f}"))
            for gt,box in enumerate(boxes):
                measurement = contrast_stats(a, box)
                if family == "background_diff":
                    measurement = contrast_stats(-score, box)
                local_rows.append({"image":name,"gt":gt+1,"family":family,"variant":config["variant"],**measurement})
        order = ["baseline","contrast","clahe","background_diff"]
        titles += [selected[f]["variant"] + (" (128+4*residual)" if f=="background_diff" else "") for f in order[1:]]
        arrays = [displays[f] for f in order]
        montage(arrays,titles).save(out / "05_comparison" / f"{name}_comparison.png")
        montage([draw_boxes(a,boxes) for a in arrays], titles).save(out / "05_comparison" / f"{name}_gt.png")
        for gt,box in enumerate(boxes):
            cx,cy = (box[0]+box[2])/2,(box[1]+box[3])/2
            side = max(64,int(np.ceil(max(box[2]-box[0],box[3]-box[1])+40)))
            x1,y1 = max(0,int(cx-side/2)),max(0,int(cy-side/2))
            x2,y2 = min(gray.shape[1],x1+side),min(gray.shape[0],y1+side)
            crops = [a[y1:y2,x1:x2] for a in arrays]
            prefix = f"{name}_gt{gt+1:02d}"
            save_image(out / "07_crops" / f"{prefix}_original_native.png", crops[0])
            montage(crops,titles,width=(x2-x1)*4,nearest=True).save(out / "07_crops" / f"{prefix}_comparison_4x.png")
            crop_rows.append({"image":name,"gt":gt+1,"x1":x1,"y1":y1,"x2":x2,"y2":y2,
                              "gt_x1":box[0],"gt_y1":box[1],"gt_x2":box[2],"gt_y2":box[3]})
            # All enhancement settings at exactly the same crop coordinates.
            settings = [("gray baseline",gray)] + [(v,a) for v,(f,a,s) in variants.items() if f in {"contrast","clahe"}]
            sheets = [montage([a[y1:y2,x1:x2] for _,a in settings[k:k+3]],
                             [v for v,_ in settings[k:k+3]],width=(x2-x1)*3,nearest=True) for k in range(0,len(settings),3)]
            sheet = Image.new("RGB",(max(s.width for s in sheets),sum(s.height for s in sheets)),"#202020")
            ypos=0
            for s in sheets:
                sheet.paste(s,(0,ypos)); ypos+=s.height
            sheet.save(out / "07_crops" / f"{prefix}_parameter_grid.png")
        if name in PILOT:
            variants_p = [(v,a) for v,(f,a,s) in variants.items() if f in {"contrast","clahe"}]
            montage([a for _,a in variants_p],[v for v,_ in variants_p],width=512).save(out / "05_comparison" / f"{name}_all_enhancements.png")
    for family,examples in fp_examples.items():
        rows = [montage([a for a,_ in examples[k:k+4]],[t for _,t in examples[k:k+4]],width=288,nearest=True) for k in range(0,len(examples),4)]
        if rows:
            sheet = Image.new("RGB",(max(r.width for r in rows),sum(r.height for r in rows)),"#202020")
            y=0
            for r in rows:
                sheet.paste(r,(0,y)); y+=r.height
            sheet.save(out / "06_detection" / f"unmatched_examples_{family}.png")
    for family in selected:
        for split in ["pilot", "remaining", "all"]:
            for criterion in ["center","iou"]:
                rows = [r for r in per_image if r["family"]==family and r["criterion"]==criterion and (split=="all" or r["split"]==split)]
                sums = [sum(r[k] for r in rows) for k in ["tp","fp","fn","images"]]
                metric_rows.append({"family":family,"variant":selected[family]["variant"],"split":split,"criterion":criterion,**summarize(*sums)})
    csv_out(out/"metrics.csv",metric_rows)
    csv_out(out/"metrics_per_image.csv",per_image)
    csv_out(out/"detections.csv",detection_rows)
    csv_out(out/"local_contrast.csv",local_rows)
    csv_out(out/"07_crops/crop_coordinates.csv",crop_rows)
    return metric_rows,local_rows


def measure_all_variants(records, out):
    """Independent contrast measurements, without retuning selected detector settings."""
    rows = []
    for rec in records:
        if not rec['boxes']:
            continue
        gray = cv2.cvtColor(read_image(rec['path']), cv2.COLOR_RGB2GRAY)
        for variant, family, a, sigma in transformations(gray):
            if family == 'background_diff':
                a = -residual(a, sigma)
            for gt, box in enumerate(rec['boxes']):
                rows.append({'image':rec['name'], 'split':'pilot' if rec['name'] in PILOT else 'remaining',
                             'gt':gt+1,'variant':variant,'family':family,**contrast_stats(a,box)})
    csv_out(out/'local_contrast_all_variants.csv',rows)
    summary=[]
    for variant in dict.fromkeys(r['variant'] for r in rows):
        for split in ['pilot','remaining','all']:
            part=[r for r in rows if r['variant']==variant and (split=='all' or r['split']==split)]
            summary.append({'variant':variant,'split':split,'count':len(part),
                **{key+'_median':float(np.median([r[key] for r in part])) for key in
                   ['dark_delta_p10','bright_delta_p90','ring_std','tail_cnr']}})
    csv_out(out/'local_contrast_summary.csv',summary)


def finalize_report(out, metrics, local):
    """Interpret bounded parameter search, not a claim of general detection performance."""
    lookup={(r['family'],r['split'],r['criterion']):r for r in metrics}
    bg=lookup['background_diff','all','center']
    bg_remaining=lookup['background_diff','remaining','center']
    base=lookup['baseline','all','center']
    st=lookup['contrast','all','center']
    cl=lookup['clahe','all','center']
    med={f:{k:float(np.median([float(r[k]) for r in local if r['family']==f])) for k in
            ['dark_delta_p10','ring_std','tail_cnr']} for f in ['baseline','contrast','clahe','background_diff']}
    discussion=[
        '1. **元画像での確認**：拡大図では暗い点・小さな線状の特徴を確認できる例がある（image1 GT1、image16 GT1/GT2、image20 GT1）。image16 GT2では通常強調で暗点が目立つ。一方、image18 GT4は背景の明暗模様に埋もれ、強調しても明瞭な分離は難しい。全景縮小だけでは微小欠陥が見えにくいため、元画素の切り出しを併用する。全27件の視認性を人による盲検試験で判定したものではない。',
        f"2. **強調の程度**：全27矩形の暗側輝度差中央値は元画像{med['baseline']['dark_delta_p10']:.2f}、ストレッチ{med['contrast']['dark_delta_p10']:.2f}、CLAHE{med['clahe']['dark_delta_p10']:.2f}。リング標準偏差はそれぞれ{med['baseline']['ring_std']:.2f}、{med['contrast']['ring_std']:.2f}、{med['clahe']['ring_std']:.2f}。差が増えても背景変動が同時に増える場合があり、輝度差だけを視認性改善と同一視できない。",
        f"3. **通常強調とCLAHE**：選択条件での全画像中心対応は通常強調{st['tp']}/{st['gt']}件・F1={st['f1']:.4f}、CLAHE{cl['tp']}/{cl['gt']}件・F1={cl['f1']:.4f}。全景の明暗を広げる効果と局所模様を強める効果は見られるが、今回の探索条件でCLAHEが常に有効とはいえない。選択基準は予備F1であり、主観的に最も見やすいパラメータを選んだわけではない。全8強調条件の拡大図と `local_contrast_summary.csv` で比較できる。",
        f"4. **背景差分による二値化**：全画像で{bg['tp']}/{bg['gt']}件、Recall={bg['recall']:.1%}、暫定Precision={bg['precision']:.4%}、FP/枚={bg['fp_per_image']:.2f}。予備以外では{bg_remaining['tp']}/{bg_remaining['gt']}件だが、FP/枚={bg_remaining['fp_per_image']:.2f}であり、多数の背景候補の中に正解対応候補が含まれる状況。局所差だけで抽出できる傷はあるが、全点状異常を安定して選別する性能は確認できない。元輝度の局所差では全画像{base['tp']}/{base['gt']}件、FP/枚={base['fp_per_image']:.2f}。",
        '5. **誤検出の原因と評価限界**：確認した未対応例ではimage1の細かな表面模様、image8の周期的な明暗、image16/20の札の縁、image18の対象物境界、image21の線状特徴が候補になった。札の文字や対象外領域も除外していない（image8のレビュー画像参照）。予備画像のFP/枚10に対し残り画像では約2989で、同じ条件が背景の異なる画像に適合していない。未対応候補には未注釈の傷や他ラベルの異常も含み得るため、正常とは断定しない。全画像の原因別FP内訳は人手確認していない。密な候補が偶然矩形に入る可能性もあり、中心対応Recallだけでは実用性を示せない。',
        '6. **格子ライトなしで検出可能か**：入力には周期的な明暗模様が明瞭に見える（特にimage18）。撮像条件のメタデータは確認できていない。本実験が示すのは「周期解析・FFTを使わず、与えられた画像の輝度差で一部候補を抽出できる」ことに限る。格子ライトなしの独立撮像で同じ結果が得られるかは未検証。',
        '今回のしきい値範囲・サイズ条件内の比較であり、全パラメータに対する最適性は主張しない。3枚13件による選択は不安定になり得る。image18の低解像度、矩形内の背景混入、近接模様、対象物や撮像の重複が結果に影響する。明点/両極性も予備探索したが、選ばれた条件は暗点のみである。これは明点が存在しないことを意味しない。'
    ]
    conclusion=(
        '基本的なコントラスト強調は点状特徴の見え方を変え、一部は元画像でも確認できる。ただし背景模様も強まり、今回の条件では単純なしきい値処理だけによる高いRecallと低い誤検出数の両立は確認できなかった。'
        'Precisionは注釈の完全性に依存する暫定値だが、見逃しも残るため、そのまま実用的な全自動検査が成立するとはいえない。\n\n'
        '次の比較には、格子ライトなしと確認できる同一対象の撮像、札・対象外領域のROI定義、未注釈候補の再確認、解像度・物理スケールの統一を用意し、独立した画像で条件を固定して評価することが必要。'
    )
    path=out/'report.md'
    content=path.read_text(encoding='utf-8')
    content=content.replace('<!-- VISUAL_REVIEW -->','\n\n'.join(discussion)).replace('<!-- CONCLUSION -->',conclusion)
    summary_path=out/'local_contrast_summary.csv'
    if summary_path.exists():
        with summary_path.open(encoding='utf-8-sig') as f:
            summary=[r for r in csv.DictReader(f) if r['split']=='all']
        table=['### 全パラメータの局所コントラスト比較','',
               '|変換|暗側輝度差中央値|背景標準偏差中央値|CNR中央値|',
               '|---|---:|---:|---:|']
        for r in summary:
            table.append(f"|{r['variant']}|{float(r['dark_delta_p10_median']):.2f}|{float(r['ring_std_median']):.2f}|{float(r['tail_cnr_median']):.2f}|")
        table+=['','ストレッチ2条件のCNRはほぼ同等。CLAHEのclipLimitを増やすと暗側差と背景変動の両方が増え、CNR中央値は低下した。背景差分sigma=16のCNR改善も、低い誤検出数での検出成功を意味しない。','']
        content=content.replace('## 考察','\n'.join(table)+'\n## 考察')
    content += ('\n`07_crops/*_gt_comparison_4x.png` は正解矩形を緑で加えた補助図。'
                '`06_detection/unmatched_examples_remaining_background_diff.png` はimage1/8/21の上位2件とスコア中央値付近2件ずつの未対応候補。'
                'レビュー用切り出しは評価ROIを変更しない。\n')
    # Separate prose paragraphs without breaking Markdown tables.
    lines=content.splitlines()
    formatted=[]
    for line in lines:
        if formatted and line and formatted[-1] and not line.startswith(('|','#')) and not formatted[-1].startswith(('|','#')):
            formatted.append('')
        formatted.append(line)
    content='\n'.join(formatted)+'\n'
    path.write_text(content,encoding='utf-8')


def supplements_and_verify(records, out):
    """Check crop pixels and aggregate counts; add review aids without changing detection."""
    with (out/'07_crops/crop_coordinates.csv').open(encoding='utf-8-sig') as f:
        crop_rows=list(csv.DictReader(f))
    sources={r['name']:read_image(r['path']) for r in records}
    for row in crop_rows:
        name,gt=row['image'],int(row['gt'])
        x1,y1,x2,y2=[int(row[k]) for k in ['x1','y1','x2','y2']]
        prefix=f'{name}_gt{gt:02d}'
        native=np.array(Image.open(out/'07_crops'/f'{prefix}_original_native.png'))
        assert np.array_equal(native,sources[name][y1:y2,x1:x2])
        sheet=Image.open(out/'07_crops'/f'{prefix}_comparison_4x.png').convert('RGB')
        # Pixel-exact original panel under its 36-pixel caption strip.
        expected=np.repeat(np.repeat(native,4,axis=0),4,axis=1)
        assert np.array_equal(np.asarray(sheet)[36:36+expected.shape[0],:expected.shape[1]],expected)
        draw=ImageDraw.Draw(sheet)
        box=[float(row[k]) for k in ['gt_x1','gt_y1','gt_x2','gt_y2']]
        for panel in range(4):
            offset=panel*4*(x2-x1)
            draw.rectangle([offset+4*(box[0]-x1),36+4*(box[1]-y1),
                            offset+4*(box[2]-x1),36+4*(box[3]-y1)],outline=(0,255,0),width=1)
        sheet.save(out/'07_crops'/f'{prefix}_gt_comparison_4x.png')
    totals=Counter()
    matched={}
    review=[]
    samples={name:[] for name in ['image1','image8','image21']}
    with (out/'detections.csv').open(encoding='utf-8-sig') as f:
        for d in csv.DictReader(f):
            family,name=d['family'],d['image']
            totals[family,name]+=1
            for criterion,key in [('center','center_gt'),('iou','iou_gt')]:
                if d[key]:
                    group=(family,name,criterion)
                    matched.setdefault(group,set())
                    assert d[key] not in matched[group], 'Non-unique GT match'
                    matched[group].add(d[key])
            if family=='background_diff' and name in samples and not d['center_gt']:
                samples[name].append(d)
    with (out/'metrics_per_image.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            key=(r['family'],r['image'])
            assert totals[key]==int(r['tp'])+int(r['fp'])
            assert len(matched.get((*key,r['criterion']),set()))==int(r['tp'])
    panels=[]
    for name,ds in samples.items():
        ds.sort(key=lambda d:-float(d['score']))
        if not ds:
            continue
        # Include highest scores and examples around median score; deterministic, not prevalence estimate.
        selected=ds[:2]+ds[len(ds)//2:len(ds)//2+2]
        crops,titles=[],[]
        for d in selected:
            x,y=int(float(d['cx'])),int(float(d['cy']))
            coords={k:int(d[k]) for k in ['x1','y1','x2','y2']}
            rgb=sources[name]
            crop=draw_boxes(rgb,[],[coords])[max(0,y-40):min(rgb.shape[0],y+40),max(0,x-40):min(rgb.shape[1],x+40)]
            crops.append(crop);titles.append(f"{name} ({x},{y}) s={float(d['score']):.1f}")
            review.append({'image':name,'candidate':d['candidate'],**coords,'score':d['score'],'note':'unmatched, not confirmed normal'})
        panels.append(montage(crops,titles,width=256,nearest=True))
    sheet=Image.new('RGB',(max(p.width for p in panels),sum(p.height for p in panels)),'#202020')
    y=0
    for panel in panels:
        sheet.paste(panel,(0,y));y+=panel.height
    sheet.save(out/'06_detection/unmatched_examples_remaining_background_diff.png')
    csv_out(out/'06_detection/review_examples.csv',review)
    if 'image8' in sources:
        with (out/'detections.csv').open(encoding='utf-8-sig') as f:
            ds=[{k:int(d[k]) for k in ['x1','y1','x2','y2']} for d in csv.DictReader(f)
                if d['image']=='image8' and d['family']=='background_diff']
        rgb=sources['image8']
        overlay=draw_boxes(rgb,[],ds)
        for tag,(x1,y1,x2,y2) in {'text':(200,280,750,800),'outside':(1000,1250,1800,1536)}.items():
            montage([rgb[y1:y2,x1:x2],overlay[y1:y2,x1:x2]],
                    ['Original RGB', 'Background-difference detections'],width=550).save(out/'06_detection'/f'image8_review_{tag}.png')
    write_json(out/'artifact_validation.json',{'native_and_nearest_neighbor_crops_exact':len(crop_rows),
        'one_to_one_matches_verified':True,'detection_metrics_counts_consistent':True})


def report(out, records, selected, metrics, local):
    lines = ["# 点状異常のコントラスト強調・自動検出比較", "", "## 目的", "",
        "通常画像の点状異常について、基本的な輝度処理による見やすさと自動検出の限界を調べる。FFT・周期解析・機械学習・画像生成AIは使用しない。", "", "## 手法", "",
        f"実画像{len(records)}枚、点状異常{sum(len(r['boxes']) for r in records)}件。画像参照とSHA-256は `01_original/source_manifest.json`、欠損等は `skipped.json`。元画像の画素を書き換えず、処理結果は出力フォルダへ保存した。",
        "image18は428×321 px、その他は2048×1536 px。寸法はJSONと一致。解像度を変更せず同じpixel単位の条件を適用するため、物理的な検出サイズは同一とは限らない。",
        "予備検討はimage16・18・20（13件）。残り18画像（14件）は選択済み条件で評価。ラベルのない画像も誤検出評価に含む。撮像対象の重複や独立性は未確認であり、残り画像の結果も未知撮像条件への汎化の証明ではない。",
        "処理はOpenCV RGB→gray、全画面の1–99%/2–98%ストレッチ、CLAHE clipLimit=1/2/3 × tileGridSize=8/16、Gaussian背景差分 sigma=3/8/16 px。tileGridSizeはタイル個数。Black-hatは今回未実施。",
        "baseline/ストレッチ/CLAHEは処理画像にsigma=8のGaussian背景推定を適用し、R=背景−処理画像を局所輝度差とする。背景差分法は元grayから直接Rを求める。従ってbaselineと背景差分sigma=8は同じ検出器で、baseline表示だけが無加工RGB。",
        "暗点R≥t、明点−R≥t、両方|R|≥tを比較。t=8/16/24/40（8bit輝度単位）。8近傍連結成分を用い、モルフォロジー・ROI・正解領域による候補制限は行わない。",
        "サイズ制約 loose: 面積2–2500 px、幅/高さ2–120 px、最大縦横比8。moderate: 面積6–1200 px、幅/高さ2–80 px、最大縦横比4。矩形面積ではなく連結成分画素数を面積とする。",
        "予備3画像だけで各手法群の中心対応micro-F1を最大化し、同点ならRecall、次いで誤検出数を優先。全288条件（12画像変換×4しきい値×3極性×2サイズ制約）を `pilot_parameter_sweep.csv` に記録。選択後は残り画像で再調整しない。",
        "中心対応: 候補bounding box中心が正解矩形内（左上含む・右下含まない）。IoU対応: IoU≥0.10を別評価。どちらも最大対応数を優先しIoUを次の優先値とする二部割当で1対1対応。重複候補はFP。候補ごとの対応ID・中心対応時IoUを `detections.csv` に保存。",
        "局所輝度差: 正解矩形内のp10/p90と周囲リング中央値との差。リング幅は矩形最大辺の半分（最低8px）、画像端で切り詰める。CNR代理値=max(暗差,明差)/max(リング標準偏差,1)。正解矩形は輪郭ではないため背景混入があり、これらはコントラストの代理指標である。背景差分では−Rを用いる。",
        "比較図の背景差分は全画像共通でclip(128+4R,0,255)。暗異常は白、明異常は黒、差分0は灰色。検出には表示用の増幅・クリップを用いずfloat残差を使用。signed.npyはRの生値、dark/bright.pngは片側差分の4倍表示。",
        "全景は縮小表示。拡大比較は同じ座標を切り出し4倍nearest-neighborで表示。original_native.pngは元RGB画素を保持する。正解は緑、検出候補はマゼンタ。無注記比較図と正解付き図を両方保存。", "", "## 結果", "", "### 選択された条件", "",
        "|手法|画像処理|sigma|閾値|極性|サイズ条件|", "|---|---|---:|---:|---|---|"]
    for family,c in selected.items():
        lines.append(f"|{family}|{c['variant']}|{c['sigma']}|{c['threshold']}|{c['polarity']}|{c['filter']}|")
    lines += ["", "### 検出評価（Precision・FPは未注釈傷を含み得る暫定値）", "",
              "|手法|区分|基準|TP/GT|Precision|Recall|F1|FP/枚|", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in metrics:
        lines.append(f"|{r['family']}|{r['split']}|{r['criterion']}|{r['tp']}/{r['gt']}|{r['precision']:.4f}|{r['recall']:.4f}|{r['f1']:.4f}|{r['fp_per_image']:.2f}|")
    lines += ["", "### 局所コントラスト（全27矩形の中央値）", "", "|手法|暗側輝度差|明側輝度差|リング標準偏差|CNR代理値|", "|---|---:|---:|---:|---:|"]
    for family in selected:
        rows = [r for r in local if r['family']==family]
        vals = [np.median([r[k] for r in rows]) for k in ['dark_delta_p10','bright_delta_p90','ring_std','tail_cnr']]
        lines.append(f"|{family}|"+'|'.join(f'{v:.2f}' for v in vals)+'|')
    lines += ["", "## 考察", "", "<!-- VISUAL_REVIEW -->", "", "## 結論", "", "<!-- CONCLUSION -->", "", "### 再実行と出力", "",
              "`python point_anomaly_contrast.py`。依存: numpy, opencv-python, Pillow, scipy。環境バージョンと実行条件は `run_config.json`。",
              "`05_comparison/image16_comparison.png`・`image18_comparison.png`・`image20_comparison.png` を最初に参照し、微小欠陥は `07_crops/*_comparison_4x.png` で確認する。`*_parameter_grid.png` は通常強調2条件とCLAHE6条件の同一座標比較。",
              "`06_detection/unmatched_examples_*.png` は予備画像ごとにスコア上位4件の未対応候補例。無作為標本ではなく、原因別頻度や真の正常判定を表すものではない。"]
    (out/"report.md").write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--annotation',type=Path,default=ROOT/'data20260903/data20260903/全傷.json')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/point_anomaly_contrast')
    args=parser.parse_args()
    cv2.setNumThreads(4)
    out=args.output
    for folder in ['01_original','02_contrast','03_clahe','04_background_diff','05_comparison','06_detection','07_crops']:
        (out/folder).mkdir(parents=True,exist_ok=True)
    records,manifest=load_data(args.annotation,out)
    selected=pilot_sweep(records,out)
    metrics,local=evaluate(records,selected,out)
    measure_all_variants(records,out)
    report(out,records,selected,metrics,local)
    finalize_report(out,metrics,local)
    supplements_and_verify(records,out)
    import scipy, PIL, sys
    write_json(out/'run_config.json',{'annotation':str(args.annotation.resolve()),'annotation_sha256':digest(args.annotation),
        'pilot':sorted(PILOT),'thresholds':THRESHOLDS,'filters':FILTERS,'selected':selected,
        'versions':{'python':sys.version,'opencv':cv2.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'Pillow':PIL.__version__}})
    assert all(digest(Path(m['path']))==m['sha256'] for m in manifest), 'Source changed during processing'
    write_json(out/'verification.json',{'source_hashes_unchanged':True,'images':len(records),
        'point_annotations':sum(len(r['boxes']) for r in records),'crop_count':len(list((out/'07_crops').glob('*_original_native.png')))})
    print('Finished:',out,flush=True)


if __name__=='__main__':
    main()
