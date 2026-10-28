"""Render the top-10 Gaussian sigma=8 Method-0 candidates for every image."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import point_anomaly_feature_analysis as fa

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data20260903" / "data20260903"
ANALYSIS = ROOT / "outputs" / "point_anomaly_peak_analysis"
OUT = ANALYSIS / "09_gaussian_s8_top10"


def candidates(response, threshold):
    binary = response >= threshold
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    rows = []
    for label in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[label]]
        if not (2 <= area <= 2000 and w <= 100 and h <= 100):
            continue
        rows.append({
            "component_x": x, "component_y": y,
            "component_width": w, "component_height": h,
            "area": area,
            "cx": float(centroids[label, 0]), "cy": float(centroids[label, 1]),
            "score": float(response[labels == label].max()),
        })
    return sorted(rows, key=lambda r: r["score"], reverse=True)


def display_box(row, width, height, minimum=18):
    cx, cy = row["cx"], row["cy"]
    bw = max(minimum, row["component_width"] + 6)
    bh = max(minimum, row["component_height"] + 6)
    x1 = max(0, int(round(cx - bw / 2))); y1 = max(0, int(round(cy - bh / 2)))
    x2 = min(width - 1, int(round(cx + bw / 2))); y2 = min(height - 1, int(round(cy + bh / 2)))
    return x1, y1, x2, y2


def render(rec, top, threshold, path):
    image = Image.fromarray(rec["rgb"].copy())
    draw = ImageDraw.Draw(image)
    line = max(2, rec["width"] // 700)
    for rank, row in enumerate(top, 1):
        box = display_box(row, rec["width"], rec["height"])
        draw.rectangle(box, outline=(255, 0, 0), width=line)
        tx, ty = box[0], max(42, box[1] - 16)
        draw.rectangle((tx, ty, tx + 25, ty + 15), fill=(0, 0, 0))
        draw.text((tx + 2, ty), str(rank), fill=(255, 255, 0))
    for shape in fa.point_shapes(rec):
        draw.rectangle(shape["box"], outline=(0, 255, 0), width=line)
    draw.rectangle((5, 5, min(rec["width"] - 5, 830), 38), fill=(0, 0, 0))
    draw.text((12, 12), f"Gaussian sigma=8 top 10 (red, ranked); point GT (green); threshold={threshold:.3f}", fill=(255, 255, 255))
    image.save(path)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def make_sheets(paths):
    thumb_size = (640, 480)
    for page, start in enumerate(range(0, len(paths), 4), 1):
        selected = paths[start:start + 4]
        sheet = Image.new("RGB", (1280, 960), "white")
        for index, path in enumerate(selected):
            image = Image.open(path).convert("RGB")
            image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
            x = (index % 2) * 640 + (640 - image.width) // 2
            y = (index // 2) * 480 + (480 - image.height) // 2
            sheet.paste(image, (x, y))
        sheet.save(OUT / f"top10_review_page_{page:02d}.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ANALYSIS / "parameter_summary.json").read_text(encoding="utf-8"))
    threshold = float(config["base_threshold"])
    rows, rendered = [], []
    for rec in fa.load_records(DATA):
        gray = rec["gray"].astype(np.float32)
        response = cv2.GaussianBlur(gray, (0, 0), 8, borderType=cv2.BORDER_REFLECT_101) - gray
        top = candidates(response, threshold)[:10]
        output = OUT / f"{rec['name']}_gaussian_s8_top10.png"
        render(rec, top, threshold, output); rendered.append(output)
        for rank, row in enumerate(top, 1):
            rows.append({"image": rec["name"], "rank": rank, "sigma": 8,
                         "threshold": threshold, **row,
                         "display_box": list(display_box(row, rec["width"], rec["height"]))})
        print(rec["name"], len(top), flush=True)
    write_csv(OUT / "top10_candidates.csv", rows)
    make_sheets(rendered)


if __name__ == "__main__":
    main()
