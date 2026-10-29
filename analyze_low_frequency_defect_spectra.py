#!/usr/bin/env python3
"""Measure target-vs-reference spectral differences below 0.1 cycles/pixel."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

import radial_fft_experiment as radial
import window_size_comparison_unnormalized as experiment


LABEL_JA = {
    experiment.labels_impl.CLASSES[0]: "局所変形",
    experiment.labels_impl.CLASSES[1]: "点状異常",
    experiment.labels_impl.CLASSES[2]: "線状異常",
    experiment.labels_impl.CLASSES[3]: "面状異常",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=experiment.DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for image_id in range(1, 22):
        path = args.data / f"image{image_id}.png"
        if not path.exists():
            continue
        image = np.asarray(Image.open(path).convert("L"), np.float32)
        for window in experiment.WINDOWS:
            xs, ys = experiment.target_centres(image.shape, window, "available")
            if not len(xs) or not len(ys):
                continue
            masks = experiment.masks_for(image_id, image.shape, xs, ys, args.data)
            grid_x, grid_y = np.meshgrid(xs, ys)
            xs_set, ys_set = set(map(int, xs)), set(map(int, ys))
            chosen: list[tuple[dict, int, int]] = []
            all_coords: set[tuple[int, int]] = set()
            for instance in masks["instances"]:
                candidates = instance["selection"]
                if not candidates.any():
                    continue
                annotation_mask = masks["labels"][instance["annotation_index"]][1]
                py, px = np.nonzero(annotation_mask)
                anchor_x, anchor_y = float(px.mean()), float(py.mean())
                distance2 = (grid_x-anchor_x)**2 + (grid_y-anchor_y)**2
                iy, ix = np.unravel_index(np.argmin(np.where(candidates, distance2, np.inf)), candidates.shape)
                x, y = int(xs[ix]), int(ys[iy])
                chosen.append((instance, x, y))
                all_coords.add((x, y))
                all_coords.update(
                    (x+dx*window, y+dy*window)
                    for dy, dx in experiment.OFFSETS
                    if x+dx*window in xs_set and y+dy*window in ys_set
                )
            if not chosen:
                continue
            coords = sorted(all_coords, key=lambda p: (p[1], p[0]))
            spectra, cache, _, _ = experiment.calculate_spectra(
                image, coords, window, args.workers, args.batch_size
            )
            _, _, _, specs = radial.grids(window, experiment.BIN_WIDTH, [experiment.FREQ])
            valid = specs[experiment.FREQ]["valid"]
            frequency = ((np.arange(len(valid))+.5)*experiment.BIN_WIDTH)[valid]
            low = frequency <= 0.100 + 1e-12
            for instance, x, y in chosen:
                ref_coords = [
                    (x+dx*window, y+dy*window)
                    for dy, dx in experiment.OFFSETS
                    if (x+dx*window, y+dy*window) in cache
                ]
                refs = np.stack([spectra[cache[p]] for p in ref_coords])
                target = spectra[cache[(x, y)]]
                reference = np.median(refs, axis=0)
                difference = target-reference
                low_difference = difference[low]
                rows.append({
                    "image_id": f"image{image_id}",
                    "annotation_index": instance["annotation_index"],
                    "defect_label": LABEL_JA.get(instance["label"], instance["label"]),
                    "window_size": window,
                    "target_x": x, "target_y": y,
                    "reference_count": len(ref_coords),
                    "low_frequency_min": float(frequency[low].min()),
                    "low_frequency_max": float(frequency[low].max()),
                    "low_frequency_bin_count": int(low.sum()),
                    "low_frequency_rms_db": float(np.sqrt(np.mean(low_difference**2))),
                    "low_frequency_mean_abs_db": float(np.mean(np.abs(low_difference))),
                    "low_frequency_max_abs_db": float(np.max(np.abs(low_difference))),
                    "full_band_rms_db": float(np.sqrt(np.mean(difference**2))),
                    "below_1db_low_frequency_rms": bool(np.sqrt(np.mean(low_difference**2)) <= 1.0),
                })
        print(f"image{image_id}: done", flush=True)

    rows.sort(key=lambda row: row["low_frequency_rms_db"])
    write_csv(args.output / "all_defects_low_frequency_difference.csv", rows)
    close = [row for row in rows if row["below_1db_low_frequency_rms"]]
    lines = [
        "# 傷窓と周辺中央値スペクトルの低周波差",
        "",
        "条件: 正規化なし、放射平均dBスペクトル、周辺参照は画像内に存在する窓の周波数ビン別中央値。",
        "低周波範囲は有効ビン中心が0.100 cycles/pixel以下（解析帯域下限0.020）。",
        "対象位置は各傷アノテーション中心に最も近い有効な16px評価格子点。",
        "",
        f"評価組合せ数: {len(rows)}",
        f"低周波RMSが厳密に0 dB: {sum(row['low_frequency_rms_db'] == 0 for row in rows)}",
        f"低周波RMSが1 dB以下: {len(close)}",
        "",
        "## 低周波差が小さい順",
        "",
        "|順位|画像|傷index|種類|W|中心|参照数|低周波RMS (dB)|平均絶対差 (dB)|全帯域RMS (dB)|",
        "|---:|---|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows[:20], 1):
        lines.append(
            f"|{rank}|{row['image_id']}|{row['annotation_index']}|{row['defect_label']}|"
            f"{row['window_size']}|({row['target_x']},{row['target_y']})|{row['reference_count']}|"
            f"{row['low_frequency_rms_db']:.3f}|{row['low_frequency_mean_abs_db']:.3f}|"
            f"{row['full_band_rms_db']:.3f}|"
        )
    (args.output / "low_frequency_difference_summary.md").write_text(
        "\n".join(lines)+"\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
