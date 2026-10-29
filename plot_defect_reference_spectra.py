#!/usr/bin/env python3
"""Plot target-defect and surrounding-reference 1-D radial FFT spectra."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle

import radial_fft_experiment as radial
import window_size_comparison_unnormalized as experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=int, default=15)
    parser.add_argument("--fixed-annotation-index", type=int, default=None,
                        help="Keep one defect annotation fixed across all window sizes")
    parser.add_argument("--reference-policy", choices=("strict8", "available"), default="available")
    parser.add_argument("--reference-aggregation", choices=("mean", "median"), default="mean")
    parser.add_argument("--data", type=Path, default=experiment.DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    image_path = args.data / f"image{args.image}.png"
    image = np.asarray(Image.open(image_path).convert("L"), np.float32)
    rows: list[dict] = []
    selections: list[dict] = []

    for window in experiment.WINDOWS:
        xs, ys = experiment.target_centres(image.shape, window, args.reference_policy)
        if not len(xs) or not len(ys):
            continue
        coords = experiment.required_coordinates(xs, ys, window, args.reference_policy)
        spectra, cache, valid_bins, _ = experiment.calculate_spectra(
            image, coords, window, args.workers, args.batch_size
        )
        scores, _ = experiment.calculate_score(spectra, cache, xs, ys, window)
        masks = experiment.masks_for(args.image, image.shape, xs, ys, args.data)
        fixed_instance = None
        if args.fixed_annotation_index is not None:
            fixed_instance = next(
                (item for item in masks["instances"]
                 if item["annotation_index"] == args.fixed_annotation_index), None
            )
            if fixed_instance is None:
                raise ValueError(
                    f"annotation index {args.fixed_annotation_index} is not an evaluated defect"
                )
            candidates = fixed_instance["selection"] & np.isfinite(scores)
        else:
            candidates = masks["defect"] & np.isfinite(scores)
        if not candidates.any():
            continue
        if fixed_instance is None:
            candidate_scores = np.where(candidates, scores, -np.inf)
            iy, ix = np.unravel_index(np.argmax(candidate_scores), scores.shape)
            selection_rule = "maximum-score defect window"
            annotation_index = None
            defect_label = None
        else:
            label_record = next(
                (item for item in masks["labels"]
                 if item[0] == fixed_instance["label"] and np.array_equal(item[1],
                    masks["labels"][fixed_instance["annotation_index"]][1])),
                masks["labels"][fixed_instance["annotation_index"]],
            )
            py, px = np.nonzero(label_record[1])
            anchor_x, anchor_y = float(px.mean()), float(py.mean())
            grid_x, grid_y = np.meshgrid(xs, ys)
            distance2 = (grid_x-anchor_x)**2 + (grid_y-anchor_y)**2
            iy, ix = np.unravel_index(np.argmin(np.where(candidates, distance2, np.inf)), scores.shape)
            selection_rule = f"nearest valid centre to fixed annotation {args.fixed_annotation_index}"
            annotation_index = args.fixed_annotation_index
            defect_label = fixed_instance["label"]
        x, y = int(xs[ix]), int(ys[iy])
        target = spectra[cache[(x, y)]]
        ref_coords = [
            (x + dx * window, y + dy * window)
            for dy, dx in experiment.OFFSETS
            if (x + dx * window, y + dy * window) in cache
        ]
        refs = np.stack([spectra[cache[p]] for p in ref_coords])
        reference = (refs.mean(axis=0) if args.reference_aggregation == "mean"
                     else np.median(refs, axis=0))
        difference = target - reference
        aggregation = args.reference_aggregation
        comparison_rms = float(np.sqrt(np.mean(difference**2)))
        _, _, _, specs = radial.grids(window, experiment.BIN_WIDTH, [experiment.FREQ])
        valid = specs[experiment.FREQ]["valid"]
        frequency = ((np.arange(len(valid)) + 0.5) * experiment.BIN_WIDTH)[valid]
        assert len(frequency) == valid_bins == len(target)

        selections.append({
            "window": window, "x": x, "y": y, "score": comparison_rms,
            "annotation_index": annotation_index, "defect_label": defect_label,
            "selection_rule": selection_rule,
            "reference_aggregation": aggregation,
            "frequency": frequency, "target": target, "refs": refs,
            "reference": reference, "difference": difference, "ref_coords": ref_coords,
        })
        for k, f in enumerate(frequency):
            row = {
                "image_id": f"image{args.image}", "window_size": window,
                "fixed_annotation_index": annotation_index, "defect_label": defect_label,
                "selection_rule": selection_rule,
                "target_x": x, "target_y": y, "reference_count": len(ref_coords),
                "reference_aggregation": aggregation,
                "comparison_rms_db": comparison_rms,
                "frequency_cycles_per_pixel": float(f),
                "target_db": float(target[k]),
                f"reference_{aggregation}_db": float(reference[k]),
                f"target_minus_reference_{aggregation}_db": float(difference[k]),
            }
            for ri, values in enumerate(refs, 1):
                row[f"reference_{ri}_db"] = float(values[k])
            rows.append(row)

        fig, (ax0, ax1, ax2) = plt.subplots(
            1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.05, 1.55, 1.3]}
        )
        ax0.imshow(image, cmap="gray", vmin=0, vmax=255)
        half = window / 2
        ax0.add_patch(Rectangle((x-half, y-half), window, window, fill=False,
                                edgecolor="#ff3b30", linewidth=2.0, label="defect target"))
        for ri, (rx, ry) in enumerate(ref_coords):
            ax0.add_patch(Rectangle((rx-half, ry-half), window, window, fill=False,
                                    edgecolor="#00d8ff", linewidth=1.0,
                                    label="available references" if ri == 0 else None))
        ax0.scatter([x], [y], c="#ff3b30", s=18)
        ax0.set_title(f"positions: target=({x}, {y}), refs={len(ref_coords)}")
        ax0.legend(loc="lower left", fontsize=8)
        ax0.axis("off")

        for ri, values in enumerate(refs):
            ax1.plot(frequency, values, color="0.65", alpha=.65, linewidth=.8,
                     label="individual references" if ri == 0 else None)
        ax1.plot(frequency, reference, color="#0072b2", linewidth=2.1,
                 label=f"reference {aggregation}")
        ax1.plot(frequency, target, color="#d55e00", linewidth=2.1, label="defect target")
        ax1.set(xlabel="frequency (cycles/pixel)", ylabel="radial power spectrum (dB)",
                title="1-D radial FFT spectra (no power normalisation)")
        ax1.grid(alpha=.25); ax1.legend(fontsize=8)

        ax2.axhline(0, color="0.35", linewidth=.8)
        ax2.plot(frequency, difference, color="#cc0066", linewidth=2)
        ax2.fill_between(frequency, 0, difference, color="#cc0066", alpha=.18)
        ax2.set(xlabel="frequency (cycles/pixel)",
                ylabel=f"target - reference {aggregation} (dB)",
                title=f"spectral difference; RMS={comparison_rms:.3f} dB")
        ax2.grid(alpha=.25)
        fixed_text = (f", fixed defect annotation={annotation_index}" if annotation_index is not None else "")
        fig.suptitle(
            f"image{args.image}, W={window}, d=W, {args.reference_policy} references, "
            f"reference={aggregation}{fixed_text}"
        )
        fig.tight_layout()
        suffix = "" if aggregation == "mean" else f"_{aggregation}"
        fig.savefig(args.output / f"image{args.image}_W{window}_spectra{suffix}.png", dpi=180)
        plt.close(fig)

    if selections:
        fig, axes = plt.subplots(len(selections), 2, figsize=(13, 3.1*len(selections)), squeeze=False)
        for row_index, item in enumerate(selections):
            ax, diff_ax = axes[row_index]
            f = item["frequency"]
            for ri, values in enumerate(item["refs"]):
                ax.plot(f, values, color="0.72", alpha=.55, linewidth=.7,
                        label="individual refs" if ri == 0 else None)
            ax.plot(f, item["reference"], color="#0072b2", linewidth=1.8,
                    label=f"reference {item['reference_aggregation']}")
            ax.plot(f, item["target"], color="#d55e00", linewidth=1.8, label="defect target")
            ax.set_title(f"W={item['window']}, target=({item['x']},{item['y']}), refs={len(item['ref_coords'])}")
            ax.set_ylabel("power (dB)"); ax.grid(alpha=.2); ax.legend(fontsize=7, ncol=3)
            diff_ax.axhline(0, color="0.35", linewidth=.7)
            diff_ax.plot(f, item["difference"], color="#cc0066", linewidth=1.7)
            diff_ax.fill_between(f, 0, item["difference"], color="#cc0066", alpha=.16)
            diff_ax.set_title(
                f"target - reference {item['reference_aggregation']}; RMS={item['score']:.3f} dB"
            )
            diff_ax.set_ylabel("difference (dB)"); diff_ax.grid(alpha=.2)
            if row_index == len(selections)-1:
                ax.set_xlabel("frequency (cycles/pixel)")
                diff_ax.set_xlabel("frequency (cycles/pixel)")
        fixed_text = (f", fixed defect annotation={args.fixed_annotation_index}"
                      if args.fixed_annotation_index is not None else "")
        fig.suptitle(
            f"image{args.image}: defect-window vs surrounding-window 1-D radial FFT, "
            f"reference={args.reference_aggregation}{fixed_text}"
        )
        fig.tight_layout()
        suffix = "" if args.reference_aggregation == "mean" else f"_{args.reference_aggregation}"
        fig.savefig(args.output / f"image{args.image}_all_windows_spectra{suffix}.png", dpi=180)
        plt.close(fig)

    fields = list(dict.fromkeys(key for row in rows for key in row))
    suffix = "" if args.reference_aggregation == "mean" else f"_{args.reference_aggregation}"
    with (args.output / f"image{args.image}_spectral_values{suffix}.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
