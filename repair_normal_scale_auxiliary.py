#!/usr/bin/env python3
"""Regenerate auxiliary normal-p05/p99 sheets with one colorbar per W."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import window_size_comparison_unnormalized as experiment


def repair(data: Path, output: Path, reference_policy: str, workers: int, batch_size: int) -> None:
    destination = output / "heatmaps_normal_scale_auxiliary"
    destination.mkdir(parents=True, exist_ok=True)
    for image_id in range(1, 22):
        path = data / f"image{image_id}.png"
        if not path.exists():
            continue
        image = np.asarray(Image.open(path).convert("L"), np.float32)
        height, width = image.shape
        fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
        for ax in axes.flat:
            ax.axis("off")
        plotted = 0
        for slot, window in enumerate(experiment.WINDOWS):
            xs, ys = experiment.target_centres(image.shape, window, reference_policy)
            if not len(xs) or not len(ys):
                continue
            coords = experiment.required_coordinates(xs, ys, window, reference_policy)
            spectra, cache, _, _ = experiment.calculate_spectra(image, coords, window, workers, batch_size)
            score, _ = experiment.calculate_score(spectra, cache, xs, ys, window)
            if not np.isfinite(score).any():
                continue
            masks = experiment.masks_for(image_id, image.shape, xs, ys, data)
            record = {"image": image, "window": window, "xs": xs, "ys": ys,
                      "score": score, "masks": masks}
            lo, hi = experiment.percentile_range(score[masks["normal"] & np.isfinite(score)])
            ax = axes.flat[slot]
            ax.axis("on")
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            heat = ax.imshow(score, extent=experiment.extent(record), cmap="inferno",
                             vmin=lo, vmax=hi, alpha=.68, interpolation="nearest")
            experiment.overlay_annotations(ax, record)
            ax.set(xlim=(0, width), ylim=(height, 0), title=f"W={window}: {lo:.3f}–{hi:.3f}")
            ax.set_axis_off()
            fig.colorbar(heat, ax=ax, label="RMS (dB)", shrink=.72, pad=.02)
            plotted += 1
        if plotted:
            fig.suptitle(f"image{image_id}: auxiliary normal-region p05–p99 scales\n"
                         "Each panel has its own independent color scale")
            fig.savefig(destination / f"image{image_id}_normal_scale.png", dpi=135)
        plt.close(fig)
        print(f"image{image_id}: repaired ({plotted} W)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=experiment.DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-policy", choices=("strict8", "available"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    repair(args.data, args.output, args.reference_policy, args.workers, args.batch_size)


if __name__ == "__main__":
    main()
