"""Redraw saved 4x4 FFT spectra with user-selected scratch regions in red."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(image_name, regions):
    if not regions or any(not isinstance(region, int) or region < 1 or region > 16 for region in regions):
        raise ValueError("Region numbers must be integers between 1 and 16.")
    root = Path(__file__).resolve().parent / "outputs" / "⑦_16分割FFT比較"
    directory = root / image_name
    with (directory / "features.csv").open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    spectra = {int(record["region"]): [] for record in records}
    with (directory / "radial_spectra.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            spectra[int(row["region"])].append(row)

    selected = set(regions)
    scores = [float(record["fft_difference_score"]) for record in records]
    norm = plt.Normalize(min(scores), max(scores) + 1e-9)
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.subplots_adjust(left=.08, right=.74, bottom=.09, top=.87, hspace=.20)
    handles = {}
    # Draw the selected curves last so overlapping curves do not hide them.
    for record in sorted(records, key=lambda row: int(row["region"]) in selected):
        region = int(record["region"])
        score = float(record["fft_difference_score"])
        rows = spectra[region]
        frequency = [float(row["frequency_cycles_per_pixel"]) for row in rows]
        magnitude = [float(row["mean_magnitude"]) for row in rows]
        power = [float(row["radial_bin_power_fraction"]) for row in rows]
        highlight = region in selected
        style = dict(color="red" if highlight else cmap(norm(score)),
                     linewidth=2.2 if highlight else 1.0,
                     alpha=1.0 if highlight else .65,
                     zorder=5 if highlight else 2)
        axes[0].plot(frequency, np.log1p(magnitude), **style)
        label = f"R{region:02d} (E={score:.2f})" + (" [scratch]" if highlight else "")
        handles[region], = axes[1].plot(frequency, power, label=label, **style)
    axes[0].set_ylabel("log(1 + mean |FFT|)")
    axes[0].set_title("Radial mean amplitude (1-D spectrum)")
    axes[1].set(xlabel="Spatial frequency [cycles/pixel]",
                ylabel="Fraction of non-DC power per radial bin",
                title="Radial power distribution (normalized per region)")
    for ax in axes:
        ax.grid(alpha=.25)
    legend_handles = [handles[region] for region in sorted(handles)]
    legend = fig.legend(legend_handles, [handle.get_label() for handle in legend_handles],
                        loc="center left", bbox_to_anchor=(.83, .5), fontsize=7,
                        title="Region / score")
    for label, region in zip(legend.get_texts(), sorted(handles)):
        if region in selected:
            label.set_color("red")
            label.set_weight("bold")
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar_ax = fig.add_axes([.765, .17, .016, .62])
    fig.colorbar(scalar, cax=colorbar_ax, label="FFT difference score")
    regions_text = ", ".join(f"R{region:02d}" for region in sorted(selected))
    fig.suptitle("16 regions: frequency vs strength (radial average; direction is collapsed)\n"
                 f"{image_name}: scratch regions in red: {regions_text}", fontsize=12)
    output = directory / "03_fft_16regions_1d_scratch_red.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="image7")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--regions", type=int, nargs="+")
    selection.add_argument("--annotations", type=Path,
                           help="JSON mapping image names to user-selected region numbers")
    args = parser.parse_args()
    if args.annotations:
        annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
        for image_name, regions in annotations.items():
            render(image_name, regions)
    else:
        render(args.image, args.regions)


if __name__ == "__main__":
    main()
