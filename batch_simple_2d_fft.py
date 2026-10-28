"""Run plain 2-D FFT on every original and existing trimmed image.

Usage: python batch_simple_2d_fft.py
Only grayscale conversion precedes FFT: no resize, mean subtraction or window.
"""

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from analyze_2d_fft import read_grayscale


ROOT = Path(__file__).resolve().parent


def fft_spectrum(gray):
    """Centered log amplitude (natural log), computed at original resolution."""
    fft = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))
    return np.log1p(np.abs(fft))


def image_number(path):
    return int(re.search(r"image(\d+)", path.stem).group(1))


def save_overview(records, kind, output, upper):
    selected = [r for r in records if r["kind"] == kind]
    if not selected:
        return
    columns = 4
    rows = math.ceil(len(selected) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(14, 3 * rows),
                             squeeze=False, layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax, record in zip(axes.flat, selected):
        with Image.open(output / record["spectrum_png"]) as image:
            # Resizing is for this overview only; FFT uses all input pixels.
            image.thumbnail((600, 600))
            ax.imshow(np.asarray(image).copy(), interpolation="nearest")
        ax.set_title(f"image{record['number']} ({record['width']} x {record['height']})")
    fig.suptitle(f"{kind.capitalize()}: plain 2D FFT / log(1 + |F|)\n"
                 f"Shared grayscale: 0 to {upper:.3f}; DC at center")
    fig.savefig(output / f"overview_{kind}.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=ROOT / "data20260903" / "data20260903")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs" / "simple_2d_fft")
    args = parser.parse_args()
    source, output = args.input_dir.resolve(), args.output_dir.resolve()
    paths = {
        "original": sorted(source.glob("image[0-9]*.png"), key=image_number),
        "trimmed": sorted((source / "triming").glob("image[0-9]*_triming.png"),
                          key=image_number),
    }
    if not any(paths.values()):
        parser.error(f"No input images found in {source}")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    # Nonnegative grayscale images have their largest FFT amplitude at DC:
    # abs(F[0,0]) == gray.sum(). This gives one shared scale without caching FFTs.
    upper = 0.0
    for kind, files in paths.items():
        (output / kind).mkdir(exist_ok=True)
        for path in files:
            gray = read_grayscale(path)
            upper = max(upper, float(np.log1p(gray.sum(dtype=np.float64))))
            records.append({"number": image_number(path), "kind": kind,
                            "source": str(path), "width": gray.shape[1],
                            "height": gray.shape[0],
                            "spectrum_png": f"{kind}/{path.stem}_fft.png"})
    upper = max(upper, 1e-12)
    comparisons = output / "comparisons"
    comparisons.mkdir(exist_ok=True)
    for number in sorted({r["number"] for r in records}):
        pair = [r for r in records if r["number"] == number]
        fig, axes = plt.subplots(len(pair), 2, figsize=(11, 4.6 * len(pair)),
                                 squeeze=False, layout="constrained")
        for row, record in enumerate(pair):
            gray = read_grayscale(record["source"])
            spectrum = fft_spectrum(gray)
            plt.imsave(output / record["spectrum_png"], spectrum,
                       cmap="gray", vmin=0, vmax=upper)
            axes[row, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
            axes[row, 0].set_title(f"{record['kind'].capitalize()}: image{number}\n"
                                   f"{record['width']} x {record['height']} pixels")
            plotted = axes[row, 1].imshow(spectrum, cmap="gray", vmin=0,
                                          vmax=upper, interpolation="nearest")
            axes[row, 1].set_title("2D FFT: log(1 + |F|), DC at center")
            for ax in axes[row]:
                ax.axis("off")
        fig.colorbar(plotted, ax=list(axes[:, 1]), label="Log amplitude", shrink=.8)
        fig.suptitle("Plain FFT | no mean subtraction | no window | no resize")
        fig.savefig(comparisons / f"image{number}_comparison.png", dpi=140)
        plt.close(fig)
        print(f"image{number}: {', '.join(r['kind'] for r in pair)}", flush=True)
    for kind in paths:
        save_overview(records, kind, output, upper)
    report = {
        "formula": "log1p(abs(fftshift(fft2(grayscale_8bit))))",
        "preprocessing": "8-bit grayscale only; original resolution",
        "display_limits": [0, upper],
        "counts": {kind: len(files) for kind, files in paths.items()},
        "originals_without_trim": sorted(
            {r["number"] for r in records if r["kind"] == "original"}
            - {r["number"] for r in records if r["kind"] == "trimmed"}),
        "note": "Unnormalized FFT amplitude depends on image size. PNGs are display values, not raw FFT arrays.",
        "images": records,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Done: {len(records)} images -> {output}")


if __name__ == "__main__":
    main()
