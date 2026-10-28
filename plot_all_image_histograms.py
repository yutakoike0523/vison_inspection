"""Create 256-level grayscale histograms for image1 through image21."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from colorize_grid_lighting import select_bands


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data20260903" / "data20260903"
OUTPUT = ROOT / "outputs" / "all_image_histograms"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = [SOURCE / f"image{i}.png" for i in range(1, 22)]
    records = []
    fig, axes = plt.subplots(7, 3, figsize=(16, 24), constrained_layout=True)

    for index, (path, ax) in enumerate(zip(files, axes.flat), start=1):
        if not path.exists():
            ax.set_title(f"image{index}: missing")
            ax.axis("off")
            continue
        gray = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        counts = np.bincount(gray.ravel(), minlength=256)
        levels = np.arange(256)
        bands = select_bands(gray) if 2 <= index <= 18 else None
        records.append((index, gray.shape[1], gray.shape[0], counts, bands))

        ax.plot(levels, counts, color="#555555", linewidth=1.0)
        if bands:
            d0, d1 = bands["dark_range"]
            b0, b1 = bands["bright_range"]
            ax.axvspan(d0 - 0.5, d1 + 0.5, color="#0d52c2", alpha=0.45,
                       label=f"Blue {d0}-{d1}")
            ax.axvspan(b0 - 0.5, b1 + 0.5, color="#f04d0d", alpha=0.45,
                       label=f"Orange {b0}-{b1}")
            ax.set_title(f"image{index}: blue {d0}-{d1}, orange {b0}-{b1}")
        else:
            ax.set_title(f"image{index}: no blue/orange overlay")
        ax.set_xlim(0, 255)
        ax.set_xlabel("Brightness (0-255)")
        ax.set_ylabel("Pixel count")
        ax.grid(True, alpha=0.22)

        individual, iax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        iax.plot(levels, counts, color="#555555", linewidth=1.2)
        if bands:
            d0, d1 = bands["dark_range"]
            b0, b1 = bands["bright_range"]
            iax.axvspan(d0 - 0.5, d1 + 0.5, color="#0d52c2", alpha=0.45,
                        label=f"Blue in image: brightness {d0}-{d1}")
            iax.axvspan(b0 - 0.5, b1 + 0.5, color="#f04d0d", alpha=0.45,
                        label=f"Orange in image: brightness {b0}-{b1}")
            iax.legend(loc="upper right", fontsize=8)
        iax.set_title(f"image{index} grayscale histogram ({gray.shape[1]}x{gray.shape[0]})")
        iax.set_xlim(0, 255)
        iax.set_xlabel("Brightness (0-255)")
        iax.set_ylabel("Pixel count")
        iax.grid(True, alpha=0.25)
        individual.savefig(OUTPUT / f"image{index}_histogram.png", dpi=160)
        plt.close(individual)

    fig.suptitle("Grayscale histograms: shaded ranges match blue/orange pixels in images 2-18", fontsize=16)
    fig.savefig(OUTPUT / "all_images_histograms.png", dpi=170)
    plt.close(fig)

    csv_path = OUTPUT / "all_images_histograms.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        f.write("image,width,height,brightness,pixel_count,blue_min,blue_max,orange_min,orange_max\n")
        for index, width, height, counts, bands in records:
            for brightness, count in enumerate(counts):
                blue = bands["dark_range"] if bands else ["", ""]
                orange = bands["bright_range"] if bands else ["", ""]
                f.write(f"image{index},{width},{height},{brightness},{int(count)},{blue[0]},{blue[1]},{orange[0]},{orange[1]}\n")

    print(f"Created {len(records)} individual histograms and contact sheet: {OUTPUT}")
    print(f"Contact sheet: {OUTPUT / 'all_images_histograms.png'}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
