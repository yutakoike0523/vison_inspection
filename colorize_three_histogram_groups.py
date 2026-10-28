"""Color three histogram brightness groups on selected source images."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data20260903" / "data20260903"
OUTPUT = ROOT / "outputs" / "three_histogram_groups"
IMAGE_NUMBERS = [4, 7, 8, 11, 13, 15, 16, 18]
COLORS = [np.array(c, dtype=np.float32) for c in [
    (30, 100, 220), (245, 105, 35), (25, 170, 85)
]]


def three_class_otsu(counts):
    """Find two thresholds maximizing between-class variance."""
    p = counts.astype(np.float64)
    levels = np.arange(256, dtype=np.float64)
    w = np.r_[0.0, np.cumsum(p)]
    m = np.r_[0.0, np.cumsum(p * levels)]
    total_mean = m[-1] / w[-1]
    best_score, best = -1.0, (85, 170)
    for t1 in range(1, 254):
        w0 = w[t1 + 1]
        if w0 == 0:
            continue
        mean0 = m[t1 + 1] / w0
        t2s = np.arange(t1 + 1, 255)
        w1 = w[t2s + 1] - w[t1 + 1]
        w2 = w[-1] - w[t2s + 1]
        valid = (w1 > 0) & (w2 > 0)
        if not valid.any():
            continue
        m1 = m[t2s + 1] - m[t1 + 1]
        m2 = m[-1] - m[t2s + 1]
        means1 = np.divide(m1, w1, out=np.zeros_like(m1), where=w1 > 0)
        means2 = np.divide(m2, w2, out=np.zeros_like(m2), where=w2 > 0)
        scores = w0 * (mean0-total_mean)**2 + w1 * (means1-total_mean)**2 + w2 * (means2-total_mean)**2
        scores[~valid] = -1
        j = int(np.argmax(scores))
        if scores[j] > best_score:
            best_score, best = float(scores[j]), (t1, int(t2s[j]))
    return best


def analyze(gray):
    counts = np.bincount(gray.ravel(), minlength=256)
    smooth = gaussian_filter1d(counts.astype(float), 2)
    # Count stable histogram modes, rather than forcing every image into
    # three intensity classes. This keeps small shoulders within one broad
    # hill from becoming separate colors.
    candidates, _ = find_peaks(smooth, distance=6, prominence=smooth.max() * 0.003)
    prominences = peak_prominences(smooth, candidates)[0]
    order = np.argsort(prominences)[::-1][:3]
    peaks = sorted(int(candidates[i]) for i in order)
    # A cropped image can have one broad dominant mode. Keep it as one valid
    # band instead of rejecting the image; downstream code can still build a
    # candidate mask from that mode.
    if not peaks:
        peaks = [int(np.argmax(smooth))]
    boundaries = []
    for left, right in zip(peaks[:-1], peaks[1:]):
        boundaries.append(left + int(np.argmin(smooth[left:right + 1])))
    color_ranges = []
    for idx, peak in enumerate(peaks):
        neighbor_gaps = []
        if idx:
            neighbor_gaps.append(peak - peaks[idx - 1])
        if idx + 1 < len(peaks):
            neighbor_gaps.append(peaks[idx + 1] - peak)
        radius = (int(np.clip(round(min(neighbor_gaps) / 4), 4, 12))
                  if neighbor_gaps else 8)
        color_ranges.append([max(0, peak - radius), min(255, peak + radius)])
    return counts, smooth, color_ranges, peaks, boundaries


def tint(gray, ranges, alpha=0.66):
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)
    masks = []
    for (lo, hi), color in zip(ranges, COLORS):
        mask = (gray >= lo) & (gray <= hi)
        masks.append(mask)
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * color
    return np.clip(rgb, 0, 255).astype(np.uint8), masks


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records, tiles = [], []
    for number in IMAGE_NUMBERS:
        path = SOURCE / f"image{number}.png"
        gray = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        counts, smooth, ranges, peaks, thresholds = analyze(gray)
        colored, masks = tint(gray, ranges)
        Image.fromarray(colored).save(OUTPUT / f"image{number}_three_colors.png")

        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.plot(np.arange(256), counts, color="#555555", linewidth=1, label="Histogram")
        for idx, ((lo, hi), peak, color) in enumerate(zip(ranges, peaks, COLORS), start=1):
            hex_color = "#%02x%02x%02x" % tuple(color.astype(int))
            ax.axvspan(lo - .5, hi + .5, color=hex_color, alpha=.32,
                       label=f"Color {idx}: {lo}-{hi}; peak {peak}")
            ax.axvline(peak, color=hex_color, linewidth=1.4)
        ax.set(xlim=(0, 255), xlabel="Brightness (0-255)", ylabel="Pixel count",
               title=f"image{number}: highlighted peak bands")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        fig.savefig(OUTPUT / f"image{number}_histogram_groups.png", dpi=150)
        plt.close(fig)

        records.append({
            "image": f"image{number}.png", "valley_boundaries": list(thresholds),
            "highlight_ranges": ranges, "peak_positions": peaks,
            "pixel_counts": [int(m.sum()) for m in masks],
            "method": "up to three strongest stable local peaks; narrow bands around peaks with half-width 4-12 levels; histogram smoothing sigma=2"
        })
        thumb = Image.fromarray(colored).convert("RGB")
        thumb.thumbnail((480, 360))
        tile = Image.new("RGB", (500, 400), "white")
        tile.paste(thumb, ((500-thumb.width)//2, 30))
        draw = ImageDraw.Draw(tile)
        draw.text((10, 8), f"image{number}  {len(peaks)} stable peak band(s)", fill="black")
        tiles.append(tile)

    sheet = Image.new("RGB", (1000, 1600), "white")
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % 2) * 500, (idx // 2) * 400))
    sheet.save(OUTPUT / "all_eight_colorized.png")
    (OUTPUT / "settings.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
