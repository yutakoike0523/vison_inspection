"""Apply brightness bands found in trimmed crops back onto the full images."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from colorize_three_histogram_groups import COLORS, tint


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data20260903" / "data20260903"
SUMMARY = ROOT / "outputs" / "trimmed_histograms" / "trimmed_histogram_summary.json"
OUT = ROOT / "outputs" / "trimmed_histograms" / "originals_colorized_by_trim_bands"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = json.loads(SUMMARY.read_text(encoding="utf-8"))
    tiles = []
    pair_tiles = []
    result = []
    for record in records:
        number = record["image_number"]
        source_path = SOURCE / f"image{number}.png"
        trimmed_path = ROOT / record["trimmed_file"]
        gray = np.asarray(Image.open(source_path).convert("L"), dtype=np.uint8)
        trim_gray = np.asarray(Image.open(trimmed_path).convert("L"), dtype=np.uint8)
        bands = record["trimmed_peak_bands_inclusive"]
        colorized, masks = tint(gray, bands)
        trim_colorized, trim_masks = tint(trim_gray, bands)
        output_path = OUT / f"image{number}_trim_bands_on_original.png"
        trim_output_path = OUT / f"image{number}_trimmed_colorized.png"
        pair_output_path = OUT / f"image{number}_trim_and_original_colorized.png"
        Image.fromarray(colorized).save(output_path)
        Image.fromarray(trim_colorized).save(trim_output_path)

        pair = Image.new("RGB", (1200, 700), "white")
        draw_pair = ImageDraw.Draw(pair)
        draw_pair.text((20, 12), f"image{number}: trimmed image bands {bands}", fill="black")
        left = ImageOps.contain(Image.fromarray(trim_colorized).convert("RGB"), (570, 640))
        right = ImageOps.contain(Image.fromarray(colorized).convert("RGB"), (570, 640))
        pair.paste(left, (20 + (570-left.width)//2, 45 + (640-left.height)//2))
        pair.paste(right, (610 + (570-right.width)//2, 45 + (640-right.height)//2))
        draw_pair.text((20, 680), f"Trimmed: {trimmed_path.name}", fill="black")
        draw_pair.text((610, 680), f"Original: {source_path.name}", fill="black")
        pair.save(pair_output_path)

        preview = Image.fromarray(colorized).convert("RGB")
        preview.thumbnail((440, 330))
        tile = Image.new("RGB", (480, 380), "white")
        tile.paste(preview, ((480-preview.width)//2, 32))
        draw = ImageDraw.Draw(tile)
        band_text = ", ".join(f"{lo}-{hi}" for lo, hi in bands)
        draw.text((10, 8), f"image{number}: trim bands {band_text}", fill="black")
        tiles.append(tile)
        small_pair = Image.new("RGB", (640, 390), "white")
        small_draw = ImageDraw.Draw(small_pair)
        small_draw.text((8, 7), f"image{number}: bands {band_text}", fill="black")
        trim_thumb = ImageOps.contain(Image.fromarray(trim_colorized).convert("RGB"), (310, 340))
        original_thumb = ImageOps.contain(Image.fromarray(colorized).convert("RGB"), (310, 340))
        small_pair.paste(trim_thumb, (5 + (310-trim_thumb.width)//2, 35 + (340-trim_thumb.height)//2))
        small_pair.paste(original_thumb, (325 + (310-original_thumb.width)//2, 35 + (340-original_thumb.height)//2))
        small_draw.text((10, 372), "Trimmed", fill="black")
        small_draw.text((330, 372), "Original", fill="black")
        pair_tiles.append(small_pair)
        result.append({
            "image_number": number,
            "source_file": str(source_path.relative_to(ROOT)),
            "trimmed_brightness_bands_inclusive": bands,
            "colored_pixels_per_band_in_original": [int(mask.sum()) for mask in masks],
            "output_file": str(output_path.relative_to(ROOT)),
            "note": "Brightness bands were detected from the trimmed crop, then applied to every matching grayscale pixel in the full original image.",
        })

    columns = 3
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 480, rows * 380), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns) * 480, (i // columns) * 380))
    sheet_path = OUT / "all_originals_colorized_by_trim_bands.png"
    sheet.save(sheet_path)
    pair_columns = 2
    pair_rows = (len(pair_tiles) + pair_columns - 1) // pair_columns
    pairs = Image.new("RGB", (pair_columns * 640, pair_rows * 390), "white")
    for i, tile in enumerate(pair_tiles):
        pairs.paste(tile, ((i % pair_columns) * 640, (i // pair_columns) * 390))
    pairs_path = OUT / "all_trimmed_and_original_colorized.png"
    pairs.save(pairs_path)
    (OUT / "colorization_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Colorized {len(result)} original images using bands from trimmed-image histograms.")
    print(f"Colorized trimmed crops as well; same bands are applied to both crop and original.")
    print(f"Contact sheet: {sheet_path}")
    print(f"Trimmed + original comparison: {pairs_path}")
    for item in result:
        print(f"image{item['image_number']}: bands={item['trimmed_brightness_bands_inclusive']} "
              f"pixels={item['colored_pixels_per_band_in_original']}")


if __name__ == "__main__":
    main()
