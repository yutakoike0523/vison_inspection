"""Create non-destructive, tag-suppressed copies of the inspection images."""
from pathlib import Path
import json
import numpy as np
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data20260903" / "data20260903"
OUT = ROOT / "data_tag_suppressed_inpainted"

def tag_mask(shape, annotations):
    mask = np.zeros(shape, np.uint8)
    for item in annotations:
        if item.get("label", "").strip().lower() not in {"tag", "札"}:
            continue
        points = np.rint(np.asarray(item.get("points", []), np.float32)).astype(np.int32)
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 255)
    # Include the dark tag boundary so it cannot remain as an artificial edge.
    return cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)

def main():
    OUT.mkdir(exist_ok=True); (OUT / "masks").mkdir(exist_ok=True)
    for image_id in range(1, 22):
        source = np.asarray(Image.open(DATA / f"image{image_id}.png").convert("L"))
        annotations = json.loads((DATA / f"image{image_id}.json").read_text(encoding="utf-8"))["shapes"]
        mask = tag_mask(source.shape, annotations)
        # Navier-Stokes inpainting preserves boundary direction better than a flat fill.
        filled = cv2.inpaint(source, mask, 9, cv2.INPAINT_NS)
        Image.fromarray(filled).save(OUT / f"image{image_id}_tag_suppressed.png")
        Image.fromarray(mask).save(OUT / "masks" / f"image{image_id}_tag_mask.png")
        print(f"image{image_id}", flush=True)

if __name__ == "__main__":
    main()
