"""Create viewer-compatible FFT maps from the patch-repaired inspection images."""
from __future__ import annotations
from pathlib import Path
import csv, os, time
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evaluate_all_radial_fft as radial
from strict_b8_window_maps import rms8

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data_tag_patch_matched"
OUT = ROOT / "output_tag_patch_matched_fft"
WS, STRIDE = (32, 64, 128, 256, 512), 16

def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def analyze_one(image: np.ndarray, window: int, workers: int) -> tuple[dict, np.ndarray, np.ndarray, float]:
    old_n, old_spacing = radial.N, radial.SPACING
    radial.N, radial.SPACING = window, window
    try:
        started = time.perf_counter()
        result = radial.analyze(image, workers=workers, batch=32, also_2d=False)
        score, counts = rms8(result["spectra"], window)
        return result, score, counts, time.perf_counter() - started
    finally:
        radial.N, radial.SPACING = old_n, old_spacing

def main() -> None:
    (OUT / "maps").mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    paths = [(int(p.stem.split("_")[0].replace("image", "")), p) for p in DATA.glob("image*_tag_patch.png")]
    paths.sort()
    for window in WS:
        contact: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for image_id, path in paths:
            image = np.asarray(Image.open(path).convert("L"), np.float32)
            if image.shape[0] < 2 * window or image.shape[1] < 2 * window:
                rows.append(dict(image=f"image{image_id}", window=window, status="too small for a non-overlapping reference", workers=1,
                                 fft_seconds=None, radial_seconds=None, map_seconds=None, total_seconds=0, locations=0)); continue
            result, score, counts, elapsed = analyze_one(image, window, workers=1)
            valid = score[np.isfinite(score)]
            np.savez_compressed(OUT / "maps" / f"image{image_id}_w{window}.npz", anomaly_map=score, x_centers=result["xs"], y_centers=result["ys"],
                                window=window, stride=STRIDE, reference_spacing=window, reference_counts=counts,
                                source_image=path.name, normalisation="power total 0.01-0.50", rms_band="0.01-0.50")
            contact[image_id] = (score, result["xs"], result["ys"])
            rows.append(dict(image=f"image{image_id}", window=window, status="ok", workers=1, fft_seconds=result["fft_seconds"],
                             radial_seconds=result["radial_seconds"], map_seconds=result["map_seconds"], total_seconds=elapsed, locations=len(valid)))
            print(f"map image{image_id} W={window}: {elapsed:.2f}s", flush=True)
        fig, axes = plt.subplots(7, 3, figsize=(18, 27), constrained_layout=True)
        for ax, image_id in zip(axes.flat, range(1, 22)):
            if image_id not in contact:
                ax.set_title(f"image{image_id}: no patch-repaired input"); ax.set_axis_off(); continue
            score, xs, ys = contact[image_id]; low, high = np.percentile(score[np.isfinite(score)], (1, 99))
            ax.imshow(score, extent=[xs[0]-8, xs[-1]+8, ys[-1]+8, ys[0]-8], cmap="inferno", vmin=low, vmax=high)
            ax.set_title(f"image{image_id}: own p01-p99"); ax.set_axis_off()
        fig.suptitle(f"Patch-repaired input, W={window}: normalised 0.01-0.50, RMS 0.01-0.50, surrounding 8-reference mean")
        fig.savefig(OUT / f"window_{window}_all_images_per_image_p01_p99.png", dpi=150); plt.close(fig)
    write_csv(OUT / "processing_times.csv", rows)

    # FFT-worker benchmark: use image1 (patch-repaired and a full-size image), W=128.
    image = np.asarray(Image.open(DATA / "image1_tag_patch.png").convert("L"), np.float32)
    logical = os.cpu_count() or 1
    worker_values = [value for value in (1, 2, 4, 8) if value <= logical] + [-1]
    benchmarks: list[dict] = []
    for workers in worker_values:
        for repeat in range(2):
            result, score, counts, elapsed = analyze_one(image, 128, workers=workers)
            benchmarks.append(dict(image="image1_tag_patch", window=128, workers=workers, repeat=repeat + 1,
                                   fft_seconds=result["fft_seconds"], radial_seconds=result["radial_seconds"], map_seconds=result["map_seconds"], total_seconds=elapsed))
            print(f"benchmark workers={workers}, repeat={repeat+1}: {elapsed:.2f}s", flush=True)
    write_csv(OUT / "core_benchmark.csv", benchmarks)
    means = []
    for workers in worker_values:
        group = [r for r in benchmarks if r["workers"] == workers]
        means.append(dict(workers=workers, fft_seconds=np.mean([r["fft_seconds"] for r in group]),
                          radial_seconds=np.mean([r["radial_seconds"] for r in group]), map_seconds=np.mean([r["map_seconds"] for r in group]),
                          total_seconds=np.mean([r["total_seconds"] for r in group])))
    write_csv(OUT / "core_benchmark_mean.csv", means)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["all" if r["workers"] == -1 else str(r["workers"]) for r in means]
    ax.plot(labels, [r["fft_seconds"] for r in means], marker="o", label="FFT")
    ax.plot(labels, [r["total_seconds"] for r in means], marker="o", label="total")
    ax.set(xlabel="FFT workers (-1 = all logical cores)", ylabel="seconds / image", title="image1, W=128; mean of two runs")
    ax.legend(); fig.savefig(OUT / "core_benchmark.png", dpi=150, bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    main()
