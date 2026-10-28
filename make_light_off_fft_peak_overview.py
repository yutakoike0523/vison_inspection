"""Create the same FFT-centre peak comparison figure for the four light-off images."""
from __future__ import annotations

import json
from pathlib import Path

import compare_fft_center_peaks as fft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
NAMES = ("image1.png", "image19.png", "image20.png", "image21.png")
OUT = ROOT / "outputs" / "fft_center_peaks_light_off"


def save_overview(focus, destination: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout="constrained")
    try:
        for ax in axes.flat:
            ax.axis("off")
        for ax, (record, masked, extent) in zip(axes.flat, focus):
            low, high = record["zoom_display_limits"]
            ax.imshow(masked, cmap="gray", vmin=low, vmax=high,
                      extent=extent, interpolation="nearest")
            fft.annotate_peaks(ax, record["peaks"])
            ax.axis("on")
            ax.set_title(
                f"{record['filename']} | {record['width']} x {record['height']}\n"
                f"{record['detected_peak_count']} points / {record['detected_pair_count']} pairs"
            )
            ax.set(xlabel="x offset [FFT bins]", ylabel="y offset [FFT bins]")
        axes.flat[-1].text(
            .05, .8,
            "Grid light OFF: image1 / 19 / 20 / 21\n"
            "Full original images; no crop / resize\n"
            "Mean subtraction + 2D Hann + FFT\n\n"
            "Zoom: +/-150 bins; DC mask: r=10 bins\n\n"
            "Each image has its own display scale.\n"
            "A/B labels identify conjugate pairs.\n"
            "Symmetry alone does not identify a lattice.",
            fontsize=12, va="top",
        )
        fig.suptitle("FFT center peaks: grid light OFF", fontsize=16)
        fig.savefig(destination, dpi=150)
    finally:
        plt.close(fig)


def main() -> None:
    source = ROOT / "data20260903" / "data20260903"
    OUT.mkdir(parents=True, exist_ok=True)
    focus = []
    for name in NAMES:
        path = source / name
        gray = fft.read_gray(path)
        transformed, spectrum = fft.hann_fft(gray)
        zoom, outside_dc, bounds, extent = fft.center_zoom(
            spectrum, fft.ZOOM_RADIUS, fft.DC_MASK_RADIUS
        )
        peaks, threshold = fft.detect_peaks(zoom, outside_dc, bounds, gray.shape)
        record = {
            "filename": path.name,
            "width": gray.shape[1], "height": gray.shape[0],
            "zoom_radius": fft.ZOOM_RADIUS, "dc_mask_radius": fft.DC_MASK_RADIUS,
            "detected_peak_count": len(peaks), "detected_pair_count": len(peaks) // 2,
            "peak_percentile": 99., "min_peak_distance": 7,
            "peak_threshold_log_amplitude": threshold,
            "full_display_limits": fft.display_limits(spectrum),
            "zoom_display_limits": fft.display_limits(zoom[outside_dc]),
            "peaks": peaks,
        }
        fft.save_comparison(
            gray, spectrum, zoom, outside_dc, bounds, extent, record,
            OUT / f"{path.name}_fft_center_comparison.png",
        )
        (OUT / f"{path.stem}_peaks.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        focus.append((record, zoom * outside_dc, extent))

    save_overview(focus, OUT / "light_off_images_peaks.png")
    print(OUT)


if __name__ == "__main__":
    main()
