"""Interactive block brightness / FFT comparison viewer.

Run: python block_spectral_viewer.py
Click blocks in the image. First choice is the focus (orange); further
choices are comparison targets. Full-image measurements are always gray.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data20260903" / "data20260903"
COLORS = ("#ff8c00", "#00bcd4", "#e83e8c", "#7cb342", "#9c6ade")
DAMAGE_COLOR = "#ff3030"
NON_DAMAGE_LABELS = {"tag", "background", "surface", "normal_surface"}


def image_number(path: Path) -> int:
    match = re.search(r"\d+", path.stem)
    return int(match.group()) if match else 10**9


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image.astype(np.float64)


def spectrum_1d(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-dimensional horizontal spectrum of the row-mean brightness."""
    waveform = gray.mean(axis=0)
    waveform -= waveform.mean()
    magnitude = np.abs(np.fft.rfft(waveform * np.hanning(waveform.size)))
    frequency = np.fft.rfftfreq(waveform.size, d=1.0)
    return frequency[1:], magnitude[1:] / max(1, waveform.size)


def spectrum_2d(gray: np.ndarray) -> np.ndarray:
    centered = gray - gray.mean()
    window = np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))
    # sqrt(N) normalization makes spectra from different block sizes comparable.
    fft = np.fft.fftshift(np.fft.fft2(centered * window)) / np.sqrt(gray.size)
    return np.log1p(np.abs(fft))


def load_damage_annotations(image_path: Path, shape: tuple[int, int]) -> tuple[np.ndarray, list[tuple[float, float, float, float, str]]]:
    """Read LabelMe damage shapes for display/comparison only.

    `tag`, `background`, and explicit surface labels are excluded.  All other
    labelled shapes are treated as damage so newly added damage label names do
    not silently disappear from this viewer.
    """
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    boxes: list[tuple[float, float, float, float, str]] = []
    json_path = image_path.with_suffix(".json")
    if not json_path.is_file():
        return mask, boxes
    try:
        document = json.loads(json_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return mask, boxes
    for item in document.get("shapes", []):
        label = str(item.get("label", ""))
        if label.casefold() in NON_DAMAGE_LABELS:
            continue
        points = np.asarray(item.get("points", []), dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all() or not len(points):
            continue
        kind = str(item.get("shape_type", "polygon"))
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        x0, x1 = np.clip((x0, x1), 0, width)
        y0, y1 = np.clip((y0, y1), 0, height)
        if x1 <= x0 or y1 <= y0:
            continue
        boxes.append((float(x0), float(y0), float(x1), float(y1), label))
        if kind == "rectangle" and len(points) == 2:
            mask[int(math.floor(y0)):int(math.ceil(y1)), int(math.floor(x0)):int(math.ceil(x1))] = True
        elif kind == "circle" and len(points) >= 2:
            canvas = np.zeros((height, width), dtype=np.uint8)
            center = tuple(np.rint(points[0]).astype(int))
            radius = int(round(np.hypot(*(points[1] - points[0]))))
            cv2.circle(canvas, center, max(1, radius), 1, thickness=-1)
            mask |= canvas.astype(bool)
        elif len(points) >= 3:
            canvas = np.zeros((height, width), dtype=np.uint8)
            vertices = np.rint(points).astype(np.int32)
            vertices[:, 0] = np.clip(vertices[:, 0], 0, width - 1)
            vertices[:, 1] = np.clip(vertices[:, 1], 0, height - 1)
            cv2.fillPoly(canvas, [vertices], 1)
            mask |= canvas.astype(bool)
    return mask, boxes


def center_zoom(spectrum: np.ndarray, radius: int) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Return a DC-centred FFT crop, with axes expressed in FFT-bin offsets."""
    height, width = spectrum.shape
    cy, cx = height // 2, width // 2
    x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
    extent = (x0 - cx - .5, x1 - cx - .5, y1 - cy - .5, y0 - cy - .5)
    return spectrum[y0:y1, x0:x1], extent


def top_conjugate_peaks(spectrum: np.ndarray, radius: int, max_peaks: int = 20,
                        dc_radius: int = 10, percentile: float = 99.0,
                        min_distance: int = 7) -> list[tuple[int, int]]:
    """Pick high local maxima as complete FFT conjugate pairs.

    Red circles are only visual peak candidates: real-valued images always
    produce conjugate symmetry, including for noise and non-grid texture.
    """
    height, width = spectrum.shape
    cy, cx = height // 2, width // 2
    yy, xx = np.indices(spectrum.shape)
    eligible = ((np.abs(xx - cx) <= radius) & (np.abs(yy - cy) <= radius) &
                ((xx - cx) ** 2 + (yy - cy) ** 2 > dc_radius ** 2))
    values = spectrum[eligible]
    if not values.size:
        return []
    cutoff = float(np.percentile(values, percentile))
    local_max = cv2.dilate(spectrum, np.ones((5, 5), dtype=np.uint8))
    candidates = {(int(y), int(x)) for y, x in np.argwhere(
        eligible & (spectrum >= cutoff) & (spectrum >= local_max - 1e-12)
    )}
    ordered = sorted(candidates, key=lambda point: (-float(spectrum[point]), point))
    peaks: list[tuple[int, int]] = []
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for point in ordered:
        if len(peaks) + 2 > max_peaks:
            break
        y, x = point
        partner = ((2 * cy - y) % height, (2 * cx - x) % width)
        key = tuple(sorted((point, partner)))
        if key in visited:
            continue
        visited.add(key)
        if partner == point or partner not in candidates:
            continue
        if any(np.hypot(py - sy, px - sx) < min_distance
               for py, px in key for sy, sx in peaks):
            continue
        peaks.extend(key)
    return peaks


def draw_peak_circles(ax, peaks: list[tuple[int, int]], spectrum_shape: tuple[int, int],
                      zoom_coordinates: bool = False) -> None:
    """Overlay visual peak candidates in either pixel-index or DC-offset axes."""
    if not peaks:
        return
    height, width = spectrum_shape
    cy, cx = height // 2, width // 2
    xs = [x - cx if zoom_coordinates else x for y, x in peaks]
    ys = [y - cy if zoom_coordinates else y for y, x in peaks]
    ax.scatter(xs, ys, s=48, facecolors="none", edgecolors="#ff3030", linewidths=1.0)


class BlockSpectralViewer(tk.Tk):
    def __init__(self, paths: list[Path], rows: int, cols: int):
        super().__init__()
        self.title("Block brightness and FFT comparison")
        self.geometry("1580x980")
        self.minsize(1100, 720)
        self.paths = paths
        self.path_by_name = {path.name: path for path in paths}
        self.gray: np.ndarray | None = None
        self.damage_mask: np.ndarray | None = None
        self.damage_boxes: list[tuple[float, float, float, float, str]] = []
        self.selected: list[tuple[int, int]] = []
        self.rows_var = tk.IntVar(value=rows)
        self.cols_var = tk.IntVar(value=cols)
        self.zoom_bins_var = tk.IntVar(value=32)
        self.show_damage_var = tk.BooleanVar(value=True)
        self.image_var = tk.StringVar(value=paths[0].name)
        self.status_var = tk.StringVar()
        self.fft_compare_window = None
        self.fft_compare_figure = None
        self.fft_compare_canvas = None
        self._build_ui()
        self.load_image()

    def _build_ui(self) -> None:
        controls = ttk.Frame(self, padding=8)
        controls.pack(fill="x")
        ttk.Label(controls, text="Image").pack(side="left")
        chooser = ttk.Combobox(controls, textvariable=self.image_var,
                                values=[path.name for path in self.paths],
                                state="readonly", width=18)
        chooser.pack(side="left", padx=(5, 16))
        chooser.bind("<<ComboboxSelected>>", lambda _event: self.load_image())
        ttk.Label(controls, text="Grid").pack(side="left")
        ttk.Spinbox(controls, from_=1, to=32, textvariable=self.rows_var,
                    width=4, command=self.change_grid).pack(side="left", padx=(5, 2))
        ttk.Label(controls, text="rows x").pack(side="left")
        ttk.Spinbox(controls, from_=1, to=32, textvariable=self.cols_var,
                    width=4, command=self.change_grid).pack(side="left", padx=(2, 5))
        ttk.Label(controls, text="cols").pack(side="left")
        ttk.Button(controls, text="Apply grid", command=self.change_grid).pack(side="left", padx=8)
        ttk.Button(controls, text="Clear", command=self.clear_selection).pack(side="left")
        ttk.Button(controls, text="Selected FFT + center zoom", command=self.show_2d_comparison).pack(side="left", padx=8)
        ttk.Button(controls, text="All block center FFT", command=self.show_all_center_fft).pack(side="left", padx=3)
        ttk.Button(controls, text="Select damage blocks", command=self.select_damage_blocks).pack(side="left", padx=3)
        ttk.Label(controls, text="Center zoom ±").pack(side="left", padx=(10, 1))
        ttk.Spinbox(controls, from_=4, to=128, increment=4, textvariable=self.zoom_bins_var,
                    width=4, command=self.redraw).pack(side="left")
        ttk.Label(controls, text="FFT bins").pack(side="left", padx=(1, 8))
        ttk.Checkbutton(controls, text="Damage annotation", variable=self.show_damage_var,
                        command=self.redraw).pack(side="left")
        ttk.Label(controls, textvariable=self.status_var).pack(side="right")
        ttk.Label(self, text=("Click blocks to compare. Red outlines are LabelMe damage annotations; "
                              "a block is ‘damage’ when it overlaps one or more damage pixels.")).pack(anchor="w", padx=10)

        self.figure = Figure(figsize=(15, 8.5), layout="constrained")
        axes = self.figure.subplot_mosaic([["image", "hist", "fft1d"],
                                           ["image", "full2d", "block2d"]])
        self.image_ax = axes["image"]
        self.hist_ax = axes["hist"]
        self.fft1d_ax = axes["fft1d"]
        self.full2d_ax = axes["full2d"]
        self.block2d_ax = axes["block2d"]
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8)
        NavigationToolbar2Tk(self.canvas, self).update()
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def grid_shape(self) -> tuple[int, int]:
        return max(1, int(self.rows_var.get())), max(1, int(self.cols_var.get()))

    def load_image(self) -> None:
        path = self.path_by_name[self.image_var.get()]
        self.gray = read_gray(path)
        self.damage_mask, self.damage_boxes = load_damage_annotations(path, self.gray.shape)
        self.selected.clear()
        self.redraw()

    def change_grid(self) -> None:
        self.selected.clear()
        self.redraw()

    def clear_selection(self) -> None:
        self.selected.clear()
        self.redraw()

    def block_damage_fraction(self, block: tuple[int, int]) -> float:
        if self.damage_mask is None:
            return 0.0
        x0, y0, x1, y1 = self.block_bounds(block)
        return float(self.damage_mask[y0:y1, x0:x1].mean())

    def block_kind(self, block: tuple[int, int]) -> str:
        return "damage" if self.block_damage_fraction(block) > 0 else "no damage"

    def select_damage_blocks(self) -> None:
        rows, cols = self.grid_shape()
        self.selected = [(row, col) for row in range(rows) for col in range(cols)
                         if self.block_damage_fraction((row, col)) > 0]
        self.redraw()

    def show_2d_comparison(self) -> None:
        if not self.selected:
            self.status_var.set("Select one or more blocks before comparing 2D FFT.")
            return
        if self.fft_compare_window is None or not self.fft_compare_window.winfo_exists():
            self.fft_compare_window = tk.Toplevel(self)
            self.fft_compare_window.title("Selected blocks: 2D FFT comparison")
            self.fft_compare_window.geometry("1200x820")
            self.fft_compare_figure = Figure(figsize=(11, 7.5), layout="constrained")
            self.fft_compare_canvas = FigureCanvasTkAgg(self.fft_compare_figure, master=self.fft_compare_window)
            self.fft_compare_canvas.get_tk_widget().pack(fill="both", expand=True)
            NavigationToolbar2Tk(self.fft_compare_canvas, self.fft_compare_window).update()
            self.fft_compare_window.protocol("WM_DELETE_WINDOW", self.close_2d_comparison)
        self.render_2d_comparison()
        self.fft_compare_window.lift()

    def close_2d_comparison(self) -> None:
        if self.fft_compare_window is not None:
            self.fft_compare_window.destroy()
        self.fft_compare_window = self.fft_compare_figure = self.fft_compare_canvas = None

    def render_2d_comparison(self) -> None:
        if self.fft_compare_figure is None or self.gray is None:
            return
        sources = [(f"B{r + 1},{c + 1}", (r, c), self.block_data((r, c)))
                   for r, c in self.selected[:12]]
        spectra = [(label, spectrum_2d(data)) for label, _block, data in sources]
        vmax = max(float(np.percentile(value, 99.5)) for _, value in spectra)
        self.fft_compare_figure.clear()
        axes = self.fft_compare_figure.subplots(len(spectra), 2, squeeze=False)
        radius = max(1, int(self.zoom_bins_var.get()))
        for (label, block, _data), (_same_label, value), (full_ax, zoom_ax) in zip(sources, spectra, axes):
            fraction = self.block_damage_fraction(block)
            kind = self.block_kind(block)
            radius = min(radius, min(value.shape) // 2 - 1)
            peaks = top_conjugate_peaks(value, radius)
            full_ax.imshow(value, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
            full_ax.plot(value.shape[1] // 2, value.shape[0] // 2, "+", color="white", ms=4)
            draw_peak_circles(full_ax, peaks, value.shape)
            full_ax.set_title(f"{label} | {kind} ({fraction:.2%})\nFull 2D FFT | {len(peaks)} peak candidates", fontsize=9,
                              color=DAMAGE_COLOR if fraction > 0 else "black")
            full_ax.set_axis_off()
            cropped, extent = center_zoom(value, radius)
            zoom_ax.imshow(cropped, cmap="gray", vmin=0, vmax=vmax, extent=extent, interpolation="nearest")
            zoom_ax.axhline(0, color="white", lw=.35, alpha=.6)
            zoom_ax.axvline(0, color="white", lw=.35, alpha=.6)
            draw_peak_circles(zoom_ax, peaks, value.shape, zoom_coordinates=True)
            zoom_ax.set_title(f"{label} | center ±{radius} bins | red circles: top {len(peaks)}", fontsize=9,
                              color=DAMAGE_COLOR if fraction > 0 else "black")
            zoom_ax.set(xlabel="x FFT-bin offset", ylabel="y FFT-bin offset")
        self.fft_compare_figure.suptitle(
            "Selected blocks: full 2D FFT (left) and DC-centre zoom (right) | common color scale",
            fontsize=12,
        )
        self.fft_compare_canvas.draw_idle()

    def show_all_center_fft(self) -> None:
        """Show DC-centre crops for every grid cell; red titles denote damage blocks."""
        if self.gray is None:
            return
        window = tk.Toplevel(self)
        window.title("All blocks: 2D FFT centre zoom (red = damage block)")
        window.geometry("1480x950")
        figure = Figure(figsize=(14, 9), layout="constrained")
        canvas = FigureCanvasTkAgg(figure, master=window)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(canvas, window).update()
        rows, cols = self.grid_shape()
        blocks = [(row, col) for row in range(rows) for col in range(cols)]
        spectra = [(block, spectrum_2d(self.block_data(block))) for block in blocks]
        vmax = max(float(np.percentile(value, 99.5)) for _, value in spectra)
        axes = figure.subplots(rows, cols, squeeze=False)
        radius = max(1, int(self.zoom_bins_var.get()))
        for ax, (block, value) in zip(axes.ravel(), spectra):
            cropped, extent = center_zoom(value, radius)
            fraction = self.block_damage_fraction(block)
            actual_radius = min(radius, min(value.shape) // 2 - 1)
            peaks = top_conjugate_peaks(value, actual_radius)
            ax.imshow(cropped, cmap="gray", vmin=0, vmax=vmax, extent=extent, interpolation="nearest")
            ax.axhline(0, color="white", lw=.25, alpha=.55)
            ax.axvline(0, color="white", lw=.25, alpha=.55)
            draw_peak_circles(ax, peaks, value.shape, zoom_coordinates=True)
            ax.set_title(f"B{block[0] + 1},{block[1] + 1} {'damage' if fraction else 'no damage'} | {len(peaks)} peaks",
                         fontsize=7, color=DAMAGE_COLOR if fraction else "black")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(DAMAGE_COLOR if fraction else "#555555")
                spine.set_linewidth(1.8 if fraction else .7)
        figure.suptitle(
            f"All block 2D FFT centre zoom ±{radius} bins | red title/frame = damage; red circles = top FFT peaks",
            fontsize=12,
        )
        canvas.draw_idle()

    def block_bounds(self, block: tuple[int, int]) -> tuple[int, int, int, int]:
        assert self.gray is not None
        row, col = block
        rows, cols = self.grid_shape()
        height, width = self.gray.shape
        y_edges = np.linspace(0, height, rows + 1, dtype=int)
        x_edges = np.linspace(0, width, cols + 1, dtype=int)
        return x_edges[col], y_edges[row], x_edges[col + 1], y_edges[row + 1]

    def block_data(self, block: tuple[int, int]) -> np.ndarray:
        x0, y0, x1, y1 = self.block_bounds(block)
        assert self.gray is not None
        return self.gray[y0:y1, x0:x1]

    def on_click(self, event) -> None:
        if event.inaxes != self.image_ax or event.xdata is None or event.ydata is None:
            return
        assert self.gray is not None
        x, y = int(event.xdata), int(event.ydata)
        if not (0 <= x < self.gray.shape[1] and 0 <= y < self.gray.shape[0]):
            return
        rows, cols = self.grid_shape()
        block = (min(rows - 1, y * rows // self.gray.shape[0]),
                 min(cols - 1, x * cols // self.gray.shape[1]))
        if block in self.selected:
            self.selected.remove(block)
        else:
            self.selected.append(block)
        self.redraw()

    def redraw(self) -> None:
        if self.gray is None:
            return
        rows, cols = self.grid_shape()
        for ax in (self.image_ax, self.hist_ax, self.fft1d_ax, self.full2d_ax, self.block2d_ax):
            ax.clear()
        self.image_ax.imshow(self.gray, cmap="gray", vmin=0, vmax=255)
        self.image_ax.set_title("Click blocks to compare (red outlines = damage annotation)")
        for row in range(1, rows):
            self.image_ax.axhline(row * self.gray.shape[0] / rows, color="lime", lw=.65)
        for col in range(1, cols):
            self.image_ax.axvline(col * self.gray.shape[1] / cols, color="lime", lw=.65)
        if self.show_damage_var.get():
            for x0, y0, x1, y1, label in self.damage_boxes:
                self.image_ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                                  fill=False, lw=1.4, edgecolor=DAMAGE_COLOR))
                self.image_ax.text(x0 + 2, y0 + 12, label, color=DAMAGE_COLOR, fontsize=7,
                                   bbox={"facecolor": "black", "alpha": .5, "pad": 1})
        for index, block in enumerate(self.selected):
            x0, y0, x1, y1 = self.block_bounds(block)
            color = COLORS[index % len(COLORS)]
            self.image_ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                              fill=False, lw=2.5, edgecolor=color))
            fraction = self.block_damage_fraction(block)
            self.image_ax.text(x0 + 3, y0 + 16,
                               f"B{block[0] + 1},{block[1] + 1} {'damage' if fraction else 'no damage'}",
                               color=color,
                               fontsize=9, bbox={"facecolor": "black", "alpha": .6, "pad": 1})
        self.image_ax.set_axis_off()

        self.hist_ax.hist(self.gray.ravel(), bins=256, range=(0, 256), density=True,
                          histtype="step", color="0.35", lw=1.5, label="Full image")
        f, m = spectrum_1d(self.gray)
        self.fft1d_ax.plot(f, m, color="0.35", lw=1.4, label="Full image")
        for index, block in enumerate(self.selected):
            data = self.block_data(block)
            color = COLORS[index % len(COLORS)]
            label = f"B{block[0] + 1},{block[1] + 1}"
            self.hist_ax.hist(data.ravel(), bins=256, range=(0, 256), density=True,
                              histtype="step", color=color, lw=1.4, label=label)
            freq, magnitude = spectrum_1d(data)
            self.fft1d_ax.plot(freq, magnitude, color=color, lw=1.3, label=label)
        self.hist_ax.set(title="Brightness histogram", xlabel="Brightness (0-255)", ylabel="Density")
        self.fft1d_ax.set(title="1D FFT: horizontal row-mean spectrum",
                           xlabel="Spatial frequency (cycles/pixel)", ylabel="Amplitude")
        for ax in (self.hist_ax, self.fft1d_ax):
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)

        self.full2d_ax.imshow(spectrum_2d(self.gray), cmap="magma", interpolation="nearest")
        self.full2d_ax.set_title("Full image 2D FFT (log amplitude; DC centered)")
        self.full2d_ax.set_axis_off()
        if self.selected:
            focus = self.selected[0]
            self.block2d_ax.imshow(spectrum_2d(self.block_data(focus)), cmap="magma", interpolation="nearest")
            self.block2d_ax.set_title(f"Block B{focus[0] + 1},{focus[1] + 1} 2D FFT")
        else:
            self.block2d_ax.text(.5, .5, "Click a block to show its 2D FFT", ha="center", va="center",
                                 transform=self.block2d_ax.transAxes)
            self.block2d_ax.set_title("Selected block 2D FFT")
        self.block2d_ax.set_axis_off()
        selected_text = ", ".join(
            f"B{r + 1},{c + 1} ({self.block_kind((r, c))})" for r, c in self.selected
        ) or "none"
        self.status_var.set(f"{self.gray.shape[1]}x{self.gray.shape[0]} px | selected: {selected_text}")
        self.canvas.draw_idle()
        if self.fft_compare_window is not None and self.fft_compare_window.winfo_exists():
            self.render_2d_comparison()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, help="optional source image")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    args = parser.parse_args()
    paths = [args.path.resolve()] if args.path else sorted(DEFAULT_INPUT.glob("image[0-9]*.png"), key=image_number)
    if not paths:
        parser.error("No images found")
    BlockSpectralViewer(paths, args.rows, args.cols).mainloop()


if __name__ == "__main__":
    main()
