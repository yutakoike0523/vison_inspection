"""Two-window FFT explorer. Run: python interactive_2d_fft.py --image 2"""

import argparse
from pathlib import Path
import re
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from analyze_2d_fft import read_grayscale


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data20260903" / "data20260903"


class FrequencySelection:
    """One selected bin includes its conjugate, so reconstruction is real."""

    def __init__(self, gray):
        self.gray = gray
        self.height, self.width = gray.shape
        self.fft = np.fft.fft2(gray.astype(np.float64))
        self.spectrum = np.log1p(np.abs(np.fft.fftshift(self.fft)))
        self.selected = []
        self.total = np.zeros(gray.shape, dtype=np.float64)

    def key(self, x, y):
        # Map displayed fftshift coordinates back to unshifted FFT indices.
        ky = (y - self.height // 2) % self.height
        kx = (x - self.width // 2) % self.width
        partner = ((-ky) % self.height, (-kx) % self.width)
        return min((ky, kx), partner)

    def points(self, key):
        ky, kx = key
        return sorted({((kx + self.width // 2) % self.width,
                        (ky + self.height // 2) % self.height),
                       ((-kx + self.width // 2) % self.width,
                        (-ky + self.height // 2) % self.height)})

    def component(self, key):
        """Exact inverse DFT of one conjugate pair, without a full FFT per hover."""
        ky, kx = key
        ey = np.exp(2j * np.pi * ky * np.arange(self.height) / self.height)
        ex = np.exp(2j * np.pi * kx * np.arange(self.width) / self.width)
        wave = np.real((self.fft[ky, kx] / self.gray.size) * ey[:, None] * ex[None, :])
        # DC and even-size Nyquist corner bins are self-conjugate: count once.
        if ((-ky) % self.height, (-kx) % self.width) != key:
            wave *= 2
        return wave

    def add(self, key):
        if key in self.selected:
            return False
        self.total += self.component(key)
        self.selected.append(key)
        return True

    def undo(self):
        if self.selected:
            self.total -= self.component(self.selected.pop())
        if not self.selected:
            self.total.fill(0)

    def clear(self):
        self.selected.clear()
        self.total.fill(0)


class FFTViewer:
    def __init__(self, root, paths, initial=0):
        self.root, self.paths = root, paths
        self.index = initial
        self.hover_key = None
        self.hover_wave = None
        self.pending_point = None
        self.timer = None
        screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
        main_width = int(screen_width * .52)
        window_height = max(500, screen_height - 110)
        root.title("2D FFT - select frequencies")
        root.geometry(f"{main_width}x{window_height}+0+30")
        controls = ttk.Frame(root)
        controls.pack(fill="x", padx=8, pady=6)
        self.choice = ttk.Combobox(controls, state="readonly", width=40,
                                   values=[p.name for p in paths])
        self.choice.pack(side="left")
        self.choice.bind("<<ComboboxSelected>>", lambda event: self.load(self.choice.current()))
        for label, command in [("Previous", lambda: self.load((self.index - 1) % len(paths))),
                               ("Next", lambda: self.load((self.index + 1) % len(paths))),
                               ("Original / Trimmed", self.toggle_kind),
                               ("Undo", self.undo), ("Clear", self.clear)]:
            ttk.Button(controls, text=label, command=command).pack(side="left", padx=3)
        ttk.Label(root, text="FFT上にカーソル: 成分を表示 / 左クリック: 固定して加算 / "
                  "右クリック・U: 戻す / C: 全解除 / 画像切替時は選択をリセット").pack()
        self.fig = Figure(figsize=(10, 6), layout="constrained")
        self.original_ax, self.fft_ax = self.fig.subplots(1, 2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, root)
        self.status = tk.StringVar()
        ttk.Label(root, textvariable=self.status).pack(fill="x", padx=8)

        self.result = tk.Toplevel(root)
        self.result.title("Frequency components - inverse FFT")
        result_width = screen_width - main_width - 30
        self.result.geometry(f"{result_width}x{window_height}+{main_width + 10}+30")
        ttk.Label(self.result, text="上: カーソルの成分 / 中: 固定した成分の合計 / "
                  "下: 固定した合計 + カーソル（クリック前のプレビュー）").pack(pady=6)
        ttk.Label(self.result, text="青: 負、白: 0、赤: 正。3枚共通の自動表示スケール。"
                  "元画像の振幅・位相を保持し、対称な周波数を1組として加算します。").pack()
        self.result_fig = Figure(figsize=(8, 9), layout="constrained")
        self.result_axes = self.result_fig.subplots(3, 1)
        self.result_canvas = FigureCanvasTkAgg(self.result_fig, master=self.result)
        self.result_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.result_toolbar = NavigationToolbar2Tk(self.result_canvas, self.result)
        self.result.protocol("WM_DELETE_WINDOW", self.close)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("figure_leave_event", lambda event: self.queue_point(None))
        for window in (root, self.result):
            window.bind("<Key-u>", lambda event: self.undo())
            window.bind("<Key-c>", lambda event: self.clear())
        self.load(initial)

    def load(self, index):
        if self.timer is not None:
            self.root.after_cancel(self.timer)
            self.timer = None
        self.pending_point = None
        self.hover_key = self.hover_wave = None
        self.index = index
        self.choice.current(index)
        self.model = FrequencySelection(read_grayscale(self.paths[index]))
        self.original_ax.clear()
        self.fft_ax.clear()
        self.original_ax.imshow(self.model.gray, cmap="gray", vmin=0, vmax=255)
        self.original_ax.set_title(self.paths[index].name)
        self.original_ax.axis("off")
        self.fft_ax.imshow(self.model.spectrum, cmap="gray", interpolation="nearest")
        self.fft_ax.set_title("2D FFT: log(1 + |F|)\nHover / click a frequency bin")
        self.fft_ax.set(xlabel="FFT bin x (DC at center)", ylabel="FFT bin y")
        self.fixed_marker, = self.fft_ax.plot([], [], "o", ms=7, mfc="none", mec="#ff6547")
        self.hover_marker, = self.fft_ax.plot([], [], "+", ms=12, color="#00ffff")
        self.result_artists = []
        for ax in self.result_axes:
            ax.clear()
            self.result_artists.append(ax.imshow(self.model.total, cmap="RdBu_r",
                                                 vmin=-1, vmax=1, interpolation="nearest"))
            ax.axis("off")
        self.toolbar.update()
        self.result_toolbar.update()
        self.refresh()

    def toggle_kind(self):
        path = self.paths[self.index]
        if path.stem.endswith("_triming"):
            target = path.parent.parent / (path.stem.removesuffix("_triming") + ".png")
        else:
            target = path.parent / "triming" / (path.stem + "_triming.png")
        if target in self.paths:
            self.load(self.paths.index(target))
        else:
            self.status.set("この画像には対応するトリミング画像がありません。")

    def event_point(self, event):
        if event.inaxes != self.fft_ax or self.toolbar.mode or event.xdata is None or event.ydata is None:
            return None
        x, y = int(np.floor(event.xdata + .5)), int(np.floor(event.ydata + .5))
        if 0 <= x < self.model.width and 0 <= y < self.model.height:
            return x, y
        return None

    def queue_point(self, point):
        self.pending_point = point
        if self.timer is None:
            self.timer = self.root.after(60, self.apply_hover)

    def on_motion(self, event):
        self.queue_point(self.event_point(event))

    def apply_hover(self):
        self.timer = None
        key = self.model.key(*self.pending_point) if self.pending_point is not None else None
        if key != self.hover_key:
            self.hover_key = key
            self.hover_wave = self.model.component(key) if key is not None else None
            self.refresh()

    def on_click(self, event):
        if self.toolbar.mode:
            return
        if event.button == 3:
            self.undo()
            return
        point = self.event_point(event)
        if event.button == 1 and point is not None:
            self.pending_point = point
            # Use the actual click position even if a hover timer is pending.
            if self.timer is not None:
                self.root.after_cancel(self.timer)
                self.timer = None
            self.hover_key = self.model.key(*point)
            self.hover_wave = self.model.component(self.hover_key)
            self.model.add(self.hover_key)
            self.refresh()

    def undo(self):
        self.model.undo()
        self.refresh()

    def clear(self):
        self.model.clear()
        self.refresh()

    def refresh(self):
        hover = self.hover_wave if self.hover_wave is not None else np.zeros_like(self.model.total)
        duplicate = self.hover_key in self.model.selected
        preview = self.model.total if duplicate else self.model.total + hover
        arrays = (hover, self.model.total, preview)
        upper = max(1e-12, *(float(np.max(np.abs(a))) for a in arrays))
        titles = ("Cursor component", f"Fixed sum: {len(self.model.selected)} pairs",
                  "Fixed + cursor" + (" (already fixed)" if duplicate else ""))
        for ax, artist, array, title in zip(self.result_axes, self.result_artists, arrays, titles):
            artist.set_data(array)
            artist.set_clim(-upper, upper)
            ax.set_title(title)
        self.result_fig.suptitle(f"Shared display range: {-upper:.4g} to {upper:.4g} gray levels")
        fixed = [point for key in self.model.selected for point in self.model.points(key)]
        self.fixed_marker.set_data([p[0] for p in fixed], [p[1] for p in fixed])
        points = self.model.points(self.hover_key) if self.hover_key is not None else []
        self.hover_marker.set_data([p[0] for p in points], [p[1] for p in points])
        status = f"固定: {len(self.model.selected)} 組"
        if self.hover_key is not None:
            ky, kx = self.hover_key
            fy = (ky if ky <= (self.model.height - 1) // 2 else ky - self.model.height) / self.model.height
            fx = (kx if kx <= (self.model.width - 1) // 2 else kx - self.model.width) / self.model.width
            freq = np.hypot(fx, fy)
            period = f"{1 / freq:.2f} px" if freq else "DC (mean brightness)"
            status += f" | fx={fx:.6f}, fy={fy:.6f} cycles/pixel | period={period}"
            if duplicate:
                status += " | 固定済み（重複加算しません）"
        self.status.set(status)
        self.canvas.draw_idle()
        self.result_canvas.draw_idle()

    def close(self):
        if self.timer is not None:
            self.root.after_cancel(self.timer)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, help="Optional image file")
    parser.add_argument("--image", type=int, default=2)
    parser.add_argument("--trimmed", "--triming", action="store_true")
    args = parser.parse_args()
    if args.path:
        paths, initial = [args.path.resolve()], 0
    else:
        paths = sorted(SOURCE.glob("image[0-9]*.png"),
                       key=lambda p: int(re.search(r"\d+", p.stem).group()))
        paths += sorted((SOURCE / "triming").glob("image[0-9]*_triming.png"),
                        key=lambda p: int(re.search(r"\d+", p.stem).group()))
        target = (SOURCE / "triming" / f"image{args.image}_triming.png" if args.trimmed
                  else SOURCE / f"image{args.image}.png")
        if target not in paths:
            parser.error(f"Image not found: {target}")
        initial = paths.index(target)
    root = tk.Tk()
    FFTViewer(root, paths, initial)
    root.mainloop()


if __name__ == "__main__":
    main()
