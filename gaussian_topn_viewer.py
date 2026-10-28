"""Interactive viewer for Gaussian sigma=8 detection candidates.

Select an image and the number of ranked candidates to display.  Candidate
scores are drawn on the image and listed in a table.  Source images and JSON
files are read-only.
"""
from __future__ import annotations

import argparse
import json
import math
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

import fft_mask_ground_truth_evaluation as gt
import point_anomaly_feature_analysis as fa
from render_gaussian_top10 import candidates, display_box


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data20260903" / "data20260903"
ANALYSIS = ROOT / "outputs" / "point_anomaly_peak_analysis"
DEFAULT_EXPORT = ANALYSIS / "09_gaussian_s8_top10" / "interactive_exports"
SIGMA = 8
THRESHOLD = float(json.loads(
    (ANALYSIS / "parameter_summary.json").read_text(encoding="utf-8")
)["base_threshold"])
_DAMAGE_LABELS = list(dict.fromkeys(gt.DAMAGE_MAP.values()))
LABEL_DISPLAY = {
    _DAMAGE_LABELS[0]: ("点状異常", "P", (0, 255, 0)),
    _DAMAGE_LABELS[1]: ("線状異常", "L", (0, 220, 255)),
    _DAMAGE_LABELS[2]: ("面状異常", "A", (255, 0, 255)),
    _DAMAGE_LABELS[3]: ("局所変形", "D", (255, 165, 0)),
}


def image_number(path: Path) -> int:
    suffix = path.stem.replace("image", "")
    return int(suffix) if suffix.isdigit() else 10**9


def image_jsons() -> list[Path]:
    return sorted(DATA.glob("image*.json"), key=image_number)


def analyze(json_path: Path):
    rec = gt.load_record(json_path)
    gray = rec["gray"].astype(np.float32)
    response = cv2.GaussianBlur(
        gray, (0, 0), SIGMA, borderType=cv2.BORDER_REFLECT_101
    ) - gray
    annotations = []
    for shape in rec["shapes"]:
        label = shape["normalized_label"]
        if label not in LABEL_DISPLAY or shape["box"] is None:
            continue
        x1, y1, x2, y2 = shape["box"]
        ix1 = max(0, int(math.floor(x1)))
        iy1 = max(0, int(math.floor(y1)))
        ix2 = min(rec["width"], int(math.ceil(x2)))
        iy2 = min(rec["height"], int(math.ceil(y2)))
        region = response[iy1:iy2, ix1:ix2]
        display_name, code, color = LABEL_DISPLAY[label]
        annotations.append({
            "id": len(annotations) + 1,
            "label": display_name,
            "code": code,
            "color": color,
            "box": (x1, y1, x2, y2),
            "score": float(region.max()) if region.size else float("nan"),
        })
    return rec, candidates(response, THRESHOLD), annotations


def load_font(size: int):
    for name in ("C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/YuGothM.ttc",
                 "arial.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_overlay(rec, ranked, annotations, top_n: int) -> Image.Image:
    image = Image.fromarray(rec["rgb"].copy())
    draw = ImageDraw.Draw(image)
    line = max(2, rec["width"] // 700)
    font = load_font(max(15, rec["width"] // 100))
    small_font = load_font(max(13, rec["width"] // 120))

    for rank, row in enumerate(ranked[:top_n], 1):
        box = display_box(row, rec["width"], rec["height"])
        draw.rectangle(box, outline=(255, 0, 0), width=line)
        label = f"#{rank}  {row['score']:.2f}"
        label_box = draw.textbbox((0, 0), label, font=font, stroke_width=1)
        label_w = label_box[2] - label_box[0] + 8
        label_h = label_box[3] - label_box[1] + 6
        tx = min(max(0, box[0]), max(0, rec["width"] - label_w))
        ty = box[1] - label_h
        if ty < 0:
            ty = min(rec["height"] - label_h, box[3] + 2)
        draw.rectangle((tx, ty, tx + label_w, ty + label_h), fill=(0, 0, 0))
        draw.text((tx + 4, ty + 2), label, fill=(255, 255, 0), font=font,
                  stroke_width=1, stroke_fill=(0, 0, 0))

    # Draw manual annotations after candidates so their locations remain visible.
    for ann in annotations:
        x1, y1, x2, y2 = ann["box"]
        color = ann["color"]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line + 1)
        label = f"GT{ann['id']} {ann['code']} {ann['score']:.2f}"
        label_box = draw.textbbox((0, 0), label, font=small_font, stroke_width=1)
        label_w = label_box[2] - label_box[0] + 6
        label_h = label_box[3] - label_box[1] + 4
        tx = min(max(0, int(x1)), max(0, rec["width"] - label_w))
        ty = int(y1) - label_h
        if ty < 0:
            ty = min(rec["height"] - label_h, int(y2) + 2)
        draw.rectangle((tx, ty, tx + label_w, ty + label_h), fill=(0, 0, 0))
        draw.text((tx + 3, ty + 1), label, fill=color, font=small_font,
                  stroke_width=1, stroke_fill=(0, 0, 0))

    header = (
        f"{rec['name']} | s={SIGMA} | top {min(top_n, len(ranked))}/{len(ranked):,} "
        f"| th={THRESHOLD:.3f}\n"
        "red=candidate; P/L/A/D=user annotations; value=score"
    )
    header_box = draw.multiline_textbbox((0, 0), header, font=small_font, spacing=2)
    header_h = header_box[3] - header_box[1] + 12
    draw.rectangle((0, 0, rec["width"], header_h), fill=(0, 0, 0))
    draw.multiline_text((6, 4), header, fill=(255, 255, 255),
                        font=small_font, spacing=2)
    return image


class Viewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gaussian sigma=8 candidate viewer")
        self.geometry("1550x920")
        self.minsize(1050, 650)
        self.paths = image_jsons()
        self.path_by_name = {p.stem: p for p in self.paths}
        self.cache = {}
        self.current = None
        self.rendered = None
        self.tk_image = None
        self.job_id = 0
        self.resize_job = None
        self._build_ui()
        self.after(100, self.load_selected)

    def _build_ui(self):
        controls = ttk.Frame(self, padding=8)
        controls.pack(fill="x")

        ttk.Label(controls, text="画像:").pack(side="left")
        self.image_var = tk.StringVar(value=self.paths[0].stem if self.paths else "")
        self.image_box = ttk.Combobox(
            controls, textvariable=self.image_var,
            values=[p.stem for p in self.paths], state="readonly", width=12,
        )
        self.image_box.pack(side="left", padx=(4, 16))
        self.image_box.bind("<<ComboboxSelected>>", lambda _event: self.load_selected())

        ttk.Label(controls, text="表示する上位件数:").pack(side="left")
        self.top_var = tk.IntVar(value=10)
        self.top_spin = ttk.Spinbox(
            controls, from_=1, to=100000, textvariable=self.top_var, width=9,
            command=self.redraw,
        )
        self.top_spin.pack(side="left", padx=(4, 6))
        self.top_spin.bind("<Return>", lambda _event: self.redraw())
        self.top_spin.bind("<FocusOut>", lambda _event: self.redraw())
        ttk.Button(controls, text="表示更新", command=self.redraw).pack(side="left", padx=4)
        ttk.Button(controls, text="表示画像を保存", command=self.save_image).pack(side="left", padx=12)

        self.status_var = tk.StringVar(value="準備中")
        ttk.Label(controls, textvariable=self.status_var).pack(side="left", padx=12)

        pane = ttk.Panedwindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.image_frame = ttk.Frame(pane)
        table_frame = ttk.Frame(pane, width=520)
        pane.add(self.image_frame, weight=3)
        pane.add(table_frame, weight=2)

        self.image_label = ttk.Label(self.image_frame, anchor="center")
        self.image_label.pack(fill="both", expand=True)
        self.image_frame.bind("<Configure>", self._schedule_fit)

        notebook = ttk.Notebook(table_frame)
        notebook.pack(fill="both", expand=True)
        candidate_tab = ttk.Frame(notebook)
        annotation_tab = ttk.Frame(notebook)
        notebook.add(candidate_tab, text="検出候補")
        notebook.add(annotation_tab, text="指定した傷")

        columns = ("rank", "score", "x", "y", "width", "height", "area")
        self.table = ttk.Treeview(candidate_tab, columns=columns, show="headings")
        headings = {
            "rank": "順位", "score": "スコア", "x": "x", "y": "y",
            "width": "幅", "height": "高さ", "area": "面積",
        }
        widths = {"rank": 55, "score": 85, "x": 65, "y": 65,
                  "width": 55, "height": 55, "area": 70}
        for column in columns:
            self.table.heading(column, text=headings[column])
            self.table.column(column, width=widths[column], anchor="center")
        scrollbar = ttk.Scrollbar(candidate_tab, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        self.table.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        annotation_columns = ("id", "type", "score", "x", "y", "width", "height")
        self.annotation_table = ttk.Treeview(
            annotation_tab, columns=annotation_columns, show="headings"
        )
        annotation_headings = {
            "id": "ID", "type": "種類", "score": "スコア", "x": "x", "y": "y",
            "width": "幅", "height": "高さ",
        }
        annotation_widths = {
            "id": 45, "type": 95, "score": 80, "x": 65, "y": 65,
            "width": 55, "height": 55,
        }
        for column in annotation_columns:
            self.annotation_table.heading(column, text=annotation_headings[column])
            self.annotation_table.column(
                column, width=annotation_widths[column], anchor="center"
            )
        annotation_scroll = ttk.Scrollbar(
            annotation_tab, orient="vertical", command=self.annotation_table.yview
        )
        self.annotation_table.configure(yscrollcommand=annotation_scroll.set)
        self.annotation_table.pack(side="left", fill="both", expand=True)
        annotation_scroll.pack(side="right", fill="y")

    def _schedule_fit(self, _event=None):
        if self.resize_job is not None:
            self.after_cancel(self.resize_job)
        self.resize_job = self.after(120, self._fit_image)

    def load_selected(self):
        name = self.image_var.get()
        if not name or name not in self.path_by_name:
            return
        self.job_id += 1
        job = self.job_id
        if name in self.cache:
            self.current = self.cache[name]
            self.redraw()
            return
        self.status_var.set(f"{name} を解析中...")
        self.image_box.configure(state="disabled")

        def worker():
            try:
                result = analyze(self.path_by_name[name])
                self.after(0, lambda: self._analysis_done(job, name, result, None))
            except Exception as exc:
                self.after(0, lambda: self._analysis_done(job, name, None, exc))

        threading.Thread(target=worker, daemon=True).start()

    def _analysis_done(self, job, name, result, error):
        self.image_box.configure(state="readonly")
        if job != self.job_id:
            return
        if error is not None:
            self.status_var.set("解析エラー")
            messagebox.showerror("解析エラー", str(error))
            return
        self.cache[name] = result
        self.current = result
        self.redraw()

    def _top_n(self):
        if self.current is None:
            return 1
        maximum = max(1, len(self.current[1]))
        try:
            value = int(self.top_var.get())
        except (tk.TclError, ValueError):
            value = 10
        value = min(maximum, max(1, value))
        self.top_var.set(value)
        return value

    def redraw(self):
        if self.current is None:
            return
        rec, ranked, annotations = self.current
        top_n = self._top_n()
        self.rendered = render_overlay(rec, ranked, annotations, top_n)
        self.table.delete(*self.table.get_children())
        for rank, row in enumerate(ranked[:top_n], 1):
            self.table.insert("", "end", values=(
                rank, f"{row['score']:.3f}", row["component_x"],
                row["component_y"], row["component_width"],
                row["component_height"], row["area"],
            ))
        self.annotation_table.delete(*self.annotation_table.get_children())
        for ann in annotations:
            x1, y1, x2, y2 = ann["box"]
            self.annotation_table.insert("", "end", values=(
                f"GT{ann['id']}", ann["label"], f"{ann['score']:.3f}",
                int(round(x1)), int(round(y1)), int(round(x2 - x1)),
                int(round(y2 - y1)),
            ))
        self.status_var.set(
            f"{rec['name']}: 上位{top_n}/{len(ranked):,}候補、指定傷{len(annotations)}件"
        )
        self._fit_image()

    def _fit_image(self):
        self.resize_job = None
        if self.rendered is None:
            return
        width = max(200, self.image_frame.winfo_width() - 12)
        height = max(200, self.image_frame.winfo_height() - 12)
        shown = self.rendered.copy()
        shown.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(shown)
        self.image_label.configure(image=self.tk_image)

    def save_image(self):
        if self.current is None or self.rendered is None:
            return
        rec, _, _ = self.current
        top_n = self._top_n()
        DEFAULT_EXPORT.mkdir(parents=True, exist_ok=True)
        initial = f"{rec['name']}_gaussian_s8_top{top_n}_scores.png"
        path = filedialog.asksaveasfilename(
            title="表示画像を保存", initialdir=str(DEFAULT_EXPORT),
            initialfile=initial, defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
        )
        if path:
            self.rendered.save(path)
            self.status_var.set(f"保存しました: {path}")


def smoke_test(output: Path):
    paths = image_jsons()
    if len(paths) != 21:
        raise RuntimeError(f"expected 21 JSON files, found {len(paths)}")
    target = next(path for path in paths if path.stem == "image18")
    rec, ranked, annotations = analyze(target)
    if len(ranked) < 10:
        raise RuntimeError("image18 has fewer than 10 candidates")
    output.parent.mkdir(parents=True, exist_ok=True)
    render_overlay(rec, ranked, annotations, 10).save(output)
    print(
        f"OK: {rec['name']}, candidates={len(ranked)}, "
        f"annotations={len(annotations)}, output={output}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", type=Path)
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test(args.smoke_test)
    else:
        Viewer().mainloop()


if __name__ == "__main__":
    main()
