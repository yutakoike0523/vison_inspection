"""Compare FFT spectra of grid-aligned image15 patches near a visible mark.

This is an exploratory comparison, not a defect classifier. It reuses patch
centers produced by analyze_grid_patches.py and reports how much the selected
grid patch overlaps the previously marked candidate region.

Run from the repository root:
    python compare_grid_cell_fft_image15.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data20260903" / "data20260903" / "triming" / "image15_triming.png"
PATCH_REPORT = ROOT / "outputs" / "trimmed_grid_patch_analysis"
OUTPUT = ROOT / "outputs" / "⑬_格子セルFFT比較_image15"
# Previously examined black C-shaped candidate region (xyxy, image15_triming coordinates).
CANDIDATE_ROI = (705, 894, 775, 983)
CONTROL_COUNT = 8


def overlap_fraction(center_x: int, center_y: int, size: int, roi: tuple[int, int, int, int]) -> float:
    half = size / 2
    left, top = center_x - half, center_y - half
    right, bottom = center_x + half, center_y + half
    ix = max(0.0, min(right, roi[2]) - max(left, roi[0]))
    iy = max(0.0, min(bottom, roi[3]) - max(top, roi[1]))
    return float(ix * iy / (size * size))


def fft_features(patch: np.ndarray) -> dict:
    image = patch.astype(np.float64)
    centered = image - image.mean()
    wy = np.hanning(image.shape[0])
    wx = np.hanning(image.shape[1])
    window = np.outer(wy, wx)
    spectrum = np.fft.fftshift(np.fft.fft2(centered * window))
    power = np.abs(spectrum) ** 2
    h, w = patch.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    radius = np.hypot(fx_grid, fy_grid)
    df = max(1.0 / h, 1.0 / w)
    power_no_dc = power.copy()
    power_no_dc[h // 2, w // 2] = 0.0  # Remove only the zero-frequency sample.
    ring_index = np.rint(radius.ravel() / df).astype(int)
    ring_count = int(ring_index.max()) + 1
    radial = np.bincount(ring_index, weights=power_no_dc.ravel(), minlength=ring_count)
    centers = np.arange(ring_count, dtype=float) * df
    non_dc_total = float(power_no_dc.sum())
    normalized = radial / max(non_dc_total, 1e-12)

    def band(lo: float, hi: float) -> float:
        mask = (centers >= lo) & (centers < hi)
        return float(normalized[mask].sum())

    valid = (centers > 0) & (centers <= 0.5)
    peak_index = int(np.argmax(normalized[valid]))
    peak_frequency = float(centers[valid][peak_index])
    return {
        "spectrum": spectrum,
        "power": power,
        "radial_frequency": centers,
        "radial_power_fraction": normalized,
        "mean_brightness": float(image.mean()),
        "std_brightness": float(image.std()),
        "total_fft_power": float(power.sum()),
        "non_dc_fft_power": non_dc_total,
        "low_band_fraction_0_0.025": band(0.0, 0.025),
        "mid_band_fraction_0.025_0.08": band(0.025, 0.08),
        "high_band_fraction_0.08_0.5": band(0.08, 0.500001),
        "dominant_radial_frequency_cycles_per_pixel": peak_frequency,
        "dominant_radial_power_fraction": float(normalized[valid][peak_index]),
    }


def load_rows() -> tuple[Path, dict, list[dict]]:
    reports = list(PATCH_REPORT.rglob("image15_patch_report.json"))
    if len(reports) != 1:
        raise FileNotFoundError(f"Expected one image15 patch report under {PATCH_REPORT}; found {len(reports)}")
    path = reports[0]
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("patches") or not data.get("patch_size_px"):
        raise ValueError(f"Patch report has no usable patch data: {path}")
    return path, data, data["patches"]


def save_position_plot(gray: np.ndarray, candidate: dict, controls: list[dict], size: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    x1, y1, x2, y2 = CANDIDATE_ROI
    ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="red", linewidth=2.5,
                           label="Visible candidate ROI"))
    half = size / 2
    ax.add_patch(Rectangle((candidate["center_x"]-half, candidate["center_y"]-half), size, size,
                           fill=False, edgecolor="#ffcc00", linewidth=2.5, label="Candidate grid patch"))
    ax.scatter([candidate["center_x"]], [candidate["center_y"]], c="#ffcc00", edgecolors="black", s=65, zorder=4)
    ax.text(candidate["center_x"], candidate["center_y"], f"  C{candidate['patch_id']}", color="#ffcc00",
            fontsize=9, weight="bold", va="center")
    for row in controls:
        ax.add_patch(Rectangle((row["center_x"]-half, row["center_y"]-half), size, size,
                               fill=False, edgecolor="#00e5ff", linewidth=1.3))
        ax.text(row["center_x"], row["center_y"], f"{row['patch_id']}", color="#00e5ff",
                fontsize=8, weight="bold", ha="center", va="center")
    ax.set(title="image15: grid-aligned candidate and same-phase control patches", xlabel="x [px]", ylabel="y [px]")
    ax.legend(loc="upper right")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_patch_montage(patches: list[tuple[str, np.ndarray, str]], path: Path, title: str) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(9, 9), constrained_layout=True)
    for ax, item in zip(axes.ravel(), patches):
        label, patch, note = item
        ax.imshow(patch, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"{label}\n{note}", fontsize=8)
        ax.axis("off")
    for ax in axes.ravel()[len(patches):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_fft_montage(items: list[tuple[str, dict]], path: Path) -> None:
    logs = [np.log1p(item[1]["power"]) for item in items]
    non_dc = []
    for image in logs:
        tmp = image.copy()
        cy, cx = np.array(tmp.shape) // 2
        tmp[max(0, cy-1):cy+2, max(0, cx-1):cx+2] = np.nan
        non_dc.extend(tmp[np.isfinite(tmp)].ravel().tolist())
    vmax = float(np.percentile(non_dc, 99)) if non_dc else 1.0
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)
    image_handle = None
    for ax, ((label, _), log_image) in zip(axes.ravel(), zip(items, logs)):
        image_handle = ax.imshow(log_image, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        cy, cx = np.array(log_image.shape) // 2
        ax.add_patch(Rectangle((cx-1.5, cy-1.5), 3, 3, fill=False, edgecolor="cyan", linewidth=.8))
        ax.set_title(label, fontsize=8)
        ax.axis("off")
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    fig.colorbar(image_handle, ax=axes.ravel().tolist(), shrink=.72, label="log(1 + FFT power), shared scale")
    fig.suptitle("2D FFT log power (center DC area outlined)", fontsize=13)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Input image not found: {INPUT}")
    report_path, report, rows = load_rows()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    gray = np.asarray(Image.open(INPUT).convert("L"), dtype=np.uint8)
    height, width = gray.shape
    size = int(report["patch_size_px"])
    roi_center = ((CANDIDATE_ROI[0]+CANDIDATE_ROI[2])/2, (CANDIDATE_ROI[1]+CANDIDATE_ROI[3])/2)
    valid = [row for row in rows if 0 <= row["center_x"]-size//2 and row["center_x"]+size//2 <= width
             and 0 <= row["center_y"]-size//2 and row["center_y"]+size//2 <= height]
    candidate = min(valid, key=lambda row: math.hypot(row["center_x"]-roi_center[0], row["center_y"]-roi_center[1]))
    candidate_overlap = overlap_fraction(candidate["center_x"], candidate["center_y"], size, CANDIDATE_ROI)
    controls_available = [row for row in valid if row["patch_id"] != candidate["patch_id"]
                          and overlap_fraction(row["center_x"], row["center_y"], size, CANDIDATE_ROI) == 0.0]
    controls = sorted(controls_available,
                      key=lambda row: math.hypot(row["center_x"]-roi_center[0], row["center_y"]-roi_center[1]))[:CONTROL_COUNT]
    if len(controls) < 3:
        raise ValueError(f"Only {len(controls)} non-overlapping same-phase controls found; need at least 3")

    selected = [("candidate", candidate)] + [("control", row) for row in controls]
    patch_items, feature_items = [], []
    for role, row in selected:
        cx, cy = int(row["center_x"]), int(row["center_y"])
        half = size // 2
        patch = gray[cy-half:cy-half+size, cx-half:cx-half+size]
        if patch.shape != (size, size):
            continue
        features = fft_features(patch)
        overlap = overlap_fraction(cx, cy, size, CANDIDATE_ROI)
        record = {
            "role": "傷候補" if role == "candidate" else "正常候補（目視未確定）",
            "patch_id": int(row["patch_id"]), "center_x": cx, "center_y": cy,
            "patch_size_px": size, "candidate_roi_overlap_fraction": overlap,
            "distance_to_candidate_roi_center_px": math.hypot(cx-roi_center[0], cy-roi_center[1]),
            "normalized_template_error_from_existing_report": row.get("normalized_median_absolute_error"),
            **{k: v for k, v in features.items() if k not in {"spectrum", "power", "radial_frequency", "radial_power_fraction"}},
        }
        feature_items.append((role, row, patch, features, record))
        patch_items.append((f"{'CANDIDATE' if role == 'candidate' else 'CONTROL'} patch {row['patch_id']}", patch,
                            f"center=({cx},{cy}), ROI overlap={overlap:.1%}"))

    control_features = [item[3] for item in feature_items if item[0] == "control"]
    candidate_features = next(item[3] for item in feature_items if item[0] == "candidate")
    median_curve = np.median(np.stack([f["radial_power_fraction"] for f in control_features]), axis=0)
    control_curve_stack = np.stack([f["radial_power_fraction"] for f in control_features])
    curve_l1 = float(np.abs(candidate_features["radial_power_fraction"] - median_curve).sum())
    control_loo = []
    for index, feat in enumerate(control_features):
        others = np.delete(control_curve_stack, index, axis=0)
        control_loo.append(float(np.abs(feat["radial_power_fraction"] - np.median(others, axis=0)).sum()))
    for item in feature_items:
        item[4]["radial_spectrum_L1_distance_to_control_median"] = (
            curve_l1 if item[0] == "candidate" else float(np.abs(item[3]["radial_power_fraction"] - median_curve).sum())
        )

    save_position_plot(gray, candidate, controls, size, OUTPUT / "01_patch_positions.png")
    save_patch_montage(patch_items, OUTPUT / "02_patch_montage.png", "Candidate and same-phase grid patches")
    save_fft_montage([(f"{'CANDIDATE' if role == 'candidate' else 'CONTROL'} #{row['patch_id']}", feat)
                      for role, row, _, feat, _ in feature_items], OUTPUT / "03_fft_2d_montage.png")

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for idx, feat in enumerate(control_features):
        ax.plot(feat["radial_frequency"], feat["radial_power_fraction"], color="#73b7c8", alpha=.55,
                linewidth=1.1, label="Controls" if idx == 0 else None)
    ax.plot(control_features[0]["radial_frequency"], median_curve, color="black", linestyle="--", linewidth=2,
            label="Control median")
    ax.plot(candidate_features["radial_frequency"], candidate_features["radial_power_fraction"],
            color="#d62728", linewidth=2.4, label="Candidate")
    ax.set(xlim=(0, .5), xlabel="Spatial frequency [cycles/pixel]",
           ylabel="Fraction of non-DC radial FFT power", title="Radial power spectra: candidate vs same-phase controls")
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(OUTPUT / "04_radial_spectrum_comparison.png", dpi=180)
    plt.close(fig)

    csv_path = OUTPUT / "features.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(feature_items[0][4].keys()))
        writer.writeheader()
        writer.writerows([item[4] for item in feature_items])

    candidate_record = next(item[4] for item in feature_items if item[0] == "candidate")
    control_records = [item[4] for item in feature_items if item[0] == "control"]
    def robust_z(value: float, values: list[float]) -> float | None:
        median = float(np.median(values))
        mad = float(np.median(np.abs(np.asarray(values) - median)))
        return float((value - median) / (1.4826 * mad)) if mad > 1e-12 else None
    metric_names = ["total_fft_power", "non_dc_fft_power", "low_band_fraction_0_0.025",
                    "mid_band_fraction_0.025_0.08", "high_band_fraction_0.08_0.5",
                    "dominant_radial_frequency_cycles_per_pixel", "dominant_radial_power_fraction",
                    "radial_spectrum_L1_distance_to_control_median"]
    metrics = {}
    for name in metric_names:
        values = [float(r[name]) for r in control_records]
        metrics[name] = {
            "candidate": float(candidate_record[name]),
            "control_median": float(np.median(values)),
            "control_mad": float(np.median(np.abs(np.asarray(values)-np.median(values)))),
            "candidate_robust_z_vs_controls": robust_z(float(candidate_record[name]), values),
        }
    result = {
        "image": INPUT.name,
        "input_path": str(INPUT.relative_to(ROOT)),
        "patch_report_path": str(report_path.relative_to(ROOT)),
        "grid_basis_px": {"a": report.get("vector_a_px"), "b": report.get("vector_b_px")},
        "checker_parity": report.get("selected_checker_parity"),
        "candidate_region": {"region": 171, "xyxy": list(CANDIDATE_ROI),
                             "description": "Previously marked black C-shaped visible candidate; not ground-truth labeled."},
        "patch_size_px": size,
        "candidate_patch": {"patch_id": int(candidate["patch_id"]), "center_xy": [candidate["center_x"], candidate["center_y"]],
                            "roi_overlap_fraction": candidate_overlap},
        "control_patches": [{"patch_id": int(row["patch_id"]), "center_xy": [row["center_x"], row["center_y"]],
                             "roi_overlap_fraction": 0.0} for row in controls],
        "preprocessing": "grayscale; subtract patch mean; separable Hann window; fft2; fftshift; power=abs(F)^2; nearest-frequency radial bins with spacing max(1/h,1/w); remove only the center DC sample from normalized radial curve",
        "metric_comparison": metrics,
        "candidate_radial_curve_L1_distance": curve_l1,
        "control_leave_one_out_radial_curve_L1_distances": control_loo,
        "interpretation": {
            "candidate_overlap_warning": candidate_overlap < 0.5,
            "candidate_overlap_note": "Candidate grid patch overlaps less than half of the marked candidate ROI; FFT difference may be diluted or spatially misaligned." if candidate_overlap < 0.5 else "Candidate patch substantially overlaps the marked region.",
            "ground_truth_status": "unknown",
            "control_status": "visual normal candidates only; not verified ground truth",
            "conclusion": "Exploratory image15 comparison only. Spectral differences do not establish that the visible feature is a scratch or that FFT is a reliable defect detector.",
        },
        "outputs": ["01_patch_positions.png", "02_patch_montage.png", "03_fft_2d_montage.png",
                    "04_radial_spectrum_comparison.png", "features.csv"],
    }
    (OUTPUT / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = f"""# image15 格子パッチFFT比較\n\n## 目的\n\n目視で確認できる黒いC字状の領域（従来のregion171）に近い格子パッチと、同じ白側格子位相の周辺パッチを比較する探索実験です。傷の自動判定ではありません。\n\n## 今回の切り出し\n\n- 入力: `{INPUT.relative_to(ROOT)}`\n- 格子位相・中心: `{report_path.relative_to(ROOT)}` の既存レポート\n- パッチ: {size}×{size} px\n- 傷候補領域: `xyxy={CANDIDATE_ROI}`\n- 最近傍候補パッチ: ID {candidate['patch_id']}、中心 ({candidate['center_x']}, {candidate['center_y']})、領域との面積重なり {candidate_overlap:.1%}\n- 比較用: 同じ checker parity で候補領域と重ならない周辺パッチ {len(controls)} 個。正常ラベルは目視候補で、正解確認済みではありません。\n\n**重なり率が50%未満**の場合、傷候補の模様がパッチ内に一部しか入らず、FFT差が薄まる可能性があります。このため結果は参考値として扱います。\n\n## 出力画像の見方\n\n- `01_patch_positions.png`: 赤枠が従来の傷候補範囲、黄枠が最近傍の格子候補パッチ、水色枠が比較用パッチです。まず位置の対応を確認します。\n- `02_patch_montage.png`: 切り出した生パッチです。候補部の特徴が実際に入っているか確認します。\n- `03_fft_2d_montage.png`: 各パッチの2次元FFT対数パワーです。中央の水色枠は0周波数（DC）近傍です。共通スケールで周波数方向・ピークの違いを見ます。\n- `04_radial_spectrum_comparison.png`: 横軸は空間周波数（cycles/pixel）、縦軸は非DC FFTパワーの半径方向分布を全体で正規化した割合です。赤が候補、薄青が比較用、黒破線が比較用中央値です。\n\n## 計算と制約\n\n各パッチをグレースケール化し、平均値を引いてHann窓を適用した後、2次元FFTのパワー `|F|²` を計算しました。1次元スペクトルは周波数半径ごとのパワー和を非DC総パワーで正規化しています。総FFTパワーは別の特徴量として保存します。\n\n`features.csv` と `report.json` に明るさ、総パワー、低・中・高周波パワー比、主ピーク、候補と比較用中央値との差を保存しています。小さいパッチでは周波数分解能が粗く、曲面・照明差・位相合わせ誤差もスペクトルに影響します。image15一枚、少数の未確認正常候補から一般的な傷特徴を断定できません。\n"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote outputs to {OUTPUT}")
    print(f"Candidate patch #{candidate['patch_id']} at ({candidate['center_x']}, {candidate['center_y']}), ROI overlap={candidate_overlap:.1%}")
    print(f"Controls: {[row['patch_id'] for row in controls]}")
    print(f"Candidate normalized radial-spectrum distance={curve_l1:.4f}; control leave-one-out median={np.median(control_loo):.4f}")


if __name__ == "__main__":
    main()
