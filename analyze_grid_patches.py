"""Cut predicted checker-grid patches and rank them by normalized template error.

Usage:
    python analyze_grid_patches.py
    python analyze_grid_patches.py --input-dir PATH --phase-dir PATH --output-dir PATH
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data20260903" / "data20260903" / "triming"
DEFAULT_PHASE = ROOT / "outputs" / "trimmed_grid_phase_local_contrast"
DEFAULT_OUTPUT = ROOT / "outputs" / "trimmed_grid_patch_analysis"
STAGES = {
    "①_白側パッチ": "位相推定した白側格子中心から切り出したグレースケール画像",
    "②_パッチ一覧": "切り出した全パッチと明るさ正規化後の一覧",
    "③_代表パターンと差分": "全パッチの中央値テンプレートと高スコア例",
    "④_位置とスコア": "元画像上の格子点と相対パッチ差スコア",
    "⑤_数値レポート": "画像別・パッチ別のCSVとJSON",
}


def predicted_centers(width, height, vector_a, vector_b, origin, parity, margin):
    a, b = np.asarray(vector_a, float), np.asarray(vector_b, float)
    matrix = np.column_stack([a, b])
    corners = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], float)
    uv = (corners - origin) @ np.linalg.inv(matrix).T
    imin, jmin = np.floor(uv.min(axis=0)).astype(int) - 2
    imax, jmax = np.ceil(uv.max(axis=0)).astype(int) + 2
    result = []
    for i in range(imin, imax + 1):
        for j in range(jmin, jmax + 1):
            if (i + j) % 2 != parity:
                continue
            x, y = origin + i * a + j * b
            if margin <= x < width - margin and margin <= y < height - margin:
                result.append((float(x), float(y), i, j))
    return sorted(result, key=lambda p: (p[1], p[0]))


def make_montage(patches, labels, output, title, normalized=False, scores=None, columns=8):
    if not patches:
        return
    rows = math.ceil(len(patches) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.65, rows * 1.8), squeeze=False,
                             constrained_layout=True)
    for index, ax in enumerate(axes.ravel()):
        if index >= len(patches):
            ax.axis("off")
            continue
        image = patches[index]
        if normalized:
            ax.imshow(image, cmap="gray", vmin=-3, vmax=3, interpolation="nearest")
        else:
            ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        score_text = f" E={scores[index]:.2f}" if scores is not None else ""
        ax.set_title(f"{labels[index]}{score_text}", fontsize=7, pad=2)
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--phase-dir", type=Path, default=DEFAULT_PHASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--patch-fraction", type=float, default=.75,
                        help="Patch width as a fraction of the shorter autocorrelation vector")
    parser.add_argument("--columns", type=int, default=8)
    args = parser.parse_args()
    input_dir, phase_dir, output_dir = args.input_dir.resolve(), args.phase_dir.resolve(), args.output_dir.resolve()
    image_paths = sorted(input_dir.glob("image*_triming.png"),
                         key=lambda p: int(re.search(r"image(\d+)", p.stem).group(1)))
    if not image_paths:
        parser.error(f"No image*_triming.png images in {input_dir}")
    for folder in STAGES:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    summary_rows, patch_rows = [], []
    for image_path in image_paths:
        key = re.search(r"image\d+", image_path.stem).group(0)
        report_path = phase_dir / "⑥_数値レポート" / f"{key}_report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing phase report: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        phase = report.get("grid_phase_alignment")
        if not phase:
            summary_rows.append({"image": image_path.name, "status": "no_grid_phase"})
            continue
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
        height, width = gray.shape
        a, b = phase["grid_vector_a"], phase["grid_vector_b"]
        origin = np.asarray(phase["grid_origin"], dtype=float)
        spacing = min(np.linalg.norm(a), np.linalg.norm(b))
        patch_size = max(24, int(round(spacing * args.patch_fraction)))
        if patch_size % 2:
            patch_size += 1
        half = patch_size // 2
        centers = predicted_centers(width, height, a, b, origin,
                                    int(phase["selected_checker_parity"]), half)

        raw_patches, normalized_patches, labels, records = [], [], [], []
        image_raw_dir = output_dir / "①_白側パッチ" / key
        image_raw_dir.mkdir(parents=True, exist_ok=True)
        for patch_id, (x, y, i, j) in enumerate(centers, start=1):
            cx, cy = int(round(x)), int(round(y))
            patch = gray[cy-half:cy+half, cx-half:cx+half]
            if patch.shape != (patch_size, patch_size):
                continue
            patch_float = patch.astype(np.float32)
            mean, std = float(patch_float.mean()), float(patch_float.std())
            normalized = (patch_float - mean) / max(std, 1e-6)
            raw_patches.append(patch.copy())
            normalized_patches.append(normalized)
            labels.append(f"{patch_id:03d}")
            filename = f"patch_{patch_id:03d}_x{cx}_y{cy}.png"
            Image.fromarray(patch).save(image_raw_dir / filename)
            records.append({"patch_id": patch_id, "center_x": cx, "center_y": cy,
                            "lattice_i": i, "lattice_j": j, "mean_brightness": mean,
                            "std_brightness": std, "patch_file": filename})

        if not normalized_patches:
            summary_rows.append({"image": image_path.name, "status": "no_valid_patches"})
            continue
        normalized_stack = np.stack(normalized_patches)
        template = np.median(normalized_stack, axis=0)
        errors = np.mean(np.abs(normalized_stack - template[None, :, :]), axis=(1, 2))
        for record, error in zip(records, errors):
            record["normalized_median_absolute_error"] = float(error)
            record["relative_rank_descending"] = int(np.count_nonzero(errors > error) + 1)
            patch_rows.append({"image": image_path.name, **record})

        label_by_id = [f"{r['patch_id']}" for r in records]
        make_montage(raw_patches, label_by_id,
                     output_dir / "②_パッチ一覧" / f"{key}_patches_raw.png",
                     f"{key}: predicted white-side patches ({patch_size}x{patch_size})",
                     columns=args.columns)
        make_montage(normalized_patches, label_by_id,
                     output_dir / "②_パッチ一覧" / f"{key}_patches_normalized.png",
                     f"{key}: per-patch normalized patches", normalized=True,
                     columns=args.columns)

        template_path = output_dir / "③_代表パターンと差分" / f"{key}_median_template.png"
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
        im = axes[0].imshow(template, cmap="gray", vmin=-3, vmax=3, interpolation="nearest")
        axes[0].set_title("Median normalized template")
        axes[1].imshow(np.median(np.abs(normalized_stack - template[None, :, :]), axis=0),
                       cmap="magma", interpolation="nearest")
        axes[1].set_title("Median absolute pixel difference")
        for ax in axes: ax.axis("off")
        fig.colorbar(im, ax=axes[0], fraction=.046)
        fig.savefig(template_path, dpi=160)
        plt.close(fig)

        top_indices = np.argsort(errors)[::-1][:min(16, len(errors))]
        make_montage([raw_patches[k] for k in top_indices],
                     [f"{records[k]['patch_id']}" for k in top_indices],
                     output_dir / "③_代表パターンと差分" / f"{key}_highest_difference_patches.png",
                     f"{key}: patches with highest template difference",
                     scores=[float(errors[k]) for k in top_indices], columns=4)

        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
        ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
        scatter = ax.scatter([r["center_x"] for r in records], [r["center_y"] for r in records],
                             c=errors, cmap="inferno", s=34, edgecolors="white", linewidths=.35)
        ax.set(xlim=(0, width), ylim=(height, 0), title=f"{key}: relative patch deviation E = mean(|P' - T|)")
        fig.colorbar(scatter, ax=ax, label="Normalized mean absolute difference")
        fig.savefig(output_dir / "④_位置とスコア" / f"{key}_patch_score_overlay.png", dpi=160)
        plt.close(fig)

        score_csv = output_dir / "⑤_数値レポート" / f"{key}_patch_scores.csv"
        with score_csv.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
        result = {
            "image": image_path.name, "status": "relative_scores_only",
            "patch_count": len(records), "patch_size_px": patch_size,
            "vector_a_px": a, "vector_b_px": b, "grid_origin_xy": origin.tolist(),
            "selected_checker_parity": phase["selected_checker_parity"],
            "patch_normalization": "(patch - patch_mean) / patch_std",
            "template": "pixelwise median of all normalized patches from this image",
            "score": "mean absolute difference from this image's normalized median template",
            "median_E": float(np.median(errors)), "p90_E": float(np.percentile(errors, 90)),
            "max_E": float(np.max(errors)),
            "highest_score_patch_ids": [int(records[k]["patch_id"]) for k in top_indices[:10]],
            "patches": records,
            "caveat": "Relative novelty only; without labeled normal patches or a validated threshold, high E is not a confirmed defect.",
        }
        (output_dir / "⑤_数値レポート" / f"{key}_patch_report.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_rows.append({"image": image_path.name, "status": result["status"],
                             "patch_count": len(records), "patch_size_px": patch_size,
                             "median_E": result["median_E"], "p90_E": result["p90_E"],
                             "max_E": result["max_E"],
                             "top_patch_ids": ",".join(map(str, result["highest_score_patch_ids"]))})
        print(f"{image_path.name}: {len(records)} patches, size={patch_size}, median E={result['median_E']:.3f}", flush=True)

    summary_path = output_dir / "⑤_数値レポート" / "全画像_patch_summary.csv"
    if summary_rows:
        with summary_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in summary_rows for k in r)))
            writer.writeheader(); writer.writerows(summary_rows)
    readme = ["# 格子点パッチ抽出と相対差スコア", "",
              f"入力画像: `{input_dir}`", f"位相・ベクトルのレポート: `{phase_dir}`", "",
              "## 工程", ""]
    readme.extend(f"- `{folder}/` — {description}" for folder, description in STAGES.items())
    readme += ["", "## 手順と解釈", "",
               "自己相関から得た2本の格子ベクトルを固定し、局所コントラストで最適化した位相とチェッカー parity から白側格子点を再生成しました。格子点ごとに、短いベクトルの75%を一辺とする正方形パッチを切り出しています。",
               "各パッチは個別に平均0・標準偏差1へ正規化し、その画像内の全正規化パッチの画素ごとの中央値をテンプレートにしました。`E = mean(abs(P' - T))` が大きいほど、その画像内の多数派パターンから異なります。",
               "これは教師なしの相対スコアです。画像内に傷が多い場合、照明むら・曲面・位相ずれも高スコアになる場合があり、傷の確定判定には使えません。高スコアパッチの一覧と元画像上の位置を合わせて確認してください。", ""]
    (output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
