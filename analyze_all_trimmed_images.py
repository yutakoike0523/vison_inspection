"""Run grid-period analysis for every trimmed image and sort outputs by stage.

Usage:
    python analyze_all_trimmed_images.py
    python analyze_all_trimmed_images.py --input-dir PATH --output-dir PATH
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data20260903" / "data20260903" / "triming"
DEFAULT_OUTPUT = ROOT / "outputs" / "trimmed_grid_phase_local_contrast"
STAGES = {
    "①_入力と明度マスク": "元画像・明度分類・格子候補マスク",
    "②_2次元自己相関": "ずらし量ごとの一致度と周期ピーク候補",
    "③_局所周期比較": "左上・中央・右下の周期比較",
    "④_FFT確認": "周期候補に基づくFFT窓サイズ比較",
    "⑤_格子位置合わせ": "中心円と外側リングの局所コントラストで位相を最適化",
    "⑥_数値レポート": "画像ごとの report.json と一覧表",
}


def stage_for(filename: str) -> str:
    if filename in {"original_grayscale.png", "histogram_classification.png",
                    "grid_candidate_mask.png", "01_input_classification_mask.png"}:
        return "①_入力と明度マスク"
    if filename == "02_autocorrelation.png":
        return "②_2次元自己相関"
    if filename in {"04_local_region_masks.png", "local_periods.json"}:
        return "③_局所周期比較"
    if (filename == "05_fft_size_comparison.png" or filename.startswith((
            "fft_window_", "roi_", "candidate_mask_roi_", "hann_and_logfft_", "mask_fft_"))):
        return "④_FFT確認"
    if filename in {"00_original_grayscale.png", "phase_optimization.json"} or filename.startswith(("06_", "07_", "08_", "09_")):
        return "⑤_格子位置合わせ"
    return "⑥_数値レポート"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--group-index", type=int, default=2,
                        help="1-based histogram peak band used as the grid mask")
    parser.add_argument("--autocorr-source", choices=("mask", "gray"), default="mask")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    images = sorted(input_dir.glob("*.png"), key=lambda p: (int(''.join(filter(str.isdigit, p.stem)) or 0), p.name))
    if not images:
        parser.error(f"No PNG images found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for dirname in STAGES:
        (output_dir / dirname).mkdir(parents=True, exist_ok=True)
    work = output_dir / ".working"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()

    script = ROOT / "grid_period_fft.py"
    rows = []
    for index, image in enumerate(images, start=1):
        image_key = image.stem.replace("_triming", "")
        run_dir = work / image_key
        print(f"[{index}/{len(images)}] {image.name}", flush=True)
        command = [sys.executable, str(script), str(image), "--output", str(run_dir),
                   "--group-index", str(args.group_index), "--autocorr-source", args.autocorr_source]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            error = (result.stderr or result.stdout)[-5000:]
            (output_dir / "⑥_数値レポート" / f"{image_key}_error.txt").write_text(error, encoding="utf-8")
            rows.append({"image": image.name, "status": "error", "error": error[-500:]})
            print(f"  ERROR: {error[-400:]}", flush=True)
            continue
        report_path = run_dir / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        prior_error = output_dir / "⑥_数値レポート" / f"{image_key}_error.txt"
        if prior_error.exists():
            prior_error.unlink()
        for produced in run_dir.iterdir():
            destination = output_dir / stage_for(produced.name) / f"{image_key}_{produced.name}"
            shutil.copy2(produced, destination)
        report_destination = output_dir / "⑥_数値レポート" / f"{image_key}_report.json"
        report_destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        phase = report.get("grid_phase_alignment") or {}
        rows.append({
            "image": image.name,
            "status": report.get("status"),
            "image_size": "x".join(map(str, report.get("image_size_wh", []))),
            "histogram_peaks": ";".join(map(str, report.get("histogram_peak_levels", []))),
            "mask_brightness_range": str(report.get("selected_brightness_range")),
            "mask_fraction": report.get("candidate_mask_fraction"),
            "credible_autocorr_peaks": len(report.get("autocorrelation_peak_candidates", [])),
            "estimated_translation_px": report.get("estimated_grid_spacing_px"),
            "basis_a_dx_dy": str([report["selected_grid_translation_basis"][0][k]
                                    for k in ("dx", "dy")]) if report.get("selected_grid_translation_basis") else "",
            "basis_b_dx_dy": str([report["selected_grid_translation_basis"][1][k]
                                    for k in ("dx", "dy")]) if len(report.get("selected_grid_translation_basis", [])) > 1 else "",
            "selected_checker_parity": phase.get("selected_checker_parity"),
            "phase_score_before": phase.get("phase_score_before"),
            "phase_score_after": phase.get("phase_score_after"),
            "phase_score_improvement": phase.get("phase_score_improvement"),
        })

    csv_path = output_dir / "⑥_数値レポート" / "全画像_summary.csv"
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    readme = ["# トリミング画像の格子周期・自己相関解析", "",
              f"入力: `{input_dir}`", f"画像数: {len(images)}", "",
              "## 工程フォルダ", ""]
    readme.extend(f"- `{folder}/` — {description}" for folder, description in STAGES.items())
    readme += ["", "## 読み方", "",
               "`②_2次元自己相関` の画像は横軸が dx、縦軸が dy のずらし量です。中央 (0,0) の周囲に対称に現れるピークが、元画像と似た模様に戻るシフト候補です。",
               "自己相関は周期と方向を推定しますが、格子の絶対位置は決めません。`⑤_格子位置合わせ` では、固定した2本の格子ベクトルを使い、中心円の平均輝度から外側リングの平均輝度を引いた値が格子全体で最大になる位相を探索します。偶奇2つのサブ格子も比較し、強い側を白側として表示します。",
               "`estimated_grid_spacing_px` は選択された自己相関の平行移動長の代表値です。チェッカー格子では隣接要素・同じ色要素・対角方向などが候補になるため、必ずしも1マス幅を意味しません。",
               "`⑥_数値レポート/全画像_summary.csv` に画像別の推定値と位相スコアをまとめています。スコアは局所コントラスト（輝度階調）で、最適化前後の増加は探索した画像上での適合度を示します。スコア改善だけでは格子中心への一致を独立に証明しないため、重ね合わせ画像も目視確認してください。", ""]
    (output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    shutil.rmtree(work)
    errors = sum(row.get("status") == "error" for row in rows)
    print(f"Finished: {len(images)} images, {errors} errors. Output: {output_dir}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
