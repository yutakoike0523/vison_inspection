"""Compare local FFT, local brightness contrast, and edge features on image15 region 171."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data20260903" / "data20260903" / "triming" / "image15_triming.png"
OUT = ROOT / "outputs" / "⑩_単純手法比較" / "image15"
GRID = 16
TARGET = 171


def boundaries(length, n):
    return [(i * length // n, (i + 1) * length // n) for i in range(n)]


def robust_score(values):
    a = np.asarray(values, float)
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    scale = 1.4826 * mad
    if scale < 1e-9:
        scale = np.std(a)
    return np.zeros_like(a) if scale < 1e-9 else np.clip(np.abs((a - med) / scale), 0, 10)


def high_rank(values):
    a = np.asarray(values, float)
    return np.array([1 + np.count_nonzero(a > x) for x in a])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gray = np.asarray(Image.open(INPUT).convert("L"), dtype=np.uint8)
    h, w = gray.shape
    ys, xs = boundaries(h, GRID), boundaries(w, GRID)

    # Low-frequency background estimates illumination; positive residual is local darkness.
    gf = gray.astype(np.float32)
    local_bg = cv2.GaussianBlur(gf, (0, 0), sigmaX=12, sigmaY=12)
    dark_residual = np.maximum(local_bg - gf, 0)
    gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.hypot(gx, gy)
    lo, hi = np.percentile(gradient, [82, 94])
    edges = cv2.Canny(gray, threshold1=float(lo), threshold2=float(hi), L2gradient=True)

    fft_features_path = ROOT / "outputs" / "⑧_マルチスケールFFT解析" / "image15" / "16x16" / "features.csv"
    with fft_features_path.open(encoding="utf-8-sig", newline="") as f:
        fft = {int(r["region_id"]): r for r in csv.DictReader(f)}

    records = []
    rid = 0
    for r, (y0, y1) in enumerate(ys):
        for c, (x0, x1) in enumerate(xs):
            rid += 1
            patch = gray[y0:y1, x0:x1].astype(float)
            dark = dark_residual[y0:y1, x0:x1]
            grad = gradient[y0:y1, x0:x1]
            ep = edges[y0:y1, x0:x1]
            med = float(np.median(patch))
            mad = float(np.median(np.abs(patch - med)))
            dark_tail_threshold = med - 2.5 * max(1.4826 * mad, 1.0)
            records.append({
                "region_id": rid, "row": r+1, "column": c+1,
                "x1": x0, "y1": y0, "x2": x1, "y2": y1,
                "mean_intensity": float(patch.mean()), "median_intensity": med,
                "std_intensity": float(patch.std()),
                "dark_pixel_fraction_below_local_median_minus_2p5mad": float(np.mean(patch < dark_tail_threshold)),
                "mean_positive_dark_residual": float(dark.mean()),
                "p95_positive_dark_residual": float(np.percentile(dark, 95)),
                "mean_gradient_magnitude": float(grad.mean()),
                "canny_edge_fraction": float(np.mean(ep > 0)),
                "canny_low_threshold_global": float(lo), "canny_high_threshold_global": float(hi),
                "local_fft_difference_score": float(fft[rid]["combined_score"]),
                "local_fft_rank": int(fft[rid]["relative_rank"]),
            })
    feature_fields = ["mean_intensity", "mean_positive_dark_residual", "p95_positive_dark_residual",
                      "dark_pixel_fraction_below_local_median_minus_2p5mad", "mean_gradient_magnitude", "canny_edge_fraction"]
    for key in feature_fields:
        vals = [r[key] for r in records]
        name = key + "_relative_score"
        rankname = key + "_rank_high"
        scores, ranks = robust_score(vals), high_rank(vals)
        for i, rec in enumerate(records):
            rec[name] = float(scores[i]); rec[rankname] = int(ranks[i])
    # A low mean brightness can be a useful cue, so keep low-intensity rank explicitly.
    means = np.asarray([r["mean_intensity"] for r in records])
    low_rank = np.array([1 + np.count_nonzero(means < x) for x in means])
    for i, rec in enumerate(records):
        rec["mean_intensity_rank_low"] = int(low_rank[i])

    with (OUT / "features_comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)

    # Context is a 3x3 neighborhood around region 171.
    target = records[TARGET-1]
    tr, tc = target["row"]-1, target["column"]-1
    context = [rec for rec in records if abs(rec["row"]-1-tr) <= 1 and abs(rec["column"]-1-tc) <= 1]

    # Whole-image maps with identical target/control boxes.
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
    panes = [(gray, "gray", "Original grayscale"), (dark_residual, "magma", "Positive local-dark residual (background σ=12 px)"),
             (edges, "gray", f"Canny edges (global thresholds {lo:.1f}/{hi:.1f})")]
    for ax, (im, cmap, title) in zip(axes, panes):
        ax.imshow(im, cmap=cmap)
        for rec in context:
            col = "red" if rec["region_id"] == TARGET else "yellow"
            ax.add_patch(Rectangle((rec["x1"],rec["y1"]),rec["x2"]-rec["x1"],rec["y2"]-rec["y1"],
                                   fill=False,edgecolor=col,linewidth=2 if col=="red" else 1))
        ax.set(title=title,xlim=(0,w),ylim=(h,0)); ax.axis("off")
    fig.savefig(OUT/"01_brightness_and_edge_maps.png",dpi=150);plt.close(fig)

    # Same 9 crops, showing brightness contrast, edges and actual 2D FFT.
    fig, axes = plt.subplots(4, len(context), figsize=(3.1*len(context), 12), constrained_layout=True)
    for j, rec in enumerate(context):
        x0,y0,x1,y1 = rec["x1"],rec["y1"],rec["x2"],rec["y2"]
        crop=gray[y0:y1,x0:x1]
        axes[0,j].imshow(crop,cmap="gray",vmin=0,vmax=255)
        axes[0,j].set_title(f"R{rec['region_id']}"+(" (target)" if rec["region_id"]==TARGET else ""));axes[0,j].axis("off")
        axes[1,j].imshow(dark_residual[y0:y1,x0:x1],cmap="magma",vmin=0,vmax=max(1,float(np.percentile(dark_residual[y0:y1,x0:x1],99))))
        axes[1,j].set_title(f"Dark resid mean {rec['mean_positive_dark_residual']:.1f}; p95 {rec['p95_positive_dark_residual']:.1f}\nmean intensity {rec['mean_intensity']:.1f}");axes[1,j].axis("off")
        axes[2,j].imshow(edges[y0:y1,x0:x1],cmap="gray",vmin=0,vmax=255)
        axes[2,j].set_title(f"Edge fraction {rec['canny_edge_fraction']:.3f}\nrank {rec['canny_edge_fraction_rank_high']}/256");axes[2,j].axis("off")
        f=crop.astype(np.float32);f=(f-f.mean())*np.outer(np.hanning(f.shape[0]),np.hanning(f.shape[1]))
        spec=np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(f))))
        axes[3,j].imshow(spec,cmap="magma",origin="lower")
        axes[3,j].set_title(f"Local FFT {rec['local_fft_difference_score']:.2f}\nrank {rec['local_fft_rank']}/256");axes[3,j].axis("off")
    for ax, lab in zip(axes[:,0],["Original","Brightness","Canny edges","2D FFT"]):ax.set_ylabel(lab,fontsize=11)
    fig.suptitle("Same image15 regions: brightness / edge / local FFT")
    fig.savefig(OUT/"02_same_region_method_comparison.png",dpi=160);plt.close(fig)

    # Full 16x16 heatmaps for each family, all using rank so units are not confused.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    fields=[("local_fft_rank","Local FFT rank (1 = most different)"),
            ("mean_positive_dark_residual_rank_high","Dark residual rank"),
            ("canny_edge_fraction_rank_high","Canny edge-density rank"),
            ("mean_intensity_rank_low","Low-brightness rank (1 = darkest)")]
    for ax,(field,title) in zip(axes.flat,fields):
        grid=np.array([rec[field] for rec in records]).reshape(GRID,GRID)
        im=ax.imshow(grid,cmap="viridis_r" if "low" not in field else "viridis",vmin=1,vmax=GRID*GRID)
        rr,cc=divmod(TARGET-1,GRID);ax.add_patch(Rectangle((cc-.5,rr-.5),1,1,fill=False,edgecolor="red",linewidth=2))
        ax.set(title=title,xlabel="Column",ylabel="Row");fig.colorbar(im,ax=ax,shrink=.78)
    fig.savefig(OUT/"03_full_image_rank_maps.png",dpi=150);plt.close(fig)

    target_out = next(r for r in records if r["region_id"]==TARGET)
    summary={"image":INPUT.name,"target_region":TARGET,"target_bbox_xyxy":[target["x1"],target["y1"],target["x2"],target["y2"]],
             "target_values":target_out,"context_region_ids":[r["region_id"] for r in context],
             "methods":{"brightness":"local Gaussian background subtraction, sigma=12 px; region mean and dark-tail fraction",
                        "edges":f"Sobel magnitude and Canny with fixed whole-image percentile thresholds {lo:.3f}/{hi:.3f}",
                        "fft":"existing 16x16 local FFT composite difference score"},
             "note":"Relative ranks are within image15 16x16 regions. They indicate candidates, not defect decisions."}
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    readme=f"""# image15 単純手法比較：局所FFT・輝度差・エッジ

region 171（x={target['x1']}–{target['x2']}, y={target['y1']}–{target['y2']} px）を傷候補として、その周囲8区画と比較しました。各順位はimage15の16×16全256区画内の相対順位で、1位が最も高い値です。

## 今回の測定

- 平均輝度：region171は {target_out['mean_intensity']:.1f}（暗い側の順位 {target_out['mean_intensity_rank_low']}/256）。
- 暗い局所残差：region171は平均 {target_out['mean_positive_dark_residual']:.2f}（順位 {target_out['mean_positive_dark_residual_rank_high']}/256）、p95 {target_out['p95_positive_dark_residual']:.2f}（順位 {target_out['p95_positive_dark_residual_rank_high']}/256）。各画素からσ=12 pxのGaussian背景を引いた正の暗さを集計しました。暗い画素比はregion171が {target_out['dark_pixel_fraction_below_local_median_minus_2p5mad']:.3f}、256区画中{target_out['dark_pixel_fraction_below_local_median_minus_2p5mad_rank_high']}/256位でした。
- エッジ：Cannyのエッジ画素比は {target_out['canny_edge_fraction']:.4f}、順位 {target_out['canny_edge_fraction_rank_high']}/256。閾値は画像全体のSobel勾配分布から共通設定し、閾値は {lo:.1f}/{hi:.1f} です。
- 局所FFT：既存のスペクトル差スコアは {target_out['local_fft_difference_score']:.2f}、順位 {target_out['local_fft_rank']}/256。

## 出力

- `01_brightness_and_edge_maps.png`: 元画像、局所的な暗さ残差、Cannyエッジ。赤枠がregion171、黄枠が比較区画です。
- `02_same_region_method_comparison.png`: 同じ9区画の元画像、暗さ残差、エッジ、2D FFTを比較します。
- `03_full_image_rank_maps.png`: FFT、暗さ残差、エッジ密度、暗い輝度の全画像内順位です。
- `features_comparison.csv`: 全256区画の輝度・暗部・勾配・エッジ・FFTの数値と順位です。
- `summary.json`: 設定とregion171の数値です。

高い順位は傷の確定ではありません。黒い模様そのものが輝度残差・エッジ・周波数を変えるため、格子の乱れを検出したかは別に確認する必要があります。正常領域にも同程度の反応があるか、画像とヒートマップを照合してください。
"""
    (OUT/"README.md").write_text(readme,encoding="utf-8")
    print(f"Saved simple-baseline comparison to {OUT}")


if __name__ == "__main__":
    main()
