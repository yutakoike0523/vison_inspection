"""Compare Gabor, local normal-template residual, and local FFT around image15 region 171."""
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
OUT = ROOT / "outputs" / "⑨_手法比較" / "image15"
GRID = 16
TARGET_REGION = 171
CONTEXT_RADIUS = 1
GABOR_FREQS = [0.018, 0.025, 0.04]  # cycles/pixel; centered around observed local grid frequencies
GABOR_ANGLES_DEG = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]


def boundaries(length: int, n: int):
    return [(i * length // n, (i + 1) * length // n) for i in range(n)]


def robust_z(values):
    v = np.asarray(values, float)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    den = 1.4826 * mad
    if den < 1e-9:
        den = np.std(v)
    return np.zeros_like(v) if den < 1e-9 else np.clip(np.abs((v - med) / den), 0, 10)


def zscore_patch(p):
    x = p.astype(np.float32)
    x = cv2.GaussianBlur(x, (0, 0), 1.2)
    x -= x.mean()
    return x / max(float(x.std()), 1e-5)


def fit_shape(patch, shape):
    h, w = shape
    y0 = max(0, (patch.shape[0] - h) // 2)
    x0 = max(0, (patch.shape[1] - w) // 2)
    crop = patch[y0:y0 + h, x0:x0 + w]
    out = np.zeros((h, w), np.float32)
    out[:crop.shape[0], :crop.shape[1]] = crop
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gray = np.asarray(Image.open(INPUT).convert("L"), dtype=np.float32)
    h, w = gray.shape
    ys, xs = boundaries(h, GRID), boundaries(w, GRID)
    regions = {}
    rid = 0
    for r, (y0, y1) in enumerate(ys):
        for c, (x0, x1) in enumerate(xs):
            rid += 1
            regions[rid] = {"row": r, "col": c, "x1": x0, "y1": y0, "x2": x1, "y2": y1,
                            "patch": gray[y0:y1, x0:x1]}
    target = regions[TARGET_REGION]
    tr, tc = target["row"], target["col"]
    context = [n for n, v in regions.items()
               if abs(v["row"] - tr) <= CONTEXT_RADIUS and abs(v["col"] - tc) <= CONTEXT_RADIUS]
    th, tw = target["patch"].shape

    # Gabor bank: quadrature energy in several frequencies and orientations.
    gabor_maps = []
    gabor_vectors = []
    for freq in GABOR_FREQS:
        sigma = max(2.0, 0.56 / freq)
        ksize = int(np.ceil(6 * sigma)) | 1
        for deg in GABOR_ANGLES_DEG:
            theta = np.deg2rad(deg)
            responses = []
            for phase in (0, np.pi / 2):
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, 1.0 / freq,
                                            0.7, phase, ktype=cv2.CV_32F)
                responses.append(cv2.filter2D(gray, cv2.CV_32F, kernel,
                                              borderType=cv2.BORDER_REFLECT101))
            energy = np.hypot(responses[0], responses[1])
            gabor_maps.append(energy)
    for n, v in regions.items():
        x1, y1, x2, y2 = v["x1"], v["y1"], v["x2"], v["y2"]
        vec = []
        for energy in gabor_maps:
            # Ignore filter boundary pixels to avoid artificial edge responses.
            margin = min(ksize // 2, max(0, min(y2-y1, x2-x1)//4))
            crop = energy[y1+margin:y2-margin or y2, x1+margin:x2-margin or x2]
            vec.append(float(np.mean(crop)))
        gabor_vectors.append(vec)
    gabor_vectors = np.asarray(gabor_vectors)
    # Compare normalized feature profiles; retain total response as a separate diagnostic.
    gabor_profiles = gabor_vectors / np.maximum(gabor_vectors.sum(axis=1, keepdims=True), 1e-12)
    gabor_ref = np.median(gabor_profiles, axis=0)
    gabor_raw_diff = 0.5 * np.abs(gabor_profiles - gabor_ref).sum(axis=1)
    gabor_scores = robust_z(gabor_raw_diff)
    gabor_total = gabor_vectors.sum(axis=1)
    gabor_ranks = np.array([1 + np.count_nonzero(gabor_scores > score) for score in gabor_scores])

    # Local normal-template baseline: phase-align the eight adjacent same-sized patches,
    # take their median as a reference, then compare the center after local normalization.
    template_residual = {}
    template_images = {}
    template_residual_images = {}
    template_meta = {}
    window = cv2.createHanningWindow((tw, th), cv2.CV_32F)
    for n in context:
        v = regions[n]
        ref_patch = fit_shape(v["patch"], (th, tw))
        ref_norm = zscore_patch(ref_patch)
        donors, responses, donor_ids = [], [], []
        for dn in context:
            if dn == n:
                continue
            d = fit_shape(regions[dn]["patch"], (th, tw))
            d_norm = zscore_patch(d)
            shift, response = cv2.phaseCorrelate(ref_norm, d_norm, window)
            # phaseCorrelate reports the translation from ref to donor; invert it to align donor to ref.
            # Reject implausibly large translations; they usually indicate a weak/ambiguous match.
            sx, sy = (float(shift[0]), float(shift[1]))
            if abs(sx) > 15 or abs(sy) > 15:
                sx, sy = 0.0, 0.0
                response = 0.0
            matrix = np.float32([[1, 0, -sx], [0, 1, -sy]])
            aligned = cv2.warpAffine(d_norm, matrix, (tw, th), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT101)
            donors.append(aligned)
            responses.append(float(response))
            donor_ids.append(dn)
        template = np.median(np.stack(donors), axis=0)
        residual = np.abs(ref_norm - template)
        # Exclude a small border where translated donors are padded/warped.
        residual_core = residual[4:-4, 4:-4]
        template_residual[n] = float(np.mean(residual_core))
        template_images[n] = template
        template_residual_images[n] = residual
        template_meta[n] = {"mean_phase_correlation_response": float(np.mean(responses)),
                            "max_phase_correlation_response": float(np.max(responses)),
                            "best_donor_region": int(donor_ids[int(np.argmax(responses))]),
                            "reference_donors": donor_ids}

    # Existing local FFT comparison scores on the same 16x16 regions.
    fft_csv = ROOT / "outputs" / "⑧_マルチスケールFFT解析" / "image15" / "16x16" / "features.csv"
    with fft_csv.open(encoding="utf-8-sig", newline="") as f:
        fft_rows = {int(row["region_id"]): row for row in csv.DictReader(f)}

    rows = []
    for i, (n, v) in enumerate(regions.items()):
        rows.append({"region_id": n, "row": v["row"]+1, "column": v["col"]+1,
                     "x1": v["x1"], "y1": v["y1"], "x2": v["x2"], "y2": v["y2"],
                     "is_in_target_context": n in context,
                     "gabor_profile_difference": float(gabor_raw_diff[i]),
                     "gabor_difference_score": float(gabor_scores[i]),
                     "gabor_rank": int(gabor_ranks[i]),
                     "gabor_total_response": float(gabor_total[i]),
                     "normal_template_difference": template_residual.get(n, ""),
                     "template_mean_phase_response": template_meta.get(n, {}).get("mean_phase_correlation_response", ""),
                     "template_registration_reliable": (template_meta[n]["max_phase_correlation_response"] >= 0.25) if n in template_meta else "",
                     "template_best_donor_region": template_meta.get(n, {}).get("best_donor_region", ""),
                     "local_fft_difference_score": float(fft_rows[n]["combined_score"]),
                     "local_fft_rank": int(fft_rows[n]["relative_rank"])})
    with (OUT / "features_comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    # 1) target/context boxes
    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    for n in context:
        v = regions[n]
        color = "red" if n == TARGET_REGION else "yellow"
        ax.add_patch(Rectangle((v["x1"], v["y1"]), v["x2"]-v["x1"], v["y2"]-v["y1"],
                               fill=False, edgecolor=color, linewidth=2 if n == TARGET_REGION else 1))
        ax.text(v["x1"]+3, v["y1"]+12, str(n), color=color, fontsize=8,
                bbox={"facecolor":"black", "alpha":.5, "pad":1})
    ax.set(title="image15: region 171 (red) and 8 neighboring comparison regions", xlim=(0,w), ylim=(h,0))
    ax.axis("off"); fig.savefig(OUT/"01_target_and_controls.png", dpi=160); plt.close(fig)

    # 2) summed Gabor energy map (locally contrast normalized for display)
    gsum = np.sum(gabor_maps, axis=0)
    gdisp = np.log1p(gsum)
    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    ax.imshow(gdisp, cmap="magma", extent=[0,w,h,0]);
    v=target; ax.add_patch(Rectangle((v["x1"],v["y1"]),tw,th,fill=False,edgecolor="cyan",linewidth=2))
    ax.set(title="Gabor bank response energy (all tested frequencies/orientations)", xlim=(0,w), ylim=(h,0)); ax.axis("off")
    fig.colorbar(ax.images[0], ax=ax, label="log(1 + quadrature response energy)")
    fig.savefig(OUT/"02_gabor_response.png", dpi=160); plt.close(fig)

    # 3) shared-context panels: each candidate crop, Gabor response, template residual, FFT score.
    cols = len(context)
    fig, axes = plt.subplots(4, cols, figsize=(3.3*cols, 12), constrained_layout=True)
    for j, n in enumerate(context):
        v=regions[n]; crop=v["patch"]
        axes[0,j].imshow(crop,cmap="gray",vmin=0,vmax=255); axes[0,j].set_title(f"R{n} original" + (" (target)" if n==TARGET_REGION else "")); axes[0,j].axis("off")
        axes[1,j].imshow(np.log1p(gsum[v["y1"]:v["y2"],v["x1"]:v["x2"]]),cmap="magma");
        axes[1,j].set_title(f"Gabor score {gabor_scores[n-1]:.2f}"); axes[1,j].axis("off")
        if n in template_residual:
            axes[2,j].imshow(template_residual_images[n],cmap="inferno")
            axes[2,j].set_title(f"Template diff {template_residual[n]:.2f}\nphase response {template_meta[n]['mean_phase_correlation_response']:.2f}")
        axes[2,j].axis("off")
        fft_crop = crop.astype(np.float32) - float(crop.mean())
        fft_win = np.outer(np.hanning(crop.shape[0]), np.hanning(crop.shape[1]))
        fft_show = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(fft_crop * fft_win))))
        axes[3,j].imshow(fft_show, cmap="magma", origin="lower")
        axes[3,j].set_title(f"2D FFT; score {float(fft_rows[n]['combined_score']):.2f}\nrank {fft_rows[n]['relative_rank']}/256")
        axes[3,j].axis("off")
    axes[0,0].set_ylabel("Crop", fontsize=11); axes[1,0].set_ylabel("Gabor", fontsize=11)
    axes[2,0].set_ylabel("Template residual", fontsize=11); axes[3,0].set_ylabel("FFT", fontsize=11)
    fig.suptitle("Same local regions: Gabor / neighboring normal-template difference / existing local FFT")
    fig.savefig(OUT/"03_method_comparison.png", dpi=160); plt.close(fig)

    target_row = rows[TARGET_REGION-1]
    result = {"image": INPUT.name, "target_region_id": TARGET_REGION,
              "target_bbox_xyxy": [target["x1"], target["y1"], target["x2"], target["y2"]],
              "context_regions": context, "target_method_scores": target_row,
              "template_registration": template_meta[TARGET_REGION],
              "gabor_parameters": {"frequencies_cycles_per_pixel": GABOR_FREQS,
                                   "orientations_degrees": GABOR_ANGLES_DEG,
                                   "quadrature_energy": "sqrt(response_phase0^2 + response_phase90^2)"},
              "interpretation": "Exploratory comparison only. Template registration response must be checked; scores are not defect decisions."}
    (OUT/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    readme = f"""# image15 局所傷候補手法の比較\n\n対象: image15 region 171（x={target['x1']}–{target['x2']}, y={target['y1']}–{target['y2']} px）と周囲8区画です。赤枠が比較対象、黄枠が同画像内の対照候補です。\n\n## 同じ場所で比較した3手法\n\n- **Gabor**: 周波数 {GABOR_FREQS} cycles/pixel、方向 {GABOR_ANGLES_DEG}°の複素直交応答エネルギーを計算し、256区画の応答特徴プロファイルの中央値からの差をスコア化しました。局所位置を保ちながら方向・周期への反応を見ます。\n- **正常テンプレート差分**: 対象の周囲8区画を位相相関で平行移動合わせし、中央値テンプレートと局所平均・コントラスト正規化画像との差を計算しました。region171の平均位相相関応答は {template_meta[TARGET_REGION]['mean_phase_correlation_response']:.3f}（最大 {template_meta[TARGET_REGION]['max_phase_correlation_response']:.3f}）です。応答が低い場合は位置合わせが不確かなので、差分を正常模様との差と解釈できません。\n- **局所FFT**: 既存の16×16解析で計算した合成相対スコアを同じ区画に表示しました（region171: {target_row['local_fft_difference_score']:.2f}、順位 {target_row['local_fft_rank']}/256）。\n\n## 出力の見方\n\n- `01_target_and_controls.png`: 対象・対照区画の位置。\n- `02_gabor_response.png`: 全方向・周波数のGabor応答エネルギー。傷らしい暗点だけでなく格子の周期成分にも反応します。\n- `03_method_comparison.png`: 同じ9区画で元画像、Gabor応答、テンプレート差分、FFT順位を比較します。テンプレート差のタイトルには位置合わせ応答を併記しています。\n- `features_comparison.csv`: 全256区画のGabor・FFT値と、対象近傍9区画のテンプレート差・位置合わせ品質。\n- `summary.json`: 設定と対象の数値。\n\n## 現時点の限界\n\nregion171には黒い線状模様があり、局所FFTでは高スコアでした。ただし黒い模様そのものがスペクトルを変えた可能性があり、格子の乱れを検出した証拠ではありません。Gaborも暗い輪郭や照明むらに反応します。さらに隣接区画の位相相関応答が低ければ、正常テンプレート差分は模様位置合わせ誤差を含みます。各出力を見て、傷候補だけでなく正常対照の誤反応も評価してください。\n"""
    (OUT/"README.md").write_text(readme, encoding="utf-8")
    with (OUT/"README.md").open("a", encoding="utf-8") as f:
        f.write(f"\n\nregion171のGabor差スコアは {target_row['gabor_difference_score']:.2f}（256区画中{target_row['gabor_rank']}位）で、隣接region172は {rows[171]['gabor_difference_score']:.2f}（{rows[171]['gabor_rank']}位）でした。今回の設定では黒い領域だけが最も強く反応したわけではありません。テンプレート差はregion171で {template_residual[TARGET_REGION]:.2f} ですが、位置合わせ品質が低く、近傍との差から傷を区別できませんでした。3手法のうち今回は局所FFTがregion171を最も強く目立たせましたが、原因が格子の乱れか黒い輝度領域かは未確定です。\n")
    print(f"Saved method comparison to {OUT}")


if __name__ == "__main__":
    main()
