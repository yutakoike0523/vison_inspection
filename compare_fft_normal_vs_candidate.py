"""Direct FFT spectrum comparison: image15 region171 vs eight nearby normal candidates."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data20260903" / "data20260903" / "triming" / "image15_triming.png"
OUT = ROOT / "outputs" / "⑪_FFT正常候補比較" / "image15"
GRID = 16
TARGET_ID = 171
CONTROL_IDS = [154, 155, 156, 170, 172, 186, 187, 188]
FREQ_BANDS = [(0.0, 0.025, "low"), (0.025, 0.08, "mid"), (0.08, 0.5, "high")]


def bounds(length, n):
    return [(i * length // n, (i + 1) * length // n) for i in range(n)]


def analyze_patch(patch):
    p = patch.astype(np.float64)
    p -= p.mean()  # same DC removal for every sample
    hann = np.outer(np.hanning(p.shape[0]), np.hanning(p.shape[1]))
    f = np.fft.fftshift(np.fft.fft2(p * hann))
    mag = np.abs(f)
    power = mag ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(p.shape[0]))
    fx = np.fft.fftshift(np.fft.fftfreq(p.shape[1]))
    fxg, fyg = np.meshgrid(fx, fy)
    radius = np.hypot(fxg, fyg)
    dr = max(1.0 / p.shape[1], 1.0 / p.shape[0])
    bins = np.floor(radius / dr).astype(int)
    n_bins = int(np.floor(np.hypot(0.5, 0.5) / dr)) + 1
    valid = radius > 0
    counts = np.bincount(bins[valid].ravel(), minlength=n_bins)
    amp_sum = np.bincount(bins[valid].ravel(), weights=mag[valid].ravel(), minlength=n_bins)
    power_sum = np.bincount(bins[valid].ravel(), weights=power[valid].ravel(), minlength=n_bins)
    mean_amp = amp_sum / np.maximum(counts, 1)
    mean_amp /= max(float(mean_amp.max()), 1e-20)
    power_fraction = power_sum / max(float(power_sum.sum()), 1e-20)
    centers = (np.arange(n_bins) + 0.5) * dr
    valid_bin = counts > 0
    # Shape-only 2D display: normalize each magnitude map to its own maximum.
    mag_shape = mag / max(float(mag.max()), 1e-20)
    return {"fx": fx, "fy": fy, "mag_db": 20*np.log10(np.maximum(mag_shape, 1e-5)),
            "frequency": centers[valid_bin], "radial_mean_magnitude": mean_amp[valid_bin],
            "radial_power_fraction": power_fraction[valid_bin], "dr": dr,
            "hann": hann, "mean": float(patch.mean()), "std": float(patch.std()),
            "total_power": float(power.sum())}


def band_fraction(freq, profile, low, high):
    m = (freq >= low) & (freq < high)
    return float(profile[m].sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gray = np.asarray(Image.open(INPUT).convert("L"), dtype=np.float64)
    h, w = gray.shape
    ys, xs = bounds(h, GRID), bounds(w, GRID)
    tr0, tc0 = divmod(TARGET_ID - 1, GRID)
    ty0, ty1 = ys[tr0]; tx0, tx1 = xs[tc0]
    patch_h, patch_w = ty1 - ty0, tx1 - tx0
    ids = [*CONTROL_IDS, TARGET_ID]
    regions = {}
    for region_id in ids:
        row, col = divmod(region_id - 1, GRID)
        y0, y1 = ys[row]; x0, x1 = xs[col]
        # 16-way integer splits can differ by 1 px; center-crop every patch to
        # the target's exact 70x89 dimensions for a fair frequency-bin comparison.
        y0 += (y1 - y0 - patch_h) // 2
        x0 += (x1 - x0 - patch_w) // 2
        y1, x1 = y0 + patch_h, x0 + patch_w
        patch = gray[y0:y1, x0:x1]
        regions[region_id] = {"row": row+1, "column": col+1, "x1": x0, "y1": y0,
                              "x2": x1, "y2": y1, "patch": patch, **analyze_patch(patch)}
    target = regions[TARGET_ID]
    controls = [regions[i] for i in CONTROL_IDS]
    # 1D plots: equal-energy normalized spectra with normal-candidate median and IQR.
    freq = target["frequency"]
    amp_controls = np.stack([np.interp(freq, r["frequency"], r["radial_mean_magnitude"]) for r in controls])
    pow_controls = np.stack([np.interp(freq, r["frequency"], r["radial_power_fraction"]) for r in controls])
    amp_target = target["radial_mean_magnitude"]
    pow_target = target["radial_power_fraction"]
    amp_med, amp_q1, amp_q3 = np.median(amp_controls, axis=0), np.percentile(amp_controls,25,axis=0), np.percentile(amp_controls,75,axis=0)
    pow_med, pow_q1, pow_q3 = np.median(pow_controls, axis=0), np.percentile(pow_controls,25,axis=0), np.percentile(pow_controls,75,axis=0)
    target_power_distance = 0.5 * np.abs(pow_target - pow_med).sum()
    control_loo_distances = []
    for i in range(len(CONTROL_IDS)):
        loo_ref = np.median(np.delete(pow_controls, i, axis=0), axis=0)
        loo_ref /= max(float(loo_ref.sum()), 1e-20)
        control_loo_distances.append(float(0.5*np.abs(pow_controls[i]-loo_ref).sum()))

    # Save the numeric curve data so peak and band observations can be reproduced.
    curve_rows=[]
    for j, fval in enumerate(freq):
        item={"frequency_cycles_per_pixel":float(fval),
              "candidate_R171_radial_mean_magnitude":float(amp_target[j]),
              "candidate_R171_radial_power_fraction":float(pow_target[j]),
              "normal_candidate_median_radial_mean_magnitude":float(amp_med[j]),
              "normal_candidate_q1_radial_mean_magnitude":float(amp_q1[j]),
              "normal_candidate_q3_radial_mean_magnitude":float(amp_q3[j]),
              "normal_candidate_median_radial_power_fraction":float(pow_med[j]),
              "normal_candidate_q1_radial_power_fraction":float(pow_q1[j]),
              "normal_candidate_q3_radial_power_fraction":float(pow_q3[j])}
        for rid, profile, power_profile in zip(CONTROL_IDS, amp_controls, pow_controls):
            item[f"R{rid}_radial_mean_magnitude"]=float(profile[j])
            item[f"R{rid}_radial_power_fraction"]=float(power_profile[j])
        curve_rows.append(item)
    with (OUT/"radial_spectra.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(curve_rows[0]));writer.writeheader();writer.writerows(curve_rows)

    fig, axes = plt.subplots(2,1,figsize=(11,9),sharex=True,constrained_layout=True)
    for r in controls:
        axes[0].plot(r["frequency"],r["radial_mean_magnitude"],color="gray",alpha=.45,lw=.9)
        axes[1].plot(r["frequency"],r["radial_power_fraction"],color="gray",alpha=.45,lw=.9)
    axes[0].fill_between(freq,amp_q1,amp_q3,color="gray",alpha=.22,label="Normal-candidate IQR")
    axes[0].plot(freq,amp_med,color="black",lw=2,label="Normal-candidate median")
    axes[0].plot(freq,amp_target,color="red",lw=1.8,label=f"Candidate R{TARGET_ID}")
    axes[0].set(ylabel="Radial mean magnitude / own max",title="1D radial FFT amplitude: same-sized patches, same Hann preprocessing")
    axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].fill_between(freq,pow_q1,pow_q3,color="gray",alpha=.22,label="Normal-candidate IQR")
    axes[1].plot(freq,pow_med,color="black",lw=2,label="Normal-candidate median")
    axes[1].plot(freq,pow_target,color="red",lw=1.8,label=f"Candidate R{TARGET_ID}")
    axes[1].set(xlabel="Spatial frequency [cycles/pixel]",ylabel="Power fraction per radial bin",title="Radial power distribution (direction averaged)",xlim=(0,.5))
    axes[1].legend(); axes[1].grid(alpha=.25)
    fig.savefig(OUT/"02_radial_spectra_overlay.png",dpi=160);plt.close(fig)

    # Region map with comparable same-sized boxes.
    fig,ax=plt.subplots(figsize=(10,9),constrained_layout=True)
    ax.imshow(gray,cmap="gray",vmin=0,vmax=255)
    for rid in ids:
        r=regions[rid]; target_flag=rid==TARGET_ID
        color="red" if target_flag else "lime"
        ax.add_patch(Rectangle((r["x1"],r["y1"]),r["x2"]-r["x1"],r["y2"]-r["y1"],fill=False,
                               edgecolor=color,linewidth=2 if target_flag else 1.2))
        ax.text(r["x1"]+2,r["y1"]+12,str(rid),color=color,fontsize=8,
                bbox={"facecolor":"black","alpha":.55,"pad":1})
    ax.set(title="image15: candidate region 171 (red) and nearby normal candidates (green)",xlim=(0,w),ylim=(h,0));ax.axis("off")
    fig.savefig(OUT/"01_region_selection.png",dpi=160);plt.close(fig)

    # 2D FFT montage, all panels share dB limits and frequency axes.
    fig,axes=plt.subplots(3,3,figsize=(12,12),constrained_layout=True)
    all_db=[]
    for rid in ids: all_db.extend(regions[rid]["mag_db"].ravel())
    vmin=-45;vmax=0
    for ax,rid in zip(axes.flat,ids):
        r=regions[rid]
        im=ax.imshow(r["mag_db"],origin="lower",cmap="magma",vmin=vmin,vmax=vmax,
                     extent=[r["fx"][0],r["fx"][-1],r["fy"][0],r["fy"][-1]],aspect="auto")
        ax.axhline(0,color="white",lw=.4,alpha=.55);ax.axvline(0,color="white",lw=.4,alpha=.55)
        ax.set_title(f"R{rid}"+(" candidate" if rid==TARGET_ID else " control"),color="red" if rid==TARGET_ID else "black")
        ax.set_xlabel("fx [cycles/pixel]");ax.set_ylabel("fy [cycles/pixel]")
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.75,label="Relative magnitude [dB; each patch max=0 dB]")
    fig.savefig(OUT/"03_2d_fft_montage.png",dpi=160);plt.close(fig)

    rows=[]
    for rid in ids:
        r=regions[rid]
        row={"region_id":rid,"role":"candidate" if rid==TARGET_ID else "normal_candidate",
             "row":r["row"],"column":r["column"],"x1":r["x1"],"y1":r["y1"],"x2":r["x2"],"y2":r["y2"],
             "width_px":r["x2"]-r["x1"],"height_px":r["y2"]-r["y1"],"mean_intensity":r["mean"],"std_intensity":r["std"],
             "total_fft_power":r["total_power"]}
        for low,high,name in FREQ_BANDS:
            row[f"radial_power_fraction_{name}"]=band_fraction(r["frequency"],r["radial_power_fraction"],low,high)
        peak_i=int(np.argmax(r["radial_power_fraction"]))
        row["radial_power_peak_frequency_cycles_per_pixel"]=float(r["frequency"][peak_i])
        row["radial_power_peak_fraction"]=float(r["radial_power_fraction"][peak_i])
        rows.append(row)
    with (OUT/"spectral_band_features.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    # Quantify target relative to the small local control group, with caution.
    band_result={}
    for low,high,name in FREQ_BANDS:
        vals=np.array([band_fraction(regions[i]["frequency"],regions[i]["radial_power_fraction"],low,high) for i in CONTROL_IDS])
        tv=band_fraction(target["frequency"],target["radial_power_fraction"],low,high)
        sd=float(vals.std(ddof=1))
        band_result[name]={"frequency_range_cycles_per_pixel":[low,high],"normal_candidate_mean":float(vals.mean()),
                           "normal_candidate_sd":sd,"candidate_region171":tv,
                           "candidate_minus_normal_mean":float(tv-vals.mean()),
                           "difference_in_normal_sd":float((tv-vals.mean())/sd) if sd>1e-12 else None}
    normal_power_median=float(np.median([r["total_power"] for r in controls]))
    normal_peak_fractions=[float(np.max(r["radial_power_fraction"])) for r in controls]
    result={"image":INPUT.name,"target_region":TARGET_ID,"target_coordinates_xyxy":[target["x1"],target["y1"],target["x2"],target["y2"]],
            "normal_candidate_regions":CONTROL_IDS,"patch_size_px":[target["x2"]-target["x1"],target["y2"]-target["y1"]],
            "preprocessing":"grayscale; subtract patch mean; separable Hann window; FFT2; fftshift; power; radial bins; normalize radial power to sum 1",
            "frequency_resolution_cycles_per_pixel":{"fx":1.0/(target["x2"]-target["x1"]),"fy":1.0/(target["y2"]-target["y1"])},
            "band_comparison":band_result,
            "target_total_fft_power":target["total_power"],
            "normal_candidate_median_total_fft_power":normal_power_median,
            "target_to_normal_median_total_fft_power_ratio":float(target["total_power"]/normal_power_median),
            "target_mean_intensity":target["mean"],
            "normal_candidate_mean_intensity_median":float(np.median([r["mean"] for r in controls])),
            "radial_power_spectrum_distance_to_normal_candidate_median":float(target_power_distance),
            "normal_candidate_leave_one_out_distance_median":float(np.median(control_loo_distances)),
            "normal_candidate_leave_one_out_distance_max":float(np.max(control_loo_distances)),
            "candidate_distance_exceeds_all_normal_leave_one_out_distances":bool(target_power_distance>max(control_loo_distances)),
            "radial_power_peak_frequency_cycles_per_pixel":{"candidate_R171":float(target["frequency"][np.argmax(target["radial_power_fraction"])]),
                "normal_candidate_range":[float(min(r["frequency"][np.argmax(r["radial_power_fraction"])] for r in controls)),
                                          float(max(r["frequency"][np.argmax(r["radial_power_fraction"])] for r in controls))]},
            "radial_power_peak_height_fraction":{"candidate_R171":float(np.max(target["radial_power_fraction"])),
                "normal_candidate_median":float(np.median(normal_peak_fractions)),
                "normal_candidate_range":[float(min(normal_peak_fractions)),float(max(normal_peak_fractions))]},
            "interpretation_note":"Nearby controls are only visually normal candidates, not verified ground truth. Eight controls from one image do not establish a general defect signature."}
    (OUT/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")

    normal_ids = ", ".join(map(str,CONTROL_IDS))
    readme=f"""# image15 正常候補・傷候補のFFTスペクトル比較

## 対象

- 傷候補: region171、x={target['x1']}–{target['x2']}、y={target['y1']}–{target['y2']} px（黒いC字状模様を含む）
- 正常候補: region {normal_ids}（同画像内の周囲区画。正常の確認ラベルはない）
- 全領域は同じ {target['x2']-target['x1']}×{target['y2']-target['y1']} px、同じ平均値除去・Hann窓で処理しました。

## 直接観察

FFTの1D比較は`02_radial_spectra_overlay.png`です。赤線がregion171、細い灰色線が正常候補8区画、黒線が正常候補の中央値、灰色帯が四分位範囲です。上段は各領域の最大値を1にしたRadial平均振幅、下段は合計1にしたRadialパワー分布です。Radial平均は方向を平均しているため、角度差は`03_2d_fft_montage.png`で確認します。

2D FFT画像は各領域の最大振幅を0 dBにそろえ、同じ−45～0 dB表示範囲です。絶対的なFFT強度の比較ではなく、ピークの形・方向・広がりを見ます。総パワー・輝度・帯域別パワーはCSVに分けてあります。

### 周波数帯域の参考値

{chr(10).join(f"- {name}: {val['frequency_range_cycles_per_pixel'][0]:.3f}–{val['frequency_range_cycles_per_pixel'][1]:.3f} cycles/pixel — controls mean {val['normal_candidate_mean']:.4f} (SD {val['normal_candidate_sd']:.4f}); region171 {val['candidate_region171']:.4f}; difference {val['difference_in_normal_sd'] if val['difference_in_normal_sd'] is not None else float('nan'):.2f} SD" for name,val in band_result.items())}

正規化したRadialパワー曲線と正常候補中央値との距離はregion171が {target_power_distance:.3f}、正常候補を1つずつ外して計算した距離は中央値 {np.median(control_loo_distances):.3f}、最大 {np.max(control_loo_distances):.3f} でした（候補が最大より大きい: {target_power_distance>max(control_loo_distances)}）。最大パワー位置はregion171が {target['frequency'][np.argmax(target['radial_power_fraction'])]:.4f} cycles/pixel、正常候補では {result['radial_power_peak_frequency_cycles_per_pixel']['normal_candidate_range'][0]:.4f}–{result['radial_power_peak_frequency_cycles_per_pixel']['normal_candidate_range'][1]:.4f} の範囲でした。

平均輝度はregion171が {target['mean']:.1f}、近傍候補の中央値が {np.median([r['mean'] for r in controls]):.1f} です。平均輝度が近くても、パワー分布や総パワーは異なり得ます。絶対FFT総パワーはregion171が対照中央値の {target['total_power']/normal_power_median:.2f} 倍でした。これは黒い模様のコントラストでも変わります。

Radialパワーの最大ピーク高さ（全パワーに対する割合）はregion171が {result['radial_power_peak_height_fraction']['candidate_R171']:.3f}、正常候補中央値が {result['radial_power_peak_height_fraction']['normal_candidate_median']:.3f}、候補範囲が {result['radial_power_peak_height_fraction']['normal_candidate_range'][0]:.3f}–{result['radial_power_peak_height_fraction']['normal_candidate_range'][1]:.3f} でした。ピーク位置はregion171が正常候補の位置範囲内なので、単一ピークの位置だけで傷を分けられる結果ではありません。

帯域境界は現象を整理するための暫定設定で、データを見て後付けした探索的比較です。画像サイズが小さいため周波数分解能はfx={result['frequency_resolution_cycles_per_pixel']['fx']:.4f}、fy={result['frequency_resolution_cycles_per_pixel']['fy']:.4f} cycles/pixelで、特に低周波の細かなピーク位置を区別しにくいです。

## 結論と限界

region171は既存16×16解析のFFT相対スコアで256区画中1位でした。この資料では初めて周囲正常候補とのRadialスペクトルと2D FFTを直接並べました。ここから分かるのは、region171の周波数分布が8つの近傍候補の範囲とどの程度重なるかです。正常と傷のラベルがないため、差があっても傷由来とは断定できません。また近傍8区画は独立画像でも独立サンプルでもありません。

本来の検証には、傷のない領域を複数箇所・複数画像から取り、同じ照明・面位置・サイズで比較する必要があります。region171の黒い特徴は輝度変化だけでもスペクトルを変えるため、格子の乱れが原因かを見るには、局所輝度を正規化したスペクトルや円環状の方向別2D成分も今後比較します。

## 出力

- `01_region_selection.png`: 赤=region171候補、緑=正常候補の位置。
- `02_radial_spectra_overlay.png`: 1D周波数スペクトルの重ね合わせと正常候補のばらつき。
- `03_2d_fft_montage.png`: 2Dスペクトルの方向・ピーク形状比較。
- `spectral_band_features.csv`: 領域座標、輝度、総パワー、帯域別パワー。
- `radial_spectra.csv`: プロットの元になった各領域のRadial曲線データ。
- `summary.json`: 処理条件・解像度・比較値。
"""
    (OUT/"README.md").write_text(readme,encoding="utf-8")
    print(f"Saved direct spectrum comparison to {OUT}")


if __name__ == "__main__":
    main()
