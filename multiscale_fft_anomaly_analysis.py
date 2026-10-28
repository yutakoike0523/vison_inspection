"""Independent 4x4 / 8x8 / 16x16 local FFT comparison for trimmed images.

No autocorrelation, grid vector, or phase output is used. Feature scores compare
each region's normalized spectrum with the pixelwise median spectrum of all
regions at the same scale in the same image.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data20260903" / "data20260903" / "triming"
DEFAULT_OUTPUT = ROOT / "outputs" / "⑧_マルチスケールFFT解析"
GRID_SIZES = [4, 8, 16]
LOW_FREQ_EXCLUDE_RADIUS_RATIO = 1.25  # multiplied by the FFT frequency-bin spacing
DIRECTION_CENTERS_DEG = [0, 45, 90, 135]


def robust_abs_z(values):
    x = np.asarray(values, dtype=float)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        scale = float(np.std(x))
    if scale <= 1e-12:
        return np.zeros_like(x)
    return np.clip(np.abs((x - med) / scale), 0, 10)


def region_bounds(length, count):
    # Integer floor boundaries include every pixel, with remainder pixels at edges.
    return [(i * length // count, (i + 1) * length // count) for i in range(count)]


def radial_profile(values, radius, valid, dr):
    bins = np.floor(radius / dr).astype(int)
    n_bins = int(bins.max()) + 1
    sums = np.bincount(bins[valid].ravel(), weights=values[valid].ravel(), minlength=n_bins)
    counts = np.bincount(bins[valid].ravel(), minlength=n_bins)
    profile = sums / np.maximum(counts, 1)
    centers = (np.arange(n_bins) + .5) * dr
    keep = counts > 0
    return centers[keep], profile[keep]


def directional_energy(power, radius, angle, valid):
    total = max(float(power[valid].sum()), 1e-20)
    # Orientations are axial: +theta and theta+180 degrees belong to one group.
    angle_axial = np.mod(angle, np.pi)
    energies = []
    for center_deg in DIRECTION_CENTERS_DEG:
        center = np.deg2rad(center_deg)
        delta = np.abs(np.angle(np.exp(1j * (angle_axial - center))))
        delta = np.minimum(delta, np.pi - delta)
        sector = valid & (delta <= np.deg2rad(22.5))
        energies.append(float(power[sector].sum() / total))
    return energies


def analyze_scale(gray, count, low_freq_ratio):
    h, w = gray.shape
    ys, xs = region_bounds(h, count), region_bounds(w, count)
    fft_h = max(y1 - y0 for y0, y1 in ys)
    fft_w = max(x1 - x0 for x0, x1 in xs)
    fy = np.fft.fftshift(np.fft.fftfreq(fft_h))
    fx = np.fft.fftshift(np.fft.fftfreq(fft_w))
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    radius = np.hypot(fx_grid, fy_grid)
    angle = np.arctan2(fy_grid, fx_grid)
    df = max(1.0 / fft_w, 1.0 / fft_h)
    low_cut = low_freq_ratio * df
    valid = radius > low_cut
    dr = df

    records, spectra, radial_records = [], [], []
    region_id = 0
    for row, (y0, y1) in enumerate(ys):
        for col, (x0, x1) in enumerate(xs):
            region_id += 1
            crop = gray[y0:y1, x0:x1].astype(np.float64)
            mean = float(crop.mean())
            std = float(crop.std())
            centered = crop - mean  # removes local DC / average brightness
            hann = np.outer(np.hanning(crop.shape[0]), np.hanning(crop.shape[1]))
            tapered = centered * hann
            # Zero-pad remainder-sized edge regions to one common FFT shape.
            spectrum = np.fft.fftshift(np.fft.fft2(tapered, s=(fft_h, fft_w)))
            magnitude = np.abs(spectrum)
            power = magnitude ** 2
            outside_power = float(power[valid].sum())
            normalized_magnitude = np.zeros_like(magnitude)
            normalized_magnitude[valid] = magnitude[valid] / max(float(magnitude[valid].sum()), 1e-20)
            normalized_power = np.zeros_like(power)
            normalized_power[valid] = power[valid] / max(outside_power, 1e-20)
            peak_map = np.where(valid, magnitude, -np.inf)
            py, px = np.unravel_index(int(np.argmax(peak_map)), peak_map.shape)
            peak_mag = float(magnitude[py, px])
            total_power = float(power.sum())
            energies = directional_energy(power, radius, angle, valid)
            rf, rp = radial_profile(normalized_power, radius, valid, dr)
            record = {
                "grid_size": count, "region_id": region_id, "row": row + 1, "column": col + 1,
                "x1": x0, "y1": y0, "x2": x1, "y2": y1,
                "region_width_px": x1 - x0, "region_height_px": y1 - y0,
                "mean_intensity": mean, "std_intensity": std,
                "fft_total_energy": total_power, "fft_non_low_frequency_energy": outside_power,
                "fft_energy_fraction_outside_low_freq": outside_power / max(total_power, 1e-20),
                "dominant_frequency_x": float(fx[px]), "dominant_frequency_y": float(fy[py]),
                "dominant_frequency_magnitude": peak_mag,
                "dominant_frequency_radius": float(radius[py, px]),
                "low_freq_exclude_radius_cycles_per_pixel": low_cut,
                "directional_energy_0": energies[0], "directional_energy_45": energies[1],
                "directional_energy_90": energies[2], "directional_energy_135": energies[3],
                "fft_2d_difference_score": 0.0, "radial_difference_score": 0.0,
                "directional_difference_score": 0.0, "combined_score": 0.0,
                "score_z_fft_2d": 0.0, "score_z_radial": 0.0, "score_z_directional": 0.0,
                "hann_window": True, "mean_subtracted": True,
            }
            records.append(record)
            spectra.append(normalized_magnitude)
            radial_records.append({"frequency": rf, "power_profile": rp})

    # Robust coordinate-wise median reference spectrum, normalized to unit L1.
    reference = np.median(np.stack(spectra, axis=0), axis=0)
    reference[~valid] = 0
    reference /= max(float(reference.sum()), 1e-20)
    two_d = np.array([.5 * float(np.abs(s - reference).sum()) for s in spectra])

    # Interpolate radial profiles onto the common frequency centers before comparing.
    common_freq = np.arange(1, int(np.ceil(np.hypot(.5, .5) / dr))) * dr
    radial_matrix = []
    for item in radial_records:
        profile = np.interp(common_freq, item["frequency"], item["power_profile"], left=0, right=0)
        profile /= max(float(profile.sum()), 1e-20)
        radial_matrix.append(profile)
    radial_matrix = np.asarray(radial_matrix)
    radial_ref = np.median(radial_matrix, axis=0)
    radial_ref /= max(float(radial_ref.sum()), 1e-20)
    radial_diff = np.array([.5 * np.abs(p - radial_ref).sum() for p in radial_matrix])

    direction_matrix = np.array([[r[f"directional_energy_{d}"] for d in DIRECTION_CENTERS_DEG]
                                 for r in records], dtype=float)
    direction_matrix /= np.maximum(direction_matrix.sum(axis=1, keepdims=True), 1e-20)
    direction_ref = np.median(direction_matrix, axis=0)
    direction_ref /= max(float(direction_ref.sum()), 1e-20)
    direction_diff = .5 * np.abs(direction_matrix - direction_ref[None, :]).sum(axis=1)

    z_2d = robust_abs_z(two_d)
    z_radial = robust_abs_z(radial_diff)
    z_direction = robust_abs_z(direction_diff)
    combined = (z_2d + z_radial + z_direction) / 3.0
    for i, record in enumerate(records):
        record["fft_2d_difference_score"] = float(two_d[i])
        record["radial_difference_score"] = float(radial_diff[i])
        record["directional_difference_score"] = float(direction_diff[i])
        record["combined_score"] = float(combined[i])
        record["score_z_fft_2d"] = float(z_2d[i])
        record["score_z_radial"] = float(z_radial[i])
        record["score_z_directional"] = float(z_direction[i])
        record["relative_rank"] = int(np.count_nonzero(combined > combined[i]) + 1)
    return records, spectra, radial_records, radial_matrix, direction_matrix, reference, valid, fx, fy, low_cut, common_freq


def save_region_map(gray, records, count, path):
    h, w = gray.shape
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    for rec in records:
        ax.add_patch(Rectangle((rec["x1"], rec["y1"]), rec["x2"]-rec["x1"], rec["y2"]-rec["y1"],
                               fill=False, edgecolor="cyan", linewidth=.8 if count > 4 else 1.5))
        if count <= 8:
            ax.text((rec["x1"]+rec["x2"])/2, (rec["y1"]+rec["y2"])/2, str(rec["region_id"]),
                    color="white", ha="center", va="center", fontsize=9,
                    bbox={"facecolor":"black", "alpha":.55, "pad":1.5})
    ax.set(xlim=(0,w), ylim=(h,0), title=f"{count}x{count} region map; row-major region IDs")
    fig.savefig(path, dpi=150); plt.close(fig)


def save_fft_grid(spectra, records, count, valid, path):
    shown = [np.log1p(s * 1e5) for s in spectra]
    pool = np.concatenate([a[valid].ravel() for a in shown])
    vmin, vmax = float(np.percentile(pool, 5)), float(np.percentile(pool, 99.5))
    if vmax <= vmin: vmax = vmin + 1e-6
    fig, axes = plt.subplots(count, count, figsize=(max(10,count*.9), max(9,count*.8)), squeeze=False,
                             constrained_layout=True)
    im = None
    for ax, spec, rec in zip(axes.ravel(), shown, records):
        display = spec.copy(); display[~valid] = vmin
        im = ax.imshow(display, cmap="magma", origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{rec['region_id']} E={rec['combined_score']:.1f}", fontsize=max(5, 9-count//4))
        ax.axis("off")
    fig.suptitle(f"{count}x{count}: log(1 + normalized |FFT|); common scale; central low-frequency mask suppressed")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.7, label="log(1 + normalized magnitude × 1e5)")
    fig.savefig(path, dpi=150); plt.close(fig)


def save_radial_spectra(radial_matrix, records, frequencies, path):
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    colors = plt.get_cmap("viridis")
    scores = np.asarray([r["combined_score"] for r in records])
    norm = plt.Normalize(float(scores.min()), float(scores.max()+1e-9))
    for profile, rec in zip(radial_matrix, records):
        ax.plot(frequencies[:len(profile)], profile, color=colors(norm(rec["combined_score"])),
                linewidth=.8, alpha=.72, label=f"R{rec['region_id']}")
    ax.set(xlabel="Spatial frequency [cycles/pixel]", ylabel="Normalized radial power",
           title="Radial 1-D spectrum (reference information; direction is collapsed)")
    ax.grid(alpha=.25)
    if len(records) <= 16:
        ax.legend(ncol=2, fontsize=7, bbox_to_anchor=(1.02,1), loc="upper left")
    else:
        scalar = plt.cm.ScalarMappable(norm=norm, cmap=colors)
        fig.colorbar(scalar, ax=ax, label="Combined score (line color)")
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def save_directional(directions, records, path):
    count = len(records)
    fig, ax = plt.subplots(figsize=(max(10,count*.35), 5), constrained_layout=True)
    x = np.arange(count); width = .19
    for j, deg in enumerate(DIRECTION_CENTERS_DEG):
        ax.bar(x + (j-1.5)*width, directions[:,j], width, label=f"{deg}°")
    ax.set_xticks(x, [str(r["region_id"]) for r in records], fontsize=7)
    ax.set(xlabel="Region ID", ylabel="Fraction of non-low-frequency power",
           title="Directional energy (each region normalized to sum=1)")
    ax.legend(ncol=4); ax.grid(axis="y", alpha=.2)
    fig.savefig(path, dpi=140); plt.close(fig)


def save_heatmap(gray, records, count, vmax, path, field="combined_score"):
    h,w = gray.shape
    values=np.array([r[field] for r in records]).reshape(count,count)
    fig,ax=plt.subplots(figsize=(9,8),constrained_layout=True)
    ax.imshow(gray,cmap="gray",vmin=0,vmax=255)
    im=ax.imshow(values,cmap="inferno",alpha=.55,interpolation="nearest",extent=[0,w,h,0],vmin=0,vmax=max(vmax,1e-6))
    for r in records:
        ax.text((r["x1"]+r["x2"])/2,(r["y1"]+r["y2"])/2,
                f"{r['region_id']}\n{r[field]:.2f}",ha="center",va="center",color="white",fontsize=8,
                bbox={"facecolor":"black","alpha":.5,"pad":1.5})
    ax.set(xlim=(0,w),ylim=(h,0),title=f"{count}x{count} combined FFT difference (relative score)")
    fig.colorbar(im,ax=ax,label="Combined robust score")
    fig.savefig(path,dpi=150);plt.close(fig)


def save_detailed_image15(gray, scale_results, out):
    selected=[3,4,10,11]
    data=scale_results[4]
    records,spectra,radial_matrix,directions=data[0],data[1],data[3],data[4]
    common_freq=data[10]
    indices=[next(i for i,r in enumerate(records) if r["region_id"]==n) for n in selected]
    fig=plt.figure(figsize=(15,10),constrained_layout=True)
    gs=fig.add_gridspec(3,4)
    for col,idx in enumerate(indices):
        r=records[idx]
        crop=gray[r["y1"]:r["y2"],r["x1"]:r["x2"]]
        ax=fig.add_subplot(gs[0,col]);ax.imshow(crop,cmap="gray",vmin=0,vmax=255)
        ax.set_title(f"Region {r['region_id']} crop\ncombined={r['combined_score']:.2f}");ax.axis("off")
        ax=fig.add_subplot(gs[1,col]);disp=np.log1p(spectra[idx]*1e5);disp[~data[6]]=0
        ax.imshow(disp,cmap="magma",origin="lower");ax.set_title("2D FFT shape");ax.axis("off")
    ax=fig.add_subplot(gs[2,:2])
    for idx in indices:ax.plot(common_freq,radial_matrix[idx],label=f"R{records[idx]['region_id']}")
    ax.set(title="Radial power profiles",xlabel="Frequency [cycles/pixel]",ylabel="Normalized power");ax.grid(alpha=.2);ax.legend()
    ax=fig.add_subplot(gs[2,2:])
    x=np.arange(4);barw=.18
    for j,d in enumerate(DIRECTION_CENTERS_DEG):ax.bar(x+(j-1.5)*barw,[directions[idx,j] for idx in indices],barw,label=f"{d}°")
    ax.set_xticks(x,[f"R{n}" for n in selected]);ax.set(title="Directional energy",ylabel="Normalized energy");ax.legend(ncol=2);ax.grid(axis="y",alpha=.2)
    fig.suptitle("image15 debug view: regions 3, 4, 10, 11 (not defect labels)")
    fig.savefig(out,dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",type=Path,default=DEFAULT_INPUT)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--grid-sizes",type=int,nargs="+",default=GRID_SIZES)
    parser.add_argument("--low-freq-exclude-radius-ratio",type=float,default=LOW_FREQ_EXCLUDE_RADIUS_RATIO,
                        help="low-frequency radius as a multiple of the FFT-bin spacing (default 1.25)")
    parser.add_argument("--only",type=str,default=None,help="Optional image stem filter, e.g. image15")
    args=parser.parse_args()
    if any(n<2 for n in args.grid_sizes):parser.error("grid sizes must be >=2")
    paths=sorted(args.input_dir.resolve().glob("image*_triming.png"),key=lambda p:int(re.search(r"image(\d+)",p.stem).group(1)))
    if args.only: paths=[p for p in paths if re.match(r"image\d+",p.stem).group(0)==args.only]
    if not paths:parser.error(f"No matching trimmed images in {args.input_dir}")
    root=args.output_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    all_csv=[];global_reports=[]
    for path in paths:
        image_key=re.search(r"image\d+",path.stem).group(0)
        image_out=root/image_key;image_out.mkdir(parents=True,exist_ok=True)
        gray=np.asarray(Image.open(path).convert("L"),dtype=np.uint8)
        scale_results={};all_scores=[]
        for count in args.grid_sizes:
            result=analyze_scale(gray,count,args.low_freq_exclude_radius_ratio)
            records,spectra,radial_records,radial_matrix,directions,reference,valid,fx,fy,low_cut,common_freq=result
            scale_results[count]=result
            all_scores.extend(r["combined_score"] for r in records)
        shared_vmax=max(all_scores) if all_scores else 1.0
        scale_reports=[]
        for count,result in scale_results.items():
            records,spectra,radial_records,radial_matrix,directions,reference,valid,fx,fy,low_cut,common_freq=result
            scale_dir=image_out/f"{count}x{count}";scale_dir.mkdir(parents=True,exist_ok=True)
            save_region_map(gray,records,count,scale_dir/"01_region_map.png")
            save_fft_grid(spectra,records,count,valid,scale_dir/"02_fft_2d_grid.png")
            save_radial_spectra(radial_matrix,records,common_freq,scale_dir/"03_radial_spectra.png")
            save_directional(directions,records,scale_dir/"04_directional_energy.png")
            save_heatmap(gray,records,count,shared_vmax,scale_dir/"05_anomaly_heatmap.png")
            with (scale_dir/"features.csv").open("w",encoding="utf-8-sig",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
            rank=sorted(records,key=lambda r:r["combined_score"],reverse=True)
            scale_reports.append({"grid_size":count,"region_count":len(records),"low_freq_exclude_radius_cycles_per_pixel":low_cut,
                                  "top_regions":[{"region_id":r["region_id"],"fft_2d_difference_score":r["fft_2d_difference_score"],
                                                  "radial_difference_score":r["radial_difference_score"],
                                                  "directional_difference_score":r["directional_difference_score"],
                                                  "combined_score":r["combined_score"]} for r in rank[:8]]})
            all_csv.extend({"image_name":path.name,**r} for r in records)

        # Side-by-side scale comparison using one shared combined-score color limit.
        fig,axes=plt.subplots(2,2,figsize=(14,12),constrained_layout=True)
        h,w=gray.shape;axes[0,0].imshow(gray,cmap="gray",vmin=0,vmax=255);axes[0,0].set(title="Original",xlim=(0,w),ylim=(h,0));axes[0,0].axis("off")
        ims=[]
        for ax,count in zip(axes.ravel()[1:],args.grid_sizes):
            records=scale_results[count][0]
            values=np.array([r["combined_score"] for r in records]).reshape(count,count)
            ax.imshow(gray,cmap="gray",vmin=0,vmax=255)
            im=ax.imshow(values,cmap="inferno",alpha=.55,extent=[0,w,h,0],vmin=0,vmax=max(shared_vmax,1e-6))
            ims.append(im);ax.set(title=f"{count}x{count} combined score",xlim=(0,w),ylim=(h,0));ax.axis("off")
            if count<=8:
                for r in records:ax.text((r["x1"]+r["x2"])/2,(r["y1"]+r["y2"])/2,str(r["region_id"]),ha="center",va="center",color="white",fontsize=7)
        if ims:fig.colorbar(ims[-1],ax=axes.ravel().tolist(),shrink=.75,label="Combined robust score; common scale")
        fig.suptitle(f"{image_key}: compare 4x4 / 8x8 / 16x16 spatial scales")
        fig.savefig(image_out/"scale_comparison.png",dpi=150);plt.close(fig)

        if image_key=="image15" and 4 in scale_results:
            save_detailed_image15(gray,scale_results,image_out/"detailed_region_comparison.png")
        per_image={"image_name":path.name,"image_size_wh":[w,h],"low_freq_exclude_radius_ratio":args.low_freq_exclude_radius_ratio,
                   "score_definition":{"fft_2d_difference_score":"0.5 * L1 distance from coordinate-wise median normalized 2D magnitude spectrum",
                                       "radial_difference_score":"0.5 * L1 distance between normalized radial power profiles",
                                       "directional_difference_score":"0.5 * L1 distance between normalized [0,45,90,135] degree energy fractions",
                                       "combined_score":"mean of the three absolute robust z-scores; median/MAD per scale"},
                   "scale_results":scale_reports,
                   "notes":["The pixelwise median spectrum is a within-image reference, not guaranteed ground truth normal.",
                            "Frequency bins inside the configured low-frequency radius are excluded from anomaly scores, not from saved FFT images.",
                            "No damage labels or automatic defect decisions are used."]}
        (image_out/"summary.json").write_text(json.dumps(per_image,ensure_ascii=False,indent=2),encoding="utf-8")
        global_reports.append(per_image)
        for s in scale_reports:
            print(image_key,f"{s['grid_size']}x{s['grid_size']}","top:",", ".join(f"{r['region_id']}={r['combined_score']:.2f}" for r in s["top_regions"][:3]),flush=True)

    csv_path=root/"全画像_multiscale_fft_summary.csv"
    with csv_path.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(all_csv[0]));writer.writeheader();writer.writerows(all_csv)
    (root/"全画像_summary.json").write_text(json.dumps(global_reports,ensure_ascii=False,indent=2),encoding="utf-8")
    readme=["# マルチスケール局所FFT比較", "",
            f"入力: `{args.input_dir.resolve()}`",f"対象画像: {len(paths)}",f"分割スケール: {args.grid_sizes}","",
            "## 処理と出力", "",
            "各画像を4×4、8×8、16×16に分割します。各領域でグレースケール化済みの画素から領域平均を引き、Hann窓を掛け、2D FFT→fftshift→振幅/パワーを計算します。余りの画素も全て使い、端の小領域は共通FFTサイズにゼロパディングします。",
            "低周波マスクはDC/緩やかな照明むらを抑える目的で中心近傍をスコア計算から除外します。`LOW_FREQ_EXCLUDE_RADIUS_RATIO` はFFT基本ビン幅に掛ける係数です。マスク範囲は各CSV/JSONとスペクトル図に記録されます。",
            "各領域の使用可能な2D振幅スペクトルは総和1に正規化します。基準は同じ画像・同じ分割スケールにおける周波数座標ごとの中央値スペクトルです。Radialは参考値、2Dスペクトル差を主要な形状比較、方向別エネルギー差も独立指標として出し、combinedは3つの絶対ロバストz-scoreの平均です。", "",
            "各 `imageXX/<scale>/` には領域マップ、2D FFT一覧、Radialスペクトル、方向別エネルギー、異常度ヒートマップ、features.csvを保存します。画像15にはregion 3/4/10/11の詳細比較もあります。全画像CSVは `全画像_multiscale_fft_summary.csv` です。", "",
            "## 解釈上の注意", "",
            "高スコアは同一画像・同一スケール内の多数派スペクトルからの違いを表します。基準スペクトル自体に傷領域が含まれる可能性があるため、スコアは傷の確定判定ではありません。方向別エネルギーは4方向の扇形領域ごとのパワーを、非低周波パワー合計で割った割合です。", "",
            "Radial 1D spectrumは方向情報を平均で失います。周波数方向の情報が必要な時は2D FFT図および方向別エネルギーを見てください。", "",
            "実行: `python multiscale_fft_anomaly_analysis.py`。画像15のみ試す場合: `python multiscale_fft_anomaly_analysis.py --only image15`。", ""]
    (root/"README.md").write_text("\n".join(readme),encoding="utf-8")
    print(f"Processed {len(paths)} images. Output: {root}")


if __name__=="__main__":
    main()
