import os
import glob
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------
# 1. 設定・ハイパーパラメータ
# ----------------------------------------------------------------------
BLOCK_SIZES = [256, 128]
DC_RADIUS = 5            # 中央から±5ピクセルを除外
USE_LOG_TRANSFORM = False # コサイン類似度計算時のスペクトルのログ変換（True / False）
OUTPUT_ROOT = "outputs/fft2_cosine_similarity"

# ----------------------------------------------------------------------
# 2. 周波数領域の統一（正規化周波数の対応）
# ----------------------------------------------------------------------
def match_frequency_resolution(global_spec, block_size, global_shape):
    """
    画像全体のFFTスペクトルから、ブロックFFTの最高周波数（ナイキスト周波数）に
    相当する領域を正規化周波数座標ベースで切り出し、ブロックサイズと同じ大きさにリサイズします。
    """
    gh, gw = global_shape
    bh, bw = block_size, block_size
    
    # 画像全体とブロックの最大正規化周波数は 0.5 cycles/pixel で共通。
    # したがって、画像全体FFTの「中心からブロックサイズに比例した範囲」が、
    # ブロックFFTの全周波数領域（-0.5 〜 0.5）に対応します。
    crop_h = int(gh * (bh / gh)) if gh > bh else gh
    crop_w = int(gw * (bw / gw)) if gw > bw else gw
    
    cy, cx = gh // 2, gw // 2
    
    # 偶数サイズを想定した正確な切り出し座標
    y1 = cy - crop_h // 2
    y2 = y1 + crop_h
    x1 = cx - crop_w // 2
    x2 = x1 + crop_w
    
    cropped = global_spec[y1:y2, x1:x2]
    
    # ブロックサイズ (bw, bh) と同じピクセル数になるようバイリニア補間
    # これにより、1ピクセルあたりの周波数解像度（Hz/pixel）が完全に一致します。
    global_spec_matched = cv2.resize(cropped, (bw, bh), interpolation=cv2.INTER_LINEAR)
    return global_spec_matched

# ----------------------------------------------------------------------
# 3. FFT基本処理 & DC成分除外
# ----------------------------------------------------------------------
def compute_fft_spectrum(img, apply_hann=True):
    """2次元FFTを行い、振幅スペクトル（fftshift済み）を取得します"""
    img_f = img.astype(np.float64)
    if apply_hann:
        hann_y = np.hanning(img.shape[0])
        hann_x = np.hanning(img.shape[1])
        window = np.outer(hann_y, hann_x)
        img_f *= window
        
    f_co = np.fft.fft2(img_f)
    f_shift = np.fft.fftshift(f_co)
    amplitude = np.abs(f_shift)
    return amplitude

def mask_dc_component(spectrum, radius=5):
    """スペクトルの中心（DC成分付近）を指定半径で0（除外）にします"""
    masked = spectrum.copy()
    cy, cx = masked.shape[0] // 2, masked.shape[1] // 2
    masked[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1] = 0
    return masked

# ----------------------------------------------------------------------
# 4. コサイン類似度計算
# ----------------------------------------------------------------------
def compute_cosine_similarity(spec_a, spec_b, log_transform=False):
    """2つのスペクトルのコサイン類似度を計算します"""
    if log_transform:
        a = np.log1p(spec_a).flatten()
        b = np.log1p(spec_b).flatten()
    else:
        a = spec_a.flatten()
        b = spec_b.flatten()
        
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(np.dot(a, b) / (norm_a * norm_b + 1e-12))

# ----------------------------------------------------------------------
# 5. メインパイプライン
# ----------------------------------------------------------------------
def process_inspection_images(image_dict, label_dict):
    """
    image_dict: { image_name: gray_img_ndarray }
    label_dict: { image_name: { (x, y, w, h): "scratch" } } 
    """
    all_blocks_data = []
    
    for block_size in BLOCK_SIZES:
        print(f"--- Processing Block Size: {block_size} ---")
        
        for img_name, img in image_dict.items():
            h, w = img.shape
            img_labels = label_dict.get(img_name, {})
            
            # 出力ディレクトリ作成
            img_out_dir = os.path.join(OUTPUT_ROOT, str(block_size), img_name)
            os.makedirs(img_out_dir, exist_ok=True)
            
            # --- 全体FFTの計算 ---
            global_amp = compute_fft_spectrum(img, apply_hann=True)
            global_amp_masked = mask_dc_component(global_amp, radius=DC_RADIUS)
            
            # 全体基準スペクトルの周波数統一化
            global_ref_spec = match_frequency_resolution(global_amp_masked, block_size, (h, w))
            
            # --- ブロック分割とFFT ---
            blocks_info = []
            block_idx = 0
            
            # 重複なしの単純なグリッド分割
            for y in range(0, h - block_size + 1, block_size):
                for x in range(0, w - block_size + 1, block_size):
                    block_img = img[y : y + block_size, x : x + block_size]
                    
                    # ブロックFFT
                    b_amp = compute_fft_spectrum(block_img, apply_hann=True)
                    b_amp_masked = mask_dc_component(b_amp, radius=DC_RADIUS)
                    
                    # コサイン類似度
                    sim = compute_cosine_similarity(global_ref_spec, b_amp_masked, log_transform=USE_LOG_TRANSFORM)
                    
                    # ラベル判定 (ブロックの中心や重複度合いに応じて既存仕様に合わせ可能)
                    # 簡易的に、ブロック領域が傷ラベルの矩形と交差しているかで判定
                    label = "unknown"
                    for (lx, ly, lw, lh), l_type in img_labels.items():
                        # 交差判定
                        if not (x + block_size <= lx or lx + lw <= x or y + block_size <= ly or ly + lh <= y):
                            label = l_type
                            break
                            
                    blocks_info.append({
                        "image_name": img_name,
                        "block_size": block_size,
                        "block_id": block_idx,
                        "x": x, "y": y, "width": block_size, "height": block_size,
                        "cosine_similarity": sim,
                        "label": label,
                        "global_fft_energy": float(np.sum(global_ref_spec)),
                        "block_fft_energy": float(np.sum(b_amp_masked))
                    })
                    block_idx += 1
            
            if not blocks_info:
                continue
                
            df_img_blocks = pd.DataFrame(blocks_info)
            
            # --- 画像内統計（順位・中央値差）の計算 ---
            medi = df_img_blocks["cosine_similarity"].median()
            df_img_blocks["similarity_minus_image_median"] = df_img_blocks["cosine_similarity"] - medi
            # 類似度が高い順に1位〜とするためのランク
            df_img_blocks["similarity_rank"] = df_img_blocks["cosine_similarity"].rank(ascending=False, method="min")
            # 傷が低類似度になる仮説のため、パーセンタイル（低い方から何％か）を算出
            df_img_blocks["image_percentile"] = df_img_blocks["cosine_similarity"].rank(pct=True) * 100
            
            all_blocks_data.extend(df_img_blocks.to_dict(orient="records"))
            
            # --- 6 & 7. 視覚化画像の作成・保存 ---
            # A. 元画像
            cv2.imwrite(os.path.join(img_out_dir, "original.png"), img)
            
            # B. 全体FFT (可視化用はログ変換)
            global_vis = np.log1p(global_amp)
            global_vis = cv2.normalize(global_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(os.path.join(img_out_dir, "global_fft.png"), global_vis)
            
            # C & D 可視化キャンバスの準備
            sim_map = np.zeros((h, w), dtype=np.float32)
            overlay_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            for b in df_img_blocks.to_dict(orient="records"):
                bx, by, bw, bh = b["x"], b["y"], b["width"], b["height"]
                bsim = b["cosine_similarity"]
                blbl = b["label"]
                
                # ヒートマップ用配列への代入
                sim_map[by:by+bh, bx:bx+bw] = bsim
                
                # 類似度が低いほど赤くなるカラー（1に近いほど青、0に近いほど赤）
                # 可視化用に値を0〜255に反転マッピング
                color_val = int((1.0 - np.clip(bsim, 0, 1)) * 255)
                # 簡易的なカラーリング：低類似度＝赤強く、高類似度＝緑
                bgr_color = (0, 255 - color_val, color_val)
                
                # ブロック境界の描画
                cv2.rectangle(overlay_img, (bx, by), (bx+bw, by+bh), bgr_color, 1)
                
                # 類似度テキストの描画
                cv2.putText(overlay_img, f"{bsim:.3f}", (bx + 5, by + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr_color, 1, cv2.LINE_AA)
                
                # 傷ラベルの重ね合わせ（赤太枠）
                if blbl == "scratch":
                    cv2.rectangle(overlay_img, (bx, by), (bx+bw, by+bh), (0, 0, 255), 3)
            
            # C. コサイン類似度ヒートマップの保存
            sim_map_vis = cv2.normalize(sim_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            sim_map_color = cv2.applyColorMap(sim_map_vis, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(img_out_dir, "similarity_map.png"), sim_map_color)
            
            # D. オーバーレイ画像の保存
            cv2.imwrite(os.path.join(img_out_dir, "overlay.png"), overlay_img)

    # --- 8. 全ブロック統合CSV出力 ---
    df_all = pd.DataFrame(all_blocks_data)
    df_all.to_csv(os.path.join(OUTPUT_ROOT, "all_blocks.csv"), index=False)
    
    # --- 9. 比較グラフの作成 ---
    for b_size in BLOCK_SIZES:
        df_sub = df_all[df_all["block_size"] == b_size]
        if df_sub.empty:
            continue
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 箱ひげ図 & ストリッププロット
        sns.boxplot(data=df_sub, x="label", y="cosine_similarity", ax=axes[0], palette="Set2", order=["scratch", "unknown"])
        sns.stripplot(data=df_sub, x="label", y="cosine_similarity", ax=axes[0], color="black", alpha=0.3, order=["scratch", "unknown"])
        axes[0].set_title(f"Block Size {b_size}: Boxplot & Strip Plot")
        
        # ヒストグラム
        sns.histplot(data=df_sub, x="cosine_similarity", hue="label", kde=True, ax=axes[1], element="step", stat="density", common_norm=False, hue_order=["unknown", "scratch"])
        axes[1].set_title(f"Block Size {b_size}: Histogram Density")
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_ROOT, f"scratch_vs_unknown_{b_size}.png"), dpi=150)
        plt.close()
