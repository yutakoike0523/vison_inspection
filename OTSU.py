from pathlib import Path
import json
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 設定
# ============================================================

IMAGE_PATH = Path(
    r"C:\work\Visual_inspection\data20260903\data20260903\image15.png"
)

LABEL_JSON_PATH = Path(
    r"C:\work\Visual_inspection\data20260903\data20260903\image15.json"
)

OUTPUT_DIR = Path(
    r"C:\work\Visual_inspection\outputs\fft2_cosine_otsu"
)

BLOCK_SIZES = [256, 128]

# FFT設定
USE_HANN = True
USE_LOG_FOR_SIMILARITY = False

# block FFTにそろえた後の中央DC除外半径
DC_RADIUS = 5

# 可視化の透明度
OVERLAY_ALPHA = 0.35


# ============================================================
# LabelMe JSON読み込み
# ============================================================

def load_labelme_boxes(json_path):
    """
    LabelMe形式jsonから、
    各shapeをBounding Boxとして読み込む。

    戻り値:
        [
            {
                "label": "scratch",
                "x1": ...,
                "y1": ...,
                "x2": ...,
                "y2": ...
            },
            ...
        ]
    """
    if not json_path.exists():
        print(f"[WARNING] JSON not found: {json_path}")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    boxes = []

    for shape in data.get("shapes", []):
        points = shape.get("points", [])
        if len(points) < 2:
            continue

        pts = np.array(points, dtype=np.float32)

        x1 = float(np.min(pts[:, 0]))
        y1 = float(np.min(pts[:, 1]))
        x2 = float(np.max(pts[:, 0]))
        y2 = float(np.max(pts[:, 1]))

        boxes.append(
            {
                "label": shape.get("label", "scratch"),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )

    return boxes


# ============================================================
# BlockとBBoxの重なり判定
# ============================================================

def block_label(x, y, w, h, boxes):
    """
    ブロックと傷BBoxが少しでも重なれば、その傷ラベルを返す。
    重ならなければ unknown。
    """

    bx1 = x
    by1 = y
    bx2 = x + w
    by2 = y + h

    labels = []

    for box in boxes:
        ix1 = max(bx1, box["x1"])
        iy1 = max(by1, box["y1"])
        ix2 = min(bx2, box["x2"])
        iy2 = min(by2, box["y2"])

        if ix2 > ix1 and iy2 > iy1:
            labels.append(box["label"])

    if not labels:
        return "unknown"

    return "+".join(sorted(set(labels)))


# ============================================================
# 2D Hann窓
# ============================================================

def hann2d(h, w):
    hy = np.hanning(h)
    hx = np.hanning(w)
    return np.outer(hy, hx)


# ============================================================
# 2次元FFT
# ============================================================

def calc_fft_spectrum(gray, use_hann=True):
    """
    画像からfftshift済みの振幅スペクトルを作る。
    """

    img = gray.astype(np.float32)

    # 平均値を引いてDCを少し抑える
    img = img - np.mean(img)

    if use_hann:
        img = img * hann2d(*img.shape)

    fft = np.fft.fft2(img)
    fft = np.fft.fftshift(fft)

    amplitude = np.abs(fft)

    return amplitude.astype(np.float32)


# ============================================================
# DC除外
# ============================================================

def remove_dc(spectrum, radius=5):
    """
    FFT中央付近を0にする。
    """

    out = spectrum.copy()

    h, w = out.shape
    cy = h // 2
    cx = w // 2

    yy, xx = np.ogrid[:h, :w]

    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2

    out[mask] = 0.0

    return out


# ============================================================
# 全体FFTをBlock FFTと同じ周波数グリッドへ
# ============================================================

def resize_global_spectrum(global_spectrum, block_size):
    """
    fftshift後のスペクトルは、
    横・縦とも -0.5 ～ +0.5 cycles/pixel に対応する。

    そのため全体FFTスペクトルを block_size × block_size に
    リサイズすることで、正規化周波数軸をそろえる。

    INTER_AREAで全体スペクトルを縮小。
    """

    resized = cv2.resize(
        global_spectrum,
        (block_size, block_size),
        interpolation=cv2.INTER_AREA,
    )

    return resized.astype(np.float32)


# ============================================================
# コサイン類似度
# ============================================================

def cosine_similarity(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)

    numerator = np.dot(a, b)

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
        + 1e-12
    )

    return float(numerator / denominator)


# ============================================================
# Otsu法
# ============================================================

def otsu_threshold_float(values, bins=256):
    """
    0～1程度のfloat値に対してOtsu閾値を求める。
    OpenCVの8bit変換をせず、そのまま計算。
    """

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    vmin = values.min()
    vmax = values.max()

    if np.isclose(vmin, vmax):
        return float(vmin)

    hist, edges = np.histogram(
        values,
        bins=bins,
        range=(vmin, vmax)
    )

    centers = (edges[:-1] + edges[1:]) / 2.0

    prob = hist.astype(np.float64)
    prob /= prob.sum()

    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)

    mu_total = mu[-1]

    numerator = (mu_total * omega - mu) ** 2
    denominator = omega * (1.0 - omega)

    sigma_b = np.zeros_like(numerator)

    valid = denominator > 1e-12

    sigma_b[valid] = (
        numerator[valid]
        / denominator[valid]
    )

    idx = np.argmax(sigma_b)

    return float(centers[idx])


# ============================================================
# FFT表示保存
# ============================================================

def save_fft_image(spectrum, save_path):
    display = np.log1p(spectrum)

    plt.figure(figsize=(8, 6))
    plt.imshow(display, cmap="gray")
    plt.title("Global 2D FFT")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


# ============================================================
# ヒストグラム
# ============================================================

def save_histogram(similarities, threshold, save_path):
    plt.figure(figsize=(8, 5))

    plt.hist(
        similarities,
        bins=40,
        edgecolor="black",
        alpha=0.8
    )

    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"Otsu = {threshold:.4f}"
    )

    plt.xlabel("Cosine similarity")
    plt.ylabel("Block count")
    plt.title("Cosine Similarity Histogram")

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


# ============================================================
# 傷BBox描画
# ============================================================

def draw_boxes(image, boxes):
    out = image.copy()

    for box in boxes:
        p1 = (
            int(round(box["x1"])),
            int(round(box["y1"]))
        )

        p2 = (
            int(round(box["x2"])),
            int(round(box["y2"]))
        )

        cv2.rectangle(
            out,
            p1,
            p2,
            (0, 255, 255),
            3
        )

        cv2.putText(
            out,
            box["label"],
            (p1[0], max(p1[1] - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    return out


# ============================================================
# 格子/非格子Overlay
# ============================================================

def make_grid_overlay(
    image,
    rows,
    threshold,
    boxes,
    alpha=0.35
):
    """
    高類似度 = 格子候補 = 緑
    低類似度 = 非格子候補 = 赤
    傷BBox = 黄
    """

    overlay = image.copy()

    for row in rows:

        x = row["x"]
        y = row["y"]
        w = row["width"]
        h = row["height"]
        sim = row["cosine_similarity"]

        if sim >= threshold:
            # 格子候補
            color = (0, 255, 0)
        else:
            # 非格子候補
            color = (0, 0, 255)

        cv2.rectangle(
            overlay,
            (x, y),
            (x + w, y + h),
            color,
            -1
        )

    result = cv2.addWeighted(
        overlay,
        alpha,
        image,
        1.0 - alpha,
        0
    )

    # ブロック境界 + similarity
    for row in rows:

        x = row["x"]
        y = row["y"]
        w = row["width"]
        h = row["height"]

        sim = row["cosine_similarity"]

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            1
        )

        cv2.putText(
            result,
            f"{sim:.3f}",
            (x + 4, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # 傷BBox
    result = draw_boxes(result, boxes)

    return result


# ============================================================
# 二値Grid Mask
# ============================================================

def make_grid_mask(image_shape, rows, threshold):
    h, w = image_shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)

    for row in rows:

        if row["cosine_similarity"] >= threshold:

            x = row["x"]
            y = row["y"]
            bw = row["width"]
            bh = row["height"]

            mask[
                y:y + bh,
                x:x + bw
            ] = 255

    return mask


# ============================================================
# 類似度マップ
# ============================================================

def save_similarity_map(
    rows,
    image_h,
    image_w,
    block_size,
    save_path
):

    rows_n = image_h // block_size
    cols_n = image_w // block_size

    sim_map = np.full(
        (rows_n, cols_n),
        np.nan,
        dtype=np.float32
    )

    for row in rows:

        r = row["y"] // block_size
        c = row["x"] // block_size

        sim_map[r, c] = row["cosine_similarity"]

    plt.figure(figsize=(10, 7))

    im = plt.imshow(
        sim_map,
        cmap="viridis",
        vmin=np.nanmin(sim_map),
        vmax=np.nanmax(sim_map)
    )

    plt.colorbar(
        im,
        label="Cosine similarity"
    )

    for r in range(rows_n):
        for c in range(cols_n):

            value = sim_map[r, c]

            if np.isfinite(value):
                plt.text(
                    c,
                    r,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6
                )

    plt.title(
        f"Cosine Similarity Map "
        f"({block_size}x{block_size})"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


# ============================================================
# CSV
# ============================================================

def save_csv(rows, threshold, save_path):

    fieldnames = [
        "image_name",
        "block_size",
        "block_id",
        "x",
        "y",
        "width",
        "height",
        "cosine_similarity",
        "similarity_minus_image_median",
        "similarity_percentile",
        "otsu_threshold",
        "grid_candidate",
        "region_class",
        "label",
    ]

    with open(
        save_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in rows:

            out = row.copy()

            is_grid = (
                row["cosine_similarity"]
                >= threshold
            )

            out["otsu_threshold"] = threshold
            out["grid_candidate"] = int(is_grid)

            out["region_class"] = (
                "grid_candidate"
                if is_grid
                else "non_grid_candidate"
            )

            writer.writerow(out)


# ============================================================
# Block Sizeごとの処理
# ============================================================

def process_block_size(
    image,
    gray,
    global_spectrum,
    boxes,
    block_size,
    output_dir
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    h, w = gray.shape

    # --------------------------------------------
    # Global FFTをblock FFTサイズに変換
    # --------------------------------------------

    global_ref = resize_global_spectrum(
        global_spectrum,
        block_size
    )

    if USE_LOG_FOR_SIMILARITY:
        global_ref = np.log1p(global_ref)

    global_ref = remove_dc(
        global_ref,
        DC_RADIUS
    )

    rows = []

    block_id = 0

    # --------------------------------------------
    # Block処理
    # --------------------------------------------

    for y in range(
        0,
        h - block_size + 1,
        block_size
    ):

        for x in range(
            0,
            w - block_size + 1,
            block_size
        ):

            block = gray[
                y:y + block_size,
                x:x + block_size
            ]

            block_spectrum = calc_fft_spectrum(
                block,
                USE_HANN
            )

            if USE_LOG_FOR_SIMILARITY:
                block_spectrum = np.log1p(
                    block_spectrum
                )

            block_spectrum = remove_dc(
                block_spectrum,
                DC_RADIUS
            )

            sim = cosine_similarity(
                global_ref,
                block_spectrum
            )

            label = block_label(
                x,
                y,
                block_size,
                block_size,
                boxes
            )

            rows.append(
                {
                    "image_name": IMAGE_PATH.name,
                    "block_size": block_size,
                    "block_id": block_id,
                    "x": x,
                    "y": y,
                    "width": block_size,
                    "height": block_size,
                    "cosine_similarity": sim,
                    "label": label,
                }
            )

            block_id += 1

    # --------------------------------------------
    # 画像内統計量
    # --------------------------------------------

    sims = np.array(
        [
            row["cosine_similarity"]
            for row in rows
        ]
    )

    median_sim = np.median(sims)

    for row in rows:

        s = row["cosine_similarity"]

        row[
            "similarity_minus_image_median"
        ] = s - median_sim

        # 値が低いものほど低パーセンタイル
        percentile = (
            np.sum(sims <= s)
            / len(sims)
            * 100.0
        )

        row[
            "similarity_percentile"
        ] = percentile

    # --------------------------------------------
    # Otsu
    # --------------------------------------------

    threshold = otsu_threshold_float(sims)

    print()
    print("=" * 60)
    print(
        f"Block size = {block_size}"
    )

    print(
        f"Otsu threshold = "
        f"{threshold:.6f}"
    )

    high_count = int(
        np.sum(sims >= threshold)
    )

    low_count = int(
        np.sum(sims < threshold)
    )

    print(
        f"grid candidate = {high_count}"
    )

    print(
        f"non-grid candidate = {low_count}"
    )

    # --------------------------------------------
    # Histogram
    # --------------------------------------------

    save_histogram(
        sims,
        threshold,
        output_dir
        / "similarity_histogram.png"
    )

    # --------------------------------------------
    # Similarity map
    # --------------------------------------------

    save_similarity_map(
        rows,
        h,
        w,
        block_size,
        output_dir
        / "similarity_map.png"
    )

    # --------------------------------------------
    # Grid mask
    # --------------------------------------------

    mask = make_grid_mask(
        image.shape,
        rows,
        threshold
    )

    cv2.imwrite(
        str(
            output_dir
            / "grid_mask.png"
        ),
        mask
    )

    # --------------------------------------------
    # Overlay
    # --------------------------------------------

    overlay = make_grid_overlay(
        image,
        rows,
        threshold,
        boxes,
        OVERLAY_ALPHA
    )

    cv2.imwrite(
        str(
            output_dir
            / "grid_overlay.png"
        ),
        overlay
    )

    # --------------------------------------------
    # CSV
    # --------------------------------------------

    save_csv(
        rows,
        threshold,
        output_dir
        / "blocks.csv"
    )

    # --------------------------------------------
    # summary
    # --------------------------------------------

    summary = {
        "image_name": IMAGE_PATH.name,
        "block_size": block_size,
        "num_blocks": len(rows),
        "otsu_threshold": threshold,
        "num_high_similarity": high_count,
        "num_low_similarity": low_count,
        "high_similarity_ratio":
            high_count / len(rows),
        "low_similarity_ratio":
            low_count / len(rows),
    }

    with open(
        output_dir / "summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2
        )

    return rows, threshold


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------
    # 画像読み込み
    # --------------------------------------------

    image = cv2.imread(
        str(IMAGE_PATH)
    )

    if image is None:
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    print(
        f"image size: "
        f"{gray.shape[1]} x {gray.shape[0]}"
    )

    # --------------------------------------------
    # LabelMe
    # --------------------------------------------

    boxes = load_labelme_boxes(
        LABEL_JSON_PATH
    )

    print(
        f"labels: {len(boxes)}"
    )

    for box in boxes:
        print(box)

    # 元画像保存
    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "original.png"
        ),
        image
    )

    # 傷BBox付き元画像
    labeled_image = draw_boxes(
        image,
        boxes
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "original_labels.png"
        ),
        labeled_image
    )

    # --------------------------------------------
    # Global FFT
    # --------------------------------------------

    global_spectrum = calc_fft_spectrum(
        gray,
        USE_HANN
    )

    global_for_display = remove_dc(
        global_spectrum,
        DC_RADIUS
    )

    save_fft_image(
        global_for_display,
        OUTPUT_DIR
        / "global_fft.png"
    )

    # --------------------------------------------
    # 256 / 128
    # --------------------------------------------

    for block_size in BLOCK_SIZES:

        block_output = (
            OUTPUT_DIR
            / str(block_size)
        )

        process_block_size(
            image=image,
            gray=gray,
            global_spectrum=global_spectrum,
            boxes=boxes,
            block_size=block_size,
            output_dir=block_output,
        )

    print()
    print("Finished.")
    print(
        f"Output: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()