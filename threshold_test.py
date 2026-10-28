from pathlib import Path
import json
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 設定
# ============================================================

DATA_DIR = Path(
    r"C:\work\Visual_inspection\data20260903\data20260903"
)

OUTPUT_DIR = Path(
    r"C:\work\Visual_inspection\outputs\fixed_threshold_comparison"
)

BLOCK_SIZE = 128

THRESHOLDS = [
    0.85,
    0.86,
    0.87,
    0.88,
    0.89,
]

USE_HANN = True
USE_LOG_FOR_SIMILARITY = False
DC_RADIUS = 5
OVERLAY_ALPHA = 0.35


# ============================================================
# LabelMe読み込み
# ============================================================

def load_labelme_boxes(json_path):

    if not json_path.exists():
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    boxes = []

    for shape in data.get("shapes", []):

        points = shape.get("points", [])

        if len(points) < 2:
            continue

        pts = np.asarray(points, dtype=np.float32)

        boxes.append({
            "label": shape.get("label", "scratch"),
            "x1": float(np.min(pts[:, 0])),
            "y1": float(np.min(pts[:, 1])),
            "x2": float(np.max(pts[:, 0])),
            "y2": float(np.max(pts[:, 1])),
        })

    return boxes


# ============================================================
# Blockが傷BBoxと重なるか
# ============================================================

def get_block_label(x, y, size, boxes):

    bx1 = x
    by1 = y
    bx2 = x + size
    by2 = y + size

    labels = []

    for box in boxes:

        ix1 = max(bx1, box["x1"])
        iy1 = max(by1, box["y1"])
        ix2 = min(bx2, box["x2"])
        iy2 = min(by2, box["y2"])

        if ix2 > ix1 and iy2 > iy1:
            labels.append(box["label"])

    if len(labels) == 0:
        return "unknown"

    return "+".join(sorted(set(labels)))


# ============================================================
# 2D Hann
# ============================================================

def hann2d(h, w):

    return np.outer(
        np.hanning(h),
        np.hanning(w)
    )


# ============================================================
# FFT
# ============================================================

def calc_fft_spectrum(gray):

    img = gray.astype(np.float32)

    # DCを抑える
    img = img - np.mean(img)

    if USE_HANN:
        img *= hann2d(*img.shape)

    f = np.fft.fft2(img)
    f = np.fft.fftshift(f)

    return np.abs(f).astype(np.float32)


# ============================================================
# DC除去
# ============================================================

def remove_dc(spectrum, radius=5):

    result = spectrum.copy()

    h, w = result.shape
    cy = h // 2
    cx = w // 2

    yy, xx = np.ogrid[:h, :w]

    mask = (
        (xx - cx) ** 2
        +
        (yy - cy) ** 2
        <= radius ** 2
    )

    result[mask] = 0

    return result


# ============================================================
# コサイン類似度
# ============================================================

def cosine_similarity(a, b):

    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)

    return float(
        np.dot(a, b)
        /
        (
            np.linalg.norm(a)
            *
            np.linalg.norm(b)
            +
            1e-12
        )
    )


# ============================================================
# 全体FFTを128×128へ変換
# ============================================================

def make_global_reference(global_spectrum, block_size):

    ref = cv2.resize(
        global_spectrum,
        (block_size, block_size),
        interpolation=cv2.INTER_AREA
    )

    if USE_LOG_FOR_SIMILARITY:
        ref = np.log1p(ref)

    ref = remove_dc(
        ref,
        DC_RADIUS
    )

    return ref


# ============================================================
# 傷BBoxを描画
# ============================================================

def draw_boxes(image, boxes):

    out = image.copy()

    for box in boxes:

        p1 = (
            int(box["x1"]),
            int(box["y1"])
        )

        p2 = (
            int(box["x2"]),
            int(box["y2"])
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
            (p1[0], max(15, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    return out


# ============================================================
# 1画像の類似度を計算
# ============================================================

def calculate_blocks(image_path, boxes):

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"Could not read {image_path}"
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    h, w = gray.shape

    # 全体FFT
    global_fft = calc_fft_spectrum(gray)

    global_ref = make_global_reference(
        global_fft,
        BLOCK_SIZE
    )

    rows = []

    block_id = 0

    for y in range(
        0,
        h - BLOCK_SIZE + 1,
        BLOCK_SIZE
    ):

        for x in range(
            0,
            w - BLOCK_SIZE + 1,
            BLOCK_SIZE
        ):

            block = gray[
                y:y + BLOCK_SIZE,
                x:x + BLOCK_SIZE
            ]

            block_fft = calc_fft_spectrum(
                block
            )

            if USE_LOG_FOR_SIMILARITY:
                block_fft = np.log1p(
                    block_fft
                )

            block_fft = remove_dc(
                block_fft,
                DC_RADIUS
            )

            sim = cosine_similarity(
                global_ref,
                block_fft
            )

            label = get_block_label(
                x,
                y,
                BLOCK_SIZE,
                boxes
            )

            rows.append({
                "block_id": block_id,
                "x": x,
                "y": y,
                "width": BLOCK_SIZE,
                "height": BLOCK_SIZE,
                "cosine_similarity": sim,
                "label": label,
            })

            block_id += 1

    return image, rows


# ============================================================
# Overlay
# ============================================================

def make_overlay(
    image,
    rows,
    threshold,
    boxes
):

    color_layer = image.copy()

    for row in rows:

        x = row["x"]
        y = row["y"]

        sim = row["cosine_similarity"]

        if sim >= threshold:

            # 格子候補 = 緑
            color = (0, 255, 0)

        else:

            # 非格子候補 = 赤
            color = (0, 0, 255)

        cv2.rectangle(
            color_layer,
            (x, y),
            (
                x + BLOCK_SIZE,
                y + BLOCK_SIZE
            ),
            color,
            -1
        )

    result = cv2.addWeighted(
        color_layer,
        OVERLAY_ALPHA,
        image,
        1.0 - OVERLAY_ALPHA,
        0
    )

    # block線
    for row in rows:

        x = row["x"]
        y = row["y"]

        cv2.rectangle(
            result,
            (x, y),
            (
                x + BLOCK_SIZE,
                y + BLOCK_SIZE
            ),
            (255, 255, 255),
            1
        )

        cv2.putText(
            result,
            f'{row["cosine_similarity"]:.3f}',
            (x + 3, y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1
        )

    result = draw_boxes(
        result,
        boxes
    )

    return result


# ============================================================
# 5閾値を1枚に並べる
# ============================================================

def save_comparison_figure(
    image_path,
    rows,
    boxes,
    save_path
):

    image = cv2.imread(
        str(image_path)
    )

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    fig, axes = plt.subplots(
        1,
        len(THRESHOLDS),
        figsize=(25, 7)
    )

    for ax, threshold in zip(
        axes,
        THRESHOLDS
    ):

        overlay = make_overlay(
            image,
            rows,
            threshold,
            boxes
        )

        overlay = cv2.cvtColor(
            overlay,
            cv2.COLOR_BGR2RGB
        )

        ax.imshow(overlay)

        ax.set_title(
            f"threshold = {threshold:.2f}"
        )

        ax.axis("off")

    plt.suptitle(
        image_path.name,
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=140
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # image1.png, image2.png ...
    image_paths = sorted(
        DATA_DIR.glob("image*.png"),
        key=lambda p: int(
            "".join(
                c for c in p.stem
                if c.isdigit()
            )
        )
    )

    all_summary = []

    print(
        f"Found {len(image_paths)} images"
    )

    for image_path in image_paths:

        print()
        print("=" * 60)
        print(image_path.name)

        json_path = (
            image_path.with_suffix(".json")
        )

        boxes = load_labelme_boxes(
            json_path
        )

        image, rows = calculate_blocks(
            image_path,
            boxes
        )

        image_output_dir = (
            OUTPUT_DIR
            / image_path.stem
        )

        image_output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------
        # 5閾値比較画像
        # ----------------------------

        save_comparison_figure(
            image_path,
            rows,
            boxes,
            image_output_dir
            / "threshold_comparison.png"
        )

        # ----------------------------
        # 各threshold個別出力
        # ----------------------------

        for threshold in THRESHOLDS:

            overlay = make_overlay(
                image,
                rows,
                threshold,
                boxes
            )

            cv2.imwrite(
                str(
                    image_output_dir
                    /
                    f"threshold_{threshold:.2f}.png"
                ),
                overlay
            )

            sims = np.array([
                row["cosine_similarity"]
                for row in rows
            ])

            grid_count = int(
                np.sum(
                    sims >= threshold
                )
            )

            nongrid_count = int(
                np.sum(
                    sims < threshold
                )
            )

            # ----------------------------
            # 傷ラベルを含むblock
            # ----------------------------

            defect_rows = [
                row
                for row in rows
                if row["label"] != "unknown"
            ]

            defect_total = len(
                defect_rows
            )

            defect_grid = sum(
                row[
                    "cosine_similarity"
                ] >= threshold
                for row in defect_rows
            )

            if defect_total > 0:

                defect_retention = (
                    defect_grid
                    /
                    defect_total
                )

            else:

                defect_retention = np.nan

            all_summary.append({

                "image_name":
                    image_path.name,

                "threshold":
                    threshold,

                "num_blocks":
                    len(rows),

                "grid_count":
                    grid_count,

                "nongrid_count":
                    nongrid_count,

                "grid_ratio":
                    grid_count
                    / len(rows),

                "defect_blocks":
                    defect_total,

                "defect_blocks_kept":
                    defect_grid,

                "defect_retention":
                    defect_retention,

                "similarity_min":
                    float(
                        np.min(sims)
                    ),

                "similarity_median":
                    float(
                        np.median(sims)
                    ),

                "similarity_max":
                    float(
                        np.max(sims)
                    ),
            })

            print(
                f"T={threshold:.2f}  "
                f"grid={grid_count}/{len(rows)}  "
                f"defect kept="
                f"{defect_grid}/{defect_total}"
            )

    # ========================================================
    # Summary CSV
    # ========================================================

    summary_path = (
        OUTPUT_DIR
        / "threshold_summary.csv"
    )

    fieldnames = [
        "image_name",
        "threshold",
        "num_blocks",
        "grid_count",
        "nongrid_count",
        "grid_ratio",
        "defect_blocks",
        "defect_blocks_kept",
        "defect_retention",
        "similarity_min",
        "similarity_median",
        "similarity_max",
    ]

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(
            all_summary
        )

    # ========================================================
    # thresholdごとの傷残存率
    # ========================================================

    thresholds = []
    retention_means = []

    for threshold in THRESHOLDS:

        values = [
            row["defect_retention"]
            for row in all_summary
            if (
                row["threshold"]
                == threshold
                and
                not np.isnan(
                    row["defect_retention"]
                )
            )
        ]

        if len(values) > 0:
            mean_retention = np.mean(
                values
            )
        else:
            mean_retention = np.nan

        thresholds.append(
            threshold
        )

        retention_means.append(
            mean_retention
        )

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        thresholds,
        retention_means,
        marker="o"
    )

    plt.ylim(
        0,
        1.05
    )

    plt.xlabel(
        "Cosine similarity threshold"
    )

    plt.ylabel(
        "Mean defect-block retention"
    )

    plt.title(
        "Defect retention vs threshold"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        /
        "defect_retention_vs_threshold.png",
        dpi=160
    )

    plt.close()

    print()
    print("Finished.")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()