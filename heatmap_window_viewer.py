"""厳密8参照 FFT 異常度、格子周期、傷アノテーションの対話ビューア。"""
from pathlib import Path
import json
import re

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
from matplotlib.patches import Rectangle, Polygon

# Windows の日本語フォントを明示し、環境依存の文字化けを防ぐ。
plt.rcParams['font.family'] = 'Meiryo'
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data20260903' / 'data20260903'
PATCHED_DATA = ROOT / 'data_tag_patch_matched'
MAPS = ROOT / 'output_tag_patch_matched_fft' / 'maps'
WINDOW_SIZES = (32, 64, 128, 256, 512)


def shapes(image_id):
    return json.loads((DATA / f'image{image_id}.json').read_text(encoding='utf-8'))['shapes']


def display_image_path(image_id):
    """Show the tag-patch-repaired source when it exists."""
    patched = PATCHED_DATA / f'image{image_id}_tag_patch.png'
    return patched if patched.exists() else DATA / f'image{image_id}.png'


def grid_period(image_id):
    """格子アノテーションから横・縦の1周期[pixel]を推定する。"""
    points = {}
    for shape in shapes(image_id):
        match = re.match(r'grid\d+_r(\d+)-?_c(\d+)', shape['label'])
        if match and shape.get('points'):
            points[(int(match.group(1)), int(match.group(2)))] = np.mean(shape['points'], axis=0)
    horizontal = [np.linalg.norm(points[r, c + 1] - points[r, c]) for r, c in points if (r, c + 1) in points]
    vertical = [np.linalg.norm(points[r + 1, c] - points[r, c]) for r, c in points if (r + 1, c) in points]
    if not horizontal or not vertical:
        return np.nan, np.nan, np.nan
    px, py = np.median(horizontal), np.median(vertical)
    return px, py, np.sqrt(px * py)


def draw(image_id, window_size):
    ax_img.clear()
    ax_map.clear()
    ax_hist.clear()
    ax_img.imshow(Image.open(display_image_path(image_id)).convert('L'), cmap='gray')
    px, py, period = grid_period(image_id)
    map_path = MAPS / f'image{image_id}_w{window_size}.npz'
    if map_path.exists():
        with np.load(map_path) as data:
            score, xs, ys = data['anomaly_map'], data['x_centers'], data['y_centers']
        valid = score[np.isfinite(score)]
        ax_map.imshow(score, extent=[xs[0]-8, xs[-1]+8, ys[-1]+8, ys[0]-8], cmap='inferno',
                      vmin=np.percentile(valid, 1), vmax=np.percentile(valid, 99))
        # 同一の選択条件での異常度分布。外れ値も省かずに表示する。
        ax_hist.hist(valid, bins=60, color='darkorange', alpha=.82, edgecolor='none')
        median = np.median(valid)
        p95 = np.percentile(valid, 95)
        ax_hist.axvline(median, color='deepskyblue', linewidth=1.5, label=f'median {median:.3f} dB')
        ax_hist.axvline(p95, color='lime', linewidth=1.5, linestyle='--', label=f'p95 {p95:.3f} dB')
        ax_hist.set(title='異常度ヒストグラム（計算できた全地点）', xlabel='周囲ブロック平均との差の RMS 異常度 (dB)', ylabel='地点数')
        ax_hist.legend(fontsize=8, loc='upper right')
        status = f'有効地点: {len(valid):,} | RMS 中央値: {np.median(valid):.3f} dB | RMS 95%点: {np.percentile(valid, 95):.3f} dB'
        ax_map.set_title(f'異常度ヒートマップ: W={window_size}px | 札パッチ置換後・利用可能な周囲参照の平均 RMS | 黒=計算対象外')
    else:
        ax_map.set_facecolor('black')
        ax_hist.text(.5, .5, '評価可能な異常度はありません', ha='center', va='center', transform=ax_hist.transAxes)
        ax_hist.set_axis_off()
        status = '周囲の参照ブロックが入る地点がありません'
        ax_map.set_title(f'異常度ヒートマップ: W={window_size}px | 評価不可')
    ratio = window_size / period if np.isfinite(period) else np.nan
    info.set_text(
        f'image{image_id} | 格子1周期: {px:.1f} × {py:.1f} px（横 × 縦） | '
        f'FFT窓（バッチサイズ）: {window_size} × {window_size} px | '
        f'窓/格子比 W/p={ratio:.2f}\n{status}'
    )

    grid_points = []
    for shape in shapes(image_id):
        label, points = shape['label'], shape.get('points', [])
        if re.match(r'grid\d+_r\d+-?_c\d+', label):
            grid_points.append(np.mean(points, axis=0))
        elif label == 'tag' and shape.get('shape_type') == 'polygon':
            # The repaired source already removes this region; do not redraw its old outline.
            continue
        elif shape.get('shape_type') == 'rectangle' and points:
            (x0, y0), (x1, y1) = points
            for ax in (ax_img, ax_map):
                ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor='lime', linewidth=1.2))

    # Explain a lower-left FFT batch. Its score is drawn as a 16 x 16 px cell
    # centred on the batch centre, so it does not cover the grid-point overlay.
    batch_x = 0
    batch_y = 1536 - window_size
    batch_center_x = batch_x + window_size / 2
    batch_center_y = batch_y + window_size / 2
    display_cell = 16
    cell_x = batch_center_x - display_cell / 2
    cell_y = batch_center_y - display_cell / 2
    for ax in (ax_img, ax_map):
        ax.add_patch(Rectangle(
            (batch_x, batch_y), window_size, window_size, fill=False,
            edgecolor='deepskyblue', linewidth=2.0, linestyle='--', zorder=10,
        ))
        ax.scatter(
            [batch_center_x], [batch_center_y], marker='x', s=52,
            c='deepskyblue', linewidths=1.7, zorder=12,
        )
        ax.add_patch(Rectangle(
            (cell_x, cell_y), display_cell, display_cell,
            facecolor='none', edgecolor='magenta', linewidth=1.8, zorder=11,
        ))
    ax_map.text(
        .01, -.12,
        f'水色破線：左下FFTバッチ (0, {batch_y})〜{window_size}×{window_size}px　'
        f'×：中心　紫枠：中心16×16pxの表示セル',
        transform=ax_map.transAxes, ha='left', va='top', fontsize=7,
        color='deepskyblue', clip_on=False,
    )
    if grid_points:
        grid_points = np.asarray(grid_points)
        for ax in (ax_img, ax_map):
            ax.scatter(grid_points[:, 0], grid_points[:, 1], s=18, c='deepskyblue', edgecolors='black', linewidths=.4)
    for ax in (ax_img, ax_map):
        ax.set_xlim(0, 2048)
        ax.set_ylim(1536, 0)
        ax.set_xlabel('x (px)')
        ax.set_ylabel('y (px)')
    ax_img.set_title(f'image{image_id}: 元画像 | 水色=格子点・tag輪郭、黄緑=傷アノテーション')
    fig.canvas.draw_idle()


fig = plt.figure(figsize=(15, 10))
info = fig.text(.06, .955, '', fontsize=10, va='top')
ax_img = fig.add_axes([.06, .34, .38, .56])
ax_map = fig.add_axes([.50, .34, .38, .56])
ax_hist = fig.add_axes([.06, .07, .82, .17])
ax_image = fig.add_axes([.90, .55, .08, .35])
ax_window = fig.add_axes([.90, .18, .08, .27])
ax_image.set_title('画像選択')
ax_window.set_title('FFT窓\nサイズ W')
image_radio = RadioButtons(ax_image, [str(i) for i in range(1, 22)], active=1)
window_radio = RadioButtons(ax_window, [str(w) for w in WINDOW_SIZES], active=2)
state = {'image_id': 2, 'window_size': 128}


def change_image(label):
    state['image_id'] = int(label)
    draw(**state)


def change_window(label):
    state['window_size'] = int(label)
    draw(**state)


image_radio.on_clicked(change_image)
window_radio.on_clicked(change_window)
draw(**state)
plt.show()
