"""Small, conservative PPTX text revisions without changing the source deck."""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path
from copy import deepcopy
import xml.etree.ElementTree as ET


P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS = {"p": P_NS, "a": A_NS}


SOURCE = Path(r"C:\Users\kyuta\Downloads\データサイエンス・人工知能技術-10-20進捗_実験結果追加版.pptx")
OUTPUT = Path(r"C:\work\Visual_inspection\outputs\presentations\データサイエンス・人工知能技術-10-20進捗_実験結果追加版_修正版.pptx")


def replace_once(data: bytes, old: str, new: str, slide: int) -> bytes:
    old_bytes, new_bytes = old.encode("utf-8"), new.encode("utf-8")
    count = data.count(old_bytes)
    # Slide 15 intentionally has two consecutive "未報告" cells; the caller
    # replaces them in table order.  All other replacements must be unique.
    if count == 0 or (count != 1 and not (slide == 15 and old == "未報告")):
        raise RuntimeError(f"slide {slide}: expected occurrence of {old!r}, found {count}")
    return data.replace(old_bytes, new_bytes, 1)


def edit_slide(data: bytes, slide: int, changes: list[tuple[str, str]]) -> bytes:
    for old, new in changes:
        data = replace_once(data, old, new, slide)
    return data


def _shape_id(shape: ET.Element) -> int:
    return int(shape.find("p:nvSpPr/p:cNvPr", NS).attrib["id"])


def _set_shape_text(shape: ET.Element, value: str) -> None:
    """Keep the source shape's font/style while replacing its visible text."""
    text_nodes = shape.findall(".//a:t", NS)
    if not text_nodes:
        raise RuntimeError("source text shape did not contain a:t")
    text_nodes[0].text = value
    for node in text_nodes[1:]:
        node.text = ""


def _set_shape_rect(shape: ET.Element, x: int, y: int, cx: int, cy: int) -> None:
    xfrm = shape.find("p:spPr/a:xfrm", NS)
    if xfrm is None:
        raise RuntimeError("source text shape did not contain transform")
    xfrm.find("a:off", NS).attrib.update({"x": str(x), "y": str(y)})
    xfrm.find("a:ext", NS).attrib.update({"cx": str(cx), "cy": str(cy)})


def replace_slide16_table(data: bytes) -> bytes:
    """Replace the former three-column list by the requested threshold matrix."""
    root = ET.fromstring(data)
    tree = root.find("p:cSld/p:spTree", NS)
    shapes = tree.findall("p:sp", NS)
    by_id = {_shape_id(shape): shape for shape in shapes}
    heading_template = by_id[9]
    header_template = by_id[10]
    cell_template = by_id[12]

    # Old right-side heading and four three-column rows.
    for shape in shapes:
        if _shape_id(shape) in {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23}:
            tree.remove(shape)

    next_id = max(_shape_id(shape) for shape in tree.findall("p:sp", NS)) + 1

    def add(template: ET.Element, text: str, x: int, y: int, cx: int, cy: int) -> None:
        nonlocal next_id
        shape = deepcopy(template)
        props = shape.find("p:nvSpPr/p:cNvPr", NS)
        props.attrib.update({"id": str(next_id), "name": f"ThresholdTable {next_id}"})
        next_id += 1
        _set_shape_text(shape, text)
        _set_shape_rect(shape, x, y, cx, cy)
        tree.append(shape)

    x0, y0, total_w = 6_760_000, 1_550_000, 4_400_000
    add(heading_template, "画像別：選択した閾値での候補数・検出数", x0, y0, total_w, 350_000)
    col_w = [1_050_000, 825_000, 825_000, 825_000, 875_000]
    headers = ["画像", "t=10", "t=15", "t=20", "t=30"]
    rows = [
        ["image1", "—", "—", "182件・1/1", "—"],
        ["image19", "—", "—", "—", "25件・1/1"],
        ["image20", "17件・5/5", "—", "—", "—"],
        ["image21", "—", "24件・2/2", "—", "—"],
    ]
    y_header, header_h, y_row0, row_h = 2_180_000, 315_000, 2_600_000, 490_000
    x = x0
    for label, width in zip(headers, col_w):
        add(header_template, label, x, y_header, width, header_h)
        x += width
    for index, row in enumerate(rows):
        x = x0
        for value, width in zip(row, col_w):
            add(cell_template, value, x, y_row0 + index * row_h, width, row_h)
            x += width

    note = by_id[24]
    _set_shape_text(note, "※ 表は各画像でFPR≤1%となるよう選んだ閾値のみ。件数＝全検出候補の連結成分数／右側＝点状異常の検出数。")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


CHANGES = {
    13: [
        ("点状異常検出｜周辺との輝度差に注目", "点状異常検出｜21×21／63×63リング平均との差分"),
        ("着想：点状異常は「局所的に暗い」。ただし正常な粒状模様も暗い点を作る。", "着想：点状異常は局所的に暗い。一方で正常な粒状模様も暗い点を作る。"),
        ("ぼかした画像を基準にして、暗い場所との差を求める（σ=15）", "D_G=max(Gaussian(gray, σ=15)−gray, 0)"),
        ("問題：正常な粒状模様にも大量に反応", "問題：粒状模様にも反応し、候補が増える"),
        ("改善案：中心と周辺リングの平均差", "改善案：中心と周辺リングの平均を比較"),
        ("周辺平均と中心平均の差で暗い点を強調", "D=max(μ_ring−μ_center, 0)"),
        ("面積ではなく局所的な明暗を検出する方法。今後の評価では異なる傷サイズも考慮", "各画素を中心に1画素ずつ走査し、D>tを候補とする。周辺リングは63×63から中心21×21を除いた領域。"),
    ],
    14: [
        ("ライトなし4画像／点状異常9件、画像ごとにFPR 1%以下の閾値を探索", "ライトなし4画像／点状異常9件。画像ごとにFPR 1%以下となる閾値を探索"),
        ("※ 全画像共通の固定閾値では同等の9/9検出・FPR≤1%は未達成。候補数はユーザー報告値", "※ 候補数は点状異常に重ならない連結成分数の平均。画像別の閾値選定による探索結果。固定閾値での9/9・FPR≤1%は未達成。"),
    ],
    15: [
        ("未報告", "90.8"),
        ("未報告", "62.5"),
        ("考察：中心を大きくしすぎると、小さな暗点の影響が平均化されて目立たなくなる", "考察：中心を31×31px以上へ大きくすると、小さな暗点の影響が平均化されて見逃しが生じる"),
    ],
    16: [
        ("※ 画像ごとに閾値tが異なる。赤点は検出候補であり、すべてが誤検出と確定したわけではない", "※ 画像ごとに閾値tが異なる。候補数は傷候補の全連結成分数であり、全てが誤検出と確定したものではない。"),
    ],
    17: [
        ("→ 似ている粒状模様でも、画素の場所はずれる", "→ 似ている粒状模様でも、画素位置は一致しない"),
        ("比較手法：最も一致画素数が多い隣接ブロック／上下左右4ブロックの画素ごとの中央値", "比較手法：一致画素数が最多の隣接ブロック／上下左右4ブロックの画素ごとの中央値。±2px位置合わせも比較したが改善しなかった。"),
    ],
    18: [
        ("・9/9件に対応（画像別閾値）", "・点状異常9/9件を検出（画像別閾値）"),
        ("次：image1の正常粒状模様と点状異常を比較し、固定閾値でも候補を絞れる特徴を探す", "次：image1の正常粒状模様と点状異常を比較し、固定閾値でも候補を絞れる追加特徴を探す"),
    ],
}


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        OUTPUT.unlink()
    with zipfile.ZipFile(SOURCE, "r") as src, zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            content = src.read(info.filename)
            if info.filename.startswith("ppt/slides/slide") and info.filename.endswith(".xml"):
                number = int(info.filename.removeprefix("ppt/slides/slide").removesuffix(".xml"))
                if number in CHANGES:
                    content = edit_slide(content, number, CHANGES[number])
                if number == 16:
                    content = replace_slide16_table(content)
            dst.writestr(info, content)
    with zipfile.ZipFile(OUTPUT) as check:
        for slide, replacements in CHANGES.items():
            data = check.read(f"ppt/slides/slide{slide}.xml")
            if slide == 16:
                for expected in ("画像別：選択した閾値での候補数・検出数", "182件・1/1", "25件・1/1", "17件・5/5", "24件・2/2"):
                    if expected.encode("utf-8") not in data:
                        raise RuntimeError(f"slide 16: revised table value missing: {expected}")
                continue
            for _old, new in replacements:
                if new.encode("utf-8") not in data:
                    raise RuntimeError(f"slide {slide}: revised text missing: {new}")
    print(OUTPUT)


if __name__ == "__main__":
    main()
