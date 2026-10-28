"""Append a quantitative background/surface validation slide to the revised deck.

The source deck is left untouched.  The new slide clones the existing summary
layout, so fonts, theme, and existing visual treatment are preserved.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs" / "presentations" / "データサイエンス・人工知能技術-10-20進捗_実験結果追加版_修正版_閾値表追加.pptx"
OUTPUT = ROOT / "outputs" / "presentations" / "データサイエンス・人工知能技術-10-20進捗_実験結果追加版_修正版_閾値表追加_背景判別評価追加.pptx"
P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
PR = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"p": P, "a": A, "r": R}


def set_shape_text(root: ET.Element, shape_id: int, text: str) -> None:
    shape = next(shape for shape in root.findall(".//p:sp", NS)
                 if shape.find("p:nvSpPr/p:cNvPr", NS).attrib.get("id") == str(shape_id))
    nodes = shape.findall(".//a:t", NS)
    if not nodes:
        raise RuntimeError(f"shape {shape_id} has no text node")
    nodes[0].text = text
    for node in nodes[1:]:
        node.text = ""


def edit_slide(xml: bytes) -> bytes:
    root = ET.fromstring(xml)
    # Layout cloned from slide 18: header, subtitle, metric cards, conclusion, footnote.
    replacements = {
        3: "追加評価｜背景・表面の自動判別",
        4: "輝度特徴量による定量評価｜全21画像・7×7ブロック",
        6: "19",
        8: "評価条件：純粋背景 background≥90% ／ 純粋表面 surface≥90% ／ tag・damage含有ブロックは除外",
        11: "全体の分類性能",
        12: "正解率 91.9%\n表面F1 95.4%\nmacro F1 81.3%",
        13: "背景除去の性能",
        14: "背景再現率 61.4%\n背景F1 67.1%\n背景32/83ブロックを表面と判定",
        16: "6特徴量で分離度を比較。エントロピー・IQRの差は大きいが、LOIOで最も安定したのは平均輝度。表面は保持しやすい一方、背景の約4割を見逃す。",
        17: "評価：テスト画像の閾値は残り20画像の純粋ブロックだけから決定。テスト画像の注釈で閾値は調整していない。ライトあり macro F1=81.5%、ライトなし=79.7%。",
    }
    for shape_id, text in replacements.items():
        set_shape_text(root, shape_id, text)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def next_rel_id(rels: ET.Element) -> str:
    numbers = [int(match.group(1)) for rel in rels.findall(f"{{{PR}}}Relationship")
               if (match := re.fullmatch(r"rId(\d+)", rel.attrib["Id"]))]
    return f"rId{max(numbers, default=0) + 1}"


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    with zipfile.ZipFile(SOURCE) as source:
        existing = {info.filename: source.read(info.filename) for info in source.infolist()}
    slide_numbers = [int(match.group(1)) for name in existing
                     if (match := re.fullmatch(r"ppt/slides/slide(\d+)\.xml", name))]
    base, new = max(slide_numbers), max(slide_numbers) + 1
    existing[f"ppt/slides/slide{new}.xml"] = edit_slide(existing[f"ppt/slides/slide{base}.xml"])
    existing[f"ppt/slides/_rels/slide{new}.xml.rels"] = existing[f"ppt/slides/_rels/slide{base}.xml.rels"]

    presentation = ET.fromstring(existing["ppt/presentation.xml"])
    slide_ids = presentation.find("p:sldIdLst", NS)
    highest_id = max(int(node.attrib["id"]) for node in slide_ids)
    rels = ET.fromstring(existing["ppt/_rels/presentation.xml.rels"])
    rel_id = next_rel_id(rels)
    slide_ids.append(ET.Element(f"{{{P}}}sldId", {"id": str(highest_id + 1), f"{{{R}}}id": rel_id}))
    rels.append(ET.Element(f"{{{PR}}}Relationship", {
        "Id": rel_id,
        "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide",
        "Target": f"slides/slide{new}.xml",
    }))
    existing["ppt/presentation.xml"] = ET.tostring(presentation, encoding="utf-8", xml_declaration=True)
    existing["ppt/_rels/presentation.xml.rels"] = ET.tostring(rels, encoding="utf-8", xml_declaration=True)

    content_types = ET.fromstring(existing["[Content_Types].xml"])
    content_types.append(ET.Element(f"{{{CT}}}Override", {
        "PartName": f"/ppt/slides/slide{new}.xml",
        "ContentType": "application/vnd.openxmlformats-officedocument.presentationml.slide+xml",
    }))
    existing["[Content_Types].xml"] = ET.tostring(content_types, encoding="utf-8", xml_declaration=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as target:
        for name, content in existing.items():
            target.writestr(name, content)
    with zipfile.ZipFile(OUTPUT) as check:
        if check.testzip() is not None:
            raise RuntimeError("invalid ZIP output")
        added = check.read(f"ppt/slides/slide{new}.xml").decode("utf-8")
        for expected in ("91.9%", "61.4%", "95.4%", "background"):
            if expected not in added:
                raise RuntimeError(f"missing expected slide text: {expected}")
    print(OUTPUT)


if __name__ == "__main__":
    main()
