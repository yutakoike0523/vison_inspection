import sys
import zipfile
import xml.etree.ElementTree as ET

path = sys.argv[1]
slides_to_inspect = [int(x) for x in sys.argv[2:]]
ns = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}

with zipfile.ZipFile(path) as archive:
    for slide_number in slides_to_inspect:
        root = ET.fromstring(archive.read(f"ppt/slides/slide{slide_number}.xml"))
        print(f"\nSLIDE {slide_number}")
        for shape in root.findall(".//p:sp", ns):
            props = shape.find("p:nvSpPr/p:cNvPr", ns)
            transform = shape.find("p:spPr/a:xfrm", ns)
            position = ""
            if transform is not None:
                position = f"{transform.find('a:off', ns).attrib} {transform.find('a:ext', ns).attrib}"
            text = " | ".join(
                "".join(run.text or "" for run in paragraph.findall(".//a:t", ns))
                for paragraph in shape.findall(".//a:p", ns)
            )
            if text:
                print(props.attrib.get("id"), props.attrib.get("name"), position, repr(text))
