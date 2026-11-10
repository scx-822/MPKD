"""Embed the generated robustness figure at the existing Figure 5 marker."""

from __future__ import annotations

import argparse
import copy
import shutil
import zipfile
from pathlib import Path

from lxml import etree
from PIL import Image


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"w": W, "r": R}


def patch(document: Path, figure: Path) -> None:
    backup = document.with_name(document.stem + ".before_figure5_embed.docx")
    shutil.copy2(document, backup)
    with zipfile.ZipFile(document, "r") as src:
        document_xml = etree.fromstring(src.read("word/document.xml"))
        rels_xml = etree.fromstring(src.read("word/_rels/document.xml.rels"))
        paragraphs = document_xml.xpath("//w:p", namespaces=NS)
        marker = None
        for paragraph in paragraphs:
            text = "".join(paragraph.itertext())
            if "[[FIGURE5]]" in text:
                marker = paragraph
                break
        if marker is None:
            raise RuntimeError("Figure 5 marker not found")

        rel_ids = [rel.get("Id") for rel in rels_xml]
        number = 15
        while f"rId{number}" in rel_ids:
            number += 1
        rel = etree.SubElement(rels_xml, f"{{{REL}}}Relationship")
        rel.set("Id", f"rId{number}")
        rel.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
        rel.set("Target", f"media/figure5_robustness.png")

        source_run = None
        for paragraph in paragraphs:
            drawings = paragraph.xpath(".//w:drawing", namespaces=NS)
            if drawings:
                source_run = paragraph.find("w:r", namespaces=NS)
                if source_run is not None:
                    break
        if source_run is None:
            raise RuntimeError("No existing drawing template found")
        run = copy.deepcopy(source_run)
        for blip in run.xpath(".//*[local-name()='blip']"):
            blip.set(f"{{{R}}}embed", f"rId{number}")

        width_px, height_px = Image.open(figure).size
        width_emu = 5_334_000
        height_emu = round(width_emu * height_px / width_px)
        for extent in run.xpath(".//*[local-name()='extent']"):
            extent.set("cx", str(width_emu))
            extent.set("cy", str(height_emu))
        for ext in run.xpath(".//*[local-name()='ext' and @cx and @cy]"):
            ext.set("cx", str(width_emu))
            ext.set("cy", str(height_emu))

        ppr = marker.find("w:pPr", namespaces=NS)
        for child in list(marker):
            if child is not ppr:
                marker.remove(child)
        marker.append(run)

        patched_document = etree.tostring(document_xml, xml_declaration=True, encoding="UTF-8", standalone=True)
        patched_rels = etree.tostring(rels_xml, xml_declaration=True, encoding="UTF-8", standalone=True)
        temp = document.with_suffix(".figure5patched.docx")
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as out:
            for item in src.infolist():
                if item.filename == "word/document.xml":
                    data = patched_document
                elif item.filename == "word/_rels/document.xml.rels":
                    data = patched_rels
                else:
                    data = src.read(item.filename)
                out.writestr(item, data)
            out.writestr("word/media/figure5_robustness.png", figure.read_bytes())
    shutil.move(str(temp), str(document))
    print(f"Embedded {figure} into {document}")
    print(f"Backup: {backup}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("document", type=Path)
    parser.add_argument("figure", type=Path)
    args = parser.parse_args()
    patch(args.document, args.figure)


if __name__ == "__main__":
    main()
