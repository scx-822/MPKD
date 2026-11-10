"""Hide the incompatible historical K=3 result until the fair rerun finishes."""

from __future__ import annotations

import copy
import shutil
import zipfile
from pathlib import Path

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}
SPACE = "http://www.w3.org/XML/1998/namespace"


def text_of(node: etree._Element) -> str:
    return "".join(x.text or "" for x in node.xpath(".//w:t", namespaces=NS))


def replace_paragraph(node: etree._Element, text: str) -> None:
    ppr = node.find("w:pPr", namespaces=NS)
    old_run = node.find("w:r", namespaces=NS)
    rpr = copy.deepcopy(old_run.find("w:rPr", namespaces=NS)) if old_run is not None and old_run.find("w:rPr", namespaces=NS) is not None else None
    for child in list(node):
        if child is not ppr:
            node.remove(child)
    run = etree.SubElement(node, f"{{{W}}}r")
    if rpr is not None:
        run.append(rpr)
    t = etree.SubElement(run, f"{{{W}}}t")
    t.set(f"{{{SPACE}}}space", "preserve")
    t.text = text


def set_cell(cell: etree._Element, value: str) -> None:
    paragraphs = cell.findall("w:p", namespaces=NS)
    if not paragraphs:
        p = etree.SubElement(cell, f"{{{W}}}p")
    else:
        p = paragraphs[0]
        for extra in paragraphs[1:]:
            cell.remove(extra)
    replace_paragraph(p, value)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("document", type=Path)
    path = parser.parse_args().document
    backup = path.with_name(path.stem + ".before_k_table_finalize.docx")
    shutil.copy2(path, backup)

    with zipfile.ZipFile(path, "r") as src:
        root = etree.fromstring(src.read("word/document.xml"))
        body = root.find("w:body", namespaces=NS)
        tables = body.findall("w:tbl", namespaces=NS)
        k_table = None
        for table in tables:
            rows = table.findall("w:tr", namespaces=NS)
            if not rows:
                continue
            first_row_cells = rows[0].findall("w:tc", namespaces=NS)
            if first_row_cells and text_of(first_row_cells[0]).strip() == "K":
                k_table = table
                break
        if k_table is None:
            raise RuntimeError("K sensitivity table not found")

        rows = k_table.findall("w:tr", namespaces=NS)
        k3_row = None
        for row in rows[1:]:
            cells = row.findall("w:tc", namespaces=NS)
            if cells and text_of(cells[0]).strip() == "3":
                k3_row = cells
                break
        if k3_row is None or len(k3_row) < 3:
            raise RuntimeError("K=3 row not found")
        set_cell(k3_row[1], "—")
        set_cell(k3_row[2], "—")

        paragraphs = root.xpath("/w:document/w:body/w:p", namespaces=NS)
        replacements = {
            "为评估每类子原型数量 K": (
                "为评估每类子原型数量 K 对 MPKD 性能的影响，本文在 CIFAR-100 的 ResNet32×4→ResNet8×4 设置下比较 K=1、2、3、4 和 5。K=1、2、4 和 5 的实验采用相同随机种子、训练轮数、学习率策略、门控和原型筛选阈值。此前记录的 K=3 结果使用了不同筛选阈值，故不与其余设置直接比较；同协议 K=3 复核正在进行，表10暂以“—”标记该项。",
            ),
            "当前同协议结果中，K=1": (
                "表10中已完成的同协议实验显示，K=1、2、4 和 5 的 Top-1 分别为 77.23%、77.05%、77.06% 和 77.05%，Top-5 分别为 94.41%、94.78%、94.55% 和 94.46%。其中 K=1 的 Top-1 最高，K=2 的 Top-5 最高；四组单次运行的差距较小，且各指标的最佳值可能对应不同 epoch。由于 K=3 的公平复核尚未完成，目前结果不足以确定最优原型数量，也不据此改变默认 K=3。",
            ),
        }
        for prefix, replacement in replacements.items():
            found = [p for p in paragraphs if text_of(p).startswith(prefix)]
            if len(found) != 1:
                raise RuntimeError(f"Expected one K ablation paragraph for {prefix!r}, got {len(found)}")
            replace_paragraph(found[0], replacement)

        output = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
        temp = path.with_suffix(".kfinal.docx")
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as dst:
            for item in src.infolist():
                data = output if item.filename == "word/document.xml" else src.read(item.filename)
                dst.writestr(item, data)
    shutil.move(str(temp), str(path))
    print(f"Updated {path}; backup={backup}")


if __name__ == "__main__":
    main()
