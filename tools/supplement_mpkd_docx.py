"""Add the pending structure-ablation section to the current MPKD paper."""

from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = next(
    (Path.home() / "Desktop").glob("*/Our_Method_MPKD_V4.docx"),
    None,
)
DEFAULT_OUTPUT = ROOT / "experiments" / "Our_Method_MPKD_V4_supplemented.docx"


def insert_paragraph_before(target, text, style=None, bold=False):
    paragraph = target._parent.add_paragraph(text, style=style)
    target._p.addprevious(paragraph._p)
    for run in paragraph.runs:
        run.bold = bold
        run.font.name = "Times New Roman"
        run.font.size = Pt(10.5)
    return paragraph


def insert_table_before(target, headers, rows):
    document = target._parent._parent
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = str(value)
    for values in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, values):
            cell.text = str(value)

    target._p.addprevious(table._tbl)
    for row_index, row in enumerate(table.rows):
        for cell in row.cells:
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                paragraph.paragraph_format.first_line_indent = Pt(0)
                for run in paragraph.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(9)
                    run.bold = row_index == 0
    return table


def add_structure_section(document):
    target = next(
        paragraph
        for paragraph in document.paragraphs
        if paragraph.text.strip() == "5. Conclusion"
    )

    insert_paragraph_before(
        target,
        "4.4.4 原型中心数与聚合方式分析",
        style="Heading 2",
        bold=True,
    )
    insert_paragraph_before(
        target,
        "为分别考察原型数量和类别内相似度聚合方式的影响，本文在 CIFAR-100 的 ResNet32×4→ResNet8×4 设置下进行结构消融。除待考察变量外，其余训练配置、随机种子和可靠性门控保持不变。K=1 表示单中心原型，K≥2 表示多中心原型；因此，K 值实验同时覆盖了单中心与多中心原型的直接比较。表7中的 K=3 数值来自已有正式 MPKD 结果，其余数值待完成对应训练后补入。",
    )
    insert_paragraph_before(target, "Table 7: Ablation of the number of prototypes per class on CIFAR-100 (ResNet32×4→ResNet8×4).", bold=True)
    insert_table_before(
        target,
        ["K", "原型结构", "Top-1 (%)", "状态"],
        [
            ("1", "单中心原型", "待运行", "待补充"),
            ("2", "双中心原型", "待运行", "待补充"),
            ("3", "三中心原型", "77.72", "已有正式结果"),
            ("4", "四中心原型", "待运行", "待补充"),
            ("5", "五中心原型", "待运行", "待补充"),
        ],
    )
    insert_paragraph_before(
        target,
        "在此基础上，固定 K=3，进一步比较 Log-Mean-Exp、Mean 和 Max 三种类别内聚合方式。三种方式均作用于温度缩放后的原型相似度：Log-Mean-Exp 保留多个局部原型的平滑响应，Mean 使用类内平均相似度，Max 仅保留最匹配的局部原型。表8中的 Log-Mean-Exp 为当前正式 MPKD 结果，Mean 与 Max 的结果待运行后补充。",
    )
    insert_paragraph_before(target, "Table 8: Ablation of prototype aggregation with K=3 on CIFAR-100 (ResNet32×4→ResNet8×4).", bold=True)
    insert_table_before(
        target,
        ["聚合方式", "定义", "Top-1 (%)", "状态"],
        [
            ("Log-Mean-Exp", "log(mean(exp(s / τp)))", "77.72", "已有正式结果"),
            ("Mean", "mean(s / τp)", "待运行", "待补充"),
            ("Max", "max(s / τp)", "待运行", "待补充"),
        ],
    )
    insert_paragraph_before(
        target,
        "上述两组实验分别回答两个问题：多中心原型相对于单中心原型是否能够带来稳定收益，以及 Log-Mean-Exp 是否比简单平均或最大响应更适合将类内局部证据转换为类别分布。由于新增实验尚未完成，本文暂不对未运行配置作出性能判断。",
    )

    insert_paragraph_before(
        target,
        "4.4.5 跨教师–学生架构的稳定性分析",
        style="Heading 2",
        bold=True,
    )
    insert_paragraph_before(
        target,
        "表1和表2已经覆盖六组异构及六组同构教师–学生架构。异构设置中，MPKD 的平均 Top-1 准确率为 74.76%，高于 CRLD 的 74.10%，提升 0.66 个百分点；同构设置中，MPKD 的平均准确率为 75.76%，高于 CRLD 的 75.24%，提升 0.52 个百分点。逐组结果显示，MPKD 在当前十二组 CIFAR-100 架构组合上均高于对应的 CRLD 结果，说明原型目标重构并不依赖单一网络族，能够在同构和异构蒸馏场景中保持较稳定的收益。表5中的 KD、DKD 和 NKD 兼容性结果进一步表明，该结构证据主要作用于教师目标重构，而不是依赖某一种特定的基础 Logit 损失。",
    )


def main():
    parser = argparse.ArgumentParser(description="Supplement the MPKD V4 Word draft.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.source is None or not args.source.is_file():
        raise FileNotFoundError(
            "未找到 Our_Method_MPKD_V4.docx，请使用 --source 指定文件。"
        )

    document = Document(args.source)
    add_structure_section(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    document.save(args.output)
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
