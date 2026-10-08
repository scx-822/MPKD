"""Finalize the MPKD Word draft without changing reported experiment values."""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path(r"C:\Users\scx\Desktop")
SOURCE = next(DESKTOP.glob("*/Our_Method_MPKD_V2_revised.docx"))
OUTPUT = SOURCE.with_name("Our_Method_MPKD_V2_final.docx")
ANALYSIS_DIR = ROOT / "experiments" / "analysis"

METHOD_REFS = {
    "KD": 1,
    "FitNets": 11,
    "AT": 12,
    "OFD": 13,
    "FCFD": 14,
    "RKD": 15,
    "CRD": 16,
    "SRRL": 17,
    "ICKD": 18,
    "PEFD": 19,
    "CAT-KD": 20,
    "TaT": 21,
    "ReviewKD": 22,
    "SimKD": 23,
    "DKD": 24,
    "NKD": 25,
    "JCS-MKD": 26,
    "CTKD": 27,
    "NormKD": 28,
    "LSKD": 29,
    "TopKD": 30,
    "TTM": 31,
    "BDD": 32,
    "TAKD": 33,
    "MLLD": 34,
    "CRLD": 3,
}

REFERENCES = [
    "[1] Hinton, G., Vinyals, O., and Dean, J. Distilling the Knowledge in a Neural Network. NIPS Deep Learning Workshop, 2015; arXiv:1503.02531.",
    "[2] Sohn, K., Berthelot, D., Li, C.-L., Zhang, Z., Carlini, N., Cubuk, E. D., Kurakin, A., Zhang, H., and Raffel, C. A. FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence. Advances in Neural Information Processing Systems, 33:596–608, 2020.",
    "[3] Zhang, W., Liu, D., Cai, W., and Ma, C. Cross-View Consistency Regularisation for Knowledge Distillation. Proceedings of the 32nd ACM International Conference on Multimedia, 2011–2020, 2024. https://doi.org/10.1145/3664647.3681206.",
    "[4] Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. On Calibration of Modern Neural Networks. Proceedings of the 34th International Conference on Machine Learning, PMLR 70:1321–1330, 2017.",
    "[5] Qian, Q., Shang, L., Sun, B., Hu, J., Li, H., and Jin, R. SoftTriple Loss: Deep Metric Learning Without Triplet Sampling. Proceedings of the IEEE/CVF International Conference on Computer Vision, 6450–6458, 2019.",
    "[6] Banerjee, A., Dhillon, I. S., Ghosh, J., and Sra, S. Clustering on the Unit Hypersphere Using von Mises–Fisher Distributions. Journal of Machine Learning Research, 6:1345–1382, 2005.",
    "[7] Snell, J., Swersky, K., and Zemel, R. Prototypical Networks for Few-Shot Learning. Advances in Neural Information Processing Systems, 30, 2017.",
    "[8] Zhang, P., Zhang, B., Zhang, T., Chen, D., Wang, Y., and Wen, F. Prototypical Pseudo Label Denoising and Target Structure Learning for Domain Adaptive Semantic Segmentation. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 12414–12424, 2021.",
    "[9] Gneiting, T., and Raftery, A. E. Strictly Proper Scoring Rules, Prediction, and Estimation. Journal of the American Statistical Association, 102(477):359–378, 2007.",
    "[10] Gal, Y., Islam, R., and Ghahramani, Z. Deep Bayesian Active Learning with Image Data. Proceedings of the 34th International Conference on Machine Learning, PMLR 70:1183–1192, 2017.",
    "[11] Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., and Bengio, Y. FitNets: Hints for Thin Deep Nets. International Conference on Learning Representations, 2015.",
    "[12] Zagoruyko, S., and Komodakis, N. Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer. International Conference on Learning Representations, 2017.",
    "[13] Heo, B., Kim, J., Yun, S., Park, H., Kwak, N., and Choi, J. Y. A Comprehensive Overhaul of Feature Distillation. Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019.",
    "[14] Liu, D., Kan, M., Shan, S., and Chen, X. Function-Consistent Feature Distillation. International Conference on Learning Representations, 2023.",
    "[15] Park, W., Kim, D., Lu, Y., and Cho, M. Relational Knowledge Distillation. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.",
    "[16] Tian, Y., Krishnan, D., and Isola, P. Contrastive Representation Distillation. International Conference on Learning Representations, 2020.",
    "[17] Yang, J., Martinez, B., Bulat, A., and Tzimiropoulos, G. Knowledge Distillation via Softmax Regression Representation Learning. International Conference on Learning Representations, 2021.",
    "[18] Liu, L., Huang, Q., Lin, S., Xie, H., Wang, B., Chang, X., and Liang, X. Exploring Inter-Channel Correlation for Diversity-Preserved Knowledge Distillation. Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021.",
    "[19] Chen, Y., Wang, S., Liu, J., Xu, X., de Hoog, F., and Huang, Z. Improved Feature Distillation via Projector Ensemble. Advances in Neural Information Processing Systems, 35:12084–12095, 2022.",
    "[20] Guo, Z., Yan, H., Li, H., and Lin, X. Class Attention Transfer Based Knowledge Distillation. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 11868–11877, 2023.",
    "[21] Lin, S., Xie, H., Wang, B., Yu, K., Chang, X., Liang, X., and Wang, G. Knowledge Distillation via the Target-Aware Transformer. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.",
    "[22] Chen, P., Liu, S., Zhao, H., and Jia, J. Distilling Knowledge via Knowledge Review. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021.",
    "[23] Chen, D., Mei, J.-P., Zhang, H., Wang, C., Feng, Y., and Chen, C. Knowledge Distillation with the Reused Teacher Classifier. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.",
    "[24] Zhao, B., Cui, Q., Song, R., Qiu, Y., and Liang, J. Decoupled Knowledge Distillation. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.",
    "[25] Yang, Z., Zeng, A., Li, Z., Zhang, T., Yuan, C., and Li, Y. From Knowledge Distillation to Self-Knowledge Distillation: A Unified Approach with Normalized Loss and Customized Soft Labels. Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.",
    "[26] Ding, Y., Yang, G., Ye, X., Wang, X., and Liu, Z. Joint Class Attention Knowledge and Self-Knowledge for Multi-Teacher Knowledge Distillation. Engineering Applications of Artificial Intelligence, 160:111922, 2025. https://doi.org/10.1016/j.engappai.2025.111922.",
    "[27] Li, Z., Li, X., Yang, L., Zhao, B., Song, R., Luo, L., Li, J., and Yang, J. Curriculum Temperature for Knowledge Distillation. Proceedings of the AAAI Conference on Artificial Intelligence, 2023.",
    "[28] Chi, Z., Zheng, T., Li, H., Yang, Z., Wu, B., Lin, B., and Cai, D. NormKD: Normalized Logits for Knowledge Distillation. arXiv:2308.00520, 2023.",
    "[29] Sun, S., Ren, W., Li, J., Wang, R., and Cao, X. Logit Standardization in Knowledge Distillation. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 15731–15740, 2024. https://doi.org/10.1109/CVPR52733.2024.01489.",
    "[30] Wang, Q., and Zhou, J. Top-KD: Top-Scaled Contrastive Knowledge Distillation. Neurocomputing, 685:133636, 2026. https://doi.org/10.1016/j.neucom.2026.133636.",
    "[31] Zheng, K., and Yang, E.-H. Knowledge Distillation Based on Transformed Teaching Matching. International Conference on Learning Representations, 2024.",
    "[32] Qi, Y., Wang, C., Zhang, Z., Liu, Y., and Zhang, Y. Balance Divergence for Knowledge Distillation. Engineering Applications of Artificial Intelligence, 167:113943, 2026. https://doi.org/10.1016/j.engappai.2026.113943.",
    "[33] Mirzadeh, S.-I., Farajtabar, M., Li, A., Levine, N., Matsukawa, A., and Ghasemzadeh, H. Improved Knowledge Distillation via Teacher Assistant. Proceedings of the AAAI Conference on Artificial Intelligence, 2020.",
    "[34] Jin, Y., Wang, J., and Lin, D. Multi-Level Logit Distillation. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.",
]


def replace_in_text_nodes(container, old: str, new: str) -> None:
    for node in container.iter(qn("w:t")):
        if node.text and old in node.text:
            node.text = node.text.replace(old, new)


def add_method_citations(paragraph) -> None:
    """Add one stable citation to method names in the experimental narrative."""
    nodes = list(paragraph._p.iter(qn("w:t")))
    for node in nodes:
        if not node.text:
            continue
        text = node.text
        for name, number in sorted(METHOD_REFS.items(), key=lambda item: -len(item[0])):
            pattern = rf"(?<![A-Za-z-]){re.escape(name)}(?![A-Za-z])(?!(?:\s*)\[)"
            text = re.sub(pattern, f"{name} [{number}]", text)
        node.text = text


def replace_table_citations(cell) -> None:
    text = cell.text
    updated = text
    for name, number in sorted(METHOD_REFS.items(), key=lambda item: -len(item[0])):
        pattern = rf"(?<![A-Za-z-]){re.escape(name)}\s*\[\d+\]"
        updated = re.sub(pattern, f"{name}[{number}]", updated)
    for name, number in sorted(METHOD_REFS.items(), key=lambda item: -len(item[0])):
        pattern = rf"(?<![A-Za-z-]){re.escape(name)}(?![A-Za-z])(?!(?:\s*)\[)"
        updated = re.sub(pattern, f"{name}[{number}]", updated)
    if updated == text:
        return
    cell.text = updated
    for paragraph in cell.paragraphs:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.first_line_indent = Pt(0)
        for run in paragraph.runs:
            run.font.name = "Times New Roman"
            run.font.size = Pt(8.5)
            run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")


def set_cell_margins(cell, top=0, start=80, bottom=0, end=80) -> None:
    properties = cell._tc.get_or_add_tcPr()
    margins = properties.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        properties.append(margins)
    for name, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = margins.find(qn(f"w:{name}"))
        if node is None:
            node = OxmlElement(f"w:{name}")
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def remove_table_borders(table) -> None:
    properties = table._tbl.tblPr
    borders = properties.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        properties.append(borders)
    for name in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = borders.find(qn(f"w:{name}"))
        if node is None:
            node = OxmlElement(f"w:{name}")
            borders.append(node)
        node.set(qn("w:val"), "nil")


def rebuild_algorithm(document: Document) -> None:
    outer = document.tables[1]
    cell = outer.cell(1, 0)
    old_inner = cell.tables[0]
    cell._tc.remove(old_inner._tbl)
    table = cell.add_table(rows=5, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    remove_table_borders(table)

    rows = [
        ("1. Freeze the teacher and mine reliable features from weak views.", "Eqs. (6)-(8)"),
        ("2. Cluster each class into K spherical K-Means prototypes and keep them fixed.", "Eqs. (9)-(10)"),
        ("3. For each batch, form both views and reconstruct the teacher strong-view target using prototype evidence.", "Eqs. (1)-(3), (11)-(21)"),
        ("4. Compute the two-view classification, within-view KD (WV-KD), cross-view KD (CV-KD), and total loss.", "Eqs. (4)-(5)"),
        ("5. Update only the student; repeat Steps 3-5 through training and return the optimized student.", "Eq. (5)"),
    ]
    for row, (step, formula) in zip(table.rows, rows):
        left, right = row.cells
        left.width = Cm(12.1)
        right.width = Cm(3.5)
        for column, cell_item in enumerate((left, right)):
            cell_item.width = Cm(12.1 if column == 0 else 3.5)
            cell_item.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell_item)
            paragraph = cell_item.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if column == 0 else WD_ALIGN_PARAGRAPH.RIGHT
            paragraph.paragraph_format.first_line_indent = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.0
            run = paragraph.add_run(step if column == 0 else formula)
            run.font.name = "Times New Roman"
            run.font.size = Pt(10.5)
            run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")


def replace_images(document: Document) -> None:
    image_paths = [
        ANALYSIS_DIR / "mpkd_prototype_feature_distribution.png",
        ANALYSIS_DIR / "mpkd_target_reconstruction_ablation.png",
    ]
    widths = (15.2, 15.2)
    for shape, image_path, width in zip(document.inline_shapes, image_paths, widths):
        blip = shape._inline.graphic.graphicData.pic.blipFill.blip
        relationship_id = blip.get(qn("r:embed"))
        image_part = document.part.related_parts[relationship_id]
        image_part._blob = image_path.read_bytes()
        with Image.open(image_path) as image:
            ratio = image.height / image.width
        shape.width = Cm(width)
        shape.height = Cm(width * ratio)


def add_references(document: Document) -> None:
    for paragraph in list(document.paragraphs):
        if re.match(r"^\[\d+\]\s", paragraph.text.strip()):
            paragraph._element.getparent().remove(paragraph._element)
    for reference in REFERENCES:
        paragraph = document.add_paragraph(reference, style="Body Text")
        paragraph.paragraph_format.first_line_indent = Pt(0)
        paragraph.paragraph_format.line_spacing = 1.0
        paragraph.paragraph_format.space_after = Pt(2)
        for run in paragraph.runs:
            run.font.name = "Times New Roman"
            run.font.size = Pt(9.5)
            run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")


def main() -> None:
    document = Document(SOURCE)
    replace_images(document)
    rebuild_algorithm(document)

    # The experimental narrative is the only prose block that needs the new method map.
    for paragraph in document.paragraphs[123:167]:
        add_method_citations(paragraph)
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                replace_table_citations(cell)

    add_references(document)
    document.save(OUTPUT)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
