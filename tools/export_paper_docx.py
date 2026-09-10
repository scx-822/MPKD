"""Export the Chinese MPKD paper draft to a formatted Word document.

The exporter intentionally has no Pandoc or Microsoft Word runtime dependency.
It converts the limited LaTeX subset used by the draft to MathML, transforms
MathML to native Word OMML with Office's MML2OMML stylesheet, and writes the
result with python-docx.
"""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from lxml import etree


MATHML_NS = "http://www.w3.org/1998/Math/MathML"
MML = f"{{{MATHML_NS}}}"
OFFICE_MML2OMML = Path(
    r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"
)


def element(tag: str, text: str | None = None, **attributes):
    node = etree.Element(MML + tag)
    if text is not None:
        node.text = text
    for key, value in attributes.items():
        node.set(key.replace("_", "-"), value)
    return node


def row(*nodes):
    result = element("mrow")
    for node in nodes:
        if node is not None:
            result.append(node)
    return result


class LatexSubsetParser:
    """Parse the LaTeX subset present in MPKD论文草稿.md."""

    GREEK = {
        "alpha": "α",
        "beta": "β",
        "epsilon": "ε",
        "gamma": "γ",
        "lambda": "λ",
        "mu": "μ",
        "sigma": "σ",
        "tau": "τ",
        "theta": "θ",
    }
    SYMBOLS = {
        "cdot": "·",
        "geq": "≥",
        "in": "∈",
       "langle": "⟨",
       "ldots": "…",
        "leq": "≤",
        "lVert": "‖",
        "neq": "≠",
        "pm": "±",
        "rangle": "⟩",
        "rightarrow": "→",
        "rVert": "‖",
        "times": "×",
        "Vert": "‖",
        "%": "%",
   }
    FUNCTIONS = {"arg", "cos", "exp", "log", "max"}
    ACCENTS = {
        "bar": "¯",
        "widehat": "^",
        "widetilde": "~",
    }
    STYLES = {
        "mathbb": "double-struck",
        "mathcal": "script",
        "mathrm": "normal",
    }
    SPACING = {
        "!": "0em",
        ",": "0.17em",
        ";": "0.28em",
        "qquad": "2em",
    }

    def __init__(self, source: str):
        self.source = source.strip()
        self.position = 0
        self.unknown_commands: set[str] = set()

    def parse(self):
        parsed = self._parse_sequence()
        self._skip_space()
        if self.position != len(self.source):
            raise ValueError(
                f"Unparsed LaTeX at {self.position}: {self.source[self.position:]}"
            )
        return parsed

    def _skip_space(self):
        while self.position < len(self.source) and self.source[self.position].isspace():
            self.position += 1

    def _peek(self):
        self._skip_space()
        if self.position >= len(self.source):
            return ""
        return self.source[self.position]

    def _parse_sequence(self, terminator: str | None = None):
        nodes = []
        while True:
            self._skip_space()
            if self.position >= len(self.source):
                if terminator is not None:
                    raise ValueError(f"Missing closing delimiter {terminator!r}")
                break
            if terminator is not None and self.source[self.position] == terminator:
                self.position += 1
                break

            atom = self._parse_atom()
            if atom is None:
                continue
            atom = self._attach_scripts(atom)
            nodes.append(atom)
        return row(*nodes)

    def _attach_scripts(self, base):
        subscript = None
        superscript = None
        while True:
            next_char = self._peek()
            if next_char not in {"_", "^"}:
                break
            self.position += 1
            script = self._parse_argument()
            if next_char == "_":
                subscript = script
            else:
                superscript = script

        if subscript is not None and superscript is not None:
            result = element("msubsup")
            result.extend([base, subscript, superscript])
            return result
        if subscript is not None:
            result = element("msub")
            result.extend([base, subscript])
            return result
        if superscript is not None:
            result = element("msup")
            result.extend([base, superscript])
            return result
        return base

    def _parse_argument(self):
        self._skip_space()
        if self.position >= len(self.source):
            raise ValueError("Expected a LaTeX argument")
        if self.source[self.position] == "{":
            self.position += 1
            return self._parse_sequence("}")
        atom = self._parse_atom()
        if atom is None:
            raise ValueError("Expected a non-empty LaTeX argument")
        return self._attach_scripts(atom)

    def _read_command(self):
        self.position += 1
        if self.position >= len(self.source):
            return ""
        if self.source[self.position].isalpha():
            start = self.position
            while (
                self.position < len(self.source)
                and self.source[self.position].isalpha()
            ):
                self.position += 1
            return self.source[start:self.position]
        command = self.source[self.position]
        self.position += 1
        return command

    def _read_raw_group(self):
        self._skip_space()
        if self.position >= len(self.source) or self.source[self.position] != "{":
            raise ValueError("Expected a braced argument")
        self.position += 1
        start = self.position
        depth = 1
        while self.position < len(self.source):
            char = self.source[self.position]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    value = self.source[start:self.position]
                    self.position += 1
                    return value
            self.position += 1
        raise ValueError("Unterminated braced argument")

    def _parse_atom(self):
        self._skip_space()
        if self.position >= len(self.source):
            return None

        char = self.source[self.position]
        if char == "\\":
            command = self._read_command()
            return self._parse_command(command)
        if char == "{":
            self.position += 1
            return self._parse_sequence("}")
        if char.isdigit():
            start = self.position
            while self.position < len(self.source):
                current = self.source[self.position]
                if current.isdigit() or current == ".":
                    self.position += 1
                else:
                    break
            return element("mn", self.source[start:self.position])
        if char.isalpha():
            self.position += 1
            return element("mi", char)

        self.position += 1
        replacements = {
            "-": "−",
            "*": "×",
            "(": "(",
            ")": ")",
            "[": "[",
            "]": "]",
            "|": "|",
            "+": "+",
            "=": "=",
            ">": ">",
            "<": "<",
            ",": ",",
            ".": ".",
            ":": ":",
            "/": "/",
        }
        return element("mo", replacements.get(char, char))

    def _parse_command(self, command: str):
        if command in {"left", "right", "middle"}:
            return self._parse_atom()
        if command in self.SPACING:
            return element("mspace", width=self.SPACING[command])
        if command == "frac":
            numerator = self._parse_argument()
            denominator = self._parse_argument()
            fraction = element("mfrac")
            fraction.extend([numerator, denominator])
            return fraction
        if command in self.ACCENTS:
            base = self._parse_argument()
            accent = element("mover", accent="true")
            accent.extend([base, element("mo", self.ACCENTS[command])])
            return accent
        if command in self.STYLES:
            styled = self._parse_argument()
            self._apply_variant(styled, self.STYLES[command])
            return styled
        if command == "operatorname":
            value = self._read_raw_group()
            return element("mi", value, mathvariant="normal")
        if command == "sum":
            return element("mo", "∑", largeop="true", movablelimits="true")
        if command in self.FUNCTIONS:
            return element("mi", command, mathvariant="normal")
        if command in self.GREEK:
            return element("mi", self.GREEK[command])
        if command in self.SYMBOLS:
            return element("mo", self.SYMBOLS[command])
        if command == "{":
            return element("mo", "{")
        if command == "}":
            return element("mo", "}")
        if command == "|":
            return element("mo", "‖")

        self.unknown_commands.add(command)
        return element("mi", command, mathvariant="normal")

    @staticmethod
    def _apply_variant(node, variant: str):
        if etree.QName(node).localname == "mi":
            node.set("mathvariant", variant)
        for child in node:
            LatexSubsetParser._apply_variant(child, variant)


class OmmlConverter:
    def __init__(self, stylesheet_path: Path):
        if not stylesheet_path.exists():
            raise FileNotFoundError(
                "Office MathML-to-OMML stylesheet was not found: "
                f"{stylesheet_path}"
            )
        self.transform = etree.XSLT(etree.parse(str(stylesheet_path)))
        self.unknown_commands: set[str] = set()

    def convert(self, latex: str, display: bool = False):
        latex = re.sub(r"\\tag\{[^{}]+\}", "", latex).strip()
        parser = LatexSubsetParser(latex)
        parsed = parser.parse()
        self.unknown_commands.update(parser.unknown_commands)

        math = element("math", display="block" if display else "inline")
        math.append(parsed)
        result = self.transform(math)
        if result.getroot() is None:
            raise ValueError(f"MML2OMML returned no root for: {latex}")
        return copy.deepcopy(result.getroot())


def set_east_asia_font(run, east_asia: str, latin: str = "Times New Roman"):
    run.font.name = latin
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east_asia)


def set_cell_shading(cell, fill: str):
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=70, start=80, bottom=70, end=80):
    properties = cell._tc.get_or_add_tcPr()
    margins = properties.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        properties.append(margins)
    for margin_name, margin_value in (
        ("top", top),
        ("start", start),
        ("bottom", bottom),
        ("end", end),
    ):
        margin = margins.find(qn(f"w:{margin_name}"))
        if margin is None:
            margin = OxmlElement(f"w:{margin_name}")
            margins.append(margin)
        margin.set(qn("w:w"), str(margin_value))
        margin.set(qn("w:type"), "dxa")


def remove_table_borders(table):
    properties = table._tbl.tblPr
    borders = properties.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        properties.append(borders)
    for name in ("top", "left", "bottom", "right", "insideH", "insideV"):
        edge = borders.find(qn(f"w:{name}"))
        if edge is None:
            edge = OxmlElement(f"w:{name}")
            borders.append(edge)
        edge.set(qn("w:val"), "nil")


def repeat_table_header(row):
    row_properties = row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    row_properties.append(repeat)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for item in (begin, instruction, separate, text, end):
        run._r.append(item)


def configure_document(document: Document):
    section = document.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.8)
    section.right_margin = Cm(2.5)
    section.header_distance = Cm(1.2)
    section.footer_distance = Cm(1.2)
    add_page_number(section.footer.paragraphs[0])

    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.first_line_indent = Pt(21)

    title = document.styles["Title"]
    title.font.name = "Times New Roman"
    title.font.size = Pt(18)
    title.font.bold = True
    title._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(18)

    heading_specs = {
        "Heading 1": (16, "黑体", 12, 6),
        "Heading 2": (14, "黑体", 10, 4),
        "Heading 3": (12, "黑体", 8, 3),
        "Heading 4": (11, "黑体", 6, 3),
    }
    for style_name, (size, east_asia, before, after) in heading_specs.items():
        style = document.styles[style_name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), east_asia)
        style.paragraph_format.first_line_indent = Pt(0)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True


def append_inline_content(paragraph, text: str, converter: OmmlConverter, size=10.5):
    token_pattern = re.compile(r"(\*\*.*?\*\*|\\\(.*?\\\))")
    cursor = 0
    for match in token_pattern.finditer(text):
        if match.start() > cursor:
            run = paragraph.add_run(text[cursor:match.start()])
            run.font.size = Pt(size)
            set_east_asia_font(run, "宋体")
        token = match.group(0)
        if token.startswith("**"):
            run = paragraph.add_run(token[2:-2])
            run.bold = True
            run.font.size = Pt(size)
            set_east_asia_font(run, "黑体")
        else:
            latex = token[2:-2]
            paragraph._p.append(converter.convert(latex, display=False))
        cursor = match.end()
    if cursor < len(text):
        run = paragraph.add_run(text[cursor:])
        run.font.size = Pt(size)
        set_east_asia_font(run, "宋体")


def add_body_paragraph(document, text, converter, style=None, indent=True):
    paragraph = document.add_paragraph(style=style)
    if not indent:
        paragraph.paragraph_format.first_line_indent = Pt(0)
    append_inline_content(paragraph, text, converter)
    return paragraph


def add_display_equation(document, latex: str, converter: OmmlConverter):
    tag_match = re.search(r"\\tag\{([^{}]+)\}", latex)
    tag = tag_match.group(1) if tag_match else None
    clean_latex = re.sub(r"\\tag\{[^{}]+\}", "", latex).strip()

    if tag is None:
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.first_line_indent = Pt(0)
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph._p.append(converter.convert(clean_latex, display=True))
        return

    table = document.add_table(rows=1, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    remove_table_borders(table)
    widths = (Cm(1.2), Cm(13.7), Cm(1.2))
    for cell, width in zip(table.rows[0].cells, widths):
        cell.width = width
        set_cell_margins(cell, top=0, start=0, bottom=0, end=0)

    equation_paragraph = table.cell(0, 1).paragraphs[0]
    equation_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    equation_paragraph.paragraph_format.first_line_indent = Pt(0)
    equation_paragraph._p.append(converter.convert(clean_latex, display=True))

    number_paragraph = table.cell(0, 2).paragraphs[0]
    number_paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    number_paragraph.paragraph_format.first_line_indent = Pt(0)
    number_run = number_paragraph.add_run(f"({tag})")
    number_run.font.size = Pt(10.5)
    set_east_asia_font(number_run, "宋体")


def add_markdown_image(document, source_path: Path, line: str):
    """Embed a local Markdown image at a stable paper-width size."""
    match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
    if match is None:
        raise ValueError(f"Unsupported Markdown image syntax: {line}")

    image_path = (source_path.parent / match.group(2).strip()).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Markdown image referenced by {source_path.name} was not found: "
            f"{image_path}"
        )

    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.first_line_indent = Pt(0)
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(6)
    run = paragraph.add_run()
    run.add_picture(str(image_path), width=Cm(15.0))


def parse_table_cells(line: str):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_table_separator(line: str):
    cells = parse_table_cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def add_markdown_table(document, lines, converter: OmmlConverter):
    rows = [parse_table_cells(line) for line in lines]
    if len(rows) >= 2 and is_table_separator(lines[1]):
        rows.pop(1)

    is_algorithm_table = (
        bool(rows)
        and len(rows[0]) >= 2
        and rows[0][0] == "算法步骤"
        and rows[0][1] == "公式编号"
    )
    if is_algorithm_table:
        rows.pop(0)
        table = document.add_table(rows=len(rows), cols=2)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        remove_table_borders(table)
        widths = (Cm(14.0), Cm(1.6))
        for cell, width in zip(table.rows[0].cells, widths):
            cell.width = width
            set_cell_margins(cell, top=0, start=0, bottom=0, end=0)

        for row_index, row_data in enumerate(rows):
            for column_index in range(2):
                cell = table.cell(row_index, column_index)
                cell.width = widths[column_index]
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                set_cell_margins(cell, top=0, start=0, bottom=0, end=0)
                paragraph = cell.paragraphs[0]
                paragraph.alignment = (
                    WD_ALIGN_PARAGRAPH.LEFT
                    if column_index == 0
                    else WD_ALIGN_PARAGRAPH.RIGHT
                )
                paragraph.paragraph_format.first_line_indent = Pt(0)
                paragraph.paragraph_format.line_spacing = 1.0
                value = row_data[column_index] if column_index < len(row_data) else ""
                append_inline_content(paragraph, value, converter, size=10.5)
        document.add_paragraph().paragraph_format.space_after = Pt(0)
        return

    column_count = max(len(row_data) for row_data in rows)
    table = document.add_table(rows=len(rows), cols=column_count)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    repeat_table_header(table.rows[0])

    for row_index, row_data in enumerate(rows):
        for column_index in range(column_count):
            cell = table.cell(row_index, column_index)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.first_line_indent = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.0
            value = row_data[column_index] if column_index < len(row_data) else ""
            append_inline_content(paragraph, value, converter, size=8.5)
            if row_index == 0:
                set_cell_shading(cell, "D9E2F3")
                for run in paragraph.runs:
                    run.bold = True
                    set_east_asia_font(run, "黑体")
    document.add_paragraph().paragraph_format.space_after = Pt(0)

def export_markdown(source_path: Path, output_path: Path):
    converter = OmmlConverter(OFFICE_MML2OMML)
    document = Document()
    configure_document(document)
    lines = source_path.read_text(encoding="utf-8").splitlines()

    index = 0
    while index < len(lines):
        raw_line = lines[index]
        line = raw_line.strip()

        if not line:
            index += 1
            continue

        if line == "$$":
            math_lines = []
            index += 1
            while index < len(lines) and lines[index].strip() != "$$":
                math_lines.append(lines[index].strip())
                index += 1
            if index >= len(lines):
                raise ValueError("Unterminated display-math block")
            add_display_equation(document, " ".join(math_lines), converter)
            index += 1
            continue

        if line.startswith("|") and index + 1 < len(lines):
            if lines[index + 1].strip().startswith("|") and is_table_separator(
                lines[index + 1]
            ):
                table_lines = []
                while index < len(lines) and lines[index].strip().startswith("|"):
                    table_lines.append(lines[index].strip())
                    index += 1
                add_markdown_table(document, table_lines, converter)
                continue

        if re.fullmatch(r"!\[[^\]]*\]\([^)]+\)", line):
            add_markdown_image(document, source_path, line)
            index += 1
            continue

        heading_match = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2)
            if level == 1:
                paragraph = document.add_paragraph(style="Title")
                append_inline_content(paragraph, heading_text, converter, size=18)
            else:
                paragraph = document.add_paragraph(
                    style=f"Heading {min(level - 1, 4)}"
                )
                paragraph.paragraph_format.first_line_indent = Pt(0)
                append_inline_content(
                    paragraph,
                    heading_text,
                    converter,
                    size={2: 16, 3: 14, 4: 12}.get(level, 11),
                )
            index += 1
            continue

        if line.startswith(">"):
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.first_line_indent = Pt(0)
            paragraph.paragraph_format.left_indent = Cm(0.6)
            paragraph.paragraph_format.right_indent = Cm(0.6)
            paragraph.paragraph_format.space_after = Pt(6)
            append_inline_content(
                paragraph, line.lstrip("> ").strip(), converter, size=9
            )
            for run in paragraph.runs:
                run.font.color.rgb = RGBColor(89, 89, 89)
            index += 1
            continue

        bullet_match = re.match(r"^-\s+(.+)$", line)
        numbered_match = re.match(r"^\d+\.\s+(.+)$", line)
        if bullet_match or numbered_match:
            content = (bullet_match or numbered_match).group(1)
            style = "List Bullet" if bullet_match else "List Number"
            paragraph = add_body_paragraph(
                document, content, converter, style=style, indent=False
            )
            paragraph.paragraph_format.left_indent = Cm(0.75)
            paragraph.paragraph_format.first_line_indent = Cm(-0.35)
            index += 1
            continue

        paragraph = add_body_paragraph(document, line, converter)
        if line.startswith("**表 ") or line.startswith("**关键词**"):
            paragraph.paragraph_format.first_line_indent = Pt(0)
            if line.startswith("**表 "):
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                paragraph.paragraph_format.keep_with_next = True
        index += 1

    if converter.unknown_commands:
        unknown = ", ".join(sorted(converter.unknown_commands))
        raise ValueError(f"Unsupported LaTeX commands: {unknown}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(output_path)
    return {
        "paragraphs": len(document.paragraphs),
        "tables": len(document.tables),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Export MPKD Markdown draft to native-Word DOCX."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "experiments" / "MPKD论文草稿.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "experiments" / "MPKD论文草稿.docx",
    )
    arguments = parser.parse_args()

    stats = export_markdown(arguments.source.resolve(), arguments.output.resolve())
    print(
        f"Generated {arguments.output.resolve()} "
        f"({stats['paragraphs']} paragraphs, {stats['tables']} tables)."
    )


if __name__ == "__main__":
    main()
