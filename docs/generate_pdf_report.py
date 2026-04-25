from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_PATH = ROOT / "docs" / "Algoritmo_Genetico_no_Diagnostico_de_Cancer_de_Mama_em_Mulheres.md"
PDF_PATH = ROOT / "docs" / "Algoritmo_Genetico_no_Diagnostico_de_Cancer_de_Mama_em_Mulheres.pdf"


def _register_fonts() -> str:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    ]
    for font_path in candidates:
        if font_path.exists():
            pdfmetrics.registerFont(TTFont("RelatorioSans", str(font_path)))
            return "RelatorioSans"
    return "Helvetica"


def _build_styles(font_name: str) -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCustom",
            parent=styles["Title"],
            fontName=font_name,
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "Heading1Custom",
            parent=styles["Heading1"],
            fontName=font_name,
            fontSize=16,
            leading=20,
            spaceBefore=12,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "Heading2Custom",
            parent=styles["Heading2"],
            fontName=font_name,
            fontSize=13,
            leading=17,
            spaceBefore=10,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "BodyCustom",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=10.5,
            leading=14,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "BulletCustom",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=10.5,
            leading=14,
            leftIndent=12,
            spaceAfter=2,
        ),
    }


def _format_inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", escaped)

    def replace_link(match: re.Match[str]) -> str:
        label = match.group(1)
        url = match.group(2)
        return f"<a href='{html.escape(url, quote=True)}'>{html.escape(label)}</a>"

    escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace_link, escaped)
    return escaped


def _resolve_image_from_markdown(line: str) -> Path | None:
    match = re.fullmatch(r"- \[([^\]]+)\]\(([^)]+\.(?:png|jpg|jpeg))\)", line.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    target = match.group(2)
    if target.startswith("/"):
        return Path(target)
    return (ROOT / target).resolve()


def build_pdf() -> Path:
    font_name = _register_fonts()
    styles = _build_styles(font_name)

    story = []
    bullet_buffer: list[str] = []
    table_buffer: list[str] = []

    def flush_bullets() -> None:
        if not bullet_buffer:
            return
        flowable = ListFlowable(
            [ListItem(Paragraph(_format_inline(item), styles["bullet"])) for item in bullet_buffer],
            bulletType="bullet",
            start="circle",
            leftIndent=16,
        )
        story.append(flowable)
        story.append(Spacer(1, 0.15 * cm))
        bullet_buffer.clear()

    def flush_table() -> None:
        if not table_buffer:
            return
        rows: list[list[Paragraph]] = []
        for index, raw in enumerate(table_buffer):
            if index == 1:
                continue
            cells = [cell.strip() for cell in raw.strip().strip("|").split("|")]
            rows.append([Paragraph(_format_inline(cell), styles["body"]) for cell in cells])
        if rows:
            table = Table(rows, repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e6f2")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("BOX", (0, 0), (-1, -1), 0.75, colors.grey),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("LEADING", (0, 0), (-1, -1), 11),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 0.2 * cm))
        table_buffer.clear()

    lines = MARKDOWN_PATH.read_text(encoding="utf-8").splitlines()
    in_code_block = False
    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_table()
            in_code_block = not in_code_block
            continue
        if in_code_block:
            flush_bullets()
            flush_table()
            story.append(Paragraph(_format_inline(line or " "), styles["body"]))
            continue
        if not stripped:
            flush_bullets()
            flush_table()
            story.append(Spacer(1, 0.12 * cm))
            continue

        image_path = _resolve_image_from_markdown(stripped)
        if image_path is not None and image_path.exists():
            flush_bullets()
            flush_table()
            story.append(Paragraph(_format_inline(stripped), styles["body"]))
            story.append(Spacer(1, 0.1 * cm))
            image = Image(str(image_path))
            image._restrictSize(16 * cm, 10 * cm)
            story.append(image)
            story.append(Spacer(1, 0.3 * cm))
            continue

        if stripped.startswith("|"):
            flush_bullets()
            table_buffer.append(stripped)
            continue

        if stripped.startswith("- "):
            flush_table()
            bullet_buffer.append(stripped[2:].strip())
            continue

        flush_bullets()
        flush_table()
        if stripped.startswith("# "):
            story.append(Paragraph(_format_inline(stripped[2:].strip()), styles["title"]))
        elif stripped.startswith("## "):
            story.append(Paragraph(_format_inline(stripped[3:].strip()), styles["h1"]))
        elif stripped.startswith("### "):
            story.append(Paragraph(_format_inline(stripped[4:].strip()), styles["h2"]))
        else:
            story.append(Paragraph(_format_inline(stripped), styles["body"]))

    flush_bullets()
    flush_table()

    document = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="Algoritmo Genetico no Diagnostico de Cancer de Mama em Mulheres",
    )
    document.build(story)
    return PDF_PATH


if __name__ == "__main__":
    print(build_pdf())
