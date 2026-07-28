"""Routage automatique entre PyMuPDF4LLM et Docling."""

import re
from pathlib import Path
from typing import Any, cast

import pymupdf4llm
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from phosprocess.ingestion.schemas import (
    PageContent,
    PageProvenance,
    PageQuality,
    ParsedPage,
)

_COMPLEX_LAYOUT_CLASSES = {"picture", "table", "formula"}


def _markdown_to_plain_text(markdown: str) -> str:
    """Retirer les principaux marqueurs Markdown."""

    text = re.sub(r"!\[[^\]]*]\([^)]*\)", "", markdown)
    text = re.sub(r"\[([^]]+)]\([^)]*\)", r"\1", text)
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")

    return text.strip()


def _requires_docling(page_chunk: dict[str, Any]) -> tuple[bool, list[str]]:
    """Déterminer automatiquement si la page nécessite Docling."""

    markdown = str(page_chunk.get("text", "")).strip()
    boxes = page_chunk.get("page_boxes", [])

    box_classes = [
        str(box.get("class", ""))
        for box in boxes
        if isinstance(box, dict)
    ]

    complex_elements = sum(
        item in _COMPLEX_LAYOUT_CLASSES for item in box_classes
    )

    short_numeric_lines = sum(
        1
        for line in markdown.splitlines()
        if len(line.split()) <= 3 and re.search(r"\d", line)
    )

    warnings: list[str] = []

    if len(markdown) < 100:
        warnings.append("insufficient_text")

    if complex_elements >= 2:
        warnings.append("complex_layout")

    if short_numeric_lines >= 8:
        warnings.append("possible_broken_table_or_figure")

    return bool(warnings), warnings


def _build_docling_converter() -> DocumentConverter:
    """Configurer Docling avec OCR et extraction des tableaux."""

    options = PdfPipelineOptions()
    options.do_ocr = True
    options.do_table_structure = True

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=options,
            )
        },
    )


def _parse_page_with_docling(
    pdf_path: Path,
    page_number: int,
    converter: DocumentConverter,
    warnings: list[str],
) -> ParsedPage:
    """Retraiter une page complexe avec Docling."""

    result = converter.convert(
        pdf_path,
        page_range=(page_number, page_number),
    )

    document = result.document
    markdown = document.export_to_markdown()
    plain_text = document.export_to_text()

    tables = [
        table.export_to_markdown(doc=document)
        for table in document.tables
    ]

    figures = [
        f"picture_{index}"
        for index, _ in enumerate(document.pictures, start=1)
    ]

    return ParsedPage(
        content=PageContent(
            plain_text=plain_text,
            markdown=markdown,
            tables=tables,
            figures=figures,
        ),
        provenance=PageProvenance(
            source_file=pdf_path.name,
            document_id=pdf_path.stem,
            page_number=page_number,
            parser="docling",
            ocr_used=True,
        ),
        quality=PageQuality(
            character_count=len(plain_text),
            word_count=len(plain_text.split()),
            is_empty=not plain_text.strip(),
            needs_review=not plain_text.strip(),
            warnings=warnings,
        ),
    )


def parse_pdf_automatically(pdf_path: Path) -> list[ParsedPage]:
    """Analyser toutes les pages et choisir automatiquement le parseur."""

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

    raw_chunks = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
        use_ocr=True,
        header=False,
        footer=False,
        show_progress=True,
    )

    page_chunks = cast(list[dict[str, Any]], raw_chunks)

    parsed_pages: list[ParsedPage] = []
    docling_converter: DocumentConverter | None = None

    for chunk in page_chunks:
        metadata = chunk.get("metadata", {})
        page_number = int(metadata["page_number"])

        requires_docling, warnings = _requires_docling(chunk)

        if requires_docling:
            if docling_converter is None:
                docling_converter = _build_docling_converter()

            parsed_pages.append(
                _parse_page_with_docling(
                    pdf_path=pdf_path,
                    page_number=page_number,
                    converter=docling_converter,
                    warnings=warnings,
                )
            )
            continue

        markdown = str(chunk.get("text", "")).strip()
        plain_text = _markdown_to_plain_text(markdown)

        boxes = chunk.get("page_boxes", [])
        figures = [
            f"picture_{index}"
            for index, box in enumerate(boxes, start=1)
            if isinstance(box, dict) and box.get("class") == "picture"
        ]

        parsed_pages.append(
            ParsedPage(
                content=PageContent(
                    plain_text=plain_text,
                    markdown=markdown,
                    figures=figures,
                ),
                provenance=PageProvenance(
                    source_file=pdf_path.name,
                    document_id=pdf_path.stem,
                    page_number=page_number,
                    parser="pymupdf4llm",
                    ocr_used=False,
                ),
                quality=PageQuality(
                    character_count=len(plain_text),
                    word_count=len(plain_text.split()),
                    is_empty=not plain_text,
                    needs_review=False,
                    warnings=[],
                ),
            )
        )

    return parsed_pages