"""Strict text-only PyMuPDF fallback for failed Docling conversions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from phosprocess.ingestion.extraction_quality import (
    PageQualityRecord,
    assess_page,
)

_HEADING = re.compile(r"^[A-ZÀ-ÖØ-Þ0-9][A-ZÀ-ÖØ-Þ0-9 .,:;()/-]{3,80}$")


@dataclass(frozen=True, slots=True)
class FallbackPage:
    """One page extracted in deterministic reading order."""

    page_number: int
    text: str
    markdown: str
    quality: PageQualityRecord


@dataclass(frozen=True, slots=True)
class FallbackExtraction:
    """Serializable fallback document."""

    pages: tuple[FallbackPage, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_name": "pymupdf_fallback_v1",
            "pages": [
                {
                    "page_number": page.page_number,
                    "text": page.text,
                    "markdown": page.markdown,
                }
                for page in self.pages
            ],
        }


def _text_to_markdown(text: str) -> str:
    lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()

        if stripped and _HEADING.fullmatch(stripped) and len(stripped.split()) <= 12:
            lines.append(f"## {stripped}")
        else:
            lines.append(line.rstrip())

    return "\n".join(lines).strip()


def extract_with_pymupdf(path: Path) -> FallbackExtraction:
    """Extract native text without OCR and retain exact one-based pages."""

    try:
        document = pymupdf.open(path)
    except Exception as error:
        raise ValueError(f"PDF illisible : {path}") from error

    with document:
        if not document.is_pdf or document.page_count <= 0:
            raise ValueError(f"PDF invalide : {path}")

        if document.needs_pass:
            raise ValueError(f"PDF protégé par mot de passe : {path}")

        pages: list[FallbackPage] = []

        for page_index, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True).strip()
            markdown = _text_to_markdown(text)
            image_count = len(page.get_images(full=True))
            quality = assess_page(
                page_number=page_index,
                text=text,
                markdown=markdown,
                parser="pymupdf",
                image_count=image_count,
                native_text_present=True,
                ocr_enabled=False,
            )
            pages.append(
                FallbackPage(
                    page_number=page_index,
                    text=text,
                    markdown=markdown,
                    quality=quality,
                )
            )

    return FallbackExtraction(pages=tuple(pages))
