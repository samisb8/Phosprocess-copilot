"""Deterministic page and document extraction-quality assessment."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field

_WORD = re.compile(r"(?u)[^\W\d_]{3,}")
_FORMULA_HINT = re.compile(
    r"<!--\s*formula|(?:^|\s)[A-Za-zΑ-ω]\s*=|[∑∫√≈≤≥∆Δ]",
    flags=re.IGNORECASE,
)
_FIGURE_HINT = re.compile(r"(?im)^\s*(?:figure|fig\.|diagram|schéma)\s+\d+")


class PageExtractionStatus(StrEnum):
    """Quality class assigned to one extracted page."""

    NATIVE_TEXT = "native_text"
    EXTRACTED_SUCCESSFULLY = "extracted_successfully"
    LOW_TEXT = "low_text"
    IMAGE_ONLY = "image_only"
    CORRUPTED_TEXT = "corrupted_text"
    OCR_REQUIRED = "ocr_required"


class PageQualityRecord(BaseModel):
    """Persisted page-level extraction diagnostics."""

    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(gt=0)
    status: PageExtractionStatus
    parser: str = Field(min_length=1)
    character_count: int = Field(ge=0)
    word_count: int = Field(ge=0)
    letter_ratio: float = Field(ge=0.0, le=1.0)
    replacement_character_ratio: float = Field(ge=0.0, le=1.0)
    control_character_ratio: float = Field(ge=0.0, le=1.0)
    table_count: int = Field(ge=0)
    formula_count: int = Field(ge=0)
    figure_count: int = Field(ge=0)
    title_count: int = Field(ge=0)
    section_count: int = Field(ge=0)
    image_count: int = Field(ge=0)
    warnings: tuple[str, ...] = ()

    @computed_field
    @property
    def usable(self) -> bool:
        """Whether the page can contribute trustworthy text."""

        return self.status in {
            PageExtractionStatus.NATIVE_TEXT,
            PageExtractionStatus.EXTRACTED_SUCCESSFULLY,
            PageExtractionStatus.LOW_TEXT,
        }


class DocumentExtractionReport(BaseModel):
    """Aggregated extraction metrics used as an activation gate."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    source_filename: str
    source_sha256: str
    parser: str
    fallback_used: bool
    ocr_enabled: bool
    ocr_used: bool
    pipeline_version: str
    total_pages: int = Field(gt=0)
    pages_with_text: int = Field(ge=0)
    pages_without_text: int = Field(ge=0)
    suspicious_pages: int = Field(ge=0)
    tables: int = Field(ge=0)
    formulas: int = Field(ge=0)
    figures: int = Field(ge=0)
    replacement_character_ratio: float = Field(ge=0.0, le=1.0)
    average_page_characters: float = Field(ge=0.0)
    titles_detected: int = Field(ge=0)
    sections_detected: int = Field(ge=0)
    status_counts: dict[str, int]
    activation_allowed: bool
    rejection_reasons: tuple[str, ...] = ()


def _control_character_ratio(text: str) -> float:
    visible = max(1, sum(not character.isspace() for character in text))
    controls = sum(
        unicodedata.category(character) == "Cc" and character not in "\n\r\t" for character in text
    )
    return controls / visible


def assess_page(
    *,
    page_number: int,
    text: str,
    parser: str,
    markdown: str = "",
    table_count: int = 0,
    formula_count: int | None = None,
    figure_count: int | None = None,
    title_count: int = 0,
    section_count: int = 0,
    image_count: int = 0,
    native_text_present: bool = True,
    ocr_enabled: bool = False,
) -> PageQualityRecord:
    """Classify one page without invoking OCR or an LLM."""

    stripped = text.strip()
    visible_count = sum(not character.isspace() for character in stripped)
    letter_count = sum(character.isalpha() for character in stripped)
    word_count = len(_WORD.findall(stripped))
    replacement_count = stripped.count("\ufffd")
    replacement_ratio = replacement_count / max(1, visible_count)
    control_ratio = _control_character_ratio(stripped)
    letter_ratio = letter_count / max(1, visible_count)
    formulas = (
        len(_FORMULA_HINT.findall(markdown or stripped)) if formula_count is None else formula_count
    )
    figures = (
        len(_FIGURE_HINT.findall(markdown or stripped)) if figure_count is None else figure_count
    )
    warnings: list[str] = []

    if not stripped:
        if image_count:
            status = (
                PageExtractionStatus.OCR_REQUIRED
                if not ocr_enabled
                else PageExtractionStatus.IMAGE_ONLY
            )
            warnings.append("image_without_extracted_text")
        else:
            status = PageExtractionStatus.LOW_TEXT
            warnings.append("empty_page")
    elif visible_count >= 200 and (
        letter_ratio < 0.15 or replacement_ratio > 0.05 or control_ratio > 0.08 or word_count < 5
    ):
        status = PageExtractionStatus.CORRUPTED_TEXT
        warnings.append("unusable_character_distribution")
    elif visible_count < 80 or word_count < 10:
        status = PageExtractionStatus.LOW_TEXT
        warnings.append("low_text")
    elif native_text_present:
        status = PageExtractionStatus.NATIVE_TEXT
    else:
        status = PageExtractionStatus.EXTRACTED_SUCCESSFULLY

    return PageQualityRecord(
        page_number=page_number,
        status=status,
        parser=parser,
        character_count=len(stripped),
        word_count=word_count,
        letter_ratio=round(letter_ratio, 6),
        replacement_character_ratio=round(replacement_ratio, 6),
        control_character_ratio=round(control_ratio, 6),
        table_count=table_count,
        formula_count=formulas,
        figure_count=figures,
        title_count=title_count,
        section_count=section_count,
        image_count=image_count,
        warnings=tuple(warnings),
    )


def build_extraction_report(
    *,
    document_id: str,
    source_filename: str,
    source_sha256: str,
    parser: str,
    fallback_used: bool,
    ocr_enabled: bool,
    ocr_used: bool,
    pipeline_version: str,
    pages: list[PageQualityRecord],
    maximum_corrupted_fraction: float = 0.30,
) -> DocumentExtractionReport:
    """Aggregate page metrics and decide whether indexing is safe."""

    if not pages:
        raise ValueError("Une extraction doit contenir au moins une page.")

    status_counts = Counter(page.status.value for page in pages)
    usable_pages = sum(page.usable and page.character_count > 0 for page in pages)
    suspicious_statuses = {
        PageExtractionStatus.CORRUPTED_TEXT,
        PageExtractionStatus.OCR_REQUIRED,
    }
    suspicious_pages = sum(page.status in suspicious_statuses for page in pages)
    corrupted_fraction = suspicious_pages / len(pages)
    reasons: list[str] = []

    if usable_pages == 0:
        reasons.append("no_usable_page")

    if corrupted_fraction > maximum_corrupted_fraction:
        reasons.append("more_than_30_percent_corrupted_or_ocr_required")

    if (
        not ocr_enabled
        and status_counts[PageExtractionStatus.OCR_REQUIRED.value]
        and usable_pages == 0
    ):
        reasons.append("ocr_required_but_disabled")

    total_characters = sum(page.character_count for page in pages)
    total_replacements = sum(
        round(page.replacement_character_ratio * max(1, page.character_count)) for page in pages
    )

    return DocumentExtractionReport(
        document_id=document_id,
        source_filename=source_filename,
        source_sha256=source_sha256,
        parser=parser,
        fallback_used=fallback_used,
        ocr_enabled=ocr_enabled,
        ocr_used=ocr_used,
        pipeline_version=pipeline_version,
        total_pages=len(pages),
        pages_with_text=sum(page.character_count > 0 for page in pages),
        pages_without_text=sum(page.character_count == 0 for page in pages),
        suspicious_pages=suspicious_pages,
        tables=sum(page.table_count for page in pages),
        formulas=sum(page.formula_count for page in pages),
        figures=sum(page.figure_count for page in pages),
        replacement_character_ratio=round(
            total_replacements / max(1, total_characters),
            6,
        ),
        average_page_characters=round(total_characters / len(pages), 2),
        titles_detected=sum(page.title_count for page in pages),
        sections_detected=sum(page.section_count for page in pages),
        status_counts=dict(sorted(status_counts.items())),
        activation_allowed=not reasons,
        rejection_reasons=tuple(reasons),
    )
