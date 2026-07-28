"""Structured extraction quality, fallback and SHA-cache tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest
from docling.datamodel.base_models import ConversionStatus
from docling_core.types.doc import DocItemLabel, DoclingDocument

from phosprocess.ingestion.docling_extractor import (
    DoclingStructuredExtractor,
    StructuredExtractionError,
    _docling_page_quality,
    recover_formula_text,
)
from phosprocess.ingestion.extraction_quality import (
    PageExtractionStatus,
    assess_page,
    build_extraction_report,
)
from phosprocess.ingestion.pdf_fallback import extract_with_pymupdf


def make_pdf(path: Path, pages: tuple[str, ...]) -> None:
    with pymupdf.open() as document:
        for text in pages:
            page = document.new_page()

            if text:
                page.insert_textbox(
                    page.rect + (40, 40, -40, -40),
                    text,
                    fontsize=10,
                )

        document.save(path)


def test_page_quality_detects_native_empty_and_corrupted_text() -> None:
    native = assess_page(
        page_number=1,
        text="Heat transfer occurs by conduction and convection. " * 20,
        parser="docling",
    )
    empty = assess_page(
        page_number=2,
        text="",
        parser="docling",
        image_count=1,
        ocr_enabled=False,
    )
    corrupted = assess_page(
        page_number=3,
        text=("! # $ % 1 2 3 " * 100),
        parser="docling",
    )

    assert native.status is PageExtractionStatus.NATIVE_TEXT
    assert empty.status is PageExtractionStatus.OCR_REQUIRED
    assert corrupted.status is PageExtractionStatus.CORRUPTED_TEXT


def test_document_quality_rejects_more_than_thirty_percent_corrupted() -> None:
    pages = [
        assess_page(
            page_number=number,
            text=(
                "Usable technical process description. " * 20
                if number <= 6
                else "! # $ % 1 2 3 " * 100
            ),
            parser="docling",
        )
        for number in range(1, 11)
    ]
    report = build_extraction_report(
        document_id="document",
        source_filename="document.pdf",
        source_sha256="a" * 64,
        parser="docling",
        fallback_used=False,
        ocr_enabled=False,
        ocr_used=False,
        pipeline_version="quality-v1",
        pages=pages,
    )

    assert report.activation_allowed is False
    assert "more_than_30_percent_corrupted_or_ocr_required" in (
        report.rejection_reasons
    )


def test_pymupdf_fallback_preserves_pages_and_empty_page(tmp_path: Path) -> None:
    source = tmp_path / "textual.pdf"
    make_pdf(
        source,
        (
            "CHAPTER 1\nHeat transfer coefficient and temperature.",
            "",
        ),
    )

    extraction = extract_with_pymupdf(source)

    assert len(extraction.pages) == 2
    assert extraction.pages[0].page_number == 1
    assert "## CHAPTER 1" in extraction.pages[0].markdown
    assert extraction.pages[1].quality.status is PageExtractionStatus.LOW_TEXT


def test_invalid_pdf_is_refused_by_fallback(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.pdf"
    invalid.write_bytes(b"not a pdf")

    with pytest.raises(ValueError, match="PDF illisible"):
        extract_with_pymupdf(invalid)


class FailingConverter:
    def __init__(self) -> None:
        self.calls = 0

    def convert(self, _source: Path, **_kwargs: object) -> object:
        self.calls += 1
        raise RuntimeError("Docling unavailable")


def test_docling_failure_uses_fallback_and_sha_cache(tmp_path: Path) -> None:
    source = tmp_path / "textual.pdf"
    make_pdf(
        source,
        ("Industrial heat exchanger operation and energy balance. " * 20,),
    )
    converter = FailingConverter()
    extractor = DoclingStructuredExtractor(
        parsed_root=tmp_path / "parsed",
        converter=converter,
    )

    first = extractor.extract(pdf_path=source, document_id="textual")
    second = extractor.extract(pdf_path=source, document_id="textual")

    assert first.cached is False
    assert first.report.fallback_used is True
    assert first.report.activation_allowed is True
    assert second.cached is True
    assert converter.calls == 1
    assert first.document_path.is_file()
    assert first.markdown_path.is_file()
    assert first.page_quality_path.is_file()


def test_unusable_fallback_is_not_cached(tmp_path: Path) -> None:
    source = tmp_path / "encoded.pdf"
    make_pdf(source, (("! # $ % 1 2 3 " * 100),))
    extractor = DoclingStructuredExtractor(
        parsed_root=tmp_path / "parsed",
        converter=FailingConverter(),
    )

    with pytest.raises(StructuredExtractionError, match="Extraction refusée"):
        extractor.extract(pdf_path=source, document_id="encoded")

    assert not list((tmp_path / "parsed").rglob("extraction_report.json"))


def test_large_document_conversion_uses_bounded_page_ranges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int] | None] = []

    class Recorder:
        def convert(self, _source: Path, **kwargs: object) -> object:
            page_range = kwargs.get("page_range")
            calls.append(
                page_range
                if isinstance(page_range, tuple)
                else None
            )
            return SimpleNamespace(
                status=ConversionStatus.SUCCESS,
                document=SimpleNamespace(page_range=page_range),
            )

    monkeypatch.setattr(
        DoclingDocument,
        "concatenate",
        lambda documents: SimpleNamespace(documents=documents),
    )
    extractor = DoclingStructuredExtractor(
        parsed_root=tmp_path,
        converter=Recorder(),
    )

    result = extractor._convert(
        extractor.converter,
        tmp_path / "large.pdf",
        page_count=1201,
    )

    assert calls == [(1, 500), (501, 1000), (1001, 1201)]
    assert len(result.documents) == 3


def test_large_document_conversion_reuses_sha_batch_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    class Recorder:
        def convert(self, _source: Path, **kwargs: object) -> object:
            page_range = kwargs["page_range"]
            assert isinstance(page_range, tuple)
            calls.append(page_range)
            return SimpleNamespace(
                status=ConversionStatus.SUCCESS,
                document=DoclingDocument(
                    name=f"pages-{page_range[0]}-{page_range[1]}"
                ),
            )

    monkeypatch.setattr(
        DoclingDocument,
        "concatenate",
        lambda documents: SimpleNamespace(documents=documents),
    )
    extractor = DoclingStructuredExtractor(
        parsed_root=tmp_path,
        converter=Recorder(),
    )
    batch_cache = tmp_path / "document" / ".sha256.batches"

    first = extractor._convert(
        extractor.converter,
        tmp_path / "large.pdf",
        page_count=501,
        batch_cache_directory=batch_cache,
    )
    second = extractor._convert(
        extractor.converter,
        tmp_path / "large.pdf",
        page_count=501,
        batch_cache_directory=batch_cache,
    )

    assert calls == [(1, 500), (501, 501)]
    assert len(first.documents) == 2
    assert len(second.documents) == 2
    assert sorted(path.name for path in batch_cache.glob("*.json")) == [
        "000001-000500.json",
        "000501-000501.json",
    ]


def test_native_formula_text_is_preserved_without_optional_enrichment() -> None:
    document = DoclingDocument(name="formula")
    formula = document.add_formula(
        text="",
        orig="\x00 q = U A delta-T \x00",
    )

    recovered = recover_formula_text(document)

    assert recovered == 1
    assert formula.text == "q = U A delta-T"


def test_page_quality_aggregates_docling_items_in_one_pass() -> None:
    technical_text = "Industrial heat transfer and energy balance. " * 20
    item = SimpleNamespace(
        label=DocItemLabel.TEXT,
        text=technical_text,
        orig="",
        prov=(SimpleNamespace(page_no=1),),
    )

    class SinglePassDocument:
        def __init__(self) -> None:
            self.iterations = 0

        def iterate_items(self, **_kwargs: object) -> object:
            self.iterations += 1
            return iter(((item, 0),))

        @staticmethod
        def num_pages() -> int:
            return 1

    document = SinglePassDocument()
    native = assess_page(
        page_number=1,
        text=technical_text,
        parser="pymupdf",
    )

    pages = _docling_page_quality(
        document,  # type: ignore[arg-type]
        native_pages=(native,),
        ocr_enabled=False,
    )

    assert document.iterations == 1
    assert pages[0].character_count == len(technical_text.strip())
    assert pages[0].status is PageExtractionStatus.NATIVE_TEXT
