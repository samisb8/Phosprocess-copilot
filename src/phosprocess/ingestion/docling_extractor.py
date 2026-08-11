"""Cached Docling-first structured extraction with a strict PDF fallback."""

from __future__ import annotations

import gc
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DocItemLabel, DoclingDocument

from phosprocess.ingestion.extraction_quality import (
    DocumentExtractionReport,
    PageQualityRecord,
    assess_page,
    build_extraction_report,
)
from phosprocess.ingestion.pdf_fallback import extract_with_pymupdf
from phosprocess.knowledge_base.models import sha256_file

EXTRACTION_PIPELINE_VERSION = "quality-v1"
FORMULA_RECOVERY_VERSION = "formula-orig-v1"


class ConverterProtocol(Protocol):
    """Small dependency boundary used by unit tests."""

    def convert(self, source: Path, **kwargs: Any) -> Any:
        """Convert one source document."""


class StructuredExtractionError(RuntimeError):
    """Raised when neither Docling nor fallback provides safe text."""


@dataclass(frozen=True, slots=True)
class DoclingExtractionConfig:
    """Docling and quality settings for one immutable extraction."""

    ocr_enabled: bool = False
    ocr_only_when_required: bool = True
    table_structure: bool = True
    formula_enrichment: bool = False
    maximum_corrupted_fraction: float = 0.30
    fallback_parser: str = "pymupdf"
    batch_page_count: int = 100

    def __post_init__(self) -> None:
        if self.batch_page_count <= 0:
            raise ValueError("batch_page_count doit être positif.")


@dataclass(frozen=True, slots=True)
class StructuredExtractionResult:
    """Paths and report for one cached structured document."""

    cache_directory: Path
    document_path: Path
    markdown_path: Path
    page_quality_path: Path
    report_path: Path
    report: DocumentExtractionReport
    cached: bool


def build_docling_converter(
    config: DoclingExtractionConfig,
) -> DocumentConverter:
    """Build one reusable local converter without remote services."""

    options = PdfPipelineOptions()
    options.do_ocr = config.ocr_enabled
    options.do_table_structure = config.table_structure
    options.do_formula_enrichment = config.formula_enrichment
    options.enable_remote_services = False
    options.heading_hierarchy_options.enabled = True

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)},
    )


def _page_item_data(
    document: DoclingDocument,
) -> tuple[dict[int, Counter[str]], dict[int, list[str]]]:
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    texts: dict[int, list[str]] = defaultdict(list)

    for item, _level in document.iterate_items(
        with_groups=False,
        traverse_pictures=True,
    ):
        label = getattr(item, "label", None)
        label_value = str(label.value) if hasattr(label, "value") else str(label or "unknown")
        provenances = getattr(item, "prov", None) or ()
        item_text = str(getattr(item, "text", "") or getattr(item, "orig", "") or "").strip()

        for provenance in provenances:
            page_number = int(provenance.page_no)
            counts[page_number][label_value] += 1
            if item_text:
                texts[page_number].append(item_text)

    return counts, texts


def _docling_page_quality(
    document: DoclingDocument,
    *,
    native_pages: tuple[PageQualityRecord, ...],
    ocr_enabled: bool,
) -> list[PageQualityRecord]:
    counts, item_texts = _page_item_data(document)
    native_by_page = {page.page_number: page for page in native_pages}
    pages: list[PageQualityRecord] = []

    for page_number in range(1, document.num_pages() + 1):
        text = "\n".join(item_texts.get(page_number, ()))
        item_counts = counts.get(page_number, Counter())
        native = native_by_page.get(page_number)
        pages.append(
            assess_page(
                page_number=page_number,
                text=text,
                markdown=text,
                parser="docling",
                table_count=item_counts[DocItemLabel.TABLE.value],
                formula_count=item_counts[DocItemLabel.FORMULA.value],
                figure_count=item_counts[DocItemLabel.PICTURE.value],
                title_count=item_counts[DocItemLabel.TITLE.value],
                section_count=item_counts[DocItemLabel.SECTION_HEADER.value],
                image_count=item_counts[DocItemLabel.PICTURE.value],
                native_text_present=(
                    native is not None
                    and native.character_count > 0
                    and native.status.value != "corrupted_text"
                ),
                ocr_enabled=ocr_enabled,
            )
        )

    return pages


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_quality(path: Path, pages: list[PageQualityRecord]) -> None:
    path.write_text(
        "\n".join(page.model_dump_json() for page in pages) + "\n",
        encoding="utf-8",
    )


def recover_formula_text(document: DoclingDocument) -> int:
    """Retain native formula text when optional formula enrichment is off."""

    recovered = 0

    for item, _level in document.iterate_items(
        with_groups=False,
        traverse_pictures=True,
    ):
        if getattr(item, "label", None) != DocItemLabel.FORMULA:
            continue

        if str(getattr(item, "text", "") or "").strip():
            continue

        original = str(getattr(item, "orig", "") or "")
        normalized = "".join(
            character if character.isprintable() else " " for character in original
        )
        normalized = " ".join(normalized.split())

        if not normalized:
            continue

        item.text = normalized
        recovered += 1

    return recovered


def _publish_cache(temporary: Path, final: Path) -> None:
    if final.exists():
        raise FileExistsError(f"Cache déjà présent : {final}")

    try:
        os.rename(temporary, final)
    except PermissionError:
        shutil.copytree(temporary, final)
        shutil.rmtree(temporary)


class DoclingStructuredExtractor:
    """Convert PDFs once per SHA and persist structured provenance."""

    def __init__(
        self,
        *,
        parsed_root: Path,
        config: DoclingExtractionConfig | None = None,
        converter: ConverterProtocol | None = None,
    ) -> None:
        self.parsed_root = parsed_root.resolve()
        self.config = config or DoclingExtractionConfig()
        self.converter = converter

    def _cache_directory(self, document_id: str, digest: str) -> Path:
        return self.parsed_root / document_id / digest

    @staticmethod
    def _successful_document(
        conversion: Any,
        *,
        expected_pages: int,
    ) -> DoclingDocument:
        """Accept only a complete successful Docling conversion."""

        if conversion.status != ConversionStatus.SUCCESS:
            errors = getattr(conversion, "errors", None) or ()

            detail = "; ".join(str(error) for error in list(errors)[:3])

            message = f"Docling a retourn? {conversion.status} au lieu de SUCCESS."

            if detail:
                message += f" Erreurs : {detail}"

            raise StructuredExtractionError(message)

        document = conversion.document
        actual_pages = document.num_pages()

        if actual_pages != expected_pages:
            raise StructuredExtractionError(
                f"Couverture Docling incomplète : {actual_pages}/{expected_pages} pages."
            )

        return document

    @staticmethod
    def _batch_cache_path(
        batch_cache_directory: Path | None,
        *,
        first_page: int,
        last_page: int,
    ) -> Path | None:
        """Return the immutable cache path for one page range."""

        if batch_cache_directory is None:
            return None

        return batch_cache_directory / f"{first_page:06d}-{last_page:06d}.json"

    @staticmethod
    def _write_batch_document(
        path: Path | None,
        document: DoclingDocument,
    ) -> None:
        """Atomically persist one successful Docling page range."""

        if path is None:
            return

        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")

        _write_json(
            temporary,
            document.export_to_dict(),
        )

        try:
            os.replace(
                temporary,
                path,
            )
        except PermissionError:
            shutil.copy2(
                temporary,
                path,
            )
            temporary.unlink()

    def _convert_range(
        self,
        converter: ConverterProtocol,
        pdf_path: Path,
        *,
        first_page: int,
        last_page: int,
        batch_cache_directory: Path | None,
    ) -> DoclingDocument:
        """Convert one range and recursively split it after failure."""

        expected_pages = last_page - first_page + 1

        batch_path = self._batch_cache_path(
            batch_cache_directory,
            first_page=first_page,
            last_page=last_page,
        )

        # Only caches created by a previously successful range
        # should normally reach this branch.
        if batch_path is not None and batch_path.is_file():
            try:
                cached = DoclingDocument.load_from_json(batch_path)

                if cached.num_pages() == expected_pages:
                    return cached

            except (OSError, ValueError):
                pass

        conversion: Any | None = None

        try:
            conversion = converter.convert(
                pdf_path,
                raises_on_error=False,
                page_range=(
                    first_page,
                    last_page,
                ),
            )

            document = self._successful_document(
                conversion,
                expected_pages=expected_pages,
            )

            self._write_batch_document(
                batch_path,
                document,
            )

        except Exception as error:
            conversion = None
            gc.collect()

            # A one-page range cannot be divided further.
            if first_page == last_page:
                raise StructuredExtractionError(
                    f"?chec Docling persistant sur la page {first_page}: {error}"
                ) from error

            middle = (first_page + last_page) // 2

            left = self._convert_range(
                converter,
                pdf_path,
                first_page=first_page,
                last_page=middle,
                batch_cache_directory=(batch_cache_directory),
            )

            gc.collect()

            right = self._convert_range(
                converter,
                pdf_path,
                first_page=middle + 1,
                last_page=last_page,
                batch_cache_directory=(batch_cache_directory),
            )

            document = DoclingDocument.concatenate([left, right])

            actual_pages = document.num_pages()

            if actual_pages != expected_pages:
                raise StructuredExtractionError(
                    "Assemblage Docling incomplet "
                    f"pour {first_page}-{last_page}: "
                    f"{actual_pages}/{expected_pages} pages."
                ) from error

            self._write_batch_document(
                batch_path,
                document,
            )

            gc.collect()

            return document

        conversion = None
        gc.collect()

        return document

    def _convert(
        self,
        converter: ConverterProtocol,
        pdf_path: Path,
        *,
        page_count: int,
        batch_cache_directory: Path | None = None,
    ) -> DoclingDocument:
        """Convert one PDF using bounded and adaptive page ranges."""

        batch_size = self.config.batch_page_count

        if batch_cache_directory is not None:
            batch_cache_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        # Small PDF: one bounded conversion is enough.
        if page_count <= batch_size:
            return self._convert_range(
                converter,
                pdf_path,
                first_page=1,
                last_page=page_count,
                batch_cache_directory=(batch_cache_directory),
            )

        top_level_ranges: list[tuple[int, int, Path | None]] = []

        # ---------------------------------------------------------
        # PHASE 1
        #
        # Parse each bounded range independently.
        # Do not keep every converted batch alive while parsing the
        # following batches.
        # ---------------------------------------------------------

        for first_page in range(
            1,
            page_count + 1,
            batch_size,
        ):
            last_page = min(
                page_count,
                first_page + batch_size - 1,
            )

            batch_path = self._batch_cache_path(
                batch_cache_directory,
                first_page=first_page,
                last_page=last_page,
            )

            document = self._convert_range(
                converter,
                pdf_path,
                first_page=first_page,
                last_page=last_page,
                batch_cache_directory=(batch_cache_directory),
            )

            actual_pages = document.num_pages()
            expected_pages = last_page - first_page + 1

            if actual_pages != expected_pages:
                raise StructuredExtractionError(
                    "Batch Docling incomplet "
                    f"{first_page}-{last_page}: "
                    f"{actual_pages}/{expected_pages}."
                )

            top_level_ranges.append(
                (
                    first_page,
                    last_page,
                    batch_path,
                )
            )

            del document
            gc.collect()

        # ---------------------------------------------------------
        # PHASE 2
        #
        # The expensive preprocessing work is finished.
        # Reload successful batches and build the final document.
        # ---------------------------------------------------------

        documents: list[DoclingDocument] = []

        for first_page, last_page, batch_path in top_level_ranges:
            if batch_path is not None and batch_path.is_file():
                document = DoclingDocument.load_from_json(batch_path)
            else:
                # Only relevant for injected unit-test converters
                # without a batch cache.
                document = self._convert_range(
                    converter,
                    pdf_path,
                    first_page=first_page,
                    last_page=last_page,
                    batch_cache_directory=None,
                )

            expected_pages = last_page - first_page + 1

            if document.num_pages() != expected_pages:
                raise StructuredExtractionError(
                    f"Cache Docling incomplet {first_page}-{last_page}."
                )

            documents.append(document)

        merged = DoclingDocument.concatenate(documents)

        actual_pages = merged.num_pages()

        if actual_pages != page_count:
            raise StructuredExtractionError(
                f"Extraction Docling finale incomplète : {actual_pages}/{page_count} pages."
            )

        return merged

    @staticmethod
    def _upgrade_formula_cache(cache: Path) -> None:
        marker = cache / f"{FORMULA_RECOVERY_VERSION}.ok"
        document_path = cache / "document.json"

        if marker.is_file() or not document_path.is_file():
            return

        payload = json.loads(document_path.read_text(encoding="utf-8"))

        if payload.get("schema_name") == "pymupdf_fallback_v1":
            marker.write_text("fallback\n", encoding="utf-8")
            return

        document = DoclingDocument.model_validate(payload)
        recovered = recover_formula_text(document)

        if recovered:
            temporary_json = document_path.with_suffix(".json.tmp")
            temporary_markdown = cache / "document.md.tmp"
            _write_json(temporary_json, document.export_to_dict())
            temporary_markdown.write_text(
                document.export_to_markdown(),
                encoding="utf-8",
            )
            os.replace(temporary_json, document_path)
            os.replace(temporary_markdown, cache / "document.md")

        marker.write_text(f"recovered={recovered}\n", encoding="utf-8")

    @classmethod
    def _cached_result(
        cls,
        cache: Path,
    ) -> StructuredExtractionResult | None:
        paths = {
            "document": cache / "document.json",
            "markdown": cache / "document.md",
            "quality": cache / "page_quality.jsonl",
            "report": cache / "extraction_report.json",
        }

        if not all(path.is_file() for path in paths.values()):
            return None

        cls._upgrade_formula_cache(cache)

        try:
            report = DocumentExtractionReport.model_validate_json(
                paths["report"].read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None

        return StructuredExtractionResult(
            cache_directory=cache,
            document_path=paths["document"],
            markdown_path=paths["markdown"],
            page_quality_path=paths["quality"],
            report_path=paths["report"],
            report=report,
            cached=True,
        )

    def extract(
        self,
        *,
        pdf_path: Path,
        document_id: str,
    ) -> StructuredExtractionResult:
        """Extract, quality-gate and atomically cache one PDF version."""

        digest = sha256_file(pdf_path)
        final = self._cache_directory(document_id, digest)
        batch_cache = final.with_name(f".{final.name}.batches")
        cached = self._cached_result(final)

        if cached is not None:
            return cached

        temporary = final.with_name(f".{final.name}.{uuid4().hex}.tmp")
        temporary.mkdir(parents=True, exist_ok=False)
        fallback = extract_with_pymupdf(pdf_path)
        native_quality = tuple(page.quality for page in fallback.pages)
        if self.converter is None:
            self.converter = build_docling_converter(self.config)

        converter = self.converter
        fallback_used = False

        try:
            document = self._convert(
                converter,
                pdf_path,
                page_count=len(native_quality),
                batch_cache_directory=batch_cache,
            )
            recover_formula_text(document)
            pages = _docling_page_quality(
                document,
                native_pages=native_quality,
                ocr_enabled=self.config.ocr_enabled,
            )
            document_payload: object = document.export_to_dict()
            markdown = document.export_to_markdown()
            parser = "docling"
        except Exception as docling_error:
            if self.config.fallback_parser != "pymupdf":
                shutil.rmtree(temporary)
                raise StructuredExtractionError(
                    f"Échec Docling sans fallback : {docling_error}"
                ) from docling_error

            fallback_used = True
            pages = list(native_quality)
            document_payload = fallback.to_dict()
            markdown = "\n\n".join(page.markdown for page in fallback.pages)
            parser = "pymupdf"

        report = build_extraction_report(
            document_id=document_id,
            source_filename=pdf_path.name,
            source_sha256=digest,
            parser=parser,
            fallback_used=fallback_used,
            ocr_enabled=self.config.ocr_enabled,
            ocr_used=False,
            pipeline_version=EXTRACTION_PIPELINE_VERSION,
            pages=pages,
            maximum_corrupted_fraction=(self.config.maximum_corrupted_fraction),
        )

        if not report.activation_allowed:
            shutil.rmtree(temporary)
            raise StructuredExtractionError(
                "Extraction refusée : " + ", ".join(report.rejection_reasons)
            )

        _write_json(temporary / "document.json", document_payload)
        (temporary / "document.md").write_text(markdown, encoding="utf-8")
        _write_quality(temporary / "page_quality.jsonl", pages)
        _write_json(
            temporary / "extraction_report.json",
            report.model_dump(mode="json"),
        )
        (temporary / f"{FORMULA_RECOVERY_VERSION}.ok").write_text(
            "applied\n",
            encoding="utf-8",
        )
        final.parent.mkdir(parents=True, exist_ok=True)
        _publish_cache(temporary, final)

        if batch_cache.is_dir():
            shutil.rmtree(batch_cache)

        return StructuredExtractionResult(
            cache_directory=final,
            document_path=final / "document.json",
            markdown_path=final / "document.md",
            page_quality_path=final / "page_quality.jsonl",
            report_path=final / "extraction_report.json",
            report=report,
            cached=False,
        )
