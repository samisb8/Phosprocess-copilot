"""Cache structured extraction and technical chunks for catalogue documents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from phosprocess.ingestion.chunk_serialization import (
    TechnicalChildChunk,
    TechnicalParentChunk,
    TechnicalSection,
    read_child_chunks,
    read_parent_chunks,
    read_sections,
    write_jsonl,
)
from phosprocess.ingestion.chunk_validation import validate_chunk_hierarchy
from phosprocess.ingestion.docling_extractor import DoclingStructuredExtractor
from phosprocess.ingestion.technical_chunker import TechnicalDocumentChunker
from phosprocess.knowledge_base.catalog import locate_catalogue_source
from phosprocess.knowledge_base.runtime import DEFAULT_KNOWLEDGE_BASE_ROOT
from phosprocess.knowledge_base.schemas import DocumentCatalogEntry

QUALITY_CHUNKING_VERSION = "quality-v2.0-hierarchical"


@dataclass(frozen=True, slots=True)
class PreparedQualityDocument:
    """Structured and chunked document ready for indexing."""

    entry: DocumentCatalogEntry
    children: tuple[TechnicalChildChunk, ...]
    parents: tuple[TechnicalParentChunk, ...]
    sections: tuple[TechnicalSection, ...]
    extraction_cached: bool
    chunking_cached: bool
    corpus_directory: Path


class QualityCorpusProcessor:
    """Prepare documents once per SHA using Docling and technical chunking."""

    def __init__(
        self,
        *,
        knowledge_base_root: Path = DEFAULT_KNOWLEDGE_BASE_ROOT,
        extractor: DoclingStructuredExtractor | None = None,
        chunker: TechnicalDocumentChunker | None = None,
    ) -> None:
        self.root = knowledge_base_root.resolve()
        self.extractor = extractor or DoclingStructuredExtractor(
            parsed_root=self.root / "parsed"
        )
        self.chunker = chunker or TechnicalDocumentChunker()

    def _corpus_directory(self, entry: DocumentCatalogEntry) -> Path:
        return self.root / "corpus" / entry.document_id / entry.sha256

    def prepare(
        self,
        entry: DocumentCatalogEntry,
    ) -> PreparedQualityDocument:
        """Load a validated cache or produce extraction and hierarchy."""

        source = locate_catalogue_source(
            entry,
            knowledge_base_root=self.root,
        )

        if source is None:
            raise FileNotFoundError(f"Source absente : {entry.source_filename}")

        if source.parent.name != "pdfs":
            raise ValueError(
                f"Le document {entry.document_id} n'est pas dans le corpus actif."
            )

        extraction = self.extractor.extract(
            pdf_path=source,
            document_id=entry.document_id,
        )
        corpus = self._corpus_directory(entry)
        child_path = corpus / "children.jsonl"
        parent_path = corpus / "parents.jsonl"
        section_path = corpus / "sections.jsonl"
        manifest_path = corpus / "chunking_report.json"

        if (
            child_path.is_file()
            and parent_path.is_file()
            and section_path.is_file()
            and manifest_path.is_file()
        ):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            if manifest.get("chunking_version") == QUALITY_CHUNKING_VERSION:
                children = read_child_chunks(child_path)
                parents = read_parent_chunks(parent_path)
                sections = read_sections(section_path)
                validate_chunk_hierarchy(
                    children,
                    parents,
                    maximum_child_tokens=self.chunker.config.child_maximum_tokens,
                    maximum_parent_tokens=self.chunker.config.parent_maximum_tokens,
                    sections=(sections or None),
                )
                return PreparedQualityDocument(
                    entry=entry,
                    children=tuple(children),
                    parents=tuple(parents),
                    sections=tuple(sections),
                    extraction_cached=extraction.cached,
                    chunking_cached=True,
                    corpus_directory=corpus,
                )

        result = self.chunker.chunk(
            document_path=extraction.document_path,
            entry=entry,
        )
        summary = validate_chunk_hierarchy(
            list(result.children),
            list(result.parents),
            maximum_child_tokens=self.chunker.config.child_maximum_tokens,
            maximum_parent_tokens=self.chunker.config.parent_maximum_tokens,
            sections=(list(result.sections) or None),
        )
        write_jsonl(child_path, list(result.children))
        write_jsonl(parent_path, list(result.parents))
        write_jsonl(section_path, list(result.sections))
        manifest_path.write_text(
            json.dumps(
                {
                    "pipeline_version": "quality-v2.0-hierarchical",
                    "chunking_version": QUALITY_CHUNKING_VERSION,
                    "document_id": entry.document_id,
                    "source_sha256": entry.sha256,
                    "child_count": summary.child_count,
                    "parent_count": summary.parent_count,
                    "section_count": len(result.sections),
                    "maximum_child_tokens": summary.maximum_child_tokens,
                    "maximum_parent_tokens": summary.maximum_parent_tokens,
                    "excluded_chunk_count": result.excluded_chunk_count,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return PreparedQualityDocument(
            entry=entry,
            children=result.children,
            parents=result.parents,
            sections=result.sections,
            extraction_cached=extraction.cached,
            chunking_cached=False,
            corpus_directory=corpus,
        )
