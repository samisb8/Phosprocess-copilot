"""Lightweight shared data models for knowledge-base synchronization."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phosprocess.preprocessing.chunk_schemas import DocumentChunk

KNOWLEDGE_BASE_PIPELINE_VERSION = "1.0.0"


def sha256_file(path: Path) -> str:
    """Hash one file without loading it fully into memory."""

    digest = hashlib.sha256()

    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def chunk_sha256(chunk: DocumentChunk) -> str:
    """Build a stable content digest for cache and provenance checks."""

    return hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ProcessedDocument:
    """Validated chunks and statistics for one PDF version."""

    filename: str
    document_id: str
    document_sha256: str
    page_count: int
    empty_pages: tuple[int, ...]
    chunks: tuple[DocumentChunk, ...]
    duplicates_removed: int
    ingestion_date: str
    cache_directory: Path

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def metadata_records(self) -> list[dict[str, Any]]:
        """Return enriched records accepted by existing retriever loaders."""

        return [
            {
                **chunk.model_dump(mode="json"),
                "filename": self.filename,
                "document_sha256": self.document_sha256,
                "chunk_sha256": chunk_sha256(chunk),
                "section": (
                    " > ".join(chunk.heading_path)
                    if chunk.heading_path
                    else None
                ),
                "ingestion_date": self.ingestion_date,
                "pipeline_version": KNOWLEDGE_BASE_PIPELINE_VERSION,
            }
            for chunk in self.chunks
        ]


@dataclass(frozen=True, slots=True)
class IndexBuildResult:
    """One validated version waiting for atomic activation."""

    chunk_count: int
    document_counts: dict[str, int]
    embedded_chunk_count: int
    reused_embedding_count: int
    dense_search_ok: bool
    bm25_search_ok: bool
    hybrid_search_ok: bool
