"""Build a validated hierarchical quality index with section and chunk stages."""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import bm25s
import faiss
import numpy as np

from phosprocess.ingestion.chunk_serialization import (
    TechnicalChildChunk,
    TechnicalParentChunk,
    TechnicalSection,
    write_jsonl,
)
from phosprocess.ingestion.chunk_validation import validate_chunk_hierarchy
from phosprocess.knowledge_base.indexing import VersionIndexBuilder
from phosprocess.knowledge_base.models import IndexBuildResult
from phosprocess.knowledge_base.schemas import DocumentCatalogEntry
from phosprocess.retrieval.bm25 import TOKENIZER_VERSION, technical_tokenize

QUALITY_INDEX_PIPELINE_VERSION = "quality-v2.0-hierarchical"


def child_to_runtime_record(
    child: TechnicalChildChunk,
    *,
    chunk_index: int,
    document_sha256: str,
) -> dict[str, Any]:
    """Expose hierarchy metadata through the existing chunk retrievers."""

    heading_path = [value for value in (child.chapter, child.section, child.subsection) if value]
    return {
        "chunk_id": child.chunk_id,
        "document_id": child.document_id,
        "source_file": child.source_file,
        "chunk_index": chunk_index,
        "heading_path": heading_path,
        "source_pages": list(range(child.page_start, child.page_end + 1)),
        "page_start": child.page_start,
        "page_end": child.page_end,
        "content_types": [child.chunk_type.value],
        "text": child.display_text,
        "embedding_text": child.embedding_text,
        "body_token_count": child.token_count,
        "token_count": child.token_count,
        "source_chunk_ids": [],
        "postprocessing_actions": [],
        "filename": child.source_file,
        "document_sha256": document_sha256,
        "chunk_sha256": child.sha256,
        "section": child.section,
        "ingestion_date": datetime.now(UTC).isoformat(),
        "pipeline_version": QUALITY_INDEX_PIPELINE_VERSION,
        "parent_id": child.parent_id,
        "previous_chunk_id": child.previous_chunk_id,
        "next_chunk_id": child.next_chunk_id,
        "document_title": child.document_title,
        "domains": list(child.domains),
        "chapter": child.chapter,
        "subsection": child.subsection,
        "hierarchy_path": child.hierarchy_path,
        "section_id": child.section_id,
        "chunk_type": child.chunk_type.value,
        "display_text": child.display_text,
        "bm25_text": child.bm25_text,
        "active": child.active,
        "retrieval_weight": child.retrieval_weight,
        "source_item_labels": list(child.source_item_labels),
    }


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def _section_record(section: TechnicalSection) -> dict[str, Any]:
    return section.model_dump(mode="json")


class QualityIndexBuilder:
    """Build immutable chunk indexes plus a first-stage section index."""

    def __init__(self, base_builder: VersionIndexBuilder) -> None:
        self.base_builder = base_builder

    def _build_section_indexes(
        self,
        *,
        sections: list[TechnicalSection],
        version_directory: Path,
    ) -> None:
        if not sections:
            raise ValueError("Aucune section hiérarchique à indexer.")

        dense_metadata_path = version_directory / "dense" / "metadata.jsonl"
        chunk_vectors_path = version_directory / "dense" / "embeddings.npy"
        chunk_records = self.base_builder._read_jsonl(dense_metadata_path)
        chunk_vectors = np.load(chunk_vectors_path, allow_pickle=False)
        chunk_position = {
            str(record["chunk_id"]): position for position, record in enumerate(chunk_records)
        }
        section_vectors = np.empty(
            (
                len(sections),
                self.base_builder.embedding_config.embedding_dimension,
            ),
            dtype=np.float32,
        )

        for position, section in enumerate(sections):
            missing = [
                chunk_id for chunk_id in section.child_chunk_ids if chunk_id not in chunk_position
            ]
            if missing:
                raise ValueError(
                    f"Chunks absents de l'index dense pour {section.section_id}: "
                    + ", ".join(missing[:5])
                )
            vectors = np.asarray(
                [chunk_vectors[chunk_position[item]] for item in section.child_chunk_ids],
                dtype=np.float32,
            )
            average = vectors.mean(axis=0)
            norm = float(np.linalg.norm(average))
            if norm <= 0:
                raise ValueError(f"Embedding de section nul : {section.section_id}")
            section_vectors[position] = average / norm

        section_root = version_directory / "sections"
        dense_directory = section_root / "dense"
        bm25_directory = section_root / "bm25"
        dense_directory.mkdir(parents=True, exist_ok=False)
        bm25_directory.mkdir(parents=True, exist_ok=False)

        dense_index = faiss.IndexFlatIP(self.base_builder.embedding_config.embedding_dimension)
        dense_index.add(np.ascontiguousarray(section_vectors, dtype=np.float32))
        faiss.write_index(dense_index, str(dense_directory / "index.faiss"))
        np.save(dense_directory / "embeddings.npy", section_vectors)
        _write_jsonl(
            (
                {"vector_id": index, **_section_record(section)}
                for index, section in enumerate(sections)
            ),
            dense_directory / "metadata.jsonl",
        )
        (dense_directory / "manifest.json").write_text(
            json.dumps(
                {
                    "pipeline_version": QUALITY_INDEX_PIPELINE_VERSION,
                    "created_at_utc": datetime.now(UTC).isoformat(),
                    "index_type": "IndexFlatIP",
                    "dimension": int(dense_index.d),
                    "section_count": len(sections),
                    "vector_strategy": "normalized_mean_of_child_embeddings",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        tokenized = [technical_tokenize(section.bm25_text) for section in sections]
        if any(not tokens for tokens in tokenized):
            raise ValueError("Une section ne contient aucun token BM25.")
        config = self.base_builder.bm25_config
        model = bm25s.BM25(
            method=config.method,
            k1=config.k1,
            b=config.b,
            backend=config.backend,
            csc_backend=config.csc_backend,
        )
        model.index(tokenized, show_progress=False, leave_progress=False)
        model.save(str(bm25_directory), show_progress=False)
        _write_jsonl(
            (
                {"lexical_id": index, **_section_record(section)}
                for index, section in enumerate(sections)
            ),
            bm25_directory / "metadata.jsonl",
        )
        (bm25_directory / "manifest.json").write_text(
            json.dumps(
                {
                    "pipeline_version": QUALITY_INDEX_PIPELINE_VERSION,
                    "created_at_utc": datetime.now(UTC).isoformat(),
                    "library": {
                        "name": "bm25s",
                        "version": package_version("bm25s"),
                    },
                    "tokenizer_version": TOKENIZER_VERSION,
                    "method": config.method,
                    "k1": config.k1,
                    "b": config.b,
                    "section_count": len(sections),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def build(
        self,
        *,
        children: list[TechnicalChildChunk],
        parents: list[TechnicalParentChunk],
        sections: list[TechnicalSection] | None = None,
        documents: list[DocumentCatalogEntry],
        version_directory: Path,
        previous_version_directory: Path | None,
        maximum_child_tokens: int = 560,
        maximum_parent_tokens: int = 1700,
    ) -> IndexBuildResult:
        """Validate hierarchy, build both retrieval levels and persist manifests."""

        children_by_document: dict[str, list[TechnicalChildChunk]] = defaultdict(list)
        parents_by_document: dict[str, list[TechnicalParentChunk]] = defaultdict(list)
        sections_by_document: dict[str, list[TechnicalSection]] = defaultdict(list)
        hierarchical_enabled = bool(sections)

        for child in children:
            children_by_document[child.document_id].append(child)
        for parent in parents:
            parents_by_document[parent.document_id].append(parent)
        for section in sections or ():
            sections_by_document[section.document_id].append(section)

        hierarchy_summaries: dict[str, dict[str, int]] = {}
        for document_id, document_children in children_by_document.items():
            summary = validate_chunk_hierarchy(
                document_children,
                parents_by_document[document_id],
                maximum_child_tokens=maximum_child_tokens,
                maximum_parent_tokens=maximum_parent_tokens,
                sections=(sections_by_document[document_id] if hierarchical_enabled else None),
            )
            hierarchy_summaries[document_id] = {
                "children": summary.child_count,
                "parents": summary.parent_count,
                "maximum_child_tokens": summary.maximum_child_tokens,
                "maximum_parent_tokens": summary.maximum_parent_tokens,
                "sections": summary.section_count,
            }

        document_by_id = {document.document_id: document for document in documents}
        if set(document_by_id) != set(children_by_document):
            raise ValueError("Documents catalogue et chunks qualité désalignés.")

        records: list[dict[str, Any]] = []
        for document_id in sorted(children_by_document):
            entry = document_by_id[document_id]
            for chunk_index, child in enumerate(children_by_document[document_id]):
                records.append(
                    child_to_runtime_record(
                        child,
                        chunk_index=chunk_index,
                        document_sha256=entry.sha256,
                    )
                )

        build = self.base_builder.build(
            records=records,
            version_directory=version_directory,
            previous_version_directory=previous_version_directory,
        )
        effective_sections = list(sections or ())
        write_jsonl(version_directory / "chunks.jsonl", children)
        write_jsonl(version_directory / "parents.jsonl", parents)
        write_jsonl(version_directory / "sections.jsonl", effective_sections)
        if effective_sections:
            self._build_section_indexes(
                sections=effective_sections,
                version_directory=version_directory,
            )
        (version_directory / "documents.json").write_text(
            json.dumps(
                {
                    "catalog_version": QUALITY_INDEX_PIPELINE_VERSION,
                    "documents": [document.model_dump(mode="json") for document in documents],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _link_or_copy(
            version_directory / "dense" / "index.faiss",
            version_directory / "dense.faiss",
        )
        _link_or_copy(
            version_directory / "dense" / "metadata.jsonl",
            version_directory / "dense_mapping.jsonl",
        )
        manifest = {
            "pipeline_version": QUALITY_INDEX_PIPELINE_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "document_count": len(documents),
            "child_chunk_count": len(children),
            "parent_chunk_count": len(parents),
            "section_count": len(effective_sections),
            "active_document_ids": sorted(document_by_id),
            "hierarchy_validation": hierarchy_summaries,
            "retrieval_stages": (["section", "chunk"] if effective_sections else ["chunk"]),
            "dense_format": "FAISS IndexFlatIP",
            "bm25_format": "bm25s directory",
            "runtime_compatibility": {
                "dense": "dense/index.faiss",
                "bm25": "bm25/",
                "section_dense": "sections/dense/index.faiss",
                "section_bm25": "sections/bm25/",
            },
        }
        (version_directory / "index_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (version_directory / "build_report.json").write_text(
            json.dumps(
                {
                    **manifest,
                    "embedded_chunk_count": build.embedded_chunk_count,
                    "reused_embedding_count": build.reused_embedding_count,
                    "dense_search_ok": build.dense_search_ok,
                    "bm25_search_ok": build.bm25_search_ok,
                    "hybrid_search_ok": build.hybrid_search_ok,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return build
