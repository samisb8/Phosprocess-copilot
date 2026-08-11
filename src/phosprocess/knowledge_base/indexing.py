"""Reusable PDF processing and versioned FAISS/BM25 index construction."""

from __future__ import annotations

import gc
import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import bm25s
import faiss
import numpy as np
import yaml

from phosprocess.embeddings.embedder import (
    BGEEmbedder,
    EmbeddingConfig,
    load_embedding_config,
    resolve_cached_model_source,
)
from phosprocess.ingestion.pdf_loader import extract_pdf_pages
from phosprocess.ingestion.schemas import (
    PageContent,
    PageProvenance,
    PageQuality,
    ParsedPage,
)
from phosprocess.knowledge_base.models import (
    KNOWLEDGE_BASE_PIPELINE_VERSION,
    IndexBuildResult,
    ProcessedDocument,
)
from phosprocess.preprocessing.chunk_postprocessor import (
    ChunkPostprocessingConfig,
    ChunkPostprocessor,
)
from phosprocess.preprocessing.chunk_schemas import DocumentChunk
from phosprocess.preprocessing.chunker import (
    ChunkingConfig,
    StructureAwareChunker,
)
from phosprocess.preprocessing.cleaner import (
    clean_pdf_text,
    needs_manual_review,
)
from phosprocess.retrieval.bm25 import (
    TOKENIZER_VERSION,
    BM25Retriever,
    build_lexical_text,
    load_bm25_config,
    technical_tokenize,
)
from phosprocess.retrieval.hybrid import HybridRetriever

SMOKE_QUERY = "procédé phosphorique filtration gypse concentration"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError(f"Configuration YAML invalide : {path}")

    return payload


def load_chunking_config(path: Path) -> ChunkingConfig:
    """Load the existing structure-aware chunking settings."""

    raw = _load_yaml(path)
    return ChunkingConfig(
        tokenizer_name=resolve_cached_model_source(str(raw["tokenizer_name"])),
        target_tokens=int(raw["target_tokens"]),
        max_tokens=int(raw["max_tokens"]),
        overlap_tokens=int(raw["overlap_tokens"]),
        min_chunk_tokens=int(raw["min_chunk_tokens"]),
        include_document_context=bool(raw["include_document_context"]),
    )


def load_postprocessing_config(
    path: Path,
) -> ChunkPostprocessingConfig:
    """Load the existing production chunk postprocessing settings."""

    raw = _load_yaml(path)
    return ChunkPostprocessingConfig(
        remove_boilerplate=bool(raw["remove_boilerplate"]),
        deduplicate_exact=bool(raw["deduplicate_exact"]),
        merge_small_chunks=bool(raw["merge_small_chunks"]),
        restore_uncovered_pages=bool(raw["restore_uncovered_pages"]),
        min_chunk_tokens=int(raw["min_chunk_tokens"]),
        max_tokens=int(raw["max_tokens"]),
        boilerplate_min_occurrences=int(raw["boilerplate_min_occurrences"]),
        boilerplate_min_fraction=float(raw["boilerplate_min_fraction"]),
        boilerplate_max_line_characters=int(raw["boilerplate_max_line_characters"]),
        fallback_overlap_tokens=int(raw["fallback_overlap_tokens"]),
    )


class ProductionDocumentProcessor:
    """Compose the existing parser, cleaner, chunker and postprocessor."""

    def __init__(
        self,
        *,
        chunking_config_path: Path,
        postprocessing_config_path: Path,
    ) -> None:
        self.chunker = StructureAwareChunker(load_chunking_config(chunking_config_path))
        self.postprocessor = ChunkPostprocessor(
            config=load_postprocessing_config(postprocessing_config_path),
            token_counter=self.chunker,
        )

    @staticmethod
    def _clean_pages(
        pages: list[ParsedPage],
        *,
        document_id: str,
        filename: str,
    ) -> list[ParsedPage]:
        cleaned_pages: list[ParsedPage] = []

        for page in pages:
            plain_text = clean_pdf_text(page.content.plain_text)
            markdown = clean_pdf_text(page.content.markdown)
            effective_text = plain_text or markdown
            content = page.content.model_copy(
                update={
                    "plain_text": plain_text,
                    "markdown": markdown,
                }
            )
            provenance = page.provenance.model_copy(
                update={
                    "document_id": document_id,
                    "source_file": filename,
                }
            )
            quality = page.quality.model_copy(
                update={
                    "character_count": len(effective_text),
                    "word_count": len(effective_text.split()),
                    "is_empty": not effective_text,
                    "needs_review": (
                        page.quality.needs_review
                        or needs_manual_review(
                            page.content.plain_text,
                            effective_text,
                        )
                    ),
                }
            )
            cleaned_pages.append(
                page.model_copy(
                    update={
                        "content": content,
                        "provenance": provenance,
                        "quality": quality,
                    }
                )
            )

        return cleaned_pages

    def process(
        self,
        *,
        pdf_path: Path,
        document_id: str,
        document_sha256: str,
        cache_directory: Path,
    ) -> ProcessedDocument:
        """Process a new or modified PDF and retain no empty chunk."""

        extracted_pages = extract_pdf_pages(pdf_path)
        parsed_pages = [
            ParsedPage(
                content=PageContent(
                    plain_text=page.text,
                    markdown=page.text,
                ),
                provenance=PageProvenance(
                    source_file=pdf_path.name,
                    document_id=document_id,
                    page_number=page.page_number,
                    parser="pymupdf",
                    ocr_used=False,
                ),
                quality=PageQuality(
                    character_count=len(page.text),
                    word_count=len(page.text.split()),
                    is_empty=page.is_empty,
                    needs_review=page.is_empty,
                    warnings=(["empty_page"] if page.is_empty else []),
                ),
            )
            for page in extracted_pages
        ]
        pages = self._clean_pages(
            parsed_pages,
            document_id=document_id,
            filename=pdf_path.name,
        )

        if not pages:
            raise ValueError(f"Le PDF ne contient aucune page : {pdf_path}")

        raw_chunks = [chunk for chunk in self.chunker.chunk_document(pages) if chunk.text.strip()]

        if not raw_chunks:
            raise ValueError(f"Aucun chunk exploitable extrait de {pdf_path.name}.")

        result = self.postprocessor.process(
            chunks=raw_chunks,
            pages=pages,
        )
        chunks = tuple(chunk for chunk in result.chunks if chunk.text.strip())

        if not chunks:
            raise ValueError(f"Aucun chunk final exploitable pour {pdf_path.name}.")

        chunk_ids = [chunk.chunk_id for chunk in chunks]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(f"chunk_id dupliqué dans {pdf_path.name}.")

        return ProcessedDocument(
            filename=pdf_path.name,
            document_id=document_id,
            document_sha256=document_sha256,
            page_count=len(pages),
            empty_pages=tuple(
                page.provenance.page_number for page in pages if page.quality.is_empty
            ),
            chunks=chunks,
            duplicates_removed=int(result.statistics["duplicates_removed"]),
            ingestion_date=datetime.now(UTC).isoformat(),
            cache_directory=cache_directory,
        )


class VersionIndexBuilder:
    """Build immutable dense and lexical indexes from active chunks."""

    def __init__(
        self,
        *,
        embedding_config_path: Path,
        retrieval_config_path: Path,
        embedder_factory: (Callable[[EmbeddingConfig], Any] | None) = None,
        runtime_validation: bool = True,
        legacy_dense_directory: Path | None = None,
        embedding_cache_path: Path | None = None,
    ) -> None:
        self.embedding_config_path = embedding_config_path.resolve()
        self.retrieval_config_path = retrieval_config_path.resolve()
        self.embedding_config = load_embedding_config(self.embedding_config_path)
        self.bm25_config = load_bm25_config(self.retrieval_config_path)
        self.embedder_factory = embedder_factory or BGEEmbedder
        self.runtime_validation = runtime_validation
        self.legacy_dense_directory = (
            legacy_dense_directory.resolve() if legacy_dense_directory is not None else None
        )
        self.embedding_cache_path = (
            embedding_cache_path.resolve() if embedding_cache_path is not None else None
        )

    @staticmethod
    def _embedding_cache_key(record: dict[str, Any]) -> str:
        return hashlib.sha256(str(record["embedding_text"]).encode("utf-8")).hexdigest()

    def _load_persistent_vector_cache(
        self,
    ) -> dict[str, np.ndarray]:
        path = self.embedding_cache_path

        if path is None or not path.is_file():
            return {}

        with sqlite3.connect(path) as connection:
            rows = connection.execute(
                """
                SELECT cache_key, vector
                FROM embeddings
                WHERE model_name = ? AND dimension = ?
                """,
                (
                    self.embedding_config.model_name,
                    self.embedding_config.embedding_dimension,
                ),
            ).fetchall()

        cache: dict[str, np.ndarray] = {}

        for cache_key, blob in rows:
            vector = np.frombuffer(blob, dtype=np.float32).copy()

            if vector.shape == (self.embedding_config.embedding_dimension,):
                cache[str(cache_key)] = vector

        return cache

    def _save_persistent_vectors(
        self,
        records: list[dict[str, Any]],
        positions: list[int],
        vectors: np.ndarray,
    ) -> None:
        path = self.embedding_cache_path

        if path is None or not positions:
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY(cache_key, model_name, dimension)
                )
                """
            )
            connection.executemany(
                """
                INSERT OR REPLACE INTO embeddings (
                    cache_key, model_name, dimension, vector
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        self._embedding_cache_key(records[position]),
                        self.embedding_config.model_name,
                        self.embedding_config.embedding_dimension,
                        sqlite3.Binary(
                            np.asarray(
                                vectors[row],
                                dtype=np.float32,
                            ).tobytes()
                        ),
                    )
                    for row, position in enumerate(positions)
                ],
            )

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []

        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise ValueError(f"{path.name}, ligne {line_number} vide.")

                value = json.loads(line)

                if not isinstance(value, dict):
                    raise ValueError(f"{path.name}, ligne {line_number} invalide.")

                records.append(value)

        return records

    def _load_vector_cache(
        self,
        version_directory: Path | None,
    ) -> dict[tuple[str, str], tuple[str, np.ndarray]]:
        if version_directory is None:
            return {}

        dense_directory = version_directory / "dense"
        embeddings_path = dense_directory / "embeddings.npy"
        metadata_path = dense_directory / "metadata.jsonl"

        if not embeddings_path.is_file() or not metadata_path.is_file():
            return {}

        vectors = np.load(embeddings_path, allow_pickle=False)
        records = self._read_jsonl(metadata_path)

        if vectors.shape != (
            len(records),
            self.embedding_config.embedding_dimension,
        ):
            return {}

        return {
            (
                str(record["chunk_id"]),
                str(record.get("chunk_sha256", "")),
            ): (
                str(record["embedding_text"]),
                np.asarray(vectors[index], dtype=np.float32),
            )
            for index, record in enumerate(records)
        }

    def _load_legacy_vector_cache(
        self,
    ) -> dict[str, tuple[str, np.ndarray]]:
        directory = self.legacy_dense_directory

        if directory is None:
            return {}

        index_path = directory / "index.faiss"
        metadata_path = directory / "metadata.jsonl"

        if not index_path.is_file() or not metadata_path.is_file():
            return {}

        index = faiss.read_index(str(index_path))
        records = self._read_jsonl(metadata_path)

        if (
            int(index.ntotal) != len(records)
            or int(index.d) != self.embedding_config.embedding_dimension
        ):
            return {}

        vectors = index.reconstruct_n(0, int(index.ntotal))
        return {
            str(record["chunk_id"]): (
                str(record["embedding_text"]),
                np.asarray(vectors[position], dtype=np.float32),
            )
            for position, record in enumerate(records)
        }

    def _resolve_vectors(
        self,
        records: list[dict[str, Any]],
        *,
        previous_version_directory: Path | None,
    ) -> tuple[np.ndarray, int, int]:
        current_cache = self._load_vector_cache(previous_version_directory)
        legacy_cache = self._load_legacy_vector_cache()
        persistent_cache = self._load_persistent_vector_cache()
        dimension = self.embedding_config.embedding_dimension
        vectors = np.empty(
            (len(records), dimension),
            dtype=np.float32,
        )
        missing_positions: list[int] = []
        reused = 0

        for position, record in enumerate(records):
            key = (
                str(record["chunk_id"]),
                str(record["chunk_sha256"]),
            )
            cached = current_cache.get(key)

            if cached is not None and cached[0] == record["embedding_text"]:
                vectors[position] = cached[1]
                reused += 1
                continue

            legacy = legacy_cache.get(str(record["chunk_id"]))

            if legacy is not None and legacy[0] == record["embedding_text"]:
                vectors[position] = legacy[1]
                reused += 1
                continue

            persistent = persistent_cache.get(self._embedding_cache_key(record))

            if persistent is not None:
                vectors[position] = persistent
                reused += 1
                continue

            missing_positions.append(position)

        if missing_positions:
            embedder = self.embedder_factory(self.embedding_config)
            new_vectors = embedder.embed_documents(
                [str(records[position]["embedding_text"]) for position in missing_positions]
            )

            if new_vectors.shape != (len(missing_positions), dimension):
                raise ValueError("Le modèle d'embeddings a retourné une forme invalide.")

            if self.embedding_config.normalize_embeddings:
                new_norms = np.linalg.norm(
                    new_vectors,
                    axis=1,
                    keepdims=True,
                )

                if np.any(new_norms <= 0):
                    raise ValueError("Un nouvel embedding possède une norme nulle.")

                new_vectors = new_vectors / new_norms

            for row, position in enumerate(missing_positions):
                vectors[position] = new_vectors[row]

            self._save_persistent_vectors(
                records,
                missing_positions,
                new_vectors,
            )

        if not np.isfinite(vectors).all():
            raise ValueError("Les embeddings contiennent une valeur non finie.")

        norms = np.linalg.norm(vectors, axis=1)

        if np.any(norms <= 0):
            raise ValueError("Un embedding possède une norme nulle.")

        if self.embedding_config.normalize_embeddings:
            vectors = vectors / norms[:, None]

        return (
            np.ascontiguousarray(vectors, dtype=np.float32),
            len(missing_positions),
            reused,
        )

    @staticmethod
    def _write_jsonl(
        records: Iterable[dict[str, Any]],
        path: Path,
    ) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_json(payload: dict[str, Any], path: Path) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _build_dense(
        self,
        records: list[dict[str, Any]],
        vectors: np.ndarray,
        dense_directory: Path,
        *,
        document_counts: dict[str, int],
        embedded_count: int,
        reused_count: int,
    ) -> None:
        dense_directory.mkdir(parents=True, exist_ok=False)
        index = faiss.IndexFlatIP(self.embedding_config.embedding_dimension)
        index.add(vectors)
        faiss.write_index(
            index,
            str(dense_directory / "index.faiss"),
        )
        np.save(dense_directory / "embeddings.npy", vectors)
        self._write_jsonl(
            ({"vector_id": position, **record} for position, record in enumerate(records)),
            dense_directory / "metadata.jsonl",
        )
        self._write_json(
            {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "pipeline_version": KNOWLEDGE_BASE_PIPELINE_VERSION,
                "model": {
                    "name": self.embedding_config.model_name,
                    "dimension": self.embedding_config.embedding_dimension,
                    "normalize_embeddings": (self.embedding_config.normalize_embeddings),
                },
                "index": {
                    "library": "faiss",
                    "type": "IndexFlatIP",
                    "metric": "inner_product",
                    "dimension": int(index.d),
                    "total_vectors": int(index.ntotal),
                },
                "corpus": {
                    "total_documents": len(document_counts),
                    "total_chunks": len(records),
                    "chunks_per_document": document_counts,
                },
                "embedding_cache": {
                    "generated": embedded_count,
                    "reused": reused_count,
                },
            },
            dense_directory / "manifest.json",
        )

    def _build_bm25(
        self,
        records: list[dict[str, Any]],
        bm25_directory: Path,
        *,
        document_counts: dict[str, int],
    ) -> None:
        bm25_directory.mkdir(parents=True, exist_ok=False)
        chunks = [DocumentChunk.model_validate(record) for record in records]
        tokenized = [
            technical_tokenize(str(record.get("bm25_text") or build_lexical_text(chunk)))
            for record, chunk in zip(records, chunks, strict=True)
        ]

        if any(not tokens for tokens in tokenized):
            raise ValueError("Un chunk ne contient aucun token BM25.")

        model = bm25s.BM25(
            method=self.bm25_config.method,
            k1=self.bm25_config.k1,
            b=self.bm25_config.b,
            backend=self.bm25_config.backend,
            csc_backend=self.bm25_config.csc_backend,
        )
        model.index(
            tokenized,
            show_progress=False,
            leave_progress=False,
        )
        model.save(str(bm25_directory), show_progress=False)
        self._write_jsonl(
            ({"lexical_id": position, **record} for position, record in enumerate(records)),
            bm25_directory / self.bm25_config.metadata_filename,
        )
        self._write_json(
            {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "pipeline_version": KNOWLEDGE_BASE_PIPELINE_VERSION,
                "library": {
                    "name": "bm25s",
                    "version": package_version("bm25s"),
                },
                "bm25": {
                    "method": self.bm25_config.method,
                    "k1": self.bm25_config.k1,
                    "b": self.bm25_config.b,
                    "backend": self.bm25_config.backend,
                    "csc_backend": self.bm25_config.csc_backend,
                },
                "tokenizer": {
                    "version": TOKENIZER_VERSION,
                    "normalization": "NFKC + casefold + HTML cleanup",
                    "stemming": False,
                    "stopwords_removed": False,
                },
                "corpus": {
                    "total_documents": len(document_counts),
                    "total_chunks": len(records),
                    "chunks_per_document": document_counts,
                },
                "index_statistics": {
                    "documents": int(model.scores["num_docs"]),
                },
            },
            bm25_directory / self.bm25_config.manifest_filename,
        )

    def _validate(
        self,
        version_directory: Path,
        records: list[dict[str, Any]],
        *,
        active_document_ids: set[str],
    ) -> tuple[bool, bool, bool]:
        dense_directory = version_directory / "dense"
        bm25_directory = version_directory / "bm25"
        index = faiss.read_index(str(dense_directory / "index.faiss"))
        dense_records = self._read_jsonl(dense_directory / "metadata.jsonl")
        bm25_records = self._read_jsonl(bm25_directory / self.bm25_config.metadata_filename)
        embeddings = np.load(
            dense_directory / "embeddings.npy",
            allow_pickle=False,
        )
        expected_ids = [str(record["chunk_id"]) for record in records]

        if (
            int(index.ntotal) != len(records)
            or int(index.d) != self.embedding_config.embedding_dimension
            or embeddings.shape
            != (
                len(records),
                self.embedding_config.embedding_dimension,
            )
            or [str(record["chunk_id"]) for record in dense_records] != expected_ids
            or [str(record["chunk_id"]) for record in bm25_records] != expected_ids
            or len(expected_ids) != len(set(expected_ids))
        ):
            raise ValueError("Les index et métadonnées ne correspondent pas exactement.")

        indexed_document_ids = {str(record["document_id"]) for record in dense_records}

        if indexed_document_ids != active_document_ids:
            raise ValueError("Un document inactif est présent dans les index.")

        query_vector = np.ascontiguousarray(
            embeddings[0].reshape(1, -1),
            dtype=np.float32,
        )
        _, vector_ids = index.search(
            query_vector,
            min(5, len(records)),
        )
        dense_ok = bool(vector_ids.size and int(vector_ids[0][0]) >= 0)
        bm25 = BM25Retriever(
            index_directory=bm25_directory,
            config_path=self.retrieval_config_path,
        )
        bm25_response = bm25.search(
            SMOKE_QUERY,
            top_k=min(5, len(records)),
        )
        bm25_ok = bool(bm25_response.results)
        hybrid_ok = dense_ok and bm25_ok
        hybrid: HybridRetriever | None = None

        if self.runtime_validation:
            hybrid = HybridRetriever(
                dense_index_directory=dense_directory,
                bm25_index_directory=bm25_directory,
                embedding_config_path=self.embedding_config_path,
                retrieval_config_path=self.retrieval_config_path,
            )
            response = hybrid.search(
                SMOKE_QUERY,
                top_k=min(5, len(records)),
                dense_candidate_k=min(20, len(records)),
                bm25_candidate_k=min(20, len(records)),
                use_query_expansion=True,
            )
            hybrid_ok = bool(response.results)

            if any(
                result.chunk.document_id not in active_document_ids for result in response.results
            ):
                raise ValueError("Le retrieval hybride retourne un document inactif.")

        if not (dense_ok and bm25_ok and hybrid_ok):
            raise ValueError("Une recherche de validation dense/BM25/hybride a échoué.")

        del hybrid
        del bm25
        del index
        gc.collect()
        return dense_ok, bm25_ok, hybrid_ok

    def build(
        self,
        *,
        records: list[dict[str, Any]],
        version_directory: Path,
        previous_version_directory: Path | None,
    ) -> IndexBuildResult:
        """Build and validate one temporary immutable index version."""

        if not records:
            raise ValueError("Impossible de construire une base sans chunk.")

        chunks = [DocumentChunk.model_validate(record) for record in records]
        chunk_ids = [chunk.chunk_id for chunk in chunks]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Des chunk_id actifs sont dupliqués.")

        version_directory.mkdir(parents=True, exist_ok=False)
        corpus_directory = version_directory / "corpus"
        corpus_directory.mkdir()
        records_by_document: dict[str, list[dict[str, Any]]] = {}

        for record in records:
            records_by_document.setdefault(
                str(record["document_id"]),
                [],
            ).append(record)

        for document_id, document_records in sorted(records_by_document.items()):
            self._write_jsonl(
                document_records,
                corpus_directory / f"{document_id}_chunks.jsonl",
            )

        vectors, embedded_count, reused_count = self._resolve_vectors(
            records,
            previous_version_directory=previous_version_directory,
        )
        document_counts = dict(Counter(chunk.document_id for chunk in chunks))
        self._build_dense(
            records,
            vectors,
            version_directory / "dense",
            document_counts=document_counts,
            embedded_count=embedded_count,
            reused_count=reused_count,
        )
        self._build_bm25(
            records,
            version_directory / "bm25",
            document_counts=document_counts,
        )
        dense_ok, bm25_ok, hybrid_ok = self._validate(
            version_directory,
            records,
            active_document_ids=set(document_counts),
        )
        self._write_json(
            {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "pipeline_version": KNOWLEDGE_BASE_PIPELINE_VERSION,
                "document_count": len(document_counts),
                "chunk_count": len(records),
                "chunks_per_document": document_counts,
                "active_document_ids": sorted(document_counts),
                "validation": {
                    "faiss_readable": True,
                    "bm25_readable": True,
                    "embedding_dimension": (self.embedding_config.embedding_dimension),
                    "vector_chunk_alignment": True,
                    "unique_chunk_ids": True,
                    "inactive_documents_absent": True,
                    "dense_search": dense_ok,
                    "bm25_search": bm25_ok,
                    "hybrid_search": hybrid_ok,
                },
            },
            version_directory / "manifest.json",
        )
        return IndexBuildResult(
            chunk_count=len(records),
            document_counts=document_counts,
            embedded_chunk_count=embedded_count,
            reused_embedding_count=reused_count,
            dense_search_ok=dense_ok,
            bm25_search_ok=bm25_ok,
            hybrid_search_ok=hybrid_ok,
        )
