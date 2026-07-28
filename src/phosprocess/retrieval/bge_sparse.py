"""Persistent BGE-M3 sparse retrieval over the active quality corpus."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from phosprocess.embeddings.embedder import BGEEmbedder
from phosprocess.ingestion.chunk_serialization import (
    TechnicalChildChunk,
    read_child_chunks,
)
from phosprocess.preprocessing.chunk_schemas import DocumentChunk

SPARSE_INDEX_VERSION = "bge_m3_sparse_v1"


@dataclass(frozen=True, slots=True)
class SparseSearchResult:
    """One BGE-M3 lexical retrieval result."""

    rank: int
    score: float
    chunk: DocumentChunk


@dataclass(frozen=True, slots=True)
class SparseSearchResponse:
    """Trace of one sparse retrieval call."""

    query: str
    top_k_requested: int
    search_duration_ms: float
    results: list[SparseSearchResult]


def _load_dense_metadata(path: Path) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"Ligne vide dans {path.name}: {line_number}.")
            chunks.append(DocumentChunk.model_validate_json(line))
    return chunks


def _chunk_id_digest(
    chunks: Iterable[DocumentChunk | TechnicalChildChunk],
) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def build_bge_sparse_index(
    *,
    version_directory: Path,
    embedder: BGEEmbedder,
    batch_documents: int = 64,
    force: bool = False,
) -> Path:
    """Build a CSR lexical-weight matrix without changing chunks or dense vectors."""

    version_directory = version_directory.resolve()
    chunks_path = version_directory / "chunks.jsonl"
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Corpus qualité introuvable : {chunks_path}")
    if batch_documents <= 0:
        raise ValueError("batch_documents doit être strictement positif.")

    output = version_directory / "bge_sparse"
    matrix_path = output / "matrix.npz"
    manifest_path = output / "manifest.json"
    if not force and matrix_path.is_file() and manifest_path.is_file():
        raise FileExistsError(
            "L'index BGE sparse existe déjà. Utilisez --force pour le reconstruire."
        )

    chunks = read_child_chunks(chunks_path)
    if not chunks:
        raise ValueError("Le corpus qualité est vide.")
    dense_metadata_path = version_directory / "dense" / "metadata.jsonl"
    if not dense_metadata_path.is_file():
        raise FileNotFoundError(
            f"Métadonnées denses introuvables : {dense_metadata_path}"
        )
    dense_metadata = _load_dense_metadata(dense_metadata_path)
    child_by_id: dict[str, TechnicalChildChunk] = {}
    duplicate_child_ids: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in child_by_id:
            duplicate_child_ids.add(chunk.chunk_id)
        child_by_id[chunk.chunk_id] = chunk
    if duplicate_child_ids:
        examples = ", ".join(sorted(duplicate_child_ids)[:5])
        raise ValueError(
            "chunks.jsonl contient des chunk_id dupliqués; "
            f"exemples: {examples}."
        )

    dense_ids = [chunk.chunk_id for chunk in dense_metadata]
    if len(dense_ids) != len(set(dense_ids)):
        raise ValueError("dense/metadata.jsonl contient des chunk_id dupliqués.")

    child_ids = set(child_by_id)
    dense_id_set = set(dense_ids)
    missing_from_chunks = dense_id_set - child_ids
    missing_from_dense = child_ids - dense_id_set
    if missing_from_chunks or missing_from_dense:
        raise ValueError(
            "chunks.jsonl et dense/metadata.jsonl ne contiennent pas le même "
            "ensemble de chunk_id; "
            f"absents de chunks.jsonl={len(missing_from_chunks)}, "
            f"absents de dense/metadata.jsonl={len(missing_from_dense)}."
        )

    # The sparse matrix must use the exact row order of the active dense metadata.
    # The corpus file may contain the same chunks in another deterministic order.
    aligned_chunks = [child_by_id[chunk_id] for chunk_id in dense_ids]

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    maximum_token_id = -1
    started = time.perf_counter()

    for start in range(0, len(aligned_chunks), batch_documents):
        batch = aligned_chunks[start : start + batch_documents]
        weights_batch = embedder.embed_sparse_documents(
            [chunk.embedding_text for chunk in batch]
        )
        for offset, weights in enumerate(weights_batch):
            row = start + offset
            for token_id, weight in weights.items():
                row_indices.append(row)
                column_indices.append(token_id)
                values.append(weight)
                maximum_token_id = max(maximum_token_id, token_id)
        print(
            "BGE sparse: "
            f"{min(start + len(batch), len(aligned_chunks))}/"
            f"{len(aligned_chunks)} chunks",
            flush=True,
        )

    if maximum_token_id < 0 or not values:
        raise ValueError("BGE-M3 n'a produit aucun poids lexical positif.")

    matrix = sparse.csr_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (
                np.asarray(row_indices, dtype=np.int32),
                np.asarray(column_indices, dtype=np.int32),
            ),
        ),
        shape=(len(aligned_chunks), maximum_token_id + 1),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()

    output.mkdir(parents=True, exist_ok=True)
    temporary_matrix = output / "matrix.tmp.npz"
    sparse.save_npz(temporary_matrix, matrix, compressed=True)
    temporary_matrix.replace(matrix_path)
    manifest = {
        "index_version": SPARSE_INDEX_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "model_name": embedder.model_name,
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "non_zero_values": int(matrix.nnz),
        "chunk_id_sha256": _chunk_id_digest(aligned_chunks),
        "source_chunks": "../chunks.jsonl",
        "runtime_metadata": "../dense/metadata.jsonl",
        "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    _write_json_atomic(manifest_path, manifest)
    return output


class BGESparseRetriever:
    """Search a persisted BGE-M3 sparse matrix aligned with chunks.jsonl."""

    def __init__(
        self,
        *,
        version_directory: Path,
        embedder: BGEEmbedder,
        metadata: list[DocumentChunk] | None = None,
    ) -> None:
        self.version_directory = version_directory.resolve()
        self.index_directory = self.version_directory / "bge_sparse"
        self.matrix_path = self.index_directory / "matrix.npz"
        self.manifest_path = self.index_directory / "manifest.json"
        if not self.matrix_path.is_file() or not self.manifest_path.is_file():
            raise FileNotFoundError(
                "Index BGE sparse absent. Exécutez scripts/build_bge_sparse_index.py."
            )
        self.embedder = embedder
        self.metadata = metadata or _load_dense_metadata(
            self.version_directory / "dense" / "metadata.jsonl"
        )
        self.matrix = sparse.load_npz(self.matrix_path).tocsr().astype(np.float32)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._validate_alignment()

    @classmethod
    def is_available(cls, version_directory: Path) -> bool:
        root = version_directory / "bge_sparse"
        return (root / "matrix.npz").is_file() and (root / "manifest.json").is_file()

    def _validate_alignment(self) -> None:
        if self.manifest.get("index_version") != SPARSE_INDEX_VERSION:
            raise ValueError("Version d'index BGE sparse non prise en charge.")
        if self.manifest.get("model_name") != self.embedder.model_name:
            raise ValueError(
                "L'index BGE sparse a été construit avec un autre modèle."
            )
        if self.matrix.shape[0] != len(self.metadata):
            raise ValueError("L'index BGE sparse et les métadonnées dense sont désalignés.")
        expected_digest = _chunk_id_digest(self.metadata)
        if self.manifest.get("chunk_id_sha256") != expected_digest:
            raise ValueError("Le manifeste BGE sparse ne correspond pas aux chunk IDs actifs.")

    def search(
        self,
        query: str,
        *,
        top_k: int,
        document_ids: set[str] | None = None,
        chunk_ids: set[str] | None = None,
        minimum_score: float = 0.0,
    ) -> SparseSearchResponse:
        """Rank chunks by the dot product of BGE-M3 lexical weights."""

        cleaned = query.strip()
        if not cleaned:
            raise ValueError("La requête sparse ne peut pas être vide.")
        if top_k <= 0:
            raise ValueError("top_k doit être strictement positif.")

        started = time.perf_counter()
        query_weights = self.embedder.embed_sparse_query(cleaned)
        valid_items = [
            (token_id, value)
            for token_id, value in query_weights.items()
            if 0 <= token_id < self.matrix.shape[1] and value > 0.0
        ]
        if not valid_items:
            return SparseSearchResponse(cleaned, top_k, 0.0, [])

        columns = np.asarray([item[0] for item in valid_items], dtype=np.int32)
        values = np.asarray([item[1] for item in valid_items], dtype=np.float32)
        query_vector = sparse.csc_matrix(
            (values, (columns, np.zeros_like(columns))),
            shape=(self.matrix.shape[1], 1),
            dtype=np.float32,
        )
        scores = np.asarray((self.matrix @ query_vector).toarray()).reshape(-1)

        allowed_documents = {value.strip() for value in document_ids or set() if value.strip()}
        allowed_chunks = {value.strip() for value in chunk_ids or set() if value.strip()}
        eligible = []
        for index, score in enumerate(scores):
            value = float(score)
            if value <= minimum_score:
                continue
            chunk = self.metadata[index]
            if not chunk.active:
                continue
            if allowed_documents and chunk.document_id not in allowed_documents:
                continue
            if allowed_chunks and chunk.chunk_id not in allowed_chunks:
                continue
            eligible.append((index, value))

        eligible.sort(key=lambda item: (-item[1], self.metadata[item[0]].chunk_id))
        results = [
            SparseSearchResult(rank + 1, score, self.metadata[index])
            for rank, (index, score) in enumerate(eligible[:top_k])
        ]
        return SparseSearchResponse(
            query=cleaned,
            top_k_requested=top_k,
            search_duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            results=results,
        )
