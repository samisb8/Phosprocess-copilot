"""Persistent BGE-M3 sparse-index invariants."""

from __future__ import annotations

from pathlib import Path

from phosprocess.ingestion.chunk_serialization import (
    TechnicalChildChunk,
    TechnicalChunkType,
    write_jsonl,
)
from phosprocess.preprocessing.chunk_schemas import DocumentChunk
from phosprocess.retrieval.bge_sparse import (
    BGESparseRetriever,
    build_bge_sparse_index,
)


class _FakeSparseEmbedder:
    model_name = "fake-bge-m3"

    @staticmethod
    def _weights(text: str) -> dict[int, float]:
        normalized = text.casefold()
        weights: dict[int, float] = {}
        if "pump" in normalized or "pompe" in normalized:
            weights[11] = 3.0
        if "falling film" in normalized:
            weights[17] = 4.0
        if "filter" in normalized or "filtration" in normalized:
            weights[23] = 2.0
        return weights

    def embed_sparse_documents(self, texts: list[str]) -> list[dict[int, float]]:
        return [self._weights(text) for text in texts]

    def embed_sparse_query(self, query: str) -> dict[int, float]:
        return self._weights(query)


def _technical(index: int, text: str) -> TechnicalChildChunk:
    return TechnicalChildChunk(
        chunk_id=f"chunk_{index}",
        parent_id=f"parent_{index}",
        document_id="doc",
        document_title="Document",
        source_file="doc.pdf",
        domains=("equipment",),
        section="Section",
        chunk_type=TechnicalChunkType.EQUIPMENT_DESCRIPTION,
        page_start=index + 1,
        page_end=index + 1,
        text=text,
        display_text=text,
        embedding_text=f"Document: Document\nSection: Section\n{text}",
        bm25_text=text,
        token_count=20,
        sha256=f"{index + 1:064x}",
    )


def _runtime(index: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"chunk_{index}",
        document_id="doc",
        source_file="doc.pdf",
        chunk_index=index,
        source_pages=[index + 1],
        page_start=index + 1,
        page_end=index + 1,
        text=text,
        embedding_text=f"Document: Document\nSection: Section\n{text}",
        body_token_count=20,
        token_count=20,
        document_title="Document",
        section="Section",
        active=True,
    )


def test_sparse_index_uses_existing_embedding_text_and_preserves_alignment(
    tmp_path: Path,
) -> None:
    texts = [
        "The circulation pump moves liquid through the heater.",
        "A falling film evaporator distributes a thin liquid film.",
        "This passage discusses filtration only.",
    ]
    write_jsonl(
        tmp_path / "chunks.jsonl",
        [_technical(index, text) for index, text in enumerate(texts)],
    )
    dense_directory = tmp_path / "dense"
    dense_directory.mkdir()
    write_jsonl(
        dense_directory / "metadata.jsonl",
        [_runtime(index, text) for index, text in enumerate(texts)],
    )
    embedder = _FakeSparseEmbedder()

    output = build_bge_sparse_index(
        version_directory=tmp_path,
        embedder=embedder,  # type: ignore[arg-type]
        batch_documents=2,
    )
    retriever = BGESparseRetriever(
        version_directory=tmp_path,
        embedder=embedder,  # type: ignore[arg-type]
    )
    response = retriever.search("circulation pump", top_k=2)

    assert output == tmp_path / "bge_sparse"
    assert (output / "matrix.npz").is_file()
    assert (output / "manifest.json").is_file()
    assert response.results[0].chunk.chunk_id == "chunk_0"
    assert response.results[0].score > 0.0
    assert [chunk.chunk_id for chunk in retriever.metadata] == [
        "chunk_0",
        "chunk_1",
        "chunk_2",
    ]


def test_sparse_index_reorders_corpus_to_match_dense_metadata(
    tmp_path: Path,
) -> None:
    texts = [
        "The circulation pump moves liquid through the heater.",
        "A falling film evaporator distributes a thin liquid film.",
        "This passage discusses filtration only.",
    ]
    technical = [_technical(index, text) for index, text in enumerate(texts)]
    runtime = [_runtime(index, text) for index, text in enumerate(texts)]
    write_jsonl(
        tmp_path / "chunks.jsonl",
        [technical[2], technical[0], technical[1]],
    )
    dense_directory = tmp_path / "dense"
    dense_directory.mkdir()
    write_jsonl(
        dense_directory / "metadata.jsonl",
        runtime,
    )
    embedder = _FakeSparseEmbedder()

    build_bge_sparse_index(
        version_directory=tmp_path,
        embedder=embedder,  # type: ignore[arg-type]
        batch_documents=2,
    )
    retriever = BGESparseRetriever(
        version_directory=tmp_path,
        embedder=embedder,  # type: ignore[arg-type]
    )

    pump = retriever.search("circulation pump", top_k=1)
    falling_film = retriever.search("falling film", top_k=1)

    assert pump.results[0].chunk.chunk_id == "chunk_0"
    assert falling_film.results[0].chunk.chunk_id == "chunk_1"


def test_sparse_index_rejects_different_chunk_id_sets(tmp_path: Path) -> None:
    write_jsonl(
        tmp_path / "chunks.jsonl",
        [_technical(0, "The circulation pump moves liquid.")],
    )
    dense_directory = tmp_path / "dense"
    dense_directory.mkdir()
    write_jsonl(
        dense_directory / "metadata.jsonl",
        [_runtime(1, "A falling film evaporator.")],
    )

    try:
        build_bge_sparse_index(
            version_directory=tmp_path,
            embedder=_FakeSparseEmbedder(),  # type: ignore[arg-type]
        )
    except ValueError as exc:
        assert "même ensemble de chunk_id" in str(exc)
    else:
        raise AssertionError("Un corpus réellement désaligné doit être refusé.")
