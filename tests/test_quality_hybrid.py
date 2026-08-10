"""Independent dense/BM25 query tests for the quality runtime."""

from __future__ import annotations

from types import SimpleNamespace

from phosprocess.preprocessing.chunk_schemas import DocumentChunk
from phosprocess.retrieval.quality_hybrid import (
    search_expanded_hybrid,
    search_planned_hybrid,
)
from phosprocess.retrieval.query_expansion import expand_technical_query
from phosprocess.retrieval.retrieval_planner import build_retrieval_plan


def _chunk(chunk_id: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="book",
        source_file="book.pdf",
        chunk_index=0,
        source_pages=[1],
        page_start=1,
        page_end=1,
        text="Heat exchanger technical passage.",
        embedding_text="Document: Book\nHeat exchanger technical passage.",
        body_token_count=5,
        token_count=5,
    )


class _SearchRecorder:
    def __init__(self, results: list[object]) -> None:
        self.results = results
        self.query: str | None = None

    def search(self, query: str, **_kwargs: object) -> object:
        self.query = query
        return SimpleNamespace(results=self.results, search_duration_ms=1.0)


def test_quality_hybrid_uses_distinct_query_representations() -> None:
    chunk = _chunk("chunk")
    dense = _SearchRecorder(
        [SimpleNamespace(rank=1, score=0.8, chunk=chunk)]
    )
    bm25 = _SearchRecorder(
        [SimpleNamespace(rank=1, score=3.0, chunk=chunk)]
    )
    retriever = SimpleNamespace(
        dense_retriever=dense,
        bm25_retriever=bm25,
        config=SimpleNamespace(
            dense_weight=1.0,
            bm25_weight=1.0,
            rrf_k=60,
        ),
    )
    expanded = expand_technical_query(
        "Quel est le rôle de l’échangeur thermique ?"
    )

    response = search_expanded_hybrid(
        retriever,
        expanded,
        top_k=20,
        dense_candidate_k=20,
        bm25_candidate_k=20,
    )

    assert dense.query == expanded.dense_query
    assert bm25.query == expanded.bm25_expanded_query
    assert "heat exchanger" in bm25.query
    assert response.results[0].matched_retrievers == ("dense", "bm25")

class _ColbertEmbedder:
    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []

    @staticmethod
    def embed_colbert_query(query: str) -> object:
        return query

    def embed_colbert_documents(self, texts: list[str]) -> list[object]:
        self.document_calls.append(texts)
        return list(texts)

    @staticmethod
    def colbert_score(query: object, passage: object) -> float:
        query_tokens = set(str(query).casefold().split())
        passage_tokens = set(str(passage).casefold().split())
        return float(len(query_tokens & passage_tokens))


class _RoleSearchRecorder:
    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, int, object]] = []

    def search(self, query: str, *, top_k: int, chunk_ids: object, **_kwargs: object) -> object:
        self.calls.append((query, top_k, chunk_ids))
        ranked = sorted(
            self.chunks,
            key=lambda chunk: (
                -len(set(query.casefold().split()) & set(chunk.text.casefold().split())),
                chunk.chunk_id,
            ),
        )[:top_k]
        return SimpleNamespace(
            results=[
                SimpleNamespace(rank=rank, score=1.0 / rank, chunk=chunk)
                for rank, chunk in enumerate(ranked, start=1)
            ],
            search_duration_ms=1.0,
        )


def test_planned_hybrid_searches_every_role_globally_and_uses_colbert() -> None:
    chunks = [
        DocumentChunk(
            chunk_id="forced",
            document_id="book",
            source_file="book.pdf",
            chunk_index=0,
            source_pages=[1],
            page_start=1,
            page_end=1,
            text="forced circulation evaporator pump heat transfer",
            embedding_text="forced circulation evaporator pump heat transfer",
            body_token_count=6,
            token_count=6,
        ),
        DocumentChunk(
            chunk_id="falling",
            document_id="book",
            source_file="book.pdf",
            chunk_index=1,
            source_pages=[2],
            page_start=2,
            page_end=2,
            text="falling film evaporator thin film heat transfer",
            embedding_text="falling film evaporator thin film heat transfer",
            body_token_count=7,
            token_count=7,
        ),
        DocumentChunk(
            chunk_id="filter",
            document_id="book",
            source_file="book.pdf",
            chunk_index=2,
            source_pages=[3],
            page_start=3,
            page_end=3,
            text="phosphoric acid filtration cake washing",
            embedding_text="phosphoric acid filtration cake washing",
            body_token_count=5,
            token_count=5,
        ),
    ]
    dense = _RoleSearchRecorder(chunks)
    dense.embedder = _ColbertEmbedder()
    bm25 = _RoleSearchRecorder(chunks)
    sparse = _RoleSearchRecorder(chunks)
    retriever = SimpleNamespace(
        dense_retriever=dense,
        bm25_retriever=bm25,
        config=SimpleNamespace(dense_weight=1.0, bm25_weight=1.0, rrf_k=60),
    )
    question = (
        "Compare a forced-circulation evaporator with a falling-film "
        "evaporator for phosphoric acid concentration."
    )
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="comparison",
    )

    response = search_planned_hybrid(
        retriever,
        plan,
        sparse_retriever=sparse,
        top_k=3,
        dense_candidate_k=3,
        sparse_candidate_k=3,
        bm25_candidate_k=3,
        fusion_k=3,
        colbert_candidate_k=3,
        document_ids=None,
    )

    assert len(dense.calls) == len(plan.roles)
    assert len(bm25.calls) == len(plan.roles)
    assert len(sparse.calls) == len(plan.roles)
    assert all(chunk_ids is None for _query, _top_k, chunk_ids in dense.calls)
    assert response.sparse_results_found == 3 * len(plan.roles)
    assert all(result.colbert_score is not None for result in response.results)
    assert {result.chunk.chunk_id for result in response.results}.issubset(
        {chunk.chunk_id for chunk in chunks}
    )
