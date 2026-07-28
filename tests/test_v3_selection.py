"""Tests for the DEV-designed v3 lexical safeguard."""

from __future__ import annotations

import inspect
import re
from dataclasses import replace

from phosprocess.preprocessing.chunk_schemas import DocumentChunk
from phosprocess.reranking.reranker import RerankedSearchResult
from phosprocess.retrieval.hybrid import HybridSearchResult
from phosprocess.retrieval.v3_selection import (
    select_with_lexical_safeguard,
)


def make_chunk(number: int) -> DocumentChunk:
    """Create a minimal chunk accepted by the retrieval schemas."""

    return DocumentChunk(
        chunk_id=f"chunk_{number:02d}",
        document_id="document",
        source_file="document.pdf",
        chunk_index=number,
        heading_path=[],
        source_pages=[1],
        page_start=1,
        page_end=1,
        content_types=["paragraph"],
        text=f"Passage {number}",
        embedding_text=f"Passage {number}",
        body_token_count=2,
        token_count=2,
        source_chunk_ids=[f"source_{number:02d}"],
        postprocessing_actions=[],
    )


def make_candidates() -> list[HybridSearchResult]:
    """Create six hybrid candidates with chunk 6 as BM25 rank 1."""

    candidates: list[HybridSearchResult] = []

    for number in range(1, 7):
        bm25_rank = 1 if number == 6 else number + 1
        candidates.append(
            HybridSearchResult(
                rank=number,
                rrf_score=1.0 / number,
                matched_retrievers=("dense", "bm25"),
                dense_rank=number,
                dense_score=1.0 / number,
                dense_rrf_contribution=0.0,
                bm25_rank=bm25_rank,
                bm25_score=1.0 / bm25_rank,
                bm25_rrf_contribution=0.0,
                chunk=make_chunk(number),
            )
        )

    return candidates


def make_reranked(
    candidates: list[HybridSearchResult],
) -> list[RerankedSearchResult]:
    """Create a reranker order equal to the hybrid order."""

    return [
        RerankedSearchResult(
            rank=rank,
            reranker_score=1.0 / rank,
            original_hybrid_rank=candidate.rank,
            original_rrf_score=candidate.rrf_score,
            matched_retrievers=candidate.matched_retrievers,
            dense_rank=candidate.dense_rank,
            dense_score=candidate.dense_score,
            bm25_rank=candidate.bm25_rank,
            bm25_score=candidate.bm25_score,
            chunk=candidate.chunk,
        )
        for rank, candidate in enumerate(candidates, start=1)
    ]


def test_lexical_safeguard_reserves_the_last_slot() -> None:
    candidates = make_candidates()
    reranked = make_reranked(candidates)

    selected = select_with_lexical_safeguard(
        candidates,
        reranked,
        top_k=5,
    )

    assert [result.chunk_id for result in selected] == [
        "chunk_01",
        "chunk_02",
        "chunk_03",
        "chunk_04",
        "chunk_06",
    ]
    assert selected[-1].source == "bm25_safeguard"
    assert selected[-1].reranker_rank == 6
    assert selected[-1].bm25_rank == 1


def test_duplicate_lexical_result_is_filled_from_reranker() -> None:
    candidates = make_candidates()
    candidates[0] = replace(
        candidates[0],
        bm25_rank=1,
    )
    candidates[-1] = replace(
        candidates[-1],
        bm25_rank=7,
    )
    reranked = make_reranked(candidates)

    selected = select_with_lexical_safeguard(
        candidates,
        reranked,
        top_k=5,
    )

    assert [result.chunk_id for result in selected] == [
        "chunk_01",
        "chunk_02",
        "chunk_03",
        "chunk_04",
        "chunk_05",
    ]
    assert selected[-1].source == "reranker_fill"


def test_unknown_reranked_chunk_is_rejected() -> None:
    candidates = make_candidates()
    reranked = make_reranked(candidates)
    reranked[0] = replace(
        reranked[0],
        chunk=make_chunk(99),
    )

    try:
        select_with_lexical_safeguard(
            candidates,
            reranked,
            top_k=5,
        )
    except ValueError as error:
        assert "absents des candidats" in str(error)
    else:
        raise AssertionError("Un chunk inconnu aurait dû être refusé.")


def test_selection_is_deterministic() -> None:
    candidates = make_candidates()
    reranked = make_reranked(candidates)
    expected = select_with_lexical_safeguard(
        candidates,
        reranked,
        top_k=5,
        lexical_slots=1,
    )

    for _ in range(10):
        assert (
            select_with_lexical_safeguard(
                candidates,
                reranked,
                top_k=5,
                lexical_slots=1,
            )
            == expected
        )


def test_selection_has_no_query_specific_rule() -> None:
    source = inspect.getsource(select_with_lexical_safeguard)

    assert re.search(r"\bQ\d{3}\b", source) is None
    assert re.search(r"\b[0-9]{2}_[a-z0-9_]+_[0-9]{6}_[0-9a-f]{12}\b", source) is None


def test_selection_does_not_use_gold_or_reference_answers() -> None:
    signature = inspect.signature(select_with_lexical_safeguard)
    source = inspect.getsource(
        select_with_lexical_safeguard
    ).casefold()

    assert set(signature.parameters) == {
        "candidates",
        "reranked_results",
        "top_k",
        "lexical_slots",
    }

    for forbidden_name in (
        "gold",
        "reference_answer",
        "expected_answer",
        "query_id",
    ):
        assert forbidden_name not in source


def test_selection_always_returns_five_unique_chunks() -> None:
    candidates = make_candidates()
    reranked = make_reranked(candidates)

    for lexical_slots in (0, 1, 2):
        selected = select_with_lexical_safeguard(
            candidates,
            reranked,
            top_k=5,
            lexical_slots=lexical_slots,
        )
        chunk_ids = [result.chunk_id for result in selected]

        assert len(chunk_ids) == 5
        assert len(set(chunk_ids)) == 5


def test_no_bm25_candidate_falls_back_to_reranker() -> None:
    candidates = [
        replace(
            candidate,
            bm25_rank=None,
            bm25_score=None,
        )
        for candidate in make_candidates()
    ]
    reranked = make_reranked(candidates)

    selected = select_with_lexical_safeguard(
        candidates,
        reranked,
        top_k=5,
        lexical_slots=1,
    )

    assert [result.chunk_id for result in selected] == [
        "chunk_01",
        "chunk_02",
        "chunk_03",
        "chunk_04",
        "chunk_05",
    ]
    assert selected[-1].source == "reranker_fill"
