"""DEV-designed selection policies for retrieval v3 experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from phosprocess.reranking.reranker import RerankedSearchResult
from phosprocess.retrieval.hybrid import HybridSearchResult


@dataclass(frozen=True, slots=True)
class V3SelectedResult:
    """Selection metadata for one final v3 result."""

    rank: int
    chunk_id: str
    source: str
    reranker_rank: int | None
    hybrid_rank: int
    bm25_rank: int | None


def select_with_lexical_safeguard(
    candidates: Sequence[HybridSearchResult],
    reranked_results: Sequence[RerankedSearchResult],
    *,
    top_k: int = 5,
    lexical_slots: int = 1,
) -> list[V3SelectedResult]:
    """Keep reranker leaders and reserve final slots for strong BM25 hits."""

    if top_k <= 0:
        raise ValueError("top_k doit être strictement positif.")

    if lexical_slots < 0:
        raise ValueError("lexical_slots doit être positif ou nul.")

    if lexical_slots >= top_k:
        raise ValueError("lexical_slots doit être inférieur à top_k.")

    candidate_by_id = {candidate.chunk.chunk_id: candidate for candidate in candidates}
    reranked_by_id = {result.chunk.chunk_id: result for result in reranked_results}

    if len(candidate_by_id) != len(candidates):
        raise ValueError("Les candidats hybrides contiennent un doublon.")

    if len(reranked_by_id) != len(reranked_results):
        raise ValueError("Les résultats rerankés contiennent un doublon.")

    unknown_reranked_ids = set(reranked_by_id) - set(candidate_by_id)

    if unknown_reranked_ids:
        raise ValueError(
            "Le reranker contient des chunks absents des candidats hybrides: "
            + ", ".join(sorted(unknown_reranked_ids))
        )

    reranker_slots = top_k - lexical_slots
    selected_ids = [result.chunk.chunk_id for result in reranked_results[:reranker_slots]]
    selection_source = {chunk_id: "reranker" for chunk_id in selected_ids}
    lexical_candidates = sorted(
        (candidate for candidate in candidates if candidate.bm25_rank is not None),
        key=lambda candidate: (
            candidate.bm25_rank,
            candidate.rank,
            candidate.chunk.chunk_id,
        ),
    )

    for candidate in lexical_candidates[:lexical_slots]:
        chunk_id = candidate.chunk.chunk_id

        if chunk_id in selection_source:
            continue

        selected_ids.append(chunk_id)
        selection_source[chunk_id] = "bm25_safeguard"

    for result in reranked_results:
        if len(selected_ids) >= top_k:
            break

        chunk_id = result.chunk.chunk_id

        if chunk_id in selection_source:
            continue

        selected_ids.append(chunk_id)
        selection_source[chunk_id] = "reranker_fill"

    if len(selected_ids) < min(top_k, len(candidates)):
        raise ValueError(
            "Le reranker ne fournit pas assez de résultats pour compléter la sélection v3."
        )

    return [
        V3SelectedResult(
            rank=rank,
            chunk_id=chunk_id,
            source=selection_source[chunk_id],
            reranker_rank=(reranked_by_id[chunk_id].rank if chunk_id in reranked_by_id else None),
            hybrid_rank=candidate_by_id[chunk_id].rank,
            bm25_rank=candidate_by_id[chunk_id].bm25_rank,
        )
        for rank, chunk_id in enumerate(selected_ids, start=1)
    ]
