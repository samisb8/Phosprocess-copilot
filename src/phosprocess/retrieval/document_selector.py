"""Dynamic document ranking from global retrieval evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RankedDocument:
    """Evidence-based document score produced after global retrieval."""

    document_id: str
    reranker_reciprocal_rank_sum: float
    hybrid_reciprocal_rank_sum: float
    best_reranker_rank: int | None
    best_hybrid_rank: int | None
    reranker_hits: int
    hybrid_hits: int


def _reciprocal_rank_sum(
    ranks: Sequence[int],
    *,
    maximum_hits: int = 3,
) -> float:
    """Aggregate only the strongest ranks to avoid corpus-size bias."""

    ordered = sorted(rank for rank in ranks if rank > 0)[:maximum_hits]

    return sum(1.0 / rank for rank in ordered)


def rank_documents(
    *,
    reranked_results: Sequence[Any],
    hybrid_results: Sequence[Any],
) -> tuple[RankedDocument, ...]:
    """Rank documents using retrieval evidence, never a question->book rule."""

    reranker_ranks: dict[str, list[int]] = defaultdict(list)
    hybrid_ranks: dict[str, list[int]] = defaultdict(list)

    for result in reranked_results:
        document_id = str(result.chunk.document_id)
        reranker_ranks[document_id].append(int(result.rank))

    for result in hybrid_results:
        document_id = str(result.chunk.document_id)
        hybrid_ranks[document_id].append(int(result.rank))

    document_ids = set(reranker_ranks) | set(hybrid_ranks)

    ranked: list[RankedDocument] = []

    for document_id in document_ids:
        rranks = reranker_ranks.get(
            document_id,
            [],
        )
        hranks = hybrid_ranks.get(
            document_id,
            [],
        )

        ranked.append(
            RankedDocument(
                document_id=document_id,
                reranker_reciprocal_rank_sum=(_reciprocal_rank_sum(rranks)),
                hybrid_reciprocal_rank_sum=(_reciprocal_rank_sum(hranks)),
                best_reranker_rank=(min(rranks) if rranks else None),
                best_hybrid_rank=(min(hranks) if hranks else None),
                reranker_hits=len(rranks),
                hybrid_hits=len(hranks),
            )
        )

    # Primary signal:
    #   semantic cross-encoder evidence accumulated over best 3 passages.
    #
    # Tie breakers:
    #   best reranker rank -> hybrid evidence -> hit counts -> stable id.
    ranked.sort(
        key=lambda item: (
            -item.reranker_reciprocal_rank_sum,
            (item.best_reranker_rank if item.best_reranker_rank is not None else 10**9),
            -item.hybrid_reciprocal_rank_sum,
            -item.reranker_hits,
            -item.hybrid_hits,
            item.document_id,
        )
    )

    return tuple(ranked)
