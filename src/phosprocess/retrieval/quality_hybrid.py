"""Retriever-v4 hybrid search with dense, BGE sparse, BM25 and ColBERT."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from phosprocess.preprocessing.chunk_schemas import DocumentChunk
from phosprocess.retrieval.hybrid import HybridSearchResponse, HybridSearchResult
from phosprocess.retrieval.query_expansion import (
    ExpandedTechnicalQuery,
    expand_technical_query,
)
from phosprocess.retrieval.retrieval_planner import RetrievalPlan


@dataclass(slots=True)
class _Candidate:
    chunk: DocumentChunk
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None
    dense_contribution: float = 0.0
    bm25_contribution: float = 0.0
    sparse_contribution: float = 0.0
    section_bonus: float = 0.0
    colbert_score: float | None = None
    role_matches: set[str] = field(default_factory=set)
    role_scores: dict[str, float] = field(default_factory=dict)


def _contribution(rank: int | None, *, weight: float, rrf_k: int) -> float:
    return 0.0 if rank is None else weight / (rrf_k + rank)


def _merge_best_rank(
    current_rank: int | None,
    current_score: float | None,
    rank: int,
    score: float,
) -> tuple[int, float]:
    if current_rank is None or rank < current_rank:
        return rank, score
    return current_rank, current_score if current_score is not None else score


def _ordering_score(candidate: _Candidate) -> float:
    return (
        candidate.dense_contribution
        + candidate.bm25_contribution
        + candidate.sparse_contribution
        + candidate.section_bonus
        + 0.035 * float(candidate.colbert_score or 0.0)
    )


def _add_role_score(
    candidate: _Candidate,
    *,
    role_name: str,
    value: float,
) -> None:
    candidate.role_scores[role_name] = (
        candidate.role_scores.get(role_name, 0.0) + value
    )


def _to_result(candidate: _Candidate, rank: int) -> HybridSearchResult:
    matched = []
    if candidate.dense_rank is not None:
        matched.append("dense")
    if candidate.sparse_rank is not None:
        matched.append("bge_sparse")
    if candidate.bm25_rank is not None:
        matched.append("bm25")
    if candidate.colbert_score is not None:
        matched.append("bge_colbert")
    return HybridSearchResult(
        rank=rank,
        rrf_score=_ordering_score(candidate),
        matched_retrievers=tuple(matched),
        dense_rank=candidate.dense_rank,
        dense_score=candidate.dense_score,
        dense_rrf_contribution=candidate.dense_contribution,
        bm25_rank=candidate.bm25_rank,
        bm25_score=candidate.bm25_score,
        bm25_rrf_contribution=candidate.bm25_contribution,
        chunk=candidate.chunk,
        sparse_rank=candidate.sparse_rank,
        sparse_score=candidate.sparse_score,
        sparse_rrf_contribution=candidate.sparse_contribution,
        colbert_score=candidate.colbert_score,
        section_bonus=candidate.section_bonus,
        role_matches=tuple(sorted(candidate.role_matches)),
    )


def search_expanded_hybrid(
    retriever: Any,
    expanded_query: ExpandedTechnicalQuery,
    *,
    top_k: int,
    dense_candidate_k: int,
    bm25_candidate_k: int,
    sparse_retriever: Any | None = None,
    sparse_candidate_k: int = 0,
    document_ids: set[str] | None = None,
    chunk_ids: set[str] | None = None,
    section_bonus_by_chunk: dict[str, float] | None = None,
    role_name: str = "primary",
) -> HybridSearchResponse:
    """Search one expanded query through all available first-stage engines."""

    if min(top_k, dense_candidate_k, bm25_candidate_k) <= 0:
        raise ValueError("Les tailles de retrieval doivent être positives.")

    started = time.perf_counter()
    dense_response = retriever.dense_retriever.search(
        expanded_query.dense_query,
        top_k=dense_candidate_k,
        document_ids=document_ids,
        chunk_ids=chunk_ids,
    )
    bm25_response = retriever.bm25_retriever.search(
        expanded_query.bm25_expanded_query,
        top_k=bm25_candidate_k,
        document_ids=document_ids,
        chunk_ids=chunk_ids,
    )
    sparse_response = None
    if sparse_retriever is not None and sparse_candidate_k > 0:
        sparse_response = sparse_retriever.search(
            expanded_query.dense_query,
            top_k=sparse_candidate_k,
            document_ids=document_ids,
            chunk_ids=chunk_ids,
        )

    candidates: dict[str, _Candidate] = {}
    config = retriever.config
    for result in dense_response.results:
        item = candidates.setdefault(result.chunk.chunk_id, _Candidate(result.chunk))
        item.dense_rank = result.rank
        item.dense_score = result.score
        item.dense_contribution += _contribution(
            result.rank, weight=config.dense_weight, rrf_k=config.rrf_k
        )
        _add_role_score(
            item,
            role_name=role_name,
            value=_contribution(
                result.rank,
                weight=config.dense_weight,
                rrf_k=config.rrf_k,
            ),
        )
        item.role_matches.add(role_name)
    for result in bm25_response.results:
        item = candidates.setdefault(result.chunk.chunk_id, _Candidate(result.chunk))
        item.bm25_rank = result.rank
        item.bm25_score = result.score
        item.bm25_contribution += _contribution(
            result.rank, weight=config.bm25_weight, rrf_k=config.rrf_k
        )
        _add_role_score(
            item,
            role_name=role_name,
            value=_contribution(
                result.rank,
                weight=config.bm25_weight,
                rrf_k=config.rrf_k,
            ),
        )
        item.role_matches.add(role_name)
    if sparse_response is not None:
        for result in sparse_response.results:
            item = candidates.setdefault(result.chunk.chunk_id, _Candidate(result.chunk))
            item.sparse_rank = result.rank
            item.sparse_score = result.score
            item.sparse_contribution += _contribution(
                result.rank, weight=1.0, rrf_k=config.rrf_k
            )
            _add_role_score(
                item,
                role_name=role_name,
                value=_contribution(
                    result.rank,
                    weight=1.0,
                    rrf_k=config.rrf_k,
                ),
            )
            item.role_matches.add(role_name)

    bonuses = section_bonus_by_chunk or {}
    for chunk_id, item in candidates.items():
        item.section_bonus = bonuses.get(chunk_id, 0.0)

    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            -_ordering_score(item),
            -len(item.role_matches),
            item.chunk.chunk_id,
        ),
    )[:top_k]
    results = [_to_result(item, rank) for rank, item in enumerate(ranked, start=1)]
    return HybridSearchResponse(
        query=expanded_query.dense_query,
        lexical_query=expanded_query.bm25_expanded_query,
        top_k_requested=top_k,
        dense_candidates_requested=dense_candidate_k,
        bm25_candidates_requested=bm25_candidate_k,
        dense_results_found=len(dense_response.results),
        bm25_results_found=len(bm25_response.results),
        fusion_candidates=len(candidates),
        dense_duration_ms=float(dense_response.search_duration_ms),
        bm25_duration_ms=float(bm25_response.search_duration_ms),
        total_duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
        results=results,
        sparse_candidates_requested=sparse_candidate_k if sparse_retriever is not None else 0,
        sparse_results_found=len(sparse_response.results) if sparse_response is not None else 0,
        sparse_duration_ms=(
            float(sparse_response.search_duration_ms) if sparse_response is not None else 0.0
        ),
        role_queries=((role_name, expanded_query.standalone_query),),
    )


def search_planned_hybrid(
    retriever: Any,
    plan: RetrievalPlan,
    *,
    sparse_retriever: Any | None,
    top_k: int = 30,
    dense_candidate_k: int = 50,
    sparse_candidate_k: int = 50,
    bm25_candidate_k: int = 50,
    fusion_k: int = 80,
    colbert_candidate_k: int = 80,
    document_ids: set[str] | None = None,
    section_bonus_by_chunk: dict[str, float] | None = None,
) -> HybridSearchResponse:
    """Retrieve each evidence role independently, fuse globally, then ColBERT-rank."""

    if not plan.roles:
        raise ValueError("Le plan de retrieval ne contient aucun rôle.")
    if min(top_k, dense_candidate_k, sparse_candidate_k, bm25_candidate_k) <= 0:
        raise ValueError("Les tailles de retriever v4 doivent être positives.")

    started = time.perf_counter()
    config = retriever.config
    candidates: dict[str, _Candidate] = {}
    dense_ms = bm25_ms = sparse_ms = 0.0
    dense_found = bm25_found = sparse_found = 0

    for role in plan.roles:
        expanded = expand_technical_query(
            role.query,
            standalone_query=role.query,
            question_type=plan.question_type,
        )
        dense = retriever.dense_retriever.search(
            expanded.dense_query,
            top_k=dense_candidate_k,
            document_ids=document_ids,
            chunk_ids=None,
        )
        lexical = retriever.bm25_retriever.search(
            expanded.bm25_expanded_query,
            top_k=bm25_candidate_k,
            document_ids=document_ids,
            chunk_ids=None,
        )
        sparse_response = (
            sparse_retriever.search(
                expanded.dense_query,
                top_k=sparse_candidate_k,
                document_ids=document_ids,
                chunk_ids=None,
            )
            if sparse_retriever is not None
            else None
        )
        dense_ms += float(dense.search_duration_ms)
        bm25_ms += float(lexical.search_duration_ms)
        sparse_ms += (
            float(sparse_response.search_duration_ms) if sparse_response is not None else 0.0
        )
        dense_found += len(dense.results)
        bm25_found += len(lexical.results)
        sparse_found += len(sparse_response.results) if sparse_response is not None else 0

        for result in dense.results:
            item = candidates.setdefault(result.chunk.chunk_id, _Candidate(result.chunk))
            item.dense_rank, item.dense_score = _merge_best_rank(
                item.dense_rank, item.dense_score, result.rank, result.score
            )
            item.dense_contribution += _contribution(
                result.rank, weight=config.dense_weight, rrf_k=config.rrf_k
            )
            _add_role_score(
                item,
                role_name=role.name,
                value=_contribution(
                    result.rank,
                    weight=config.dense_weight,
                    rrf_k=config.rrf_k,
                ),
            )
            item.role_matches.add(role.name)
        for result in lexical.results:
            item = candidates.setdefault(result.chunk.chunk_id, _Candidate(result.chunk))
            item.bm25_rank, item.bm25_score = _merge_best_rank(
                item.bm25_rank, item.bm25_score, result.rank, result.score
            )
            item.bm25_contribution += _contribution(
                result.rank, weight=config.bm25_weight, rrf_k=config.rrf_k
            )
            _add_role_score(
                item,
                role_name=role.name,
                value=_contribution(
                    result.rank,
                    weight=config.bm25_weight,
                    rrf_k=config.rrf_k,
                ),
            )
            item.role_matches.add(role.name)
        if sparse_response is not None:
            for result in sparse_response.results:
                item = candidates.setdefault(result.chunk.chunk_id, _Candidate(result.chunk))
                item.sparse_rank, item.sparse_score = _merge_best_rank(
                    item.sparse_rank, item.sparse_score, result.rank, result.score
                )
                item.sparse_contribution += _contribution(
                    result.rank, weight=1.0, rrf_k=config.rrf_k
                )
                _add_role_score(
                    item,
                    role_name=role.name,
                    value=_contribution(
                        result.rank,
                        weight=1.0,
                        rrf_k=config.rrf_k,
                    ),
                )
                item.role_matches.add(role.name)

    bonuses = section_bonus_by_chunk or {}
    for chunk_id, item in candidates.items():
        item.section_bonus = bonuses.get(chunk_id, 0.0)

    union = sorted(
        candidates.values(),
        key=lambda item: (
            -_ordering_score(item),
            -len(item.role_matches),
            item.chunk.chunk_id,
        ),
    )[:fusion_k]

    colbert_pool = union[: min(colbert_candidate_k, len(union))]
    embedder = getattr(retriever.dense_retriever, "embedder", None)
    colbert_available = embedder is not None and all(
        callable(getattr(embedder, method_name, None))
        for method_name in (
            "embed_colbert_query",
            "embed_colbert_documents",
            "colbert_score",
        )
    )
    if colbert_pool and colbert_available:
        query_vectors = [embedder.embed_colbert_query(role.query) for role in plan.roles]
        passage_vectors = embedder.embed_colbert_documents(
            [item.chunk.embedding_text for item in colbert_pool]
        )
        for item, vectors in zip(colbert_pool, passage_vectors, strict=True):
            item.colbert_score = max(
                embedder.colbert_score(query_vector, vectors)
                for query_vector in query_vectors
            )

    globally_ranked = sorted(
        union,
        key=lambda item: (
            -_ordering_score(item),
            -len(item.role_matches),
            item.chunk.chunk_id,
        ),
    )

    # Keep several strong candidates from every evidence-role query before
    # the cross-encoder window is truncated.  This prevents a generic passage
    # from removing an explicit conservation equation or stream definition
    # that was correctly retrieved for one role but ranked below the global
    # top-k by the multi-query fusion score.
    reserved: list[_Candidate] = []
    reserved_ids: set[str] = set()
    role_reserve_k = 3
    for role in plan.roles:
        role_ranked = sorted(
            (
                item
                for item in union
                if item.role_scores.get(role.name, 0.0) > 0.0
            ),
            key=lambda item: (
                -item.role_scores.get(role.name, 0.0),
                -_ordering_score(item),
                item.chunk.chunk_id,
            ),
        )
        added_for_role = 0
        for item in role_ranked:
            if item.chunk.chunk_id in reserved_ids:
                continue
            reserved.append(item)
            reserved_ids.add(item.chunk.chunk_id)
            added_for_role += 1
            if added_for_role >= role_reserve_k:
                break

    ranked = [
        *reserved,
        *(
            item
            for item in globally_ranked
            if item.chunk.chunk_id not in reserved_ids
        ),
    ][:top_k]
    results = [_to_result(item, rank) for rank, item in enumerate(ranked, start=1)]
    return HybridSearchResponse(
        query=plan.base_query,
        lexical_query=" || ".join(role.query for role in plan.roles),
        top_k_requested=top_k,
        dense_candidates_requested=dense_candidate_k * len(plan.roles),
        bm25_candidates_requested=bm25_candidate_k * len(plan.roles),
        dense_results_found=dense_found,
        bm25_results_found=bm25_found,
        fusion_candidates=len(candidates),
        dense_duration_ms=round(dense_ms, 3),
        bm25_duration_ms=round(bm25_ms, 3),
        total_duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
        results=results,
        sparse_candidates_requested=(
            sparse_candidate_k * len(plan.roles) if sparse_retriever is not None else 0
        ),
        sparse_results_found=sparse_found,
        sparse_duration_ms=round(sparse_ms, 3),
        role_queries=tuple((role.name, role.query) for role in plan.roles),
    )
