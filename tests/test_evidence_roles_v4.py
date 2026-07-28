"""Role-aware final evidence-selection invariants."""

from __future__ import annotations

from phosprocess.preprocessing.chunk_schemas import DocumentChunk
from phosprocess.reranking.reranker import RerankedSearchResult
from phosprocess.retrieval.evidence_roles import select_role_aware_evidence
from phosprocess.retrieval.hybrid import HybridSearchResult
from phosprocess.retrieval.retrieval_planner import build_retrieval_plan


def _chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc",
        source_file="doc.pdf",
        chunk_index=0,
        source_pages=[1],
        page_start=1,
        page_end=1,
        text=text,
        embedding_text=f"Document: Doc\n{text}",
        body_token_count=30,
        token_count=30,
        document_title="Doc",
        active=True,
    )


def _hybrid(
    rank: int,
    chunk: DocumentChunk,
    roles: tuple[str, ...],
) -> HybridSearchResult:
    return HybridSearchResult(
        rank=rank,
        rrf_score=1.0 / rank,
        matched_retrievers=("dense", "bge_sparse", "bm25"),
        dense_rank=rank,
        dense_score=0.8,
        dense_rrf_contribution=0.01,
        bm25_rank=rank,
        bm25_score=2.0,
        bm25_rrf_contribution=0.01,
        chunk=chunk,
        sparse_rank=rank,
        sparse_score=1.0,
        sparse_rrf_contribution=0.01,
        role_matches=roles,
    )


def _reranked(rank: int, candidate: HybridSearchResult) -> RerankedSearchResult:
    return RerankedSearchResult(
        rank=rank,
        reranker_score=1.0 - rank / 10,
        original_hybrid_rank=candidate.rank,
        original_rrf_score=candidate.rrf_score,
        matched_retrievers=candidate.matched_retrievers,
        dense_rank=candidate.dense_rank,
        dense_score=candidate.dense_score,
        bm25_rank=candidate.bm25_rank,
        bm25_score=candidate.bm25_score,
        chunk=candidate.chunk,
        sparse_rank=candidate.sparse_rank,
        sparse_score=candidate.sparse_score,
        role_matches=candidate.role_matches,
    )


def test_comparison_requires_explicit_evidence_for_both_equipment() -> None:
    question = (
        "Compare a forced-circulation evaporator with a falling-film "
        "evaporator for phosphoric acid concentration."
    )
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="comparison",
    )
    a = _hybrid(
        1,
        _chunk("a", "Forced-circulation evaporators use a circulation pump."),
        ("equipment_a", "comparison_criteria"),
    )
    b = _hybrid(
        2,
        _chunk("b", "A falling-film evaporator forms a thin falling liquid film."),
        ("equipment_b", "comparison_criteria"),
    )
    candidates = [a, b]

    selection = select_role_aware_evidence(
        plan,
        candidates,
        [_reranked(1, a), _reranked(2, b)],
        top_k=2,
    )

    assert selection.complete
    assert {item.chunk_id for item in selection.selected} == {"a", "b"}
    assert {"equipment_a", "equipment_b"}.issubset(selection.covered_roles)


def test_comparison_reports_missing_second_side_instead_of_inventing_it() -> None:
    question = (
        "Compare a forced-circulation evaporator with a falling-film "
        "evaporator for phosphoric acid concentration."
    )
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="comparison",
    )
    a = _hybrid(
        1,
        _chunk("a", "Forced-circulation evaporators use a circulation pump."),
        ("equipment_a", "equipment_b", "comparison_criteria"),
    )

    selection = select_role_aware_evidence(
        plan,
        [a],
        [_reranked(1, a)],
        top_k=1,
    )

    assert "equipment_b" in selection.missing_roles


def test_required_balance_roles_are_promoted_into_evidence_window() -> None:
    from phosprocess.retrieval.evidence_roles import (
        promote_required_roles_in_reranking,
    )

    question = "Establish the steady-state energy balance of an evaporator."
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="balance",
    )
    generic = _hybrid(
        1,
        _chunk("generic", "General evaporator design passage."),
        tuple(role.name for role in plan.roles),
    )
    conservation = _hybrid(
        2,
        _chunk("conservation", "Energy balance: energy in equals energy out."),
        ("energy_conservation",),
    )
    heat = _hybrid(
        3,
        _chunk("heat", "The heating medium supplies steam heat input."),
        ("heat_input",),
    )
    enthalpy = _hybrid(
        4,
        _chunk("enthalpy", "Feed enthalpy and product enthalpy are included."),
        ("feed_product_enthalpy",),
    )
    vapor = _hybrid(
        5,
        _chunk("vapor", "The vapor enthalpy includes latent heat."),
        ("vapor_enthalpy",),
    )
    candidates = [generic, conservation, heat, enthalpy, vapor]
    original = [
        _reranked(1, generic),
        _reranked(2, conservation),
        _reranked(3, heat),
        _reranked(4, enthalpy),
        _reranked(5, vapor),
    ]

    promoted = promote_required_roles_in_reranking(
        plan,
        candidates,
        original,
    )

    first_four = {item.chunk.chunk_id for item in promoted[:4]}
    assert first_four == {"conservation", "heat", "enthalpy", "vapor"}
