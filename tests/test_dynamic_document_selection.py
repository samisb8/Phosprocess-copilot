\
"""Tests for evidence-driven document selection."""

from types import SimpleNamespace

from phosprocess.knowledge_base.catalog import load_document_catalog
from phosprocess.retrieval.document_selector import rank_documents
from phosprocess.retrieval.domain_router import route_query


def _result(
    document_id: str,
    rank: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        rank=rank,
        chunk=SimpleNamespace(
            document_id=document_id,
        ),
    )


def test_rank_documents_uses_multiple_strong_hits_not_document_identity() -> None:
    reranked = [
        _result("document_b", 1),
        _result("document_a", 2),
        _result("document_a", 3),
        _result("document_a", 4),
        _result("document_c", 5),
    ]

    hybrid = [
        _result("document_a", 1),
        _result("document_b", 2),
        _result("document_c", 3),
    ]

    ranking = rank_documents(
        reranked_results=reranked,
        hybrid_results=hybrid,
    )

    assert ranking[0].document_id == "document_a"

    assert (
        ranking[0].reranker_reciprocal_rank_sum
        > ranking[1].reranker_reciprocal_rank_sum
    )


def test_automatic_router_has_no_document_prior() -> None:
    decision = route_query(
        "What is vapor-liquid equilibrium?",
        catalog=load_document_catalog(),
        source_mode="auto",
        question_type="explanation",
    )

    assert decision.preferred_documents == ()
    assert decision.soft_boosts == {}
    assert decision.hard_filter is None


def test_explicit_source_remains_a_user_controlled_lock() -> None:
    decision = route_query(
        "Selon Perry, explique cet ?quipement.",
        catalog=load_document_catalog(),
        source_mode="auto",
        question_type="explanation",
    )

    assert decision.hard_filter == frozenset(
        {"perrys_chemical_engineers_handbook"}
    )
