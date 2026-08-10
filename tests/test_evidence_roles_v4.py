"""Generic structural-role diversification tests."""

from __future__ import annotations

from types import SimpleNamespace

from phosprocess.retrieval.evidence_roles import select_role_aware_evidence
from phosprocess.retrieval.retrieval_planner import EvidenceRole, RetrievalPlan


def candidate(chunk_id: str, rank: int, *roles: str) -> SimpleNamespace:
    return SimpleNamespace(
        rank=rank,
        bm25_rank=rank,
        role_matches=roles,
        chunk=SimpleNamespace(
            chunk_id=chunk_id,
            document_title="Generic document",
            hierarchy_path="Section",
            section="Section",
            subsection="",
            text="Faithful documentary passage.",
            embedding_text="Contextual retrieval representation.",
        ),
    )


def reranked(item: SimpleNamespace, rank: int) -> SimpleNamespace:
    return SimpleNamespace(rank=rank, chunk=item.chunk)


def test_selection_diversifies_by_optional_structural_roles() -> None:
    plan = RetrievalPlan(
        question_type="process_flow",
        base_query="generic flow path",
        roles=(
            EvidenceRole(name="entry_context", query="entry beginning"),
            EvidenceRole(name="exit_context", query="exit end"),
        ),
    )
    entry = candidate("entry", 1, "entry_context")
    exit_item = candidate("exit", 2, "exit_context")
    filler = candidate("filler", 3)

    result = select_role_aware_evidence(
        plan,
        [entry, exit_item, filler],
        [reranked(entry, 1), reranked(exit_item, 2), reranked(filler, 3)],
        top_k=2,
    )

    assert tuple(item.chunk_id for item in result.selected) == ("entry", "exit")
    assert result.covered_roles == ("entry_context", "exit_context")
    assert all("evidence_role:" in item.source for item in result.selected)


def test_missing_optional_role_does_not_create_a_completeness_failure() -> None:
    plan = RetrievalPlan(
        question_type="other",
        base_query="generic question",
        roles=(EvidenceRole(name="supporting_context", query="supporting context"),),
    )
    filler = candidate("filler", 1)
    result = select_role_aware_evidence(
        plan,
        [filler],
        [reranked(filler, 1)],
        top_k=1,
    )
    assert result.covered_roles == ()
    assert result.selected[0].chunk_id == "filler"
