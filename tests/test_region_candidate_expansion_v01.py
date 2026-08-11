"""Phase-9 safety invariants for label-blind structural candidate access."""

from __future__ import annotations

import ast
import inspect

import pytest

from phosprocess.evaluation.region_aware_evaluation_v01 import _rank_by_score
from phosprocess.evaluation.region_candidate_expansion_v01 import (
    RegionCandidateExpander,
    RegionVariant,
)
from phosprocess.ingestion.chunk_serialization import (
    TechnicalChildChunk,
    TechnicalChunkType,
    TechnicalParentChunk,
)

SHA = "0" * 64


def _child(
    chunk_id: str,
    *,
    parent_id: str = "parent",
    document_id: str = "doc",
    section_id: str = "section",
    hierarchy_path: str = "chapter/section",
    previous_chunk_id: str | None = None,
    next_chunk_id: str | None = None,
) -> TechnicalChildChunk:
    text = f"Documentary passage {chunk_id}."
    return TechnicalChildChunk(
        chunk_id=chunk_id,
        parent_id=parent_id,
        previous_chunk_id=previous_chunk_id,
        next_chunk_id=next_chunk_id,
        document_id=document_id,
        document_title="Document",
        source_file="document.pdf",
        domains=("generic",),
        chapter="Chapter",
        section="Section",
        hierarchy_path=hierarchy_path,
        section_id=section_id,
        chunk_type=TechnicalChunkType.NARRATIVE,
        page_start=1,
        page_end=1,
        text=text,
        display_text=text,
        embedding_text=f"Document > {text}",
        bm25_text=text,
        token_count=4,
        sha256=SHA,
    )


def _expander() -> RegionCandidateExpander:
    children = [
        _child("previous", parent_id="other"),
        _child(
            "anchor",
            previous_chunk_id="previous",
            next_chunk_id="next",
        ),
        _child("sibling"),
        _child("next", parent_id="other"),
        _child("cross_document", parent_id="other", document_id="other-doc"),
        _child(
            "bad_hierarchy",
            parent_id="other",
            section_id="other-section",
            hierarchy_path="chapter/other-section",
        ),
    ]
    parent = TechnicalParentChunk(
        parent_id="parent",
        document_id="doc",
        document_title="Document",
        source_file="document.pdf",
        chapter="Chapter",
        section="Section",
        hierarchy_path="chapter/section",
        section_id="section",
        page_start=1,
        page_end=1,
        child_chunk_ids=("anchor", "sibling"),
        display_text="Parent documentary passage.",
        token_count=4,
        sha256=SHA,
    )
    return RegionCandidateExpander(children=children, parents=[parent])


def test_01_same_parent_candidate_can_be_added() -> None:
    related = _expander().related_candidates("anchor", RegionVariant.SAME_PARENT)
    assert [(item.chunk_id, item.provenance) for item in related] == [
        ("sibling", "same_parent")
    ]


def test_02_previous_and_next_candidates_can_be_added() -> None:
    related = _expander().related_candidates(
        "anchor", RegionVariant.PARENT_AND_NEIGHBORS
    )
    assert {item.chunk_id for item in related} >= {"previous", "next"}


def test_03_cross_document_neighbor_cannot_be_added() -> None:
    expander = _expander()
    expander.child_by_id["anchor"] = expander.child_by_id["anchor"].model_copy(
        update={"next_chunk_id": "cross_document"}
    )
    related = expander.related_candidates("anchor", RegionVariant.PARENT_AND_NEIGHBORS)
    assert "cross_document" not in {item.chunk_id for item in related}


def test_04_incompatible_hierarchy_neighbor_cannot_be_added() -> None:
    expander = _expander()
    expander.child_by_id["anchor"] = expander.child_by_id["anchor"].model_copy(
        update={"next_chunk_id": "bad_hierarchy"}
    )
    related = expander.related_candidates("anchor", RegionVariant.PARENT_AND_NEIGHBORS)
    assert "bad_hierarchy" not in {item.chunk_id for item in related}


def test_05_duplicate_chunk_ids_collapse() -> None:
    composition = _expander().compose(
        ["anchor", "sibling", "anchor", "next"],
        locked_document="doc",
        variant=RegionVariant.PARENT_AND_NEIGHBORS,
        anchor_k=3,
        candidate_budget=10,
    )
    assert len(composition.candidate_ids) == len(set(composition.candidate_ids))


def test_06_anchor_remains_present() -> None:
    composition = _expander().compose(
        ["anchor", "sibling", "next"],
        locked_document="doc",
        variant=RegionVariant.PARENT_AND_NEIGHBORS,
        anchor_k=1,
        candidate_budget=2,
    )
    assert composition.candidate_ids[0] == "anchor"


def test_07_source_lock_remains_exact() -> None:
    with pytest.raises(ValueError, match="source lock"):
        _expander().compose(
            ["anchor", "cross_document"],
            locked_document="doc",
            variant=RegionVariant.PARENT_AND_NEIGHBORS,
            anchor_k=2,
            candidate_budget=5,
        )


def test_08_candidate_budget_is_respected() -> None:
    composition = _expander().compose(
        ["anchor", "sibling", "next", "previous"],
        locked_document="doc",
        variant=RegionVariant.PARENT_NEIGHBORS_SECTION_2,
        anchor_k=2,
        candidate_budget=3,
    )
    assert len(composition.candidate_ids) == 3


def test_09_region_policy_never_imports_evaluation_labels() -> None:
    import phosprocess.evaluation.region_candidate_expansion_v01 as policy

    source = inspect.getsource(policy)
    tree = ast.parse(source)
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ] + [
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    ]
    assert not any("phosprocess.evaluation" in name for name in imports)
    assert all(
        forbidden not in source
        for forbidden in ("gold_chunk_ids", "valid_evidence_sets", "CE051", "DQ027")
    )


def test_10_region_candidates_are_reranked_without_priority_bypass() -> None:
    composition = _expander().compose(
        ["anchor"],
        locked_document="doc",
        variant=RegionVariant.SAME_PARENT,
        anchor_k=1,
        candidate_budget=2,
    )
    assert composition.candidate_ids == ("anchor", "sibling")
    scores = {"anchor": 0.1, "sibling": 0.9}
    assert _rank_by_score(list(composition.candidate_ids), scores) == [
        "sibling",
        "anchor",
    ]
