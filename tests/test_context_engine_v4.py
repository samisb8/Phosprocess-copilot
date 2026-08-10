"""Generic structural tests for parent-first context reconstruction."""

from __future__ import annotations

from phosprocess.ingestion.chunk_serialization import (
    TechnicalChildChunk,
    TechnicalChunkType,
    TechnicalParentChunk,
)
from phosprocess.retrieval.context_expander import (
    ContextExpander,
    ContextExpansionConfig,
    EvidenceAnchor,
)
from phosprocess.retrieval.evidence_bundle import EvidenceContextScope


def token_count(text: str) -> int:
    """Small deterministic tokenizer used only by these structural tests."""

    return len(text.split())


def make_child(
    chunk_id: str,
    *,
    parent_id: str,
    text: str,
    document_id: str = "document_a",
    section_id: str = "section_a",
    subsection: str = "subsection_a",
    previous_chunk_id: str | None = None,
    next_chunk_id: str | None = None,
    page: int = 1,
) -> TechnicalChildChunk:
    return TechnicalChildChunk(
        chunk_id=chunk_id,
        parent_id=parent_id,
        previous_chunk_id=previous_chunk_id,
        next_chunk_id=next_chunk_id,
        document_id=document_id,
        document_title=f"Title {document_id}",
        source_file=f"{document_id}.pdf",
        domains=("generic",),
        chapter="Chapter",
        section="Section",
        subsection=subsection,
        hierarchy_path=f"Chapter > Section > {subsection}",
        section_id=section_id,
        chunk_type=TechnicalChunkType.NARRATIVE,
        page_start=page,
        page_end=page,
        text=text,
        display_text=text,
        embedding_text=f"EMBEDDING_ONLY_MARKER {chunk_id} {text}",
        bm25_text=f"BM25_ONLY_MARKER {chunk_id} {text}",
        token_count=max(1, token_count(text)),
        sha256=f"{int(chunk_id.removeprefix('c')):064x}",
    )


def make_parent(
    parent_id: str,
    children: list[TechnicalChildChunk],
    *,
    display_text: str | None = None,
) -> TechnicalParentChunk:
    documentary_text = display_text or "\n\n".join(
        child.display_text for child in children
    )
    return TechnicalParentChunk(
        parent_id=parent_id,
        document_id=children[0].document_id,
        document_title=children[0].document_title,
        source_file=children[0].source_file,
        chapter=children[0].chapter,
        section=children[0].section,
        subsection=children[0].subsection,
        hierarchy_path=children[0].hierarchy_path,
        section_id=children[0].section_id,
        page_start=min(child.page_start for child in children),
        page_end=max(child.page_end for child in children),
        child_chunk_ids=tuple(child.chunk_id for child in children),
        display_text=documentary_text,
        token_count=max(1, token_count(documentary_text)),
        sha256=(parent_id.encode().hex() + "0" * 64)[:64],
    )


def anchor(
    child: TechnicalChildChunk,
    score: float = 0.9,
    source: str = "reranker",
) -> EvidenceAnchor:
    return EvidenceAnchor(child=child, score=score, provenance=source)


def test_multiple_anchors_with_same_parent_create_one_bundle() -> None:
    children = [
        make_child(f"c{number}", parent_id="p1", text=f"fragment {number}", page=number)
        for number in range(1, 4)
    ]
    expander = ContextExpander(
        children=children,
        parents=[make_parent("p1", children)],
        token_counter=token_count,
    )

    bundles = expander.expand([anchor(child) for child in children], question_type="generic")

    assert len(bundles) == 1
    assert bundles[0].anchor_chunk_ids == ("c1", "c2", "c3")
    assert bundles[0].supporting_chunk_ids == ("c1", "c2", "c3")


def test_full_parent_is_used_when_it_fits() -> None:
    children = [
        make_child("c1", parent_id="p1", text="first faithful fragment", next_chunk_id="c2"),
        make_child("c2", parent_id="p1", text="second faithful fragment", previous_chunk_id="c1"),
    ]
    parent = make_parent("p1", children)
    bundle = ContextExpander(
        children=children,
        parents=[parent],
        token_counter=token_count,
    ).expand([anchor(children[0])], question_type="generic")[0]

    assert bundle.context_scope is EvidenceContextScope.FULL_PARENT
    assert bundle.display_text == parent.display_text


def test_oversized_parent_preserves_anchor_and_adds_partial_context() -> None:
    children = [
        make_child("c1", parent_id="p1", text="anchor fragment", page=1),
        make_child("c2", parent_id="p1", text="nearby context fragment", page=2),
        make_child(
            "c3",
            parent_id="p1",
            text="distant context " + "word " * 30,
            page=3,
        ),
    ]
    parent = make_parent("p1", children)
    probe = ContextExpander(children=children, parents=[], token_counter=token_count)
    partial_budget = probe._rendered_tokens(  # noqa: SLF001 - exact packing fixture
        source_number=1,
        anchor=children[0],
        page_start=1,
        page_end=2,
        display_text=f"{children[0].display_text}\n\n{children[1].display_text}",
    )
    bundle = ContextExpander(
        children=children,
        parents=[parent],
        config=ContextExpansionConfig(
            neighbor_window=0,
            max_tokens_per_bundle=partial_budget,
            max_total_context_tokens=partial_budget,
        ),
        token_counter=token_count,
    ).expand([anchor(children[0])], question_type="generic")[0]

    assert bundle.context_scope is EvidenceContextScope.PARTIAL_PARENT
    assert bundle.anchor_chunk_ids == ("c1",)
    assert bundle.supporting_chunk_ids == ("c1", "c2")
    assert bundle.token_count <= partial_budget


def test_anchor_only_is_valid_without_parent_context() -> None:
    child = make_child("c1", parent_id="missing", text="standalone faithful fragment")
    bundle = ContextExpander(
        children=[child],
        parents=[],
        config=ContextExpansionConfig(neighbor_window=0),
        token_counter=token_count,
    ).expand([anchor(child)], question_type="generic")[0]

    assert bundle.context_scope is EvidenceContextScope.ANCHOR_ONLY
    assert bundle.supporting_chunk_ids == bundle.anchor_chunk_ids == ("c1",)


def test_neighbor_across_structural_boundary_is_not_included() -> None:
    first = make_child(
        "c1",
        parent_id="missing_a",
        text="boundary anchor",
        next_chunk_id="c2",
        subsection="subsection_a",
    )
    crossing = make_child(
        "c2",
        parent_id="missing_b",
        text="crossing neighbor",
        previous_chunk_id="c1",
        section_id="section_b",
        subsection="subsection_b",
    )
    bundle = ContextExpander(
        children=[first, crossing],
        parents=[],
        token_counter=token_count,
    ).expand([anchor(first)], question_type="generic")[0]

    assert bundle.supporting_chunk_ids == ("c1",)
    assert crossing.display_text not in bundle.display_text


def test_bundle_count_is_dynamic_and_not_forced_to_five() -> None:
    children = [
        make_child(
            f"c{number}",
            parent_id=f"p{number}",
            document_id=f"document_{number}",
            section_id=f"section_{number}",
            text=f"independent evidence {number}",
        )
        for number in range(1, 4)
    ]
    bundles = ContextExpander(
        children=children,
        parents=[],
        token_counter=token_count,
    ).expand([anchor(child) for child in children], question_type="generic")

    assert len(bundles) == 3


def test_exact_serialized_metadata_and_text_stay_within_total_budget() -> None:
    children = [
        make_child(
            f"c{number}",
            parent_id=f"p{number}",
            document_id=f"document_{number}",
            section_id=f"section_{number}",
            text=f"compact documentary fragment {number}",
        )
        for number in range(1, 5)
    ]
    total_budget = 60
    bundles = ContextExpander(
        children=children,
        parents=[],
        config=ContextExpansionConfig(
            max_tokens_per_bundle=40,
            max_total_context_tokens=total_budget,
        ),
        token_counter=token_count,
    ).expand([anchor(child) for child in children], question_type="generic")

    assert sum(bundle.token_count for bundle in bundles) <= total_budget
    assert all(
        bundle.token_count == token_count(bundle.render_prompt_block())
        for bundle in bundles
    )


def test_factual_evidence_never_uses_embedding_or_bm25_text() -> None:
    children = [
        make_child("c1", parent_id="p1", text="faithful alpha"),
        make_child("c2", parent_id="p1", text="faithful beta"),
    ]
    bundle = ContextExpander(
        children=children,
        parents=[make_parent("p1", children)],
        token_counter=token_count,
    ).expand([anchor(children[0])], question_type="generic")[0]

    assert "EMBEDDING_ONLY_MARKER" not in bundle.display_text
    assert "BM25_ONLY_MARKER" not in bundle.display_text
    assert bundle.display_text == "faithful alpha\n\nfaithful beta"


def test_locked_document_does_not_gain_cross_document_context() -> None:
    locked = make_child(
        "c1",
        parent_id="missing",
        document_id="locked_document",
        text="locked evidence",
        next_chunk_id="c2",
    )
    other = make_child(
        "c2",
        parent_id="other_parent",
        document_id="other_document",
        text="unlocked evidence",
        previous_chunk_id="c1",
    )
    bundles = ContextExpander(
        children=[locked, other],
        parents=[],
        token_counter=token_count,
    ).expand([anchor(locked)], question_type="generic")

    assert {bundle.document_id for bundle in bundles} == {"locked_document"}
    assert all("c2" not in bundle.supporting_chunk_ids for bundle in bundles)


def test_every_documentary_fragment_has_active_chunk_provenance() -> None:
    children = [
        make_child("c1", parent_id="p1", text="traceable first", page=1),
        make_child("c2", parent_id="p1", text="traceable second", page=2),
    ]
    child_by_id = {child.chunk_id: child for child in children}
    bundle = ContextExpander(
        children=children,
        parents=[make_parent("p1", children)],
        token_counter=token_count,
    ).expand(
        [anchor(children[0], source="reranker"), anchor(children[1], source="bm25")],
        question_type="generic",
    )[0]

    assert set(bundle.selection_provenance.split(" | ")) == {"reranker", "bm25"}
    assert all(child_by_id[chunk_id].active for chunk_id in bundle.supporting_chunk_ids)
    assert bundle.display_text == "\n\n".join(
        child_by_id[chunk_id].display_text for chunk_id in bundle.supporting_chunk_ids
    )
