"""Soft routing, technical expansion and evidence-bundle tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from phosprocess.ingestion.chunk_serialization import (
    TechnicalChildChunk,
    TechnicalChunkType,
    TechnicalParentChunk,
)
from phosprocess.knowledge_base.catalog import load_document_catalog
from phosprocess.knowledge_base.domains import KnowledgeDomain
from phosprocess.retrieval.context_expander import (
    ContextExpander,
    ContextExpansionConfig,
    EvidenceAnchor,
)
from phosprocess.retrieval.domain_router import route_query
from phosprocess.retrieval.query_expansion import expand_technical_query
from phosprocess.retrieval.source_boosting import apply_soft_boosts


def child(number: int, *, document_id: str = "document") -> TechnicalChildChunk:
    text = f"Technical process passage {number}. " * 20
    return TechnicalChildChunk(
        chunk_id=f"chunk_{number}",
        parent_id="parent",
        previous_chunk_id=f"chunk_{number - 1}" if number > 1 else None,
        next_chunk_id=f"chunk_{number + 1}" if number < 3 else None,
        document_id=document_id,
        document_title="Technical Book",
        source_file="book.pdf",
        domains=("heat_transfer",),
        chapter="Chapter 1",
        section="Section 1",
        chunk_type=TechnicalChunkType.PROCESS_DESCRIPTION,
        page_start=number,
        page_end=number,
        text=text,
        display_text=text,
        embedding_text=f"Document: Technical Book\n\n{text}",
        bm25_text=f"Technical Book\n{text}",
        token_count=100,
        sha256=f"{number:064x}",
    )


def parent(children: list[TechnicalChildChunk]) -> TechnicalParentChunk:
    text = "\n\n".join(item.display_text for item in children)
    return TechnicalParentChunk(
        parent_id="parent",
        document_id=children[0].document_id,
        document_title=children[0].document_title,
        source_file=children[0].source_file,
        chapter="Chapter 1",
        section="Section 1",
        page_start=1,
        page_end=3,
        child_chunk_ids=tuple(item.chunk_id for item in children),
        display_text=text,
        token_count=300,
        sha256="f" * 64,
    )


def test_automatic_routing_uses_soft_boosts_without_filter() -> None:
    catalog = load_document_catalog()
    decision = route_query(
        "Comment la sursaturation influence-t-elle le gypse ?",
        catalog=catalog,
    )

    domains = {domain for domain, _score in decision.detected_domains}
    assert KnowledgeDomain.CRYSTALLIZATION in domains
    assert KnowledgeDomain.PHOSPHORIC_ACID_PROCESS in domains
    assert decision.hard_filter is None
    assert "mullin_crystallization" in decision.preferred_documents
    assert "becker_phosphates_and_phosphoric_acid" in (
        decision.preferred_documents
    )


@pytest.mark.parametrize(
    ("question", "expected_document"),
    [
        (
            "phosphoric acid filtration",
            "becker_phosphates_and_phosphoric_acid",
        ),
        (
            "enthalpy and vapor pressure",
            "smith_van_ness_chemical_engineering_thermodynamics",
        ),
        (
            "heat exchanger conduction",
            "incropera_fundamentals_heat_mass_transfer",
        ),
        ("pressure drop in fluid flow", "bird_transport_phenomena"),
        ("supersaturation and nucleation", "mullin_crystallization"),
        ("MPC process control", "seborg_process_dynamics_control"),
        (
            "general chemical engineering unit operation",
            "perrys_chemical_engineers_handbook",
        ),
        (
            "What is a forced-circulation evaporator?",
            "perrys_chemical_engineers_handbook",
        ),
    ],
)
def test_domain_routes_favor_the_domain_reference(
    question: str,
    expected_document: str,
) -> None:
    decision = route_query(question, catalog=load_document_catalog())

    assert decision.preferred_documents[0] == expected_document
    assert decision.hard_filter is None


def test_non_preferred_high_score_can_beat_preferred_low_score() -> None:
    catalog = load_document_catalog()
    decision = route_query("enthalpy and vapor pressure", catalog=catalog)
    preferred = SimpleNamespace(
        document_id="smith_van_ness_chemical_engineering_thermodynamics",
        chunk_type=TechnicalChunkType.NARRATIVE,
    )
    other = SimpleNamespace(
        document_id="bird_transport_phenomena",
        chunk_type=TechnicalChunkType.NARRATIVE,
    )

    results = apply_soft_boosts(
        [preferred, other],
        [0.40, 0.70],
        routing=decision,
    )

    assert results[0].result is other


def test_explicit_source_mode_is_the_only_hard_filter() -> None:
    decision = route_query(
        "general question",
        catalog=load_document_catalog(),
        source_mode="becker",
    )

    assert decision.hard_filter == frozenset(
        {"becker_phosphates_and_phosphoric_acid"}
    )


def test_multilingual_query_expansion_preserves_original() -> None:
    result = expand_technical_query(
        "Quel est le rôle de l’échangeur thermique et de la recirculation ?"
    )

    assert result.original_query.startswith("Quel est")
    assert result.dense_query == result.original_query
    assert "heat exchanger" in result.bm25_expanded_query
    assert "recirculation" not in result.added_terms


def test_arabic_query_adds_english_technical_equivalent() -> None:
    result = expand_technical_query(
        "ما هو دور المبادل الحراري في المبخر؟"
    )
    decision = route_query(
        result.original_query,
        catalog=load_document_catalog(),
    )

    assert "heat exchanger" in result.added_terms
    assert (
        decision.preferred_documents[0]
        == "incropera_fundamentals_heat_mass_transfer"
    )


def test_process_expansion_adds_ordered_same_document_context() -> None:
    children = [child(number) for number in range(1, 4)]
    expander = ContextExpander(
        children=children,
        parents=[parent(children)],
    )

    bundles = expander.expand(
        [
            EvidenceAnchor(
                child=children[1],
                score=0.9,
                provenance="reranker",
            )
        ],
        question_type="process_flow",
    )

    assert len(bundles) == 1
    assert bundles[0].expanded_chunk_ids == (
        "chunk_1",
        "chunk_2",
        "chunk_3",
    )
    assert bundles[0].page_start == 1
    assert bundles[0].page_end == 3
    assert bundles[0].parent_included is True


def test_definition_expansion_keeps_anchor_under_budget() -> None:
    children = [child(number) for number in range(1, 4)]
    expander = ContextExpander(
        children=children,
        parents=[parent(children)],
    )

    bundles = expander.expand(
        [EvidenceAnchor(child=children[1], score=0.8, provenance="bm25")],
        question_type="definition",
    )

    assert bundles[0].expanded_chunk_ids == ("chunk_2",)
    assert bundles[0].token_count <= 650


def test_five_oversized_anchors_are_preserved_under_global_budget() -> None:
    children = [
        child(number, document_id=f"document_{number}")
        for number in range(1, 6)
    ]

    for item in children:
        item.token_count = 560

    expander = ContextExpander(
        children=children,
        parents=[],
    )
    bundles = expander.expand(
        [
            EvidenceAnchor(
                child=item,
                score=1.0,
                provenance="reranker",
            )
            for item in children
        ],
        question_type="definition",
    )

    assert len(bundles) == 5
    assert sum(bundle.token_count for bundle in bundles) <= 2600
    assert all(bundle.context_truncated for bundle in bundles)


def test_parent_expansion_never_truncates_away_the_anchor() -> None:
    children = [child(number) for number in range(1, 4)]
    expander = ContextExpander(
        children=children,
        parents=[parent(children)],
        config=ContextExpansionConfig(
            max_tokens_per_bundle=150,
            max_total_context_tokens=750,
        ),
    )

    bundle = expander.expand(
        [
            EvidenceAnchor(
                child=children[1],
                score=1.0,
                provenance="reranker",
            )
        ],
        question_type="process_flow",
    )[0]

    assert children[1].display_text in bundle.display_text
    assert bundle.anchor_chunk_id in bundle.expanded_chunk_ids


def test_process_flow_query_adds_sequence_retrieval_terms() -> None:
    result = expand_technical_query(
        "Décris le trajet de l’acide dans cet équipement.",
        standalone_query=(
            "Décris le trajet de l’acide phosphorique dans un "
            "évaporateur à circulation forcée."
        ),
        question_type="process_flow",
    )

    assert "flow path" in result.dense_query
    assert "circulation pump" in result.bm25_expanded_query
    assert "flash chamber" in result.bm25_expanded_query
    assert "product outlet" in result.added_terms


def test_large_process_anchor_receives_partial_neighbor_context() -> None:
    children = [child(number) for number in range(1, 4)]

    for item in children:
        item.token_count = 560

    expander = ContextExpander(
        children=children,
        parents=[parent(children)],
        config=ContextExpansionConfig(
            max_tokens_per_bundle=650,
            max_total_context_tokens=650,
        ),
    )
    bundle = expander.expand(
        [EvidenceAnchor(child=children[1], score=0.9, provenance="reranker")],
        question_type="process_flow",
    )[0]

    assert bundle.anchor_chunk_id in bundle.expanded_chunk_ids
    assert len(bundle.expanded_chunk_ids) >= 2
    assert bundle.context_token_count > 0
    assert bundle.token_count <= 650
