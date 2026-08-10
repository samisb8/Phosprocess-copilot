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
from phosprocess.retrieval.domain_router import (
    requests_automatic_source_scope,
    route_query,
)
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
    question = (
        "\u0645\u0627 \u0647\u0648 \u062f\u0648\u0631 "
        "\u0627\u0644\u0645\u0628\u0627\u062f\u0644 "
        "\u0627\u0644\u062d\u0631\u0627\u0631\u064a "
        "\u0641\u064a "
        "\u0627\u0644\u0645\u0628\u062e\u0631\u061f"
    )

    result = expand_technical_query(
        question
    )

    assert "heat exchanger" in result.added_terms

    decision = route_query(
        result.original_query,
        catalog=load_document_catalog(),
    )

    # Automatic routing understands the query but deliberately
    # leaves document selection to global retrieval + reranking.
    assert decision.source_mode == "auto"
    assert decision.hard_filter is None
    assert decision.preferred_documents == ()
    assert decision.soft_boosts == {}



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


def test_definition_expansion_uses_parent_when_it_fits_budget() -> None:
    children = [child(number) for number in range(1, 4)]
    expander = ContextExpander(
        children=children,
        parents=[parent(children)],
    )

    bundles = expander.expand(
        [EvidenceAnchor(child=children[1], score=0.8, provenance="bm25")],
        question_type="definition",
    )

    assert bundles[0].expanded_chunk_ids == ("chunk_1", "chunk_2", "chunk_3")
    assert bundles[0].token_count <= 650


def test_dynamic_packing_preserves_independent_anchors_within_budget() -> None:
    children = [
        child(number, document_id=f"document_{number}")
        for number in range(1, 6)
    ]

    for item in children:
        item.token_count = 560
        item.parent_id = f"parent_{item.document_id}"

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
    assert all(bundle.context_scope == "anchor_only" for bundle in bundles)


def test_parent_expansion_never_truncates_away_the_anchor() -> None:
    children = [child(number) for number in range(1, 4)]
    expander = ContextExpander(
        children=children,
        parents=[parent(children)],
        config=ContextExpansionConfig(
            max_tokens_per_bundle=250,
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
    assert {"process sequence", "entry", "transition", "exit"}.issubset(
        set(result.added_terms)
    )
    assert "product outlet" not in result.added_terms


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


def test_explicit_becker_wording_hard_filters_only_becker() -> None:
    decision = route_query(
        "Cest quoi un évaporateur à circulation forcée ? Cherche sur Becker.",
        catalog=load_document_catalog(),
        question_type="definition",
    )

    assert decision.source_mode == "becker"
    assert decision.hard_filter == frozenset(
        {"becker_phosphates_and_phosphoric_acid"}
    )




def test_ocp_report_is_routed_for_multiple_internal_technical_domains() -> None:
    catalog = load_document_catalog()
    report = next(
        document
        for document in catalog.documents
        if document.document_id == "ocp_phosphoric_acid_workshop_report"
    )

    assert {
        KnowledgeDomain.PHOSPHORIC_ACID_PROCESS,
        KnowledgeDomain.GENERAL_CHEMICAL_ENGINEERING,
        KnowledgeDomain.THERMODYNAMICS,
        KnowledgeDomain.HEAT_TRANSFER,
        KnowledgeDomain.MASS_TRANSFER,
        KnowledgeDomain.FLUID_MECHANICS,
    }.issubset(set(report.domains))



@pytest.mark.parametrize(
    "question",
    [
        "?tablis le bilan de P2O5 de l'?chelon J de JFC4.",
        "Analyse l'encrassement de l'?changeur de l'atelier OCP JFC4.",
        "Quelle est la capacit? ?vaporatoire r?elle de l'?chelon J ?",
    ],
)
def test_plant_specific_questions_do_not_preselect_a_document(
    question: str,
) -> None:
    decision = route_query(
        question,
        catalog=load_document_catalog(),
    )

    assert decision.source_mode == "auto"
    assert decision.hard_filter is None

    # Critical RAG-05.8 invariant:
    # no question -> book decision before actual retrieval.
    assert decision.preferred_documents == ()
    assert decision.soft_boosts == {}



def test_explicit_report_wording_hard_filters_ocp_report() -> None:
    decision = route_query(
        "Cherche dans le rapport le bilan thermique de l'échelon J.",
        catalog=load_document_catalog(),
    )

    assert decision.source_mode == "report"
    assert decision.hard_filter == frozenset(
        {"ocp_phosphoric_acid_workshop_report"}
    )
    assert decision.section_affinity_terms == ()




def test_kern_seaton_request_targets_report_fouling_section() -> None:
    decision = route_query(
        "Explique le modèle Kern and Seaton dans le rapport OCP.",
        catalog=load_document_catalog(),
    )

    assert decision.source_mode == "report"
    assert decision.hard_filter == frozenset(
        {"ocp_phosphoric_acid_workshop_report"}
    )




def test_explicit_report_balance_has_no_hidden_section_hints() -> None:
    decision = route_query(
        "Établis le bilan de P2O5 de l’échelon J de JFC4 selon le rapport OCP.",
        catalog=load_document_catalog(),
        source_mode="report",
        question_type="balance",
    )

    assert decision.hard_filter == frozenset(
        {"ocp_phosphoric_acid_workshop_report"}
    )
    assert decision.section_affinity_terms == ()


def test_user_can_explicitly_release_a_source_lock() -> None:
    assert requests_automatic_source_scope(
        "Cherche maintenant dans toutes les sources."
    )
    assert requests_automatic_source_scope("Use all sources for this question.")
    assert not requests_automatic_source_scope("Cherche uniquement dans Becker.")


def test_momentum_query_expansion_uses_bird_vocabulary() -> None:
    result = expand_technical_query(
        "Explique la diffusion de quantité de mouvement dans un fluide.",
        question_type="momentum_diffusion",
    )

    assert "molecular transport of momentum" in result.added_terms
    assert "concentration gradient" not in result.bm25_expanded_query


def test_automatic_routing_defers_document_selection_to_retrieval() -> None:
    decision = route_query(
        "Explique le transfert thermique dans un ?vaporateur.",
        catalog=load_document_catalog(),
        question_type="explanation",
    )

    assert decision.source_mode == "auto"
    assert decision.hard_filter is None
    assert decision.preferred_documents == ()
    assert decision.soft_boosts == {}
    assert decision.detected_domains


def test_plant_context_does_not_preselect_the_plant_report() -> None:
    decision = route_query(
        "?tablis le bilan thermique de l'?chelon J de JFC4.",
        catalog=load_document_catalog(),
        question_type="balance",
    )

    assert decision.hard_filter is None
    assert decision.preferred_documents == ()
    assert decision.soft_boosts == {}

    assert decision.source_mode == "auto"


def test_explicit_document_request_still_hard_filters() -> None:
    decision = route_query(
        "Explique le proc?d? uniquement selon Becker.",
        catalog=load_document_catalog(),
        question_type="explanation",
    )

    assert decision.source_mode == "becker"
    assert decision.hard_filter == frozenset(
        {"becker_phosphates_and_phosphoric_acid"}
    )
    assert decision.preferred_documents == (
        "becker_phosphates_and_phosphoric_acid",
    )
