"""Hierarchy metadata, intent profiles and retrieval expansion tests."""

from __future__ import annotations

from phosprocess.ingestion.chunk_serialization import (
    TechnicalChunkType,
    TechnicalSection,
)
from phosprocess.rag.question_classifier import QuestionType, classify_question
from phosprocess.retrieval.hierarchical import (
    _PROFILES,
    HierarchicalSectionRetriever,
)
from phosprocess.retrieval.query_expansion import expand_technical_query


def _section(
    *,
    hierarchy_path: str,
    chunk_types: tuple[TechnicalChunkType, ...],
) -> TechnicalSection:
    text = "Representative technical section text."
    return TechnicalSection(
        section_id="section-1",
        document_id="book",
        document_title="Technical Book",
        source_file="book.pdf",
        domains=("equipment",),
        chapter="Chapter 7",
        section="Acid concentration systems",
        subsection="Vapor body",
        hierarchy_path=hierarchy_path,
        page_start=212,
        page_end=220,
        child_chunk_ids=("chunk-1",),
        chunk_types=chunk_types,
        display_text=text,
        embedding_text=f"Hierarchy: {hierarchy_path}\n\n{text}",
        bm25_text=f"{hierarchy_path}\n{hierarchy_path}\n{text}",
        token_count=6,
        sha256="a" * 64,
    )


def test_balance_question_uses_balance_profile_and_expansion() -> None:
    question = (
        "Establish the overall mass balance and the energy balance "
        "of the evaporator at steady state."
    )
    classification = classify_question(question)
    expanded = expand_technical_query(
        question,
        question_type=classification.question_type.value,
    )

    assert classification.question_type is QuestionType.BALANCE
    assert "material balance" in expanded.added_terms
    assert "P2O5 balance" in expanded.added_terms
    assert "steady state" in expanded.bm25_expanded_query.casefold()


def test_process_flow_profile_rewards_descriptive_section() -> None:
    section = _section(
        hierarchy_path=(
            "Phosphates and Phosphoric Acid > Acid Concentration Systems "
            "> Process operation > Vapor body"
        ),
        chunk_types=(TechnicalChunkType.PROCESS_DESCRIPTION,),
    )

    score = HierarchicalSectionRetriever._profile_adjustment(
        section,
        profile=_PROFILES["process_flow"],
        query="Describe the acid flow path through the evaporator.",
    )

    assert score > 0.05


def test_process_flow_profile_penalizes_unrequested_simulation() -> None:
    section = _section(
        hierarchy_path="Workshop report > MATLAB simulation results",
        chunk_types=(TechnicalChunkType.SIMULATION_RESULTS,),
    )

    regular_score = HierarchicalSectionRetriever._profile_adjustment(
        section,
        profile=_PROFILES["process_flow"],
        query="Describe the path of the acid through the evaporator.",
    )
    simulation_score = HierarchicalSectionRetriever._profile_adjustment(
        section,
        profile=_PROFILES["process_flow"],
        query="Explain the MATLAB simulation results for the acid flow.",
    )

    assert regular_score <= -0.18
    assert simulation_score > regular_score


def test_hierarchy_representation_contains_all_levels() -> None:
    hierarchy = (
        "Technical Book > Chapter 7 > Acid concentration systems > Vapor body"
    )
    section = _section(
        hierarchy_path=hierarchy,
        chunk_types=(TechnicalChunkType.EQUIPMENT_DESCRIPTION,),
    )

    assert section.chapter == "Chapter 7"
    assert section.section == "Acid concentration systems"
    assert section.subsection == "Vapor body"
    assert hierarchy in section.embedding_text
    assert section.bm25_text.count(hierarchy) >= 2


def test_process_flow_profile_rewards_feed_section() -> None:
    section = _section(
        hierarchy_path=(
            "Phosphates and Phosphoric Acid > Acid Concentration Systems "
            "> Weak acid feed inlet"
        ),
        chunk_types=(TechnicalChunkType.PROCESS_DESCRIPTION,),
    )

    score = HierarchicalSectionRetriever._profile_adjustment(
        section,
        profile=_PROFILES["process_flow"],
        query="Describe the acid path through the evaporator.",
    )

    assert score >= 0.085


def test_process_flow_dense_hints_cover_full_path() -> None:
    question = (
        "Describe the path through a forced-circulation evaporator, "
        "from the feed inlet to the concentrated product outlet."
    )

    expanded = expand_technical_query(
        question,
        question_type="process_flow",
    )

    dense = expanded.dense_query.casefold()
    assert "weak acid feed" in dense
    assert "circulation pump heat exchanger" in dense
    assert "flash chamber recirculation" in dense
    assert "concentrated acid withdrawal" in dense


def test_process_flow_profile_rewards_product_withdrawal_section() -> None:
    section = _section(
        hierarchy_path=(
            "Phosphates and Phosphoric Acid > Acid Concentration Systems "
            "> Concentrated acid product withdrawal"
        ),
        chunk_types=(TechnicalChunkType.PROCESS_DESCRIPTION,),
    )

    score = HierarchicalSectionRetriever._profile_adjustment(
        section,
        profile=_PROFILES["process_flow"],
        query="Describe the acid path through the evaporator.",
    )

    assert score >= 0.085
