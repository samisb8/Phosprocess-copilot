"""Strict answer-only citation parsing tests."""

from __future__ import annotations

import pytest

from phosprocess.rag.citations import (
    INSUFFICIENT_CONTEXT_ANSWER,
    CitationValidationError,
    extract_citations,
    validate_grounded_answer,
)


def test_one_valid_citation() -> None:
    assert extract_citations(
        "Réponse [Source 1].",
        available_source_count=5,
    ) == [1]


def test_multiple_citations_preserve_first_appearance_order() -> None:
    assert extract_citations(
        "A [Source 4], B [Source 2], C [Source 5].",
        available_source_count=5,
    ) == [4, 2, 5]


def test_repeated_citations_are_deduplicated() -> None:
    assert extract_citations(
        "A [Source 2]. B [Source 2]. C [Source 1].",
        available_source_count=5,
    ) == [2, 1]


@pytest.mark.parametrize(
    "answer",
    [
        "Invalide [Source 0].",
        "Invalide [Source 6].",
        "Invalide [source 1].",
        "Invalide [Source1].",
        "Invalide Source 1.",
        "Invalide [Source 01].",
    ],
)
def test_unknown_or_non_exact_citations_are_rejected(
    answer: str,
) -> None:
    with pytest.raises(CitationValidationError):
        extract_citations(
            answer,
            available_source_count=5,
        )


def test_affirmative_answer_without_citation_is_rejected() -> None:
    with pytest.raises(CitationValidationError):
        validate_grounded_answer(
            "Affirmation métier sans preuve.",
            available_source_count=5,
        )


def test_memory_reference_is_rejected_even_with_valid_source() -> None:
    with pytest.raises(CitationValidationError, match="mémoire"):
        extract_citations(
            "Affirmation [Mémoire] et preuve [Source 1].",
            available_source_count=5,
        )


def test_controlled_insufficient_answer_without_citation_is_accepted() -> None:
    citations, insufficient = validate_grounded_answer(
        INSUFFICIENT_CONTEXT_ANSWER,
        available_source_count=5,
    )

    assert citations == []
    assert insufficient is True


def test_cited_partial_answer_may_end_with_an_insufficiency_notice() -> None:
    answer = (
        "Fouling reduces heat transfer [Source 1]. "
        f"{INSUFFICIENT_CONTEXT_ANSWER}"
    )

    citations, insufficient = validate_grounded_answer(
        answer,
        available_source_count=5,
    )

    assert citations == [1]
    assert insufficient is False
