"""Language, business-state, standalone-query and classifier tests."""

from __future__ import annotations

import pytest

from phosprocess.rag.citations import validate_grounded_answer
from phosprocess.rag.conversation_memory import ConversationMemory
from phosprocess.rag.conversation_state import ConversationState
from phosprocess.rag.followup_resolver import resolve_standalone_query
from phosprocess.rag.language import (
    ResponseLanguage,
    detect_response_language,
)
from phosprocess.rag.question_classifier import (
    QuestionType,
    classify_question,
)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Pourquoi la recirculation est-elle nécessaire ?", ResponseLanguage.FRENCH),
        ("What is the role of the heat exchanger?", ResponseLanguage.ENGLISH),
        ("ما هو دور المبادل الحراري؟", ResponseLanguage.ARABIC),
    ],
)
def test_current_question_controls_response_language(
    question: str,
    expected: ResponseLanguage,
) -> None:
    assert detect_response_language(question).language is expected


def test_language_switch_ignores_previous_conversation_language() -> None:
    english = detect_response_language(
        "What is the role of the heat exchanger?",
        last_explicit_language="fr",
    )
    french = detect_response_language(
        "Quel est le rôle de la pompe ?",
        last_explicit_language="en",
    )

    assert english.language is ResponseLanguage.ENGLISH
    assert french.language is ResponseLanguage.FRENCH


def test_short_question_uses_last_explicit_language() -> None:
    decision = detect_response_language(
        "And why?",
        last_explicit_language="en",
    )

    assert decision.language is ResponseLanguage.ENGLISH


def test_followup_uses_business_entities_but_not_old_evidence() -> None:
    state = ConversationState()
    state.observe_question(
        "Explique l’évaporateur à circulation forcée pour l’acide phosphorique."
    )
    state.last_cited_documents = ("old.pdf",)
    resolution = resolve_standalone_query(
        "Décris le trajet dans cet équipement.",
        state=state,
    )

    assert resolution.followup_detected is True
    assert resolution.resolver_type == "conversation_state"
    assert "évaporateur à circulation forcée" in resolution.standalone_query
    assert "acide phosphorique" in resolution.standalone_query
    assert "old.pdf" not in resolution.standalone_query


def test_evaporator_context_tracks_process_fluid_and_operation() -> None:
    state = ConversationState()
    state.observe_question(
        "C’est quoi un évaporateur à circulation forcée ?"
    )

    assert state.current_equipment == "évaporateur à circulation forcée"
    assert state.current_fluid == "acide phosphorique"
    assert state.current_operation == "concentration par évaporation"


def test_autonomous_question_is_not_rewritten() -> None:
    state = ConversationState(current_equipment="évaporateur")
    question = "Quelle relation relie la pression et la température d’ébullition ?"
    resolution = resolve_standalone_query(question, state=state)

    assert resolution.standalone_query == question
    assert resolution.followup_detected is False


def test_control_question_uses_available_process_state() -> None:
    state = ConversationState()
    state.observe_question(
        "Explique l'évaporateur à circulation forcée pour l'acide phosphorique."
    )

    resolution = resolve_standalone_query(
        "Comment peut-on contrôler la concentration de sortie ?",
        state=state,
    )

    assert resolution.followup_detected is True
    assert resolution.resolver_type == "conversation_state"
    assert "évaporateur à circulation forcée" in resolution.standalone_query
    assert "acide phosphorique" in resolution.standalone_query


def test_memory_clear_resets_business_state() -> None:
    memory = ConversationMemory()
    memory.add_turn(
        "Explique l’évaporateur pour l’acide phosphorique.",
        "Réponse fondée.",
    )
    assert memory.state.current_equipment == "évaporateur"

    memory.clear()

    assert memory.state.current_equipment is None
    assert memory.get_recent_turns() == []


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("C’est quoi la sursaturation ?", QuestionType.DEFINITION),
        ("Décris le trajet étape par étape.", QuestionType.PROCESS_FLOW),
        ("Quelle différence entre boiler et reboiler ?", QuestionType.COMPARISON),
        ("Explique cette équation et ses variables.", QuestionType.EQUATION_EXPLANATION),
        ("Quels signes indiquent un encrassement ?", QuestionType.TROUBLESHOOTING),
        ("Comment régler un MPC ?", QuestionType.CONTROL_STRATEGY),
        (
            "Comment peut-on contrôler la concentration de sortie ?",
            QuestionType.CONTROL_STRATEGY,
        ),
        (
            "Which manipulated variables regulate outlet concentration?",
            QuestionType.CONTROL_STRATEGY,
        ),
        ("What is the boiler?", QuestionType.AMBIGUOUS),
    ],
)
def test_deterministic_question_classification(
    question: str,
    expected: QuestionType,
) -> None:
    assert classify_question(question).question_type is expected


@pytest.mark.parametrize(
    "answer",
    [
        "Les passages retrouvés ne permettent pas de répondre précisément à cette question.",
        (
            "The retrieved passages do not provide enough information to "
            "answer this question precisely."
        ),
        "لا توفر المقاطع المسترجعة معلومات كافية للإجابة عن هذا السؤال بدقة.",
    ],
)
def test_multilingual_controlled_insufficiency(answer: str) -> None:
    citations, insufficient = validate_grounded_answer(
        answer,
        available_source_count=5,
    )

    assert citations == []
    assert insufficient is True
