"""Tests for compact context and deterministic follow-up handling."""

from __future__ import annotations

from phosprocess.rag.context_window import (
    prepare_document_context,
    select_relevant_window,
)
from phosprocess.rag.prompts import detect_follow_up, resolve_follow_up
from phosprocess.rag.schemas import ChatMessage


def test_context_keeps_five_sources_in_order_and_within_budget() -> None:
    source_texts = [
        (
            f"Introduction générique de la source {number}. "
            + "Phrase éloignée. " * 100
            + f"La recirculation stabilise la bouillie source {number}. "
            + "Conclusion éloignée. " * 100
        )
        for number in range(1, 6)
    ]

    prepared = prepare_document_context(
        source_texts,
        "Comment la recirculation stabilise-t-elle la bouillie ?",
        maximum_tokens_per_source=50,
        maximum_total_tokens=250,
    )

    assert len(prepared.texts) == 5
    assert len(prepared.tokens_per_source) == 5
    assert prepared.total_tokens <= 250

    for number, text in enumerate(prepared.texts, start=1):
        assert f"source {number}" in text
        assert "recirculation stabilise" in text


def test_context_window_uses_relevant_middle_not_arbitrary_prefix() -> None:
    text = (
        "En-tête sans pertinence. " * 100
        + "Le sulfate mal contrôlé dégrade la croissance cristalline. "
        + "Pied de page sans pertinence. " * 100
    )

    selected = select_relevant_window(
        text,
        "Quels effets produit un sulfate mal contrôlé ?",
        maximum_tokens=45,
    )

    assert "sulfate mal contrôlé" in selected
    assert len(selected) <= 45 * 4


def test_standalone_question_does_not_trigger_reformulation() -> None:
    history = [
        ChatMessage(role="user", content="Ancienne question"),
        ChatMessage(role="assistant", content="Ancienne réponse"),
    ]
    question = "Comment fonctionne la filtration du gypse ?"

    assert detect_follow_up(question, history) is False
    resolution = resolve_follow_up(question, history)
    assert resolution.retrieval_query == question
    assert resolution.method == "none"


def test_simple_follow_up_is_rewritten_without_model_or_old_sources() -> None:
    history = [
        ChatMessage(
            role="user",
            content=(
                "Quel est le rôle de la recirculation dans "
                "le réacteur Jacobs ?"
            ),
        ),
        ChatMessage(
            role="assistant",
            content="Réponse antérieure [Source 3].",
        ),
    ]

    resolution = resolve_follow_up(
        "Et pourquoi améliore-t-elle la stabilité du procédé ?",
        history,
    )

    assert resolution.retrieval_query == (
        "Pourquoi la recirculation améliore-t-elle la stabilité "
        "du réacteur Jacobs ?"
    )
    assert resolution.method == "deterministic_antecedent"
    assert "Réponse antérieure" not in resolution.retrieval_query
    assert "[Source 3]" not in resolution.retrieval_query


def test_complex_follow_up_is_bounded_and_does_not_answer() -> None:
    history = [
        ChatMessage(
            role="user",
            content="Quels problèmes cause un sulfate mal contrôlé ?",
        ),
        ChatMessage(
            role="assistant",
            content="Des problèmes documentés [Source 1].",
        ),
    ]

    resolution = resolve_follow_up(
        "Et dans ce cas, comment peut-on les détecter ?",
        history,
    )

    assert resolution.is_follow_up is True
    assert resolution.method == "deterministic_context"
    assert len(resolution.retrieval_query) < 500
    assert "[Source" not in resolution.retrieval_query
    assert "Des problèmes documentés" not in resolution.retrieval_query


def test_summary_follow_up_retrieves_all_remembered_topics() -> None:
    history = [
        ChatMessage(
            role="user",
            content="Quels problèmes cause le sulfate ?",
        ),
        ChatMessage(role="assistant", content="Réponse récente"),
        ChatMessage(
            role="user",
            content="Comment les détecter ?",
        ),
        ChatMessage(role="assistant", content="Autre réponse"),
    ]
    summary = (
        "Sujet : Quel est le rôle de la recirculation ? "
        "Éléments discutés : stabilité."
    )

    resolution = resolve_follow_up(
        "Résume les points importants discutés jusqu'ici.",
        history,
        summary=summary,
    )

    assert resolution.method == "deterministic_summary_topics"
    assert "recirculation" in resolution.retrieval_query
    assert "sulfate" in resolution.retrieval_query
    assert "Comment les détecter" in resolution.retrieval_query
    assert "stabilité" not in resolution.retrieval_query
