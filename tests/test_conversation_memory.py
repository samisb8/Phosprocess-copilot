"""Tests for deterministic summary-buffer conversation memory."""

from __future__ import annotations

from io import StringIO

from phosprocess.rag.conversation_memory import ConversationMemory
from phosprocess.rag.terminal_chat import ChatSessionState, handle_command


def make_memory() -> ConversationMemory:
    """Create a deliberately small memory for unit tests."""

    return ConversationMemory(
        recent_turns=2,
        summary_max_tokens=80,
        recent_history_max_tokens=120,
        total_history_max_tokens=200,
    )


def test_only_two_recent_turns_and_old_turns_move_to_summary() -> None:
    memory = make_memory()

    for number in range(1, 5):
        memory.add_turn(
            f"Question métier {number}",
            f"Réponse métier {number} [Source 1].",
        )

    context = memory.build_history_context()

    assert [turn.user for turn in context.recent_turns] == [
        "Question métier 3",
        "Question métier 4",
    ]
    assert "Question métier 1" in context.summary
    assert "Question métier 2" in context.summary
    assert "[Source" not in context.summary


def test_memory_excludes_sources_scores_and_obeys_budgets() -> None:
    memory = make_memory()
    memory.add_turn(
        "Pourquoi la recirculation est-elle utile ?",
        (
            "Elle homogénéise la bouillie [Source 2].\n"
            "Sources: ancien chunk complet\n"
            "reranker: 0.998\n"
            + "Observation opératoire. " * 100
        ),
    )

    context = memory.build_history_context()
    rendered = " ".join(
        f"{turn.user} {turn.assistant}"
        for turn in context.recent_turns
    )

    assert "[Source" not in rendered
    assert "ancien chunk complet" not in rendered
    assert "0.998" not in rendered
    assert context.recent_history_token_count <= 120
    assert context.total_token_count <= 200


def test_staged_messages_keep_order_and_clear_resets_everything() -> None:
    memory = make_memory()
    memory.add_user_message("Question")
    memory.add_assistant_message("Réponse [Source 1].")

    messages = memory.build_history_context().messages()
    assert [message.role for message in messages] == ["user", "assistant"]

    memory.clear()

    assert memory.get_recent_turns() == []
    assert memory.get_summary() == ""
    assert memory.token_usage()["total_tokens"] == 0


def test_history_command_shows_only_summary_buffer_diagnostics() -> None:
    memory = make_memory()
    memory.add_turn(
        "Question 1",
        "Réponse 1 [Source 1]. Sources: contenu secret",
    )
    state = ChatSessionState(memory=memory)
    output = StringIO()

    assert handle_command("/history", state=state, output=output)
    rendered = output.getvalue()

    assert "Mémoire conversationnelle" in rendered
    assert "Fenêtre récente" in rendered
    assert "Tokens estimés" in rendered
    assert "contenu secret" not in rendered
    assert "[Source 1]" not in rendered


def test_clear_command_removes_memory_and_last_sources() -> None:
    memory = make_memory()
    memory.add_turn("Question", "Réponse")
    state = ChatSessionState(memory=memory)
    output = StringIO()

    assert handle_command("/clear", state=state, output=output)
    assert memory.get_recent_turns() == []
    assert memory.get_summary() == ""
    assert state.last_sources == []

