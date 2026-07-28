"""Tests for terminal commands and bounded in-memory conversation."""

from __future__ import annotations

from io import StringIO

import pytest
from scripts.chat_phosprocess import build_parser, configure_utf8_console

from phosprocess.rag.conversation_memory import ConversationMemory
from phosprocess.rag.prompts import limit_history
from phosprocess.rag.schemas import ChatMessage
from phosprocess.rag.terminal_chat import (
    ChatSessionState,
    handle_command,
    print_latency_table,
)


def test_windows_console_is_reconfigured_to_utf8_with_safe_errors() -> None:
    calls: list[dict[str, str]] = []

    class Stream:
        @staticmethod
        def reconfigure(**kwargs: str) -> None:
            calls.append(kwargs)

    configure_utf8_console((Stream(), Stream(), Stream()))

    assert calls == [
        {"encoding": "utf-8", "errors": "replace"},
        {"encoding": "utf-8", "errors": "replace"},
        {"encoding": "utf-8", "errors": "replace"},
    ]


def test_clear_command_removes_history_and_last_sources() -> None:
    output = StringIO()
    state = ChatSessionState()
    state.remember("user", "Question")
    state.remember("assistant", "Réponse [Source 1].")

    should_continue = handle_command(
        "/clear",
        state=state,
        output=output,
    )

    assert should_continue is True
    assert state.history == []
    assert state.last_sources == []
    assert "Historique effacé" in output.getvalue()


def test_exit_command_stops_session() -> None:
    output = StringIO()

    should_continue = handle_command(
        "/exit",
        state=ChatSessionState(),
        output=output,
    )

    assert should_continue is False
    assert "Au revoir" in output.getvalue()


def test_disabled_history_stores_nothing() -> None:
    state = ChatSessionState(
        memory=ConversationMemory(enabled=False),
    )

    state.remember("user", "Question")

    assert state.history == []


def test_history_is_limited_by_messages_and_characters() -> None:
    history = [
        ChatMessage(role="user", content=f"Question {number} " * 10)
        for number in range(8)
    ]

    limited = limit_history(
        history,
        maximum_messages=3,
        maximum_characters=120,
    )

    assert len(limited) <= 3
    assert sum(len(item.content) for item in limited) <= 120
    assert "Question 7" in limited[-1].content


def test_latency_table_is_compact_and_contains_no_prompt() -> None:
    output = StringIO()

    print_latency_table(
        {
            "total_ms": 123.4,
            "estimated_prompt_tokens": 456,
            "document_context_token_count": 300,
            "summary_token_count": 20,
            "recent_history_token_count": 40,
            "ollama_call_count": 1,
            "source_policy_route": "general",
            "source_policy_primary": "Becker",
            "source_policy_fallback_used": False,
        },
        output=output,
    )
    rendered = output.getvalue()

    assert "123.4 ms" in rendered
    assert "prompt=456 tokens" in rendered
    assert "Politique documentaire : general" in rendered
    assert "Source prioritaire : Becker" in rendered
    assert "Fallback utilisé : non" in rendered
    assert "contenu intégral" not in rendered


def test_cli_accepts_latency_and_warmup_switches() -> None:
    arguments = build_parser().parse_args(
        ["--show-latency", "--no-warmup", "--only-source", "becker"]
    )

    assert arguments.show_latency is True
    assert arguments.no_warmup is True
    assert arguments.only_source == "becker"


@pytest.mark.parametrize(
    "source_mode",
    [
        "auto",
        "becker",
        "report",
        "thermodynamics",
        "heat_transfer",
        "perry",
        "crystallization",
        "control",
        "transport",
    ],
)
def test_source_command_changes_session_policy(
    source_mode: str,
) -> None:
    output = StringIO()
    state = ChatSessionState()

    should_continue = handle_command(
        f"/source {source_mode}",
        state=state,
        output=output,
    )

    assert should_continue is True
    assert state.source_mode == source_mode
    assert source_mode in output.getvalue()


def test_language_and_debug_commands_change_session_state() -> None:
    output = StringIO()
    state = ChatSessionState()

    handle_command("/lang ar", state=state, output=output)
    handle_command("/debug on", state=state, output=output)

    assert state.language_mode == "ar"
    assert state.debug_enabled is True

    handle_command("/debug off", state=state, output=output)

    assert state.debug_enabled is False
