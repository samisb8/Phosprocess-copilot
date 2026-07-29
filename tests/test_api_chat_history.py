"""Tests for the persisted chat history HTTP endpoint."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlalchemy import Engine

from phosprocess.api.database_dependencies import (
    DatabaseRuntimeState,
)
from phosprocess.api.main import create_app
from phosprocess.database.models import (
    ChatMessage,
    ChatSession,
    MessageCitation,
)
from phosprocess.rag.schemas import RAGResponse
from tests.support.database import (
    build_test_database_engine,
    check_test_database_connection,
)


class _HistoryRAGService:
    """Minimal RAG service for history endpoint lifecycle tests."""

    initial_loading_ms = 1.0

    def knowledge_base_status(self) -> dict[str, Any]:
        return {
            "version": "history-test-kb",
            "document_count": 1,
            "chunk_count": 10,
        }

    def warmup(self, *, enabled: bool | None = None) -> object:
        return object()

    def answer(
        self,
        question: str,
        *,
        source_mode: str = "automatic",
        language_mode: str = "auto",
    ) -> RAGResponse:
        raise AssertionError(
            "The history endpoint must not execute the RAG."
        )

    def close(self) -> None:
        return None


def _seed_history(
    runtime: DatabaseRuntimeState,
) -> UUID:
    """Insert one deterministic conversation in the test database."""

    assert runtime.session_factory is not None

    created_at = datetime(
        2026,
        7,
        30,
        0,
        0,
        tzinfo=UTC,
    )

    with runtime.session_factory.begin() as database_session:
        chat_session = ChatSession(
            title="Pompe de circulation",
            created_at=created_at,
            updated_at=created_at + timedelta(seconds=3),
        )

        user_message = ChatMessage(
            session=chat_session,
            role="user",
            content="Quel est le r?le de la pompe ?",
            created_at=created_at + timedelta(seconds=1),
            rag_metadata={},
        )

        assistant_message = ChatMessage(
            session=chat_session,
            role="assistant",
            content="Elle maintient la circulation [1].",
            created_at=created_at + timedelta(seconds=2),
            insufficient_context=False,
            model_name="qwen3:8b",
            response_language="fr",
            question_type="explanation",
            total_ms=1_800.0,
            rag_metadata={},
        )

        database_session.add_all(
            [
                chat_session,
                user_message,
                assistant_message,
            ]
        )
        database_session.flush()

        database_session.add_all(
            [
                MessageCitation(
                    message=assistant_message,
                    source_number=2,
                    chunk_id="chunk-2",
                    document_name="Perry",
                    pages=[221],
                    excerpt="Second source.",
                    is_cited=False,
                    created_at=created_at + timedelta(seconds=2),
                ),
                MessageCitation(
                    message=assistant_message,
                    source_number=1,
                    chunk_id="chunk-1",
                    document_name="Becker",
                    pages=[220],
                    excerpt="Primary cited source.",
                    is_cited=True,
                    created_at=created_at + timedelta(seconds=2),
                ),
            ]
        )
        database_session.flush()

        return chat_session.id


def test_history_endpoint_returns_complete_conversation() -> None:
    """The endpoint should expose ordered messages and citations."""

    application = create_app(
        service_factory=_HistoryRAGService,
        warmup_enabled=False,
        database_engine_factory=build_test_database_engine,
        database_health_check=check_test_database_connection,
    )

    with TestClient(application) as client:
        runtime = cast(
            DatabaseRuntimeState,
            application.state.database_runtime,
        )
        session_id = _seed_history(runtime)

        response = client.get(
            f"/api/v1/chat/sessions/{session_id}"
        )

    assert response.status_code == 200

    body = response.json()

    assert body["session_id"] == str(session_id)
    assert body["title"] == "Pompe de circulation"
    assert len(body["messages"]) == 2

    user_message, assistant_message = body["messages"]

    assert user_message["role"] == "user"
    assert user_message["citations"] == []

    assert assistant_message["role"] == "assistant"
    assert assistant_message["model_name"] == "qwen3:8b"
    assert assistant_message["total_ms"] == 1_800.0

    assert [
        citation["source_number"]
        for citation in assistant_message["citations"]
    ] == [1, 2]

    assert [
        citation["is_cited"]
        for citation in assistant_message["citations"]
    ] == [True, False]


def test_history_endpoint_returns_404_for_unknown_session() -> None:
    """An unknown conversation should return HTTP 404."""

    application = create_app(
        service_factory=_HistoryRAGService,
        warmup_enabled=False,
        database_engine_factory=build_test_database_engine,
        database_health_check=check_test_database_connection,
    )
    unknown_session_id = uuid4()

    with TestClient(application) as client:
        response = client.get(
            f"/api/v1/chat/sessions/{unknown_session_id}"
        )

    assert response.status_code == 404
    assert response.json() == {
        "detail": (
            f"Chat session '{unknown_session_id}' was not found."
        )
    }


def test_history_endpoint_requires_database_readiness() -> None:
    """An unavailable database should produce HTTP 503."""

    def failing_database_factory() -> Engine:
        raise RuntimeError("Simulated database startup failure")

    application = create_app(
        service_factory=_HistoryRAGService,
        warmup_enabled=False,
        database_engine_factory=failing_database_factory,
        database_health_check=check_test_database_connection,
    )

    with TestClient(application) as client:
        response = client.get(
            f"/api/v1/chat/sessions/{uuid4()}"
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Database service is not ready."
    }


def test_history_remains_available_when_rag_is_unavailable() -> None:
    """Reading persisted data should not depend on RAG readiness."""

    def failing_rag_factory() -> _HistoryRAGService:
        raise RuntimeError("Simulated RAG startup failure")

    application = create_app(
        service_factory=failing_rag_factory,
        warmup_enabled=False,
        database_engine_factory=build_test_database_engine,
        database_health_check=check_test_database_connection,
    )

    with TestClient(application) as client:
        runtime = cast(
            DatabaseRuntimeState,
            application.state.database_runtime,
        )
        session_id = _seed_history(runtime)

        response = client.get(
            f"/api/v1/chat/sessions/{session_id}"
        )

    assert response.status_code == 200
    assert response.json()["session_id"] == str(session_id)
