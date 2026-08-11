"""Tests for chat-session management HTTP endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from phosprocess.api.database_dependencies import (
    DatabaseRuntimeState,
)
from phosprocess.api.main import create_app
from phosprocess.database.models import ChatMessage, ChatSession
from phosprocess.rag.schemas import RAGResponse
from tests.support.database import (
    build_test_database_engine,
    check_test_database_connection,
)


class _ManagementRAGService:
    """Minimal RAG service unused by management endpoints."""

    initial_loading_ms = 1.0

    def knowledge_base_status(self) -> dict[str, Any]:
        return {
            "version": "management-test-kb",
            "document_count": 1,
            "chunk_count": 10,
        }

    def warmup(
        self,
        *,
        enabled: bool | None = None,
    ) -> object:
        return object()

    def answer(
        self,
        question: str,
        *,
        source_mode: str = "automatic",
        language_mode: str = "auto",
    ) -> RAGResponse:
        raise AssertionError(
            "Management endpoints must not execute the RAG."
        )

    def close(self) -> None:
        return None


def _seed_session(
    runtime: DatabaseRuntimeState,
) -> UUID:
    """Insert one conversation with two messages."""

    assert runtime.session_factory is not None

    created_at = datetime(
        2026,
        7,
        30,
        13,
        0,
        tzinfo=UTC,
    )

    with runtime.session_factory.begin() as database_session:
        chat_session = ChatSession(
            title="Titre original",
            created_at=created_at,
            updated_at=created_at,
        )

        database_session.add_all(
            [
                chat_session,
                ChatMessage(
                    session=chat_session,
                    role="user",
                    content="Question",
                    rag_metadata={},
                    created_at=created_at,
                ),
                ChatMessage(
                    session=chat_session,
                    role="assistant",
                    content="Réponse",
                    rag_metadata={},
                    created_at=created_at,
                ),
            ]
        )
        database_session.flush()

        return chat_session.id


def test_patch_renames_and_persists_session() -> None:
    """PATCH should normalize and persist the new title."""

    application = create_app(
        service_factory=_ManagementRAGService,
        warmup_enabled=False,
        database_engine_factory=build_test_database_engine,
        database_health_check=check_test_database_connection,
    )

    with TestClient(application) as client:
        runtime = cast(
            DatabaseRuntimeState,
            application.state.database_runtime,
        )
        session_id = _seed_session(runtime)

        response = client.patch(
            f"/api/v1/chat/sessions/{session_id}",
            json={"title": "  Pompe   de circulation  "},
        )

        assert runtime.session_factory is not None

        with runtime.session_factory() as database_session:
            chat_session = database_session.get(
                ChatSession,
                session_id,
            )
            persisted_title = (
                chat_session.title
                if chat_session is not None
                else None
            )

    assert response.status_code == 200
    assert response.json()["session_id"] == str(session_id)
    assert response.json()["title"] == "Pompe de circulation"
    assert persisted_title == "Pompe de circulation"


@pytest.mark.parametrize(
    "title",
    [
        "   ",
        "x" * 201,
    ],
)
def test_patch_rejects_invalid_title(title: str) -> None:
    """Invalid request titles should produce HTTP 422."""

    application = create_app(
        service_factory=_ManagementRAGService,
        warmup_enabled=False,
        database_engine_factory=build_test_database_engine,
        database_health_check=check_test_database_connection,
    )

    with TestClient(application) as client:
        runtime = cast(
            DatabaseRuntimeState,
            application.state.database_runtime,
        )
        session_id = _seed_session(runtime)

        response = client.patch(
            f"/api/v1/chat/sessions/{session_id}",
            json={"title": title},
        )

    assert response.status_code == 422


def test_patch_returns_404_for_unknown_session() -> None:
    """Renaming an unknown conversation should return HTTP 404."""

    application = create_app(
        service_factory=_ManagementRAGService,
        warmup_enabled=False,
        database_engine_factory=build_test_database_engine,
        database_health_check=check_test_database_connection,
    )
    session_id = uuid4()

    with TestClient(application) as client:
        response = client.patch(
            f"/api/v1/chat/sessions/{session_id}",
            json={"title": "Nouveau titre"},
        )

    assert response.status_code == 404


def test_delete_removes_session() -> None:
    """DELETE should remove the conversation and return no body."""

    application = create_app(
        service_factory=_ManagementRAGService,
        warmup_enabled=False,
        database_engine_factory=build_test_database_engine,
        database_health_check=check_test_database_connection,
    )

    with TestClient(application) as client:
        runtime = cast(
            DatabaseRuntimeState,
            application.state.database_runtime,
        )
        session_id = _seed_session(runtime)

        response = client.delete(
            f"/api/v1/chat/sessions/{session_id}"
        )
        history_response = client.get(
            f"/api/v1/chat/sessions/{session_id}"
        )

        assert runtime.session_factory is not None

        with runtime.session_factory() as database_session:
            chat_session = database_session.get(
                ChatSession,
                session_id,
            )

    assert response.status_code == 204
    assert response.content == b""
    assert chat_session is None
    assert history_response.status_code == 404


def test_delete_returns_404_for_unknown_session() -> None:
    """Deleting an unknown conversation should return HTTP 404."""

    application = create_app(
        service_factory=_ManagementRAGService,
        warmup_enabled=False,
        database_engine_factory=build_test_database_engine,
        database_health_check=check_test_database_connection,
    )

    with TestClient(application) as client:
        response = client.delete(
            f"/api/v1/chat/sessions/{uuid4()}"
        )

    assert response.status_code == 404


@pytest.mark.parametrize(
    "method",
    [
        "PATCH",
        "DELETE",
    ],
)
def test_management_requires_database_readiness(
    method: str,
) -> None:
    """Management endpoints should return 503 without PostgreSQL."""

    def failing_database_factory() -> Engine:
        raise RuntimeError("Simulated database failure")

    application = create_app(
        service_factory=_ManagementRAGService,
        warmup_enabled=False,
        database_engine_factory=failing_database_factory,
        database_health_check=check_test_database_connection,
    )
    session_id = uuid4()

    with TestClient(application) as client:
        if method == "PATCH":
            response = client.patch(
                f"/api/v1/chat/sessions/{session_id}",
                json={"title": "Nouveau titre"},
            )
        else:
            response = client.delete(
                f"/api/v1/chat/sessions/{session_id}"
            )

    assert response.status_code == 503


def test_management_remains_available_without_rag() -> None:
    """Renaming and deleting should not depend on RAG readiness."""

    def failing_rag_factory() -> _ManagementRAGService:
        raise RuntimeError("Simulated RAG failure")

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
        session_id = _seed_session(runtime)

        rename_response = client.patch(
            f"/api/v1/chat/sessions/{session_id}",
            json={"title": "Conversation disponible"},
        )
        delete_response = client.delete(
            f"/api/v1/chat/sessions/{session_id}"
        )

    assert rename_response.status_code == 200
    assert delete_response.status_code == 204
