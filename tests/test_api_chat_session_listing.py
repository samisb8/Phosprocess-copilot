"""Tests for the paginated chat-session listing endpoint."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID

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


class _ListingRAGService:
    """Minimal RAG service unused by the listing endpoint."""

    initial_loading_ms = 1.0

    def knowledge_base_status(self) -> dict[str, Any]:
        return {
            "version": "listing-test-kb",
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
            "The listing endpoint must not execute the RAG."
        )

    def close(self) -> None:
        return None


def _seed_session(
    runtime: DatabaseRuntimeState,
    *,
    title: str,
    updated_at: datetime,
    message_count: int,
) -> UUID:
    """Insert one conversation with a known message count."""

    assert runtime.session_factory is not None

    with runtime.session_factory.begin() as database_session:
        chat_session = ChatSession(
            title=title,
            created_at=updated_at - timedelta(minutes=5),
            updated_at=updated_at,
        )

        database_session.add(chat_session)

        for index in range(message_count):
            database_session.add(
                ChatMessage(
                    session=chat_session,
                    role=(
                        "user"
                        if index % 2 == 0
                        else "assistant"
                    ),
                    content=f"Message {index + 1}",
                    created_at=(
                        updated_at
                        - timedelta(
                            seconds=message_count - index
                        )
                    ),
                    rag_metadata={},
                )
            )

        database_session.flush()
        return chat_session.id


def test_listing_endpoint_returns_paginated_sessions() -> None:
    """Sessions should be ordered by latest activity."""

    application = create_app(
        service_factory=_ListingRAGService,
        warmup_enabled=False,
        database_engine_factory=build_test_database_engine,
        database_health_check=check_test_database_connection,
    )

    base_time = datetime(
        2026,
        7,
        30,
        12,
        0,
        tzinfo=UTC,
    )

    with TestClient(application) as client:
        runtime = cast(
            DatabaseRuntimeState,
            application.state.database_runtime,
        )

        oldest_id = _seed_session(
            runtime,
            title="Ancienne conversation",
            updated_at=base_time,
            message_count=1,
        )
        middle_id = _seed_session(
            runtime,
            title="Conversation intermédiaire",
            updated_at=base_time + timedelta(hours=1),
            message_count=2,
        )
        newest_id = _seed_session(
            runtime,
            title="Conversation récente",
            updated_at=base_time + timedelta(hours=2),
            message_count=3,
        )

        first_page = client.get(
            "/api/v1/chat/sessions?limit=2&offset=0"
        )
        second_page = client.get(
            "/api/v1/chat/sessions?limit=2&offset=2"
        )

    assert first_page.status_code == 200
    first_body = first_page.json()

    assert first_body["total"] == 3
    assert first_body["limit"] == 2
    assert first_body["offset"] == 0

    assert [
        item["session_id"]
        for item in first_body["items"]
    ] == [
        str(newest_id),
        str(middle_id),
    ]

    assert [
        item["message_count"]
        for item in first_body["items"]
    ] == [3, 2]

    assert second_page.status_code == 200
    second_body = second_page.json()

    assert second_body["total"] == 3
    assert second_body["offset"] == 2
    assert len(second_body["items"]) == 1
    assert second_body["items"][0]["session_id"] == str(
        oldest_id
    )
    assert second_body["items"][0]["message_count"] == 1


def test_listing_endpoint_returns_empty_page() -> None:
    """An empty database should return an empty valid page."""

    application = create_app(
        service_factory=_ListingRAGService,
        warmup_enabled=False,
        database_engine_factory=build_test_database_engine,
        database_health_check=check_test_database_connection,
    )

    with TestClient(application) as client:
        response = client.get("/api/v1/chat/sessions")

    assert response.status_code == 200
    assert response.json() == {
        "items": [],
        "total": 0,
        "limit": 20,
        "offset": 0,
    }


@pytest.mark.parametrize(
    "query",
    [
        "limit=0",
        "limit=101",
        "offset=-1",
    ],
)
def test_listing_endpoint_rejects_invalid_pagination(
    query: str,
) -> None:
    """FastAPI should reject invalid query parameters."""

    application = create_app(
        service_factory=_ListingRAGService,
        warmup_enabled=False,
        database_engine_factory=build_test_database_engine,
        database_health_check=check_test_database_connection,
    )

    with TestClient(application) as client:
        response = client.get(
            f"/api/v1/chat/sessions?{query}"
        )

    assert response.status_code == 422


def test_listing_endpoint_requires_database_readiness() -> None:
    """An unavailable database should return HTTP 503."""

    def failing_database_factory() -> Engine:
        raise RuntimeError(
            "Simulated database startup failure"
        )

    application = create_app(
        service_factory=_ListingRAGService,
        warmup_enabled=False,
        database_engine_factory=failing_database_factory,
        database_health_check=check_test_database_connection,
    )

    with TestClient(application) as client:
        response = client.get("/api/v1/chat/sessions")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Database service is not ready."
    }


def test_listing_remains_available_when_rag_is_unavailable() -> None:
    """Listing persisted sessions should not depend on the RAG."""

    def failing_rag_factory() -> _ListingRAGService:
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

        session_id = _seed_session(
            runtime,
            title="Conversation disponible",
            updated_at=datetime.now(UTC),
            message_count=2,
        )

        response = client.get("/api/v1/chat/sessions")

    assert response.status_code == 200
    assert response.json()["items"][0]["session_id"] == str(
        session_id
    )
