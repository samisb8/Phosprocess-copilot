"""Tests for the public RAG chat endpoint."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlalchemy import Engine, func, select

from phosprocess.api.database_dependencies import (
    DatabaseRuntimeState,
)
from phosprocess.api.main import create_app
from phosprocess.database.models import (
    ChatMessage,
    ChatSession,
    MessageCitation,
)
from phosprocess.rag.schemas import (
    ChatMessage as RAGChatMessage,
)
from phosprocess.rag.schemas import (
    RAGResponse,
    RAGSource,
    RAGStreamEvent,
    RAGTimings,
)
from tests.support.database import (
    build_test_database_engine,
    check_test_database_connection,
)


class _FakeChatRAGService:
    """Small fake service that avoids loading production models."""

    def __init__(self) -> None:
        self.initial_loading_ms = 100.0
        self.answer_calls: list[tuple[str, str, str]] = []
        self.stream_calls: list[
            tuple[str, list[tuple[str, str]], str, str]
        ] = []
        self.closed = False

    def knowledge_base_status(self) -> dict[str, Any]:
        return {
            "version": "kb-test-chat",
            "document_count": 8,
            "chunk_count": 27_096,
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
        self.answer_calls.append(
            (question, source_mode, language_mode)
        )

        return RAGResponse(
            question=question,
            answer=(
                "La pompe maintient la circulation requise dans "
                "la boucle d'?vaporation [1]."
            ),
            sources=[
                RAGSource(
                    source_number=index,
                    chunk_id=f"becker-{219 + index:03d}-{index:03d}",
                    document_name="Becker",
                    pages=[219 + index, 220 + index],
                    section="Forced-circulation evaporators",
                    excerpt=(
                        "The circulating pump provides the required "
                        "liquid circulation."
                    ),
                    document_title="Phosphates and Phosphoric Acid",
                    filename="becker.pdf",
                    chapter="Evaporation",
                    page_start=219 + index,
                    page_end=220 + index,
                    domain="heat_transfer",
                    chunk_type="text",
                    selection_source="reranker",
                    hybrid_rank=index,
                    rrf_score=1.0 / (60 + index),
                    reranker_rank=index,
                    reranker_score=1.0 - (index * 0.05),
                )
                for index in range(1, 6)
            ],
            cited_source_numbers=[1],
            insufficient_context=False,
            model_name="qwen3:8b",
            selected_variant="production",
            snapshot_sha256="A" * 64,
            candidate_count=20,
            selected_count=5,
            source_policy_route="becker",
            source_policy_mode="automatic",
            source_policy_primary="becker",
            source_policy_fallback_used=False,
            source_policy_forced=False,
            response_language="fr",
            standalone_query=question,
            question_type="technical_definition",
            detected_domains=["heat_transfer"],
            timings=RAGTimings(
                hybrid_ms=100.0,
                reranking_ms=200.0,
                generation_ms=1_500.0,
                total_ms=1_800.0,
                first_token_ms=450.0,
            ),
            latency={},
        )

    def stream_answer(
        self,
        question: str,
        history: list[RAGChatMessage] | None = None,
        *,
        source_mode: str = "automatic",
        language_mode: str = "auto",
    ) -> Iterator[RAGStreamEvent]:
        self.stream_calls.append(
            (
                question,
                [
                    (message.role, message.content)
                    for message in history or []
                ],
                source_mode,
                language_mode,
            )
        )
        yield RAGStreamEvent(
            event_type="completed",
            response=self.answer(
                question,
                source_mode=source_mode,
                language_mode=language_mode,
            ),
        )

    def close(self) -> None:
        self.closed = True


def test_chat_returns_public_rag_response() -> None:
    """The endpoint should persist and expose one complete exchange."""

    service = _FakeChatRAGService()
    application = create_app(
        service_factory=lambda: service,
        warmup_enabled=False,
        database_engine_factory=build_test_database_engine,
        database_health_check=check_test_database_connection,
    )

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "question": " Quel est le r?le de la pompe ? ",
                "source_mode": "automatic",
                "language_mode": "auto",
            },
        )

        assert response.status_code == 200

        body = response.json()

        session_id = UUID(body["session_id"])
        user_message_id = UUID(body["user_message_id"])
        assistant_message_id = UUID(
            body["assistant_message_id"]
        )

        runtime = cast(
            DatabaseRuntimeState,
            application.state.database_runtime,
        )
        assert runtime.session_factory is not None

        with runtime.session_factory() as database_session:
            chat_session = database_session.get(
                ChatSession,
                session_id,
            )
            user_message = database_session.get(
                ChatMessage,
                user_message_id,
            )
            assistant_message = database_session.get(
                ChatMessage,
                assistant_message_id,
            )
            citation_count = database_session.scalar(
                select(func.count())
                .select_from(MessageCitation)
                .where(
                    MessageCitation.message_id
                    == assistant_message_id
                )
            )
            cited_count = database_session.scalar(
                select(func.count())
                .select_from(MessageCitation)
                .where(
                    MessageCitation.message_id
                    == assistant_message_id,
                    MessageCitation.is_cited.is_(True),
                )
            )

    assert body["question"] == "Quel est le r?le de la pompe ?"
    assert body["insufficient_context"] is False
    assert body["model_name"] == "qwen3:8b"
    assert body["cited_source_numbers"] == [1]
    assert body["sources"][0]["document_name"] == "Becker"
    assert body["sources"][0]["pages"] == [220, 221]
    assert body["source_policy"]["primary"] == "becker"
    assert body["timings"]["total_ms"] == 1_800.0

    assert chat_session is not None
    assert chat_session.title == "Quel est le r?le de la pompe ?"

    assert user_message is not None
    assert user_message.role == "user"
    assert user_message.content == "Quel est le r?le de la pompe ?"

    assert assistant_message is not None
    assert assistant_message.role == "assistant"
    assert assistant_message.model_name == "qwen3:8b"

    assert citation_count == 5
    assert cited_count == 1

    assert service.answer_calls == [
        (
            "Quel est le r?le de la pompe ?",
            "automatic",
            "auto",
        )
    ]


def test_chat_appends_to_an_existing_session() -> None:
    """A returned session identifier should continue the conversation."""

    service = _FakeChatRAGService()
    application = create_app(
        service_factory=lambda: service,
        warmup_enabled=False,
        database_engine_factory=build_test_database_engine,
        database_health_check=check_test_database_connection,
    )

    with TestClient(application) as client:
        first_response = client.post(
            "/api/v1/chat",
            json={"question": "Premi?re question"},
        )
        assert first_response.status_code == 200
        first_body = first_response.json()

        second_response = client.post(
            "/api/v1/chat",
            json={
                "question": "Deuxi?me question",
                "session_id": first_body["session_id"],
            },
        )
        assert second_response.status_code == 200
        second_body = second_response.json()

        session_id = UUID(first_body["session_id"])

        runtime = cast(
            DatabaseRuntimeState,
            application.state.database_runtime,
        )
        assert runtime.session_factory is not None

        with runtime.session_factory() as database_session:
            session_count = database_session.scalar(
                select(func.count())
                .select_from(ChatSession)
            )
            message_count = database_session.scalar(
                select(func.count())
                .select_from(ChatMessage)
                .where(ChatMessage.session_id == session_id)
            )
            citation_count = database_session.scalar(
                select(func.count())
                .select_from(MessageCitation)
                .join(ChatMessage)
                .where(ChatMessage.session_id == session_id)
            )
            chat_session = database_session.get(
                ChatSession,
                session_id,
            )

    assert second_body["session_id"] == first_body["session_id"]
    assert (
        second_body["user_message_id"]
        != first_body["user_message_id"]
    )
    assert (
        second_body["assistant_message_id"]
        != first_body["assistant_message_id"]
    )

    assert session_count == 1
    assert message_count == 4
    assert citation_count == 10

    assert chat_session is not None
    assert chat_session.title == "Premi?re question"

    assert service.stream_calls == [
        (
            "Deuxi?me question",
            [
                ("user", "Premi?re question"),
                (
                    "assistant",
                    "La pompe maintient la circulation requise dans "
                    "la boucle d'?vaporation [1].",
                ),
            ],
            "automatic",
            "auto",
        )
    ]


def test_chat_returns_404_for_unknown_session() -> None:
    """An unknown session identifier should not create partial data."""

    service = _FakeChatRAGService()
    application = create_app(
        service_factory=lambda: service,
        warmup_enabled=False,
        database_engine_factory=build_test_database_engine,
        database_health_check=check_test_database_connection,
    )
    unknown_session_id = uuid4()

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "question": "Question technique",
                "session_id": str(unknown_session_id),
            },
        )

        runtime = cast(
            DatabaseRuntimeState,
            application.state.database_runtime,
        )
        assert runtime.session_factory is not None

        with runtime.session_factory() as database_session:
            session_count = database_session.scalar(
                select(func.count())
                .select_from(ChatSession)
            )
            message_count = database_session.scalar(
                select(func.count())
                .select_from(ChatMessage)
            )

    assert response.status_code == 404
    assert response.json() == {
        "detail": (
            f"Chat session '{unknown_session_id}' was not found."
        )
    }
    assert session_count == 0
    assert message_count == 0
    assert service.answer_calls == []
    assert service.stream_calls == []


def test_chat_returns_503_when_database_is_unavailable() -> None:
    """Chat persistence requires a ready database service."""

    service = _FakeChatRAGService()

    def failing_database_factory() -> Engine:
        raise RuntimeError("Simulated database startup failure")

    application = create_app(
        service_factory=lambda: service,
        warmup_enabled=False,
        database_engine_factory=failing_database_factory,
        database_health_check=check_test_database_connection,
    )

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/chat",
            json={"question": "Question technique"},
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Database service is not ready."
    }
    assert service.answer_calls == []


def test_chat_rejects_an_empty_question() -> None:
    """A blank question should fail before reaching the RAG."""

    service = _FakeChatRAGService()
    application = create_app(
        service_factory=lambda: service,
        warmup_enabled=False,
        database_engine_factory=build_test_database_engine,
        database_health_check=check_test_database_connection,
    )

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/chat",
            json={"question": "   "},
        )

    assert response.status_code == 422
    assert service.answer_calls == []


def test_chat_returns_503_when_rag_is_unavailable() -> None:
    """The endpoint should reject requests when startup failed."""

    def failing_factory() -> _FakeChatRAGService:
        raise RuntimeError("Simulated startup failure")

    application = create_app(
        service_factory=failing_factory,
        warmup_enabled=False,
        database_engine_factory=build_test_database_engine,
        database_health_check=check_test_database_connection,
    )

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/chat",
            json={"question": "Question technique"},
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "RAG service is not ready."
    }
