"""Tests for transactional RAG conversation persistence."""

from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from phosprocess.database.base import Base
from phosprocess.database.models import (
    ChatMessage,
    ChatSession,
    MessageCitation,
)
from phosprocess.database.repositories.chat_repository import (
    ChatSessionNotFoundError,
)
from phosprocess.database.services.chat_persistence import (
    ChatPersistenceService,
)
from phosprocess.database.session import create_session_factory
from phosprocess.rag.schemas import (
    RAGResponse,
    RAGSource,
    RAGTimings,
)


@pytest.fixture
def session_factory() -> Iterator[sessionmaker[Session]]:
    """Provide an isolated in-memory database."""

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)

    try:
        yield factory
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def _build_response(
    question: str = "Quel est le r?le de la pompe ?",
) -> RAGResponse:
    """Build a deterministic response for persistence tests."""

    sources = [
        RAGSource(
            source_number=index,
            chunk_id=f"chunk-{index}",
            document_name="Becker",
            pages=[219 + index],
            section="Forced-circulation evaporators",
            excerpt="The pump maintains liquid circulation.",
            document_title="Phosphates and Phosphoric Acid",
            filename="becker.pdf",
            chapter="Evaporation",
            page_start=219 + index,
            page_end=219 + index,
            domain="heat_transfer",
            chunk_type="text",
            selection_source="reranker",
            hybrid_rank=index,
            rrf_score=1.0 / (60 + index),
            reranker_rank=index,
            reranker_score=1.0 - (index * 0.05),
        )
        for index in range(1, 6)
    ]

    return RAGResponse(
        question=question,
        answer="La pompe maintient la circulation [1].",
        sources=sources,
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


def test_persist_exchange_creates_complete_conversation(
    session_factory: sessionmaker[Session],
) -> None:
    """A new exchange should persist all related entities."""

    service = ChatPersistenceService(session_factory)

    result = service.persist_exchange(
        response=_build_response()
    )

    with session_factory() as database_session:
        chat_session = database_session.get(
            ChatSession,
            result.session_id,
        )
        user_message = database_session.get(
            ChatMessage,
            result.user_message_id,
        )
        assistant_message = database_session.get(
            ChatMessage,
            result.assistant_message_id,
        )
        citations = database_session.scalars(
            select(MessageCitation)
            .where(
                MessageCitation.message_id
                == result.assistant_message_id
            )
            .order_by(MessageCitation.source_number)
        ).all()

    assert chat_session is not None
    assert chat_session.title == "Quel est le r?le de la pompe ?"

    assert user_message is not None
    assert user_message.role == "user"

    assert assistant_message is not None
    assert assistant_message.role == "assistant"
    assert assistant_message.total_ms == 1_800.0
    assert assistant_message.rag_metadata[
        "candidate_count"
    ] == 20

    assert len(citations) == 5
    assert [citation.is_cited for citation in citations] == [
        True,
        False,
        False,
        False,
        False,
    ]


def test_persist_exchange_appends_to_existing_session(
    session_factory: sessionmaker[Session],
) -> None:
    """A supplied session identifier should reuse the conversation."""

    service = ChatPersistenceService(session_factory)

    first = service.persist_exchange(
        response=_build_response("Premi?re question")
    )
    second = service.persist_exchange(
        response=_build_response("Deuxi?me question"),
        session_id=first.session_id,
    )

    with session_factory() as database_session:
        session_count = database_session.scalar(
            select(func.count())
            .select_from(ChatSession)
        )
        message_count = database_session.scalar(
            select(func.count())
            .select_from(ChatMessage)
            .where(
                ChatMessage.session_id == first.session_id
            )
        )

    assert second.session_id == first.session_id
    assert session_count == 1
    assert message_count == 4


def test_unknown_session_rolls_back_exchange(
    session_factory: sessionmaker[Session],
) -> None:
    """An unknown conversation should not persist partial data."""

    service = ChatPersistenceService(session_factory)

    with pytest.raises(ChatSessionNotFoundError):
        service.persist_exchange(
            response=_build_response(),
            session_id=uuid4(),
        )

    with session_factory() as database_session:
        message_count = database_session.scalar(
            select(func.count())
            .select_from(ChatMessage)
        )

    assert message_count == 0


def test_constraint_failure_rolls_back_entire_exchange(
    session_factory: sessionmaker[Session],
) -> None:
    """A citation error should roll back session and messages."""

    response = _build_response()
    duplicated_source = response.sources[0]

    invalid_response = response.model_copy(
        update={
            "sources": [
                duplicated_source,
                duplicated_source,
            ]
        }
    )

    service = ChatPersistenceService(session_factory)

    with pytest.raises(IntegrityError):
        service.persist_exchange(
            response=invalid_response
        )

    with session_factory() as database_session:
        session_count = database_session.scalar(
            select(func.count())
            .select_from(ChatSession)
        )
        message_count = database_session.scalar(
            select(func.count())
            .select_from(ChatMessage)
        )
        citation_count = database_session.scalar(
            select(func.count())
            .select_from(MessageCitation)
        )

    assert session_count == 0
    assert message_count == 0
    assert citation_count == 0
