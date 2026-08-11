"""Tests for transactional chat-session management."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from phosprocess.database.base import Base
from phosprocess.database.models import (
    ChatMessage,
    ChatSession,
    MessageCitation,
)
from phosprocess.database.repositories.chat_repository import (
    ChatRepository,
    ChatSessionNotFoundError,
)
from phosprocess.database.services.chat_session_management import (
    ChatSessionManagementService,
)
from phosprocess.database.session import create_session_factory


@pytest.fixture
def session_factory(
) -> Iterator[sessionmaker[Session]]:
    """Provide a shared SQLite database with cascades enabled."""

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(
        database_connection: Any,
        _connection_record: Any,
    ) -> None:
        cursor = database_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)

    try:
        yield factory
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def _seed_conversation(
    factory: sessionmaker[Session],
) -> UUID:
    """Create one session with messages and a citation."""

    created_at = datetime(
        2026,
        7,
        30,
        12,
        0,
        tzinfo=UTC,
    )

    with factory.begin() as database_session:
        chat_session = ChatSession(
            title="Titre original",
            created_at=created_at,
            updated_at=created_at,
        )

        user_message = ChatMessage(
            session=chat_session,
            role="user",
            content="Question utilisateur",
            rag_metadata={},
            created_at=created_at,
        )

        assistant_message = ChatMessage(
            session=chat_session,
            role="assistant",
            content="Réponse assistant [1].",
            rag_metadata={},
            created_at=created_at,
        )

        citation = MessageCitation(
            message=assistant_message,
            source_number=1,
            chunk_id="chunk-1",
            document_name="document.pdf",
            pages=[1],
            excerpt="Extrait documentaire.",
            is_cited=True,
            created_at=created_at,
        )

        database_session.add_all(
            [
                chat_session,
                user_message,
                assistant_message,
                citation,
            ]
        )
        database_session.flush()

        return chat_session.id


def test_rename_session_normalizes_and_persists_title(
    session_factory: sessionmaker[Session],
) -> None:
    """Renaming should normalize spaces and update the record."""

    session_id = _seed_conversation(session_factory)
    service = ChatSessionManagementService(session_factory)

    result = service.rename_session(
        session_id,
        title="  Pompe   de circulation  ",
    )

    assert result.session_id == session_id
    assert result.title == "Pompe de circulation"

    with session_factory() as database_session:
        chat_session = database_session.get(
            ChatSession,
            session_id,
        )

        assert chat_session is not None
        assert chat_session.title == "Pompe de circulation"


@pytest.mark.parametrize(
    "title",
    [
        "   ",
        "x" * 201,
    ],
)
def test_rename_rejects_invalid_title(
    session_factory: sessionmaker[Session],
    title: str,
) -> None:
    """Blank or oversized titles should be rejected."""

    session_id = _seed_conversation(session_factory)
    service = ChatSessionManagementService(session_factory)

    with pytest.raises(ValueError):
        service.rename_session(
            session_id,
            title=title,
        )


def test_rename_unknown_session_raises(
    session_factory: sessionmaker[Session],
) -> None:
    """Renaming an unknown conversation should fail explicitly."""

    service = ChatSessionManagementService(session_factory)

    with pytest.raises(ChatSessionNotFoundError):
        service.rename_session(
            uuid4(),
            title="Nouveau titre",
        )


def test_delete_removes_session_messages_and_citations(
    session_factory: sessionmaker[Session],
) -> None:
    """Deletion should cascade to every dependent entity."""

    session_id = _seed_conversation(session_factory)
    service = ChatSessionManagementService(session_factory)

    service.delete_session(session_id)

    with session_factory() as database_session:
        session_count = database_session.scalar(
            select(func.count()).select_from(ChatSession)
        )
        message_count = database_session.scalar(
            select(func.count()).select_from(ChatMessage)
        )
        citation_count = database_session.scalar(
            select(func.count()).select_from(MessageCitation)
        )

    assert session_count == 0
    assert message_count == 0
    assert citation_count == 0


def test_delete_unknown_session_raises(
    session_factory: sessionmaker[Session],
) -> None:
    """Deleting an unknown conversation should fail explicitly."""

    service = ChatSessionManagementService(session_factory)

    with pytest.raises(ChatSessionNotFoundError):
        service.delete_session(uuid4())


def test_rename_rolls_back_when_flush_fails(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed transaction must preserve the previous title."""

    session_id = _seed_conversation(session_factory)
    service = ChatSessionManagementService(session_factory)

    def failing_flush(
        _repository: ChatRepository,
    ) -> None:
        raise RuntimeError("Simulated database failure")

    monkeypatch.setattr(
        ChatRepository,
        "flush",
        failing_flush,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated database failure",
    ):
        service.rename_session(
            session_id,
            title="Titre non validé",
        )

    with session_factory() as database_session:
        chat_session = database_session.get(
            ChatSession,
            session_id,
        )

        assert chat_session is not None
        assert chat_session.title == "Titre original"
