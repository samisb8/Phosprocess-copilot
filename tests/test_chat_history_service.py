"""Tests for optimized persistent chat history reads."""

from __future__ import annotations

from collections.abc import Iterator
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Engine, create_engine, event
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
from phosprocess.database.services.chat_history import (
    ChatHistoryService,
)
from phosprocess.database.session import create_session_factory


@pytest.fixture
def database_resources(
) -> Iterator[tuple[sessionmaker[Session], Engine]]:
    """Provide an isolated shared in-memory database."""

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)

    try:
        yield factory, engine
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def _seed_history(
    session_factory: sessionmaker[Session],
) -> UUID:
    """Insert one deterministic conversation."""

    with session_factory.begin() as database_session:
        chat_session = ChatSession(
            title="Fonctionnement de la pompe"
        )

        user_message = ChatMessage(
            session=chat_session,
            role="user",
            content="Quel est le r?le de la pompe ?",
            rag_metadata={},
        )

        assistant_message = ChatMessage(
            session=chat_session,
            role="assistant",
            content="Elle maintient la circulation [1].",
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
                ),
                MessageCitation(
                    message=assistant_message,
                    source_number=1,
                    chunk_id="chunk-1",
                    document_name="Becker",
                    pages=[220],
                    excerpt="Primary cited source.",
                    is_cited=True,
                ),
            ]
        )
        database_session.flush()

        return chat_session.id


def test_history_returns_ordered_messages_and_citations(
    database_resources: tuple[
        sessionmaker[Session],
        Engine,
    ],
) -> None:
    """The service should return the complete detached history."""

    session_factory, _engine = database_resources
    session_id = _seed_history(session_factory)

    service = ChatHistoryService(session_factory)
    history = service.get_session_history(session_id)

    assert history.session_id == session_id
    assert history.title == "Fonctionnement de la pompe"
    assert len(history.messages) == 2

    user_message, assistant_message = history.messages

    assert user_message.role == "user"
    assert user_message.citations == ()

    assert assistant_message.role == "assistant"
    assert assistant_message.model_name == "qwen3:8b"
    assert assistant_message.total_ms == 1_800.0

    assert [
        citation.source_number
        for citation in assistant_message.citations
    ] == [1, 2]

    assert [
        citation.is_cited
        for citation in assistant_message.citations
    ] == [True, False]


def test_history_raises_for_unknown_session(
    database_resources: tuple[
        sessionmaker[Session],
        Engine,
    ],
) -> None:
    """An unknown conversation should produce an explicit error."""

    session_factory, _engine = database_resources
    service = ChatHistoryService(session_factory)
    unknown_session_id = uuid4()

    with pytest.raises(
        ChatSessionNotFoundError,
        match=str(unknown_session_id),
    ):
        service.get_session_history(unknown_session_id)


def test_history_uses_three_select_queries(
    database_resources: tuple[
        sessionmaker[Session],
        Engine,
    ],
) -> None:
    """Select-in loading should avoid one query per message."""

    session_factory, engine = database_resources
    session_id = _seed_history(session_factory)
    statements: list[str] = []

    def capture_statement(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        normalized = statement.lstrip().upper()

        if normalized.startswith("SELECT"):
            statements.append(statement)

    event.listen(
        engine,
        "before_cursor_execute",
        capture_statement,
    )

    try:
        service = ChatHistoryService(session_factory)
        service.get_session_history(session_id)
    finally:
        event.remove(
            engine,
            "before_cursor_execute",
            capture_statement,
        )

    assert len(statements) == 3
