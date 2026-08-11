"""Tests for paginated persistent chat-session listings."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from phosprocess.database.base import Base
from phosprocess.database.models import ChatMessage, ChatSession
from phosprocess.database.services.chat_session_listing import (
    ChatSessionListingService,
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
    session_factory = create_session_factory(engine)

    try:
        yield session_factory, engine
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def _seed_session(
    session_factory: sessionmaker[Session],
    *,
    title: str,
    updated_at: datetime,
    message_count: int,
) -> UUID:
    """Insert one conversation with a known message count."""

    with session_factory.begin() as database_session:
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
                    role="user" if index % 2 == 0 else "assistant",
                    content=f"Message {index + 1}",
                    rag_metadata={},
                    created_at=updated_at
                    - timedelta(seconds=message_count - index),
                )
            )

        database_session.flush()
        return chat_session.id


def test_listing_orders_sessions_and_counts_messages(
    database_resources: tuple[
        sessionmaker[Session],
        Engine,
    ],
) -> None:
    """Sessions should be ordered by latest activity."""

    session_factory, _engine = database_resources
    base_time = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)

    oldest_id = _seed_session(
        session_factory,
        title="Ancienne conversation",
        updated_at=base_time,
        message_count=1,
    )
    newest_id = _seed_session(
        session_factory,
        title="Conversation récente",
        updated_at=base_time + timedelta(hours=2),
        message_count=3,
    )
    middle_id = _seed_session(
        session_factory,
        title="Conversation intermédiaire",
        updated_at=base_time + timedelta(hours=1),
        message_count=2,
    )

    service = ChatSessionListingService(session_factory)
    page = service.list_sessions()

    assert page.total == 3
    assert page.limit == 20
    assert page.offset == 0

    assert [
        item.session_id
        for item in page.items
    ] == [
        newest_id,
        middle_id,
        oldest_id,
    ]

    assert [
        item.message_count
        for item in page.items
    ] == [3, 2, 1]


def test_listing_applies_limit_and_offset(
    database_resources: tuple[
        sessionmaker[Session],
        Engine,
    ],
) -> None:
    """Pagination should return only the requested window."""

    session_factory, _engine = database_resources
    base_time = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)

    _seed_session(
        session_factory,
        title="Session 1",
        updated_at=base_time,
        message_count=1,
    )
    expected_id = _seed_session(
        session_factory,
        title="Session 2",
        updated_at=base_time + timedelta(hours=1),
        message_count=2,
    )
    _seed_session(
        session_factory,
        title="Session 3",
        updated_at=base_time + timedelta(hours=2),
        message_count=3,
    )

    service = ChatSessionListingService(session_factory)
    page = service.list_sessions(limit=1, offset=1)

    assert page.total == 3
    assert page.limit == 1
    assert page.offset == 1
    assert len(page.items) == 1
    assert page.items[0].session_id == expected_id


def test_listing_returns_empty_page(
    database_resources: tuple[
        sessionmaker[Session],
        Engine,
    ],
) -> None:
    """An empty database should return a valid empty page."""

    session_factory, _engine = database_resources
    service = ChatSessionListingService(session_factory)

    page = service.list_sessions()

    assert page.total == 0
    assert page.items == ()


@pytest.mark.parametrize(
    ("limit", "offset"),
    [
        (0, 0),
        (101, 0),
        (20, -1),
    ],
)
def test_listing_rejects_invalid_pagination(
    database_resources: tuple[
        sessionmaker[Session],
        Engine,
    ],
    limit: int,
    offset: int,
) -> None:
    """Invalid pagination values should be rejected."""

    session_factory, _engine = database_resources
    service = ChatSessionListingService(session_factory)

    with pytest.raises(ValueError):
        service.list_sessions(
            limit=limit,
            offset=offset,
        )


def test_listing_uses_two_select_queries(
    database_resources: tuple[
        sessionmaker[Session],
        Engine,
    ],
) -> None:
    """Listing should use one count query and one page query."""

    session_factory, engine = database_resources

    _seed_session(
        session_factory,
        title="Session test",
        updated_at=datetime(2026, 7, 30, 12, 0, tzinfo=UTC),
        message_count=4,
    )

    statements: list[str] = []

    def capture_statement(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    event.listen(
        engine,
        "before_cursor_execute",
        capture_statement,
    )

    try:
        service = ChatSessionListingService(session_factory)
        service.list_sessions()
    finally:
        event.remove(
            engine,
            "before_cursor_execute",
            capture_statement,
        )

    assert len(statements) == 2
