"""Service for paginated persistent chat-session listings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from phosprocess.database.repositories.chat_repository import (
    ChatRepository,
    ChatSessionSummaryRecord,
)


@dataclass(frozen=True, slots=True)
class ChatSessionSummary:
    """Detached summary of one persisted conversation."""

    session_id: UUID
    title: str | None
    created_at: datetime
    updated_at: datetime
    message_count: int


@dataclass(frozen=True, slots=True)
class ChatSessionPage:
    """One paginated group of conversation summaries."""

    items: tuple[ChatSessionSummary, ...]
    total: int
    limit: int
    offset: int


def _map_summary(
    record: ChatSessionSummaryRecord,
) -> ChatSessionSummary:
    """Detach one summary from its SQLAlchemy entity."""

    chat_session = record.chat_session

    return ChatSessionSummary(
        session_id=chat_session.id,
        title=chat_session.title,
        created_at=chat_session.created_at,
        updated_at=chat_session.updated_at,
        message_count=record.message_count,
    )


class ChatSessionListingService:
    """List persisted conversations with bounded pagination."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
    ) -> None:
        self._session_factory = session_factory

    def list_sessions(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> ChatSessionPage:
        """Return conversations ordered by latest activity."""

        if not 1 <= limit <= 100:
            raise ValueError("The limit must be between 1 and 100.")

        if offset < 0:
            raise ValueError("The offset must be greater than or equal to zero.")

        with self._session_factory() as database_session:
            repository = ChatRepository(database_session)

            total = repository.count_sessions()
            records = repository.list_session_summaries(
                limit=limit,
                offset=offset,
            )

            return ChatSessionPage(
                items=tuple(_map_summary(record) for record in records),
                total=total,
                limit=limit,
                offset=offset,
            )
