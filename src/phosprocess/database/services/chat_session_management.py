"""Transactional management of persistent chat sessions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from phosprocess.database.repositories.chat_repository import (
    ChatRepository,
)


@dataclass(frozen=True, slots=True)
class RenamedChatSession:
    """Result returned after a conversation is renamed."""

    session_id: UUID
    title: str
    updated_at: datetime


def _normalize_title(title: str) -> str:
    """Normalize and validate one conversation title."""

    normalized = " ".join(title.split())

    if not normalized:
        raise ValueError(
            "The chat session title must not be empty."
        )

    if len(normalized) > 200:
        raise ValueError(
            "The chat session title must not exceed 200 characters."
        )

    return normalized


class ChatSessionManagementService:
    """Rename and delete conversations transactionally."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
    ) -> None:
        self._session_factory = session_factory

    def rename_session(
        self,
        session_id: UUID,
        *,
        title: str,
    ) -> RenamedChatSession:
        """Rename one existing conversation atomically."""

        normalized_title = _normalize_title(title)
        updated_at = datetime.now(UTC)

        with self._session_factory.begin() as database_session:
            repository = ChatRepository(database_session)
            chat_session = repository.require_session(session_id)

            repository.rename_session(
                chat_session,
                title=normalized_title,
                updated_at=updated_at,
            )
            repository.flush()

            return RenamedChatSession(
                session_id=chat_session.id,
                title=normalized_title,
                updated_at=chat_session.updated_at,
            )

    def delete_session(
        self,
        session_id: UUID,
    ) -> None:
        """Delete one conversation and all dependent records."""

        with self._session_factory.begin() as database_session:
            repository = ChatRepository(database_session)
            chat_session = repository.require_session(session_id)

            repository.delete_session(chat_session)
            repository.flush()
