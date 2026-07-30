"""Repository for persistent chat entities."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from phosprocess.database.models import (
    ChatMessage,
    ChatSession,
    MessageCitation,
)


@dataclass(frozen=True, slots=True)
class ChatSessionSummaryRecord:
    """Database result used to summarize one conversation."""

    chat_session: ChatSession
    message_count: int


class ChatSessionNotFoundError(LookupError):
    """Raised when a requested conversation does not exist."""

    def __init__(self, session_id: UUID) -> None:
        self.session_id = session_id
        super().__init__(
            f"Chat session '{session_id}' was not found."
        )


class ChatRepository:
    """Perform chat persistence operations in one SQLAlchemy session."""

    def __init__(self, database_session: Session) -> None:
        self._database_session = database_session

    def create_session(
        self,
        *,
        title: str | None = None,
    ) -> ChatSession:
        """Create a new persistent conversation."""

        chat_session = ChatSession(title=title)
        self._database_session.add(chat_session)
        self._database_session.flush()

        return chat_session

    def require_session(
        self,
        session_id: UUID,
    ) -> ChatSession:
        """Return an existing conversation or raise an explicit error."""

        chat_session = self._database_session.get(
            ChatSession,
            session_id,
        )

        if chat_session is None:
            raise ChatSessionNotFoundError(session_id)

        return chat_session

    def require_session_with_history(
        self,
        session_id: UUID,
    ) -> ChatSession:
        """Load one conversation with all messages and citations."""

        statement = (
            select(ChatSession)
            .options(
                selectinload(
                    ChatSession.messages
                ).selectinload(
                    ChatMessage.citations
                )
            )
            .where(ChatSession.id == session_id)
        )

        chat_session = self._database_session.scalar(
            statement
        )

        if chat_session is None:
            raise ChatSessionNotFoundError(session_id)

        return chat_session

    def count_sessions(self) -> int:
        """Return the total number of persisted conversations."""

        total = self._database_session.scalar(
            select(func.count()).select_from(ChatSession)
        )

        return int(total or 0)

    def list_session_summaries(
        self,
        *,
        limit: int,
        offset: int,
    ) -> list[ChatSessionSummaryRecord]:
        """Return one ordered and paginated conversation page."""

        statement = (
            select(
                ChatSession,
                func.count(ChatMessage.id),
            )
            .outerjoin(ChatSession.messages)
            .group_by(ChatSession.id)
            .order_by(
                ChatSession.updated_at.desc(),
                ChatSession.created_at.desc(),
            )
            .limit(limit)
            .offset(offset)
        )

        result = self._database_session.execute(statement)

        return [
            ChatSessionSummaryRecord(
                chat_session=chat_session,
                message_count=int(message_count),
            )
            for chat_session, message_count in result
        ]

    def add_message(
        self,
        message: ChatMessage,
    ) -> ChatMessage:
        """Stage one chat message for persistence."""

        self._database_session.add(message)
        return message

    def add_citation(
        self,
        citation: MessageCitation,
    ) -> MessageCitation:
        """Stage one documentary citation for persistence."""

        self._database_session.add(citation)
        return citation

    def flush(self) -> None:
        """Send staged changes to the database transaction."""

        self._database_session.flush()
