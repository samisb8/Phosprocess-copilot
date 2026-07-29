"""Repository for persistent chat entities."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from phosprocess.database.models import (
    ChatMessage,
    ChatSession,
    MessageCitation,
)


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
