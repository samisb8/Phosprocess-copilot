"""Read service for persistent chat conversation history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from phosprocess.database.models import (
    ChatMessage,
    ChatSession,
    MessageCitation,
)
from phosprocess.database.repositories.chat_repository import (
    ChatRepository,
)


@dataclass(frozen=True, slots=True)
class ChatHistoryCitation:
    """Immutable citation returned by the history service."""

    id: UUID
    source_number: int
    chunk_id: str
    document_name: str
    pages: tuple[int, ...]
    section: str | None
    excerpt: str
    document_title: str | None
    filename: str | None
    chapter: str | None
    page_start: int | None
    page_end: int | None
    domain: str | None
    chunk_type: str | None
    is_cited: bool
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ChatHistoryMessage:
    """Immutable message returned by the history service."""

    id: UUID
    role: str
    content: str
    created_at: datetime
    insufficient_context: bool | None
    model_name: str | None
    response_language: str | None
    question_type: str | None
    total_ms: float | None
    citations: tuple[ChatHistoryCitation, ...]


@dataclass(frozen=True, slots=True)
class ChatSessionHistory:
    """Complete immutable representation of one conversation."""

    session_id: UUID
    title: str | None
    created_at: datetime
    updated_at: datetime
    messages: tuple[ChatHistoryMessage, ...]


def _map_citation(
    citation: MessageCitation,
) -> ChatHistoryCitation:
    """Detach one citation from its SQLAlchemy entity."""

    return ChatHistoryCitation(
        id=citation.id,
        source_number=citation.source_number,
        chunk_id=citation.chunk_id,
        document_name=citation.document_name,
        pages=tuple(citation.pages),
        section=citation.section,
        excerpt=citation.excerpt,
        document_title=citation.document_title,
        filename=citation.filename,
        chapter=citation.chapter,
        page_start=citation.page_start,
        page_end=citation.page_end,
        domain=citation.domain,
        chunk_type=citation.chunk_type,
        is_cited=citation.is_cited,
        created_at=citation.created_at,
    )


def _map_message(
    message: ChatMessage,
) -> ChatHistoryMessage:
    """Detach one message and its loaded citations."""

    return ChatHistoryMessage(
        id=message.id,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
        insufficient_context=message.insufficient_context,
        model_name=message.model_name,
        response_language=message.response_language,
        question_type=message.question_type,
        total_ms=message.total_ms,
        citations=tuple(
            _map_citation(citation)
            for citation in message.citations
        ),
    )


def _map_session(
    chat_session: ChatSession,
) -> ChatSessionHistory:
    """Detach a complete loaded conversation."""

    return ChatSessionHistory(
        session_id=chat_session.id,
        title=chat_session.title,
        created_at=chat_session.created_at,
        updated_at=chat_session.updated_at,
        messages=tuple(
            _map_message(message)
            for message in chat_session.messages
        ),
    )


class ChatHistoryService:
    """Read complete conversation histories from the database."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
    ) -> None:
        self._session_factory = session_factory

    def get_session_history(
        self,
        session_id: UUID,
    ) -> ChatSessionHistory:
        """Return one conversation with ordered messages and citations."""

        with self._session_factory() as database_session:
            repository = ChatRepository(database_session)
            chat_session = (
                repository.require_session_with_history(
                    session_id
                )
            )

            return _map_session(chat_session)
