"""SQLAlchemy models for persistent chat conversations."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from phosprocess.database.base import Base

JSON_DOCUMENT = JSON().with_variant(
    JSONB(),
    "postgresql",
)


class ChatSession(Base):
    """One persistent conversation between a client and the assistant."""

    __tablename__ = "chat_sessions"
    __table_args__ = (
        Index(
            "ix_chat_sessions_created_at",
            "created_at",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    title: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ChatMessage.created_at",
    )


class ChatMessage(Base):
    """One user question or assistant response in a chat session."""

    __tablename__ = "chat_messages"
    __table_args__ = (
        CheckConstraint(
            "role IN ('user', 'assistant')",
            name="ck_chat_messages_role_allowed",
        ),
        CheckConstraint(
            "length(trim(content)) > 0",
            name="ck_chat_messages_content_not_blank",
        ),
        CheckConstraint(
            "total_ms IS NULL OR total_ms >= 0",
            name="ck_chat_messages_total_ms_non_negative",
        ),
        Index(
            "ix_chat_messages_session_created_at",
            "session_id",
            "created_at",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    session_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey(
            "chat_sessions.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    insufficient_context: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
    )
    model_name: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    response_language: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
    )
    question_type: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    total_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    rag_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSON_DOCUMENT,
        nullable=False,
        default=dict,
        server_default=text("'{}'"),
    )

    session: Mapped[ChatSession] = relationship(
        back_populates="messages",
    )
    citations: Mapped[list[MessageCitation]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="MessageCitation.source_number",
    )


class MessageCitation(Base):
    """One documentary source attached to an assistant response."""

    __tablename__ = "message_citations"
    __table_args__ = (
        CheckConstraint(
            "source_number > 0",
            name="ck_message_citations_source_number_positive",
        ),
        UniqueConstraint(
            "message_id",
            "source_number",
            name=("uq_message_citations_message_source_number"),
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    message_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey(
            "chat_messages.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    source_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    chunk_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    document_name: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )
    pages: Mapped[list[int]] = mapped_column(
        JSON_DOCUMENT,
        nullable=False,
        default=list,
        server_default=text("'[]'"),
    )
    section: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    excerpt: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    document_title: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )
    filename: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )
    chapter: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    page_start: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    page_end: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    domain: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    chunk_type: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    is_cited: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    message: Mapped[ChatMessage] = relationship(
        back_populates="citations",
    )
