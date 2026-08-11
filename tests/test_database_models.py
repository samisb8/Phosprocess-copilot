"""Tests for the initial persistent chat database schema."""

from phosprocess.database.base import Base
from phosprocess.database.models import (
    ChatMessage,
    ChatSession,
    MessageCitation,
)


def test_chat_tables_are_registered_in_metadata() -> None:
    """All initial persistence tables should be known by Alembic."""

    assert {
        "chat_sessions",
        "chat_messages",
        "message_citations",
    }.issubset(Base.metadata.tables)


def test_chat_message_references_chat_session() -> None:
    """Deleting a session should cascade to its messages."""

    foreign_key = next(
        iter(ChatMessage.__table__.foreign_keys)
    )

    assert foreign_key.target_fullname == "chat_sessions.id"
    assert foreign_key.ondelete == "CASCADE"


def test_citation_references_chat_message() -> None:
    """Deleting a message should cascade to its citations."""

    foreign_key = next(
        iter(MessageCitation.__table__.foreign_keys)
    )

    assert foreign_key.target_fullname == "chat_messages.id"
    assert foreign_key.ondelete == "CASCADE"


def test_chat_models_use_uuid_primary_keys() -> None:
    """All public persistence identifiers should be UUID values."""

    assert ChatSession.__table__.primary_key.columns.keys() == [
        "id"
    ]
    assert ChatMessage.__table__.primary_key.columns.keys() == [
        "id"
    ]
    assert MessageCitation.__table__.primary_key.columns.keys() == [
        "id"
    ]


def test_citation_source_number_is_unique_per_message() -> None:
    """One response cannot contain duplicate source numbers."""

    constraint_names = {
        constraint.name
        for constraint in MessageCitation.__table__.constraints
        if constraint.name is not None
    }

    assert (
        "uq_message_citations_message_source_number"
        in constraint_names
    )
