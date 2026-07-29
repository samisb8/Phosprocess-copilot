"""Transactional persistence service for RAG conversations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from phosprocess.database.models import (
    ChatMessage,
    MessageCitation,
)
from phosprocess.database.repositories.chat_repository import (
    ChatRepository,
)
from phosprocess.rag.schemas import RAGResponse, RAGSource


@dataclass(frozen=True, slots=True)
class PersistedChatExchange:
    """Identifiers produced after one exchange is committed."""

    session_id: UUID
    user_message_id: UUID
    assistant_message_id: UUID


def _build_session_title(question: str) -> str:
    """Build a compact initial title from the first question."""

    normalized = " ".join(question.split())
    return normalized[:200]


def _build_source_diagnostic(
    source: RAGSource,
) -> dict[str, Any]:
    """Preserve retrieval and reranking diagnostics for one source."""

    return {
        "source_number": source.source_number,
        "selection_source": source.selection_source,
        "hybrid_rank": source.hybrid_rank,
        "rrf_score": source.rrf_score,
        "reranker_rank": source.reranker_rank,
        "reranker_score": source.reranker_score,
    }


def _build_rag_metadata(
    response: RAGResponse,
) -> dict[str, Any]:
    """Build the flexible audit metadata stored with the answer."""

    return {
        "selected_variant": response.selected_variant,
        "knowledge_base_snapshot_sha256": (
            response.snapshot_sha256
        ),
        "candidate_count": response.candidate_count,
        "selected_count": response.selected_count,
        "cited_source_numbers": list(
            response.cited_source_numbers
        ),
        "source_policy": {
            "route": response.source_policy_route,
            "mode": response.source_policy_mode,
            "primary": response.source_policy_primary,
            "fallback_used": (
                response.source_policy_fallback_used
            ),
            "forced": response.source_policy_forced,
        },
        "standalone_query": response.standalone_query,
        "detected_domains": list(response.detected_domains),
        "timings": {
            "hybrid_ms": response.timings.hybrid_ms,
            "reranking_ms": response.timings.reranking_ms,
            "generation_ms": response.timings.generation_ms,
            "total_ms": response.timings.total_ms,
            "first_token_ms": response.timings.first_token_ms,
        },
        "latency": dict(response.latency),
        "source_diagnostics": [
            _build_source_diagnostic(source)
            for source in response.sources
        ],
    }


class ChatPersistenceService:
    """Persist one complete RAG exchange atomically."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
    ) -> None:
        self._session_factory = session_factory

    def persist_exchange(
        self,
        *,
        response: RAGResponse,
        session_id: UUID | None = None,
    ) -> PersistedChatExchange:
        """Persist a question, answer and citations in one transaction."""

        with self._session_factory.begin() as database_session:
            repository = ChatRepository(database_session)

            if session_id is None:
                chat_session = repository.create_session(
                    title=_build_session_title(
                        response.question
                    )
                )
            else:
                chat_session = repository.require_session(
                    session_id
                )

            user_message = repository.add_message(
                ChatMessage(
                    session=chat_session,
                    role="user",
                    content=response.question,
                    rag_metadata={},
                )
            )

            assistant_message = repository.add_message(
                ChatMessage(
                    session=chat_session,
                    role="assistant",
                    content=response.answer,
                    insufficient_context=(
                        response.insufficient_context
                    ),
                    model_name=response.model_name,
                    response_language=(
                        response.response_language
                    ),
                    question_type=response.question_type,
                    total_ms=response.timings.total_ms,
                    rag_metadata=_build_rag_metadata(response),
                )
            )

            cited_numbers = set(
                response.cited_source_numbers
            )

            for source in response.sources:
                repository.add_citation(
                    MessageCitation(
                        message=assistant_message,
                        source_number=source.source_number,
                        chunk_id=source.chunk_id,
                        document_name=source.document_name,
                        pages=list(source.pages),
                        section=source.section,
                        excerpt=source.excerpt,
                        document_title=(
                            source.document_title
                        ),
                        filename=source.filename,
                        chapter=source.chapter,
                        page_start=source.page_start,
                        page_end=source.page_end,
                        domain=source.domain,
                        chunk_type=source.chunk_type,
                        is_cited=(
                            source.source_number
                            in cited_numbers
                        ),
                    )
                )

            chat_session.updated_at = datetime.now(UTC)

            repository.flush()

            return PersistedChatExchange(
                session_id=chat_session.id,
                user_message_id=user_message.id,
                assistant_message_id=assistant_message.id,
            )
