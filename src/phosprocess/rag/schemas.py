"""Public schemas returned by the production RAG pipeline."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class GroundedAnswerPayload(BaseModel):
    """Strict JSON payload expected from the local Qwen model."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)

    @field_validator("answer")
    @classmethod
    def normalize_answer(cls, value: str) -> str:
        """Reject whitespace-only model answers."""

        normalized = value.strip()

        if not normalized:
            raise ValueError("La réponse ne peut pas être vide.")

        return normalized


class RAGSource(BaseModel):
    """One documentary source selected by retrieval v3."""

    model_config = ConfigDict(frozen=True)

    source_number: int = Field(ge=1)
    chunk_id: str = Field(min_length=1)
    document_name: str = Field(min_length=1)
    pages: list[int] = Field(min_length=1)
    section: str | None = None
    excerpt: str = Field(min_length=1)
    document_title: str | None = None
    filename: str | None = None
    chapter: str | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    anchor_chunk_id: str | None = None
    anchor_chunk_ids: list[str] = Field(default_factory=list)
    expanded_chunk_ids: list[str] = Field(default_factory=list)
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    display_text: str | None = None
    anchor_text: str | None = None
    domain: str | None = None
    chunk_type: str | None = None
    parent_id: str | None = None
    context_scope: str | None = None
    best_anchor_score: float | None = None
    source_boost: float | None = None
    context_added_tokens: int | None = Field(default=None, ge=0)
    context_truncated: bool = False

    selection_source: str
    hybrid_rank: int = Field(ge=1)
    rrf_score: float

    dense_rank: int | None = Field(default=None, ge=1)
    dense_score: float | None = None
    bm25_rank: int | None = Field(default=None, ge=1)
    bm25_score: float | None = None
    reranker_rank: int | None = Field(default=None, ge=1)
    reranker_score: float | None = None


class RAGTimings(BaseModel):
    """Measured durations for one production request."""

    model_config = ConfigDict(frozen=True)

    hybrid_ms: float = Field(ge=0)
    reranking_ms: float = Field(ge=0)
    generation_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)
    first_token_ms: float | None = Field(default=None, ge=0)


class RAGResponse(BaseModel):
    """Grounded answer and complete retrieval provenance."""

    model_config = ConfigDict(frozen=True)

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    sources: list[RAGSource] = Field(default_factory=list)
    cited_source_numbers: list[int] = Field(default_factory=list)
    insufficient_context: bool

    model_name: str = Field(min_length=1)
    selected_variant: str = Field(min_length=1)
    snapshot_sha256: str = Field(pattern=r"^[A-F0-9]{64}$")

    candidate_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    source_policy_route: str = "disabled"
    source_policy_mode: str = "automatic"
    source_policy_primary: str | None = None
    source_policy_fallback_used: bool = False
    source_policy_forced: bool = False
    response_language: str | None = None
    standalone_query: str | None = None
    question_type: str | None = None
    detected_domains: list[str] = Field(default_factory=list)
    timings: RAGTimings
    latency: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_retrieval_cardinality(self) -> RAGResponse:
        """Allow a direct bypass or any non-empty retrieval selection."""

        counts = (self.candidate_count, self.selected_count)
        if counts == (0, 0):
            if self.sources or self.cited_source_numbers:
                raise ValueError("Une réponse directe ne peut pas contenir de sources RAG.")
            return self

        if self.candidate_count < 1 or self.selected_count < 1:
            raise ValueError(
                "Une réponse RAG exige au moins un candidat et une source sélectionnée."
            )

        if self.selected_count > self.candidate_count:
            raise ValueError(
                "Le nombre de sources sélectionnées ne peut pas dépasser les candidats."
            )

        return self


class ChatMessage(BaseModel):
    """One in-memory conversational message."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        """Reject empty history entries."""

        normalized = value.strip()

        if not normalized:
            raise ValueError("Un message de conversation ne peut pas être vide.")

        return normalized


class RAGStreamEvent(BaseModel):
    """Typed event emitted by the streaming RAG pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    event_type: Literal[
        "retrieval_started",
        "retrieval_completed",
        "token",
        "validation_started",
        "sources",
        "completed",
        "error",
    ]
    content: str | None = None
    sources: list[RAGSource] = Field(default_factory=list)
    response: RAGResponse | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
