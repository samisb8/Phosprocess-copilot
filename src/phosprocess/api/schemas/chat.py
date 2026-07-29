"""Public request and response schemas for the chat API."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    """Question and execution options accepted by the chat endpoint."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        max_length=4_000,
        description="Technical question sent to PhosProcess Copilot.",
    )
    session_id: UUID | None = Field(
        default=None,
        description=(
            "Existing conversation identifier. "
            "Omit it to create a new conversation."
        ),
    )
    source_mode: str = Field(
        default="automatic",
        min_length=1,
        max_length=64,
    )
    language_mode: str = Field(
        default="auto",
        min_length=1,
        max_length=32,
    )

    @field_validator(
        "question",
        "source_mode",
        "language_mode",
    )
    @classmethod
    def strip_text_fields(cls, value: str) -> str:
        """Reject fields containing only spaces."""

        stripped = value.strip()

        if not stripped:
            raise ValueError("The value must not be empty.")

        return stripped


class ChatSourceResponse(BaseModel):
    """Public citation metadata returned to the client."""

    model_config = ConfigDict(frozen=True)

    source_number: int
    chunk_id: str
    document_name: str
    pages: list[int] = Field(default_factory=list)
    section: str | None = None
    excerpt: str
    document_title: str | None = None
    filename: str | None = None
    chapter: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    domain: str | None = None
    chunk_type: str | None = None


class ChatHistoryCitationResponse(BaseModel):
    """Citation persisted with one historical assistant message."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    source_number: int = Field(gt=0)
    chunk_id: str
    document_name: str
    pages: list[int] = Field(default_factory=list)
    section: str | None = None
    excerpt: str
    document_title: str | None = None
    filename: str | None = None
    chapter: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    domain: str | None = None
    chunk_type: str | None = None
    is_cited: bool
    created_at: datetime


class ChatHistoryMessageResponse(BaseModel):
    """One persisted user or assistant message."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    role: str
    content: str
    created_at: datetime
    insufficient_context: bool | None = None
    model_name: str | None = None
    response_language: str | None = None
    question_type: str | None = None
    total_ms: float | None = Field(default=None, ge=0)
    citations: list[ChatHistoryCitationResponse] = Field(
        default_factory=list
    )


class ChatSessionHistoryResponse(BaseModel):
    """Complete persisted conversation returned by the history API."""

    model_config = ConfigDict(frozen=True)

    session_id: UUID
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    messages: list[ChatHistoryMessageResponse] = Field(
        default_factory=list
    )


class ChatTimingsResponse(BaseModel):
    """Main RAG latency measurements returned to the client."""

    model_config = ConfigDict(frozen=True)

    hybrid_ms: float = Field(ge=0)
    reranking_ms: float = Field(ge=0)
    generation_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)
    first_token_ms: float | None = Field(default=None, ge=0)


class ChatPolicyResponse(BaseModel):
    """Routing and source-selection policy applied by the RAG."""

    model_config = ConfigDict(frozen=True)

    route: str
    mode: str
    primary: str | None = None
    fallback_used: bool
    forced: bool


class ChatResponse(BaseModel):
    """Stable public API response built from the internal RAG response."""

    model_config = ConfigDict(frozen=True)

    session_id: UUID
    user_message_id: UUID
    assistant_message_id: UUID

    question: str
    answer: str
    sources: list[ChatSourceResponse] = Field(default_factory=list)
    cited_source_numbers: list[int] = Field(default_factory=list)
    insufficient_context: bool

    model_name: str
    selected_variant: str
    knowledge_base_snapshot_sha256: str

    candidate_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)

    source_policy: ChatPolicyResponse

    response_language: str | None = None
    standalone_query: str | None = None
    question_type: str | None = None
    detected_domains: list[str] = Field(default_factory=list)

    timings: ChatTimingsResponse
