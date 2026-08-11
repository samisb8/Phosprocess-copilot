"""Schemas for the API readiness endpoint."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeBaseReadiness(BaseModel):
    """Metadata about the active RAG knowledge base."""

    model_config = ConfigDict(frozen=True)

    version: str = Field(min_length=1)
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)


class DatabaseReadiness(BaseModel):
    """State and safe metadata for the PostgreSQL dependency."""

    model_config = ConfigDict(frozen=True)

    connected: bool
    current_user: str | None = None
    current_database: str | None = None
    server_version: str | None = None


class ReadinessResponse(BaseModel):
    """Response describing whether required services are ready."""

    model_config = ConfigDict(frozen=True)

    status: Literal["ready", "not_ready"]
    rag_loaded: bool
    database: DatabaseReadiness
    knowledge_base: KnowledgeBaseReadiness | None = None
    initial_loading_ms: float | None = Field(default=None, ge=0)
    detail: str | None = None
