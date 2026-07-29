"""Dependencies and shared runtime state for the HTTP API."""

from __future__ import annotations

from asyncio import Lock
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

from fastapi import HTTPException, Request, status

from phosprocess.rag.pipeline import PhosProcessRAG, load_runtime_config
from phosprocess.rag.schemas import RAGResponse


class RAGService(Protocol):
    """Minimal RAG interface required by the API."""

    initial_loading_ms: float

    def knowledge_base_status(self) -> dict[str, Any] | None:
        """Return metadata about the active knowledge base."""

    def warmup(self, *, enabled: bool | None = None) -> object:
        """Warm up the retrieval and generation components."""

    def answer(
        self,
        question: str,
        *,
        source_mode: str = "automatic",
        language_mode: str = "auto",
    ) -> RAGResponse:
        """Generate a grounded answer for one question."""

    def close(self) -> None:
        """Release resources owned by the RAG service."""


RAGServiceFactory = Callable[[], RAGService]


@dataclass(frozen=True, slots=True)
class RAGRuntimeState:
    """State created once during the FastAPI lifespan."""

    service: RAGService | None
    ready: bool
    knowledge_base: dict[str, Any] | None
    initial_loading_ms: float | None
    startup_error: str | None = None


def build_rag_service() -> RAGService:
    """Build the production RAG service from its runtime configuration."""

    runtime = load_runtime_config()
    return PhosProcessRAG(runtime_config=runtime)


def get_rag_runtime_state(request: Request) -> RAGRuntimeState:
    """Return the application RAG state or report unavailability."""

    runtime_state = getattr(request.app.state, "rag_runtime", None)

    if runtime_state is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG runtime state is unavailable.",
        )

    return cast(RAGRuntimeState, runtime_state)


def get_rag_service(request: Request) -> RAGService:
    """Return the ready RAG service used by chat routes."""

    runtime_state = get_rag_runtime_state(request)

    if not runtime_state.ready or runtime_state.service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is not ready.",
        )

    return runtime_state.service


def get_rag_inference_lock(request: Request) -> Lock:
    """Return the lock protecting the shared local RAG/GPU service."""

    inference_lock = getattr(
        request.app.state,
        "rag_inference_lock",
        None,
    )

    if inference_lock is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG inference controller is unavailable.",
        )

    return cast(Lock, inference_lock)
