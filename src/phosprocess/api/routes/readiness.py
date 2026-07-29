"""RAG readiness-check route."""

from __future__ import annotations

from math import isfinite
from typing import Any

from fastapi import APIRouter, Request, Response, status

from phosprocess.api.dependencies import get_rag_runtime_state
from phosprocess.api.schemas.readiness import (
    KnowledgeBaseReadiness,
    ReadinessResponse,
)

router = APIRouter(tags=["health"])


def _non_negative_int(value: Any) -> int:
    """Convert a metadata value to a non-negative integer."""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0

    return max(parsed, 0)


def _non_negative_float(value: float | None) -> float | None:
    """Return a finite non-negative duration."""

    if value is None:
        return None

    parsed = float(value)

    if not isfinite(parsed):
        return None

    return max(parsed, 0.0)


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ReadinessResponse,
            "description": "The RAG service is not ready.",
        }
    },
    summary="Check whether the RAG service is ready",
)
def readiness(request: Request, response: Response) -> ReadinessResponse:
    """Report whether the RAG and knowledge base are ready."""

    runtime_state = get_rag_runtime_state(request)

    if not runtime_state.ready or runtime_state.service is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

        return ReadinessResponse(
            status="not_ready",
            rag_loaded=False,
            detail="RAG service is not ready.",
        )

    knowledge_base = runtime_state.knowledge_base or {}

    return ReadinessResponse(
        status="ready",
        rag_loaded=True,
        knowledge_base=KnowledgeBaseReadiness(
            version=str(knowledge_base.get("version", "unknown")),
            document_count=_non_negative_int(
                knowledge_base.get("document_count")
            ),
            chunk_count=_non_negative_int(
                knowledge_base.get("chunk_count")
            ),
        ),
        initial_loading_ms=_non_negative_float(
            runtime_state.initial_loading_ms
        ),
    )
