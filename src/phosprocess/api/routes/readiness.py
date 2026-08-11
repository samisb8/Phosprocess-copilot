"""Application dependency readiness route."""

from __future__ import annotations

from math import isfinite
from typing import Any, cast

from fastapi import APIRouter, Request, Response, status

from phosprocess.api.database_dependencies import (
    DatabaseHealthCheck,
    get_database_runtime_state,
)
from phosprocess.api.dependencies import get_rag_runtime_state
from phosprocess.api.schemas.readiness import (
    DatabaseReadiness,
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
            "description": ("The RAG service or PostgreSQL dependency is not ready."),
        }
    },
    summary="Check whether required application services are ready",
)
def readiness(
    request: Request,
    response: Response,
) -> ReadinessResponse:
    """Report readiness of the RAG and PostgreSQL services."""

    rag_state = get_rag_runtime_state(request)
    database_state = get_database_runtime_state(request)

    rag_ready = rag_state.ready and rag_state.service is not None

    database_health = None
    database_ready = False

    database_health_check = cast(
        DatabaseHealthCheck | None,
        getattr(request.app.state, "database_health_check", None),
    )

    if (
        database_state.ready
        and database_state.engine is not None
        and database_health_check is not None
    ):
        try:
            database_health = database_health_check(database_state.engine)
            database_ready = database_health.connected
        except Exception:
            database_ready = False

    issues: list[str] = []

    if not rag_ready:
        issues.append("RAG service is not ready.")

    if not database_ready:
        issues.append("Database is not ready.")

    if issues:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    knowledge_base = rag_state.knowledge_base or {}

    return ReadinessResponse(
        status=("ready" if rag_ready and database_ready else "not_ready"),
        rag_loaded=rag_ready,
        database=DatabaseReadiness(
            connected=database_ready,
            current_user=(database_health.current_user if database_health is not None else None),
            current_database=(
                database_health.current_database if database_health is not None else None
            ),
            server_version=(
                database_health.server_version if database_health is not None else None
            ),
        ),
        knowledge_base=(
            KnowledgeBaseReadiness(
                version=str(knowledge_base.get("version", "unknown")),
                document_count=_non_negative_int(knowledge_base.get("document_count")),
                chunk_count=_non_negative_int(knowledge_base.get("chunk_count")),
            )
            if rag_ready
            else None
        ),
        initial_loading_ms=(
            _non_negative_float(rag_state.initial_loading_ms) if rag_ready else None
        ),
        detail=" ".join(issues) or None,
    )
