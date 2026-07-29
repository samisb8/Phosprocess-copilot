"""PostgreSQL dependencies and runtime state for the HTTP API."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from fastapi import HTTPException, Request, status
from sqlalchemy import Engine

from phosprocess.database.engine import create_database_engine
from phosprocess.database.health import DatabaseHealth

DatabaseEngineFactory = Callable[[], Engine]
DatabaseHealthCheck = Callable[[Engine], DatabaseHealth]


@dataclass(frozen=True, slots=True)
class DatabaseRuntimeState:
    """Database resources created once during the FastAPI lifespan."""

    engine: Engine | None
    ready: bool
    health: DatabaseHealth | None
    startup_error: str | None = None


def build_database_engine() -> Engine:
    """Build the production SQLAlchemy engine from environment settings."""

    return create_database_engine()


def get_database_runtime_state(
    request: Request,
) -> DatabaseRuntimeState:
    """Return the shared database runtime state."""

    runtime_state = getattr(
        request.app.state,
        "database_runtime",
        None,
    )

    if runtime_state is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database runtime state is unavailable.",
        )

    return cast(DatabaseRuntimeState, runtime_state)


def get_database_engine(request: Request) -> Engine:
    """Return the ready shared SQLAlchemy engine."""

    runtime_state = get_database_runtime_state(request)

    if not runtime_state.ready or runtime_state.engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service is not ready.",
        )

    return runtime_state.engine
