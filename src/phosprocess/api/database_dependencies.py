"""PostgreSQL dependencies and runtime state for the HTTP API."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from fastapi import HTTPException, Request, status
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from phosprocess.database.engine import create_database_engine
from phosprocess.database.health import DatabaseHealth
from phosprocess.database.services.chat_history import (
    ChatHistoryService,
)
from phosprocess.database.services.chat_persistence import (
    ChatPersistenceService,
)
from phosprocess.database.services.chat_session_listing import (
    ChatSessionListingService,
)
from phosprocess.database.services.chat_session_management import (
    ChatSessionManagementService,
)

DatabaseEngineFactory = Callable[[], Engine]
DatabaseHealthCheck = Callable[[Engine], DatabaseHealth]


@dataclass(frozen=True, slots=True)
class DatabaseRuntimeState:
    """Database resources created once during the FastAPI lifespan."""

    engine: Engine | None
    session_factory: sessionmaker[Session] | None
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


def get_database_session_factory(
    request: Request,
) -> sessionmaker[Session]:
    """Return the ready SQLAlchemy session factory."""

    runtime_state = get_database_runtime_state(request)

    if not runtime_state.ready or runtime_state.session_factory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service is not ready.",
        )

    return runtime_state.session_factory


def get_chat_session_management_service(
    request: Request,
) -> ChatSessionManagementService:
    """Build the transactional session-management service."""

    return ChatSessionManagementService(get_database_session_factory(request))


def get_chat_session_listing_service(
    request: Request,
) -> ChatSessionListingService:
    """Build the paginated conversation-listing service."""

    return ChatSessionListingService(get_database_session_factory(request))


def get_chat_history_service(
    request: Request,
) -> ChatHistoryService:
    """Build the read-only chat history service."""

    return ChatHistoryService(get_database_session_factory(request))


def get_chat_persistence_service(
    request: Request,
) -> ChatPersistenceService:
    """Build the transactional chat persistence service."""

    return ChatPersistenceService(get_database_session_factory(request))
