"""FastAPI application entry point."""

from __future__ import annotations

import logging
from asyncio import Lock
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import Engine

from phosprocess.api.database_dependencies import (
    DatabaseEngineFactory,
    DatabaseHealthCheck,
    DatabaseRuntimeState,
    build_database_engine,
)
from phosprocess.api.dependencies import (
    RAGRuntimeState,
    RAGService,
    RAGServiceFactory,
    build_rag_service,
)
from phosprocess.api.routes.chat import router as chat_router
from phosprocess.api.routes.health import router as health_router
from phosprocess.api.routes.readiness import router as readiness_router
from phosprocess.database.health import check_database_connection

LOGGER = logging.getLogger(__name__)


def _safe_close(service: RAGService | None) -> None:
    """Close a RAG service without preventing application shutdown."""

    if service is None:
        return

    try:
        service.close()
    except Exception:
        LOGGER.exception("Failed to close the RAG service cleanly.")


def _safe_dispose(engine: Engine | None) -> None:
    """Dispose of the SQLAlchemy pool without blocking shutdown."""

    if engine is None:
        return

    try:
        engine.dispose()
    except Exception:
        LOGGER.exception(
            "Failed to dispose of the database engine cleanly."
        )


def create_app(
    *,
    service_factory: RAGServiceFactory = build_rag_service,
    warmup_enabled: bool | None = None,
    database_engine_factory: DatabaseEngineFactory = (
        build_database_engine
    ),
    database_health_check: DatabaseHealthCheck = (
        check_database_connection
    ),
) -> FastAPI:
    """Create and configure the PhosProcess API application."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.rag_inference_lock = Lock()

        database_engine: Engine | None = None

        try:
            database_engine = database_engine_factory()
            database_health = database_health_check(
                database_engine
            )

            if not database_health.connected:
                raise RuntimeError(
                    "The PostgreSQL health check failed."
                )

            database_runtime = DatabaseRuntimeState(
                engine=database_engine,
                ready=True,
                health=database_health,
            )
        except Exception as exception:
            LOGGER.exception(
                "PostgreSQL failed during API startup."
            )
            _safe_dispose(database_engine)

            database_runtime = DatabaseRuntimeState(
                engine=None,
                ready=False,
                health=None,
                startup_error=(
                    f"{type(exception).__name__}: {exception}"
                ),
            )

        application.state.database_runtime = database_runtime

        service: RAGService | None = None

        try:
            service = service_factory()
            knowledge_base = service.knowledge_base_status()

            if knowledge_base is None:
                raise RuntimeError(
                    "The active knowledge base status is unavailable."
                )

            service.warmup(enabled=warmup_enabled)

            rag_runtime = RAGRuntimeState(
                service=service,
                ready=True,
                knowledge_base=knowledge_base,
                initial_loading_ms=float(service.initial_loading_ms),
            )
        except Exception as exception:
            LOGGER.exception(
                "The RAG service failed during API startup."
            )
            _safe_close(service)

            rag_runtime = RAGRuntimeState(
                service=None,
                ready=False,
                knowledge_base=None,
                initial_loading_ms=None,
                startup_error=(
                    f"{type(exception).__name__}: {exception}"
                ),
            )

        application.state.rag_runtime = rag_runtime

        try:
            yield
        finally:
            _safe_close(rag_runtime.service)
            _safe_dispose(database_runtime.engine)

    application = FastAPI(
        title="PhosProcess Copilot API",
        description=(
            "API for the wet-process phosphoric acid production assistant."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    application.include_router(health_router)
    application.include_router(readiness_router)
    application.include_router(chat_router)

    return application


app = create_app()
