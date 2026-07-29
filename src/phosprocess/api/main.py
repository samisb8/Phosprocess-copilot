"""FastAPI application entry point."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from phosprocess.api.dependencies import (
    RAGRuntimeState,
    RAGService,
    RAGServiceFactory,
    build_rag_service,
)
from phosprocess.api.routes.health import router as health_router
from phosprocess.api.routes.readiness import router as readiness_router

LOGGER = logging.getLogger(__name__)


def _safe_close(service: RAGService | None) -> None:
    """Close a RAG service without preventing application shutdown."""

    if service is None:
        return

    try:
        service.close()
    except Exception:
        LOGGER.exception("Failed to close the RAG service cleanly.")


def create_app(
    *,
    service_factory: RAGServiceFactory = build_rag_service,
    warmup_enabled: bool | None = None,
) -> FastAPI:
    """Create and configure the PhosProcess API application."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        service: RAGService | None = None

        try:
            service = service_factory()
            knowledge_base = service.knowledge_base_status()

            if knowledge_base is None:
                raise RuntimeError(
                    "The active knowledge base status is unavailable."
                )

            service.warmup(enabled=warmup_enabled)

            runtime_state = RAGRuntimeState(
                service=service,
                ready=True,
                knowledge_base=knowledge_base,
                initial_loading_ms=float(service.initial_loading_ms),
            )
        except Exception as exception:
            LOGGER.exception("The RAG service failed during API startup.")
            _safe_close(service)

            runtime_state = RAGRuntimeState(
                service=None,
                ready=False,
                knowledge_base=None,
                initial_loading_ms=None,
                startup_error=(
                    f"{type(exception).__name__}: {exception}"
                ),
            )

        application.state.rag_runtime = runtime_state

        try:
            yield
        finally:
            _safe_close(runtime_state.service)

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

    return application


app = create_app()
