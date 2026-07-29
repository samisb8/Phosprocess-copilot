"""FastAPI application entry point."""

from fastapi import FastAPI

from phosprocess.api.routes.health import router as health_router


def create_app() -> FastAPI:
    """Create and configure the PhosProcess API application."""

    application = FastAPI(
        title="PhosProcess Copilot API",
        description=(
            "API for the wet-process phosphoric acid production assistant."
        ),
        version="0.1.0",
    )
    application.include_router(health_router)

    return application


app = create_app()
