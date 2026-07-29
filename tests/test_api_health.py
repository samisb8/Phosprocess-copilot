"""Tests for the API health endpoint."""

from typing import Any

from fastapi.testclient import TestClient

from phosprocess.api.main import create_app
from tests.support.database import (
    build_test_database_engine,
    check_test_database_connection,
)


class _HealthyRAGService:
    """Small fake service that avoids loading production models in tests."""

    initial_loading_ms = 1.0

    def knowledge_base_status(self) -> dict[str, Any]:
        return {
            "version": "test-kb",
            "document_count": 1,
            "chunk_count": 10,
        }

    def warmup(self, *, enabled: bool | None = None) -> object:
        return object()

    def close(self) -> None:
        return None


def test_health_returns_ok() -> None:
    """The API should expose a lightweight health response."""

    service = _HealthyRAGService()
    application = create_app(
        service_factory=lambda: service,
        warmup_enabled=False,
        database_engine_factory=build_test_database_engine,
        database_health_check=check_test_database_connection,
    )

    with TestClient(application) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
