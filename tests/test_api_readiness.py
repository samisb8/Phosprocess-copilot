"""Tests for the RAG readiness endpoint."""

from typing import Any

from fastapi.testclient import TestClient

from phosprocess.api.main import create_app


class _FakeRAGService:
    """Fake RAG service used to test application lifecycle behavior."""

    def __init__(self) -> None:
        self.initial_loading_ms = 125.5
        self.warmup_calls: list[bool | None] = []
        self.closed = False

    def knowledge_base_status(self) -> dict[str, Any]:
        return {
            "version": "kb-test-001",
            "document_count": 8,
            "chunk_count": 27_096,
        }

    def warmup(self, *, enabled: bool | None = None) -> object:
        self.warmup_calls.append(enabled)
        return object()

    def close(self) -> None:
        self.closed = True


def test_ready_returns_active_knowledge_base() -> None:
    """A successfully loaded RAG service should be reported as ready."""

    service = _FakeRAGService()
    application = create_app(
        service_factory=lambda: service,
        warmup_enabled=False,
    )

    with TestClient(application) as client:
        response = client.get("/ready")

        assert response.status_code == 200
        assert response.json() == {
            "status": "ready",
            "rag_loaded": True,
            "knowledge_base": {
                "version": "kb-test-001",
                "document_count": 8,
                "chunk_count": 27_096,
            },
            "initial_loading_ms": 125.5,
            "detail": None,
        }
        assert service.warmup_calls == [False]

    assert service.closed is True


def test_health_remains_available_when_rag_startup_fails() -> None:
    """The process can be alive while its RAG dependency is unavailable."""

    def failing_factory() -> _FakeRAGService:
        raise RuntimeError("Simulated startup failure")

    application = create_app(
        service_factory=failing_factory,
        warmup_enabled=False,
    )

    with TestClient(application) as client:
        health_response = client.get("/health")
        readiness_response = client.get("/ready")

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}

    assert readiness_response.status_code == 503
    assert readiness_response.json() == {
        "status": "not_ready",
        "rag_loaded": False,
        "knowledge_base": None,
        "initial_loading_ms": None,
        "detail": "RAG service is not ready.",
    }
