"""Tests for the API health endpoint."""

from fastapi.testclient import TestClient

from phosprocess.api.main import app


def test_health_returns_ok() -> None:
    """The API should expose a lightweight health response."""

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
