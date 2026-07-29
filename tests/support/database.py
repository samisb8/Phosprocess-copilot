"""Lightweight database dependencies used by API tests."""

from sqlalchemy import Engine, create_engine

from phosprocess.database.health import DatabaseHealth


def build_test_database_engine() -> Engine:
    """Create a local in-memory engine without network access."""

    return create_engine("sqlite+pysqlite:///:memory:")


def check_test_database_connection(
    _engine: Engine,
) -> DatabaseHealth:
    """Return deterministic PostgreSQL-like metadata for tests."""

    return DatabaseHealth(
        connected=True,
        current_user="test_user",
        current_database="test_database",
        server_version="17-test",
    )
