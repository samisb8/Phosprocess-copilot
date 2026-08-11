"""Lightweight database dependencies used by API tests."""

from sqlalchemy import Engine, create_engine
from sqlalchemy.pool import StaticPool

from phosprocess.database import models as database_models
from phosprocess.database.base import Base
from phosprocess.database.health import DatabaseHealth


def build_test_database_engine() -> Engine:
    """Create a shared in-memory database with the ORM schema."""

    # Accessing the module ensures every ORM table is registered.
    _ = database_models

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={
            "check_same_thread": False,
        },
        poolclass=StaticPool,
    )

    Base.metadata.create_all(engine)

    return engine


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
