"""SQLAlchemy engine construction."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url

from phosprocess.database.settings import (
    DatabaseSettings,
    get_database_settings,
)


def create_database_engine(
    settings: DatabaseSettings | None = None,
) -> Engine:
    """Create the shared PostgreSQL connection engine and pool."""

    configuration = settings or get_database_settings()
    database_url = make_url(configuration.database_url.get_secret_value())

    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=configuration.database_pool_size,
        max_overflow=configuration.database_max_overflow,
        pool_timeout=(configuration.database_pool_timeout_seconds),
        pool_recycle=1_800,
        echo=configuration.database_echo_sql,
        connect_args={"connect_timeout": 5},
    )
