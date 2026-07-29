"""Alembic migration environment for PhosProcess Copilot."""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import Connection

# Import for side effects: register ORM models in Base.metadata.
from phosprocess.database import (
    models as _models,  # noqa: F401
)
from phosprocess.database.base import Base
from phosprocess.database.engine import create_database_engine
from phosprocess.database.settings import get_database_settings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_database_url() -> str:
    """Return the private PostgreSQL URL loaded from the environment."""

    settings = get_database_settings()
    return settings.database_url.get_secret_value()


def run_migrations_offline() -> None:
    """Generate SQL without opening a PostgreSQL connection."""

    context.configure(
        url=get_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_with_connection(
    connection: Connection,
) -> None:
    """Run migrations using an active SQLAlchemy connection."""

    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect to PostgreSQL and execute pending migrations."""

    engine = create_database_engine()

    try:
        with engine.connect() as connection:
            run_migrations_with_connection(connection)
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
