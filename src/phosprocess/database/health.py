"""PostgreSQL connectivity checks."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Engine, text


@dataclass(frozen=True, slots=True)
class DatabaseHealth:
    """Verified information returned by PostgreSQL."""

    connected: bool
    current_user: str
    current_database: str
    server_version: str


def check_database_connection(
    engine: Engine,
) -> DatabaseHealth:
    """Open one pooled connection and verify PostgreSQL."""

    statement = text(
        """
        SELECT
            current_user AS current_user,
            current_database() AS current_database,
            current_setting('server_version') AS server_version
        """
    )

    with engine.connect() as connection:
        result = connection.execute(statement).mappings().one()

    return DatabaseHealth(
        connected=True,
        current_user=str(result["current_user"]),
        current_database=str(result["current_database"]),
        server_version=str(result["server_version"]),
    )
