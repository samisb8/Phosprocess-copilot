"""Verify the complete Python-to-PostgreSQL connection path."""

from phosprocess.database.engine import create_database_engine
from phosprocess.database.health import check_database_connection


def main() -> int:
    """Connect, query PostgreSQL, print safe metadata, then close."""

    engine = create_database_engine()

    try:
        health = check_database_connection(engine)
    finally:
        engine.dispose()

    print("Database connection: OK")
    print(f"User: {health.current_user}")
    print(f"Database: {health.current_database}")
    print(f"PostgreSQL: {health.server_version}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
