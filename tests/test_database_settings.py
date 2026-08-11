"""Tests for PostgreSQL configuration and engine construction."""

from pydantic import SecretStr

from phosprocess.database.engine import create_database_engine
from phosprocess.database.settings import DatabaseSettings


def test_database_settings_build_psycopg_engine() -> None:
    """The engine should use PostgreSQL and the Psycopg 3 driver."""

    settings = DatabaseSettings(
        database_url=SecretStr(
            "postgresql+psycopg://"
            "phosprocess_app:test-password@"
            "127.0.0.1:5432/phosprocess"
        )
    )

    engine = create_database_engine(settings)

    try:
        assert engine.url.drivername == "postgresql+psycopg"
        assert engine.url.database == "phosprocess"
        assert engine.url.username == "phosprocess_app"
    finally:
        engine.dispose()


def test_database_password_is_hidden_from_settings_repr() -> None:
    """Configuration representations must not expose the password."""

    settings = DatabaseSettings(
        database_url=SecretStr(
            "postgresql+psycopg://"
            "phosprocess_app:very-secret-password@"
            "127.0.0.1:5432/phosprocess"
        )
    )

    assert "very-secret-password" not in repr(settings)
