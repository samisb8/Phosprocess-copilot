"""Database configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Validated PostgreSQL and connection-pool configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_default=True,
    )

    database_url: SecretStr = SecretStr("")
    database_pool_size: int = 5
    database_max_overflow: int = 10
    database_pool_timeout_seconds: float = 30.0
    database_echo_sql: bool = False

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: SecretStr) -> SecretStr:
        """Require the modern SQLAlchemy Psycopg dialect."""

        raw_url = value.get_secret_value().strip()

        if not raw_url:
            raise ValueError("DATABASE_URL is required.")

        if not raw_url.startswith("postgresql+psycopg://"):
            raise ValueError("DATABASE_URL must use postgresql+psycopg://.")

        return SecretStr(raw_url)


@lru_cache
def get_database_settings() -> DatabaseSettings:
    """Load and cache database settings for the current process."""

    return DatabaseSettings()
