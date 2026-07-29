"""SQLAlchemy session factory."""

from __future__ import annotations

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker


def create_session_factory(
    engine: Engine,
) -> sessionmaker[Session]:
    """Create database sessions bound to the shared engine."""

    return sessionmaker(
        bind=engine,
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )
