"""Tests for the shared SQLAlchemy declarative base."""

from phosprocess.database.base import Base


def test_base_uses_stable_constraint_names() -> None:
    """Alembic should generate deterministic database object names."""

    convention = Base.metadata.naming_convention

    assert convention is not None
    assert convention["pk"] == "pk_%(table_name)s"
    assert convention["ix"] == (
        "ix_%(table_name)s_%(column_0_name)s"
    )
    assert convention["fk"] == (
        "fk_%(table_name)s_%(column_0_name)s_"
        "%(referred_table_name)s"
    )
