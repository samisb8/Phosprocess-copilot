"""Tests for the main module."""

from phosprocess.main import get_project_name


def test_get_project_name() -> None:
    """The project must expose its official name."""
    assert get_project_name() == "PhosProcess Copilot"
