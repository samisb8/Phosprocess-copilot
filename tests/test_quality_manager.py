"""Transactional quality-index activation tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import phosprocess.knowledge_base.quality_manager as quality_manager
from phosprocess.knowledge_base.quality_manager import (
    _atomic_write,
    _publish_directory,
)


def test_atomic_pointer_replaces_complete_content(tmp_path: Path) -> None:
    pointer = tmp_path / "current_index.json"
    pointer.write_text("old", encoding="utf-8")

    _atomic_write(pointer, b'{"version":"new"}')

    assert pointer.read_text(encoding="utf-8") == '{"version":"new"}'
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_publish_keeps_existing_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary = tmp_path / "quality_tmp"
    final = tmp_path / "quality"
    temporary.mkdir()
    final.mkdir()
    (temporary / "new.txt").write_text("new", encoding="utf-8")
    (final / "old.txt").write_text("old", encoding="utf-8")
    monkeypatch.setattr(os, "rename", lambda *_args: (_ for _ in ()).throw(PermissionError()))
    monkeypatch.setattr(quality_manager.time, "sleep", lambda _seconds: None)

    with pytest.raises(FileExistsError):
        _publish_directory(temporary, final)

    assert (final / "old.txt").read_text(encoding="utf-8") == "old"
