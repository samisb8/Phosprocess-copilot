"""Tests for network-free resolution of installed embedding snapshots."""

from __future__ import annotations

from phosprocess.embeddings.embedder import resolve_cached_model_source


def test_cached_hub_snapshot_is_preferred(tmp_path) -> None:
    model_cache = tmp_path / "models--BAAI--bge-m3"
    snapshot = model_cache / "snapshots" / "revision-123"
    snapshot.mkdir(parents=True)
    references = model_cache / "refs"
    references.mkdir()
    (references / "main").write_text("revision-123\n", encoding="utf-8")

    resolved = resolve_cached_model_source(
        "BAAI/bge-m3",
        cache_dir=str(tmp_path),
    )

    assert resolved == str(snapshot.resolve())


def test_missing_cache_preserves_model_identifier(tmp_path) -> None:
    assert resolve_cached_model_source(
        "BAAI/bge-m3",
        cache_dir=str(tmp_path),
    ) == "BAAI/bge-m3"
