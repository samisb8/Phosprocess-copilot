"""Resolve the atomically activated production knowledge-base version."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_KNOWLEDGE_BASE_ROOT = PROJECT_ROOT / "data" / "knowledge_base"
DEFAULT_CURRENT_INDEX_PATH = DEFAULT_KNOWLEDGE_BASE_ROOT / "current_index.json"


class ActiveKnowledgeBaseError(RuntimeError):
    """Raised when the active production pointer is missing or unsafe."""


@dataclass(frozen=True, slots=True)
class ActiveKnowledgeBase:
    """Validated paths and counts for one immutable index version."""

    version: str
    version_directory: Path
    dense_index_directory: Path
    bm25_index_directory: Path
    corpus_directory: Path
    document_count: int
    chunk_count: int
    documents: tuple[dict[str, Any], ...]
    pointer_path: Path


def _required_non_empty_string(
    payload: dict[str, Any],
    field_name: str,
) -> str:
    value = payload.get(field_name)

    if not isinstance(value, str) or not value.strip():
        raise ActiveKnowledgeBaseError(f"Champ invalide dans current_index.json : {field_name}.")

    return value.strip()


def _resolve_version_directory(
    path_value: str,
    *,
    project_root: Path,
    knowledge_base_root: Path,
) -> Path:
    candidate = Path(path_value)

    if not candidate.is_absolute():
        candidate = project_root / candidate

    resolved = candidate.resolve()
    versions_root = (knowledge_base_root / "indexes" / "versions").resolve()

    if resolved.parent != versions_root:
        raise ActiveKnowledgeBaseError(
            "Le chemin actif doit désigner une version directe sous "
            "data/knowledge_base/indexes/versions."
        )

    return resolved


def load_active_knowledge_base(
    pointer_path: Path = DEFAULT_CURRENT_INDEX_PATH,
    *,
    project_root: Path = PROJECT_ROOT,
    knowledge_base_root: Path | None = None,
) -> ActiveKnowledgeBase:
    """Load and validate the current production index pointer."""

    pointer_path = pointer_path.resolve()
    project_root = project_root.resolve()
    effective_root = (
        knowledge_base_root.resolve()
        if knowledge_base_root is not None
        else pointer_path.parent.resolve()
    )

    if not pointer_path.is_file():
        raise ActiveKnowledgeBaseError(
            "Aucune base documentaire active. Exécutez scripts/sync_knowledge_base.py."
        )

    try:
        payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ActiveKnowledgeBaseError(
            f"Pointeur de base documentaire invalide : {pointer_path}"
        ) from error

    if not isinstance(payload, dict):
        raise ActiveKnowledgeBaseError("current_index.json doit contenir un objet JSON.")

    version = _required_non_empty_string(payload, "version")
    version_directory = _resolve_version_directory(
        _required_non_empty_string(payload, "path"),
        project_root=project_root,
        knowledge_base_root=effective_root,
    )

    if version_directory.name != version:
        raise ActiveKnowledgeBaseError("La version et le chemin actif ne correspondent pas.")

    dense_directory = version_directory / "dense"
    bm25_directory = version_directory / "bm25"
    corpus_directory = version_directory / "corpus"
    required_files = (
        dense_directory / "index.faiss",
        dense_directory / "metadata.jsonl",
        dense_directory / "manifest.json",
        bm25_directory / "metadata.jsonl",
        bm25_directory / "manifest.json",
        version_directory / "manifest.json",
    )
    missing = [path for path in required_files if not path.is_file()]

    if missing or not corpus_directory.is_dir():
        formatted = ", ".join(str(path) for path in missing)
        raise ActiveKnowledgeBaseError(
            "La version active est incomplète" + (f" : {formatted}" if formatted else ".")
        )

    document_count = payload.get("document_count")
    chunk_count = payload.get("chunk_count")
    documents = payload.get("documents", [])

    if (
        not isinstance(document_count, int)
        or isinstance(document_count, bool)
        or document_count <= 0
        or not isinstance(chunk_count, int)
        or isinstance(chunk_count, bool)
        or chunk_count <= 0
        or not isinstance(documents, list)
        or len(documents) != document_count
        or not all(isinstance(item, dict) for item in documents)
    ):
        raise ActiveKnowledgeBaseError(
            "Les compteurs ou documents de current_index.json sont invalides."
        )

    return ActiveKnowledgeBase(
        version=version,
        version_directory=version_directory,
        dense_index_directory=dense_directory,
        bm25_index_directory=bm25_directory,
        corpus_directory=corpus_directory,
        document_count=document_count,
        chunk_count=chunk_count,
        documents=tuple(documents),
        pointer_path=pointer_path,
    )
