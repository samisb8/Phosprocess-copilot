"""Load and verify the explicit production document catalogue."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import yaml

from phosprocess.knowledge_base.models import sha256_file
from phosprocess.knowledge_base.runtime import DEFAULT_KNOWLEDGE_BASE_ROOT, PROJECT_ROOT
from phosprocess.knowledge_base.schemas import (
    DocumentCatalogEntry,
    KnowledgeBaseCatalog,
)

DEFAULT_CATALOG_PATH = PROJECT_ROOT / "configs" / "knowledge_base_catalog.yaml"


class KnowledgeBaseCatalogError(RuntimeError):
    """Raised when catalogue metadata and physical sources disagree."""


def load_document_catalog(
    path: Path = DEFAULT_CATALOG_PATH,
) -> KnowledgeBaseCatalog:
    """Load the catalogue without inferring domains from filenames."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise KnowledgeBaseCatalogError(f"Catalogue illisible : {path}") from error

    if not isinstance(raw, dict):
        raise KnowledgeBaseCatalogError("Le catalogue YAML doit contenir un objet.")

    try:
        return KnowledgeBaseCatalog.model_validate(raw)
    except ValueError as error:
        raise KnowledgeBaseCatalogError(f"Catalogue invalide : {error}") from error


def locate_catalogue_source(
    entry: DocumentCatalogEntry,
    *,
    knowledge_base_root: Path = DEFAULT_KNOWLEDGE_BASE_ROOT,
) -> Path | None:
    """Locate an observed source in active, rejected or archive storage."""

    for directory_name in ("pdfs", "rejected", "archive"):
        candidate = knowledge_base_root / directory_name / entry.source_filename

        if candidate.is_file():
            return candidate

    return None


def verify_catalogue_sources(
    catalog: KnowledgeBaseCatalog,
    *,
    knowledge_base_root: Path = DEFAULT_KNOWLEDGE_BASE_ROOT,
) -> dict[str, tuple[bool, str]]:
    """Verify exact SHA and page count for every physically available PDF."""

    results: dict[str, tuple[bool, str]] = {}

    for entry in catalog.documents:
        source = locate_catalogue_source(
            entry,
            knowledge_base_root=knowledge_base_root,
        )

        if source is None:
            results[entry.document_id] = (False, "source_missing")
            continue

        digest = sha256_file(source)

        if digest != entry.sha256:
            results[entry.document_id] = (False, "sha256_mismatch")
            continue

        try:
            with pymupdf.open(source) as document:
                page_count = int(document.page_count)
        except Exception as error:
            results[entry.document_id] = (
                False,
                f"unreadable:{type(error).__name__}",
            )
            continue

        if page_count != entry.page_count:
            results[entry.document_id] = (False, "page_count_mismatch")
            continue

        results[entry.document_id] = (True, "verified")

    return results
