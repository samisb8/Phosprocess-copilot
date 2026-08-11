"""Catalog-driven document identity and explicit-source resolution."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from phosprocess.knowledge_base.catalog import load_document_catalog
from phosprocess.knowledge_base.schemas import (
    DocumentCatalogEntry,
    KnowledgeBaseCatalog,
)

_EXPLICIT_SOURCE_REQUEST = re.compile(
    r"\b(?:according\s+to|based\s+on|what\s+does|as\s+stated\s+in|"
    r"selon|d\s+apres|que\s+dit|indique(?:e|es|s)?\s+par|dans|"
    r"cherche(?:r|z)?\s+(?:dans|sur)|"
    r"uniquement\s+(?:dans|selon)|use\s+only|search\s+(?:in|within))\b",
    re.I,
)


@dataclass(frozen=True, slots=True)
class SourceResolution:
    """One unambiguous document identity resolved from catalog metadata."""

    document_id: str
    source_mode: str
    matched_alias: str
    entry: DocumentCatalogEntry


def normalize_source_reference(value: str) -> str:
    """Normalize accents and punctuation for identity matching only."""

    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return re.sub(r"[^a-z0-9%\u0600-\u06ff]+", " ", without_marks).strip()


def source_mode_for_entry(entry: DocumentCatalogEntry) -> str:
    """Return the stable user-facing mode declared as the first alias."""

    return entry.aliases[0].casefold()


def supported_source_modes(
    catalog: KnowledgeBaseCatalog | None = None,
) -> frozenset[str]:
    active_catalog = catalog or load_document_catalog()
    return frozenset(
        source_mode_for_entry(entry) for entry in active_catalog.documents if entry.active
    )


def _identity_aliases(entry: DocumentCatalogEntry) -> tuple[str, ...]:
    """Build searchable identity strings solely from catalog metadata."""

    values = [
        *entry.aliases,
        entry.document_id,
        entry.display_title,
        entry.canonical_filename,
        Path(entry.canonical_filename).stem,
        entry.source_filename,
        Path(entry.source_filename).stem,
        *entry.authors,
    ]
    normalized = [normalize_source_reference(value) for value in values]
    return tuple(dict.fromkeys(value for value in normalized if value))


def _contains_alias(normalized_text: str, alias: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized_text) is not None


def _resolution(
    entry: DocumentCatalogEntry,
    matched_alias: str,
) -> SourceResolution:
    return SourceResolution(
        document_id=entry.document_id,
        source_mode=source_mode_for_entry(entry),
        matched_alias=matched_alias,
        entry=entry,
    )


def resolve_source_mode(
    value: str,
    *,
    catalog: KnowledgeBaseCatalog | None = None,
) -> SourceResolution | None:
    """Resolve an exact CLI/API mode or document identity."""

    normalized = normalize_source_reference(value)
    if not normalized:
        return None

    active_catalog = catalog or load_document_catalog()
    matches: list[SourceResolution] = []
    for entry in active_catalog.documents:
        if not entry.active:
            continue
        for alias in _identity_aliases(entry):
            if normalized == alias:
                matches.append(_resolution(entry, alias))
                break

    document_ids = {match.document_id for match in matches}
    return matches[0] if len(document_ids) == 1 else None


def resolve_explicit_source(
    question: str,
    *,
    catalog: KnowledgeBaseCatalog | None = None,
) -> SourceResolution | None:
    """Resolve a named source only when the wording explicitly requests it."""

    normalized = normalize_source_reference(question)
    if not normalized or _EXPLICIT_SOURCE_REQUEST.search(normalized) is None:
        return None

    active_catalog = catalog or load_document_catalog()
    matches: list[SourceResolution] = []
    for entry in active_catalog.documents:
        if not entry.active:
            continue
        aliases = sorted(_identity_aliases(entry), key=len, reverse=True)
        matched_alias = next(
            (alias for alias in aliases if _contains_alias(normalized, alias)),
            None,
        )
        if matched_alias is not None:
            matches.append(_resolution(entry, matched_alias))

    document_ids = {match.document_id for match in matches}
    return matches[0] if len(document_ids) == 1 else None
