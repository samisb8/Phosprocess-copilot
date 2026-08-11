"""Retrieval scope routing with catalog-driven explicit source locking.

Automatic mode never chooses a document from a hand-written domain rule.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from phosprocess.knowledge_base.catalog import load_document_catalog
from phosprocess.knowledge_base.domains import KnowledgeDomain
from phosprocess.knowledge_base.schemas import KnowledgeBaseCatalog
from phosprocess.knowledge_base.source_resolution import (
    resolve_explicit_source,
    resolve_source_mode,
    supported_source_modes,
)

SUPPORTED_SOURCE_MODES = frozenset({"auto", *supported_source_modes()})

_SOURCE_SCOPE_RELEASE = re.compile(
    r"\b(?:toutes? les sources|tous les documents|cherche partout|"
    r"sans source imposee|mode automatique|source automatique|"
    r"all sources|all documents|search everywhere|automatic source|"
    r"no source restriction)\b",
    re.I,
)


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return re.sub(r"[^a-z0-9%\u0600-\u06ff]+", " ", without_marks).strip()


def detect_explicit_source_mode(
    question: str,
    *,
    catalog: KnowledgeBaseCatalog | None = None,
) -> str | None:
    """Return the stable catalog mode for one explicit source request."""

    resolution = resolve_explicit_source(
        question,
        catalog=catalog or load_document_catalog(),
    )
    return resolution.source_mode if resolution is not None else None


def requests_automatic_source_scope(question: str) -> bool:
    return _SOURCE_SCOPE_RELEASE.search(_normalize(question)) is not None


def _infer_question_type(normalized: str) -> str:
    """Small domain-neutral intent fallback used only for retrieval structure."""

    if any(term in normalized for term in ("balance", "bilan")):
        return "balance"
    if any(term in normalized for term in ("step by step", "etape par etape", "trajet", "path")):
        return "process_flow"
    if any(term in normalized for term in ("compare", "difference", "versus")):
        return "comparison"
    if any(term in normalized for term in ("why", "pourquoi", "how", "comment", "role")):
        return "explanation"
    if any(term in normalized for term in ("define", "definition", "c est quoi", "what is")):
        return "definition"
    return "explanation"


@dataclass(frozen=True, slots=True)
class DomainRoutingDecision:
    detected_domains: tuple[tuple[KnowledgeDomain, float], ...]
    confidence: float
    preferred_documents: tuple[str, ...]
    soft_boosts: dict[str, float]
    explanation: str
    hard_filter: frozenset[str] | None
    source_mode: str
    question_type: str = "explanation"
    explicit_source: str | None = None
    temporal_scope: str = "static"
    section_affinity_terms: tuple[str, ...] = ()


def route_query(
    question: str,
    *,
    catalog: KnowledgeBaseCatalog,
    source_mode: str = "auto",
    question_type: str | None = None,
    focus_entity: str | None = None,
    maximum_source_boost: float = 0.06,
) -> DomainRoutingDecision:
    """Return scope metadata without automatic document selection."""

    del maximum_source_boost
    mode = source_mode.strip().casefold()
    if mode == "automatic":
        mode = "auto"

    explicit = resolve_explicit_source(question, catalog=catalog) if mode == "auto" else None
    selected = explicit if explicit is not None else (
        resolve_source_mode(mode, catalog=catalog) if mode != "auto" else None
    )

    if mode != "auto" and selected is None:
        supported = ", ".join(sorted({"auto", *supported_source_modes(catalog)}))
        raise ValueError("Mode source invalide : " + supported)

    normalized = _normalize(" ".join(part for part in (question, focus_entity or "") if part))
    resolved_type = (question_type or "").strip().casefold() or _infer_question_type(normalized)
    temporal_scope = (
        "live_or_current"
        if any(
            marker in normalized
            for marker in ("today", "now", "current", "actuel", "aujourd hui", "maintenant")
        )
        else "static"
    )
    detected = ((KnowledgeDomain.GENERAL_CHEMICAL_ENGINEERING, 0.0),)

    if selected is not None:
        return DomainRoutingDecision(
            detected_domains=detected,
            confidence=1.0,
            preferred_documents=(selected.document_id,),
            soft_boosts={},
            explanation="Explicit user source filter resolved from catalog metadata.",
            hard_filter=frozenset({selected.document_id}),
            source_mode=selected.source_mode,
            question_type=resolved_type,
            explicit_source=selected.source_mode,
            temporal_scope=temporal_scope,
        )

    return DomainRoutingDecision(
        detected_domains=detected,
        confidence=0.0,
        preferred_documents=(),
        soft_boosts={},
        explanation="Automatic mode defers document selection to retrieval evidence.",
        hard_filter=None,
        source_mode="auto",
        question_type=resolved_type,
        temporal_scope=temporal_scope,
    )
