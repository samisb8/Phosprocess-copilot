"""Deterministic production-only document routing for the RAG pipeline."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

BECKER_SOURCE = "01_becker_phosphates_and_phosphoric_acid.pdf"
JACOBS_SOURCE = "02_jacobs_largest_phosphoric_acid_plant.pdf"
UNIDO_SOURCE = "03_unido_phosphate_process_technologies.pdf"
ATELIER_SOURCE = "04_rapport_atelier_acide_phosphorique.pdf"

SOURCE_ALIASES = {
    "becker": BECKER_SOURCE,
    "jacobs": JACOBS_SOURCE,
    "atelier": ATELIER_SOURCE,
}
SOURCE_LABELS = {
    BECKER_SOURCE: "Becker",
    JACOBS_SOURCE: "Jacobs",
    UNIDO_SOURCE: "UNIDO",
    ATELIER_SOURCE: "Atelier OCP",
    "02_chemical_engineering_thermodynamics_9e.pdf": "Smith–Van Ness",
    "03_fundamentals_heat_mass_transfer.pdf": "Incropera",
    "05_perrys_chemical_engineers_handbook_9e.pdf": "Perry",
    "06_mullin_crystallization_4e.pdf": "Mullin",
    "07_process_dynamics_control_seborg_4e.pdf": "Seborg",
    "08_transport_phenomena_bird_2e.pdf": "Bird",
}
SUPPORTED_SOURCE_MODES = frozenset({"automatic", *SOURCE_ALIASES})

_JACOBS_INTENT = re.compile(r"\bjacobs\b", flags=re.IGNORECASE)
_ATELIER_INTENT = re.compile(
    r"\b(?:ocp|atelier)\b|rapport\s+(?:de\s+l['’])?atelier",
    flags=re.IGNORECASE,
)
_DOCUMENT_TOKEN = re.compile(r"[a-z][a-z0-9]{3,}")
_GENERIC_DOCUMENT_TOKENS = frozenset(
    {
        "acid",
        "acide",
        "chemical",
        "compress",
        "edition",
        "engineering",
        "fundamentals",
        "heat",
        "mass",
        "phosphates",
        "phosphoric",
        "phenomena",
        "process",
        "rapport",
        "revised",
        "thermodynamics",
        "transfer",
    }
)


@dataclass(frozen=True, slots=True)
class SourcePolicyConfig:
    """Application-layer source-priority configuration."""

    enabled: bool
    default_priority: tuple[str, ...]
    domain_routes: dict[str, tuple[str, ...]]
    minimum_preferred_chunks: int
    allow_fallback: bool

    def __post_init__(self) -> None:
        if not self.default_priority:
            raise ValueError("default_priority ne peut pas être vide.")

        if len(self.default_priority) != len(set(self.default_priority)):
            raise ValueError("default_priority contient des doublons.")

        if self.minimum_preferred_chunks <= 0:
            raise ValueError(
                "minimum_preferred_chunks doit être strictement positif."
            )

        required_routes = {"general", "jacobs", "ocp_atelier"}

        if set(self.domain_routes) != required_routes:
            raise ValueError(
                "Les routes documentaires requises sont "
                "general, jacobs et ocp_atelier."
            )

        known_sources = set(self.default_priority)

        for route, sources in self.domain_routes.items():
            if not sources:
                raise ValueError(
                    f"La route documentaire {route} est vide."
                )

            if len(sources) != len(set(sources)):
                raise ValueError(
                    f"La route documentaire {route} contient des doublons."
                )

            unknown = set(sources) - known_sources

            if unknown:
                raise ValueError(
                    f"Sources inconnues pour {route}: {sorted(unknown)}"
                )


@dataclass(frozen=True, slots=True)
class SourcePolicyDecision:
    """Deterministic document scope selected before retrieval."""

    route: str
    mode: str
    preferred_sources: tuple[str, ...]
    primary_source: str | None
    forced: bool
    allow_fallback: bool

    @property
    def primary_label(self) -> str:
        """Return a concise human-facing primary-source label."""

        if self.primary_source is None:
            return "Aucune"

        return source_label(self.primary_source)


@dataclass(frozen=True, slots=True)
class AppliedSourcePolicy:
    """Policy outcome attached to one validated RAG response."""

    route: str
    mode: str
    primary_source: str | None
    preferred_sources: tuple[str, ...]
    selected_scope: tuple[str, ...]
    fallback_used: bool
    forced: bool
    attempt_count: int
    sufficient_preferred_chunks: int

    @property
    def primary_label(self) -> str:
        """Return a concise human-facing primary-source label."""

        if self.primary_source is None:
            return "Aucune"

        return source_label(self.primary_source)


def document_id_from_source(source_file: str) -> str:
    """Convert one configured PDF filename to the indexed document ID."""

    return Path(source_file).stem


def source_label(source_file: str) -> str:
    """Return the configured display label for one PDF."""

    return SOURCE_LABELS.get(source_file, Path(source_file).stem)


def normalize_source_mode(mode: str) -> str:
    """Normalize and validate an automatic or forced source mode."""

    normalized = mode.strip().casefold()

    if normalized not in SUPPORTED_SOURCE_MODES:
        raise ValueError(
            "Mode source invalide. Utilisez automatic, becker, "
            "jacobs ou atelier."
        )

    return normalized


def detect_document_route(question: str) -> str:
    """Classify document intent without an LLM call."""

    if _JACOBS_INTENT.search(question):
        return "jacobs"

    if _ATELIER_INTENT.search(question):
        return "ocp_atelier"

    return "general"


def _source_tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    ascii_value = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    tokens = set(_DOCUMENT_TOKEN.findall(ascii_value))
    variants = {
        token[:-1]
        for token in tokens
        if token.endswith("s") and len(token) > 5
    }
    return (tokens | variants) - _GENERIC_DOCUMENT_TOKENS


def detect_explicit_active_source(
    question: str,
    active_sources: Sequence[str],
) -> str | None:
    """Match an explicitly named active document without an LLM call.

    Only distinctive filename tokens are considered, so generic domain words
    cannot bypass the normal Becker priority. Ambiguous matches deliberately
    fall back to the configured domain policy.
    """

    question_tokens = _source_tokens(question)
    matches: list[tuple[int, str]] = []

    for source in active_sources:
        overlap = question_tokens & _source_tokens(Path(source).stem)

        if overlap:
            matches.append((max(map(len, overlap)), source))

    if not matches:
        return None

    best_score = max(score for score, _source in matches)
    best_sources = [
        source
        for score, source in matches
        if score == best_score
    ]
    return best_sources[0] if len(best_sources) == 1 else None


def decide_source_policy(
    question: str,
    *,
    config: SourcePolicyConfig,
    mode: str = "automatic",
) -> SourcePolicyDecision:
    """Choose the preferred document scope deterministically."""

    normalized_mode = normalize_source_mode(mode)

    if normalized_mode != "automatic":
        source = SOURCE_ALIASES[normalized_mode]
        route = (
            "ocp_atelier"
            if normalized_mode == "atelier"
            else normalized_mode
        )
        return SourcePolicyDecision(
            route=route,
            mode=normalized_mode,
            preferred_sources=(source,),
            primary_source=source,
            forced=True,
            allow_fallback=False,
        )

    if not config.enabled:
        return SourcePolicyDecision(
            route="disabled",
            mode="automatic",
            preferred_sources=config.default_priority,
            primary_source=None,
            forced=False,
            allow_fallback=False,
        )

    route = detect_document_route(question)
    preferred_sources = config.domain_routes[route]
    return SourcePolicyDecision(
        route=route,
        mode="automatic",
        preferred_sources=preferred_sources,
        primary_source=preferred_sources[0],
        forced=False,
        allow_fallback=config.allow_fallback,
    )
