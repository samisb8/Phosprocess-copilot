"""Legacy-index document policy aligned with the eight-document router.

Quality indexes use :mod:`phosprocess.retrieval.domain_router` directly.  This
module keeps rollback/non-quality indexes consistent with the same source
vocabulary so a legacy path cannot silently route to removed Jacobs/UNIDO PDFs.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from phosprocess.retrieval.domain_router import detect_explicit_source_mode

BECKER_SOURCE = "01_becker_phosphates_and_phosphoric_acid.pdf"
THERMODYNAMICS_SOURCE = "02_chemical_engineering_thermodynamics_9e.pdf"
HEAT_TRANSFER_SOURCE = "03_fundamentals_heat_mass_transfer.pdf"
ATELIER_SOURCE = "04_rapport_atelier_acide_phosphorique.pdf"
PERRY_SOURCE = "05_perrys_chemical_engineers_handbook_9e.pdf"
CRYSTALLIZATION_SOURCE = "06_mullin_crystallization_4e.pdf"
CONTROL_SOURCE = "07_process_dynamics_control_seborg_4e.pdf"
TRANSPORT_SOURCE = "08_transport_phenomena_bird_2e.pdf"

# Backward-compatible names retained for imports in old tests/scripts. They no
# longer appear in the active eight-document catalogue.
JACOBS_SOURCE = "02_jacobs_largest_phosphoric_acid_plant.pdf"
UNIDO_SOURCE = "03_unido_phosphate_process_technologies.pdf"

SOURCE_ALIASES = {
    "becker": BECKER_SOURCE,
    "report": ATELIER_SOURCE,
    "atelier": ATELIER_SOURCE,
    "thermodynamics": THERMODYNAMICS_SOURCE,
    "heat_transfer": HEAT_TRANSFER_SOURCE,
    "perry": PERRY_SOURCE,
    "crystallization": CRYSTALLIZATION_SOURCE,
    "control": CONTROL_SOURCE,
    "transport": TRANSPORT_SOURCE,
}
SOURCE_LABELS = {
    BECKER_SOURCE: "Becker",
    THERMODYNAMICS_SOURCE: "Smith–Van Ness",
    HEAT_TRANSFER_SOURCE: "Incropera",
    ATELIER_SOURCE: "Rapport OCP JFC4",
    PERRY_SOURCE: "Perry",
    CRYSTALLIZATION_SOURCE: "Mullin",
    CONTROL_SOURCE: "Seborg",
    TRANSPORT_SOURCE: "Bird",
}
SUPPORTED_SOURCE_MODES = frozenset({"automatic", "auto", *SOURCE_ALIASES})

_DOCUMENT_TOKEN = re.compile(r"[a-z][a-z0-9]{2,}")
_GENERIC_DOCUMENT_TOKENS = frozenset(
    {
        "acid",
        "acide",
        "chemical",
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
        if "general" not in self.domain_routes:
            raise ValueError("La route documentaire general est requise.")

        known_sources = set(self.default_priority)
        for route, sources in self.domain_routes.items():
            if not sources:
                raise ValueError(f"La route documentaire {route} est vide.")
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
        if self.primary_source is None:
            return "Aucune"
        return source_label(self.primary_source)


def document_id_from_source(source_file: str) -> str:
    return Path(source_file).stem


def source_label(source_file: str) -> str:
    return SOURCE_LABELS.get(source_file, Path(source_file).stem)


def normalize_source_mode(mode: str) -> str:
    normalized = mode.strip().casefold()
    if normalized == "auto":
        normalized = "automatic"
    if normalized not in SUPPORTED_SOURCE_MODES:
        raise ValueError(
            "Mode source invalide. Utilisez automatic, becker, report, "
            "thermodynamics, heat_transfer, perry, crystallization, "
            "control ou transport."
        )
    return normalized


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", without_marks).strip()


def detect_document_route(question: str) -> str:
    """Choose a legacy soft scope; quality routing remains globally searchable."""

    normalized = _normalize(question)
    if any(term in normalized for term in ("ocp", "jfc4", "atelier", "echelon")):
        return "plant_specific"
    if any(
        term in normalized
        for term in (
            "p2o5",
            "acide phosphorique",
            "phosphoric acid",
            "wet process",
            "voie humide",
        )
    ):
        return "phosphoric_acid"
    if any(
        term in normalized
        for term in (
            "enthalp",
            "pression de vapeur",
            "vapor pressure",
            "thermodynam",
        )
    ):
        return "thermodynamics"
    if any(
        term in normalized
        for term in (
            "echangeur",
            "heat exchanger",
            "lmtd",
            "transfert thermique",
            "heat transfer",
        )
    ):
        return "heat_transfer"
    if any(term in normalized for term in ("cristall", "sursaturation", "nucleation")):
        return "crystallization"
    if any(term in normalized for term in ("pid", "mpc", "regulation", "control")):
        return "control"
    if any(
        term in normalized
        for term in (
            "diffusion",
            "reynolds",
            "perte de charge",
            "pressure drop",
        )
    ):
        return "transport"
    if any(
        term in normalized
        for term in (
            "evaporateur",
            "evaporator",
            "pompe",
            "pump",
            "equipment",
        )
    ):
        return "equipment"
    return "general"


def _source_tokens(value: str) -> set[str]:
    normalized = _normalize(value)
    tokens = set(_DOCUMENT_TOKEN.findall(normalized))
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
    """Match one explicitly requested active document, never a mere context mention."""

    mode = detect_explicit_source_mode(question)
    if mode is not None:
        expected = SOURCE_ALIASES[mode]
        return expected if expected in active_sources else None

    # Preserve support for an exact active filename mentioned with a request
    # phrase even when it is not one of the canonical aliases.
    normalized = _normalize(question)
    if not re.search(
        r"\b(?:selon|d apres|cherche dans|cherche sur|based on|according to|use only)\b",
        normalized,
    ):
        return None
    question_tokens = _source_tokens(question)
    matches: list[tuple[int, str]] = []
    for source in active_sources:
        overlap = question_tokens & _source_tokens(Path(source).stem)
        if overlap:
            matches.append((max(map(len, overlap)), source))
    if not matches:
        return None
    best_score = max(score for score, _source in matches)
    best_sources = [source for score, source in matches if score == best_score]
    return best_sources[0] if len(best_sources) == 1 else None


def decide_source_policy(
    question: str,
    *,
    config: SourcePolicyConfig,
    mode: str = "automatic",
) -> SourcePolicyDecision:
    """Choose one preferred legacy scope deterministically."""

    normalized_mode = normalize_source_mode(mode)
    if normalized_mode != "automatic":
        source = SOURCE_ALIASES[normalized_mode]
        return SourcePolicyDecision(
            route=normalized_mode,
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
    preferred_sources = config.domain_routes.get(
        route,
        config.domain_routes["general"],
    )
    return SourcePolicyDecision(
        route=route,
        mode="automatic",
        preferred_sources=preferred_sources,
        primary_source=preferred_sources[0],
        forced=False,
        allow_fallback=config.allow_fallback,
    )
