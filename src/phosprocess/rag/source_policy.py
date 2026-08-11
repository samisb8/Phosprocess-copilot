"""Legacy-index source policy backed by the production document catalog."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from phosprocess.knowledge_base.catalog import load_document_catalog
from phosprocess.knowledge_base.source_resolution import (
    resolve_explicit_source,
    resolve_source_mode,
    supported_source_modes,
)

_CATALOG = load_document_catalog()


def _compatibility_source(mode: str) -> str:
    resolution = resolve_source_mode(mode, catalog=_CATALOG)
    if resolution is None:
        raise RuntimeError(f"Alias de catalogue introuvable : {mode}")
    return resolution.entry.source_filename


# Transitional public constants. Their values are derived from catalog metadata;
# document identity is not duplicated in this module.
BECKER_SOURCE = _compatibility_source("becker")
THERMODYNAMICS_SOURCE = _compatibility_source("thermodynamics")
HEAT_TRANSFER_SOURCE = _compatibility_source("heat_transfer")
ATELIER_SOURCE = _compatibility_source("report")
PERRY_SOURCE = _compatibility_source("perry")
CRYSTALLIZATION_SOURCE = _compatibility_source("crystallization")
CONTROL_SOURCE = _compatibility_source("control")
TRANSPORT_SOURCE = _compatibility_source("transport")

SUPPORTED_SOURCE_MODES = frozenset({"automatic", "auto", *supported_source_modes(_CATALOG)})


@dataclass(frozen=True, slots=True)
class SourcePolicyConfig:
    """Runtime compatibility settings for the legacy retrieval path."""

    enabled: bool
    default_priority: tuple[str, ...] = ()
    domain_routes: dict[str, tuple[str, ...]] | None = None
    minimum_preferred_chunks: int = 1
    allow_fallback: bool = False

    def __post_init__(self) -> None:
        if self.minimum_preferred_chunks <= 0:
            raise ValueError("minimum_preferred_chunks doit être strictement positif.")


@dataclass(frozen=True, slots=True)
class SourcePolicyDecision:
    route: str
    mode: str
    preferred_sources: tuple[str, ...]
    primary_source: str | None
    forced: bool
    allow_fallback: bool

    @property
    def primary_label(self) -> str:
        return "Aucune" if self.primary_source is None else source_label(self.primary_source)


@dataclass(frozen=True, slots=True)
class AppliedSourcePolicy:
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
        return "Aucune" if self.primary_source is None else source_label(self.primary_source)


def document_id_from_source(source_file: str) -> str:
    return Path(source_file).stem


def source_label(source_file: str) -> str:
    normalized = Path(source_file).name.casefold()
    for entry in _CATALOG.documents:
        if normalized in {
            Path(entry.source_filename).name.casefold(),
            Path(entry.canonical_filename).name.casefold(),
        }:
            return entry.display_title
    return Path(source_file).stem


def normalize_source_mode(mode: str) -> str:
    normalized = mode.strip().casefold()
    if normalized in {"auto", "automatic"}:
        return "automatic"

    resolution = resolve_source_mode(normalized, catalog=_CATALOG)
    if resolution is None:
        supported = ", ".join(sorted(SUPPORTED_SOURCE_MODES))
        raise ValueError("Mode source invalide. Utilisez : " + supported)
    return resolution.source_mode


def detect_explicit_active_source(
    question: str,
    active_sources: Sequence[str],
) -> str | None:
    """Resolve an explicit catalog source and require it to be active."""

    resolution = resolve_explicit_source(question, catalog=_CATALOG)
    if resolution is None:
        return None

    active_by_name = {Path(source).name.casefold(): source for source in active_sources}
    for filename in (
        resolution.entry.source_filename,
        resolution.entry.canonical_filename,
    ):
        active = active_by_name.get(Path(filename).name.casefold())
        if active is not None:
            return active
    return None


def decide_source_policy(
    question: str,
    *,
    config: SourcePolicyConfig,
    mode: str = "automatic",
) -> SourcePolicyDecision:
    """Lock only a source explicitly selected by mode or question wording."""

    del config
    normalized_mode = normalize_source_mode(mode)

    resolution = (
        resolve_explicit_source(question, catalog=_CATALOG)
        if normalized_mode == "automatic"
        else resolve_source_mode(normalized_mode, catalog=_CATALOG)
    )
    if resolution is not None:
        source = resolution.entry.source_filename
        return SourcePolicyDecision(
            route="explicit_document",
            mode=resolution.source_mode,
            preferred_sources=(source,),
            primary_source=source,
            forced=True,
            allow_fallback=False,
        )

    return SourcePolicyDecision(
        route="automatic_global",
        mode="automatic",
        preferred_sources=(),
        primary_source=None,
        forced=False,
        allow_fallback=False,
    )
