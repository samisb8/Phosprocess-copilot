"""Context-safe multilingual query expansion without an LLM call."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from phosprocess.retrieval.technical_lexicon import TECHNICAL_EQUIVALENTS

# Intent terms are deliberately short and generic. They are not facts and are
# used only to improve retrieval recall for query types whose wording is often
# underspecified (for example: "décris le trajet dans cet équipement").
_INTENT_EQUIVALENTS: dict[str, tuple[str, ...]] = {
    "process_flow": (
        "flow path",
        "circulation loop",
        "feed inlet",
        "weak acid feed",
        "dilute phosphoric acid feed",
        "feed line",
        "acid is introduced",
        "concentrated acid withdrawal",
        "product acid withdrawal",
        "product draw-off",
        "acid sent to storage",
        "circulation pump",
        "heat exchanger",
        "flash chamber",
        "recirculation line",
        "product outlet",
    ),
    "procedure": (
        "process sequence",
        "operating steps",
        "inlet",
        "outlet",
    ),
    "balance": (
        "mass balance",
        "material balance",
        "component balance",
        "P2O5 balance",
        "heat balance",
        "energy balance",
        "enthalpy balance",
        "steady state",
    ),
    "troubleshooting": (
        "operating problems",
        "cause effect",
        "fouling scaling",
        "performance loss",
    ),
    "momentum_diffusion": (
        "molecular transport of momentum",
        "momentum flux",
        "velocity gradient",
        "shear stress",
        "Newton law of viscosity",
        "dynamic viscosity",
    ),
}

# Dense retrieval benefits from a concise and balanced semantic hint. BM25 still
# receives every lexical variant. Process-flow hints deliberately cover the inlet,
# circulation/heating, vapor separation/return and product withdrawal so one stage
# cannot displace the others merely because it has more synonyms.
_DENSE_INTENT_LIMIT = 5
_DENSE_INTENT_HINTS: dict[str, tuple[str, ...]] = {
    "process_flow": (
        "flow path",
        "weak acid feed",
        "circulation pump heat exchanger",
        "flash chamber recirculation",
        "concentrated acid withdrawal",
    ),
    "momentum_diffusion": (
        "molecular transport of momentum",
        "momentum flux velocity gradient",
        "Newton law of viscosity",
    ),
}


def normalize_query(value: str) -> str:
    """Normalize only for matching while preserving the original query."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


@dataclass(frozen=True, slots=True)
class ExpandedTechnicalQuery:
    """Separate dense and BM25 representations with traceable additions."""

    original_query: str
    standalone_query: str
    dense_query: str
    bm25_expanded_query: str
    added_terms: tuple[str, ...]


def _append_unique(additions: list[str], candidate: str, normalized: str) -> None:
    """Append one expansion term once and only when it is not already present."""

    if normalize_query(candidate) in normalized:
        return

    if candidate not in additions:
        additions.append(candidate)


def expand_technical_query(
    original_query: str,
    *,
    standalone_query: str | None = None,
    question_type: str | None = None,
) -> ExpandedTechnicalQuery:
    """Add lexical equivalents plus a small intent-specific retrieval hint.

    The original and standalone questions remain unchanged. Intent expansion is
    deterministic and does not inject domain facts; it only names common parts
    of the requested information structure, such as inlet, pump, exchanger,
    flash chamber and outlet for a process-flow question.
    """

    original = original_query.strip()
    standalone = (standalone_query or original).strip()

    if not original or not standalone:
        raise ValueError("Les requêtes ne peuvent pas être vides.")

    normalized = normalize_query(standalone)
    additions: list[str] = []

    for expression, equivalents in TECHNICAL_EQUIVALENTS.items():
        if normalize_query(expression) not in normalized:
            continue

        for equivalent in equivalents:
            _append_unique(additions, equivalent, normalized)

    intent_additions: list[str] = []

    for equivalent in _INTENT_EQUIVALENTS.get(question_type or "", ()):
        if normalize_query(equivalent) in normalized:
            continue

        if equivalent not in additions and equivalent not in intent_additions:
            intent_additions.append(equivalent)

    all_additions = [*additions, *intent_additions]
    configured_dense_hints = _DENSE_INTENT_HINTS.get(
        question_type or "",
        tuple(intent_additions),
    )
    dense_hints = [
        hint
        for hint in configured_dense_hints
        if normalize_query(hint) not in normalized
    ][:_DENSE_INTENT_LIMIT]
    dense = " ".join([standalone, *dense_hints]).strip()
    bm25 = " ".join([standalone, *all_additions]).strip()
    return ExpandedTechnicalQuery(
        original_query=original,
        standalone_query=standalone,
        dense_query=dense,
        bm25_expanded_query=bm25,
        added_terms=tuple(all_additions),
    )
