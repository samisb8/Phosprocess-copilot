"""Explicit non-evidentiary process state for conversational references."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

_EQUIPMENT_PATTERNS = (
    (
        re.compile(
            r"\b(?:évaporateur à circulation forcée|evaporateur a circulation forcee|"
            r"forced-circulation evaporator|forced circulation evaporator)\b",
            re.I,
        ),
        "évaporateur à circulation forcée",
    ),
    (
        re.compile(r"\b(?:pompe de circulation|circulation pump)\b", re.I),
        "pompe de circulation",
    ),
    (
        re.compile(
            r"\b(?:chambre de flash|flash chamber|vapor body|vapour body|"
            r"bouilleur|غرفة التبخير|جسم المبخر)\b",
            re.I,
        ),
        "chambre de vaporisation",
    ),
    (
        re.compile(r"\b(?:échangeur thermique|echangeur thermique|heat exchanger)\b", re.I),
        "échangeur thermique",
    ),
    (re.compile(r"\b(?:évaporateur|evaporator)\b", re.I), "évaporateur"),
    (re.compile(r"\b(?:pompe à vide|pompe a vide|vacuum pump)\b", re.I), "pompe à vide"),
    (re.compile(r"\b(?:condenseur|condenser)\b", re.I), "condenseur"),
    (re.compile(r"\b(?:séparateur|separateur|separator)\b", re.I), "séparateur"),
    (re.compile(r"\b(?:réacteur|reactor)\b", re.I), "réacteur"),
    (re.compile(r"\b(?:filtre|filter)\b", re.I), "filtre"),
)
_FLUID_PATTERNS = (
    (re.compile(r"\b(?:acide phosphorique|phosphoric acid)\b", re.I), "acide phosphorique"),
    (re.compile(r"\b(?:bouillie|slurry)\b", re.I), "bouillie phosphorique"),
    (re.compile(r"\b(?:vapeur|steam)\b", re.I), "vapeur"),
)
_OPERATION_PATTERNS = (
    (re.compile(r"\b(?:évaporation|evaporation)\b", re.I), "évaporation"),
    (re.compile(r"\b(?:concentration|concentrat)\b", re.I), "concentration"),
    (re.compile(r"\b(?:filtration|filtering)\b", re.I), "filtration"),
    (re.compile(r"\b(?:cristallisation|crystallization)\b", re.I), "cristallisation"),
)
_VARIABLE_PATTERNS = (
    (re.compile(r"\b(?:température|temperature)\b", re.I), "température"),
    (re.compile(r"\b(?:pression|pressure)\b", re.I), "pression"),
    (re.compile(r"\b(?:débit|flow rate)\b", re.I), "débit"),
    (re.compile(r"\b(?:niveau|level)\b", re.I), "niveau"),
    (re.compile(r"\b(?:concentration)\b", re.I), "concentration"),
)
_PROBLEM_PATTERNS = (
    (re.compile(r"\b(?:encrassement|fouling)\b", re.I), "encrassement"),
    (re.compile(r"\b(?:entartrage|scaling)\b", re.I), "entartrage"),
    (re.compile(r"\b(?:instabilité|instability)\b", re.I), "instabilité"),
    (re.compile(r"\b(?:perte de charge|pressure drop)\b", re.I), "perte de charge"),
)


def _last_match(
    question: str,
    patterns: tuple[tuple[re.Pattern[str], str], ...],
) -> str | None:
    matches = [
        (match.start(), match.end() - match.start(), value)
        for pattern, value in patterns
        if (match := pattern.search(question)) is not None
    ]
    return max(matches, default=(-1, -1, None))[2]


def _focus_equipment(question: str) -> str | None:
    """Return the equipment that is grammatically central to the question."""

    normalized = question.strip()
    focus_patterns = (
        re.compile(
            r"(?:r[oô]le|fonction)\s+(?:de|du|d['’])\s*"
            r"(?:la|le|l['’])?\s*(pompe de circulation|"
            r"[eé]vaporateur [aà] circulation forc[eé]e|"
            r"[eé]changeur thermique|[eé]vaporateur|filtre|r[eé]acteur)",
            re.I,
        ),
        re.compile(
            r"(?:role|function)\s+of\s+(?:the\s+)?"
            r"(circulation pump|forced-circulation evaporator|"
            r"heat exchanger|evaporator|filter|reactor)",
            re.I,
        ),
        re.compile(
            r"(?:why is|how does|why does)\s+(?:the\s+)?"
            r"(circulation pump|forced-circulation evaporator|"
            r"heat exchanger|evaporator|filter|reactor)",
            re.I,
        ),
    )
    aliases = {
        "circulation pump": "pompe de circulation",
        "forced-circulation evaporator": "évaporateur à circulation forcée",
        "heat exchanger": "échangeur thermique",
        "evaporator": "évaporateur",
        "filter": "filtre",
        "reactor": "réacteur",
    }

    for pattern in focus_patterns:
        match = pattern.search(normalized)
        if match is not None:
            value = match.group(1).casefold()
            return aliases.get(value, value)

    matches = []
    for pattern, value in _EQUIPMENT_PATTERNS:
        match = pattern.search(normalized)
        if match is not None:
            matches.append((match.start(), -(match.end() - match.start()), value))
    return min(matches, default=(0, 0, None))[2]


@dataclass(slots=True)
class ConversationState:
    """Mutable entity state that helps retrieval but never acts as evidence."""

    current_process: str | None = None
    current_unit: str | None = None
    current_equipment: str | None = None
    current_fluid: str | None = None
    current_operation: str | None = None
    current_variable: str | None = None
    current_problem: str | None = None
    focus_entity: str | None = None
    current_document_scope: str = "auto"
    current_source_mode: str = "auto"
    source_scope_explicit: bool = False
    source_scope_origin: str | None = None
    last_question_type: str | None = None
    last_language: str | None = None
    last_user_question: str | None = None
    last_standalone_query: str | None = None
    last_cited_documents: tuple[str, ...] = ()
    recent_turns: list[str] = field(default_factory=list)
    rolling_summary: str = ""

    def observe_question(self, question: str) -> None:
        """Update only entities explicitly present in a user question."""

        focus_equipment = _focus_equipment(question)
        equipment = focus_equipment or _last_match(question, _EQUIPMENT_PATTERNS)
        fluid = _last_match(question, _FLUID_PATTERNS)
        operation = _last_match(question, _OPERATION_PATTERNS)
        variable = _last_match(question, _VARIABLE_PATTERNS)
        problem = _last_match(question, _PROBLEM_PATTERNS)

        if equipment:
            self.current_equipment = equipment
            self.focus_entity = focus_equipment or equipment

            if "évaporateur" in question.casefold() or "evaporator" in question.casefold():
                self.current_process = "production d’acide phosphorique"
                self.current_fluid = self.current_fluid or "acide phosphorique"
                self.current_operation = (
                    self.current_operation
                    or "concentration par évaporation"
                )

        if fluid:
            self.current_fluid = fluid

        if operation:
            self.current_operation = operation

        if variable:
            self.current_variable = variable

        if problem:
            self.current_problem = problem

        if re.search(r"\b(?:atelier|ocp|workshop)\b", question, re.I):
            self.current_unit = "atelier d’acide phosphorique"

        if "acide phosphorique" in question.casefold() or "phosphoric acid" in question.casefold():
            self.current_process = "production d’acide phosphorique"

        self.last_user_question = question.strip()
        self.recent_turns.append(question.strip())
        self.recent_turns[:] = self.recent_turns[-4:]

    def record_resolution(self, standalone_query: str) -> None:
        self.last_standalone_query = standalone_query.strip()

    def record_source_scope(
        self,
        mode: str,
        *,
        explicit: bool,
        origin: str = "question",
    ) -> None:
        """Persist an explicit source only for the active conversation thread."""

        normalized = mode.strip().casefold() or "auto"
        self.current_source_mode = normalized
        self.current_document_scope = normalized
        self.source_scope_explicit = explicit and normalized != "auto"
        self.source_scope_origin = (
            origin.strip().casefold()
            if self.source_scope_explicit
            else None
        )

    def release_source_scope(self) -> None:
        """Return to automatic routing for a new autonomous question."""

        self.current_source_mode = "auto"
        self.current_document_scope = "auto"
        self.source_scope_explicit = False
        self.source_scope_origin = None

    def record_question_type(self, question_type: str) -> None:
        self.last_question_type = question_type.strip().casefold()

    def record_response(
        self,
        *,
        cited_documents: tuple[str, ...],
        language: str,
    ) -> None:
        self.last_cited_documents = cited_documents
        self.last_language = language

    def clear(self) -> None:
        """Reset all business and conversational fields."""

        fresh = ConversationState()

        for key, value in asdict(fresh).items():
            setattr(self, key, value)

    def debug_view(self) -> dict[str, object]:
        """Return state values without documentary passages."""

        return asdict(self)
