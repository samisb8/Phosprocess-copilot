"""Intent-aware retrieval planning for retriever v4.

The planner decomposes one user question into independently retrievable
information roles. It does not inject documentary facts and never generates an
answer. Its only purpose is to make every required side of a question visible
to dense, sparse and lexical retrieval before reranking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from phosprocess.retrieval.technical_lexicon import TECHNICAL_EQUIVALENTS


@dataclass(frozen=True, slots=True)
class EvidenceRole:
    """One independently retrievable evidence need."""

    name: str
    query: str
    subject: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    """Structured retrieval contract for one standalone question."""

    question_type: str
    base_query: str
    roles: tuple[EvidenceRole, ...]
    comparison_subjects: tuple[str, str] | None = None
    balance_kind: str | None = None
    answer_intent: str | None = None

    @property
    def queries(self) -> tuple[str, ...]:
        return tuple(role.query for role in self.roles)


_COMPARISON_PATTERNS = (
    re.compile(
        r"\bcompare\s+(?P<a>.+?)\s+(?:with|to|versus|vs\.?)\s+(?P<b>.+?)"
        r"(?:\s+for\b|\s+in\b|[?.]|$)",
        re.I,
    ),
    re.compile(
        r"\bdifference\s+between\s+(?P<a>.+?)\s+and\s+(?P<b>.+?)"
        r"(?:\s+for\b|\s+in\b|[?.]|$)",
        re.I,
    ),
    re.compile(
        r"\bcompar(?:e|er|ez)\s+(?P<a>.+?)\s+(?:avec|à|et|versus|vs\.?)\s+"
        r"(?P<b>.+?)(?:\s+pour\b|\s+dans\b|[?.]|$)",
        re.I,
    ),
)


def _compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t\r\n,;:-")


def _english_technical_projection(query: str) -> str:
    """Project only terminology from the shared multilingual lexicon."""
    normalized = query.casefold()
    additions: list[str] = []
    for expression, equivalents in TECHNICAL_EQUIVALENTS.items():
        if expression.casefold() not in normalized:
            continue
        for equivalent in equivalents:
            if equivalent.casefold() not in normalized and equivalent not in additions:
                additions.append(equivalent)
    return " ".join(additions)


def _extract_comparison_subjects(question: str) -> tuple[str, str] | None:
    for pattern in _COMPARISON_PATTERNS:
        match = pattern.search(question)
        if match is None:
            continue
        subject_a = _compact(match.group("a"))
        subject_b = _compact(match.group("b"))
        if subject_a and subject_b:
            return subject_a, subject_b
    return None


def _balance_kind(question: str) -> str:
    """Classify a balance structurally without encoding document facts."""

    normalized = question.casefold().replace("₂", "2").replace("₅", "5")

    energy_terms = (
        "energy",
        "heat",
        "enthalpy",
        "energie",
        "énergie",
        "thermique",
        "chaleur",
    )
    if any(term in normalized for term in energy_terms):
        return "energy"

    species_terms = (
        "species",
        "component",
        "constituent",
        "espèce",
        "espece",
        "composant",
        "constituant",
    )
    if any(term in normalized for term in species_terms):
        return "species"

    # Generic chemical-formula detection, e.g. H2O, CO2, X2Y5.
    if re.search(
        r"\b[a-z]{1,3}\d+(?:[a-z]\d*)+\b",
        normalized,
    ):
        return "species"

    return "overall_mass"


def build_retrieval_plan(
    original_question: str,
    *,
    standalone_query: str,
    question_type: str,
) -> RetrievalPlan:
    """Create generic information-needs for retrieval.

    Roles describe only the structure of information to search for.
    They never encode the expected documentary answer.
    """

    original = _compact(original_question)
    standalone = _compact(standalone_query)

    if not original or not standalone:
        raise ValueError("Les questions du plan de retrieval ne peuvent pas être vides.")

    projected = _english_technical_projection(f"{original} {standalone}")
    base_query = _compact(" ".join(part for part in (standalone, projected) if part))

    def role(
        name: str,
        hint: str,
        *,
        subject: str | None = None,
    ) -> EvidenceRole:
        query = _compact(f"{base_query} {hint}")
        return EvidenceRole(
            name=name,
            query=query,
            subject=subject,
        )

    if question_type == "comparison":
        subjects = _extract_comparison_subjects(original) or _extract_comparison_subjects(
            standalone
        )

        if subjects is None:
            return RetrievalPlan(
                question_type=question_type,
                base_query=base_query,
                roles=(
                    role(
                        "comparison_context",
                        "comparison similarities differences criteria",
                    ),
                ),
            )

        subject_a, subject_b = subjects

        return RetrievalPlan(
            question_type=question_type,
            base_query=base_query,
            roles=(
                role(
                    "subject_a",
                    subject_a,
                    subject=subject_a,
                ),
                role(
                    "subject_b",
                    subject_b,
                    subject=subject_b,
                ),
                role(
                    "comparison_criteria",
                    "comparison criteria similarities differences",
                ),
            ),
            comparison_subjects=subjects,
        )

    role_hints: dict[str, tuple[tuple[str, str], ...]] = {
        "definition": (
            ("definition", "definition nature"),
            ("mechanism", "mechanism operation"),
            ("function", "function purpose"),
        ),
        "explanation": (
            ("core_explanation", "explanation mechanism"),
            ("relations", "relationships causes effects"),
        ),
        "process_flow": (
            ("sequence_overview", "process sequence flow path"),
            ("entry_context", "entry inlet beginning"),
            (
                "transitions",
                "transitions connections intermediate stages",
            ),
            ("exit_context", "exit outlet end"),
        ),
        "procedure": (
            ("prerequisites", "prerequisites initial conditions"),
            ("ordered_actions", "ordered actions procedure sequence"),
            ("outcome", "result outcome completion"),
        ),
        "balance": (
            ("governing_relation", "conservation relation balance equation"),
            ("inputs", "input quantities variables"),
            ("outputs", "output quantities variables"),
            ("assumptions_units", "assumptions units basis"),
        ),
        "equation_explanation": (
            ("governing_relation", "governing relation equation"),
            ("variables", "variables symbols definitions"),
            ("assumptions", "assumptions applicability"),
        ),
        "thermodynamic_relation": (
            ("governing_relation", "governing relation equation"),
            ("variables", "variables properties definitions"),
            ("conditions", "conditions assumptions applicability"),
        ),
        "calculation": (
            ("governing_relation", "governing relation calculation"),
            ("inputs", "input values variables units"),
            ("method", "calculation method"),
        ),
        "troubleshooting": (
            ("symptom", "observed symptom problem"),
            ("causes", "possible documented causes"),
            ("mechanism", "physical mechanism"),
            ("effects", "documented effects consequences"),
            ("actions", "documented corrective actions mitigation"),
        ),
        "momentum_diffusion": (
            ("concept", "definition concept"),
            ("governing_relation", "governing relation"),
            ("variables", "variables physical meaning"),
        ),
        "control_strategy": (
            ("objective", "control objective"),
            ("observations", "measured observed variables"),
            ("actions", "control actions manipulated variables"),
            ("strategy", "control strategy operation"),
        ),
        "table_question": (("table_context", "table data values units"),),
        "plant_specific": (
            ("plant_context", "documented plant context data"),
            ("requested_fact", "requested documented information"),
        ),
    }

    hints = role_hints.get(question_type)

    if hints is None:
        roles = (
            role(
                "primary",
                "relevant supporting evidence",
            ),
        )
    else:
        roles = tuple(role(name, hint) for name, hint in hints)

    return RetrievalPlan(
        question_type=question_type,
        base_query=base_query,
        roles=roles,
        balance_kind=(_balance_kind(base_query) if question_type == "balance" else None),
    )
