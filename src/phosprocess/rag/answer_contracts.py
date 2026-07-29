"""Question-type answer contracts and deterministic repair selection."""

from __future__ import annotations

import re
from dataclasses import dataclass

from phosprocess.rag.citation_binding import (
    _fallback_answer,
    _iter_answer_claims,
    build_atomic_process_flow_answer,
)
from phosprocess.rag.claim_support import _CITATION, _normalize
from phosprocess.rag.deterministic_builders import (
    _COMPARISON_CRITERIA_MARKERS,
    _DEFINITION_FUNCTION_MARKERS,
    _DEFINITION_MARKERS,
    _DEFINITION_MECHANISM_MARKERS,
    _TROUBLESHOOTING_PROBLEM_MARKERS,
    _TROUBLESHOOTING_ROLE_MARKERS,
    _infer_balance_kind,
    build_deterministic_balance_answer,
    build_deterministic_definition_answer,
    build_deterministic_fouling_answer,
    build_deterministic_momentum_diffusion_answer,
    build_deterministic_scoped_explanation,
)
from phosprocess.retrieval.evidence_bundle import EvidenceBundle


@dataclass(frozen=True, slots=True)
class AnswerContractResult:
    """Deterministic end-to-end answer contract normalization."""

    answer: str
    changed: bool
    fallback_used: bool
    missing_roles: tuple[str, ...] = ()
    removed_claims: tuple[str, ...] = ()
    atomic_plan_used: bool = False


def _answer_claim_records(answer: str) -> list[tuple[str, str]]:
    """Return ``(claim_with_citations, normalized_claim)`` records."""

    records: list[tuple[str, str]] = []
    for claim in _iter_answer_claims(answer):
        clean = _CITATION.sub("", claim).strip()
        if clean:
            records.append((claim.strip(), _normalize(clean)))
    return records


def _contains_any_marker(text: str, markers: tuple[str, ...]) -> bool:
    return any(_normalize(marker) in text for marker in markers)


def _subject_aliases(subject: str) -> tuple[str, ...]:
    normalized = _normalize(subject)
    aliases = {normalized}
    aliases.add(re.sub(r"^(?:a|an|the|un|une|le|la|les)\s+", "", normalized))
    if "forced circulation" in normalized or "circulation forcee" in normalized:
        aliases.update(
            {
                "forced circulation evaporator",
                "forced circulation",
                "evaporateur a circulation forcee",
                "circulation forcee",
            }
        )
    if "falling film" in normalized or "film tombant" in normalized:
        aliases.update(
            {
                "falling film evaporator",
                "falling film",
                "evaporateur a film tombant",
                "film tombant",
            }
        )
    return tuple(alias for alias in aliases if alias)


def _definition_contract(answer: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    records = _answer_claim_records(answer)
    combined = " ".join(text for _claim, text in records)
    roles: set[str] = set()

    if _contains_any_marker(f" {combined} ", _DEFINITION_MARKERS):
        roles.add("definition")
    if _contains_any_marker(combined, _DEFINITION_MECHANISM_MARKERS):
        roles.add("mechanism")
    if _contains_any_marker(combined, _DEFINITION_FUNCTION_MARKERS):
        roles.add("function")

    required = ("definition", "mechanism", "function")
    missing = tuple(role for role in required if role not in roles)
    return tuple(sorted(roles)), missing


def _comparison_contract(
    answer: str,
    *,
    subjects: tuple[str, ...],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    records = _answer_claim_records(answer)
    subject_alias_groups = tuple(_subject_aliases(subject) for subject in subjects[:2])
    kept: list[str] = []
    removed: list[str] = []
    covered_subjects: set[int] = set()
    criterion_present = False

    for claim, normalized in records:
        matched_subjects = {
            index
            for index, aliases in enumerate(subject_alias_groups)
            if any(alias in normalized for alias in aliases)
        }
        has_criterion = _contains_any_marker(
            normalized,
            _COMPARISON_CRITERIA_MARKERS,
        )
        if not matched_subjects or not has_criterion:
            removed.append(_CITATION.sub("", claim).strip())
            continue
        kept.append(claim)
        covered_subjects.update(matched_subjects)
        criterion_present = True

    missing: list[str] = []
    if len(subject_alias_groups) >= 1 and 0 not in covered_subjects:
        missing.append("equipment_a")
    if len(subject_alias_groups) >= 2 and 1 not in covered_subjects:
        missing.append("equipment_b")
    if not criterion_present:
        missing.append("comparison_criteria")

    return "\n".join(kept), tuple(missing), tuple(removed)


def _troubleshooting_contract(
    answer: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    records = _answer_claim_records(answer)
    retained: list[tuple[int, int, str, set[str]]] = []
    removed: list[str] = []
    covered_roles: set[str] = set()

    for index, (claim, normalized) in enumerate(records):
        roles = {
            role
            for role, markers in _TROUBLESHOOTING_ROLE_MARKERS.items()
            if _contains_any_marker(normalized, markers)
        }
        problem_related = _contains_any_marker(
            normalized,
            _TROUBLESHOOTING_PROBLEM_MARKERS,
        )
        action_specific = "action" in roles and problem_related
        if not problem_related and not action_specific:
            removed.append(_CITATION.sub("", claim).strip())
            continue
        if not roles:
            removed.append(_CITATION.sub("", claim).strip())
            continue

        covered_roles.update(roles)
        priority = min(
            ("cause", "mechanism", "effect", "action").index(role)
            for role in roles
        )
        retained.append((priority, index, claim, roles))

    retained.sort(key=lambda item: (item[0], item[1]))
    required = ("cause", "mechanism", "effect", "action")
    missing = tuple(role for role in required if role not in covered_roles)
    return (
        "\n".join(item[2] for item in retained),
        missing,
        tuple(removed),
    )


def enforce_answer_contract(
    answer: str,
    bundles: list[EvidenceBundle],
    *,
    question_type: str | None,
    language: str,
    comparison_subjects: tuple[str, ...] = (),
    question: str = "",
    balance_kind: str | None = None,
) -> AnswerContractResult:
    """Apply deterministic task contracts after grounding validation.

    The contract never creates comparison or troubleshooting facts.  It only
    removes grounded-but-off-task claims, orders the remaining claims, or uses
    the existing source-local atomic planner for process flow.
    """

    normalized_type = (question_type or "").strip().lower()

    if normalized_type == "process_flow":
        atomic = build_atomic_process_flow_answer(bundles, language=language)
        if atomic is None:
            return AnswerContractResult(
                answer=_fallback_answer(language),
                changed=True,
                fallback_used=True,
                missing_roles=(
                    "feed_inlet",
                    "conical_bottom",
                    "pump_heat_exchanger",
                    "recirculation_vapor_body",
                    "product_outlet",
                ),
                atomic_plan_used=True,
            )
        return AnswerContractResult(
            answer=atomic,
            changed=atomic != answer,
            fallback_used=False,
            atomic_plan_used=True,
        )

    if normalized_type == "definition":
        deterministic = build_deterministic_definition_answer(
            bundles,
            language=language,
        )
        if deterministic is not None:
            return AnswerContractResult(
                answer=deterministic,
                changed=deterministic != answer,
                fallback_used=False,
            )

        _covered, missing = _definition_contract(answer)
        if missing:
            return AnswerContractResult(
                answer=_fallback_answer(language),
                changed=True,
                fallback_used=True,
                missing_roles=missing,
            )
        return AnswerContractResult(
            answer=answer,
            changed=False,
            fallback_used=False,
        )

    if normalized_type == "balance":
        kind = balance_kind or _infer_balance_kind(question)
        deterministic = build_deterministic_balance_answer(
            bundles,
            balance_kind=kind,
            language=language,
        )
        if deterministic is None:
            return AnswerContractResult(
                answer=_fallback_answer(language),
                changed=True,
                fallback_used=True,
                missing_roles=(f"{kind}_balance",),
            )
        return AnswerContractResult(
            answer=deterministic,
            changed=deterministic != answer,
            fallback_used=False,
        )

    if normalized_type == "momentum_diffusion":
        deterministic = build_deterministic_momentum_diffusion_answer(
            bundles,
            language=language,
        )
        if deterministic is None:
            return AnswerContractResult(
                answer=_fallback_answer(language),
                changed=True,
                fallback_used=True,
                missing_roles=(
                    "momentum_transport",
                    "velocity_gradient",
                    "newton_viscosity_law",
                ),
            )
        return AnswerContractResult(
            answer=deterministic,
            changed=deterministic != answer,
            fallback_used=False,
        )

    if normalized_type == "explanation":
        scoped = build_deterministic_scoped_explanation(
            question,
            bundles,
            language=language,
        )
        if scoped is not None:
            return AnswerContractResult(
                answer=scoped,
                changed=scoped != answer,
                fallback_used=False,
            )

    if normalized_type == "comparison":
        normalized, missing, removed = _comparison_contract(
            answer,
            subjects=comparison_subjects,
        )
        if missing or not normalized.strip():
            return AnswerContractResult(
                answer=_fallback_answer(language),
                changed=True,
                fallback_used=True,
                missing_roles=missing or ("comparison_claims",),
                removed_claims=removed,
            )
        return AnswerContractResult(
            answer=normalized,
            changed=normalized != answer,
            fallback_used=False,
            removed_claims=removed,
        )

    if normalized_type == "troubleshooting":
        normalized_question = _normalize(question)
        if _contains_any_marker(
            normalized_question,
            _TROUBLESHOOTING_PROBLEM_MARKERS,
        ):
            deterministic = build_deterministic_fouling_answer(
                bundles,
                language=language,
            )
            if deterministic is not None:
                return AnswerContractResult(
                    answer=deterministic,
                    changed=deterministic != answer,
                    fallback_used=False,
                )

        normalized, missing, removed = _troubleshooting_contract(answer)
        if missing or not normalized.strip():
            return AnswerContractResult(
                answer=_fallback_answer(language),
                changed=True,
                fallback_used=True,
                missing_roles=missing or ("troubleshooting_claims",),
                removed_claims=removed,
            )
        return AnswerContractResult(
            answer=normalized,
            changed=normalized != answer,
            fallback_used=False,
            removed_claims=removed,
        )

    return AnswerContractResult(
        answer=answer,
        changed=False,
        fallback_used=False,
    )
