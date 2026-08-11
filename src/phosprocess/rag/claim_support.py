"""Objective numeric and unit validation for cited documentary claims.

Semantic entailment, causality, direction and answer completeness are outside
this module and are assessed only during evaluation. Python only rejects
measurements that are absent from every source cited by the claim.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation

from phosprocess.rag.citation_binding import iter_answer_claims
from phosprocess.rag.citations import CitationValidationError
from phosprocess.retrieval.evidence_bundle import EvidenceBundle

_CITATION = re.compile(r"\[Source ([1-9]\d*)\]")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
_UNIT = re.compile(
    r"(?:%|°c|k|pa|kpa|mpa|bar|mbar|torr|kg/s|kg/h|t/h|m3/h|m/s|"
    r"kw|mw|w|kj/kg|kj/mol|j/mol|kg/m3)(?!\w)",
    re.I,
)
_LIST_PREFIX = re.compile(r"^\s*\d+[.)]\s*")

_NUMBER_WORDS = {
    "deux": "2",
    "trois": "3",
    "two": "2",
    "three": "3",
    "four": "4",
}

_UNIT_NORMALIZATIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:tonnes?|tons?|t)\s*(?:/|per)\s*(?:h|hr|hours?|heures?)\b"), "t/h"),
    (re.compile(r"\bkg\s*(?:/|per)\s*(?:s|sec|seconds?|secondes?)\b"), "kg/s"),
    (re.compile(r"\bkg\s*(?:/|per)\s*(?:h|hr|hours?|heures?)\b"), "kg/h"),
    (
        re.compile(
            r"\b(?:m3|m\^3|cubic\s+met(?:er|re)s?)\s*(?:/|per)\s*"
            r"(?:h|hr|hours?|heures?)\b"
        ),
        "m3/h",
    ),
    (
        re.compile(
            r"\b(?:m|met(?:er|re)s?)\s*(?:/|per)\s*"
            r"(?:s|sec|seconds?|secondes?)\b"
        ),
        "m/s",
    ),
    (re.compile(r"(?:°\s*c|degrees?\s*c(?:elsius)?|degres?\s*c(?:elsius)?)\b"), "°c"),
    (re.compile(r"\b(?:percent|pour\s*cent)\b"), "%"),
)


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", without_marks).strip()


def _measurement_text(text: str) -> str:
    normalized = _LIST_PREFIX.sub("", _normalize(text))
    normalized = normalized.translate(str.maketrans({"–": "-", "—": "-", "−": "-", "³": "3"}))
    normalized = re.sub(r"(?<=\d)\s*[.,]\s*(?=\d)", ".", normalized)

    previous = None
    while previous != normalized:
        previous = normalized
        normalized = re.sub(r"(?<=\d)\s+(?=\d{3}(?:\D|$))", "", normalized)

    for pattern, replacement in _UNIT_NORMALIZATIONS:
        normalized = pattern.sub(replacement, normalized)

    return re.sub(r"(?<=\d)\s*%", "%", normalized)


def _canonical_number(value: str) -> str:
    try:
        number = Decimal(value)
    except InvalidOperation:
        return value

    if number == number.to_integral():
        return str(number.quantize(Decimal("1")))

    return format(number.normalize(), "f")


def _numbers(text: str) -> set[str]:
    normalized = _measurement_text(text)
    found = {_canonical_number(match.group(0)) for match in _NUMBER.finditer(normalized)}
    for word, value in _NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", normalized):
            found.add(value)
    return found


def _units(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _UNIT.finditer(_measurement_text(text))}


def _measurement_gaps(claim: str, evidence_text: str) -> tuple[set[str], set[str]]:
    return _numbers(claim) - _numbers(evidence_text), _units(claim) - _units(evidence_text)


def validate_claim_support(answer: str, bundles: list[EvidenceBundle]) -> None:
    """Reject only cited numbers or units absent from every cited bundle."""

    bundle_by_number = {bundle.source_number: bundle for bundle in bundles}
    invalid_claims: list[str] = []

    for raw_claim in iter_answer_claims(answer):
        citations = tuple(dict.fromkeys(int(match) for match in _CITATION.findall(raw_claim)))
        if not citations:
            continue

        claim = _CITATION.sub("", raw_claim).strip()
        diagnostics: list[str] = []

        for source_number in citations:
            bundle = bundle_by_number.get(source_number)
            if bundle is None:
                diagnostics.append(f"source={source_number}; source inconnue")
                continue

            missing_numbers, missing_units = _measurement_gaps(claim, bundle.display_text)
            if not missing_numbers and not missing_units:
                break

            details = [f"source={source_number}"]
            if missing_numbers:
                details.append("valeurs=" + ",".join(sorted(missing_numbers)))
            if missing_units:
                details.append("unités=" + ",".join(sorted(missing_units)))
            diagnostics.append("; ".join(details))
        else:
            invalid_claims.append(f"{claim} ({' | '.join(diagnostics)})")

    if invalid_claims:
        raise CitationValidationError(
            "Valeur ou unité absente des sources citées : " + " | ".join(invalid_claims[:3])
        )
