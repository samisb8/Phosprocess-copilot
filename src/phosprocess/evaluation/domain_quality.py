"""Validation for the independent production-domain DEV question set."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phosprocess.knowledge_base.runtime import PROJECT_ROOT

DEFAULT_DOMAIN_QUALITY_DIRECTORY = PROJECT_ROOT / "data" / "evaluation" / "domain_quality" / "v1"

MINIMUM_CATEGORIES = {
    "phosphoric_acid": 10,
    "thermodynamics": 7,
    "heat_transfer": 7,
    "transport_fluids": 6,
    "crystallization": 7,
    "control_mpc": 7,
    "atelier": 6,
    "conversation": 10,
}


@dataclass(frozen=True, slots=True)
class DomainQualitySummary:
    question_count: int
    language_counts: dict[str, int]
    category_counts: dict[str, int]
    conversation_question_count: int
    unanswerable_count: int


def load_domain_questions(
    path: Path = DEFAULT_DOMAIN_QUALITY_DIRECTORY / "questions.jsonl",
) -> list[dict[str, Any]]:
    """Load strict JSONL without accepting empty or malformed records."""

    records: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        value = json.loads(line)

        if not isinstance(value, dict):
            raise ValueError(f"Question non objet à la ligne {line_number}.")

        records.append(value)

    return records


def validate_domain_questions(
    records: list[dict[str, Any]],
) -> DomainQualitySummary:
    """Enforce scope, coverage, language and conversation requirements."""

    if not 40 <= len(records) <= 60:
        raise ValueError("Le DEV métier doit contenir entre 40 et 60 questions.")

    identifiers = [str(record.get("question_id", "")) for record in records]

    if any(not identifier for identifier in identifiers):
        raise ValueError("Chaque question exige un question_id.")

    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Les question_id DEV métier doivent être uniques.")

    languages: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    unanswerable = 0

    for record in records:
        question = record.get("question")
        language = record.get("language")
        record_categories = record.get("categories")

        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Question vide : {record['question_id']}.")

        if language not in {"fr", "en", "ar"}:
            raise ValueError(f"Langue invalide : {record['question_id']}.")

        if (
            not isinstance(record_categories, list)
            or not record_categories
            or not all(isinstance(item, str) for item in record_categories)
        ):
            raise ValueError(f"Catégories invalides : {record['question_id']}.")

        languages[language] += 1
        categories.update(record_categories)
        unanswerable += record.get("answerability") == "unanswerable"

    missing = {
        category: minimum - categories[category]
        for category, minimum in MINIMUM_CATEGORIES.items()
        if categories[category] < minimum
    }

    if missing:
        raise ValueError(f"Couverture DEV métier insuffisante : {missing}")

    if not all(languages[language] > 0 for language in ("fr", "en", "ar")):
        raise ValueError("Le DEV métier doit couvrir français, anglais et arabe.")

    return DomainQualitySummary(
        question_count=len(records),
        language_counts=dict(languages),
        category_counts=dict(categories),
        conversation_question_count=categories["conversation"],
        unanswerable_count=unanswerable,
    )
