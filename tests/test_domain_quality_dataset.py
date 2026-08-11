"""Coverage tests for the independent production-domain DEV set."""

from __future__ import annotations

from phosprocess.evaluation.domain_quality import (
    load_domain_questions,
    validate_domain_questions,
)


def test_domain_quality_dataset_has_required_coverage() -> None:
    summary = validate_domain_questions(load_domain_questions())

    assert summary.question_count == 50
    assert summary.conversation_question_count >= 10
    assert set(summary.language_counts) == {"fr", "en", "ar"}


def test_domain_quality_records_are_not_historical_test_questions() -> None:
    identifiers = {
        record["question_id"]
        for record in load_domain_questions()
    }

    assert all(identifier.startswith("DQ") for identifier in identifiers)
