"""Run deterministic smoke validation on the independent domain DEV set."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime

from phosprocess.evaluation.domain_quality import (
    DEFAULT_DOMAIN_QUALITY_DIRECTORY,
    load_domain_questions,
    validate_domain_questions,
)
from phosprocess.knowledge_base.catalog import load_document_catalog
from phosprocess.rag.language import detect_response_language
from phosprocess.rag.question_classifier import classify_question
from phosprocess.retrieval.domain_router import route_query
from phosprocess.retrieval.query_expansion import expand_technical_query


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validation DEV métier indépendante des benchmarks historiques."
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--with-llm-judge", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)

    if arguments.with_llm_judge:
        raise SystemExit(
            "Le juge LLM optionnel exige d'abord des réponses et une revue humaine."
        )

    records = load_domain_questions()
    summary = validate_domain_questions(records)
    catalog = load_document_catalog()
    diagnostics = []

    for record in records:
        question = str(record["question"])
        language = detect_response_language(question)
        classification = classify_question(question)
        routing = route_query(question, catalog=catalog)
        expansion = expand_technical_query(question)
        diagnostics.append(
            {
                "question_id": record["question_id"],
                "language": language.language.value,
                "question_type": classification.question_type.value,
                "detected_domains": [
                    domain.value
                    for domain, _confidence in routing.detected_domains
                ],
                "preferred_documents": list(routing.preferred_documents),
                "hard_filter": (
                    sorted(routing.hard_filter)
                    if routing.hard_filter is not None
                    else None
                ),
                "added_terms": list(expansion.added_terms),
            }
        )

    output = DEFAULT_DOMAIN_QUALITY_DIRECTORY / "smoke_validation.json"
    output.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "scope": "domain_quality_dev_only",
                "historical_test_used": False,
                "question_count": summary.question_count,
                "language_counts": summary.language_counts,
                "category_counts": summary.category_counts,
                "conversation_question_count": (
                    summary.conversation_question_count
                ),
                "unanswerable_count": summary.unanswerable_count,
                "diagnostics": diagnostics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Questions DEV métier : {summary.question_count}")
    print(f"Langues : {summary.language_counts}")
    print(f"Scénarios conversationnels : {summary.conversation_question_count}")
    print(f"Rapport smoke : {output}")
    print("Benchmark TEST historique utilisé : non")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
