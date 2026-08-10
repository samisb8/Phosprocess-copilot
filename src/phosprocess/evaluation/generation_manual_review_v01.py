"""Build reviewer packets and compile human Phase-10 annotations."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from phosprocess.evaluation.context_engine_v01 import read_jsonl, write_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "data/evaluation/generation_baseline/v0.1"
TOKEN_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)


def _tokens(value: str) -> set[str]:
    stop = {
        "the", "and", "that", "with", "from", "this", "pour", "dans", "une",
        "les", "des", "est", "par", "sur", "qui", "plus", "من", "في", "على",
        "إلى", "التي", "هذا", "هذه", "كما", "عن",
    }
    return {token.casefold() for token in TOKEN_RE.findall(value) if token.casefold() not in stop}


def _best_evidence_windows(claim: str, text: str, limit: int = 3) -> list[str]:
    claim_tokens = _tokens(claim)
    sentences = [item.strip() for item in re.split(r"\n+|(?<=[.!?؟])\s+", text) if item.strip()]
    ranked = sorted(
        sentences,
        key=lambda item: (
            len(claim_tokens & _tokens(item)) / max(1, len(claim_tokens)),
            len(claim_tokens & _tokens(item)),
        ),
        reverse=True,
    )
    return ranked[:limit]


def build_review_packets(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, int]:
    questions = {
        row["id"]: row
        for row in read_jsonl(output_directory / "questions_snapshot.jsonl")
    }
    results = {row["id"]: row for row in read_jsonl(output_directory / "generation_results.jsonl")}
    objective = {row["id"]: row for row in read_jsonl(output_directory / "objective_checks.jsonl")}
    claims = read_jsonl(output_directory / "claim_annotations.jsonl")
    claim_packet: list[dict[str, Any]] = []
    for claim in claims:
        row = results[claim["record_id"]]
        bundles = {
            int(bundle["source_number"]): bundle
            for bundle in row["retrieval"]["evidence_bundles"]
        }
        evidence = []
        for number in claim["citation_numbers"]:
            if number in bundles:
                bundle = bundles[number]
                evidence.append(
                    {
                        "source_number": number,
                        "document_id": bundle["document_id"],
                        "pages": [bundle.get("page_start"), bundle.get("page_end")],
                        "best_windows": _best_evidence_windows(
                            claim["claim_text"], bundle.get("display_text", "")
                        ),
                    }
                )
        claim_packet.append({**claim, "cited_evidence": evidence})
    write_jsonl(output_directory / "claim_review_packet.jsonl", claim_packet)

    question_packet: list[dict[str, Any]] = []
    for record_id, row in results.items():
        source_id = row.get("source_question_id", record_id)
        question = questions[source_id]
        item = objective[record_id]
        question_packet.append(
            {
                "record_id": record_id,
                "source_question_id": source_id,
                "record_type": row["record_type"],
                "dataset_scope": row["dataset_scope"],
                "split": row["split"],
                "language": row["language"],
                "answerability": row["answerability"],
                "question": row["question"],
                "answer": row["response"]["answer"],
                "expected_concepts": question.get("expected_concepts", []),
                "documentary_justification": question.get("documentary_justification"),
                "exact_gold_available_in_context": item["evidence_available_in_context"],
                "exact_gold_context_coverage": item["evidence_context_coverage"],
                "selected_document": item["selected_document"],
                "resolved_query": row["retrieval"]["resolved_query"]["standalone_query"],
                "insufficient_context_flag": row["response"]["insufficient_context"],
                "manual": {
                    "evidence_available": None,
                    "evidence_used": None,
                    "completeness": None,
                    "answer_success": None,
                    "refusal_class": None,
                    "failure_codes": [],
                    "rationale": None,
                },
            }
        )
    write_jsonl(output_directory / "question_review_packet.jsonl", question_packet)
    annotation_path = output_directory / "question_annotations.jsonl"
    if not annotation_path.exists():
        write_jsonl(annotation_path, question_packet)
    return {"questions": len(question_packet), "claims": len(claim_packet)}


def compile_manual_annotations(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, int]:
    """Compile reviewer-authored labels after strict key/count validation."""

    claim_labels = json.loads(
        (output_directory / "manual_claim_labels.json").read_text(encoding="utf-8")
    )
    question_labels = json.loads(
        (output_directory / "manual_question_labels.json").read_text(
            encoding="utf-8"
        )
    )
    claims = read_jsonl(output_directory / "claim_annotations.jsonl")
    by_record: dict[str, list[dict[str, Any]]] = {}
    for claim in claims:
        by_record.setdefault(claim["record_id"], []).append(claim)
    if set(claim_labels) != set(by_record):
        raise RuntimeError("manual claim label IDs do not exactly match extracted records")
    allowed_support = {"SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"}
    aliases = {
        "S": "SUPPORTED",
        "P": "PARTIALLY_SUPPORTED",
        "U": "UNSUPPORTED",
    }
    compiled_claims: list[dict[str, Any]] = []
    for record_id, record_claims in by_record.items():
        entries = claim_labels[record_id]
        if len(entries) != len(record_claims):
            raise RuntimeError(f"claim-label count mismatch for {record_id}")
        for claim, entry in zip(record_claims, entries, strict=True):
            detail = {"label": entry} if isinstance(entry, str) else entry
            label = aliases.get(detail["label"], detail["label"])
            citation_value = detail.get("citation_support", label)
            citation_support = aliases.get(citation_value, citation_value)
            if label not in allowed_support or citation_support not in allowed_support:
                raise RuntimeError(f"invalid claim label for {record_id}")
            compiled_claims.append(
                {
                    **claim,
                    "support_label": label,
                    "citation_support_label": (
                        citation_support if claim["citation_numbers"] else None
                    ),
                    "supporting_source_numbers": detail.get(
                        "sources",
                        (
                            claim["citation_numbers"]
                            if citation_support != "UNSUPPORTED"
                            else []
                        ),
                    ),
                    "manual_rationale": (
                        "Compared manually with the cited EvidenceBundle wording "
                        "and frozen documentary gold."
                    ),
                    "reviewer": "Codex manual documentary review",
                }
            )
    write_jsonl(output_directory / "claim_annotations.jsonl", compiled_claims)

    question_packet = read_jsonl(output_directory / "question_review_packet.jsonl")
    if set(question_labels) != {row["record_id"] for row in question_packet}:
        raise RuntimeError("manual question label IDs do not exactly match generated records")
    allowed_use = {"YES", "PARTIAL", "NO"}
    allowed_complete = {"COMPLETE", "MOSTLY_COMPLETE", "PARTIAL", "MISSED"}
    compiled_questions: list[dict[str, Any]] = []
    for row in question_packet:
        labels = question_labels[row["record_id"]]
        if labels["evidence_used"] not in allowed_use:
            raise RuntimeError(f"invalid evidence-use label for {row['record_id']}")
        if labels["completeness"] not in allowed_complete:
            raise RuntimeError(f"invalid completeness label for {row['record_id']}")
        compiled_questions.append(
            {
                **row,
                "manual": labels,
                "reviewer": "Codex manual documentary review",
            }
        )
    write_jsonl(output_directory / "question_annotations.jsonl", compiled_questions)
    return {"questions": len(compiled_questions), "claims": len(compiled_claims)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--build-packets", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    if args.compile:
        result = compile_manual_annotations(args.output)
    elif args.build_packets:
        result = build_review_packets(args.output)
    else:
        parser.error("choose --build-packets or --compile")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
