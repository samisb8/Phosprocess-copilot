# ruff: noqa: E501
"""Objective and manual-annotation support for Phase-10 generation outputs.

This module is evaluation-only.  It never imports from production code and it
does not use an LLM as a correctness judge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from phosprocess.evaluation.context_engine_v01 import read_jsonl, write_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "data/evaluation/generation_baseline/v0.1"
EVALUATOR_MODULE = Path(__file__).resolve()

VALID_CITATION_RE = re.compile(r"\[Source\s+(\d+)\]")
ANY_CITATION_RE = re.compile(r"\[[^\]]*Source[^\]]*\]", re.IGNORECASE)
NUMBER_RE = re.compile(r"(?<![\w])[-+]?\d+(?:[.,]\d+)?(?:\s*[×x]\s*10\s*\^?\s*[-+]?\d+)?")
UNIT_RE = re.compile(
    r"(?i)(?:°\s*C|%|mmHg|torrs?|kPa|MPa|Pa|bar|kg\s*/\s*h|t\s*/\s*h|tons?\s*/\s*day|m\s*[²³23]?|m3|W\s*/\s*m\s*[²2]?\s*K|K)\b|%"
)


def _normal(value: str) -> str:
    return re.sub(r"\s+", "", value.casefold().replace(",", "."))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_atomic_claims(answer: str) -> list[dict[str, Any]]:
    """Split prose/list answers without deciding truth or correctness."""

    text = re.sub(r"\r\n?", "\n", answer).strip()
    blocks = re.split(r"\n+|(?<=[.!?؟])\s+(?=[A-ZÀ-ÖØ-Þ\u0600-\u06ff0-9])", text)
    claims: list[dict[str, Any]] = []
    for block in blocks:
        cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", block).strip()
        if not cleaned:
            continue
        # Semicolons reliably delimit independently checkable clauses in this
        # dataset.  Coordinating conjunctions are deliberately left intact so
        # extraction does not invent implicit subjects.
        for clause in re.split(r"\s*;\s*", cleaned):
            claim = clause.strip()
            if not claim:
                continue
            citations = [int(value) for value in VALID_CITATION_RE.findall(claim)]
            factual_text = VALID_CITATION_RE.sub("", claim).strip()
            if factual_text:
                claims.append(
                    {
                        "claim_text": factual_text,
                        "citation_numbers": citations,
                        "important": True,
                    }
                )
    return claims


def _bundle_chunk_ids(bundle: dict[str, Any]) -> set[str]:
    keys = ("anchor_chunk_ids", "supporting_chunk_ids", "expanded_chunk_ids")
    values: set[str] = set()
    for key in keys:
        values.update(bundle.get(key) or [])
    if bundle.get("chunk_id"):
        values.add(bundle["chunk_id"])
    return values


def evidence_coverage(
    valid_sets: list[dict[str, Any]],
    available_ids: set[str],
) -> tuple[bool, float]:
    """Apply the frozen Phase-8 alternative/complementary semantics."""

    if not valid_sets:
        return False, 0.0
    coverages: list[float] = []
    exact: list[bool] = []
    for evidence_set in valid_sets:
        if evidence_set["type"] == "alternative":
            ids = set(evidence_set["chunk_ids"])
            value = 1.0 if ids & available_ids else 0.0
        else:
            groups = evidence_set["groups"]
            satisfied = sum(bool(set(group) & available_ids) for group in groups)
            value = satisfied / len(groups) if groups else 0.0
        coverages.append(value)
        exact.append(value == 1.0)
    return any(exact), max(coverages)


def _detected_language(text: str) -> str:
    letters = [char for char in text if char.isalpha()]
    if letters and sum("\u0600" <= char <= "\u06ff" for char in letters) / len(letters) > 0.25:
        return "ar"
    lowered = f" {text.casefold()} "
    fr = sum(lowered.count(token) for token in (" le ", " la ", " les ", " des ", " une ", " est ", " dans ", " pour ", "l’", "d’"))
    en = sum(lowered.count(token) for token in (" the ", " is ", " are ", " of ", " to ", " in ", " and ", " for ", " with "))
    return "fr" if fr > en else "en"


def _objective_record(row: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    response = row["response"]
    answer = response["answer"]
    bundles = row["retrieval"]["evidence_bundles"]
    bundle_by_number = {int(bundle["source_number"]): bundle for bundle in bundles}
    valid_refs = [int(value) for value in VALID_CITATION_RE.findall(answer)]
    citation_tokens = ANY_CITATION_RE.findall(answer)
    invalid_syntax = [value for value in citation_tokens if not VALID_CITATION_RE.fullmatch(value)]
    nonexistent = sorted({value for value in valid_refs if value not in bundle_by_number})
    wrong_document = sorted(
        {
            value
            for value in valid_refs
            if value in bundle_by_number
            and bundle_by_number[value]["document_id"] not in row["retrieval"]["source_lock"]
        }
    )
    metadata_invalid = sorted(
        {
            value
            for value in valid_refs
            if value in bundle_by_number
            and (
                not bundle_by_number[value].get("document_id")
                or not bundle_by_number[value].get("filename")
                or bundle_by_number[value].get("page_start") is None
            )
        }
    )
    context_ids = set().union(*(_bundle_chunk_ids(bundle) for bundle in bundles)) if bundles else set()
    hybrid_ids = {item["chunk_id"] for item in row["retrieval"]["hybrid_candidates"]}
    if question.get("dataset_scope") == "phase8_primary":
        context_exact, context_fraction = evidence_coverage(question["valid_evidence_sets"], context_ids)
        retrieval_exact, retrieval_fraction = evidence_coverage(question["valid_evidence_sets"], hybrid_ids)
    else:
        context_exact, context_fraction = False, 0.0
        retrieval_exact, retrieval_fraction = False, 0.0

    claims = extract_atomic_claims(answer)
    numeric_checks: list[dict[str, Any]] = []
    for claim_index, claim in enumerate(claims, 1):
        for match in NUMBER_RE.finditer(claim["claim_text"]):
            number = match.group(0)
            refs = claim["citation_numbers"]
            cited_text = "\n".join(
                bundle_by_number[ref].get("display_text", "")
                for ref in refs
                if ref in bundle_by_number
            )
            nearby = claim["claim_text"][match.end() : match.end() + 30]
            unit_match = UNIT_RE.search(nearby)
            unit = unit_match.group(0) if unit_match else None
            numeric_checks.append(
                {
                    "claim_index": claim_index,
                    "number": number,
                    "unit": unit,
                    "citation_numbers": refs,
                    "number_in_cited_evidence": bool(cited_text) and _normal(number) in _normal(cited_text),
                    "unit_in_cited_evidence": (
                        None if unit is None else bool(cited_text) and _normal(unit) in _normal(cited_text)
                    ),
                }
            )

    latency = response.get("latency") or {}
    return {
        "id": row["id"],
        "source_question_id": row.get("source_question_id", row["id"]),
        "record_type": row["record_type"],
        "dataset_scope": row["dataset_scope"],
        "split": row["split"],
        "language": row["language"],
        "detected_language": _detected_language(answer),
        "language_correct": _detected_language(answer) == row["language"],
        "answerability": row["answerability"],
        "insufficient_context_flag": response["insufficient_context"],
        "citation_occurrences": len(valid_refs),
        "citation_present": bool(valid_refs),
        "citation_syntax_valid": not invalid_syntax,
        "citation_existing_bundle_valid": not nonexistent,
        "citation_locked_document_valid": not wrong_document,
        "citation_metadata_valid": not metadata_invalid,
        "invalid_citation_syntax": invalid_syntax,
        "nonexistent_citations": nonexistent,
        "wrong_document_citations": wrong_document,
        "invalid_metadata_citations": metadata_invalid,
        "claim_count": len(claims),
        "claims_with_citation": sum(bool(claim["citation_numbers"]) for claim in claims),
        "numeric_checks": numeric_checks,
        "evidence_available_in_context": context_exact,
        "evidence_context_coverage": context_fraction,
        "evidence_available_in_retrieval": retrieval_exact,
        "evidence_retrieval_coverage": retrieval_fraction,
        "context_packing_miss": retrieval_exact and not context_exact,
        "selected_document": row["retrieval"]["selected_document"],
        "source_lock_exact": row["retrieval"]["source_lock_exact"],
        "latency_ms": response["timings"]["total_ms"],
        "first_token_ms": response["timings"].get("first_token_ms"),
        "generated_tokens": latency.get("generated_token_count"),
        "estimated_input_tokens": latency.get("estimated_prompt_tokens"),
        "context_tokens": latency.get("document_context_token_count"),
        "repair_attempted": bool(latency.get("repair_attempted")),
    }


def build_objective_artifacts(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    questions = {
        row["id"]: row
        for row in read_jsonl(output_directory / "questions_snapshot.jsonl")
    }
    rows = read_jsonl(output_directory / "generation_results.jsonl")
    objective: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        source_id = row.get("source_question_id", row["id"])
        question = questions[source_id]
        item = _objective_record(row, question)
        objective.append(item)
        for index, claim in enumerate(extract_atomic_claims(row["response"]["answer"]), 1):
            claims.append(
                {
                    "record_id": row["id"],
                    "source_question_id": source_id,
                    "split": row["split"],
                    "language": row["language"],
                    "claim_index": index,
                    **claim,
                    "support_label": None,
                    "supporting_source_numbers": [],
                    "manual_rationale": None,
                    "reviewer": None,
                }
            )
    write_jsonl(output_directory / "objective_checks.jsonl", objective)
    annotation_path = output_directory / "claim_annotations.jsonl"
    if not annotation_path.exists():
        write_jsonl(annotation_path, claims)
    else:
        existing_annotations = read_jsonl(annotation_path)
        existing_by_key = {
            (row["record_id"], row["claim_index"]): row
            for row in existing_annotations
        }
        claims_by_key = {
            (row["record_id"], row["claim_index"]): row for row in claims
        }
        if not set(existing_by_key).issubset(claims_by_key):
            raise RuntimeError("claim extraction changed after annotations were created")
        for key, old in existing_by_key.items():
            if old["claim_text"] != claims_by_key[key]["claim_text"]:
                raise RuntimeError("claim text changed after annotations were created")
            claims_by_key[key] = old
        write_jsonl(annotation_path, list(claims_by_key.values()))

    primary = [
        item
        for item in objective
        if item["dataset_scope"] == "phase8_primary"
        and item["record_type"] == "primary"
    ]
    numeric = [check for item in objective for check in item["numeric_checks"]]
    with_units = [check for check in numeric if check["unit"] is not None]
    context_token_values = [
        item["context_tokens"]
        for item in objective
        if item["context_tokens"] is not None
    ]
    summary = {
        "records": len(objective),
        "primary_records": len(primary),
        "claim_records": len(claims),
        "objective": {
            "citation_validity": sum(
                item["citation_syntax_valid"]
                and item["citation_existing_bundle_valid"]
                and item["citation_locked_document_valid"]
                and item["citation_metadata_valid"]
                for item in objective
            )
            / len(objective),
            "citation_coverage": sum(item["claims_with_citation"] for item in objective)
            / max(1, sum(item["claim_count"] for item in objective)),
            "numeric_grounding": sum(item["number_in_cited_evidence"] for item in numeric)
            / max(1, len(numeric)),
            "unit_grounding": sum(item["unit_in_cited_evidence"] for item in with_units)
            / max(1, len(with_units)),
            "correct_language_rate": sum(item["language_correct"] for item in objective)
            / len(objective),
            "evidence_available_context_primary": sum(item["evidence_available_in_context"] for item in primary)
            / max(1, len(primary)),
            "context_packing_misses": sum(item["context_packing_miss"] for item in primary),
            "citation_repairs": sum(item["repair_attempted"] for item in objective),
        },
        "system": {
            "latency_ms_median": statistics.median(item["latency_ms"] for item in objective),
            "latency_ms_mean": statistics.mean(item["latency_ms"] for item in objective),
            "first_token_ms_median": statistics.median(
                item["first_token_ms"] for item in objective if item["first_token_ms"] is not None
            ),
            "generated_tokens_median": statistics.median(
                item["generated_tokens"] for item in objective if item["generated_tokens"] is not None
            ),
            "input_tokens_median": statistics.median(
                item["estimated_input_tokens"] for item in objective if item["estimated_input_tokens"] is not None
            ),
            "context_tokens_median": (
                statistics.median(context_token_values)
                if context_token_values
                else None
            ),
        },
        "counts_by_split": dict(Counter(item["split"] for item in objective)),
        "counts_by_language": dict(Counter(item["language"] for item in objective)),
        "manual_annotations_complete": all(row["support_label"] is not None for row in read_jsonl(annotation_path)),
        "semantic_judge_used": False,
    }
    (output_directory / "objective_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def freeze_evaluator(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Freeze deterministic evaluator code and DEV-derived artifacts."""

    if (output_directory / "holdout_open_manifest.json").exists() or (
        output_directory / "holdout_generation_results.jsonl"
    ).exists():
        raise RuntimeError("cannot freeze evaluator after FINAL HOLDOUT was opened")
    path = output_directory / "evaluator_freeze_manifest.json"
    if path.exists():
        raise RuntimeError("evaluator is already frozen")
    summary = build_objective_artifacts(output_directory)
    manifest = {
        "scope": "DEV only",
        "evaluator_sha256": _sha256_file(EVALUATOR_MODULE),
        "evaluation_protocol_sha256": _sha256_file(output_directory / "evaluation_protocol.json"),
        "dev_results_sha256": _sha256_file(output_directory / "dev_generation_results.jsonl"),
        "objective_checks_sha256": _sha256_file(output_directory / "objective_checks.jsonl"),
        "claim_extraction_sha256": _sha256_file(output_directory / "claim_annotations.jsonl"),
        "dev_record_count": summary["counts_by_split"].get("dev", 0),
        "holdout_seen": False,
        "semantic_judge_used": False,
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--freeze-evaluator", action="store_true")
    args = parser.parse_args()
    if args.freeze_evaluator:
        result = freeze_evaluator(args.output)
    elif args.build:
        result = build_objective_artifacts(args.output)
    else:
        parser.error("choose --build or --freeze-evaluator")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
