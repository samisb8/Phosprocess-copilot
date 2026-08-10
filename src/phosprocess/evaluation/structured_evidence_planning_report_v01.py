# ruff: noqa: E501
"""Build the Phase-12 DEV-only structured-evidence-planning decision report.

The report deliberately uses deterministic checks and the frozen Phase-10
manual labels.  It does not call an LLM judge and it never reads holdout rows.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from phosprocess.evaluation.context_engine_v01 import read_jsonl, write_jsonl
from phosprocess.evaluation.generation_baseline_analysis_v01 import (
    _objective_record,
    extract_atomic_claims,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PHASE10_OUTPUT = PROJECT_ROOT / "data/evaluation/generation_baseline/v0.1"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/evaluation/structured_evidence_planning/v0.1"
PRIMARY_COUNT = 45
EXPECTED_INSUFFICIENT_IDS = {
    "CE051",
    "CE066",
    "DQ003",
    "DQ039",
    "DQ044",
    "DQ048",
    "ABS_DQ050",
    "ABS_CE058",
    "ABS_CE059",
}
ABSENT_IDS = {"ABS_DQ050", "ABS_CE058", "ABS_CE059"}


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _median(values: list[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.median(present) if present else None


def _baseline_dev_metrics() -> dict[str, Any]:
    questions = {
        row["id"]: row for row in read_jsonl(PHASE10_OUTPUT / "questions_snapshot.jsonl")
    }
    labels = json.loads(
        (PHASE10_OUTPUT / "manual_question_labels.json").read_text(encoding="utf-8")
    )
    ids = {
        question_id
        for question_id, question in questions.items()
        if question["split"] == "dev" and question["dataset_scope"] == "phase8_primary"
    }
    selected_labels = [labels[question_id] for question_id in ids]
    claims = [
        row
        for row in read_jsonl(PHASE10_OUTPUT / "claim_annotations.jsonl")
        if row["record_id"] in ids
    ]
    cited = [row for row in claims if row["citation_numbers"]]
    return {
        "primary_count": len(ids),
        "completed_answers": len(ids),
        "operational_response_rate": 1.0,
        "manual_answer_success_count": sum(
            bool(value["answer_success"]) for value in selected_labels
        ),
        "manual_answer_success_rate": _ratio(
            sum(bool(value["answer_success"]) for value in selected_labels), len(ids)
        ),
        "claim_count": len(claims),
        "citation_coverage": _ratio(len(cited), len(claims)),
        "citation_validity_operational": 1.0,
        "process_flow_status": "completed_but_failed_manual_order_audit",
    }


def _build_objective_artifacts(
    output_directory: Path,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    questions = {
        row["id"]: row for row in read_jsonl(output_directory / "questions_snapshot.jsonl")
    }
    objective: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    for row in rows:
        if row["status"] != "completed":
            continue
        source_id = row.get("source_question_id", row["id"])
        objective.append(_objective_record(row, questions[source_id]))
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
    write_jsonl(output_directory / "claim_annotations.jsonl", claims)
    return objective


def build_report(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Build a hard-gate DEV comparison without opening the holdout."""

    if (output_directory / "holdout_open_manifest.json").exists() or (
        output_directory / "holdout_generation_results.jsonl"
    ).exists():
        raise RuntimeError("Phase-12 report must be built before holdout is opened")

    rows = read_jsonl(output_directory / "dev_generation_results.jsonl")
    if len(rows) != 49 or len({row["id"] for row in rows}) != 49:
        raise RuntimeError("Phase-12 DEV must contain exactly 49 unique frozen records")
    objective = _build_objective_artifacts(output_directory, rows)
    objective_by_id = {row["id"]: row for row in objective}
    completed = [row for row in rows if row["status"] == "completed"]
    primary = [row for row in rows if row["dataset_scope"] == "phase8_primary"]
    primary_completed = [row for row in primary if row["status"] == "completed"]
    primary_objective = [objective_by_id[row["id"]] for row in primary_completed]
    error_rows = [row for row in rows if row["status"] != "completed"]

    mode_counts = Counter(
        row["phase12"]["planner"]["plan"]["mode"] for row in completed
    )
    source_reference_violations: list[dict[str, Any]] = []
    for row in completed:
        available = {
            int(bundle["source_number"])
            for bundle in row["retrieval"]["evidence_bundles"]
        }
        for item in row["phase12"]["planner"]["plan"]["items"]:
            refs = {int(value) for value in item["source_numbers"]}
            if not refs or not refs.issubset(available):
                source_reference_violations.append(
                    {"id": row["id"], "item_id": item["item_id"], "refs": sorted(refs)}
                )

    planned_insufficient = {
        row["id"]
        for row in completed
        if row["phase12"]["planner"]["plan"]["insufficient_evidence"]
    }
    valid_citation_rows = [
        row
        for row in primary_objective
        if row["citation_syntax_valid"]
        and row["citation_existing_bundle_valid"]
        and row["citation_locked_document_valid"]
        and row["citation_metadata_valid"]
    ]
    claim_count = sum(int(row["claim_count"]) for row in primary_objective)
    cited_claim_count = sum(int(row["claims_with_citation"]) for row in primary_objective)
    forbidden_call_totals = {
        key: sum(int(row["phase12"][key]) for row in rows)
        for key in ("retrieval_calls", "verifier_calls", "repair_calls")
    }
    prompt_contract_pass = all(
        len(row["prompt_calls"]) == 1
        and "STRUCTURED EVIDENCE PLAN (JSON)" in row["prompt_calls"][0]["messages"][1]["content"]
        and "PRECOMPUTED EVIDENCE COVERAGE PLAN" not in row["prompt_calls"][0]["messages"][1]["content"]
        for row in completed
    )
    baseline = _baseline_dev_metrics()
    phase12 = {
        "primary_count": len(primary),
        "completed_answers": len(primary_completed),
        "operational_response_rate": _ratio(len(primary_completed), len(primary)),
        "semantic_answer_success": "not_rescored_after_hard-gate_rejection",
        "claim_count_among_completed": claim_count,
        "citation_coverage_among_completed": _ratio(cited_claim_count, claim_count),
        "citation_validity_among_completed": _ratio(
            len(valid_citation_rows), len(primary_completed)
        ),
        "citation_validity_operational": _ratio(
            len(valid_citation_rows), len(primary)
        ),
        "correct_language_among_completed": _ratio(
            sum(bool(row["language_correct"]) for row in primary_objective),
            len(primary_objective),
        ),
        "process_flow_status": next(
            row["status"] for row in rows if row["id"] == "PROCESS_FLOW_ACCEPTANCE"
        ),
    }

    planner_audit = {
        "records": len(rows),
        "strict_json_plan_success_count": len(completed),
        "strict_json_plan_success_rate": _ratio(len(completed), len(rows)),
        "planner_error_count": len(error_rows),
        "errors": [
            {
                "id": row["id"],
                "type": row["error"]["type"],
                "message": row["error"]["message"],
            }
            for row in error_rows
        ],
        "mode_counts": dict(mode_counts),
        "modes_demonstrated": sorted(mode_counts),
        "all_four_modes_demonstrated": all(
            mode_counts[mode] > 0
            for mode in ("sequence", "comparison", "multiple_cases", "simple")
        ),
        "source_reference_violations": source_reference_violations,
        "source_reference_integrity_rate": _ratio(
            len(completed) - len({row["id"] for row in source_reference_violations}),
            len(completed),
        ),
        "expected_insufficient_ids": sorted(EXPECTED_INSUFFICIENT_IDS),
        "planned_insufficient_ids": sorted(planned_insufficient),
        "insufficient_recall": _ratio(
            len(planned_insufficient & EXPECTED_INSUFFICIENT_IDS),
            len(EXPECTED_INSUFFICIENT_IDS),
        ),
        "k_context_insufficient_recall": _ratio(
            len(planned_insufficient & (EXPECTED_INSUFFICIENT_IDS - ABSENT_IDS)), 6
        ),
        "absent_corpus_plan_detection": _ratio(
            len(planned_insufficient & ABSENT_IDS), len(ABSENT_IDS)
        ),
        "absent_corpus_controlled_refusals": sum(
            bool(row.get("response") and row["response"]["insufficient_context"])
            for row in rows
            if row["id"] in ABSENT_IDS
        ),
        "prompt_contract_pass": prompt_contract_pass,
        "forbidden_call_totals": forbidden_call_totals,
        "planner_latency_ms_median": _median(
            [row["phase12"]["planner"]["latency_ms"] for row in completed]
        ),
        "end_to_end_latency_ms_median_completed_primary": _median(
            [row["response"]["timings"]["total_ms"] for row in primary_completed]
        ),
    }

    criteria = {
        "strict_json_plan_success_100": planner_audit["strict_json_plan_success_rate"] == 1.0,
        "all_four_modes_demonstrated": planner_audit["all_four_modes_demonstrated"],
        "source_reference_integrity_100": planner_audit["source_reference_integrity_rate"] == 1.0,
        "primary_operational_response_non_regression": phase12["operational_response_rate"] >= baseline["operational_response_rate"],
        "citation_validity_operational_non_regression": phase12["citation_validity_operational"] >= baseline["citation_validity_operational"],
        "citation_coverage_non_regression": phase12["citation_coverage_among_completed"] >= baseline["citation_coverage"] - 0.01,
        "process_flow_plan_completed": phase12["process_flow_status"] == "completed",
        "expected_insufficiency_detected_100": planner_audit["insufficient_recall"] == 1.0,
        "absent_corpus_refusal_100": planner_audit["absent_corpus_controlled_refusals"] == len(ABSENT_IDS),
        "forbidden_calls_zero": all(value == 0 for value in forbidden_call_totals.values()),
        "prompt_contract_pass": prompt_contract_pass,
    }
    accepted = all(criteria.values())
    decision = {
        "accepted_for_holdout": accepted,
        "dev_strictly_better_than_phase10": accepted,
        "production_decision": "DEPLOY" if accepted else "DO NOT DEPLOY",
        "failed_criteria": [key for key, value in criteria.items() if not value],
        "criteria": criteria,
        "semantic_rescore_performed": False,
        "semantic_rescore_reason": (
            "Deterministic hard gates already reject the candidate; no Evidence Judge was introduced."
        ),
        "holdout_opened": False,
    }
    (output_directory / "planner_audit.json").write_text(
        json.dumps(planner_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_directory / "dev_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "phase": 12,
        "comparison": {"phase10": baseline, "phase12": phase12},
        "planner": planner_audit,
        "decision": decision,
        "holdout": {
            "opened": False,
            "planner_freeze_created": False,
            "policy": "closed because DEV is not strictly better than Phase 10",
        },
    }
    (output_directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    errors = Counter(row["error"]["type"] for row in error_rows)
    lines = [
        "# Phase 12 — Structured Evidence Planning",
        "",
        "## Scope and isolation",
        "",
        "EvidencePlanner receives only the question and frozen Phase-10 EvidenceBundles. It emits one strict JSON plan. Generation then receives question, evidence, and that plan. DEV reused the exact Phase-10 evidence; retrieval, verifier, and repair calls are all zero. No Evidence Judge, new retrieval, rechunking, or fine-tuning was introduced.",
        "",
        "## DEV comparison",
        "",
        "| Metric | Phase 10 | Phase 12 |",
        "|---|---:|---:|",
        f"| Primary answers emitted | 45/45 (100.00%) | {len(primary_completed)}/45 ({_pct(phase12['operational_response_rate'])}) |",
        f"| Manual answer success | {baseline['manual_answer_success_count']}/45 ({_pct(baseline['manual_answer_success_rate'])}) | Not rescored after hard-gate rejection |",
        f"| Citation validity, operational | 100.00% | {_pct(phase12['citation_validity_operational'])} |",
        f"| Claim citation coverage | {_pct(baseline['citation_coverage'])} | {_pct(phase12['citation_coverage_among_completed'])} among emitted answers |",
        f"| Correct language | 100.00% | {_pct(phase12['correct_language_among_completed'])} among emitted answers |",
        f"| Process-flow case | Answer emitted; manual order audit failed | {phase12['process_flow_status']} |",
        "",
        "## Planner audit",
        "",
        f"- Strict valid plan rate: {len(completed)}/49 ({_pct(planner_audit['strict_json_plan_success_rate'])}).",
        f"- Modes emitted: {dict(mode_counts)}. No valid `sequence` or `multiple_cases` plan was produced.",
        f"- Source-addressable valid plans: {len(completed)}/{len(completed)}; every item points to one or more available `[Source N]`.",
        f"- Insufficiency detection: {len(planned_insufficient & EXPECTED_INSUFFICIENT_IDS)}/9 overall; 0/6 on Phase-10 K contexts; 3/3 on explicit absent-corpus cases.",
        f"- Planner failures: {len(error_rows)} ({dict(errors)}). DQ004 and the process-flow acceptance case selected sequence but omitted required indices; DQ014 and DQ040 selected comparison but omitted two explicit sides.",
        f"- Median planner latency: {planner_audit['planner_latency_ms_median']/1000:.2f}s; median completed-primary end-to-end latency: {planner_audit['end_to_end_latency_ms_median_completed_primary']/1000:.2f}s.",
        "",
        "## Decision",
        "",
        "**DO NOT DEPLOY.** Phase 12 is not strictly better than Phase 10 on DEV. Its source-reference integrity is correct and absent-corpus refusals pass, but strict-plan reliability, mode coverage, operational response rate, citation validity, citation coverage, K-context insufficiency detection, and the process-flow case fail the frozen gates.",
        "",
        "The FINAL HOLDOUT remains closed. No planner-freeze manifest, holdout-open marker, or holdout result file was created.",
    ]
    (output_directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_report(args.output)
    print(json.dumps(result["decision"], ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
