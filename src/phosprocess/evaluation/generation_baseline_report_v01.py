# ruff: noqa: E501
"""Compile the final 19-section Phase-10 baseline report."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from phosprocess.evaluation.context_engine_v01 import read_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "data/evaluation/generation_baseline/v0.1"


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    manifest = _load_json(output_directory / "baseline_manifest.json")
    holdout = _load_json(output_directory / "holdout_open_manifest.json")
    objective = read_jsonl(output_directory / "objective_checks.jsonl")
    claims = read_jsonl(output_directory / "claim_annotations.jsonl")
    questions = read_jsonl(output_directory / "question_annotations.jsonl")
    decision = _load_json(output_directory / "architectural_decision.json")
    numeric_audit = _load_json(output_directory / "numeric_audit.json")
    process_audit = _load_json(output_directory / "process_flow_audit.json")
    followup_audit = _load_json(output_directory / "followup_audit.json")
    test_gates = _load_json(output_directory / "test_gates.json")

    if holdout.get("opening_count") != 1 or holdout.get("status") != "complete":
        raise RuntimeError("FINAL HOLDOUT was not completed exactly once")
    if not claims or any(item["support_label"] is None for item in claims):
        raise RuntimeError("manual claim annotations are incomplete")
    if not questions or any(item["manual"]["answer_success"] is None for item in questions):
        raise RuntimeError("manual question annotations are incomplete")
    production_drift = {
        relative: {"expected": expected, "actual": _sha256(PROJECT_ROOT / relative)}
        for relative, expected in manifest["production_file_hashes"].items()
        if _sha256(PROJECT_ROOT / relative) != expected
    }
    if production_drift:
        raise RuntimeError(f"production drift after baseline: {sorted(production_drift)}")

    primary_questions = [
        item for item in questions
        if item["record_type"] == "primary" and item["source_question_id"].startswith(("CE", "DQ"))
    ]
    primary_ids = {item["record_id"] for item in primary_questions}
    primary_objective = [item for item in objective if item["id"] in primary_ids]
    primary_claims = [item for item in claims if item["record_id"] in primary_ids and item["important"]]
    cited_claims = [item for item in primary_claims if item["citation_numbers"]]
    support_counts = Counter(item["support_label"] for item in primary_claims)
    completeness = Counter(item["manual"]["completeness"] for item in primary_questions)
    primary_failures = Counter(
        code for item in primary_questions for code in item["manual"]["failure_codes"]
    )
    failures = Counter(
        code for item in questions for code in item["manual"]["failure_codes"]
    )
    available = [item for item in primary_questions if item["manual"]["evidence_available"] == "YES"]
    utilization = Counter(item["manual"]["evidence_used"] for item in available)
    absent = [item for item in questions if item["dataset_scope"] == "absent_corpus"]
    followups = [item for item in questions if item["record_type"] == "followup"]

    language: dict[str, Any] = {}
    for code in ("fr", "en", "ar"):
        q_items = [item for item in primary_questions if item["language"] == code]
        q_ids = {item["record_id"] for item in q_items}
        c_items = [item for item in primary_claims if item["record_id"] in q_ids]
        o_items = [item for item in primary_objective if item["id"] in q_ids]
        language[code] = {
            "questions": len(q_items),
            "answer_success_rate": _ratio(sum(item["manual"]["answer_success"] for item in q_items), len(q_items)),
            "correct_language_rate": _ratio(sum(item["language_correct"] for item in o_items), len(o_items)),
            "supported_claim_rate": _ratio(sum(item["support_label"] == "SUPPORTED" for item in c_items), len(c_items)),
            "citation_validity": _ratio(sum(item["citation_syntax_valid"] and item["citation_existing_bundle_valid"] and item["citation_locked_document_valid"] and item["citation_metadata_valid"] for item in o_items), len(o_items)),
        }

    split_metrics: dict[str, Any] = {}
    for split in ("dev", "final_holdout"):
        q_items = [item for item in primary_questions if item["split"] == split]
        q_ids = {item["record_id"] for item in q_items}
        c_items = [item for item in primary_claims if item["record_id"] in q_ids]
        o_items = [item for item in primary_objective if item["id"] in q_ids]
        split_metrics[split] = {
            "questions": len(q_items),
            "answer_success_rate": _ratio(
                sum(item["manual"]["answer_success"] for item in q_items),
                len(q_items),
            ),
            "evidence_availability_rate": _ratio(
                sum(item["manual"]["evidence_available"] == "YES" for item in q_items),
                len(q_items),
            ),
            "supported_claim_rate": _ratio(
                sum(item["support_label"] == "SUPPORTED" for item in c_items),
                len(c_items),
            ),
            "citation_validity": _ratio(
                sum(
                    item["citation_syntax_valid"]
                    and item["citation_existing_bundle_valid"]
                    and item["citation_locked_document_valid"]
                    and item["citation_metadata_valid"]
                    for item in o_items
                ),
                len(o_items),
            ),
        }

    all_numeric = [check for item in primary_objective for check in item["numeric_checks"]]
    all_units = [check for check in all_numeric if check["unit"] is not None]
    metrics = {
        "answerability": {
            "answerable_question_success_rate": _ratio(sum(item["manual"]["answer_success"] for item in primary_questions), len(primary_questions)),
            "correct_refusal_rate": _ratio(sum(item["manual"]["refusal_class"] == "I" for item in absent), len(absent)),
            "incorrect_answer_rate_absent": _ratio(sum(item["manual"]["refusal_class"] == "K" for item in absent), len(absent)),
        },
        "evidence": {
            "available_questions": len(available),
            "availability_rate": _ratio(len(available), len(primary_questions)),
            "utilization_yes_rate": _ratio(utilization["YES"], len(available)),
            "utilization_partial_rate": _ratio(utilization["PARTIAL"], len(available)),
            "utilization_no_rate": _ratio(utilization["NO"], len(available)),
            "exact_gold_context_rate": _ratio(sum(item["evidence_available_in_context"] for item in primary_objective), len(primary_objective)),
            "context_packing_misses": sum(item["context_packing_miss"] for item in primary_objective),
        },
        "claims": {
            "count": len(primary_claims),
            "supported_rate": _ratio(support_counts["SUPPORTED"], len(primary_claims)),
            "partially_supported_rate": _ratio(support_counts["PARTIALLY_SUPPORTED"], len(primary_claims)),
            "unsupported_rate": _ratio(support_counts["UNSUPPORTED"], len(primary_claims)),
        },
        "citations": {
            "validity": _ratio(sum(item["citation_syntax_valid"] and item["citation_existing_bundle_valid"] and item["citation_locked_document_valid"] and item["citation_metadata_valid"] for item in primary_objective), len(primary_objective)),
            "precision_strict": _ratio(sum(item["citation_support_label"] == "SUPPORTED" for item in cited_claims), len(cited_claims)),
            "precision_supported_or_partial": _ratio(sum(item["citation_support_label"] != "UNSUPPORTED" for item in cited_claims), len(cited_claims)),
            "coverage": _ratio(len(cited_claims), len(primary_claims)),
            "repair_count": sum(item["repair_attempted"] for item in primary_objective),
        },
        "completeness": dict(completeness),
        "numeric": {
            "objective_number_in_cited_evidence": _ratio(sum(item["number_in_cited_evidence"] for item in all_numeric), len(all_numeric)),
            "objective_unit_in_cited_evidence": _ratio(sum(item["unit_in_cited_evidence"] for item in all_units), len(all_units)),
            **numeric_audit["metrics"],
        },
        "language": language,
        "by_split": split_metrics,
        "failures": dict(failures),
        "primary_failures": dict(primary_failures),
        "system": {
            "latency_ms_mean": statistics.mean(item["latency_ms"] for item in primary_objective),
            "latency_ms_median": statistics.median(item["latency_ms"] for item in primary_objective),
            "first_token_ms_median": statistics.median(item["first_token_ms"] for item in primary_objective if item["first_token_ms"] is not None),
            "generated_tokens_median": statistics.median(item["generated_tokens"] for item in primary_objective if item["generated_tokens"] is not None),
            "input_tokens_median": statistics.median(item["estimated_input_tokens"] for item in primary_objective if item["estimated_input_tokens"] is not None),
            "context_tokens_median": statistics.median(item["context_tokens"] for item in primary_objective if item["context_tokens"] is not None),
        },
    }

    summary = {
        "phase": 10,
        "baseline_sha256": manifest["baseline_sha256"],
        "holdout_opening_count": holdout["opening_count"],
        "production_changed": False,
        "semantic_judge_used": False,
        "primary_questions": len(primary_questions),
        "dev_primary": sum(item["split"] == "dev" for item in primary_questions),
        "holdout_primary": sum(item["split"] == "final_holdout" for item in primary_questions),
        "absent_questions": len(absent),
        "followup_turns": len(followups),
        "metrics": metrics,
        "process_flow": process_audit,
        "followups": followup_audit,
        "architectural_conclusion": decision,
        "test_gates": test_gates,
        "production_drift": production_drift,
    }
    (output_directory / "citation_metrics.json").write_text(json.dumps(metrics["citations"], indent=2) + "\n", encoding="utf-8")
    (output_directory / "failure_taxonomy.json").write_text(json.dumps({"counts": dict(failures), "primary_counts": dict(primary_failures), "questions": [{"id": item["record_id"], "codes": item["manual"]["failure_codes"]} for item in questions if item["manual"]["failure_codes"]]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Phase 10 — Real end-to-end generation baseline",
        "",
        "## 1. BASELINE CONFIGURATION",
        "",
        f"Active KB `{manifest['active_kb'].get('version', manifest['active_kb'])}`; Qwen `{manifest['qwen']['model']}`, temperature `{manifest['qwen']['temperature']}`, seed `{manifest['qwen']['seed']}`, context `{manifest['qwen']['context_size']}`, max output `{manifest['qwen']['max_output_tokens']}`, thinking disabled. Retriever `{manifest['retriever']['selected_variant']}`. Baseline SHA-256 `{manifest['baseline_sha256']}`. Production drift after run: none.",
        "",
        "## 2. DATASET",
        "",
        f"64 audited Phase-8 questions: 45 DEV and 19 FINAL HOLDOUT; plus {len(absent)} absent-corpus questions, one process-flow acceptance case, and {len(followups)} generated conversational follow-up turns. FINAL HOLDOUT opening count: {holdout['opening_count']}.",
        "",
        "## 3. END-TO-END RESULTS",
        "",
        f"Answerable-question success: {_pct(metrics['answerability']['answerable_question_success_rate'])}. Every saved record traversed production discovery, source lock, retrieval, reranking, Context Engine, production prompt, and real Qwen streaming generation.",
        "",
        "| Split | N | Answer success | Evidence available | Supported claims | Citation validity |",
        "|---|---:|---:|---:|---:|---:|",
        *[f"| {split} | {data['questions']} | {_pct(data['answer_success_rate'])} | {_pct(data['evidence_availability_rate'])} | {_pct(data['supported_claim_rate'])} | {_pct(data['citation_validity'])} |" for split, data in split_metrics.items()],
        "",
        "## 4. EVIDENCE AVAILABILITY",
        "",
        f"Manual documentary availability: {metrics['evidence']['available_questions']}/64 ({_pct(metrics['evidence']['availability_rate'])}). Exact frozen evidence IDs appeared in final context for {_pct(metrics['evidence']['exact_gold_context_rate'])}; {metrics['evidence']['context_packing_misses']} cases had an exact valid route in retrieval candidates but not in final context.",
        "",
        "## 5. EVIDENCE UTILIZATION",
        "",
        f"Among questions with sufficient evidence: YES {_pct(metrics['evidence']['utilization_yes_rate'])}, PARTIAL {_pct(metrics['evidence']['utilization_partial_rate'])}, NO {_pct(metrics['evidence']['utilization_no_rate'])}. Strict evidence utilization rate is the YES rate.",
        "",
        "## 6. CLAIM SUPPORT",
        "",
        f"{metrics['claims']['count']} important atomic claims manually compared with actual cited bundles/gold: SUPPORTED {_pct(metrics['claims']['supported_rate'])}, PARTIALLY SUPPORTED {_pct(metrics['claims']['partially_supported_rate'])}, UNSUPPORTED {_pct(metrics['claims']['unsupported_rate'])}.",
        "",
        "## 7. CITATIONS",
        "",
        f"Validity {_pct(metrics['citations']['validity'])}; strict precision {_pct(metrics['citations']['precision_strict'])}; supported-or-partial precision {_pct(metrics['citations']['precision_supported_or_partial'])}; claim coverage {_pct(metrics['citations']['coverage'])}; production citation repairs {metrics['citations']['repair_count']}.",
        "",
        "## 8. COMPLETENESS",
        "",
        ", ".join(f"{key}: {completeness[key]}" for key in ("COMPLETE", "MOSTLY_COMPLETE", "PARTIAL", "MISSED")) + ".",
        "",
        "## 9. NUMBERS / UNITS",
        "",
        f"Objective number-in-cited-evidence {_pct(metrics['numeric']['objective_number_in_cited_evidence'])}; objective unit grounding {_pct(metrics['numeric']['objective_unit_in_cited_evidence'])}. Curated plant numeric accuracy {_pct(metrics['numeric']['plant_numeric_accuracy'])}; curated unit accuracy {_pct(metrics['numeric']['plant_unit_accuracy'])}. See `numeric_audit.json` for each documentary comparison.",
        "",
        "## 10. ABSENT-CORPUS BEHAVIOR",
        "",
        f"Correct refusal rate {_pct(metrics['answerability']['correct_refusal_rate'])}; answered-anyway rate {_pct(metrics['answerability']['incorrect_answer_rate_absent'])}. No threshold or Evidence Judge was added.",
        "",
        "## 11. MULTILINGUAL",
        "",
        "| Language | N | Correct language | Answer success | Supported claims | Citation validity |",
        "|---|---:|---:|---:|---:|---:|",
        *[f"| {code} | {data['questions']} | {_pct(data['correct_language_rate'])} | {_pct(data['answer_success_rate'])} | {_pct(data['supported_claim_rate'])} | {_pct(data['citation_validity'])} |" for code, data in language.items()],
        "",
        "## 12. FOLLOW-UP BEHAVIOR",
        "",
        f"{len(followups)} real follow-up turns were generated with history for query understanding only. Standalone resolution {_pct(followup_audit['metrics']['standalone_resolution_accuracy'])}; source behavior {_pct(followup_audit['metrics']['source_behavior_accuracy'])}; evidence correctness {_pct(followup_audit['metrics']['evidence_correctness'])}; grounded answers {_pct(followup_audit['metrics']['grounded_answer_rate'])}. History was never serialized as documentary evidence.",
        "",
        "## 13. PROCESS-FLOW CASE",
        "",
        process_audit["summary"],
        "",
        "## 14. DQ028 RERANKER DIAGNOSTIC",
        "",
        decision["dq028_conclusion"],
        "",
        "## 15. FAILURE TAXONOMY",
        "",
        ", ".join(f"{code}: {count}" for code, count in sorted(failures.items())) + ".",
        "",
        "## 16. LATENCY",
        "",
        f"Mean end-to-end {metrics['system']['latency_ms_mean']/1000:.2f}s; median {metrics['system']['latency_ms_median']/1000:.2f}s; median TTFT {metrics['system']['first_token_ms_median']/1000:.2f}s; median output {metrics['system']['generated_tokens_median']:.0f} tokens; median estimated input {metrics['system']['input_tokens_median']:.0f}; median documentary context {metrics['system']['context_tokens_median']:.0f}.",
        "",
        "## 17. ARCHITECTURAL CONCLUSION",
        "",
        f"{decision['case']}: {decision['conclusion']}",
        "",
        "## 18. ONE PHASE-11 RECOMMENDATION",
        "",
        decision["phase11_recommendation"],
        "",
        "## 19. TEST GATES",
        "",
        f"Compileall: {test_gates['compileall']}; Ruff: {test_gates['ruff']}; architecture guards: {test_gates['architecture_guards']}; full pytest: {test_gates['pytest']}. Production code/config/index hashes remained identical to the frozen baseline.",
        "",
        "Phase 10 stops here. Phase 11 was not implemented.",
    ]
    (output_directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not args.build:
        parser.error("choose --build")
    print(json.dumps(build_report(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
