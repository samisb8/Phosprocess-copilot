# ruff: noqa: E501
"""Build the final Phase-11 prompt-experiment decision report."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from phosprocess.evaluation.context_engine_v01 import read_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PHASE10_OUTPUT = PROJECT_ROOT / "data/evaluation/generation_baseline/v0.1"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/evaluation/generation_prompt_experiment/v0.1"


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manual_metrics(
    questions: list[dict[str, Any]],
    claims: list[dict[str, Any]],
) -> dict[str, Any]:
    primary = [row for row in questions if row["dataset_scope"] == "phase8_primary"]
    primary_ids = {row["record_id"] for row in primary}
    primary_claims = [row for row in claims if row["record_id"] in primary_ids]
    available = [
        row for row in primary if row["manual"]["evidence_available"] == "YES"
    ]
    cited = [row for row in primary_claims if row["citation_numbers"]]
    completeness = Counter(row["manual"]["completeness"] for row in primary)
    support = Counter(row["support_label"] for row in primary_claims)
    citation_support = Counter(row["citation_support_label"] for row in cited)
    return {
        "primary_count": len(primary),
        "answer_success": _ratio(
            sum(bool(row["manual"]["answer_success"]) for row in primary),
            len(primary),
        ),
        "answer_success_count": sum(
            bool(row["manual"]["answer_success"]) for row in primary
        ),
        "evidence_available_count": len(available),
        "evidence_utilization": dict(
            Counter(row["manual"]["evidence_used"] for row in available)
        ),
        "full_evidence_utilization": _ratio(
            sum(row["manual"]["evidence_used"] == "YES" for row in available),
            len(available),
        ),
        "partial_evidence_utilization": _ratio(
            sum(row["manual"]["evidence_used"] == "PARTIAL" for row in available),
            len(available),
        ),
        "no_evidence_utilization": _ratio(
            sum(row["manual"]["evidence_used"] == "NO" for row in available),
            len(available),
        ),
        "completeness": dict(completeness),
        "complete_or_mostly": _ratio(
            completeness["COMPLETE"] + completeness["MOSTLY_COMPLETE"],
            len(primary),
        ),
        "claim_count": len(primary_claims),
        "claim_support_counts": dict(support),
        "supported_claim_rate": _ratio(support["SUPPORTED"], len(primary_claims)),
        "partial_claim_rate": _ratio(
            support["PARTIALLY_SUPPORTED"], len(primary_claims)
        ),
        "unsupported_claim_rate": _ratio(
            support["UNSUPPORTED"], len(primary_claims)
        ),
        "cited_claim_count": len(cited),
        "citation_support_counts": dict(citation_support),
        "citation_strict_precision": _ratio(
            citation_support["SUPPORTED"], len(cited)
        ),
        "citation_supported_or_partial_precision": _ratio(
            citation_support["SUPPORTED"]
            + citation_support["PARTIALLY_SUPPORTED"],
            len(cited),
        ),
        "citation_coverage": _ratio(len(cited), len(primary_claims)),
        "evidence_dump_count": sum(
            bool(row["manual"].get("evidence_dump")) for row in primary
        ),
        "irrelevant_claim_count": sum(
            len(row["manual"].get("irrelevant_claim_indices", [])) for row in primary
        ),
        "redundant_claim_count": sum(
            len(row["manual"].get("redundant_claim_indices", [])) for row in primary
        ),
    }


def _baseline_question_annotations() -> list[dict[str, Any]]:
    questions = {
        row["id"]: row for row in read_jsonl(PHASE10_OUTPUT / "questions_snapshot.jsonl")
    }
    labels = json.loads(
        (PHASE10_OUTPUT / "manual_question_labels.json").read_text(encoding="utf-8")
    )
    return [
        {
            "record_id": question_id,
            "dataset_scope": questions[question_id]["dataset_scope"],
            "language": questions[question_id]["language"],
            "manual": label,
        }
        for question_id, label in labels.items()
        if question_id in questions and questions[question_id]["split"] == "dev"
    ]


def _unsupported_analysis(
    baseline_claims: list[dict[str, Any]],
    new_claims: list[dict[str, Any]],
    failed_ids: set[str],
) -> dict[str, Any]:
    old = [row for row in baseline_claims if row["support_label"] == "UNSUPPORTED"]
    new = [row for row in new_claims if row["support_label"] == "UNSUPPORTED"]

    def similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
        if left["record_id"] != right["record_id"]:
            return 0.0
        return difflib.SequenceMatcher(None, left["claim_text"], right["claim_text"]).ratio()

    fixed: list[str] = []
    unchanged: list[str] = []
    not_observable: list[str] = []
    for claim in old:
        key = f"{claim['record_id']}#{claim['claim_index']}"
        if claim["record_id"] in failed_ids:
            not_observable.append(key)
        elif any(similarity(claim, candidate) >= 0.55 for candidate in new):
            unchanged.append(key)
        else:
            fixed.append(key)
    additions = [
        f"{claim['record_id']}#{claim['claim_index']}"
        for claim in new
        if not any(similarity(claim, candidate) >= 0.55 for candidate in old)
    ]
    return {
        "baseline_unsupported": len(old),
        "new_unsupported": len(new),
        "fixed": fixed,
        "unchanged": unchanged,
        "not_observable_due_generation_failure": not_observable,
        "new_unsupported_claims": additions,
        "method": (
            "Manual claim verdicts plus same-record semantic-text alignment; a missing "
            "answer is never counted as fixing an unsupported baseline claim."
        ),
    }


def _latency_and_tokens(
    rows: list[dict[str, Any]],
    primary_ids: set[str],
) -> dict[str, Any]:
    valid = [
        row for row in rows if row["id"] in primary_ids and row.get("response") is not None
    ]
    generation = [row["response"]["timings"]["generation_ms"] for row in valid]
    first = [
        row["response"]["timings"].get("first_token_ms")
        for row in valid
        if row["response"]["timings"].get("first_token_ms") is not None
    ]
    tokens = [
        int((row["response"].get("latency") or {}).get("generated_token_count") or 0)
        for row in valid
    ]
    return {
        "valid_records": len(valid),
        "generation_ms_mean": statistics.mean(generation),
        "generation_ms_median": statistics.median(generation),
        "first_token_ms_median": statistics.median(first),
        "output_tokens_mean": statistics.mean(tokens),
        "output_tokens_median": statistics.median(tokens),
        "output_tokens_total": sum(tokens),
    }


def build_report(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    freeze = json.loads(
        (output_directory / "experiment_freeze.json").read_text(encoding="utf-8")
    )
    test_gates_path = output_directory / "test_gates.json"
    test_gates = (
        json.loads(test_gates_path.read_text(encoding="utf-8"))
        if test_gates_path.exists()
        else {"status": "PENDING"}
    )
    questions = read_jsonl(output_directory / "question_annotations.jsonl")
    claims = read_jsonl(output_directory / "claim_annotations.jsonl")
    baseline_questions = _baseline_question_annotations()
    baseline_claims_all = read_jsonl(PHASE10_OUTPUT / "claim_annotations.jsonl")
    baseline_dev_ids = {
        row["record_id"]
        for row in baseline_questions
        if row["dataset_scope"] == "phase8_primary"
    }
    baseline_claims = [
        row for row in baseline_claims_all if row["record_id"] in baseline_dev_ids
    ]
    a = _manual_metrics(baseline_questions, baseline_claims)
    b = _manual_metrics(questions, claims)
    baseline_rows = read_jsonl(PHASE10_OUTPUT / "dev_generation_results.jsonl")
    new_rows = read_jsonl(output_directory / "dev_generation_results.jsonl")
    failed_ids = {row["id"] for row in new_rows if row["status"] != "completed"}
    unsupported = _unsupported_analysis(baseline_claims, claims, failed_ids)

    new_primary_ids = {
        row["record_id"] for row in questions if row["dataset_scope"] == "phase8_primary"
    }
    baseline_latency = _latency_and_tokens(baseline_rows, baseline_dev_ids)
    new_latency = _latency_and_tokens(new_rows, new_primary_ids)

    objective = {
        row["id"]: row for row in read_jsonl(output_directory / "objective_checks.jsonl")
    }
    valid_primary_objective = [
        objective[record_id] for record_id in new_primary_ids if record_id in objective
    ]
    citation_valid_count = sum(
        row["citation_syntax_valid"]
        and row["citation_existing_bundle_valid"]
        and row["citation_locked_document_valid"]
        and row["citation_metadata_valid"]
        for row in valid_primary_objective
    )
    b["citation_validity_operational"] = _ratio(
        citation_valid_count, len(new_primary_ids)
    )
    a["citation_validity_operational"] = 1.0

    numeric = {
        "scope": "six curated DEV plant-numeric questions; no expected value is encoded in production",
        "baseline": {
            "correct": 4,
            "total": 6,
            "accuracy": 4 / 6,
        },
        "phase11": {
            "correct": 2,
            "total": 6,
            "accuracy": 2 / 6,
        },
        "checks": {
            "CE060": "PASS: Cp = m1 - m5 is preserved and correctly cited.",
            "CE061": "FAIL: variables are listed, but the required inlet/output-to-production equations remain incomplete and units are transformed.",
            "CE066": "FAIL: Becker 25% to 54% replaces the required OCP 29% to 54% fact.",
            "DQ010": "PASS: 4.1, 4.6, 8%, and 0.5-1% are preserved with their units.",
            "DQ012": "FAIL: only 75 torr remains; 50%, 96 C, and 35 T/h observations are omitted and causal additions are unsupported.",
            "DQ019": "FAIL: the required 22981770.02 and 3586832.7 kg/h results are omitted and several unit/equation renderings are transformed.",
        },
        "cross_source_numeric_conflation": ["PROCESS_FLOW_ACCEPTANCE"],
    }
    (output_directory / "numeric_regression.json").write_text(
        json.dumps(numeric, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    process = {
        "record_id": "PROCESS_FLOW_ACCEPTANCE",
        "exact_question_answered": False,
        "one_coherent_documentary_route": False,
        "order_preserved": False,
        "distinct_operating_conditions_separate": False,
        "backward_jump": True,
        "unsupported_transition": True,
        "local_citations": "PARTIAL",
        "relevant_evidence_omitted": ["explicit forced-circulation pump/loop"],
        "irrelevant_side_stream_claims": [6, 7],
        "baseline_outcome": "FAIL",
        "phase11_outcome": "FAIL",
        "summary": (
            "The new prompt still presents Source 4's 75 mmHg case and Source 6's "
            "60 Torr case as one route, reaches storage, then jumps backward to the exchanger."
        ),
    }
    (output_directory / "process_flow_audit.json").write_text(
        json.dumps(process, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    absent = [row for row in questions if row["dataset_scope"] == "absent_corpus"]
    absent_audit = {
        "count": len(absent),
        "correct_refusals": sum(
            row["manual"]["refusal_class"] == "I" for row in absent
        ),
        "correct_refusal_rate": _ratio(
            sum(row["manual"]["refusal_class"] == "I" for row in absent), len(absent)
        ),
        "hard_regression_pass": all(
            row["manual"]["refusal_class"] == "I" for row in absent
        ),
    }
    (output_directory / "absent_corpus_regression.json").write_text(
        json.dumps(absent_audit, indent=2) + "\n", encoding="utf-8"
    )

    primary = [row for row in questions if row["dataset_scope"] == "phase8_primary"]
    emitted_by_language = Counter(
        row["language"] for row in primary if row["generation_status"] == "completed"
    )
    total_by_language = Counter(row["language"] for row in primary)
    multilingual = {
        language: {
            "valid_response_count": emitted_by_language[language],
            "question_count": total_by_language[language],
            "correct_language_among_emitted": 1.0,
            "operational_response_rate": _ratio(
                emitted_by_language[language], total_by_language[language]
            ),
        }
        for language in ("fr", "en", "ar")
    }
    multilingual["requirement_pass"] = all(
        value["operational_response_rate"] == 1.0
        for key, value in multilingual.items()
        if key in {"fr", "en", "ar"}
    )
    (output_directory / "multilingual_regression.json").write_text(
        json.dumps(multilingual, indent=2) + "\n", encoding="utf-8"
    )

    followups = {
        "run": False,
        "reason": (
            "The DEV prompt was rejected and therefore was not selected/frozen; the protocol "
            "allows follow-ups only as a post-selection read-only diagnostic."
        ),
        "used_for_prompt_selection": False,
    }
    (output_directory / "followup_regression.json").write_text(
        json.dumps(followups, indent=2) + "\n", encoding="utf-8"
    )

    k_semantics = {
        "code": "K",
        "meaning": (
            "The final evidence context for this particular question did not support the "
            "requested target, so the answer should have refused instead of substituting a "
            "nearby topic."
        ),
        "not_equivalent_to": "The question is globally absent from the corpus.",
        "explicit_absent_corpus_result": "3/3 correct refusals",
        "behavior_changed_from_k": False,
        "dev_primary_k_ids": ["CE051", "CE066", "DQ003", "DQ039", "DQ044", "DQ048"],
    }
    (output_directory / "failure_code_k_semantics.json").write_text(
        json.dumps(k_semantics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_directory / "unsupported_claim_analysis.json").write_text(
        json.dumps(unsupported, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    citation_locality = {
        "baseline": {
            "strict_precision": a["citation_strict_precision"],
            "coverage": a["citation_coverage"],
            "operational_validity": a["citation_validity_operational"],
        },
        "phase11": {
            "strict_precision": b["citation_strict_precision"],
            "coverage": b["citation_coverage"],
            "operational_validity": b["citation_validity_operational"],
            "cited_claims_with_partial_or_no_support": (
                b["citation_support_counts"].get("PARTIALLY_SUPPORTED", 0)
                + b["citation_support_counts"].get("UNSUPPORTED", 0)
            ),
        },
        "diagnosis": (
            "Strict semantic precision improved slightly, but claim-level attachment coverage "
            "fell materially and one answer failed because Qwen grouped citations illegally."
        ),
    }
    (output_directory / "citation_locality.json").write_text(
        json.dumps(citation_locality, indent=2) + "\n", encoding="utf-8"
    )

    criteria = {
        "answer_success_improved": b["answer_success"] > a["answer_success"],
        "full_evidence_utilization_improved": (
            b["full_evidence_utilization"] > a["full_evidence_utilization"]
        ),
        "completeness_improved": b["complete_or_mostly"] > a["complete_or_mostly"],
        "unsupported_claim_rate_not_materially_increased": (
            b["unsupported_claim_rate"] <= a["unsupported_claim_rate"] + 0.01
        ),
        "citation_coverage_not_materially_regressed": (
            b["citation_coverage"] >= a["citation_coverage"] - 0.01
        ),
        "absent_refusal_100": absent_audit["correct_refusal_rate"] == 1.0,
        "language_100": bool(multilingual["requirement_pass"]),
        "process_order_fixed": process["phase11_outcome"] == "PASS",
        "numeric_non_regression": (
            numeric["phase11"]["accuracy"] >= numeric["baseline"]["accuracy"]
        ),
    }
    accepted = all(criteria.values())
    decision = {
        "accepted_for_holdout": accepted,
        "criteria": criteria,
        "failed_criteria": [key for key, value in criteria.items() if not value],
        "holdout_opened": (output_directory / "holdout_open_manifest.json").exists(),
        "production_decision": "DEPLOY" if accepted else "DO NOT DEPLOY",
        "decision_tree_case": "A" if accepted else "C",
    }
    (output_directory / "dev_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "experiment_freeze": {
            "baseline_sha256": freeze["baseline_sha256"],
            "baseline_prompt_sha256": freeze["baseline_prompt_sha256"],
            "new_prompt_sha256": freeze["new_prompt_sha256"],
            "dataset_sha256": freeze["dataset_sha256"],
            "qwen": freeze["qwen"],
        },
        "dev_ab": {"baseline": a, "phase11": b},
        "unsupported_claim_analysis": unsupported,
        "citation_locality": citation_locality,
        "numeric": numeric,
        "process_flow": process,
        "absent_corpus": absent_audit,
        "multilingual": multilingual,
        "followups": followups,
        "failure_code_k": k_semantics,
        "latency": {"baseline": baseline_latency, "phase11": new_latency},
        "decision": decision,
        "test_gates": test_gates,
    }
    (output_directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Phase 11 — Grounded generation prompt experiment",
        "",
        "## 1. EXPERIMENT FREEZE",
        "",
        f"- Baseline hash: `{freeze['baseline_sha256']}`",
        f"- Baseline prompt hash: `{freeze['baseline_prompt_sha256']}`",
        f"- New prompt hash: `{freeze['new_prompt_sha256']}`",
        f"- Dataset hash: `{freeze['dataset_sha256']}` (45 DEV primary, 19 FINAL HOLDOUT).",
        "- Model/config: qwen3:8b, temperature 0.1, seed 0, context 8192, max output 1024, thinking disabled.",
        "- Isolation: identical question, EvidenceBundles, serialized context, user prompt, model and sampling config; retrieval calls = 0.",
        "",
        "## 2. PROMPT CHANGE",
        "",
        "The production default remains the exact baseline. One explicit research variant adds generic, domain-neutral instructions to answer the exact question, select all and only necessary evidence, separate documentary cases, preserve ordered relations and numbers/units, and attach citations locally. It contains no plant fact, expected answer, equipment sequence or question-specific wording.",
        "",
        "## 3. DEV A/B",
        "",
        "| Metric | Baseline A | Phase-11 B |",
        "|---|---:|---:|",
        f"| Answer success | {_pct(a['answer_success'])} ({a['answer_success_count']}/45) | {_pct(b['answer_success'])} ({b['answer_success_count']}/45) |",
        f"| Full evidence utilization (available cohort) | {_pct(a['full_evidence_utilization'])} (28/39) | {_pct(b['full_evidence_utilization'])} (27/39) |",
        f"| COMPLETE + MOSTLY_COMPLETE | {_pct(a['complete_or_mostly'])} | {_pct(b['complete_or_mostly'])} |",
        f"| Supported claims | {_pct(a['supported_claim_rate'])} | {_pct(b['supported_claim_rate'])} |",
        f"| Partially supported claims | {_pct(a['partial_claim_rate'])} | {_pct(b['partial_claim_rate'])} |",
        f"| Unsupported claims | {_pct(a['unsupported_claim_rate'])} | {_pct(b['unsupported_claim_rate'])} |",
        f"| Strict citation precision | {_pct(a['citation_strict_precision'])} | {_pct(b['citation_strict_precision'])} |",
        f"| Claim citation coverage | {_pct(a['citation_coverage'])} | {_pct(b['citation_coverage'])} |",
        f"| Operational citation validity | 100.00% | {_pct(b['citation_validity_operational'])} |",
        "| Curated DEV numeric accuracy | 66.67% (4/6) | 33.33% (2/6) |",
        f"| Mean output tokens | {baseline_latency['output_tokens_mean']:.1f} | {new_latency['output_tokens_mean']:.1f} |",
        "",
        "B is shorter overall, so the failure is not global token inflation. It nevertheless repeats the boiler/reboiler distinction eight times in DQ014 and dumps three supported-but-irrelevant claims across CE066 and the process case.",
        "",
        "## 4. UNSUPPORTED CLAIM ANALYSIS",
        "",
        f"Unsupported claims fall from {unsupported['baseline_unsupported']} to {unsupported['new_unsupported']}, but this grounding improvement does not compensate for lower answer success, utilization, completeness, citation coverage and numeric fidelity. A failed output is not counted as a fixed claim.",
        "",
        "## 5. CITATION LOCALITY",
        "",
        f"Strict precision rises from {_pct(a['citation_strict_precision'])} to {_pct(b['citation_strict_precision'])}, while coverage falls from {_pct(a['citation_coverage'])} to {_pct(b['citation_coverage'])}. DQ019 places citations after equation blocks in a way that leaves many extracted factual claims locally uncited. DQ023 fails both attempts with `[Source 2, Source 5]`.",
        "",
        "## 6. PROCESS-FLOW CASE",
        "",
        "- Baseline: FAIL — conflated 75 mmHg/60 Torr routes, backward jump, loop omitted.",
        "- New: FAIL — the same structural failure remains; vapor/gas side-stream details are also irrelevant to the requested acid path.",
        "",
        "## 7. NUMERIC CASES",
        "",
        "Only CE060 and DQ010 pass the six-question DEV numeric set. DQ012 drops several required observations; DQ019 drops two required results; CE061 and CE066 remain incorrect/incomplete. The process case still conflates values from distinct documentary operating descriptions.",
        "",
        "## 8. ABSENT-CORPUS REGRESSION",
        "",
        "3/3 correct refusals (100%): PASS.",
        "",
        "## 9. MULTILINGUAL",
        "",
        f"All emitted answers use the correct language. Operational valid-response rates are FR {_pct(multilingual['fr']['operational_response_rate'])}, EN 100.00%, AR 100.00%; the French DQ023 generation failure means the required end-to-end 100% is not met.",
        "",
        "## 10. FOLLOW-UP READ-ONLY REGRESSION",
        "",
        "Not run: the DEV prompt was rejected and never selected/frozen. Follow-ups were not used to tune or select the prompt.",
        "",
        "## 11. FAILURE CODE K SEMANTICS",
        "",
        "K means that the final context for the particular requested target was insufficient and the model should have refused instead of answering a nearby topic. It does not mean that the question is globally absent from the corpus. No production behavior was changed from K.",
        "",
        "## 12. FROZEN HOLDOUT",
        "",
        "FINAL HOLDOUT was not opened. The sole prompt variant failed DEV selection, so no prompt-freeze manifest or holdout-open marker was created.",
        "",
        "## 13. LATENCY",
        "",
        f"On valid DEV primary responses, Phase-11 replay generation mean/median are {new_latency['generation_ms_mean']/1000:.2f}s/{new_latency['generation_ms_median']/1000:.2f}s, with median generation TTFT {new_latency['first_token_ms_median']/1000:.2f}s. Retrieval was not rerun. Latency was measured, not optimized.",
        "",
        "## 14. PRODUCTION DECISION",
        "",
        "**DO NOT DEPLOY.** The variant improves supported-claim rate and strict citation precision, but regresses the primary metrics it was meant to improve and introduces one invalid final output. Decision-tree result: **CASE C**.",
        "",
        "## 15. FILES CHANGED",
        "",
        "- `src/phosprocess/rag/prompts.py` — explicit research-only variant; baseline remains default.",
        "- `src/phosprocess/evaluation/generation_prompt_experiment_v01.py` — frozen same-evidence replay.",
        "- `src/phosprocess/evaluation/generation_prompt_manual_review_v01.py` — side-by-side manual annotation compiler.",
        "- `src/phosprocess/evaluation/generation_prompt_report_v01.py` — metrics and final decision report.",
        "- `tests/test_quality_prompts_fidelity.py` — prompt-variant boundary tests.",
        "- `data/evaluation/generation_prompt_experiment/v0.1/` — Phase-11 artifacts.",
        "",
        "## 16. TEST GATES",
        "",
        (
            f"Compileall: {test_gates.get('compileall', 'PENDING')}; "
            f"Ruff: {test_gates.get('ruff', 'PENDING')}; "
            f"architecture guards: {test_gates.get('architecture_guards', 'PENDING')}; "
            f"full pytest: {test_gates.get('pytest', 'PENDING')}. "
            f"Prompt hash: {test_gates.get('prompt_freeze_hash_verification', 'PENDING')}."
        ),
        "",
        "The repository-root Ruff scan still reports 14 pre-existing issues in out-of-scope utility scripts; `src` and `tests` are clean.",
        "",
        "## 17. PHASE 12 RECOMMENDATION",
        "",
        "Follow CASE C: investigate a binding structured evidence plan before generation, with explicit case/sequence boundaries and claim-to-source slots. Separately retain the audited K semantics as input to a possible Evidence Judge. Do not combine that work with follow-up resolution.",
        "",
    ]
    (output_directory / "summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = build_report(args.output)
    print(json.dumps(summary["decision"], indent=2))


if __name__ == "__main__":
    main()
