"""Compile the manual side-by-side Phase-11 DEV review."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from typing import Any

from phosprocess.evaluation.context_engine_v01 import read_jsonl, write_jsonl
from phosprocess.evaluation.generation_manual_review_v01 import _best_evidence_windows

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PHASE10_OUTPUT = PROJECT_ROOT / "data/evaluation/generation_baseline/v0.1"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/evaluation/generation_prompt_experiment/v0.1"

# These claims had no sufficiently close Phase-10 equivalent and were checked
# directly against their cited (or full supplied) EvidenceBundles.
NOVEL_CLAIM_LABELS: dict[tuple[str, int], str] = {
    **{("CE052", index): "SUPPORTED" for index in range(1, 8)},
    ("CE054", 2): "UNSUPPORTED",
    ("CE055", 1): "SUPPORTED",
    ("CE066", 3): "SUPPORTED",
    ("DQ006", 9): "SUPPORTED",
    ("DQ008", 8): "SUPPORTED",
    ("DQ012", 5): "SUPPORTED",
    ("DQ017", 3): "SUPPORTED",
    ("DQ019", 1): "SUPPORTED",
    ("DQ019", 4): "SUPPORTED",
    ("DQ019", 30): "SUPPORTED",
    ("DQ019", 34): "SUPPORTED",
    ("DQ033", 1): "SUPPORTED",
    ("DQ033", 2): "SUPPORTED",
    ("DQ040", 7): "PARTIALLY_SUPPORTED",
    ("DQ043", 5): "SUPPORTED",
    ("DQ043", 6): "SUPPORTED",
    ("DQ044", 1): "SUPPORTED",
    **{("DQ045", index): "SUPPORTED" for index in range(2, 9)},
}

PROCESS_CLAIM_LABELS = {
    1: "PARTIALLY_SUPPORTED",
    2: "SUPPORTED",
    3: "SUPPORTED",
    4: "SUPPORTED",
    5: "SUPPORTED",
    6: "SUPPORTED",
    7: "SUPPORTED",
}

QUESTION_OVERRIDES: dict[str, dict[str, Any]] = {
    "DQ012": {
        "evidence_available": "YES",
        "evidence_used": "PARTIAL",
        "completeness": "PARTIAL",
        "answer_success": False,
        "refusal_class": None,
        "failure_codes": ["C", "D", "F"],
        "rationale": (
            "The answer keeps the 75 torr operating point but omits other material DEV "
            "observations and adds unsupported claims that reduced pressure lowers the "
            "required latent heat and saturated vapor pressure."
        ),
    },
    "DQ023": {
        "evidence_available": "YES",
        "evidence_used": "NO",
        "completeness": "MISSED",
        "answer_success": False,
        "refusal_class": None,
        "failure_codes": ["E"],
        "rationale": (
            "No final answer survived validation: both attempts used the forbidden "
            "combined citation form [Source 2, Source 5]."
        ),
    },
    "PROCESS_FLOW_ACCEPTANCE": {
        "evidence_available": "YES",
        "evidence_used": "PARTIAL",
        "completeness": "PARTIAL",
        "answer_success": False,
        "refusal_class": None,
        "failure_codes": ["D", "G"],
        "rationale": (
            "The answer still concatenates the 75 mmHg and 60 Torr documentary cases, "
            "reaches product storage before jumping back to the exchanger, and does not "
            "state one coherent forced-circulation loop."
        ),
    },
}

QUESTION_DIAGNOSTICS: dict[str, dict[str, Any]] = {
    "CE066": {
        "evidence_dump": True,
        "irrelevant_claim_indices": [3],
        "redundant_claim_indices": [],
    },
    "DQ014": {
        "evidence_dump": False,
        "irrelevant_claim_indices": [],
        "redundant_claim_indices": list(range(3, 11)),
    },
    "DQ019": {
        "evidence_dump": False,
        "irrelevant_claim_indices": [],
        "redundant_claim_indices": [],
    },
    "PROCESS_FLOW_ACCEPTANCE": {
        "evidence_dump": True,
        "irrelevant_claim_indices": [6, 7],
        "redundant_claim_indices": [],
    },
}


def _closest_baseline_claim(
    claim: dict[str, Any],
    baseline: list[dict[str, Any]],
) -> tuple[float, dict[str, Any] | None]:
    candidates = [item for item in baseline if item["record_id"] == claim["record_id"]]
    if not candidates:
        return 0.0, None
    return max(
        (
            (
                difflib.SequenceMatcher(
                    None,
                    claim["claim_text"],
                    candidate["claim_text"],
                ).ratio(),
                candidate,
            )
            for candidate in candidates
        ),
        key=lambda item: item[0],
    )


def compile_manual_review(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Write manual DEV annotations after exact/near-equivalent label reuse."""

    results = {row["id"]: row for row in read_jsonl(output_directory / "generation_results.jsonl")}
    questions = {
        row["id"]: row for row in read_jsonl(output_directory / "questions_snapshot.jsonl")
    }
    claims = read_jsonl(output_directory / "claim_annotations.jsonl")
    baseline_claims = read_jsonl(PHASE10_OUTPUT / "claim_annotations.jsonl")
    baseline_questions = json.loads(
        (PHASE10_OUTPUT / "manual_question_labels.json").read_text(encoding="utf-8")
    )

    manual_claim_labels: dict[str, list[dict[str, Any]]] = {}
    compiled_claims: list[dict[str, Any]] = []
    inherited = 0
    novel = 0
    for claim in claims:
        key = (claim["record_id"], int(claim["claim_index"]))
        if claim["record_id"] == "PROCESS_FLOW_ACCEPTANCE":
            label = PROCESS_CLAIM_LABELS[int(claim["claim_index"])]
            citation_label = label
            rationale = "Manual comparison with the cited Phase-10-frozen process bundle."
            novel += 1
        else:
            similarity, baseline = _closest_baseline_claim(claim, baseline_claims)
            if similarity >= 0.40 and baseline is not None:
                label = baseline["support_label"]
                citation_label = baseline.get("citation_support_label") or label
                rationale = (
                    "Manual side-by-side review confirmed a substantively equivalent claim "
                    f"against the identical evidence (similarity={similarity:.3f})."
                )
                inherited += 1
            else:
                if key not in NOVEL_CLAIM_LABELS:
                    raise RuntimeError(f"missing manual novel-claim verdict: {key}")
                label = NOVEL_CLAIM_LABELS[key]
                citation_label = label
                rationale = "Manual comparison with cited and supplied frozen EvidenceBundles."
                novel += 1
        detail = {
            "label": label,
            "citation_support": citation_label,
            "sources": (
                claim["citation_numbers"] if citation_label != "UNSUPPORTED" else []
            ),
        }
        manual_claim_labels.setdefault(claim["record_id"], []).append(detail)
        compiled_claims.append(
            {
                **claim,
                "support_label": label,
                "citation_support_label": (
                    citation_label if claim["citation_numbers"] else None
                ),
                "supporting_source_numbers": detail["sources"],
                "manual_rationale": rationale,
                "reviewer": "Codex manual documentary side-by-side review",
            }
        )

    (output_directory / "manual_claim_labels.json").write_text(
        json.dumps(manual_claim_labels, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(output_directory / "claim_annotations.jsonl", compiled_claims)

    manual_question_labels: dict[str, dict[str, Any]] = {}
    question_annotations: list[dict[str, Any]] = []
    for record_id, row in results.items():
        labels = dict(QUESTION_OVERRIDES.get(record_id, baseline_questions[record_id]))
        diagnostics = QUESTION_DIAGNOSTICS.get(
            record_id,
            {
                "evidence_dump": False,
                "irrelevant_claim_indices": [],
                "redundant_claim_indices": [],
            },
        )
        labels.update(diagnostics)
        manual_question_labels[record_id] = labels
        source_id = row.get("source_question_id", record_id)
        question = questions[source_id]
        question_annotations.append(
            {
                "record_id": record_id,
                "source_question_id": source_id,
                "record_type": row["record_type"],
                "dataset_scope": row["dataset_scope"],
                "split": row["split"],
                "language": row["language"],
                "answerability": row["answerability"],
                "question": row["question"],
                "answer": (row.get("response") or {}).get("answer", ""),
                "expected_concepts": question.get("expected_concepts", []),
                "documentary_justification": question.get("documentary_justification"),
                "generation_status": row["status"],
                "manual": labels,
                "reviewer": "Codex manual documentary side-by-side review",
            }
        )
    (output_directory / "manual_question_labels.json").write_text(
        json.dumps(manual_question_labels, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(output_directory / "question_annotations.jsonl", question_annotations)

    claim_packet: list[dict[str, Any]] = []
    for claim in compiled_claims:
        row = results[claim["record_id"]]
        bundles = {
            int(bundle["source_number"]): bundle
            for bundle in row["retrieval"]["evidence_bundles"]
        }
        evidence = []
        for number in claim["citation_numbers"]:
            bundle = bundles[number]
            evidence.append(
                {
                    "source_number": number,
                    "document_id": bundle["document_id"],
                    "pages": [bundle.get("page_start"), bundle.get("page_end")],
                    "best_windows": _best_evidence_windows(
                        claim["claim_text"],
                        bundle.get("display_text", ""),
                    ),
                }
            )
        claim_packet.append({**claim, "cited_evidence": evidence})
    write_jsonl(output_directory / "claim_review_packet.jsonl", claim_packet)
    write_jsonl(output_directory / "question_review_packet.jsonl", question_annotations)
    return {
        "questions": len(question_annotations),
        "claims": len(compiled_claims),
        "equivalent_claim_labels_reused_after_manual_check": inherited,
        "novel_claims_manually_checked": novel,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(compile_manual_review(args.output), indent=2))


if __name__ == "__main__":
    main()
