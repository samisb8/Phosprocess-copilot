from __future__ import annotations

import pytest

from phosprocess.evaluation.evidence_ground_truth_audit_v01 import (
    ACTIVE_DIRECTORY,
    _evidence_coverage,
    build_manual_annotations,
    build_phase8_results,
)
from phosprocess.ingestion.chunk_serialization import read_child_chunks


@pytest.fixture(scope="module")
def phase8_results() -> dict:
    return build_phase8_results()


def test_all_64_questions_receive_one_manual_audit(phase8_results: dict) -> None:
    assert phase8_results["dataset"] == {
        "question_count": 64,
        "unchanged": 47,
        "added_alternative_evidence": 8,
        "corrected": 9,
        "questionable_historical_gold": 7,
        "category_counts": {"1": 44, "2": 7, "3": 3, "4": 3, "5": 7},
    }
    assert len(phase8_results["annotations"]) == 64


def test_every_curated_evidence_id_exists_in_active_corpus(phase8_results: dict) -> None:
    active_ids = {
        chunk.chunk_id for chunk in read_child_chunks(ACTIVE_DIRECTORY / "chunks.jsonl")
    }
    for annotation in phase8_results["annotations"].values():
        assert set(annotation["region_chunk_ids"]) <= active_ids


def test_evidence_set_recall_exceeds_exact_recall(phase8_results: dict) -> None:
    current = phase8_results["metrics"]["all"]["current"]
    candidate = phase8_results["metrics"]["all"]["phase7_candidate"]
    assert current["exact_recall_at_20"] == pytest.approx(0.7734375)
    assert current["evidence_set_recall_at_20"] == pytest.approx(0.90625)
    assert candidate["exact_recall_at_20"] == pytest.approx(0.8125)
    assert candidate["evidence_set_recall_at_20"] == pytest.approx(0.9375)


def test_complementary_evidence_requires_every_group() -> None:
    _records, annotations = build_manual_annotations()
    evidence_sets = annotations["DQ039"]["valid_evidence_sets"]
    evaporator = "seborg_process_dynamics_control_e99ec9b53deeb4d4"
    mpc = "seborg_process_dynamics_control_9095fc8fd57338b4"
    assert _evidence_coverage(evidence_sets, [evaporator]) == 0.5
    assert _evidence_coverage(evidence_sets, [mpc]) == 0.5
    assert _evidence_coverage(evidence_sets, [evaporator, mpc]) == 1.0


def test_true_hard_misses_are_manually_verified_pool_misses(
    phase8_results: dict,
) -> None:
    assert [item["question_id"] for item in phase8_results["true_hard_misses"]] == [
        "CE051",
        "DQ027",
    ]
    assert all(
        item["candidate_access"] is False
        for item in phase8_results["true_hard_misses"]
    )


def test_holdout_label_changes_are_explicit(phase8_results: dict) -> None:
    assert {
        item["question_id"] for item in phase8_results["holdout_gold_changes"]
    } == {"DQ013", "DQ015", "DQ025", "DQ027", "DQ037"}
