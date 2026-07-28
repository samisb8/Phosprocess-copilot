"""Tests for the DEV-only dev_best_v3 freeze decision."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from scripts import freeze_dev_best_v3 as freeze


def make_sensitivity_rows() -> list[dict[str, Any]]:
    """Create the three validated sensitivity variants."""

    return [
        {
            "variant_id": "strict_lexical_slots_0",
            "label": "plus_stricte",
            "lexical_slots": 0,
            "complexity": 0,
            "candidate_recall_at_20": 1.0,
            "evidence_recall_at_5": 0.9375,
            "hit_at_5": 0.9375,
            "mrr_at_5": 0.7760416666666666,
            "hit_at_1": 0.6875,
            "median_policy_latency_us": 50.0,
            "metrics_identical_across_runs": True,
            "eligible_candidate_recall": True,
            "selection_rank": 2,
            "selected": False,
        },
        {
            "variant_id": "lexical_safeguard_001",
            "label": "actuelle",
            "lexical_slots": 1,
            "complexity": 1,
            "candidate_recall_at_20": 1.0,
            "evidence_recall_at_5": 1.0,
            "hit_at_5": 1.0,
            "mrr_at_5": 0.7885416666666667,
            "hit_at_1": 0.6875,
            "median_policy_latency_us": 35.0,
            "metrics_identical_across_runs": True,
            "eligible_candidate_recall": True,
            "selection_rank": 1,
            "selected": True,
        },
        {
            "variant_id": "permissive_lexical_slots_2",
            "label": "plus_permissive",
            "lexical_slots": 2,
            "complexity": 2,
            "candidate_recall_at_20": 1.0,
            "evidence_recall_at_5": 0.9375,
            "hit_at_5": 0.9375,
            "mrr_at_5": 0.7760416666666666,
            "hit_at_1": 0.6875,
            "median_policy_latency_us": 30.0,
            "metrics_identical_across_runs": True,
            "eligible_candidate_recall": True,
            "selection_rank": 3,
            "selected": False,
        },
    ]


def make_per_query_rows() -> list[dict[str, str]]:
    """Create 16 stable rows for each sensitivity variant."""

    rows: list[dict[str, str]] = []

    for variant_id in freeze.EXPECTED_VARIANT_IDS:
        for number in range(1, 17):
            outcome = "unchanged"

            if (
                variant_id == "permissive_lexical_slots_2"
                and number == 6
            ):
                outcome = "regressed"

            if (
                variant_id == "lexical_safeguard_001"
                and number == 15
            ):
                outcome = "improved"

            rows.append(
                {
                    "variant_id": variant_id,
                    "query_id": f"Q{number:03d}",
                    "same_selection_across_runs": "True",
                    "outcome_vs_baseline": outcome,
                }
            )

    return rows


def make_summary(
    sensitivity_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a successful robustness summary."""

    return {
        "split": "dev",
        "test_artifacts_read": False,
        "test_evaluation_run": False,
        "reference_answers_used_for_inference": False,
        "gold_used_for_inference": False,
        "v3_frozen": False,
        "robustness_passed": True,
        "selection_rule": freeze.EXPECTED_SELECTION_RULE,
        "determinism": {
            "repetitions": 3,
            "all_variant_selections_stable": True,
            "baseline_same_top5": True,
            "current_same_metrics": True,
            "current_same_top5": True,
            "stable_current_queries": 16,
            "total_current_queries": 16,
        },
        "baseline_v2_metrics": {
            "candidate_recall_at_20": 1.0,
            "evidence_recall_at_5": 0.9375,
            "hit_at_5": 0.9375,
            "mrr_at_5": 0.7760416666666666,
            "hit_at_1": 0.6875,
        },
        "selected_variant": "lexical_safeguard_001",
        "recommendation": "freeze lexical_safeguard_001",
        "validation_id": "synthetic",
        "variants": sensitivity_rows,
    }


def test_predefined_rule_selects_lexical_safeguard_001() -> None:
    rows = make_sensitivity_rows()

    ranking = freeze.select_variant(
        rows,
        baseline_candidate_recall=1.0,
    )

    assert ranking[0]["variant_id"] == "lexical_safeguard_001"
    assert [row["variant_id"] for row in ranking] == [
        "lexical_safeguard_001",
        "strict_lexical_slots_0",
        "permissive_lexical_slots_2",
    ]


def test_candidate_recall_regression_is_ineligible() -> None:
    rows = make_sensitivity_rows()
    rows[1]["candidate_recall_at_20"] = 0.99

    ranking = freeze.select_variant(
        rows,
        baseline_candidate_recall=1.0,
    )

    assert all(
        row["variant_id"] != "lexical_safeguard_001"
        for row in ranking
    )


def test_validation_accepts_one_deterministic_winner() -> None:
    sensitivity_rows = make_sensitivity_rows()
    decision = freeze.validate_summary_consistency(
        make_summary(sensitivity_rows),
        sensitivity_rows,
        make_per_query_rows(),
        "freeze lexical_safeguard_001",
    )

    assert decision["winner"]["variant_id"] == "lexical_safeguard_001"
    assert decision["regressions"]["lexical_safeguard_001"] == 0
    assert decision["regressions"]["permissive_lexical_slots_2"] == 1


def test_validation_rejects_second_selected_variant() -> None:
    sensitivity_rows = make_sensitivity_rows()
    sensitivity_rows[0]["selected"] = True

    with pytest.raises(
        freeze.FreezeValidationError,
        match="Une seule variante",
    ):
        freeze.validate_summary_consistency(
            make_summary(sensitivity_rows),
            sensitivity_rows,
            make_per_query_rows(),
            "freeze lexical_safeguard_001",
        )


def test_validation_rejects_nondeterministic_winner() -> None:
    sensitivity_rows = make_sensitivity_rows()
    summary = make_summary(sensitivity_rows)
    summary["determinism"]["current_same_top5"] = False

    with pytest.raises(
        freeze.FreezeValidationError,
        match="déterministe",
    ):
        freeze.validate_summary_consistency(
            summary,
            sensitivity_rows,
            make_per_query_rows(),
            "freeze lexical_safeguard_001",
        )


def test_snapshot_identity_is_order_independent() -> None:
    first = {
        "a": "A" * 64,
        "b": "B" * 64,
    }
    second = {
        "b": "B" * 64,
        "a": "A" * 64,
    }

    assert freeze.snapshot_identity(first) == freeze.snapshot_identity(
        second
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("True", True),
        ("False", False),
        ("1", True),
        ("0", False),
        (True, True),
        (False, False),
    ],
)
def test_parse_bool_accepts_json_and_csv_encodings(
    value: Any,
    expected: bool,
) -> None:
    assert freeze.parse_bool(value) is expected


def test_validation_does_not_mutate_sensitivity_rows() -> None:
    rows = make_sensitivity_rows()
    original = deepcopy(rows)

    freeze.select_variant(
        rows,
        baseline_candidate_recall=1.0,
    )

    assert rows == original
