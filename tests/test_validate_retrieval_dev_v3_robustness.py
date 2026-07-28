"""Tests for the DEV-only v3 robustness protocol."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from scripts import validate_retrieval_dev_v3_robustness as robustness


def make_sensitivity_row(
    variant_id: str,
    *,
    candidate_recall: float = 1.0,
    evidence_recall: float = 1.0,
    hit_at_5: float = 1.0,
    mrr_at_5: float = 0.8,
    hit_at_1: float = 0.7,
    complexity: int = 1,
    latency: float = 10.0,
) -> dict[str, Any]:
    """Create the fields used by the predefined selection rule."""

    return {
        "variant_id": variant_id,
        "candidate_recall_at_20": candidate_recall,
        "evidence_recall_at_5": evidence_recall,
        "hit_at_5": hit_at_5,
        "mrr_at_5": mrr_at_5,
        "hit_at_1": hit_at_1,
        "complexity": complexity,
        "median_policy_latency_us": latency,
        "selection_rank": "",
        "selected": False,
    }


def test_static_audit_confirms_generic_policy() -> None:
    audit = robustness.audit_selection_policy()

    assert audit["passed"] is True
    assert audit["hardcoded_query_ids"] == []
    assert audit["hardcoded_chunk_ids"] == []
    assert audit["forbidden_terms_found"] == []
    assert audit["checks"]["only_retrieval_rank_signals_used"] is True


def test_selection_rule_prioritizes_evidence_recall() -> None:
    faster_lower_recall = make_sensitivity_row(
        "faster",
        evidence_recall=0.95,
        complexity=0,
        latency=1.0,
    )
    slower_higher_recall = make_sensitivity_row(
        "higher_recall",
        evidence_recall=1.0,
        complexity=2,
        latency=100.0,
    )
    rows = [faster_lower_recall, slower_higher_recall]

    selected = robustness.choose_variant(
        rows,
        baseline_candidate_recall=1.0,
    )

    assert selected["variant_id"] == "higher_recall"


def test_selection_rule_rejects_candidate_recall_regression() -> None:
    regressed = make_sensitivity_row(
        "regressed",
        candidate_recall=0.99,
        evidence_recall=1.0,
    )
    eligible = make_sensitivity_row(
        "eligible",
        evidence_recall=0.9,
    )
    rows = [regressed, eligible]

    selected = robustness.choose_variant(
        rows,
        baseline_candidate_recall=1.0,
    )

    assert selected["variant_id"] == "eligible"


def test_selection_rule_uses_simplicity_before_latency() -> None:
    simple = make_sensitivity_row(
        "simple",
        complexity=1,
        latency=100.0,
    )
    complex_but_fast = make_sensitivity_row(
        "complex_fast",
        complexity=2,
        latency=1.0,
    )
    rows = [deepcopy(complex_but_fast), deepcopy(simple)]

    selected = robustness.choose_variant(
        rows,
        baseline_candidate_recall=1.0,
    )

    assert selected["variant_id"] == "simple"
