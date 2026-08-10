"""Unit guards for the evaluation-only Phase-6 retriever harness."""

from __future__ import annotations

from phosprocess.evaluation.retriever_forensics_v01 import (
    aggregate_retriever_ids,
    build_traces,
    classify_gold,
    clean_standalone_query,
    fuse_raw,
    ranking_metrics,
    stable_split,
)


def _raw() -> dict[str, object]:
    return {
        "runs": {
            "dense": [
                {
                    "ids": ["dense_only", "shared", "late"],
                    "scores": [0.9, 0.8, 0.2],
                }
            ],
            "sparse": [
                {
                    "ids": ["sparse_only", "shared"],
                    "scores": [4.0, 2.0],
                }
            ],
            "bm25": [
                {
                    "ids": ["shared", "bm25_only"],
                    "scores": [8.0, 3.0],
                }
            ],
        }
    }


def test_frozen_primary_cohort_matches_phase5_denominator() -> None:
    traces = build_traces()
    primary = [trace for trace in traces if trace["cohort"] == "primary"]
    assert len(primary) == 24
    assert len(traces) == 39
    assert {trace["split"] for trace in primary} == {"dev", "test"}


def test_split_is_stable_and_label_free() -> None:
    assert stable_split("DQ001") == stable_split("DQ001")
    assert stable_split("DQ001") in {"dev", "test"}


def test_metrics_preserve_fractional_multi_gold_recall() -> None:
    metrics = ranking_metrics(
        [["a", "x"], ["z"]],
        [{"a", "b"}, {"z"}],
    )
    assert metrics["recall_at_5"] == 0.75
    assert metrics["mrr"] == 1.0


def test_multi_retriever_fusion_rewards_independent_support() -> None:
    raw = _raw()
    assert aggregate_retriever_ids(raw, "dense")[:2] == ["dense_only", "shared"]
    assert fuse_raw(raw)[0] == "shared"
    assert set(fuse_raw(raw, method="normalized")) == {
        "dense_only",
        "sparse_only",
        "bm25_only",
        "shared",
        "late",
    }


def test_failure_taxonomy_prioritizes_failure_stage() -> None:
    assert classify_gold(
        dense_rank=None,
        sparse_rank=None,
        bm25_rank=None,
        fused_rank=None,
        reranker_rank=None,
    )[0] == "H"
    assert classify_gold(
        dense_rank=8,
        sparse_rank=None,
        bm25_rank=None,
        fused_rank=44,
        reranker_rank=None,
    )[0] == "F"
    assert classify_gold(
        dense_rank=8,
        sparse_rank=9,
        bm25_rank=None,
        fused_rank=10,
        reranker_rank=23,
    )[0] == "G"
    assert classify_gold(
        dense_rank=8,
        sparse_rank=9,
        bm25_rank=None,
        fused_rank=10,
        reranker_rank=2,
    )[0] == "D"


def test_clean_query_removes_only_surface_noise() -> None:
    assert (
        clean_standalone_query(
            "Pouvez-vous expliquer le rôle du coefficient global, s'il vous plaît ?"
        )
        == "expliquer le rôle du coefficient global s'il vous plaît"
    )
