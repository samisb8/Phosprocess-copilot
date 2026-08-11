"""Tests for the DEV-only v3 baseline analysis."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from scripts import analyze_retrieval_dev_v3 as analysis


def make_row(number: int, *, final_rank: int | None = 1) -> dict[str, str]:
    """Create one coherent synthetic DEV result row."""

    query_id = f"Q{number:03d}"
    gold_id = f"gold_{number:03d}"
    candidate_ids = [
        f"candidate_{number:03d}_{position:02d}"
        for position in range(1, 21)
    ]
    candidate_ids[4] = gold_id
    reranked_ids = [
        f"reranked_{number:03d}_{position:02d}"
        for position in range(1, 6)
    ]

    if final_rank is not None:
        reranked_ids[final_rank - 1] = gold_id

    reciprocal_rank = (
        1.0 / final_rank
        if final_rank is not None
        else 0.0
    )

    return {
        "query_id": query_id,
        "category": "synthetic",
        "question": f"Question {number}",
        "gold_chunk_ids": gold_id,
        "candidate_chunk_ids": "|".join(candidate_ids),
        "reranked_chunk_ids": "|".join(reranked_ids),
        "candidate_gold_rank": "5",
        "final_gold_rank": (
            str(final_rank)
            if final_rank is not None
            else ""
        ),
        "candidate_hit": "1",
        "hit_at_1": str(int(final_rank == 1)),
        "hit_at_5": str(int(final_rank is not None)),
        "reciprocal_rank_at_5": str(reciprocal_rank),
        "evidence_recall_at_5": str(int(final_rank is not None)),
        "hybrid_latency_ms": "1.0",
        "reranking_latency_ms": "2.0",
        "total_latency_ms": "3.0",
    }


def make_rows() -> list[dict[str, str]]:
    """Create the exact Q001-Q016 DEV identifier set."""

    return [make_row(number) for number in range(1, 17)]


def test_validate_rows_accepts_exact_dev_split() -> None:
    rows = analysis.validate_and_normalize_rows(make_rows())

    assert len(rows) == 16
    assert rows[0]["query_id"] == "Q001"
    assert rows[-1]["query_id"] == "Q016"


def test_validate_rows_rejects_test_identifier() -> None:
    rows = make_rows()
    rows[-1]["query_id"] = "Q021"

    with pytest.raises(ValueError, match="Q001-Q016"):
        analysis.validate_and_normalize_rows(rows)


def test_validate_row_classifies_reranker_miss() -> None:
    row = analysis.validate_row(make_row(15, final_rank=None))

    assert row["candidate_gold_rank"] == 5
    assert row["final_gold_rank"] is None
    assert row["outcome"] == "reranker_miss"


def test_validate_row_rejects_incoherent_rank() -> None:
    row = make_row(1)
    row["candidate_gold_rank"] = "4"

    with pytest.raises(ValueError, match="candidate_gold_rank"):
        analysis.validate_row(row)


def test_compute_metrics_uses_only_validated_rows() -> None:
    raw_rows = make_rows()
    raw_rows[-1] = make_row(16, final_rank=None)
    rows = analysis.validate_and_normalize_rows(raw_rows)
    metrics = analysis.compute_metrics(rows)

    assert metrics["candidate_recall"] == 1.0
    assert metrics["hit_at_1"] == 15 / 16
    assert metrics["hit_at_5"] == 15 / 16
    assert metrics["mrr_at_5"] == 15 / 16


def test_validate_summary_rejects_non_dev_split() -> None:
    rows = analysis.validate_and_normalize_rows(make_rows())
    metrics = analysis.compute_metrics(rows)
    summary: dict[str, Any] = {
        "split": "test",
        "candidate_k": 20,
        "dense_candidates": 20,
        "bm25_candidates": 20,
        "query_expansion": True,
        "top_k": 5,
        **metrics,
    }

    with pytest.raises(ValueError, match="split DEV"):
        analysis.validate_summary(summary, metrics)


def test_build_analysis_declares_no_test_tuning() -> None:
    raw_rows = make_rows()
    raw_rows[-1] = make_row(16, final_rank=None)
    rows = analysis.validate_and_normalize_rows(raw_rows)
    metrics = analysis.compute_metrics(rows)
    payload = analysis.build_analysis(
        rows=rows,
        metrics=metrics,
        source_hashes={"dev.csv": "ABC"},
    )

    assert payload["data_policy"]["test_data_used"] is False
    assert payload["data_policy"]["test_v2_allowed_for_tuning"] is False
    assert payload["data_policy"]["independent_future_test_required"] is True
    assert payload["outcomes"]["reranker_miss"] == ["Q016"]


def test_ensure_path_within_rejects_sibling_directory(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "dev"
    forbidden = tmp_path / "test" / "results.csv"

    with pytest.raises(ValueError, match="doit rester sous"):
        analysis.ensure_path_within(
            forbidden,
            allowed,
            label="Le détail DEV",
        )


def test_duplicate_query_id_is_rejected() -> None:
    rows = make_rows()
    duplicate = deepcopy(rows[0])
    duplicate["category"] = "duplicate"
    rows[-1] = duplicate

    with pytest.raises(ValueError, match="dupliqué"):
        analysis.validate_and_normalize_rows(rows)
