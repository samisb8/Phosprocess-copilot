from __future__ import annotations

from phosprocess.evaluation.candidate_preservation_v01 import (
    DATASET_SIZE,
    HOLDOUT_SIZE,
    build_verified_dataset,
    compose_candidates,
    contribution_class,
    freeze_splits,
    round_robin_union,
)


def test_freeze_splits_is_exact_label_free_and_deterministic() -> None:
    ids = [f"Q{index:03d}" for index in range(DATASET_SIZE)]

    first = freeze_splits(ids)
    second = freeze_splits(list(reversed(ids)))

    assert first == second
    assert sum(split == "final_holdout" for split in first.values()) == HOLDOUT_SIZE
    assert sum(split == "dev" for split in first.values()) == DATASET_SIZE - HOLDOUT_SIZE


def test_round_robin_union_is_fair_and_deduplicated() -> None:
    rankings = [["a", "shared", "d"], ["b", "shared", "e"], ["c", "f"]]

    assert round_robin_union(rankings, 6) == ["a", "b", "c", "shared", "f", "d"]


def test_composer_reserves_before_fusion_and_fills_budget() -> None:
    rankings = {
        "dense": ["dense", "shared"],
        "sparse": ["sparse", "shared"],
        "bm25": ["bm25", "shared"],
    }
    fused = ["f1", "f2", "f3", "dense"]

    assert compose_candidates(rankings, fused, budget=5, reserve_k=1) == [
        "dense",
        "sparse",
        "bm25",
        "f1",
        "f2",
    ]


def test_contribution_classes_are_mutually_exclusive() -> None:
    rankings = {
        "dense": ["dense", "multiple"],
        "sparse": ["sparse", "multiple"],
        "bm25": ["bm25", "other"],
    }

    assert contribution_class(
        "dense", retriever_rankings=rankings, fused_ranking=[], budget=3, reserve_k=2
    ) == "dense_rescue"
    assert contribution_class(
        "multiple", retriever_rankings=rankings, fused_ranking=[], budget=3, reserve_k=2
    ) == "multiple"
    assert contribution_class(
        "fusion", retriever_rankings=rankings, fused_ranking=["fusion"], budget=3, reserve_k=2
    ) == "fusion_rescue"


def test_verified_dataset_has_64_real_gold_questions_and_frozen_split() -> None:
    rows = build_verified_dataset()

    assert len(rows) == DATASET_SIZE
    assert len({row["id"] for row in rows}) == DATASET_SIZE
    assert {row["language"] for row in rows} == {"fr", "en", "ar"}
    assert len({row["locked_document"] for row in rows}) == 8
    assert sum(row["split"] == "final_holdout" for row in rows) == HOLDOUT_SIZE
    assert all(row["gold_chunk_ids"] for row in rows)
    assert all(row["gold_verification"] for row in rows)
