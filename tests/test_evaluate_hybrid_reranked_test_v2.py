"""Tests des verrous de l'unique évaluation TEST."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest
from scripts import evaluate_hybrid_reranked_test_v2 as evaluate_test


def make_gold_records() -> list[dict[str, Any]]:
    """Créer les 28 entrées minimales du gold TEST."""

    return [
        {
            "query_id": f"Q{number:03d}",
            "answerable": number < 45,
            "gold_chunk_ids": (
                [f"chunk_{number:03d}"]
                if number < 45
                else []
            ),
        }
        for number in range(21, 49)
    ]


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def frozen_arguments() -> argparse.Namespace:
    return argparse.Namespace(
        candidate_k=20,
        dense_candidates=20,
        bm25_candidates=20,
        top_k=5,
        disable_query_expansion=False,
    )


def test_validate_gold_snapshot_accepts_expected_counts(
    tmp_path: Path,
) -> None:
    gold_path = tmp_path / "gold.jsonl"
    write_jsonl(gold_path, make_gold_records())
    expected_hash = evaluate_test.sha256_file(gold_path)

    records, preflight = evaluate_test.validate_gold_snapshot(
        gold_path,
        expected_sha256=expected_hash,
    )

    assert len(records) == 28
    assert preflight["entries"] == 28
    assert preflight["answerable"] == 24
    assert preflight["unanswerable"] == 4


def test_validate_gold_snapshot_rejects_wrong_hash(
    tmp_path: Path,
) -> None:
    gold_path = tmp_path / "gold.jsonl"
    write_jsonl(gold_path, make_gold_records())

    with pytest.raises(ValueError, match="Empreinte"):
        evaluate_test.validate_gold_snapshot(
            gold_path,
            expected_sha256="0" * 64,
        )


def test_frozen_arguments_refuse_retuning() -> None:
    arguments = frozen_arguments()
    arguments.top_k = 10

    with pytest.raises(ValueError, match="configuration DEV figée"):
        evaluate_test.validate_frozen_arguments(arguments)


def test_frozen_dev_components_match_snapshot() -> None:
    hashes = evaluate_test.validate_frozen_dev_snapshot()

    assert hashes == evaluate_test.EXPECTED_FROZEN_HASHES


def test_unique_run_guard_refuses_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_manifest = tmp_path / "run.json"
    summary = tmp_path / "summary.json"
    details = tmp_path / "details.csv"
    summary.write_text("already run", encoding="utf-8")

    monkeypatch.setattr(
        evaluate_test,
        "RESULTS_DIRECTORY",
        tmp_path,
    )
    monkeypatch.setattr(
        evaluate_test,
        "RUN_MANIFEST_PATH",
        run_manifest,
    )
    monkeypatch.setattr(
        evaluate_test,
        "SUMMARY_PATH",
        summary,
    )
    monkeypatch.setattr(
        evaluate_test,
        "DETAILS_PATH",
        details,
    )

    with pytest.raises(
        FileExistsError,
        match="seconde exécution refusée",
    ):
        evaluate_test.create_unique_run_manifest(
            gold_preflight={"entries": 28},
            dev_hashes={},
            arguments=frozen_arguments(),
        )
