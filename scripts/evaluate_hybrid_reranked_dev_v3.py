"""Run the first retrieval v3 experiment on the DEV split only."""

from __future__ import annotations

import csv
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_retrieval_dev_v3 import (  # noqa: E402
    EXPECTED_QUERY_IDS,
    FROZEN_DEV_DIRECTORY,
    load_snapshot_hashes,
    sha256_file,
)
from evaluate_retrieval_dev import (  # noqa: E402
    extract_query_id,
    extract_query_text,
    load_jsonl,
)

from phosprocess.reranking.reranker import (  # noqa: E402
    BGEReranker,
    load_reranking_config,
)
from phosprocess.retrieval.bm25 import load_bm25_config  # noqa: E402
from phosprocess.retrieval.hybrid import HybridRetriever  # noqa: E402
from phosprocess.retrieval.v3_selection import (  # noqa: E402
    select_with_lexical_safeguard,
)

EVALUATION_ROOT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval"
    / "v0.1"
)
QUERIES_PATH = EVALUATION_ROOT / "queries.jsonl"
GOLD_PATH = FROZEN_DEV_DIRECTORY / "gold_evidence.jsonl"
DENSE_INDEX_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "indexes"
    / "dense"
    / "bge_m3"
)
EMBEDDING_CONFIG_PATH = PROJECT_ROOT / "configs" / "embeddings.yaml"
RETRIEVAL_CONFIG_PATH = FROZEN_DEV_DIRECTORY / "retrieval_v2.yaml"
RERANKING_CONFIG_PATH = FROZEN_DEV_DIRECTORY / "reranking.yaml"
OUTPUT_DIRECTORY = (
    EVALUATION_ROOT
    / "v3"
    / "experiments"
    / "lexical_safeguard_001"
)
SUMMARY_PATH = OUTPUT_DIRECTORY / "summary.json"
DETAILS_PATH = OUTPUT_DIRECTORY / "per_query.csv"

CANDIDATE_K = 20
DENSE_CANDIDATES = 20
BM25_CANDIDATES = 20
TOP_K = 5
LEXICAL_SLOTS = 1
FORBIDDEN_TEST_TOKENS = (
    "test_pool_v2",
    "test_gold_final_v2",
    "test_hybrid_reranked_v2",
)


def ensure_not_test_path(path: Path) -> None:
    """Reject every known TEST v2 artifact namespace."""

    normalized = str(path.resolve()).replace("\\", "/").casefold()

    for token in FORBIDDEN_TEST_TOKENS:
        if token in normalized:
            raise ValueError(
                f"L'expérience v3 DEV refuse le chemin TEST: {path}"
            )


def verify_v2_starting_point() -> dict[str, str]:
    """Verify the immutable v2 components used as the v3 starting point."""

    expected_hashes = load_snapshot_hashes(
        FROZEN_DEV_DIRECTORY / "sha256.csv"
    )
    artifacts = {
        "hybrid.py": (
            PROJECT_ROOT
            / "src"
            / "phosprocess"
            / "retrieval"
            / "hybrid.py"
        ),
        "reranker.py": (
            PROJECT_ROOT
            / "src"
            / "phosprocess"
            / "reranking"
            / "reranker.py"
        ),
        "retrieval_v2.yaml": RETRIEVAL_CONFIG_PATH,
        "reranking.yaml": RERANKING_CONFIG_PATH,
        "gold_evidence.jsonl": GOLD_PATH,
    }
    actual_hashes: dict[str, str] = {}

    for file_name, path in artifacts.items():
        ensure_not_test_path(path)
        actual_hash = sha256_file(path)
        expected_hash = expected_hashes[file_name]

        if actual_hash != expected_hash:
            raise ValueError(
                f"Le point de départ v2 diffère du snapshot pour {file_name}: "
                f"{actual_hash} != {expected_hash}."
            )

        actual_hashes[file_name] = actual_hash

    ensure_not_test_path(QUERIES_PATH)
    ensure_not_test_path(EMBEDDING_CONFIG_PATH)
    ensure_not_test_path(DENSE_INDEX_DIRECTORY)
    ensure_not_test_path(OUTPUT_DIRECTORY)

    actual_hashes[QUERIES_PATH.name] = sha256_file(QUERIES_PATH)
    actual_hashes[EMBEDDING_CONFIG_PATH.name] = sha256_file(
        EMBEDDING_CONFIG_PATH
    )

    return dict(sorted(actual_hashes.items()))


def load_dev_cases() -> list[dict[str, Any]]:
    """Load exactly the DEV Q001-Q016 questions with frozen gold."""

    query_records = load_jsonl(QUERIES_PATH)
    gold_records = load_jsonl(GOLD_PATH)
    queries_by_id = {
        extract_query_id(record): record
        for record in query_records
    }
    cases: list[dict[str, Any]] = []

    for gold in gold_records:
        query_id = extract_query_id(gold)

        if gold.get("split") != "dev":
            raise ValueError(
                f"{query_id}: le snapshot de gold doit être DEV."
            )

        gold_chunk_ids = [
            str(chunk_id).strip()
            for chunk_id in gold.get("gold_chunk_ids", [])
            if str(chunk_id).strip()
        ]

        if gold.get("answerable") is not True:
            if gold_chunk_ids:
                raise ValueError(
                    f"{query_id}: un gold DEV non-answerable doit être vide."
                )

            continue

        if not gold_chunk_ids:
            raise ValueError(f"{query_id}: gold_chunk_ids est vide.")

        query = queries_by_id.get(query_id)

        if query is None:
            raise ValueError(f"{query_id}: question DEV introuvable.")

        if query.get("split") != "dev":
            raise ValueError(f"{query_id}: la question n'est pas DEV.")

        cases.append(
            {
                "query_id": query_id,
                "question": extract_query_text(query),
                "category": str(gold["category"]),
                "gold_chunk_ids": gold_chunk_ids,
            }
        )

    query_ids = [case["query_id"] for case in cases]

    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Le gold DEV contient un query_id dupliqué.")

    if set(query_ids) != EXPECTED_QUERY_IDS:
        raise ValueError(
            "L'expérience v3 exige exactement DEV Q001-Q016."
        )

    return sorted(cases, key=lambda case: case["query_id"])


def resolve_project_path(path_value: str) -> Path:
    """Resolve a configuration path relative to the project."""

    path = Path(path_value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def first_rank(
    chunk_ids: list[str],
    gold_ids: set[str],
) -> int | None:
    """Return the first one-based gold rank."""

    return next(
        (
            rank
            for rank, chunk_id in enumerate(chunk_ids, start=1)
            if chunk_id in gold_ids
        ),
        None,
    )


def metrics_from_rows(
    rows: list[dict[str, Any]],
    *,
    prefix: str,
) -> dict[str, float]:
    """Aggregate a metric family from experiment rows."""

    total = len(rows)

    return {
        "hit_at_1": (
            sum(row[f"{prefix}_hit_at_1"] for row in rows) / total
        ),
        "hit_at_5": (
            sum(row[f"{prefix}_hit_at_5"] for row in rows) / total
        ),
        "mrr_at_5": statistics.fmean(
            row[f"{prefix}_reciprocal_rank_at_5"]
            for row in rows
        ),
        "evidence_recall_at_5": statistics.fmean(
            row[f"{prefix}_evidence_recall_at_5"]
            for row in rows
        ),
    }


def validate_reproduced_baseline(
    baseline_metrics: dict[str, float],
) -> None:
    """Require the side-by-side run to reproduce the frozen DEV baseline."""

    frozen_summary = json.loads(
        (
            FROZEN_DEV_DIRECTORY
            / "dev_hybrid_reranked_v2_summary.json"
        ).read_text(encoding="utf-8")
    )

    for metric_name, actual_value in baseline_metrics.items():
        expected_value = float(frozen_summary[metric_name])

        if abs(actual_value - expected_value) > 1e-12:
            raise ValueError(
                "La reproduction DEV v2 diffère pour "
                f"{metric_name}: {actual_value} != {expected_value}."
            )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON artifact with a Windows replacement fallback."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        os.replace(temporary_path, path)
    except PermissionError:
        shutil.copyfile(temporary_path, path)
        temporary_path.unlink()


def atomic_write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write CSV with a Windows replacement fallback."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    try:
        os.replace(temporary_path, path)
    except PermissionError:
        shutil.copyfile(temporary_path, path)
        temporary_path.unlink()


def build_metric_fields(
    *,
    chunk_ids: list[str],
    gold_ids: set[str],
) -> dict[str, int | float | None]:
    """Build per-query rank and recall fields."""

    rank = first_rank(chunk_ids, gold_ids)

    return {
        "gold_rank": rank,
        "hit_at_1": int(rank == 1),
        "hit_at_5": int(rank is not None),
        "reciprocal_rank_at_5": (
            1.0 / rank
            if rank is not None
            else 0.0
        ),
        "evidence_recall_at_5": (
            len(gold_ids.intersection(chunk_ids))
            / len(gold_ids)
        ),
    }


def main() -> None:
    """Run the fixed lexical-safeguard experiment on DEV."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    source_hashes = verify_v2_starting_point()
    cases = load_dev_cases()
    bm25_config = load_bm25_config(RETRIEVAL_CONFIG_PATH)

    print(f"DEV-only queries: {len(cases)}")
    print("Loading frozen-v2 hybrid starting point...")

    retriever = HybridRetriever(
        dense_index_directory=DENSE_INDEX_DIRECTORY,
        bm25_index_directory=resolve_project_path(
            bm25_config.output_directory
        ),
        embedding_config_path=EMBEDDING_CONFIG_PATH,
        retrieval_config_path=RETRIEVAL_CONFIG_PATH,
    )

    print("Loading frozen-v2 BGE reranker...")
    reranker = BGEReranker(
        load_reranking_config(RERANKING_CONFIG_PATH)
    )
    rows: list[dict[str, Any]] = []

    for position, case in enumerate(cases, start=1):
        started = time.perf_counter()
        hybrid_response = retriever.search(
            case["question"],
            top_k=CANDIDATE_K,
            dense_candidate_k=DENSE_CANDIDATES,
            bm25_candidate_k=BM25_CANDIDATES,
            use_query_expansion=True,
        )
        reranked_response = reranker.rerank(
            case["question"],
            hybrid_response.results,
            top_k=CANDIDATE_K,
        )
        selected = select_with_lexical_safeguard(
            hybrid_response.results,
            reranked_response.results,
            top_k=TOP_K,
            lexical_slots=LEXICAL_SLOTS,
        )
        candidate_ids = [
            result.chunk.chunk_id
            for result in hybrid_response.results
        ]
        baseline_ids = [
            result.chunk.chunk_id
            for result in reranked_response.results[:TOP_K]
        ]
        v3_ids = [result.chunk_id for result in selected]
        gold_ids = set(case["gold_chunk_ids"])
        candidate_rank = first_rank(candidate_ids, gold_ids)
        baseline_fields = build_metric_fields(
            chunk_ids=baseline_ids,
            gold_ids=gold_ids,
        )
        v3_fields = build_metric_fields(
            chunk_ids=v3_ids,
            gold_ids=gold_ids,
        )
        total_latency_ms = (
            time.perf_counter() - started
        ) * 1000.0
        lexical_result = next(
            (
                result
                for result in selected
                if result.source == "bm25_safeguard"
            ),
            None,
        )
        row: dict[str, Any] = {
            "query_id": case["query_id"],
            "category": case["category"],
            "question": case["question"],
            "gold_chunk_ids": "|".join(case["gold_chunk_ids"]),
            "candidate_chunk_ids": "|".join(candidate_ids),
            "candidate_gold_rank": (
                candidate_rank
                if candidate_rank is not None
                else ""
            ),
            "baseline_chunk_ids": "|".join(baseline_ids),
            "v3_chunk_ids": "|".join(v3_ids),
            "lexical_safeguard_chunk_id": (
                lexical_result.chunk_id
                if lexical_result is not None
                else ""
            ),
            "selection_sources": "|".join(
                result.source
                for result in selected
            ),
        }

        for name, value in baseline_fields.items():
            row[f"baseline_{name}"] = (
                value
                if value is not None
                else ""
            )

        for name, value in v3_fields.items():
            row[f"v3_{name}"] = (
                value
                if value is not None
                else ""
            )

        row.update(
            {
                "hybrid_latency_ms": hybrid_response.total_duration_ms,
                "reranking_latency_ms": (
                    reranked_response.reranking_duration_ms
                ),
                "total_latency_ms": total_latency_ms,
            }
        )
        rows.append(row)

        print(
            f"[{position:02d}/{len(cases):02d}] {case['query_id']} | "
            f"v2={baseline_fields['gold_rank'] or 'MISS'} -> "
            f"v3={v3_fields['gold_rank'] or 'MISS'}"
        )

    candidate_recall = (
        sum(
            int(row["candidate_gold_rank"] != "")
            for row in rows
        )
        / len(rows)
    )
    baseline_metrics = metrics_from_rows(rows, prefix="baseline")
    v3_metrics = metrics_from_rows(rows, prefix="v3")
    validate_reproduced_baseline(baseline_metrics)
    improved_queries = [
        row["query_id"]
        for row in rows
        if row["v3_reciprocal_rank_at_5"]
        > row["baseline_reciprocal_rank_at_5"]
    ]
    regressed_queries = [
        row["query_id"]
        for row in rows
        if row["v3_reciprocal_rank_at_5"]
        < row["baseline_reciprocal_rank_at_5"]
    ]
    summary = {
        "experiment_id": "v3_lexical_safeguard_001",
        "status": "dev_candidate",
        "split": "dev",
        "test_data_used": False,
        "independent_future_test_required": True,
        "method": "hybrid_reranked_with_lexical_safeguard",
        "selection_policy": {
            "top_k": TOP_K,
            "reranker_leading_slots": TOP_K - LEXICAL_SLOTS,
            "bm25_safeguard_slots": LEXICAL_SLOTS,
            "fallback": "next_reranker_result",
        },
        "candidate_k": CANDIDATE_K,
        "dense_candidates": DENSE_CANDIDATES,
        "bm25_candidates": BM25_CANDIDATES,
        "query_expansion": True,
        "candidate_recall": candidate_recall,
        "baseline_v2_reproduced": baseline_metrics,
        "v3_metrics": v3_metrics,
        "metric_delta": {
            metric_name: (
                v3_metrics[metric_name]
                - baseline_metrics[metric_name]
            )
            for metric_name in baseline_metrics
        },
        "improved_queries": improved_queries,
        "regressed_queries": regressed_queries,
        "mean_reranking_latency_ms": statistics.fmean(
            row["reranking_latency_ms"]
            for row in rows
        ),
        "mean_total_latency_ms": statistics.fmean(
            row["total_latency_ms"]
            for row in rows
        ),
        "source_sha256": source_hashes,
    }

    atomic_write_json(SUMMARY_PATH, summary)
    atomic_write_csv(DETAILS_PATH, rows)

    print("\nDEV v3 experiment complete.")
    print(f"Candidate Recall@20: {candidate_recall:.4f}")
    print(f"Baseline Hit@5: {baseline_metrics['hit_at_5']:.4f}")
    print(f"v3 Hit@5: {v3_metrics['hit_at_5']:.4f}")
    print(f"Baseline MRR@5: {baseline_metrics['mrr_at_5']:.4f}")
    print(f"v3 MRR@5: {v3_metrics['mrr_at_5']:.4f}")
    print(f"Improved: {', '.join(improved_queries) or 'none'}")
    print(f"Regressed: {', '.join(regressed_queries) or 'none'}")
    print("TEST used: False")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Details: {DETAILS_PATH}")


if __name__ == "__main__":
    main()
