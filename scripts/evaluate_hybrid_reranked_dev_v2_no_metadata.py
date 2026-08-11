"""Evaluate Hybrid retrieval followed by BGE reranking on DEV."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "scripts"),
)

from evaluate_retrieval_dev import load_dev_cases

from phosprocess.reranking.reranker import (
    BGEReranker,
    load_reranking_config,
)
from phosprocess.retrieval.bm25 import load_bm25_config
from phosprocess.retrieval.hybrid import HybridRetriever


DENSE_INDEX_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "indexes"
    / "dense"
    / "bge_m3"
)

EMBEDDING_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "embeddings.yaml"
)

RETRIEVAL_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "retrieval_v2.yaml"
)

RERANKING_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "reranking_no_metadata.yaml"
)

RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval"
    / "v0.1"
    / "results"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--dense-candidates",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--bm25-candidates",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--disable-query-expansion",
        action="store_true",
    )

    return parser.parse_args()


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def first_rank(
    chunk_ids: list[str],
    gold_ids: set[str],
) -> int | None:
    for rank, chunk_id in enumerate(
        chunk_ids,
        start=1,
    ):
        if chunk_id in gold_ids:
            return rank

    return None


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    arguments = parse_arguments()

    if arguments.top_k > arguments.candidate_k:
        raise ValueError(
            "top-k cannot exceed candidate-k."
        )

    cases = load_dev_cases()

    bm25_config = load_bm25_config(
        RETRIEVAL_CONFIG_PATH
    )

    print(
        f"DEV answerable queries: {len(cases)}"
    )

    print("\n=== Loading Hybrid Retriever ===")

    retriever = HybridRetriever(
        dense_index_directory=DENSE_INDEX_DIRECTORY,
        bm25_index_directory=resolve_project_path(
            bm25_config.output_directory
        ),
        embedding_config_path=EMBEDDING_CONFIG_PATH,
        retrieval_config_path=RETRIEVAL_CONFIG_PATH,
    )

    print("\n=== Loading BGE Reranker ===")

    reranking_config = load_reranking_config(
        RERANKING_CONFIG_PATH
    )

    reranker = BGEReranker(
        reranking_config
    )

    rows: list[dict[str, Any]] = []

    for position, case in enumerate(cases, 1):
        started = time.perf_counter()

        hybrid_response = retriever.search(
            case["question"],
            top_k=arguments.candidate_k,
            dense_candidate_k=(
                arguments.dense_candidates
            ),
            bm25_candidate_k=(
                arguments.bm25_candidates
            ),
            use_query_expansion=(
                not arguments.disable_query_expansion
            ),
        )

        candidate_ids = [
            result.chunk.chunk_id
            for result in hybrid_response.results
        ]

        reranked_response = reranker.rerank(
            case["question"],
            hybrid_response.results,
            top_k=arguments.top_k,
        )

        final_ids = [
            result.chunk.chunk_id
            for result in reranked_response.results
        ]

        total_latency_ms = (
            time.perf_counter() - started
        ) * 1000.0

        gold_ids = set(
            case["gold_chunk_ids"]
        )

        candidate_gold_rank = first_rank(
            candidate_ids,
            gold_ids,
        )

        final_gold_rank = first_rank(
            final_ids,
            gold_ids,
        )

        candidate_hit = int(
            candidate_gold_rank is not None
        )

        hit_at_1 = int(
            final_gold_rank == 1
        )

        hit_at_5 = int(
            final_gold_rank is not None
            and final_gold_rank <= 5
        )

        reciprocal_rank = (
            1.0 / final_gold_rank
            if final_gold_rank is not None
            and final_gold_rank <= 5
            else 0.0
        )

        evidence_recall = (
            len(
                gold_ids.intersection(final_ids[:5])
            )
            / len(gold_ids)
        )

        rows.append(
            {
                "query_id": case["query_id"],
                "category": case["category"],
                "question": case["question"],
                "gold_chunk_ids": "|".join(
                    case["gold_chunk_ids"]
                ),
                "candidate_chunk_ids": "|".join(
                    candidate_ids
                ),
                "reranked_chunk_ids": "|".join(
                    final_ids
                ),
                "candidate_gold_rank": (
                    candidate_gold_rank
                    if candidate_gold_rank is not None
                    else ""
                ),
                "final_gold_rank": (
                    final_gold_rank
                    if final_gold_rank is not None
                    else ""
                ),
                "candidate_hit": candidate_hit,
                "hit_at_1": hit_at_1,
                "hit_at_5": hit_at_5,
                "reciprocal_rank_at_5": (
                    reciprocal_rank
                ),
                "evidence_recall_at_5": (
                    evidence_recall
                ),
                "hybrid_latency_ms": (
                    hybrid_response.total_duration_ms
                ),
                "reranking_latency_ms": (
                    reranked_response.reranking_duration_ms
                ),
                "total_latency_ms": total_latency_ms,
            }
        )

        before = (
            candidate_gold_rank
            if candidate_gold_rank is not None
            else "MISS"
        )

        after = (
            final_gold_rank
            if final_gold_rank is not None
            else "MISS"
        )

        print(
            f"[{position:02d}/{len(cases):02d}] "
            f"{case['query_id']} | "
            f"hybrid={before} -> reranked={after} | "
            f"{reranked_response.reranking_duration_ms:.1f} ms"
        )

    total = len(rows)

    candidate_count = sum(
        row["candidate_hit"]
        for row in rows
    )

    hit_1_count = sum(
        row["hit_at_1"]
        for row in rows
    )

    hit_5_count = sum(
        row["hit_at_5"]
        for row in rows
    )

    candidate_recall = (
        candidate_count / total
    )

    hit_at_1 = (
        hit_1_count / total
    )

    hit_at_5 = (
        hit_5_count / total
    )

    mrr_at_5 = statistics.fmean(
        row["reciprocal_rank_at_5"]
        for row in rows
    )

    evidence_recall_at_5 = statistics.fmean(
        row["evidence_recall_at_5"]
        for row in rows
    )

    reranking_latencies = [
        float(row["reranking_latency_ms"])
        for row in rows
    ]

    total_latencies = [
        float(row["total_latency_ms"])
        for row in rows
    ]

    summary = {
        "method": "hybrid_reranked",
        "split": "dev",
        "candidate_k": arguments.candidate_k,
        "dense_candidates": arguments.dense_candidates,
        "bm25_candidates": arguments.bm25_candidates,
        "query_expansion": (
            not arguments.disable_query_expansion
        ),
        "top_k": arguments.top_k,
        "candidate_recall": candidate_recall,
        "hit_at_1": hit_at_1,
        "hit_at_5": hit_at_5,
        "mrr_at_5": mrr_at_5,
        "evidence_recall_at_5": (
            evidence_recall_at_5
        ),
        "mean_reranking_latency_ms": (
            statistics.fmean(
                reranking_latencies
            )
        ),
        "median_reranking_latency_ms": (
            statistics.median(
                reranking_latencies
            )
        ),
        "mean_total_latency_ms": (
            statistics.fmean(
                total_latencies
            )
        ),
    }

    print("\n" + "=" * 88)
    print("HYBRID + BGE RERANKER — DEV")
    print("=" * 88)

    print(
        f"Candidate Recall@{arguments.candidate_k}: "
        f"{candidate_count}/{total} "
        f"= {candidate_recall:.3f}"
    )

    print(
        f"Hit@1                         : "
        f"{hit_1_count}/{total} "
        f"= {hit_at_1:.3f}"
    )

    print(
        f"Hit@5                         : "
        f"{hit_5_count}/{total} "
        f"= {hit_at_5:.3f}"
    )

    print(
        f"MRR@5                         : "
        f"{mrr_at_5:.3f}"
    )

    print(
        f"Evidence Recall@5             : "
        f"{evidence_recall_at_5:.3f}"
    )

    print(
        f"Median reranking latency       : "
        f"{statistics.median(reranking_latencies):.1f} ms"
    )

    print(
        f"Mean total latency             : "
        f"{statistics.fmean(total_latencies):.1f} ms"
    )

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        RESULTS_DIRECTORY
        / "dev_hybrid_reranked_v2_no_metadata_summary.json"
    )

    details_path = (
        RESULTS_DIRECTORY
        / "dev_hybrid_reranked_v2_no_metadata_per_query.csv"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with details_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {details_path}")


if __name__ == "__main__":
    main()
