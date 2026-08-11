"""Evaluate Dense BGE-M3 followed by BGE reranking on DEV."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass
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
from phosprocess.retrieval.dense import DenseRetriever


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

RERANKING_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "reranking.yaml"
)

RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval"
    / "v0.1"
    / "results"
)


@dataclass(frozen=True, slots=True)
class DenseCandidateAdapter:
    """Expose a dense result through the hybrid candidate interface."""

    rank: int
    rrf_score: float
    matched_retrievers: tuple[str, ...]

    dense_rank: int | None
    dense_score: float | None

    bm25_rank: int | None
    bm25_score: float | None

    chunk: Any


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Dense top-N followed by BGE reranking."
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    return parser.parse_args()


def adapt_dense_results(
    dense_results: list[Any],
) -> list[DenseCandidateAdapter]:
    candidates: list[DenseCandidateAdapter] = []

    for result in dense_results:
        candidates.append(
            DenseCandidateAdapter(
                rank=int(result.rank),
                rrf_score=float(result.score),
                matched_retrievers=("dense",),
                dense_rank=int(result.rank),
                dense_score=float(result.score),
                bm25_rank=None,
                bm25_score=None,
                chunk=result.chunk,
            )
        )

    return candidates


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    arguments = parse_arguments()

    if arguments.candidate_k <= 0:
        raise ValueError(
            "candidate-k must be positive."
        )

    if arguments.top_k <= 0:
        raise ValueError(
            "top-k must be positive."
        )

    if arguments.top_k > arguments.candidate_k:
        raise ValueError(
            "top-k cannot exceed candidate-k."
        )

    cases = load_dev_cases()

    print(
        f"DEV answerable queries: {len(cases)}"
    )

    print("\n=== Loading Dense BGE-M3 ===")

    dense_retriever = DenseRetriever(
        index_directory=DENSE_INDEX_DIRECTORY,
        embedding_config_path=EMBEDDING_CONFIG_PATH,
    )

    print("\n=== Loading BGE reranker ===")

    reranking_config = load_reranking_config(
        RERANKING_CONFIG_PATH
    )

    reranker = BGEReranker(
        reranking_config
    )

    rows: list[dict[str, Any]] = []

    for position, case in enumerate(cases, 1):
        started = time.perf_counter()

        dense_response = dense_retriever.search(
            case["question"],
            top_k=arguments.candidate_k,
        )

        dense_candidates = adapt_dense_results(
            dense_response.results
        )

        candidate_ids = [
            candidate.chunk.chunk_id
            for candidate in dense_candidates
        ]

        reranking_response = reranker.rerank(
            case["question"],
            dense_candidates,
            top_k=arguments.top_k,
        )

        final_ids = [
            result.chunk.chunk_id
            for result in reranking_response.results
        ]

        total_latency_ms = (
            time.perf_counter() - started
        ) * 1000.0

        gold_ids = set(
            case["gold_chunk_ids"]
        )

        candidate_ranks = [
            rank
            for rank, chunk_id in enumerate(
                candidate_ids,
                start=1,
            )
            if chunk_id in gold_ids
        ]

        final_ranks = [
            rank
            for rank, chunk_id in enumerate(
                final_ids,
                start=1,
            )
            if chunk_id in gold_ids
        ]

        candidate_gold_rank = (
            candidate_ranks[0]
            if candidate_ranks
            else None
        )

        final_gold_rank = (
            final_ranks[0]
            if final_ranks
            else None
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

        reciprocal_rank_at_5 = (
            1.0 / final_gold_rank
            if final_gold_rank is not None
            and final_gold_rank <= 5
            else 0.0
        )

        retrieved_gold = gold_ids.intersection(
            final_ids[:5]
        )

        evidence_recall_at_5 = (
            len(retrieved_gold)
            / len(gold_ids)
        )

        movement = None

        if (
            candidate_gold_rank is not None
            and final_gold_rank is not None
        ):
            movement = (
                candidate_gold_rank
                - final_gold_rank
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
                "movement": (
                    movement
                    if movement is not None
                    else ""
                ),
                "candidate_hit": candidate_hit,
                "hit_at_1": hit_at_1,
                "hit_at_5": hit_at_5,
                "reciprocal_rank_at_5": (
                    reciprocal_rank_at_5
                ),
                "evidence_recall_at_5": (
                    evidence_recall_at_5
                ),
                "dense_latency_ms": (
                    dense_response.search_duration_ms
                ),
                "reranking_latency_ms": (
                    reranking_response.reranking_duration_ms
                ),
                "total_latency_ms": total_latency_ms,
            }
        )

        before = (
            str(candidate_gold_rank)
            if candidate_gold_rank is not None
            else "MISS"
        )

        after = (
            str(final_gold_rank)
            if final_gold_rank is not None
            else "MISS"
        )

        print(
            f"[{position:02d}/{len(cases):02d}] "
            f"{case['query_id']} | "
            f"dense={before} -> reranked={after} | "
            f"{reranking_response.reranking_duration_ms:.1f} ms"
        )

    total = len(rows)

    candidate_recall = statistics.fmean(
        row["candidate_hit"]
        for row in rows
    )

    hit_at_1 = statistics.fmean(
        row["hit_at_1"]
        for row in rows
    )

    hit_at_5 = statistics.fmean(
        row["hit_at_5"]
        for row in rows
    )

    mrr_at_5 = statistics.fmean(
        row["reciprocal_rank_at_5"]
        for row in rows
    )

    evidence_recall_at_5 = statistics.fmean(
        row["evidence_recall_at_5"]
        for row in rows
    )

    mean_reranking_latency = statistics.fmean(
        row["reranking_latency_ms"]
        for row in rows
    )

    median_reranking_latency = statistics.median(
        row["reranking_latency_ms"]
        for row in rows
    )

    mean_total_latency = statistics.fmean(
        row["total_latency_ms"]
        for row in rows
    )

    summary = {
        "method": "dense_reranked",
        "split": "dev",
        "queries_evaluated": total,
        "candidate_k": arguments.candidate_k,
        "top_k": arguments.top_k,
        "candidate_recall": candidate_recall,
        "hit_at_1": hit_at_1,
        "hit_at_5": hit_at_5,
        "mrr_at_5": mrr_at_5,
        "evidence_recall_at_5": (
            evidence_recall_at_5
        ),
        "mean_reranking_latency_ms": (
            mean_reranking_latency
        ),
        "median_reranking_latency_ms": (
            median_reranking_latency
        ),
        "mean_total_latency_ms": (
            mean_total_latency
        ),
    }

    print("\n" + "=" * 86)
    print("DENSE + BGE RERANKER — DEV")
    print("=" * 86)
    print(
        f"Candidate Recall@{arguments.candidate_k}: "
        f"{sum(row['candidate_hit'] for row in rows)}/{total} "
        f"= {candidate_recall:.3f}"
    )
    print(
        f"Hit@1                         : "
        f"{sum(row['hit_at_1'] for row in rows)}/{total} "
        f"= {hit_at_1:.3f}"
    )
    print(
        f"Hit@5                         : "
        f"{sum(row['hit_at_5'] for row in rows)}/{total} "
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
        f"Mean reranking latency         : "
        f"{mean_reranking_latency:.1f} ms"
    )
    print(
        f"Median reranking latency       : "
        f"{median_reranking_latency:.1f} ms"
    )
    print(
        f"Mean total latency             : "
        f"{mean_total_latency:.1f} ms"
    )

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        RESULTS_DIRECTORY
        / "dev_dense_reranked_summary.json"
    )

    details_path = (
        RESULTS_DIRECTORY
        / "dev_dense_reranked_per_query.csv"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    fieldnames = list(
        rows[0].keys()
    )

    with details_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {details_path}")


if __name__ == "__main__":
    main()
