"""Evaluate retrieval configurations on the verified DEV gold set."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

from phosprocess.retrieval.bm25 import (
    BM25Retriever,
    load_bm25_config,
)
from phosprocess.retrieval.dense import DenseRetriever
from phosprocess.retrieval.hybrid import HybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVALUATION_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval"
    / "v0.1"
)

QUERIES_PATH = EVALUATION_DIRECTORY / "queries.jsonl"
GOLD_PATH = EVALUATION_DIRECTORY / "gold_evidence.jsonl"

RESULTS_DIRECTORY = EVALUATION_DIRECTORY / "results"

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
    / "retrieval.yaml"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate BM25, dense and hybrid retrieval "
            "on the verified DEV gold evidence."
        )
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
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
        "--methods",
        nargs="+",
        choices=[
            "bm25",
            "dense",
            "hybrid",
            "hybrid_no_expansion",
        ],
        default=[
            "bm25",
            "dense",
            "hybrid",
            "hybrid_no_expansion",
        ],
    )

    return parser.parse_args()


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: "
                    f"{error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected an object in {path}, "
                    f"line {line_number}."
                )

            records.append(record)

    return records


def first_value(
    record: dict[str, Any],
    keys: list[str],
) -> Any:
    for key in keys:
        value = record.get(key)

        if value is not None:
            return value

    return None


def extract_query_id(record: dict[str, Any]) -> str:
    value = first_value(
        record,
        [
            "query_id",
            "question_id",
            "id",
        ],
    )

    if value is None:
        raise KeyError(
            f"No query identifier found in record: {record}"
        )

    return str(value).strip()


def extract_query_text(record: dict[str, Any]) -> str:
    value = first_value(
        record,
        [
            "question",
            "query",
            "question_text",
            "query_text",
            "text",
        ],
    )

    if isinstance(value, str) and value.strip():
        return value.strip()

    for container_key in [
        "input",
        "data",
        "metadata",
    ]:
        container = record.get(container_key)

        if not isinstance(container, dict):
            continue

        value = first_value(
            container,
            [
                "question",
                "query",
                "question_text",
                "query_text",
                "text",
            ],
        )

        if isinstance(value, str) and value.strip():
            return value.strip()

    raise KeyError(
        "No query text found for record "
        f"{extract_query_id(record)}."
    )


def load_dev_cases() -> list[dict[str, Any]]:
    query_records = load_jsonl(QUERIES_PATH)
    gold_records = load_jsonl(GOLD_PATH)

    queries_by_id = {
        extract_query_id(record): record
        for record in query_records
    }

    cases: list[dict[str, Any]] = []

    for gold in gold_records:
        query_id = extract_query_id(gold)

        if str(gold.get("split", "")).lower() != "dev":
            continue

        if gold.get("answerable") is not True:
            continue

        gold_chunk_ids = gold.get("gold_chunk_ids", [])

        if not isinstance(gold_chunk_ids, list):
            raise TypeError(
                f"gold_chunk_ids must be a list for {query_id}."
            )

        gold_chunk_ids = [
            str(chunk_id).strip()
            for chunk_id in gold_chunk_ids
            if str(chunk_id).strip()
        ]

        if not gold_chunk_ids:
            raise ValueError(
                f"Answerable query {query_id} has no gold evidence."
            )

        query_record = queries_by_id.get(query_id)

        if query_record is None:
            raise KeyError(
                f"{query_id} exists in gold but not in queries.jsonl."
            )

        cases.append(
            {
                "query_id": query_id,
                "question": extract_query_text(query_record),
                "category": gold.get(
                    "category",
                    query_record.get("category", ""),
                ),
                "gold_chunk_ids": gold_chunk_ids,
            }
        )

    cases.sort(
        key=lambda item: int(
            "".join(
                character
                for character in item["query_id"]
                if character.isdigit()
            )
            or 0
        )
    )

    return cases


def extract_retrieved_ids(
    response: Any,
    *,
    top_k: int,
) -> list[str]:
    chunk_ids: list[str] = []

    for result in response.results[:top_k]:
        chunk = getattr(result, "chunk", None)

        if chunk is None:
            raise AttributeError(
                "A retrieval result has no chunk attribute."
            )

        chunk_id = getattr(chunk, "chunk_id", None)

        if not chunk_id:
            raise AttributeError(
                "A retrieved chunk has no chunk_id."
            )

        chunk_ids.append(str(chunk_id))

    return chunk_ids


def nearest_rank_percentile(
    values: list[float],
    percentile: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    index = max(
        0,
        math.ceil(percentile * len(ordered)) - 1,
    )

    return ordered[index]


def evaluate_method(
    method_name: str,
    cases: list[dict[str, Any]],
    search_function: Callable[[str, int], Any],
    *,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []

    print("\n" + "=" * 90)
    print(f"METHOD: {method_name}")
    print("=" * 90)

    for position, case in enumerate(cases, 1):
        started = time.perf_counter()

        response = search_function(
            case["question"],
            top_k,
        )

        wall_latency_ms = (
            time.perf_counter() - started
        ) * 1000.0

        retrieved_ids = extract_retrieved_ids(
            response,
            top_k=top_k,
        )

        gold_ids = set(case["gold_chunk_ids"])

        relevant_ranks = [
            rank
            for rank, chunk_id in enumerate(
                retrieved_ids,
                start=1,
            )
            if chunk_id in gold_ids
        ]

        first_relevant_rank = (
            relevant_ranks[0]
            if relevant_ranks
            else None
        )

        hit_at_1 = int(
            first_relevant_rank == 1
        )

        hit_at_5 = int(
            first_relevant_rank is not None
            and first_relevant_rank <= 5
        )

        reciprocal_rank_at_5 = (
            1.0 / first_relevant_rank
            if first_relevant_rank is not None
            and first_relevant_rank <= 5
            else 0.0
        )

        retrieved_gold = gold_ids.intersection(
            retrieved_ids[:5]
        )

        evidence_recall_at_5 = (
            len(retrieved_gold) / len(gold_ids)
        )

        row = {
            "method": method_name,
            "query_id": case["query_id"],
            "category": case["category"],
            "question": case["question"],
            "gold_chunk_ids": "|".join(
                case["gold_chunk_ids"]
            ),
            "retrieved_chunk_ids": "|".join(
                retrieved_ids
            ),
            "first_relevant_rank": (
                first_relevant_rank
                if first_relevant_rank is not None
                else ""
            ),
            "hit_at_1": hit_at_1,
            "hit_at_5": hit_at_5,
            "reciprocal_rank_at_5": (
                reciprocal_rank_at_5
            ),
            "evidence_recall_at_5": (
                evidence_recall_at_5
            ),
            "latency_ms": wall_latency_ms,
        }

        rows.append(row)

        marker = (
            f"rank={first_relevant_rank}"
            if first_relevant_rank is not None
            else "MISS"
        )

        print(
            f"[{position:02d}/{len(cases):02d}] "
            f"{case['query_id']} | "
            f"{marker} | "
            f"{wall_latency_ms:.1f} ms"
        )

    latencies = [
        float(row["latency_ms"])
        for row in rows
    ]

    summary = {
        "method": method_name,
        "queries_evaluated": len(rows),
        "top_k": top_k,
        "hit_at_1": statistics.fmean(
            row["hit_at_1"]
            for row in rows
        ),
        "hit_at_5": statistics.fmean(
            row["hit_at_5"]
            for row in rows
        ),
        "mrr_at_5": statistics.fmean(
            row["reciprocal_rank_at_5"]
            for row in rows
        ),
        "evidence_recall_at_5": statistics.fmean(
            row["evidence_recall_at_5"]
            for row in rows
        ),
        "mean_latency_ms": statistics.fmean(
            latencies
        ),
        "median_latency_ms": statistics.median(
            latencies
        ),
        "p95_latency_ms": nearest_rank_percentile(
            latencies,
            0.95,
        ),
    }

    return summary, rows


def release_memory() -> None:
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def create_bm25_search(
) -> tuple[Any, Callable[[str, int], Any]]:
    config = load_bm25_config(
        RETRIEVAL_CONFIG_PATH
    )

    retriever = BM25Retriever(
        index_directory=resolve_project_path(
            config.output_directory
        ),
        config_path=RETRIEVAL_CONFIG_PATH,
    )

    def search(query: str, top_k: int) -> Any:
        return retriever.search(
            query,
            top_k=top_k,
        )

    return retriever, search


def create_dense_search(
) -> tuple[Any, Callable[[str, int], Any]]:
    retriever = DenseRetriever(
        index_directory=DENSE_INDEX_DIRECTORY,
        embedding_config_path=EMBEDDING_CONFIG_PATH,
    )

    def search(query: str, top_k: int) -> Any:
        return retriever.search(
            query,
            top_k=top_k,
        )

    return retriever, search


def create_hybrid_search(
    *,
    dense_candidates: int,
    bm25_candidates: int,
    use_query_expansion: bool,
) -> tuple[Any, Callable[[str, int], Any]]:
    config = load_bm25_config(
        RETRIEVAL_CONFIG_PATH
    )

    retriever = HybridRetriever(
        dense_index_directory=DENSE_INDEX_DIRECTORY,
        bm25_index_directory=resolve_project_path(
            config.output_directory
        ),
        embedding_config_path=EMBEDDING_CONFIG_PATH,
        retrieval_config_path=RETRIEVAL_CONFIG_PATH,
    )

    def search(query: str, top_k: int) -> Any:
        return retriever.search(
            query,
            top_k=top_k,
            dense_candidate_k=dense_candidates,
            bm25_candidate_k=bm25_candidates,
            use_query_expansion=use_query_expansion,
        )

    return retriever, search


def save_results(
    summaries: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    arguments: argparse.Namespace,
) -> None:
    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        RESULTS_DIRECTORY
        / "dev_retrieval_summary.json"
    )

    rows_path = (
        RESULTS_DIRECTORY
        / "dev_retrieval_per_query.csv"
    )

    payload = {
        "split": "dev",
        "answerable_queries_only": True,
        "top_k": arguments.top_k,
        "dense_candidates": arguments.dense_candidates,
        "bm25_candidates": arguments.bm25_candidates,
        "summaries": summaries,
    }

    summary_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    fieldnames = [
        "method",
        "query_id",
        "category",
        "question",
        "gold_chunk_ids",
        "retrieved_chunk_ids",
        "first_relevant_rank",
        "hit_at_1",
        "hit_at_5",
        "reciprocal_rank_at_5",
        "evidence_recall_at_5",
        "latency_ms",
    ]

    with rows_path.open(
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
    print(f"  {rows_path}")


def print_summary_table(
    summaries: list[dict[str, Any]],
) -> None:
    print("\n" + "=" * 108)
    print("DEV RETRIEVAL SUMMARY")
    print("=" * 108)

    header = (
        f"{'Method':<24}"
        f"{'Hit@1':>10}"
        f"{'Hit@5':>10}"
        f"{'MRR@5':>10}"
        f"{'Recall@5':>12}"
        f"{'Mean ms':>12}"
        f"{'P95 ms':>12}"
    )

    print(header)
    print("-" * 108)

    for summary in summaries:
        print(
            f"{summary['method']:<24}"
            f"{summary['hit_at_1']:>10.3f}"
            f"{summary['hit_at_5']:>10.3f}"
            f"{summary['mrr_at_5']:>10.3f}"
            f"{summary['evidence_recall_at_5']:>12.3f}"
            f"{summary['mean_latency_ms']:>12.1f}"
            f"{summary['p95_latency_ms']:>12.1f}"
        )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    arguments = parse_arguments()
    cases = load_dev_cases()

    print(f"DEV answerable queries loaded: {len(cases)}")

    if len(cases) != 16:
        print(
            "Warning: 16 answerable DEV queries were expected, "
            f"but {len(cases)} were loaded."
        )

    summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    for method in arguments.methods:
        retriever = None
        search_function = None

        if method == "bm25":
            retriever, search_function = (
                create_bm25_search()
            )

        elif method == "dense":
            retriever, search_function = (
                create_dense_search()
            )

        elif method == "hybrid":
            retriever, search_function = (
                create_hybrid_search(
                    dense_candidates=(
                        arguments.dense_candidates
                    ),
                    bm25_candidates=(
                        arguments.bm25_candidates
                    ),
                    use_query_expansion=True,
                )
            )

        elif method == "hybrid_no_expansion":
            retriever, search_function = (
                create_hybrid_search(
                    dense_candidates=(
                        arguments.dense_candidates
                    ),
                    bm25_candidates=(
                        arguments.bm25_candidates
                    ),
                    use_query_expansion=False,
                )
            )

        if search_function is None:
            raise RuntimeError(
                f"Unsupported method: {method}"
            )

        summary, rows = evaluate_method(
            method,
            cases,
            search_function,
            top_k=arguments.top_k,
        )

        summaries.append(summary)
        all_rows.extend(rows)

        del search_function
        del retriever
        release_memory()

    print_summary_table(summaries)

    save_results(
        summaries,
        all_rows,
        arguments=arguments,
    )


if __name__ == "__main__":
    main()
