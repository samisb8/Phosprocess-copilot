"""Exécuter une seule évaluation TEST avec les artefacts DEV/gold figés."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "scripts"),
)


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []

    for line in path.read_text(
        encoding="utf-8-sig"
    ).splitlines():
        if line.strip():
            records.append(json.loads(line))

    return records


EXPECTED_GOLD_SHA256 = (
    "93C417B7AA6ECE47172FCC9197BCFB0AB"
    "3C84055AB782D073A739B1A64E95F99"
)
EXPECTED_TEST_IDS = {
    f"Q{number:03d}"
    for number in range(21, 49)
}

EVALUATION_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval"
    / "v0.1"
)
DEFAULT_QUERIES_PATH = (
    EVALUATION_DIRECTORY / "queries.jsonl"
)
FROZEN_DEV_DIRECTORY = (
    EVALUATION_DIRECTORY
    / "frozen"
    / "dev_best_v2"
)

ACTIVE_RETRIEVAL_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "retrieval_v2.yaml"
)
ACTIVE_RERANKING_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "reranking.yaml"
)
ACTIVE_HYBRID_PATH = (
    PROJECT_ROOT
    / "src"
    / "phosprocess"
    / "retrieval"
    / "hybrid.py"
)
ACTIVE_RERANKER_PATH = (
    PROJECT_ROOT
    / "src"
    / "phosprocess"
    / "reranking"
    / "reranker.py"
)

FROZEN_RETRIEVAL_CONFIG_PATH = (
    FROZEN_DEV_DIRECTORY / "retrieval_v2.yaml"
)
FROZEN_RERANKING_CONFIG_PATH = (
    FROZEN_DEV_DIRECTORY / "reranking.yaml"
)
FROZEN_HYBRID_PATH = (
    FROZEN_DEV_DIRECTORY / "hybrid.py"
)
FROZEN_RERANKER_PATH = (
    FROZEN_DEV_DIRECTORY / "reranker.py"
)

EXPECTED_FROZEN_HASHES = {
    "retrieval_v2.yaml": (
        "04DE6EF85B34C0E7D61AA9904392874D"
        "B136F2C7FFD62DC2129F0C59EFC37C12"
    ),
    "reranking.yaml": (
        "3A8DA7F3BFFD9592CFCC7A1139AE738B"
        "0FC919B49EAD3B652C5E81EEA761C0B0"
    ),
    "hybrid.py": (
        "074371D34A9529542F387DD0795E9FE2"
        "795229EE1AB82D7B585CC3F03168741B"
    ),
    "reranker.py": (
        "100A060F074FA2CE32AC0CA7ACBA589B"
        "5C52709A8E11DA71EBE70EC25E400737"
    ),
}


def sha256_file(path: Path) -> str:
    """Calculer l'empreinte SHA-256 d'un fichier."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest().upper()


def validate_gold_snapshot(
    gold_path: Path,
    *,
    expected_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Valider le gold TEST figé avant tout chargement de modèle."""

    if not gold_path.exists():
        raise FileNotFoundError(
            f"Snapshot gold introuvable : {gold_path}"
        )

    normalized_expected_hash = expected_sha256.strip().upper()

    if not re.fullmatch(
        r"[0-9A-F]{64}",
        normalized_expected_hash,
    ):
        raise ValueError(
            "L'empreinte SHA-256 attendue est invalide."
        )

    actual_hash = sha256_file(gold_path)

    if actual_hash != normalized_expected_hash:
        raise ValueError(
            "Empreinte du gold TEST incorrecte : "
            f"attendue={normalized_expected_hash}, obtenue={actual_hash}."
        )

    gold_records = _read_jsonl(gold_path)
    gold_by_query_id: dict[str, dict[str, Any]] = {}

    for record in gold_records:
        query_id = str(record.get("query_id", "")).strip()

        if not query_id:
            raise ValueError(
                "Le gold TEST contient une entrée sans query_id."
            )

        if query_id in gold_by_query_id:
            raise ValueError(
                f"query_id dupliqué dans le gold TEST : {query_id}."
            )

        gold_by_query_id[query_id] = record

    if set(gold_by_query_id) != EXPECTED_TEST_IDS:
        raise ValueError(
            "Le gold TEST doit couvrir exactement Q021 à Q048."
        )

    answerable_records = [
        record
        for record in gold_records
        if record.get("answerable") is True
    ]
    unanswerable_records = [
        record
        for record in gold_records
        if record.get("answerable") is False
    ]

    if len(gold_records) != 28:
        raise ValueError(
            f"28 entrées attendues, trouvé {len(gold_records)}."
        )

    if len(answerable_records) != 24:
        raise ValueError(
            "24 questions answerable attendues, "
            f"trouvé {len(answerable_records)}."
        )

    if len(unanswerable_records) != 4:
        raise ValueError(
            "4 questions unanswerable attendues, "
            f"trouvé {len(unanswerable_records)}."
        )

    for record in gold_records:
        query_id = str(record["query_id"])
        chunk_ids = record.get("gold_chunk_ids")

        if not isinstance(chunk_ids, list):
            raise ValueError(
                f"{query_id}: gold_chunk_ids doit être une liste."
            )

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                f"{query_id}: gold_chunk_ids contient un doublon."
            )

        if record["answerable"] is True:
            if not 1 <= len(chunk_ids) <= 3:
                raise ValueError(
                    f"{query_id}: 1 à 3 chunks gold attendus."
                )
        elif record["answerable"] is False:
            if chunk_ids:
                raise ValueError(
                    f"{query_id}: gold non vide pour une unanswerable."
                )
        else:
            raise ValueError(
                f"{query_id}: answerable doit être un booléen."
            )

    return gold_records, {
        "gold_path": str(gold_path),
        "gold_sha256": actual_hash,
        "entries": len(gold_records),
        "answerable": len(answerable_records),
        "unanswerable": len(unanswerable_records),
    }


def load_test_cases(
    queries_path: Path,
    gold_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Charger les 24 questions TEST répondables."""

    queries = _read_jsonl(queries_path)
    gold_by_query_id = {
        str(record["query_id"]): record
        for record in gold_records
    }
    test_queries: dict[str, dict[str, Any]] = {}

    for query_record in queries:
        if (
            str(query_record.get("split", "")).strip().casefold()
            != "test"
        ):
            continue

        query_id = str(query_record["query_id"])

        if query_id in test_queries:
            raise ValueError(
                f"Question TEST dupliquée : {query_id}."
            )

        test_queries[query_id] = query_record

    if set(test_queries) != set(gold_by_query_id):
        raise ValueError(
            "Les questions TEST et le snapshot gold ne couvrent pas "
            "les mêmes question_id."
        )

    cases: list[dict] = []

    for query_id, query_record in test_queries.items():
        gold_record = gold_by_query_id[query_id]

        if (
            bool(query_record.get("answerable"))
            != bool(gold_record["answerable"])
        ):
            raise ValueError(
                f"{query_id}: incohérence answerable query/gold."
            )

        # Exclure les questions non answerable.
        if gold_record["answerable"] is False:
            continue

        cases.append(
            {
                "query_id": query_id,
                "category": query_record.get(
                    "category",
                    "unknown",
                ),
                "question": query_record["question"],
                "gold_chunk_ids": list(
                    gold_record["gold_chunk_ids"]
                ),
            }
        )

    cases.sort(
        key=lambda case: case["query_id"]
    )

    if len(cases) != 24:
        raise ValueError(
            f"24 cas TEST répondables attendus, trouvé {len(cases)}."
        )

    return cases

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

RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval"
    / "v0.1"
    / "results"
)
SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "test_hybrid_reranked_v2_summary.json"
)
DETAILS_PATH = (
    RESULTS_DIRECTORY
    / "test_hybrid_reranked_v2_per_query.csv"
)
RUN_MANIFEST_PATH = (
    RESULTS_DIRECTORY
    / "test_hybrid_reranked_v2_run_manifest.json"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Évaluation TEST unique avec configuration DEV et gold figés."
        )
    )

    parser.add_argument(
        "--gold-path",
        type=Path,
        required=True,
        help="Chemin explicite du snapshot gold TEST figé.",
    )

    parser.add_argument(
        "--expected-gold-sha256",
        default=EXPECTED_GOLD_SHA256,
    )

    parser.add_argument(
        "--queries-path",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
    )

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

    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Valider les snapshots sans exécuter le retrieval.",
    )

    return parser.parse_args()


def validate_frozen_arguments(
    arguments: argparse.Namespace,
) -> None:
    """Interdire tout écart aux hyperparamètres DEV figés."""

    actual = (
        arguments.candidate_k,
        arguments.dense_candidates,
        arguments.bm25_candidates,
        arguments.top_k,
        arguments.disable_query_expansion,
    )
    expected = (20, 20, 20, 5, False)

    if actual != expected:
        raise ValueError(
            "Les paramètres TEST doivent rester identiques à la "
            "configuration DEV figée : candidate_k=20, "
            "dense=20, bm25=20, top_k=5, expansion activée."
        )


def validate_frozen_dev_snapshot() -> dict[str, str]:
    """Vérifier que les composants actifs égalent le snapshot DEV."""

    component_paths = {
        "retrieval_v2.yaml": (
            ACTIVE_RETRIEVAL_CONFIG_PATH,
            FROZEN_RETRIEVAL_CONFIG_PATH,
        ),
        "reranking.yaml": (
            ACTIVE_RERANKING_CONFIG_PATH,
            FROZEN_RERANKING_CONFIG_PATH,
        ),
        "hybrid.py": (
            ACTIVE_HYBRID_PATH,
            FROZEN_HYBRID_PATH,
        ),
        "reranker.py": (
            ACTIVE_RERANKER_PATH,
            FROZEN_RERANKER_PATH,
        ),
    }
    verified_hashes: dict[str, str] = {}

    for name, (active_path, frozen_path) in component_paths.items():
        expected_hash = EXPECTED_FROZEN_HASHES[name]
        active_hash = sha256_file(active_path)
        frozen_hash = sha256_file(frozen_path)

        if active_hash != expected_hash or frozen_hash != expected_hash:
            raise ValueError(
                f"Composant DEV figé non conforme : {name}. "
                f"attendu={expected_hash}, actif={active_hash}, "
                f"snapshot={frozen_hash}."
            )

        verified_hashes[name] = expected_hash

    return verified_hashes


def create_unique_run_manifest(
    *,
    gold_preflight: dict[str, Any],
    dev_hashes: dict[str, str],
    arguments: argparse.Namespace,
) -> None:
    """Créer atomiquement le verrou empêchant une seconde exécution."""

    existing_outputs = [
        path
        for path in (
            RUN_MANIFEST_PATH,
            SUMMARY_PATH,
            DETAILS_PATH,
        )
        if path.exists()
    ]

    if existing_outputs:
        raise FileExistsError(
            "Une exécution TEST existe déjà; seconde exécution refusée : "
            + ", ".join(str(path) for path in existing_outputs)
        )

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "started",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "single_test_run": True,
        "retuning_allowed": False,
        "gold": gold_preflight,
        "frozen_dev_hashes": dev_hashes,
        "parameters": {
            "candidate_k": arguments.candidate_k,
            "dense_candidates": arguments.dense_candidates,
            "bm25_candidates": arguments.bm25_candidates,
            "top_k": arguments.top_k,
            "query_expansion": True,
        },
    }

    with RUN_MANIFEST_PATH.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(
            json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n"
        )


def complete_unique_run_manifest(
    summary: dict[str, Any],
) -> None:
    """Sceller le manifeste de l'unique exécution terminée."""

    manifest = json.loads(
        RUN_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    manifest.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "summary_path": str(SUMMARY_PATH),
            "summary_sha256": sha256_file(SUMMARY_PATH),
            "details_path": str(DETAILS_PATH),
            "details_sha256": sha256_file(DETAILS_PATH),
            "metrics": summary,
        }
    )
    RUN_MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


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
    validate_frozen_arguments(arguments)

    gold_path = resolve_project_path(
        str(arguments.gold_path)
    )
    queries_path = resolve_project_path(
        str(arguments.queries_path)
    )
    dev_hashes = validate_frozen_dev_snapshot()
    gold_records, gold_preflight = validate_gold_snapshot(
        gold_path,
        expected_sha256=arguments.expected_gold_sha256,
    )
    cases = load_test_cases(
        queries_path,
        gold_records,
    )

    print("\n=== PREFLIGHT TEST FIGÉ ===")
    print(
        f"SHA-256      : {gold_preflight['gold_sha256']}"
    )
    print(f"Entrées      : {gold_preflight['entries']}")
    print(f"Answerable   : {gold_preflight['answerable']}")
    print(f"Unanswerable : {gold_preflight['unanswerable']}")
    print("Configuration: dev_best_v2 vérifiée")

    if arguments.preflight_only:
        print("PREFLIGHT=OK — aucune évaluation exécutée.")
        return

    create_unique_run_manifest(
        gold_preflight=gold_preflight,
        dev_hashes=dev_hashes,
        arguments=arguments,
    )

    # Imports différés : --preflight-only ne charge aucun modèle.
    from phosprocess.reranking.reranker import (
        BGEReranker,
        load_reranking_config,
    )
    from phosprocess.retrieval.bm25 import load_bm25_config
    from phosprocess.retrieval.hybrid import HybridRetriever

    bm25_config = load_bm25_config(
        FROZEN_RETRIEVAL_CONFIG_PATH
    )

    print(
        f"TEST answerable queries: {len(cases)}"
    )

    print("\n=== Loading Hybrid Retriever ===")

    retriever = HybridRetriever(
        dense_index_directory=DENSE_INDEX_DIRECTORY,
        bm25_index_directory=resolve_project_path(
            bm25_config.output_directory
        ),
        embedding_config_path=EMBEDDING_CONFIG_PATH,
        retrieval_config_path=(
            FROZEN_RETRIEVAL_CONFIG_PATH
        ),
    )

    print("\n=== Loading BGE Reranker ===")

    reranking_config = load_reranking_config(
        FROZEN_RERANKING_CONFIG_PATH
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
        "split": "test",
        "single_test_run": True,
        "retuning_allowed": False,
        "frozen_dev_snapshot": str(
            FROZEN_DEV_DIRECTORY
        ),
        "frozen_dev_hashes": dev_hashes,
        "gold_path": str(gold_path),
        "gold_sha256": gold_preflight["gold_sha256"],
        "queries_path": str(queries_path),
        "queries_sha256": sha256_file(queries_path),
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
    print("HYBRID + BGE RERANKER - TEST")
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

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with DETAILS_PATH.open(
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

    complete_unique_run_manifest(summary)

    print("\nSaved:")
    print(f"  {SUMMARY_PATH}")
    print(f"  {DETAILS_PATH}")
    print(f"  {RUN_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
