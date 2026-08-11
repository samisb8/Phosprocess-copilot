"""Analyze the frozen v2 DEV baseline without accessing TEST artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval"
    / "v0.1"
)
FROZEN_DEV_DIRECTORY = (
    EVALUATION_ROOT
    / "frozen"
    / "dev_best_v2"
)
DEFAULT_SUMMARY_PATH = (
    FROZEN_DEV_DIRECTORY
    / "dev_hybrid_reranked_v2_summary.json"
)
DEFAULT_PER_QUERY_PATH = (
    FROZEN_DEV_DIRECTORY
    / "dev_hybrid_reranked_v2_per_query.csv"
)
SNAPSHOT_HASH_PATH = FROZEN_DEV_DIRECTORY / "sha256.csv"
V3_ANALYSIS_DIRECTORY = EVALUATION_ROOT / "v3" / "analysis"
DEFAULT_OUTPUT_PATH = (
    V3_ANALYSIS_DIRECTORY
    / "dev_v2_baseline_error_analysis.json"
)

EXPECTED_QUERY_IDS = {
    f"Q{number:03d}"
    for number in range(1, 17)
}
EXPECTED_ROW_COUNT = 16
EXPECTED_CANDIDATE_COUNT = 20
EXPECTED_RERANKED_COUNT = 5


def sha256_file(path: Path) -> str:
    """Return an uppercase SHA-256 digest."""

    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest().upper()


def ensure_path_within(
    path: Path,
    directory: Path,
    *,
    label: str,
) -> Path:
    """Resolve a path and require it to stay under an allowed directory."""

    resolved_path = path.resolve()
    resolved_directory = directory.resolve()

    try:
        resolved_path.relative_to(resolved_directory)
    except ValueError as error:
        raise ValueError(
            f"{label} doit rester sous {resolved_directory}."
        ) from error

    return resolved_path


def parse_pipe_ids(value: str) -> list[str]:
    """Parse a pipe-separated list while rejecting empty or duplicate IDs."""

    identifiers = [
        item.strip()
        for item in value.split("|")
        if item.strip()
    ]

    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Une liste de chunk IDs contient un doublon.")

    return identifiers


def parse_optional_rank(value: str) -> int | None:
    """Parse an optional positive rank."""

    stripped = value.strip()

    if not stripped:
        return None

    rank = int(stripped)

    if rank <= 0:
        raise ValueError("Un rang doit être strictement positif.")

    return rank


def parse_binary(value: str, *, field_name: str) -> int:
    """Parse a CSV binary flag."""

    parsed = int(value)

    if parsed not in {0, 1}:
        raise ValueError(f"{field_name} doit valoir 0 ou 1.")

    return parsed


def load_snapshot_hashes(path: Path) -> dict[str, str]:
    """Load the frozen snapshot digest registry, keyed by file name."""

    hashes: dict[str, str] = {}

    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            raw_path = row["Path"].replace("\\", "/")
            file_name = raw_path.rsplit("/", maxsplit=1)[-1]
            hashes[file_name] = row["Hash"].upper()

    return hashes


def verify_frozen_inputs(
    summary_path: Path,
    per_query_path: Path,
) -> dict[str, str]:
    """Require DEV-only paths and verify their frozen SHA-256 digests."""

    summary_path = ensure_path_within(
        summary_path,
        FROZEN_DEV_DIRECTORY,
        label="Le résumé DEV",
    )
    per_query_path = ensure_path_within(
        per_query_path,
        FROZEN_DEV_DIRECTORY,
        label="Le détail DEV",
    )

    expected_hashes = load_snapshot_hashes(SNAPSHOT_HASH_PATH)
    actual_hashes = {
        summary_path.name: sha256_file(summary_path),
        per_query_path.name: sha256_file(per_query_path),
    }

    for file_name, actual_hash in actual_hashes.items():
        expected_hash = expected_hashes.get(file_name)

        if expected_hash is None:
            raise ValueError(
                f"{file_name} n'est pas enregistré dans le snapshot DEV."
            )

        if actual_hash != expected_hash:
            raise ValueError(
                f"Le snapshot DEV a changé pour {file_name}: "
                f"{actual_hash} != {expected_hash}."
            )

    return actual_hashes


def validate_row(row: dict[str, str]) -> dict[str, Any]:
    """Validate and normalize one frozen per-query DEV result."""

    query_id = row["query_id"].strip()
    gold_ids = parse_pipe_ids(row["gold_chunk_ids"])
    candidate_ids = parse_pipe_ids(row["candidate_chunk_ids"])
    reranked_ids = parse_pipe_ids(row["reranked_chunk_ids"])
    candidate_rank = parse_optional_rank(row["candidate_gold_rank"])
    final_rank = parse_optional_rank(row["final_gold_rank"])
    candidate_hit = parse_binary(
        row["candidate_hit"],
        field_name="candidate_hit",
    )
    hit_at_1 = parse_binary(row["hit_at_1"], field_name="hit_at_1")
    hit_at_5 = parse_binary(row["hit_at_5"], field_name="hit_at_5")

    if not gold_ids:
        raise ValueError(f"{query_id}: le gold DEV est vide.")

    if len(candidate_ids) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError(
            f"{query_id}: {len(candidate_ids)} candidats au lieu de "
            f"{EXPECTED_CANDIDATE_COUNT}."
        )

    if len(reranked_ids) != EXPECTED_RERANKED_COUNT:
        raise ValueError(
            f"{query_id}: {len(reranked_ids)} résultats rerankés au lieu de "
            f"{EXPECTED_RERANKED_COUNT}."
        )

    expected_candidate_rank = next(
        (
            rank
            for rank, chunk_id in enumerate(candidate_ids, start=1)
            if chunk_id in gold_ids
        ),
        None,
    )
    expected_final_rank = next(
        (
            rank
            for rank, chunk_id in enumerate(reranked_ids, start=1)
            if chunk_id in gold_ids
        ),
        None,
    )

    if candidate_rank != expected_candidate_rank:
        raise ValueError(f"{query_id}: candidate_gold_rank incohérent.")

    if final_rank != expected_final_rank:
        raise ValueError(f"{query_id}: final_gold_rank incohérent.")

    if candidate_hit != int(candidate_rank is not None):
        raise ValueError(f"{query_id}: candidate_hit incohérent.")

    if hit_at_1 != int(final_rank == 1):
        raise ValueError(f"{query_id}: hit_at_1 incohérent.")

    if hit_at_5 != int(final_rank is not None):
        raise ValueError(f"{query_id}: hit_at_5 incohérent.")

    expected_reciprocal_rank = (
        1.0 / final_rank
        if final_rank is not None
        else 0.0
    )
    reciprocal_rank = float(row["reciprocal_rank_at_5"])

    if abs(reciprocal_rank - expected_reciprocal_rank) > 1e-12:
        raise ValueError(
            f"{query_id}: reciprocal_rank_at_5 incohérent."
        )

    expected_evidence_recall = (
        len(set(gold_ids).intersection(reranked_ids))
        / len(gold_ids)
    )
    evidence_recall = float(row["evidence_recall_at_5"])

    if abs(evidence_recall - expected_evidence_recall) > 1e-12:
        raise ValueError(
            f"{query_id}: evidence_recall_at_5 incohérent."
        )

    if candidate_rank is None:
        outcome = "candidate_miss"
    elif final_rank is None:
        outcome = "reranker_miss"
    elif final_rank == 1:
        outcome = "rank_1"
    else:
        outcome = "top_5_below_rank_1"

    return {
        "query_id": query_id,
        "category": row["category"].strip(),
        "candidate_gold_rank": candidate_rank,
        "final_gold_rank": final_rank,
        "candidate_hit": candidate_hit,
        "hit_at_1": hit_at_1,
        "hit_at_5": hit_at_5,
        "reciprocal_rank_at_5": reciprocal_rank,
        "evidence_recall_at_5": evidence_recall,
        "outcome": outcome,
    }


def validate_and_normalize_rows(
    raw_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Validate that the input is exactly the frozen 16-question DEV split."""

    if len(raw_rows) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Le détail DEV doit contenir exactement {EXPECTED_ROW_COUNT} lignes."
        )

    rows = [validate_row(row) for row in raw_rows]
    query_ids = [row["query_id"] for row in rows]

    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Le détail DEV contient un query_id dupliqué.")

    if set(query_ids) != EXPECTED_QUERY_IDS:
        unexpected = sorted(set(query_ids) - EXPECTED_QUERY_IDS)
        missing = sorted(EXPECTED_QUERY_IDS - set(query_ids))
        raise ValueError(
            "Le détail n'est pas le split DEV Q001-Q016. "
            f"Manquants={missing}, inattendus={unexpected}."
        )

    return sorted(rows, key=lambda row: row["query_id"])


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Recompute official retrieval metrics from normalized DEV rows."""

    count = len(rows)

    return {
        "candidate_recall": (
            sum(row["candidate_hit"] for row in rows) / count
        ),
        "hit_at_1": sum(row["hit_at_1"] for row in rows) / count,
        "hit_at_5": sum(row["hit_at_5"] for row in rows) / count,
        "mrr_at_5": (
            sum(row["reciprocal_rank_at_5"] for row in rows) / count
        ),
        "evidence_recall_at_5": (
            sum(row["evidence_recall_at_5"] for row in rows) / count
        ),
    }


def validate_summary(
    summary: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    """Require a DEV summary consistent with the per-query records."""

    if summary.get("split") != "dev":
        raise ValueError("Le résumé fourni n'appartient pas au split DEV.")

    expected_parameters = {
        "candidate_k": 20,
        "dense_candidates": 20,
        "bm25_candidates": 20,
        "query_expansion": True,
        "top_k": 5,
    }

    for key, expected_value in expected_parameters.items():
        if summary.get(key) != expected_value:
            raise ValueError(
                f"Paramètre DEV v2 inattendu pour {key}: "
                f"{summary.get(key)!r}."
            )

    for key, computed_value in metrics.items():
        recorded_value = float(summary[key])

        if abs(recorded_value - computed_value) > 1e-12:
            raise ValueError(
                f"Métrique DEV incohérente pour {key}: "
                f"{recorded_value} != {computed_value}."
            )


def build_category_summary(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Aggregate DEV-only metrics by category."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[row["category"]].append(row)

    return {
        category: {
            "queries": len(category_rows),
            "candidate_recall": (
                sum(row["candidate_hit"] for row in category_rows)
                / len(category_rows)
            ),
            "hit_at_1": (
                sum(row["hit_at_1"] for row in category_rows)
                / len(category_rows)
            ),
            "hit_at_5": (
                sum(row["hit_at_5"] for row in category_rows)
                / len(category_rows)
            ),
            "mrr_at_5": (
                sum(
                    row["reciprocal_rank_at_5"]
                    for row in category_rows
                )
                / len(category_rows)
            ),
        }
        for category, category_rows in sorted(grouped.items())
    }


def build_analysis(
    *,
    rows: list[dict[str, Any]],
    metrics: dict[str, float],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    """Build a deterministic DEV-only v3 starting-point report."""

    outcomes: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        outcomes[row["outcome"]].append(row["query_id"])

    return {
        "analysis_version": "v3_dev_baseline_analysis_1",
        "purpose": "v3_development_on_dev_only",
        "data_policy": {
            "development_split": "dev",
            "test_data_used": False,
            "test_v2_allowed_for_tuning": False,
            "independent_future_test_required": True,
        },
        "source_snapshot": {
            "name": "dev_best_v2",
            "directory": str(
                FROZEN_DEV_DIRECTORY.relative_to(PROJECT_ROOT)
            ).replace("\\", "/"),
            "sha256": dict(sorted(source_hashes.items())),
        },
        "baseline_metrics_recomputed": metrics,
        "outcomes": {
            key: sorted(query_ids)
            for key, query_ids in sorted(outcomes.items())
        },
        "category_metrics": build_category_summary(rows),
        "query_diagnostics": rows,
        "dev_only_priorities": [
            {
                "priority": 1,
                "evidence": "Q015 est candidat au rang 9 puis absent du top 5.",
                "hypothesis": (
                    "Étudier sur DEV une fusion tardive qui évite que le "
                    "reranker élimine une preuve lexicale explicite."
                ),
            },
            {
                "priority": 2,
                "evidence": (
                    "Q001, Q006, Q007 et Q011 sont trouvées dans le top 5 "
                    "mais pas au rang 1."
                ),
                "hypothesis": (
                    "Mesurer sur DEV la calibration entre rang hybride, "
                    "score lexical et score du reranker."
                ),
            },
        ],
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically with a Windows PermissionError fallback."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    temporary_path.write_text(serialized, encoding="utf-8")

    try:
        os.replace(temporary_path, path)
    except PermissionError:
        shutil.copyfile(temporary_path, path)
        temporary_path.unlink()


def parse_arguments() -> argparse.Namespace:
    """Parse DEV-only analysis paths."""

    parser = argparse.ArgumentParser(
        description=(
            "Analyse la baseline v2 sur DEV uniquement et produit un "
            "point de départ v3 déterministe."
        )
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
    )
    parser.add_argument(
        "--per-query-path",
        type=Path,
        default=DEFAULT_PER_QUERY_PATH,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    return parser.parse_args()


def main() -> None:
    """Validate the frozen DEV input and write the v3 analysis."""

    arguments = parse_arguments()
    summary_path = ensure_path_within(
        arguments.summary_path,
        FROZEN_DEV_DIRECTORY,
        label="Le résumé DEV",
    )
    per_query_path = ensure_path_within(
        arguments.per_query_path,
        FROZEN_DEV_DIRECTORY,
        label="Le détail DEV",
    )
    output_path = ensure_path_within(
        arguments.output_path,
        V3_ANALYSIS_DIRECTORY,
        label="La sortie d'analyse v3",
    )
    source_hashes = verify_frozen_inputs(summary_path, per_query_path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    with per_query_path.open(encoding="utf-8-sig", newline="") as stream:
        raw_rows = list(csv.DictReader(stream))

    rows = validate_and_normalize_rows(raw_rows)
    metrics = compute_metrics(rows)
    validate_summary(summary, metrics)
    analysis = build_analysis(
        rows=rows,
        metrics=metrics,
        source_hashes=source_hashes,
    )
    atomic_write_json(output_path, analysis)

    print("Analyse DEV-only v3 créée.")
    print(f"Entrées DEV: {len(rows)}")
    print(f"Candidate Recall@20: {metrics['candidate_recall']:.4f}")
    print(f"Hit@1: {metrics['hit_at_1']:.4f}")
    print(f"Hit@5: {metrics['hit_at_5']:.4f}")
    print(
        "Échecs reranker: "
        + ", ".join(analysis["outcomes"].get("reranker_miss", []))
    )
    print(f"TEST utilisé pour le tuning: {analysis['data_policy']['test_data_used']}")
    print(f"Sortie: {output_path}")


if __name__ == "__main__":
    main()
