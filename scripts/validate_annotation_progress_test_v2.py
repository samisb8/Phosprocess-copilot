"""Valider et résumer la progression des annotations."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from phosprocess.evaluation.annotation import (
    JUDGMENTS_FILENAME,
    PROGRESS_FILENAME,
    load_annotation_pool,
    load_judgments,
    pool_item_key,
)
from phosprocess.evaluation.pool_builder import (
    POOL_FILENAME,
)
from phosprocess.evaluation.schemas import (
    load_evaluation_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "evaluation_test_v2.yaml"


def resolve_project_path(
    path_value: str,
) -> Path:
    """Résoudre un chemin relativement au projet."""

    path = Path(path_value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def atomic_write_json(
    data: dict[str, Any],
    path: Path,
) -> None:
    """Écrire le rapport de façon atomique."""

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    temporary_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(path)


def parse_arguments() -> argparse.Namespace:
    """Lire les options."""

    parser = argparse.ArgumentParser(description=("Valider la progression de l'annotation."))

    parser.add_argument(
        "--strict-complete",
        action="store_true",
        help=("Retourner une erreur si toutes les paires ne sont pas jugées."),
    )

    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="Afficher les premières paires restantes.",
    )

    return parser.parse_args()


def main() -> None:
    """Construire le rapport de progression."""

    arguments = parse_arguments()

    config = load_evaluation_config(CONFIG_PATH)

    output_directory = resolve_project_path(config.dataset.output_directory)

    pool_path = output_directory / POOL_FILENAME

    judgments_path = output_directory / JUDGMENTS_FILENAME

    progress_path = output_directory / PROGRESS_FILENAME

    pool_items = load_annotation_pool(pool_path)

    judgments = load_judgments(judgments_path)

    pool_map = {item.pool_item_id: item for item in pool_items}

    judgment_map = {
        pool_item_key(
            judgment.query_id,
            judgment.chunk_id,
        ): judgment
        for judgment in judgments
    }

    errors: list[str] = []
    warnings: list[str] = []

    unknown_judgments = sorted(set(judgment_map) - set(pool_map))

    if unknown_judgments:
        errors.append(
            f"Des jugements ne correspondent à aucune paire du pool : {unknown_judgments[:10]}"
        )

    for identifier, judgment in judgment_map.items():
        item = pool_map.get(identifier)

        if item is None:
            continue

        if not item.answerable and judgment.relevance > 0:
            errors.append(f"Question non répondable avec une pertinence positive : {identifier}.")

    missing_identifiers = sorted(
        set(pool_map) - set(judgment_map),
        key=lambda identifier: (
            pool_map[identifier].query_id,
            pool_map[identifier].display_order,
        ),
    )

    pool_by_query = defaultdict(list)
    judgments_by_query = defaultdict(list)

    for item in pool_items:
        pool_by_query[item.query_id].append(item)

    for identifier, judgment in judgment_map.items():
        if identifier not in pool_map:
            continue

        judgments_by_query[judgment.query_id].append(judgment)

    completed_queries: list[str] = []
    incomplete_queries: list[str] = []

    for query_id, query_items in pool_by_query.items():
        query_judgments = judgments_by_query[query_id]

        if len(query_judgments) == len(query_items):
            completed_queries.append(query_id)

            first_item = query_items[0]

            if first_item.answerable:
                strong_relevance_count = sum(
                    judgment.relevance >= 2 for judgment in query_judgments
                )

                if strong_relevance_count == 0:
                    warnings.append(
                        f"{query_id} est complète, mais aucun chunk n'a une pertinence >= 2."
                    )
        else:
            incomplete_queries.append(query_id)

    total_pairs = len(pool_items)
    judged_pairs = len(judgment_map)
    remaining_pairs = len(missing_identifiers)

    progression = 100.0 * judged_pairs / total_pairs if total_pairs else 0.0

    if errors:
        status = "invalid"
    elif judged_pairs == 0:
        status = "not_started"
    elif remaining_pairs == 0:
        status = "complete"
    else:
        status = "in_progress"

    relevance_counts = Counter(judgment.relevance for judgment in judgments)

    relevance_distribution = {
        str(relevance): relevance_counts.get(
            relevance,
            0,
        )
        for relevance in range(4)
    }

    judgment_status_counts = Counter(judgment.status.value for judgment in judgments)

    judged_by_split = Counter()
    judged_by_category = Counter()
    judged_by_language = Counter()

    for identifier in judgment_map:
        item = pool_map.get(identifier)

        if item is None:
            continue

        judged_by_split[item.split] += 1
        judged_by_category[item.category] += 1
        judged_by_language[item.language] += 1

    report: dict[str, Any] = {
        "validated_at_utc": (datetime.now(UTC).isoformat()),
        "status": status,
        "counts": {
            "pool_pairs": total_pairs,
            "judged_pairs": judged_pairs,
            "remaining_pairs": remaining_pairs,
            "progress_percent": round(
                progression,
                2,
            ),
            "total_queries": len(pool_by_query),
            "completed_queries": len(completed_queries),
            "incomplete_queries": len(incomplete_queries),
        },
        "relevance_distribution": (relevance_distribution),
        "judgment_statuses": dict(judgment_status_counts),
        "judged_by_split": dict(judged_by_split),
        "judged_by_category": dict(judged_by_category),
        "judged_by_language": dict(judged_by_language),
        "completed_query_ids": sorted(completed_queries),
        "incomplete_query_ids": sorted(incomplete_queries),
        "errors": errors,
        "warnings": warnings,
    }

    atomic_write_json(
        report,
        progress_path,
    )

    print("\n=== Progression des annotations ===")
    print(f"Statut             : {status}")
    print(f"Paires du pool     : {total_pairs}")
    print(f"Paires jugées      : {judged_pairs}")
    print(f"Paires restantes   : {remaining_pairs}")
    print(f"Progression        : {progression:.2f}%")
    print(f"Questions complètes: {len(completed_queries)}/{len(pool_by_query)}")
    print(f"Répartition notes  : {relevance_distribution}")
    print(f"Rapport            : {progress_path}")

    if warnings:
        print("\nAvertissements :")

        for warning in warnings:
            print(f"- {warning}")

    if arguments.show_missing and missing_identifiers:
        print("\nPremières paires restantes :")

        for identifier in missing_identifiers[:20]:
            item = pool_map[identifier]

            print(f"- {identifier} | {item.category} | {item.source_file}")

    if errors:
        print("\nErreurs :")

        for error in errors:
            print(f"- {error}")

        raise SystemExit(1)

    if arguments.strict_complete and remaining_pairs > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
