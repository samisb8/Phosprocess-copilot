"""Inspecter le pool sans biaiser l'annotation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from phosprocess.evaluation.pool_builder import (
    POOL_FILENAME,
    AnnotationPoolItem,
)
from phosprocess.evaluation.schemas import (
    load_evaluation_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVALUATION_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "evaluation.yaml"
)


def resolve_project_path(
    path_value: str,
) -> Path:
    """Résoudre un chemin depuis la racine."""

    path = Path(path_value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def load_pool(
    path: Path,
) -> list[AnnotationPoolItem]:
    """Charger et valider le pool JSONL."""

    if not path.exists():
        raise FileNotFoundError(
            f"Pool introuvable : {path}"
        )

    items: list[AnnotationPoolItem] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as source:
        for line_number, line in enumerate(
            source,
            start=1,
        ):
            if not line.strip():
                raise ValueError(
                    f"Ligne vide : {line_number}"
                )

            try:
                raw_record = json.loads(line)

                item = (
                    AnnotationPoolItem
                    .model_validate(raw_record)
                )
            except Exception as error:
                raise ValueError(
                    "Pool invalide à la ligne "
                    f"{line_number}."
                ) from error

            items.append(item)

    keys = [
        item.pool_item_id
        for item in items
    ]

    if len(keys) != len(set(keys)):
        raise ValueError(
            "Des pool_item_id sont dupliqués."
        )

    return items


def create_excerpt(
    text: str,
    *,
    maximum_characters: int,
) -> str:
    """Créer un extrait lisible."""

    normalized = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if len(normalized) <= maximum_characters:
        return normalized

    shortened = normalized[
        :maximum_characters
    ]

    final_space = shortened.rfind(" ")

    if final_space > 0:
        shortened = shortened[:final_space]

    return shortened.rstrip() + "..."


def format_pages(
    pages: list[int],
) -> str:
    """Compacter une liste de pages."""

    if not pages:
        return ""

    ranges: list[str] = []
    start = pages[0]
    previous = pages[0]

    for page in pages[1:]:
        if page == previous + 1:
            previous = page
            continue

        ranges.append(
            str(start)
            if start == previous
            else f"{start}-{previous}"
        )

        start = page
        previous = page

    ranges.append(
        str(start)
        if start == previous
        else f"{start}-{previous}"
    )

    return ", ".join(ranges)


def parse_arguments() -> argparse.Namespace:
    """Lire les paramètres d'inspection."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspecter le pool d'annotation."
        )
    )

    parser.add_argument(
        "--query-id",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--maximum-characters",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--list-queries",
        action="store_true",
    )

    parser.add_argument(
        "--show-system-signals",
        action="store_true",
        help=(
            "Afficher les rangs et scores. "
            "À éviter pendant l'annotation."
        ),
    )

    parser.add_argument(
        "--show-expected-answer",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    """Afficher les questions ou leurs candidats."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(
            encoding="utf-8"
        )

    arguments = parse_arguments()

    if arguments.limit <= 0:
        raise ValueError(
            "--limit doit être positif."
        )

    if arguments.maximum_characters <= 0:
        raise ValueError(
            "--maximum-characters doit être positif."
        )

    config = load_evaluation_config(
        EVALUATION_CONFIG_PATH
    )

    output_directory = resolve_project_path(
        config.dataset.output_directory
    )

    pool_path = (
        output_directory
        / POOL_FILENAME
    )

    items = load_pool(pool_path)

    grouped: dict[
        str,
        list[AnnotationPoolItem],
    ] = defaultdict(list)

    for item in items:
        grouped[item.query_id].append(
            item
        )

    for query_items in grouped.values():
        query_items.sort(
            key=lambda item: (
                item.display_order
            )
        )

    if arguments.list_queries:
        print("\n=== Questions présentes dans le pool ===")

        for query_id in sorted(grouped):
            first = grouped[query_id][0]

            print(
                f"{query_id} | "
                f"{first.split:<4} | "
                f"{first.language:<5} | "
                f"{first.category:<30} | "
                f"{len(grouped[query_id]):>2} candidats"
            )

        print(
            f"\nTotal : {len(grouped)} questions, "
            f"{len(items)} paires"
        )

        return

    if not grouped:
        raise ValueError(
            "Le pool est vide."
        )

    query_id = (
        arguments.query_id
        or sorted(grouped)[0]
    )

    if query_id not in grouped:
        raise ValueError(
            f"{query_id} absent du pool."
        )

    query_items = grouped[query_id]
    first = query_items[0]

    print("\n=== Inspection aveuglée du pool ===")
    print(f"Query ID       : {first.query_id}")
    print(f"Question       : {first.question}")
    print(f"Split          : {first.split}")
    print(f"Langue         : {first.language}")
    print(f"Catégorie      : {first.category}")
    print(f"Difficulté     : {first.difficulty}")
    print(f"Candidats      : {len(query_items)}")

    if arguments.show_expected_answer:
        print(
            "Réponse attendue: "
            f"{first.expected_answer}"
        )
        print(
            "Répondable      : "
            f"{first.answerable}"
        )

    selected_items = query_items[
        : arguments.limit
    ]

    for item in selected_items:
        print("\n" + "=" * 92)
        print(
            f"Candidat #{item.display_order} | "
            f"{item.pool_item_id}"
        )
        print(
            f"Document : {item.source_file}"
        )
        print(
            "Pages    : "
            f"{format_pages(item.source_pages)}"
        )

        heading = (
            " > ".join(item.heading_path)
            if item.heading_path
            else "Section non détectée"
        )

        print(f"Section  : {heading}")
        print(f"Chunk ID : {item.chunk_id}")

        if arguments.show_system_signals:
            print(
                "Moteurs  : "
                f"{', '.join(item.retrieved_by)}"
            )

            for system_name, evidence in sorted(
                item.systems.items()
            ):
                print(
                    f"  {system_name:<9} "
                    f"rang={evidence.rank:<2} "
                    f"score={evidence.score:.8f}"
                )

        print("-" * 92)
        print(
            create_excerpt(
                item.text,
                maximum_characters=(
                    arguments.maximum_characters
                ),
            )
        )

    print("\n" + "=" * 92)
    print(
        "Les signaux des moteurs sont cachés par défaut "
        "pour limiter le biais d'annotation."
    )


if __name__ == "__main__":
    main()
