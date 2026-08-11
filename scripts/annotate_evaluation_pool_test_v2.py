"""Annoter interactivement les paires question-chunk."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from phosprocess.evaluation.annotation import (
    JUDGMENTS_FILENAME,
    JudgmentStore,
    create_excerpt,
    create_judgment,
    format_pages,
    load_annotation_pool,
)
from phosprocess.evaluation.pool_builder import (
    POOL_FILENAME,
    AnnotationPoolItem,
)
from phosprocess.evaluation.schemas import (
    JudgmentStatus,
    RelevanceJudgment,
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


def parse_arguments() -> argparse.Namespace:
    """Lire les options CLI."""

    parser = argparse.ArgumentParser(
        description=("Annoter le pool retrieval avec une pertinence graduée de 0 à 3.")
    )

    parser.add_argument(
        "--assessor-id",
        required=True,
        help="Identifiant de l'annotateur.",
    )

    parser.add_argument(
        "--split",
        choices=("all", "dev", "test"),
        default="all",
    )

    parser.add_argument(
        "--query-id",
        action="append",
        default=None,
        help=("Limiter l'annotation à un query_id. Option répétable."),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=("Limiter le nombre de nouvelles paires à annoter."),
    )

    parser.add_argument(
        "--maximum-characters",
        type=int,
        default=1800,
    )

    parser.add_argument(
        "--show-system-signals",
        action="store_true",
        help=("Afficher les moteurs, rangs et scores. Déconseillé pendant l'annotation."),
    )

    parser.add_argument(
        "--auto-zero-unanswerable",
        action="store_true",
        help=("Attribuer automatiquement 0 aux questions déclarées non répondables."),
    )

    parser.add_argument(
        "--auto-zero-only",
        action="store_true",
        help=("Effectuer uniquement l'auto-annotation des questions non répondables puis quitter."),
    )

    return parser.parse_args()


def select_items(
    items: list[AnnotationPoolItem],
    *,
    split: str,
    query_ids: list[str] | None,
) -> list[AnnotationPoolItem]:
    """Filtrer et ordonner le pool."""

    selected = list(items)

    if split != "all":
        selected = [item for item in selected if item.split == split]

    if query_ids:
        requested_ids = set(query_ids)

        available_ids = {item.query_id for item in items}

        unknown_ids = sorted(requested_ids - available_ids)

        if unknown_ids:
            raise ValueError(f"query_id inconnus : {unknown_ids}")

        selected = [item for item in selected if item.query_id in requested_ids]

    selected.sort(
        key=lambda item: (
            item.query_id,
            item.display_order,
        )
    )

    return selected


def print_scale() -> None:
    """Afficher l'échelle d'annotation."""

    print("\nÉchelle de pertinence :")
    print("  3 = réponse directe et précise")
    print("  2 = réponse pertinente mais partielle")
    print("  1 = contexte utile seulement")
    print("  0 = passage non pertinent")


def print_item(
    item: AnnotationPoolItem,
    *,
    index: int,
    total: int,
    maximum_characters: int,
    show_system_signals: bool,
) -> None:
    """Afficher une paire à juger."""

    print("\n" + "=" * 96)
    print(f"Annotation {index}/{total} | {item.pool_item_id}")
    print("=" * 96)

    print(f"Question  : {item.question}")
    print(f"Catégorie : {item.category}")
    print(f"Split     : {item.split}")
    print(f"Langue    : {item.language}")

    if item.answerable:
        print(f"Réponse de référence : {item.expected_answer}")
    else:
        print("Réponse de référence : [QUESTION NON RÉPONDABLE]")

        if item.query_notes:
            print(f"Justification : {item.query_notes}")

    print("-" * 96)
    print(f"Document : {item.source_file}")
    print(f"Pages    : {format_pages(item.source_pages)}")

    heading = " > ".join(item.heading_path) if item.heading_path else "Section non détectée"

    print(f"Section  : {heading}")
    print(f"Chunk ID : {item.chunk_id}")

    if show_system_signals:
        print(f"Moteurs  : {', '.join(item.retrieved_by)}")

        for system_name, evidence in sorted(item.systems.items()):
            print(f"  {system_name:<9} rang={evidence.rank:<2} score={evidence.score:.8f}")

    print("-" * 96)
    print(
        create_excerpt(
            item.text,
            maximum_characters=maximum_characters,
        )
    )


def prompt_judgment(
    item: AnnotationPoolItem,
) -> tuple[str, int | None, str | None]:
    """Demander une commande à l'annotateur."""

    while True:
        raw_value = input(
            "\nNote [0/1/2/3] | f=texte complet | s=passer | q=quitter | ?=aide : "
        ).strip()

        normalized = raw_value.casefold()

        if normalized == "?":
            print_scale()
            print("\nUne justification peut suivre la note :")
            print("  2 contient la causalité mais pas la conséquence complète")
            continue

        if normalized == "f":
            print("\n--- Texte complet ---")
            print(item.text)
            print("--- Fin du texte ---")
            continue

        if normalized == "s":
            return "skip", None, None

        if normalized == "q":
            return "quit", None, None

        match = re.fullmatch(
            r"([0-3])(?:\s+(.+))?",
            raw_value,
        )

        if match is None:
            print("Commande invalide. Utilise 0, 1, 2, 3, f, s, q ou ?.")
            continue

        relevance = int(match.group(1))
        rationale = match.group(2)

        if not item.answerable and relevance > 0:
            print("Cette question est déclarée non répondable. Sa pertinence doit être 0.")
            continue

        return "save", relevance, rationale


def auto_zero_unanswerable(
    *,
    items: list[AnnotationPoolItem],
    store: JudgmentStore,
    assessor_id: str,
) -> int:
    """Noter automatiquement les questions non répondables."""

    new_judgments: list[RelevanceJudgment] = []

    for item in items:
        if item.answerable:
            continue

        if store.get(item) is not None:
            continue

        rationale = (
            "Question déclarée non répondable par le "
            "benchmark : aucun passage du corpus ne peut "
            "fournir la donnée actuelle demandée."
        )

        new_judgments.append(
            create_judgment(
                item=item,
                relevance=0,
                assessor_id=assessor_id,
                rationale=rationale,
                status=JudgmentStatus.DRAFT,
            )
        )

    if new_judgments:
        store.upsert_many(new_judgments)

    return len(new_judgments)


def main() -> None:
    """Lancer la session interactive."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    arguments = parse_arguments()

    assessor_id = arguments.assessor_id.strip()

    if len(assessor_id) < 2:
        raise ValueError("--assessor-id doit contenir au moins deux caractères.")

    if arguments.auto_zero_only and not arguments.auto_zero_unanswerable:
        raise ValueError("--auto-zero-only nécessite --auto-zero-unanswerable.")

    if arguments.limit is not None and arguments.limit <= 0:
        raise ValueError("--limit doit être strictement positif.")

    if arguments.maximum_characters <= 0:
        raise ValueError("--maximum-characters doit être positif.")

    config = load_evaluation_config(CONFIG_PATH)

    output_directory = resolve_project_path(config.dataset.output_directory)

    pool_path = output_directory / POOL_FILENAME

    judgments_path = output_directory / JUDGMENTS_FILENAME

    all_items = load_annotation_pool(pool_path)

    selected_items = select_items(
        all_items,
        split=arguments.split,
        query_ids=arguments.query_id,
    )

    store = JudgmentStore(
        path=judgments_path,
        pool_items=all_items,
    )

    print("\n=== Annotation PhosProcess Retrieval ===")
    print(f"Pool total       : {len(all_items)}")
    print(f"Sélection        : {len(selected_items)}")
    print(f"Déjà jugées      : {len(store)}")
    print(f"Annotateur       : {assessor_id}")

    if arguments.auto_zero_unanswerable:
        auto_count = auto_zero_unanswerable(
            items=selected_items,
            store=store,
            assessor_id=assessor_id,
        )

        print(f"Questions non répondables auto-notées : {auto_count}")

    if arguments.auto_zero_only:
        print(f"Jugements sauvegardés : {len(store)}")
        return

    pending_items = [item for item in selected_items if store.get(item) is None]

    if arguments.limit is not None:
        pending_items = pending_items[: arguments.limit]

    if not pending_items:
        print("\nAucune paire non jugée dans cette sélection.")
        return

    print_scale()

    skipped = 0
    saved = 0

    try:
        for index, item in enumerate(
            pending_items,
            start=1,
        ):
            print_item(
                item,
                index=index,
                total=len(pending_items),
                maximum_characters=(arguments.maximum_characters),
                show_system_signals=(arguments.show_system_signals),
            )

            action, relevance, rationale = prompt_judgment(item)

            if action == "quit":
                print("\nSession interrompue proprement.")
                break

            if action == "skip":
                skipped += 1
                print("[PASSÉ] La paire restera non jugée.")
                continue

            if relevance is None:
                raise RuntimeError("Pertinence absente lors de la sauvegarde.")

            judgment = create_judgment(
                item=item,
                relevance=relevance,
                assessor_id=assessor_id,
                rationale=rationale,
                status=JudgmentStatus.VERIFIED,
            )

            store.upsert(judgment)
            saved += 1

            progression = 100.0 * len(store) / len(all_items)

            print(f"[SAUVEGARDÉ] pertinence={relevance} | progression globale={progression:.1f}%")

    except KeyboardInterrupt:
        print("\n\nInterruption clavier détectée. Les jugements précédents sont déjà sauvegardés.")

    print("\n=== Fin de session ===")
    print(f"Nouveaux jugements : {saved}")
    print(f"Paires passées     : {skipped}")
    print(f"Total sauvegardé   : {len(store)}")
    print(f"Restantes          : {len(all_items) - len(store)}")
    print(f"Fichier            : {judgments_path}")


if __name__ == "__main__":
    main()
