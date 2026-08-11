"""Adjudication humaine assistée des gold evidence TEST."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval"
    / "v0.1"
    / "test_pool_v2"
)

DEFAULT_QUERIES = DEFAULT_DIRECTORY / "queries.jsonl"
DEFAULT_POOL = DEFAULT_DIRECTORY / "annotation_pool.jsonl"
DEFAULT_DRAFTS = DEFAULT_DIRECTORY / "gold_evidence_drafts.jsonl"
DEFAULT_OUTPUT = DEFAULT_DIRECTORY / "gold_evidence_test.jsonl"

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")
NUMBER_PATTERN = re.compile(
    r"\d+(?:[.,]\d+)?(?::\d+(?:[.,]\d+)?)?%?"
)

STOPWORDS = {
    # Français
    "afin",
    "ainsi",
    "avec",
    "dans",
    "des",
    "elle",
    "elles",
    "entre",
    "est",
    "etre",
    "leur",
    "leurs",
    "mais",
    "pour",
    "pourquoi",
    "quel",
    "quelle",
    "quelles",
    "quels",
    "qui",
    "sont",
    "sur",
    "une",
    "vers",
    # Anglais
    "about",
    "after",
    "before",
    "between",
    "does",
    "from",
    "into",
    "than",
    "that",
    "their",
    "these",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Lire un JSONL en ignorant les lignes vides."""

    if not path.exists():
        return []

    records: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        path.read_text(
            encoding="utf-8-sig"
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"JSON invalide dans {path}, ligne "
                f"{line_number}: {error}"
            ) from error

        if not isinstance(record, dict):
            raise ValueError(
                f"Objet JSON non valide dans {path}, "
                f"ligne {line_number}."
            )

        records.append(record)

    return records


def write_jsonl_atomic(
    records: list[dict[str, Any]],
    path: Path,
) -> None:
    """Sauvegarder un JSONL avec compatibilit? Windows."""

    path = path.resolve()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists() and path.is_dir():
        raise IsADirectoryError(
            f"Le chemin de sortie est un dossier : {path}"
        )

    payload = "".join(
        json.dumps(
            record,
            ensure_ascii=False,
        )
        + "\n"
        for record in records
    )

    temporary_path = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        try:
            # M?thode atomique normale.
            os.replace(
                temporary_path,
                path,
            )

        except PermissionError:
            # Certains dossiers Windows autorisent
            # l'?criture mais bloquent le renommage.
            with path.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

    finally:
        temporary_path.unlink(
            missing_ok=True
        )

def normalize_text(text: str) -> str:
    """Normaliser un texte pour les comparaisons."""

    normalized = html.unescape(str(text))

    normalized = HTML_TAG_PATTERN.sub(
        " ",
        normalized,
    )

    normalized = unicodedata.normalize(
        "NFKD",
        normalized,
    )

    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.combining(
            character
        )
    )

    normalized = normalized.casefold()

    return WHITESPACE_PATTERN.sub(
        " ",
        normalized,
    ).strip()


def clean_display_text(text: str) -> str:
    """Nettoyer un passage pour l'affichage."""

    cleaned = html.unescape(str(text))

    cleaned = re.sub(
        r"<!--.*?-->",
        " ",
        cleaned,
        flags=re.DOTALL,
    )

    cleaned = HTML_TAG_PATTERN.sub(
        " ",
        cleaned,
    )

    return WHITESPACE_PATTERN.sub(
        " ",
        cleaned,
    ).strip()


def informative_tokens(text: str) -> set[str]:
    """Extraire les termes informatifs."""

    normalized = normalize_text(text)

    tokens = set(
        re.findall(
            r"[a-z0-9]+",
            normalized,
        )
    )

    return {
        token
        for token in tokens
        if len(token) >= 3
        and token not in STOPWORDS
    }


def numeric_expressions(text: str) -> set[str]:
    """Extraire les expressions numériques."""

    return {
        value.replace(",", ".")
        for value in NUMBER_PATTERN.findall(
            normalize_text(text)
        )
    }


def get_chunk_id(item: dict[str, Any]) -> str:
    value = (
        item.get("chunk_id")
        or item.get("document_chunk_id")
        or ""
    )

    return str(value).strip()


def get_chunk_text(item: dict[str, Any]) -> str:
    value = (
        item.get("text")
        or item.get("chunk_text")
        or item.get("passage")
        or item.get("content")
        or ""
    )

    return str(value)


def get_source_file(item: dict[str, Any]) -> str:
    value = (
        item.get("source_file")
        or item.get("document")
        or item.get("document_name")
        or "document inconnu"
    )

    return str(value)


def get_pages(item: dict[str, Any]) -> str:
    pages = (
        item.get("source_pages")
        or item.get("pages")
    )

    if isinstance(pages, list):
        return ", ".join(
            str(page)
            for page in pages
        )

    if pages in (None, "", []):
        return "?"

    return str(pages)


def get_heading(item: dict[str, Any]) -> str:
    heading = (
        item.get("heading_path")
        or item.get("section")
    )

    if isinstance(heading, list):
        values = [
            str(value).strip()
            for value in heading
            if str(value).strip()
        ]

        return (
            " > ".join(values)
            if values
            else "section inconnue"
        )

    if not heading:
        return "section inconnue"

    return str(heading)


def reference_document_match(
    item: dict[str, Any],
    reference_documents: list[str],
) -> bool:
    """Vérifier si le chunk vient d'un document attendu."""

    source = normalize_text(
        Path(get_source_file(item)).stem
    )

    return any(
        normalize_text(reference_document)
        in source
        or source
        in normalize_text(reference_document)
        for reference_document
        in reference_documents
    )


def original_display_order(
    item: dict[str, Any],
) -> int:
    value = item.get(
        "display_order",
        10_000,
    )

    try:
        return int(value)
    except (TypeError, ValueError):
        return 10_000


def candidate_score(
    item: dict[str, Any],
    *,
    question: dict[str, Any],
    draft: dict[str, Any] | None,
) -> tuple[float, list[str]]:
    """Calculer une priorité d'affichage, pas une pertinence gold."""

    chunk_id = get_chunk_id(item)
    passage = get_chunk_text(item)

    answer = str(
        question.get(
            "expected_answer",
            "",
        )
    )

    question_text = str(
        question.get(
            "question",
            "",
        )
    )

    reference_documents = [
        str(value)
        for value in question.get(
            "reference_documents",
            [],
        )
    ]

    answer_tokens = informative_tokens(answer)
    question_tokens = informative_tokens(
        question_text
    )
    passage_tokens = informative_tokens(
        passage
    )

    answer_overlap = len(
        answer_tokens.intersection(
            passage_tokens
        )
    )

    question_overlap = len(
        question_tokens.intersection(
            passage_tokens
        )
    )

    answer_numbers = numeric_expressions(
        answer
    )

    passage_numbers = numeric_expressions(
        passage
    )

    numeric_matches = sorted(
        answer_numbers.intersection(
            passage_numbers
        )
    )

    proposed_ids: set[str] = set()
    shortlist_ids: set[str] = set()

    if draft is not None:
        proposed_ids = {
            str(value)
            for value in draft.get(
                "gold_chunk_ids",
                [],
            )
        }

        shortlist_ids = {
            str(value)
            for value in draft.get(
                "shortlist_ids",
                [],
            )
        }

    score = 0.0
    reasons: list[str] = []

    if chunk_id in proposed_ids:
        score += 100.0
        reasons.append("proposition Qwen")

    if chunk_id in shortlist_ids:
        score += 25.0
        reasons.append("shortlist Qwen")

    if reference_document_match(
        item,
        reference_documents,
    ):
        score += 40.0
        reasons.append("document de référence")

    if numeric_matches:
        score += 20.0 * len(
            numeric_matches
        )

        reasons.append(
            "valeur(s): "
            + ", ".join(numeric_matches)
        )

    if answer_overlap:
        score += min(
            25.0,
            answer_overlap * 2.5,
        )

        reasons.append(
            f"{answer_overlap} terme(s) réponse"
        )

    if question_overlap:
        score += min(
            10.0,
            question_overlap,
        )

    display_order = original_display_order(
        item
    )

    score += max(
        0.0,
        10.0 - display_order * 0.15,
    )

    return score, reasons


def rank_candidates(
    candidates: list[dict[str, Any]],
    *,
    question: dict[str, Any],
    draft: dict[str, Any] | None,
) -> list[tuple[dict[str, Any], float, list[str]]]:
    """Ordonner les candidats pour l'adjudication."""

    scored = []

    for candidate in candidates:
        score, reasons = candidate_score(
            candidate,
            question=question,
            draft=draft,
        )

        scored.append(
            (
                candidate,
                score,
                reasons,
            )
        )

    scored.sort(
        key=lambda value: (
            -value[1],
            original_display_order(
                value[0]
            ),
            get_chunk_id(value[0]),
        )
    )

    return scored


def display_candidate(
    item: dict[str, Any],
    *,
    index: int,
    score: float,
    reasons: list[str],
    maximum_characters: int,
) -> None:
    """Afficher un candidat avec ses vraies métadonnées."""

    passage = clean_display_text(
        get_chunk_text(item)
    )

    if len(passage) > maximum_characters:
        passage = (
            passage[:maximum_characters]
            .rstrip()
            + "…"
        )

    print("\n" + "-" * 100)
    print(
        f"[{index:02d}] "
        f"{get_chunk_id(item)}"
    )
    print(
        "Document :",
        get_source_file(item),
    )
    print(
        "Pages    :",
        get_pages(item),
    )
    print(
        "Section  :",
        get_heading(item),
    )

    if reasons:
        print(
            "Indices  :",
            " | ".join(reasons),
        )

    print(f"Priorité : {score:.2f}")
    print("Extrait  :", passage)


def save_record(
    records_by_id: dict[str, dict[str, Any]],
    *,
    question: dict[str, Any],
    gold_chunk_ids: list[str],
    assessor_id: str,
    output_path: Path,
) -> None:
    """Sauvegarder immédiatement une adjudication."""

    query_id = str(
        question["query_id"]
    )

    records_by_id[query_id] = {
        "query_id": query_id,
        "split": str(
            question.get(
                "split",
                "test",
            )
        ),
        "category": str(
            question.get(
                "category",
                "unknown",
            )
        ),
        "answerable": bool(
            question.get(
                "answerable",
                True,
            )
        ),
        "gold_chunk_ids": gold_chunk_ids,
        "status": "verified",
        "assessor_id": assessor_id,
        "updated_at": datetime.now(
            UTC
        ).isoformat(),
    }

    ordered_records = [
        records_by_id[key]
        for key in sorted(
            records_by_id
        )
    ]

    write_jsonl_atomic(
        ordered_records,
        output_path,
    )


def parse_selection(
    command: str,
    *,
    maximum_index: int,
) -> list[int]:
    """Lire une sélection comme 1 ou 1,3."""

    values = [
        value.strip()
        for value in command.split(",")
        if value.strip()
    ]

    if not values:
        raise ValueError(
            "Sélection vide."
        )

    indexes: list[int] = []

    for value in values:
        if not value.isdigit():
            raise ValueError(
                f"Index invalide : {value}"
            )

        index = int(value)

        if index < 1 or index > maximum_index:
            raise ValueError(
                f"Index hors limites : {index}"
            )

        if index not in indexes:
            indexes.append(index)

    if len(indexes) > 3:
        raise ValueError(
            "Maximum 3 chunks gold."
        )

    return indexes


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adjudication humaine assistée des "
            "gold evidence TEST."
        )
    )

    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES,
    )

    parser.add_argument(
        "--pool",
        type=Path,
        default=DEFAULT_POOL,
    )

    parser.add_argument(
        "--drafts",
        type=Path,
        default=DEFAULT_DRAFTS,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--assessor-id",
        required=True,
    )

    parser.add_argument(
        "--query-id",
        action="append",
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--maximum-characters",
        type=int,
        default=1400,
    )

    parser.add_argument(
        "--redo",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(
            encoding="utf-8"
        )

    arguments = parse_arguments()

    if arguments.max_candidates <= 0:
        raise ValueError(
            "--max-candidates doit être positif."
        )

    queries = read_jsonl(
        arguments.queries
    )

    pool = read_jsonl(
        arguments.pool
    )

    drafts = read_jsonl(
        arguments.drafts
    )

    existing = read_jsonl(
        arguments.output
    )

    query_by_id = {
        str(query["query_id"]): query
        for query in queries
        if str(
            query.get("split", "")
        ).casefold() == "test"
    }

    pool_by_query: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for item in pool:
        query_id = str(
            item.get("query_id", "")
        )

        pool_by_query.setdefault(
            query_id,
            [],
        ).append(item)

    draft_by_id = {
        str(record["query_id"]): record
        for record in drafts
    }

    records_by_id = {
        str(record["query_id"]): record
        for record in existing
    }

    # Importer uniquement les brouillons déjà validés humainement.
    imported_verified = 0

    for query_id, draft in draft_by_id.items():
        if (
            draft.get("status") != "verified"
            or query_id not in query_by_id
            or query_id in records_by_id
        ):
            continue

        gold_ids = [
            str(value)
            for value in draft.get(
                "gold_chunk_ids",
                [],
            )
        ]

        if not gold_ids:
            continue

        question = query_by_id[
            query_id
        ]

        save_record(
            records_by_id,
            question=question,
            gold_chunk_ids=gold_ids,
            assessor_id=str(
                draft.get(
                    "assessor_id",
                    arguments.assessor_id,
                )
            ),
            output_path=arguments.output,
        )

        imported_verified += 1

    # Enregistrer automatiquement les questions non répondables.
    imported_unanswerable = 0

    for query_id, question in query_by_id.items():
        if (
            question.get("answerable", True)
            or query_id in records_by_id
        ):
            continue

        save_record(
            records_by_id,
            question=question,
            gold_chunk_ids=[],
            assessor_id=arguments.assessor_id,
            output_path=arguments.output,
        )

        imported_unanswerable += 1

    selected_queries = [
        query
        for query in query_by_id.values()
        if arguments.redo
        or str(query["query_id"])
        not in records_by_id
    ]

    if arguments.query_id:
        requested = set(
            arguments.query_id
        )

        selected_queries = [
            query
            for query in selected_queries
            if str(query["query_id"])
            in requested
        ]

    selected_queries.sort(
        key=lambda query: str(
            query["query_id"]
        )
    )

    if arguments.limit is not None:
        selected_queries = selected_queries[
            : arguments.limit
        ]

    print(
        "\n=== Adjudication Gold Evidence TEST ==="
    )
    print(
        "Questions TEST       :",
        len(query_by_id),
    )
    print(
        "Déjà vérifiées       :",
        len(records_by_id),
    )
    print(
        "Drafts validés importés :",
        imported_verified,
    )
    print(
        "Unanswerable importées  :",
        imported_unanswerable,
    )
    print(
        "À examiner           :",
        len(selected_queries),
    )
    print(
        "Sortie               :",
        arguments.output,
    )

    for position, question in enumerate(
        selected_queries,
        start=1,
    ):
        query_id = str(
            question["query_id"]
        )

        candidates = pool_by_query.get(
            query_id,
            [],
        )

        if not candidates:
            print(
                f"\n[ERREUR] Aucun candidat "
                f"pour {query_id}."
            )
            continue

        draft = draft_by_id.get(
            query_id
        )

        ranked = rank_candidates(
            candidates,
            question=question,
            draft=draft,
        )

        visible_count = min(
            arguments.max_candidates,
            len(ranked),
        )

        while True:
            print("\n" + "=" * 100)
            print(
                f"Question {position}/"
                f"{len(selected_queries)} "
                f"| {query_id} "
                f"| {question.get('category')}"
            )
            print("=" * 100)
            print(
                "Question :",
                question.get("question"),
            )
            print(
                "Réponse  :",
                question.get(
                    "expected_answer",
                    "",
                ),
            )
            print(
                "Documents attendus :",
                ", ".join(
                    question.get(
                        "reference_documents",
                        [],
                    )
                )
                or "[non précisé]",
            )

            if draft is not None:
                print(
                    "Proposition Qwen   :",
                    draft.get(
                        "gold_chunk_ids",
                        [],
                    ),
                )
                print(
                    "Confiance Qwen     :",
                    draft.get(
                        "confidence",
                        0,
                    ),
                )
                print(
                    "Justification Qwen :",
                    draft.get(
                        "reason",
                        "",
                    ),
                )

            for index, (
                item,
                score,
                reasons,
            ) in enumerate(
                ranked[:visible_count],
                start=1,
            ):
                display_candidate(
                    item,
                    index=index,
                    score=score,
                    reasons=reasons,
                    maximum_characters=(
                        arguments.maximum_characters
                    ),
                )

            print(
                "\nCommande : "
                "1 ou 1,3 = sélectionner | "
                "f 2 = texte complet | "
                "m = plus de candidats | "
                "s = passer | "
                "q = quitter"
            )

            command = input(
                "\nSélection : "
            ).strip()

            if command.casefold() == "q":
                print(
                    "\nSauvegarde conservée :",
                    arguments.output,
                )
                return

            if command.casefold() == "s":
                break

            if command.casefold() == "m":
                visible_count = min(
                    len(ranked),
                    visible_count + 10,
                )
                continue

            if command.casefold().startswith(
                "f "
            ):
                value = command[2:].strip()

                if not value.isdigit():
                    print(
                        "[ERREUR] Utiliser par "
                        "exemple : f 2"
                    )
                    continue

                index = int(value)

                if (
                    index < 1
                    or index > visible_count
                ):
                    print(
                        "[ERREUR] Index hors limites."
                    )
                    continue

                selected_item = ranked[
                    index - 1
                ][0]

                print("\n" + "=" * 100)
                print(
                    get_chunk_id(
                        selected_item
                    )
                )
                print("=" * 100)
                print(
                    clean_display_text(
                        get_chunk_text(
                            selected_item
                        )
                    )
                )

                input(
                    "\nEntrée pour revenir..."
                )
                continue

            try:
                indexes = parse_selection(
                    command,
                    maximum_index=visible_count,
                )
            except ValueError as error:
                print(
                    f"[ERREUR] {error}"
                )
                continue

            selected_ids = [
                get_chunk_id(
                    ranked[index - 1][0]
                )
                for index in indexes
            ]

            print(
                "\nChunks sélectionnés :"
            )

            for chunk_id in selected_ids:
                print(" -", chunk_id)

            confirmation = input(
                "Confirmer [o/n] : "
            ).strip().casefold()

            if confirmation not in {
                "o",
                "oui",
                "y",
                "yes",
            }:
                print(
                    "Sélection annulée."
                )
                continue

            save_record(
                records_by_id,
                question=question,
                gold_chunk_ids=selected_ids,
                assessor_id=arguments.assessor_id,
                output_path=arguments.output,
            )

            print(
                f"[OK] {query_id} sauvegardée."
            )

            break

    print("\n=== Adjudication terminée ===")
    print(
        "Questions vérifiées :",
        len(records_by_id),
        "/",
        len(query_by_id),
    )
    print(
        "Fichier final       :",
        arguments.output,
    )


if __name__ == "__main__":
    main()
