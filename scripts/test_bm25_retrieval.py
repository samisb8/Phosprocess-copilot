"""Tester la recherche lexicale BM25 sur le corpus réel."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from phosprocess.retrieval.bm25 import (
    BM25Retriever,
    load_bm25_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "retrieval.yaml"

DEFAULT_QUERY = (
    "Quel est le rapport de recirculation externe "
    "du procédé Jacobs ?"
)


def resolve_project_path(path_value: str) -> Path:
    """Résoudre un chemin depuis la racine du projet."""

    path = Path(path_value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def create_excerpt(
    text: str,
    *,
    maximum_characters: int = 700,
) -> str:
    """Créer un extrait compact."""

    normalized = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if len(normalized) <= maximum_characters:
        return normalized

    shortened = normalized[:maximum_characters]
    last_space = shortened.rfind(" ")

    if last_space > 0:
        shortened = shortened[:last_space]

    return shortened.rstrip() + "..."


def format_pages(pages: list[int]) -> str:
    """Afficher les pages sous forme d'intervalles."""

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
    """Lire les paramètres du test."""

    parser = argparse.ArgumentParser(
        description="Tester BM25 sur PhosProcess."
    )

    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--minimum-score",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--document-id",
        action="append",
        default=None,
        help="Filtre documentaire répétable.",
    )

    return parser.parse_args()


def main() -> None:
    """Exécuter une recherche lexicale."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    arguments = parse_arguments()
    config = load_bm25_config(CONFIG_PATH)

    retriever = BM25Retriever(
        index_directory=resolve_project_path(
            config.output_directory
        ),
        config_path=CONFIG_PATH,
    )

    document_ids = (
        set(arguments.document_id)
        if arguments.document_id
        else None
    )

    response = retriever.search(
        arguments.query,
        top_k=arguments.top_k,
        minimum_score=arguments.minimum_score,
        document_ids=document_ids,
    )

    print("\n=== Recherche BM25 ===")
    print(f"Question   : {response.query}")
    print(
        f"Tokens     : "
        f"{', '.join(response.query_tokens)}"
    )
    print(
        f"Index      : "
        f"{retriever.total_documents} chunks"
    )
    print(
        f"Durée      : "
        f"{response.search_duration_ms:.3f} ms"
    )
    print(f"Résultats  : {len(response.results)}")

    if not response.results:
        print("\nAucun terme commun significatif trouvé.")
        return

    for result in response.results:
        chunk = result.chunk

        heading = (
            " > ".join(chunk.heading_path)
            if chunk.heading_path
            else "Section non détectée"
        )

        print("\n" + "=" * 80)
        print(
            f"Rang #{result.rank} | "
            f"Score={result.score:.6f} | "
            f"Lexical ID={result.lexical_id}"
        )
        print(f"Document : {chunk.source_file}")
        print(
            f"Pages    : "
            f"{format_pages(chunk.source_pages)}"
        )
        print(f"Section  : {heading}")
        print(f"Chunk ID : {chunk.chunk_id}")
        print("-" * 80)
        print(create_excerpt(chunk.text))

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()