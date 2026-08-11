"""Tester la recherche sémantique dense sur le corpus réel."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from phosprocess.retrieval.dense import DenseRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_DIRECTORY = (
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

DEFAULT_QUERY = (
    "Pourquoi une supersaturation excessive réduit-elle "
    "la filtrabilité des cristaux de gypse ?"
)


def create_excerpt(
    text: str,
    *,
    maximum_characters: int = 700,
) -> str:
    """Créer un extrait lisible sans espaces inutiles."""

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


def format_pages(source_pages: list[int]) -> str:
    """Afficher les pages sous une forme compacte."""

    if not source_pages:
        return ""

    ranges: list[str] = []
    start = source_pages[0]
    previous = source_pages[0]

    for page in source_pages[1:]:
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
        description="Tester le dense retrieval BGE-M3 + FAISS."
    )

    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="Question à rechercher dans le corpus.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Nombre de passages à retourner.",
    )

    parser.add_argument(
        "--minimum-score",
        type=float,
        default=None,
        help="Score minimal optionnel entre -1 et 1.",
    )

    parser.add_argument(
        "--document-id",
        action="append",
        default=None,
        help=(
            "Limiter la recherche à un document. "
            "Option répétable."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Lancer une recherche dense et afficher les résultats."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    arguments = parse_arguments()

    retriever = DenseRetriever(
        index_directory=INDEX_DIRECTORY,
        embedding_config_path=EMBEDDING_CONFIG_PATH,
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

    print("\n=== Recherche dense ===")
    print(f"Question  : {response.query}")
    print(f"Index     : {retriever.total_vectors} vecteurs")
    print(f"Dimension : {retriever.dimension}")
    print(f"Durée     : {response.search_duration_ms:.3f} ms")
    print(f"Résultats : {len(response.results)}")

    if not response.results:
        print("\nAucun passage ne respecte les critères.")
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
            f"Vector ID={result.vector_id}"
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