"""Tester la recherche hybride dense + BM25 + RRF."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from phosprocess.retrieval.bm25 import (
    load_bm25_config,
)
from phosprocess.retrieval.hybrid import (
    HybridRetriever,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
    maximum_characters: int = 900,
) -> str:
    """Créer un extrait compact et lisible."""

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


def optional_number(
    value: float | int | None,
    *,
    decimals: int = 6,
) -> str:
    """Afficher proprement une valeur optionnelle."""

    if value is None:
        return "-"

    if isinstance(value, int):
        return str(value)

    return f"{value:.{decimals}f}"


def parse_arguments() -> argparse.Namespace:
    """Lire les paramètres du test."""

    parser = argparse.ArgumentParser(
        description=(
            "Tester le retrieval hybride "
            "BGE-M3 + BM25 + RRF."
        )
    )

    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--dense-candidates",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--bm25-candidates",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--dense-minimum-score",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--bm25-minimum-score",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--document-id",
        action="append",
        default=None,
        help="Filtre documentaire répétable.",
    )

    parser.add_argument(
        "--disable-query-expansion",
        action="store_true",
        help=(
            "Désactiver les équivalents lexicaux "
            "français-anglais."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Exécuter la recherche hybride."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    arguments = parse_arguments()

    bm25_config = load_bm25_config(
        RETRIEVAL_CONFIG_PATH
    )

    retriever = HybridRetriever(
        dense_index_directory=(
            DENSE_INDEX_DIRECTORY
        ),
        bm25_index_directory=(
            resolve_project_path(
                bm25_config.output_directory
            )
        ),
        embedding_config_path=(
            EMBEDDING_CONFIG_PATH
        ),
        retrieval_config_path=(
            RETRIEVAL_CONFIG_PATH
        ),
    )

    document_ids = (
        set(arguments.document_id)
        if arguments.document_id
        else None
    )

    response = retriever.search(
        arguments.query,
        top_k=arguments.top_k,
        dense_candidate_k=(
            arguments.dense_candidates
        ),
        bm25_candidate_k=(
            arguments.bm25_candidates
        ),
        dense_minimum_score=(
            arguments.dense_minimum_score
        ),
        bm25_minimum_score=(
            arguments.bm25_minimum_score
        ),
        document_ids=document_ids,
        use_query_expansion=(
            not arguments.disable_query_expansion
        ),
    )

    print("\n=== Recherche hybride ===")
    print(f"Question       : {response.query}")
    print(f"Requête BM25   : {response.lexical_query}")
    print(f"Corpus         : {retriever.total_chunks} chunks")
    print(
        f"Candidats dense: "
        f"{response.dense_results_found}/"
        f"{response.dense_candidates_requested}"
    )
    print(
        f"Candidats BM25 : "
        f"{response.bm25_results_found}/"
        f"{response.bm25_candidates_requested}"
    )
    print(
        f"Union fusionnée: "
        f"{response.fusion_candidates}"
    )
    print(
        f"Durée dense    : "
        f"{response.dense_duration_ms:.3f} ms"
    )
    print(
        f"Durée BM25     : "
        f"{response.bm25_duration_ms:.3f} ms"
    )
    print(
        f"Durée totale   : "
        f"{response.total_duration_ms:.3f} ms"
    )
    print(f"Résultats      : {len(response.results)}")

    if not response.results:
        print("\nAucun résultat hybride trouvé.")
        return

    for result in response.results:
        chunk = result.chunk

        heading = (
            " > ".join(chunk.heading_path)
            if chunk.heading_path
            else "Section non détectée"
        )

        print("\n" + "=" * 88)
        print(
            f"Rang hybride #{result.rank} | "
            f"RRF={result.rrf_score:.8f} | "
            f"Moteurs={'+'.join(result.matched_retrievers)}"
        )

        print(
            "Dense : "
            f"rang={optional_number(result.dense_rank)} | "
            f"score={optional_number(result.dense_score)} | "
            f"contribution="
            f"{result.dense_rrf_contribution:.8f}"
        )

        print(
            "BM25  : "
            f"rang={optional_number(result.bm25_rank)} | "
            f"score={optional_number(result.bm25_score)} | "
            f"contribution="
            f"{result.bm25_rrf_contribution:.8f}"
        )

        print(f"Document : {chunk.source_file}")
        print(
            f"Pages    : "
            f"{format_pages(chunk.source_pages)}"
        )
        print(f"Section  : {heading}")
        print(f"Chunk ID : {chunk.chunk_id}")
        print("-" * 88)
        print(create_excerpt(chunk.text))

    print("\n" + "=" * 88)


if __name__ == "__main__":
    main()