"""Tester retrieval hybride puis reranking cross-encoder."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from phosprocess.reranking.reranker import (
    BGEReranker,
    load_reranking_config,
)
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
    / "retrieval_v2.yaml"
)

RERANKING_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "reranking.yaml"
)

DEFAULT_QUERY = (
    "Pourquoi une forte supersaturation produit-elle "
    "de petits cristaux de gypse difficiles à filtrer ?"
)


def resolve_project_path(
    path_value: str,
) -> Path:
    """Résoudre un chemin depuis la racine du projet."""

    path = Path(path_value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def create_excerpt(
    text: str,
    *,
    maximum_characters: int = 1600,
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
    final_space = shortened.rfind(" ")

    if final_space > 0:
        shortened = shortened[:final_space]

    return shortened.rstrip() + "..."


def format_pages(
    pages: list[int],
) -> str:
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
    """Lire les arguments de la commande."""

    parser = argparse.ArgumentParser(
        description=(
            "Tester le pipeline hybride puis reranking."
        )
    )

    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help=(
            "Nombre de candidats hybrides envoyés "
            "au reranker."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Nombre final de passages.",
    )

    parser.add_argument(
        "--disable-query-expansion",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    """Exécuter la recherche et comparer avant/après."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    arguments = parse_arguments()

    reranking_config = load_reranking_config(
        RERANKING_CONFIG_PATH
    )

    candidate_k = (
        reranking_config.hybrid_candidate_k
        if arguments.candidate_k is None
        else arguments.candidate_k
    )

    final_top_k = (
        reranking_config.final_top_k
        if arguments.top_k is None
        else arguments.top_k
    )

    if candidate_k <= 0:
        raise ValueError(
            "candidate_k doit être strictement positif."
        )

    if final_top_k <= 0:
        raise ValueError(
            "top_k doit être strictement positif."
        )

    if final_top_k > candidate_k:
        raise ValueError(
            "top_k ne peut pas dépasser candidate_k."
        )

    bm25_config = load_bm25_config(
        RETRIEVAL_CONFIG_PATH
    )

    print("\n=== Chargement du retrieval hybride ===")

    hybrid_retriever = HybridRetriever(
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

    hybrid_response = hybrid_retriever.search(
        arguments.query,
        top_k=candidate_k,
        use_query_expansion=(
            not arguments.disable_query_expansion
        ),
    )

    print("\n=== Chargement du reranker ===")

    reranker = BGEReranker(
        reranking_config
    )

    reranking_response = reranker.rerank(
        arguments.query,
        hybrid_response.results,
        top_k=final_top_k,
    )

    preview_count = min(
        final_top_k,
        len(hybrid_response.results),
    )

    print("\n=== Avant reranking : classement RRF ===")
    print(f"Question   : {arguments.query}")
    print(
        f"Candidats  : "
        f"{len(hybrid_response.results)}"
    )

    for candidate in hybrid_response.results[
        :preview_count
    ]:
        print(
            f"RRF #{candidate.rank} | "
            f"score={candidate.rrf_score:.8f} | "
            f"{candidate.chunk.source_file} | "
            f"pages "
            f"{format_pages(candidate.chunk.source_pages)}"
        )

    print("\n=== Après reranking ===")
    print(
        f"Modèle     : "
        f"{reranking_response.model_name}"
    )
    print(
        f"Device     : "
        f"{reranking_response.device}"
    )
    print(
        f"Candidats  : "
        f"{reranking_response.candidates_received}"
    )
    print(
        f"Durée      : "
        f"{reranking_response.reranking_duration_ms:.3f} ms"
    )
    print(
        f"Résultats  : "
        f"{len(reranking_response.results)}"
    )

    for result in reranking_response.results:
        chunk = result.chunk

        heading = (
            " > ".join(chunk.heading_path)
            if chunk.heading_path
            else "Section non détectée"
        )

        movement = (
            result.original_hybrid_rank
            - result.rank
        )

        movement_text = (
            f"+{movement}"
            if movement > 0
            else str(movement)
        )

        print("\n" + "=" * 88)
        print(
            f"Rang final #{result.rank} | "
            f"Reranker={result.reranker_score:.6f} | "
            f"Ancien rang RRF={result.original_hybrid_rank} | "
            f"Mouvement={movement_text}"
        )
        print(
            f"RRF original : "
            f"{result.original_rrf_score:.8f}"
        )
        print(
            "Moteurs      : "
            f"{'+'.join(result.matched_retrievers)}"
        )
        print(
            f"Document     : {chunk.source_file}"
        )
        print(
            "Pages        : "
            f"{format_pages(chunk.source_pages)}"
        )
        print(f"Section      : {heading}")
        print(f"Chunk ID     : {chunk.chunk_id}")
        print("-" * 88)
        print(create_excerpt(chunk.text))

    print("\n" + "=" * 88)


if __name__ == "__main__":
    main()