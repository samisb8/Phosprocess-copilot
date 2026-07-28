"""Construire automatiquement le pool d'annotation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from phosprocess.evaluation.pool_builder import (
    EvaluationPoolBuilder,
    build_artifact_signature,
    load_evaluation_queries,
)
from phosprocess.evaluation.schemas import (
    load_evaluation_config,
)
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

EVALUATION_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "evaluation_test_v2.yaml"
)

RETRIEVAL_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "retrieval_v2.yaml"
)

EMBEDDING_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "embeddings.yaml"
)

RERANKING_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "reranking.yaml"
)

DENSE_INDEX_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "indexes"
    / "dense"
    / "bge_m3"
)


def resolve_project_path(
    path_value: str,
) -> Path:
    """Résoudre un chemin depuis la racine du projet."""

    path = Path(path_value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def parse_arguments() -> argparse.Namespace:
    """Lire les paramètres de construction."""

    parser = argparse.ArgumentParser(
        description=(
            "Construire le pool Dense + BM25 + "
            "Hybrid + Reranker."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recommencer intégralement le pooling."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Limiter le nombre de questions pour "
            "un smoke test."
        ),
    )

    parser.add_argument(
        "--split",
        choices=("all", "dev", "test"),
        default="all",
        help="Limiter la construction à un split.",
    )

    parser.add_argument(
        "--query-id",
        action="append",
        default=None,
        help=(
            "Limiter à un query_id. Option répétable."
        ),
    )

    return parser.parse_args()


def select_queries(
    all_queries: list,
    arguments: argparse.Namespace,
) -> list:
    """Appliquer les filtres CLI."""

    selected = list(all_queries)

    if arguments.split != "all":
        selected = [
            query
            for query in selected
            if query.split.value
            == arguments.split
        ]

    if arguments.query_id:
        requested_ids = set(
            arguments.query_id
        )

        known_ids = {
            query.query_id
            for query in all_queries
        }

        unknown_ids = sorted(
            requested_ids - known_ids
        )

        if unknown_ids:
            raise ValueError(
                "query_id inconnus : "
                f"{unknown_ids}"
            )

        selected = [
            query
            for query in selected
            if query.query_id
            in requested_ids
        ]

    selected.sort(
        key=lambda query: query.query_id
    )

    if arguments.limit is not None:
        if arguments.limit <= 0:
            raise ValueError(
                "--limit doit être positif."
            )

        selected = selected[
            : arguments.limit
        ]

    if not selected:
        raise ValueError(
            "La sélection ne contient aucune question."
        )

    return selected


def main() -> None:
    """Charger les modèles une fois puis construire le pool."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(
            encoding="utf-8"
        )

    arguments = parse_arguments()

    evaluation_config = (
        load_evaluation_config(
            EVALUATION_CONFIG_PATH
        )
    )

    reranking_config = (
        load_reranking_config(
            RERANKING_CONFIG_PATH
        )
    )

    bm25_config = load_bm25_config(
        RETRIEVAL_CONFIG_PATH
    )

    output_directory = resolve_project_path(
        evaluation_config
        .dataset
        .output_directory
    )

    queries_path = (
        output_directory
        / evaluation_config
        .dataset
        .queries_filename
    )

    all_queries = load_evaluation_queries(
        queries_path
    )

    selected_queries = select_queries(
        all_queries,
        arguments,
    )

    bm25_index_directory = (
        resolve_project_path(
            bm25_config.output_directory
        )
    )

    signature_paths = [
        queries_path,
        EVALUATION_CONFIG_PATH,
        RETRIEVAL_CONFIG_PATH,
        EMBEDDING_CONFIG_PATH,
        RERANKING_CONFIG_PATH,
        (
            DENSE_INDEX_DIRECTORY
            / "manifest.json"
        ),
        (
            bm25_index_directory
            / bm25_config.manifest_filename
        ),
    ]

    build_signature = (
        build_artifact_signature(
            signature_paths
        )
    )

    print("\n=== Configuration du pooling ===")
    print(f"Questions      : {len(selected_queries)}")
    print(f"Split          : {arguments.split}")
    print(
        "Dense depth    : "
        f"{evaluation_config.pooling.dense_depth}"
    )
    print(
        "BM25 depth     : "
        f"{evaluation_config.pooling.bm25_depth}"
    )
    print(
        "Hybrid depth   : "
        f"{evaluation_config.pooling.hybrid_depth}"
    )
    print(
        "Reranker depth : "
        f"{evaluation_config.pooling.reranker_depth}"
    )
    print(
        "Signature      : "
        f"{build_signature[:16]}..."
    )

    print("\n=== Chargement des modèles ===")

    hybrid_retriever = HybridRetriever(
        dense_index_directory=(
            DENSE_INDEX_DIRECTORY
        ),
        bm25_index_directory=(
            bm25_index_directory
        ),
        embedding_config_path=(
            EMBEDDING_CONFIG_PATH
        ),
        retrieval_config_path=(
            RETRIEVAL_CONFIG_PATH
        ),
    )

    reranker = BGEReranker(
        reranking_config
    )

    builder = EvaluationPoolBuilder(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        evaluation_config=evaluation_config,
        reranking_config=reranking_config,
        output_directory=output_directory,
        build_signature=build_signature,
    )

    builder.build(
        all_queries=selected_queries,
        selected_queries=selected_queries,
        force=arguments.force,
    )


if __name__ == "__main__":
    main()
