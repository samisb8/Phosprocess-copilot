"""Build the persistent BGE-M3 sparse index for the active quality corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from phosprocess.embeddings.embedder import BGEEmbedder, load_embedding_config
from phosprocess.knowledge_base.runtime import load_active_knowledge_base
from phosprocess.retrieval.bge_sparse import build_bge_sparse_index

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDING_CONFIG = PROJECT_ROOT / "configs" / "embeddings.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construire uniquement l'index lexical BGE-M3 à partir des "
            "embedding_text existants. Les chunks et l'index dense restent inchangés."
        )
    )
    parser.add_argument(
        "--version-directory",
        type=Path,
        default=None,
        help="Version kb_quality_* explicite. Par défaut, utilise current_index.json.",
    )
    parser.add_argument(
        "--batch-documents",
        type=int,
        default=64,
        help="Nombre de chunks transmis par lot au constructeur.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version_directory = (
        args.version_directory.resolve()
        if args.version_directory is not None
        else load_active_knowledge_base().version_directory
    )
    config = load_embedding_config(DEFAULT_EMBEDDING_CONFIG)
    embedder = BGEEmbedder(config)
    output = build_bge_sparse_index(
        version_directory=version_directory,
        embedder=embedder,
        batch_documents=args.batch_documents,
        force=args.force,
    )
    print(f"Index BGE sparse créé : {output}")
    print(f"Matrice : {output / 'matrix.npz'}")
    print(f"Manifeste : {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
