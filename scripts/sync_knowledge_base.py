"""Synchronize the active production PDFs with one safe command."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Create the administration CLI without loading ML models."""

    parser = argparse.ArgumentParser(
        description=(
            "Détecter, traiter, indexer et activer atomiquement les PDF "
            "de data/knowledge_base/pdfs/."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Afficher les changements sans écrire aucun fichier.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Reconstruire une version même sans changement documentaire.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_documents",
        help="Afficher les documents actifs puis quitter.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Afficher l'état de l'index actif puis quitter.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Afficher les détails de traitement et de cache.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one mutually exclusive administrative action."""

    arguments = build_parser().parse_args(argv)

    from phosprocess.knowledge_base.quality_manager import (
        QualityKnowledgeBaseError,
        QualityKnowledgeBaseManager,
    )

    manager = QualityKnowledgeBaseManager()

    if arguments.list_documents and arguments.status:
        raise SystemExit("--list et --status sont mutuellement exclusifs.")

    if arguments.list_documents:
        manager.list_active()
        return 0

    if arguments.status:
        manager.status()
        return 0

    print("Synchronisation de la base documentaire")
    print("=======================================")

    try:
        result = manager.sync(
            dry_run=arguments.dry_run,
            rebuild=arguments.rebuild,
            verbose=arguments.verbose,
        )
    except QualityKnowledgeBaseError as error:
        print(f"\nÉchec : {error}")
        return 1

    if result.dry_run:
        print("\nMode dry-run : aucune modification effectuée.")
        return 0

    if not result.changed:
        print("\nSynchronisation terminée sans changement.")
        print(f"Version conservée : {result.version}")
        print(f"Documents actifs : {result.document_count}")
        print(f"Chunks actifs : {result.chunk_count}")
        print("Redémarrage du chat requis : non")
        return 0

    print("\nSynchronisation terminée.")
    print(f"Nouvelle version : {result.version}")
    print(f"Version rollback : {result.previous_version or 'aucune'}")
    print("FAISS : OK")
    print("BM25 : OK")
    print("Validation : OK")
    print("Activation atomique : OK")
    print(f"Documents actifs : {result.document_count}")
    print(f"Chunks actifs : {result.chunk_count}")
    print(
        "Embeddings générés : "
        f"{result.embedded_chunk_count}"
    )
    print(
        "Embeddings réutilisés : "
        f"{result.reused_embedding_count}"
    )

    for document in result.documents:
        print(
            f"- {document.filename} : "
            f"{document.chunk_count} chunks"
        )

    print("Redémarrage du chat requis : oui")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
