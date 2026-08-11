"""Interactive streaming terminal for PhosProcess Copilot."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace

from phosprocess.rag.pipeline import PhosProcessRAG, load_runtime_config
from phosprocess.rag.terminal_chat import TerminalChat


def configure_utf8_console(
    streams: Sequence[object] | None = None,
) -> None:
    """Make Windows terminal I/O tolerant of technical Unicode content."""

    targets = streams or (sys.stdin, sys.stdout, sys.stderr)

    for stream in targets:
        reconfigure = getattr(stream, "reconfigure", None)

        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    """Create terminal chat arguments without loading models."""

    parser = argparse.ArgumentParser(
        description=(
            "Conversation RAG streaming avec le retrieval figé dev_best_v3 "
            "et Qwen via Ollama."
        )
    )
    parser.add_argument(
        "--show-retrieval",
        action="store_true",
        help="Afficher les cinq chunks, leurs scores et leur provenance.",
    )
    parser.add_argument(
        "--show-query",
        action="store_true",
        help="Afficher la requête autonome, la langue, le type et le routing.",
    )
    parser.add_argument(
        "--show-latency",
        action="store_true",
        help="Afficher les métriques détaillées après chaque réponse.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Désactiver le warm-up unique au démarrage.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Désactiver la mémoire conversationnelle en session.",
    )
    parser.add_argument(
        "--only-source",
        choices=[
            "becker",
            "report",
            "thermodynamics",
            "heat_transfer",
            "perry",
            "crystallization",
            "control",
            "transport",
        ],
        help=(
            "Forcer une source documentaire pour cette session "
            "(aucun fallback)."
        ),
    )
    parser.add_argument(
        "--model",
        help="Nom du modèle Ollama.",
    )
    parser.add_argument(
        "--ollama-host",
        help="URL du serveur Ollama local.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="Timeout du flux Ollama en secondes.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Température Ollama entre 0 et 1.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Niveau des logs techniques.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load the production service once and start an interactive session."""

    configure_utf8_console()
    arguments = build_parser().parse_args(argv)
    runtime = load_runtime_config()
    ollama = replace(
        runtime.ollama,
        model=arguments.model or runtime.ollama.model,
        host=arguments.ollama_host or runtime.ollama.host,
        timeout_seconds=(
            arguments.timeout
            if arguments.timeout is not None
            else runtime.ollama.timeout_seconds
        ),
        temperature=(
            arguments.temperature
            if arguments.temperature is not None
            else runtime.ollama.temperature
        ),
    )
    runtime = replace(runtime, ollama=ollama)
    logging.basicConfig(
        level=getattr(
            logging,
            arguments.log_level or runtime.logging_level,
        ),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    service = PhosProcessRAG(runtime_config=runtime)
    print(f"Chargement initial : {service.initial_loading_ms:.1f} ms")
    knowledge_base = service.knowledge_base_status()

    if knowledge_base is not None:
        print(f"Base documentaire : {knowledge_base['version']}")
        print(
            "Documents actifs : "
            f"{knowledge_base['document_count']}"
        )
        print(
            f"Chunks actifs : {knowledge_base['chunk_count']}"
        )

    try:
        warmup = service.warmup(enabled=not arguments.no_warmup)

        if warmup.enabled:
            print(
                "Warm-up terminé "
                f"embedding_ms={warmup.embedding_ms:.1f} "
                f"reranker_ms={warmup.reranker_ms:.1f} "
                f"ollama_ms={warmup.ollama_ms:.1f}"
            )
        else:
            print("Warm-up désactivé.")

        return TerminalChat(
            service,
            show_retrieval=arguments.show_retrieval,
            show_query=arguments.show_query,
            show_latency=arguments.show_latency,
            history_enabled=not arguments.no_history,
            source_mode=arguments.only_source or "auto",
        ).run()
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
