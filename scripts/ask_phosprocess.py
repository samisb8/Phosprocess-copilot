"""Command-line interface for the production PhosProcess RAG service."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace

from phosprocess.llm.ollama_client import OllamaConfig
from phosprocess.rag.pipeline import (
    PhosProcessRAG,
    RAGError,
    load_runtime_config,
)


def build_parser() -> argparse.ArgumentParser:
    """Create CLI arguments without loading retrieval models."""

    parser = argparse.ArgumentParser(
        description=(
            "Interroger les documents de procédés phosphoriques avec "
            "le retrieval figé dev_best_v3 et Qwen via Ollama."
        )
    )
    parser.add_argument(
        "question",
        nargs="+",
        help="Question métier à poser.",
    )
    parser.add_argument(
        "--model",
        help="Nom du modèle Ollama, par défaut qwen3:8b.",
    )
    parser.add_argument(
        "--ollama-host",
        help="URL du serveur Ollama local.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="Timeout Ollama en secondes.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Température de génération entre 0 et 1.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Afficher la réponse RAG complète en JSON.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Niveau de logs.",
    )
    return parser


def configure_logging(level: str) -> None:
    """Configure concise application logs on stderr."""

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def configure_console_encoding() -> None:
    """Use Unicode output on Windows while preserving captured test streams."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)

        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def apply_ollama_overrides(
    base: OllamaConfig,
    arguments: argparse.Namespace,
) -> OllamaConfig:
    """Apply CLI-only generation overrides."""

    return replace(
        base,
        model=arguments.model or base.model,
        host=arguments.ollama_host or base.host,
        timeout_seconds=(
            arguments.timeout
            if arguments.timeout is not None
            else base.timeout_seconds
        ),
        temperature=(
            arguments.temperature
            if arguments.temperature is not None
            else base.temperature
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one production RAG request."""

    configure_console_encoding()
    parser = build_parser()
    arguments = parser.parse_args(argv)
    runtime = load_runtime_config()
    log_level = arguments.log_level or runtime.logging_level
    configure_logging(log_level)
    runtime = replace(
        runtime,
        ollama=apply_ollama_overrides(
            runtime.ollama,
            arguments,
        ),
    )
    question = " ".join(arguments.question)

    service: PhosProcessRAG | None = None

    try:
        service = PhosProcessRAG(
            runtime_config=runtime,
        )
        response = service.answer(question)
    except (RAGError, ValueError, TypeError) as error:
        logging.getLogger(__name__).error("%s", error)
        return 2
    finally:
        if service is not None:
            service.close()

    if arguments.json:
        print(response.model_dump_json(indent=2))
        return 0

    print(response.answer)
    print("\nSources:")

    for source in response.sources:
        pages = ", ".join(
            str(page)
            for page in source.pages
        )
        section = source.section or "Section non renseignée"
        print(
            f"[Source {source.source_number}] "
            f"{source.document_name}, pages {pages}"
        )
        print(f"  Section: {section}")
        print(f"  Chunk: {source.chunk_id}")
        print(
            "  Scores: "
            f"RRF={source.rrf_score:.6f}, "
            f"reranker={source.reranker_score}"
        )
        print(f"  Extrait: {source.excerpt}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
