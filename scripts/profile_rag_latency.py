"""Run the fixed five-turn production conversation and write latency reports."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from phosprocess.observability.reporting import write_latency_reports
from phosprocess.rag.pipeline import PROJECT_ROOT, PhosProcessRAG

QUESTIONS = (
    "Quel est le rôle de la recirculation dans le réacteur Jacobs ?",
    "Et pourquoi améliore-t-elle la stabilité du procédé ?",
    "Quels problèmes apparaissent lorsque le sulfate est mal contrôlé ?",
    "Comment l'opérateur peut-il les détecter ?",
    "Résume les points importants discutés jusqu'ici.",
)
DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "data" / "observability" / "latency"
)


def build_parser() -> argparse.ArgumentParser:
    """Build profiling CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Profile cinq tours du RAG de production, sans benchmark."
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Désactiver uniquement le warm-up du profil.",
    )
    return parser


def _ollama_model_audit() -> dict[str, Any]:
    """Collect local model metadata without failing the RAG profile."""

    result: dict[str, Any] = {
        "requested_model": "qwen3:8b",
        "think_disabled": True,
    }

    for key, command in (
        ("show", ["ollama", "show", "qwen3:8b"]),
        ("processes", ["ollama", "ps"]),
    ):
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as error:
            result[f"{key}_error"] = type(error).__name__
            continue

        output = completed.stdout.strip()
        result[key] = output[:4000]
        result[f"{key}_returncode"] = completed.returncode

    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Profile one long-lived service and one summary-buffer memory."""

    arguments = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    service = PhosProcessRAG()
    memory = service.create_conversation_memory()
    records: list[dict[str, Any]] = []

    try:
        print(f"Chargement initial : {service.initial_loading_ms:.1f} ms")
        warmup = service.warmup(enabled=not arguments.no_warmup)
        print(
            "Warm-up : "
            f"embedding={warmup.embedding_ms:.1f} ms "
            f"reranker={warmup.reranker_ms:.1f} ms "
            f"ollama={warmup.ollama_ms:.1f} ms"
        )

        for turn, question in enumerate(QUESTIONS, start=1):
            print(f"\nTour {turn} — {question}\nAssistant > ", end="", flush=True)
            completed_response = None

            for event in service.stream_answer(
                question,
                memory.build_history_context(),
            ):
                if event.event_type == "token":
                    print(event.content or "", end="", flush=True)
                elif event.event_type == "error":
                    raise RuntimeError(event.content)
                elif event.event_type == "completed":
                    completed_response = event.response

            if completed_response is None:
                raise RuntimeError(f"Tour {turn} sans réponse validée.")

            print()
            memory.add_turn(question, completed_response.answer)
            record = dict(completed_response.latency)
            record["turn"] = turn
            record["question"] = question
            records.append(record)
            print(
                f"TTFT={record['turn_time_to_first_token_ms'] / 1000:.2f}s "
                f"total={record['total_ms'] / 1000:.2f}s "
                f"prompt={record['estimated_prompt_tokens']} tokens "
                f"ollama_calls={record['ollama_call_count']} "
                f"citations={record['citations']}"
            )

        paths = write_latency_reports(
            arguments.output_directory.resolve(),
            records=records,
            initial_loading_ms=service.initial_loading_ms,
            warmup=warmup.to_dict(),
            model_audit=_ollama_model_audit(),
        )
    finally:
        service.close()

    print("\nRapports produits :")

    for path in paths:
        print(path)

    print(
        json.dumps(
            {
                "turns": len(records),
                "ollama_calls": sum(
                    record["ollama_call_count"]
                    for record in records
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

