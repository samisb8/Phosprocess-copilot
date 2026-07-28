"""Content-safe before/after reports for conversational RAG latency."""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from statistics import mean
from typing import Any
from uuid import uuid4

BASELINE_TURNS: tuple[dict[str, Any], ...] = (
    {
        "turn": 1,
        "question": (
            "Quel est le rôle de la recirculation dans le réacteur Jacobs ?"
        ),
        "time_to_first_token_ms": 31_410.0,
        "total_ms": 65_060.0,
    },
    {
        "turn": 2,
        "question": (
            "Et pourquoi améliore-t-elle la stabilité du procédé ?"
        ),
        "time_to_first_token_ms": 64_140.0,
        "total_ms": 130_910.0,
    },
)

PER_TURN_FIELDS = (
    "turn",
    "question",
    "retrieval_query",
    "source_policy_route",
    "source_policy_mode",
    "source_policy_primary",
    "source_policy_fallback_used",
    "source_policy_forced",
    "source_policy_attempt_count",
    "source_policy_sufficient_preferred_chunks",
    "reformulation_attempted",
    "reformulation_method",
    "ollama_call_count",
    "summary_token_count",
    "recent_history_token_count",
    "document_context_token_count",
    "estimated_prompt_tokens",
    "prompt_character_count",
    "embedding_ms",
    "dense_search_ms",
    "bm25_search_ms",
    "query_expansion_ms",
    "hybrid_fusion_ms",
    "candidate_preparation_ms",
    "reranker_tokenization_ms",
    "reranker_scoring_ms",
    "reranking_ms",
    "lexical_selection_ms",
    "source_loading_ms",
    "excerpt_preparation_ms",
    "memory_build_ms",
    "prompt_build_ms",
    "reformulation_ms",
    "ollama_connection_ms",
    "ollama_time_to_first_event_ms",
    "ollama_time_to_first_token_ms",
    "turn_time_to_first_token_ms",
    "ollama_generation_ms",
    "json_validation_ms",
    "citation_extraction_ms",
    "citation_validation_ms",
    "repair_ms",
    "total_ms",
    "repair_attempted",
    "citations",
    "displayed_source_count",
)


def _atomic_write_text(path: Path, content: str) -> None:
    """Write UTF-8 atomically, with a Windows PermissionError fallback."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="")

    try:
        os.replace(temporary, path)
    except PermissionError:
        shutil.copyfile(temporary, path)
        temporary.unlink(missing_ok=True)


def _csv_text(
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> str:
    """Serialize mappings as deterministic CSV."""

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for row in rows:
        writer.writerow(
            {
                field: (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                )
                for field, value in row.items()
                if field in fieldnames
            }
        )

    return output.getvalue()


def _average(records: Sequence[Mapping[str, Any]], field: str) -> float:
    """Average one numeric field over profile records."""

    values = [
        float(record[field])
        for record in records
        if isinstance(record.get(field), (int, float))
    ]
    return round(mean(values), 3) if values else 0.0


def _percentage_reduction(before: float, after: float) -> float:
    """Return positive percentage reduction from before to after."""

    if before <= 0:
        return 0.0

    return round((before - after) / before * 100.0, 2)


def write_latency_reports(
    output_directory: Path,
    *,
    records: Sequence[Mapping[str, Any]],
    initial_loading_ms: float,
    warmup: Mapping[str, Any],
    model_audit: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Write the six requested latency artifacts without document passages."""

    if len(records) != 5:
        raise ValueError("Le rapport optimisé exige exactement cinq tours.")

    generated_at = datetime.now(UTC).isoformat()
    all_calls = [
        call
        for record in records
        for call in record.get("ollama_calls", [])
        if isinstance(call, dict)
    ]
    token_rates = [
        float(call["generation_tokens_per_second"])
        for call in all_calls
        if isinstance(
            call.get("generation_tokens_per_second"),
            (int, float),
        )
    ]
    prompt_evaluation_durations = [
        float(call["prompt_evaluation_ms"])
        for call in all_calls
        if isinstance(call.get("prompt_evaluation_ms"), (int, float))
    ]
    baseline = {
        "recorded_before_optimization": True,
        "granular_instrumentation_available": False,
        "methodology": (
            "Latences observées avant l'instrumentation. Les tailles de prompt "
            "avant sont des estimations équivalentes calculées sur les cinq "
            "chunks complets sélectionnés lors du profil optimisé."
        ),
        "turns": list(BASELINE_TURNS),
    }
    optimized = {
        "generated_at_utc": generated_at,
        "scope": "production_rag_conversation_only",
        "benchmark_used": False,
        "initial_loading_ms": round(initial_loading_ms, 3),
        "warmup": dict(warmup),
        "model_audit": dict(model_audit or {}),
        "aggregate": {
            "average_time_to_first_token_ms": _average(
                records,
                "turn_time_to_first_token_ms",
            ),
            "average_total_ms": _average(records, "total_ms"),
            "average_prompt_tokens": _average(
                records,
                "estimated_prompt_tokens",
            ),
            "total_ollama_calls": sum(
                int(record.get("ollama_call_count", 0))
                for record in records
            ),
            "repair_count": sum(
                bool(record.get("repair_attempted"))
                for record in records
            ),
            "average_prompt_evaluation_ms": (
                round(mean(prompt_evaluation_durations), 3)
                if prompt_evaluation_durations
                else 0.0
            ),
            "average_generation_tokens_per_second": (
                round(mean(token_rates), 3)
                if token_rates
                else 0.0
            ),
        },
        "turns": [dict(record) for record in records],
    }
    prompt_rows: list[dict[str, Any]] = []
    call_rows: list[dict[str, Any]] = []

    for record in records:
        prompt_rows.append(
            {
                "turn": record["turn"],
                "baseline_equivalent_characters": record[
                    "baseline_equivalent_prompt_characters"
                ],
                "baseline_equivalent_tokens": record[
                    "baseline_equivalent_prompt_tokens"
                ],
                "optimized_characters": record[
                    "prompt_character_count"
                ],
                "optimized_tokens": record[
                    "estimated_prompt_tokens"
                ],
                "token_reduction_percent": _percentage_reduction(
                    float(record["baseline_equivalent_prompt_tokens"]),
                    float(record["estimated_prompt_tokens"]),
                ),
                "system_tokens": record["system_prompt_token_count"],
                "summary_tokens": record["summary_token_count"],
                "recent_history_tokens": record[
                    "recent_history_token_count"
                ],
                "document_context_tokens": record[
                    "document_context_token_count"
                ],
                "question_tokens": record["question_token_count"],
            }
        )

        calls = record.get("ollama_calls", [])

        for call_index, call in enumerate(calls, start=1):
            call_rows.append(
                {
                    "turn": record["turn"],
                    "call_index": call_index,
                    "call_type": call["call_type"],
                    "model": call["model"],
                    "streaming": call["streaming"],
                    "success": call["success"],
                    "prompt_characters": call["prompt_character_count"],
                    "estimated_prompt_tokens": call[
                        "estimated_prompt_tokens"
                    ],
                    "duration_ms": call["duration_ms"],
                    "time_to_first_event_ms": call[
                        "time_to_first_event_ms"
                    ],
                    "time_to_first_token_ms": call[
                        "time_to_first_token_ms"
                    ],
                    "generation_ms": call["generation_ms"],
                    "generated_tokens": call["generated_token_count"],
                    "prompt_evaluation_ms": call[
                        "prompt_evaluation_ms"
                    ],
                    "model_generation_ms": call[
                        "model_generation_ms"
                    ],
                    "generation_tokens_per_second": call[
                        "generation_tokens_per_second"
                    ],
                    "error_type": call["error_type"],
                }
            )

    ttft_target_turns = [
        str(record["turn"])
        for record in records
        if float(record["turn_time_to_first_token_ms"]) < 15_000
    ]
    total_target_turns = [
        str(record["turn"])
        for record in records
        if float(record["total_ms"]) < 45_000
    ]
    comparison_lines = [
        "# Comparaison de latence du chat RAG",
        "",
        "Périmètre : pipeline utilisateur uniquement, sans benchmark.",
        "",
        "| Tour | TTFT avant | TTFT après | Total avant | Total après |",
        "|---:|---:|---:|---:|---:|",
    ]

    for baseline_turn, record in zip(
        BASELINE_TURNS,
        records[:2],
        strict=True,
    ):
        comparison_lines.append(
            "| {turn} | {before_ttft:.2f} s | {after_ttft:.2f} s | "
            "{before_total:.2f} s | {after_total:.2f} s |".format(
                turn=baseline_turn["turn"],
                before_ttft=(
                    baseline_turn["time_to_first_token_ms"] / 1000.0
                ),
                after_ttft=(
                    float(record["turn_time_to_first_token_ms"]) / 1000.0
                ),
                before_total=baseline_turn["total_ms"] / 1000.0,
                after_total=float(record["total_ms"]) / 1000.0,
            )
        )

    comparison_lines.extend(
        [
            "",
            "## Diagnostic",
            "",
            (
                "- La croissance initiale provenait principalement du volume "
                "du prompt documentaire et de la répétition de l'historique, "
                "pas d'une reconstruction du pipeline par tour."
            ),
            (
                "- Le pipeline optimisé réutilise les modèles, index et client "
                "HTTP ; la mémoire est plafonnée et n'apporte aucune ancienne "
                "preuve documentaire."
            ),
            (
                "- Les cinq chunks restent sélectionnés par dev_best_v3 ; seul "
                "leur extrait envoyé à Qwen est compacté."
            ),
            (
                "- Le goulot restant est Ollama : évaluation moyenne du prompt "
                f"{optimized['aggregate']['average_prompt_evaluation_ms'] / 1000:.2f} s, "
                "puis génération à "
                f"{optimized['aggregate']['average_generation_tokens_per_second']:.2f} "
                "tokens/s avec l'offload CPU/GPU borné."
            ),
            "",
            "## Résultat sur cinq tours",
            "",
            (
                f"- TTFT moyen : "
                f"{optimized['aggregate']['average_time_to_first_token_ms'] / 1000:.2f} s"
            ),
            (
                f"- Durée totale moyenne : "
                f"{optimized['aggregate']['average_total_ms'] / 1000:.2f} s"
            ),
            (
                f"- Appels Ollama : "
                f"{optimized['aggregate']['total_ollama_calls']}"
            ),
            f"- Réparations : {optimized['aggregate']['repair_count']}",
            (
                "- Cible < 15 s au premier token : atteinte aux tours "
                f"{', '.join(ttft_target_turns) or 'aucun'}."
            ),
            (
                "- Cible < 45 s total : atteinte aux tours "
                f"{', '.join(total_target_turns) or 'aucun'}."
            ),
            "",
            (
                "Les métriques détaillées sont dans `latency_per_turn.csv`; "
                "aucun texte complet de chunk n'est enregistré."
            ),
        ]
    )

    paths = {
        "latency_baseline.json": (
            json.dumps(baseline, ensure_ascii=False, indent=2) + "\n"
        ),
        "latency_optimized.json": (
            json.dumps(optimized, ensure_ascii=False, indent=2) + "\n"
        ),
        "latency_per_turn.csv": _csv_text(records, PER_TURN_FIELDS),
        "latency_comparison.md": "\n".join(comparison_lines) + "\n",
        "prompt_size_comparison.csv": _csv_text(
            prompt_rows,
            tuple(prompt_rows[0]),
        ),
        "ollama_calls_per_turn.csv": _csv_text(
            call_rows,
            (
                tuple(call_rows[0])
                if call_rows
                else (
                    "turn",
                    "call_index",
                    "call_type",
                    "model",
                    "streaming",
                    "success",
                    "prompt_characters",
                    "estimated_prompt_tokens",
                    "duration_ms",
                    "time_to_first_event_ms",
                    "time_to_first_token_ms",
                    "generation_ms",
                    "generated_tokens",
                    "prompt_evaluation_ms",
                    "model_generation_ms",
                    "generation_tokens_per_second",
                    "error_type",
                )
            ),
        ),
    }
    produced: list[Path] = []

    for name, content in paths.items():
        path = output_directory / name
        _atomic_write_text(path, content)
        produced.append(path)

    return produced
