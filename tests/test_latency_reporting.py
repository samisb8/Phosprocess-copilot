"""Tests for content-safe latency artifact generation."""

from __future__ import annotations

import json

from phosprocess.observability.reporting import write_latency_reports


def make_record(turn: int) -> dict[str, object]:
    """Create one synthetic instrumented turn without document text."""

    return {
        "turn": turn,
        "question": f"Question {turn}",
        "retrieval_query": f"Question autonome {turn}",
        "reformulation_attempted": False,
        "reformulation_method": "none",
        "ollama_call_count": 1,
        "summary_token_count": 10,
        "recent_history_token_count": 20,
        "document_context_token_count": 200,
        "estimated_prompt_tokens": 260,
        "prompt_character_count": 1040,
        "embedding_ms": 1.0,
        "dense_search_ms": 2.0,
        "bm25_search_ms": 3.0,
        "query_expansion_ms": 0.1,
        "hybrid_fusion_ms": 0.2,
        "reranking_ms": 5.0,
        "reformulation_ms": 0.0,
        "turn_time_to_first_token_ms": 100.0,
        "ollama_generation_ms": 200.0,
        "total_ms": 320.0,
        "repair_attempted": False,
        "citations": [1],
        "displayed_source_count": 1,
        "baseline_equivalent_prompt_characters": 4000,
        "baseline_equivalent_prompt_tokens": 1000,
        "system_prompt_token_count": 30,
        "question_token_count": 10,
        "ollama_calls": [
            {
                "call_type": "generation_main",
                "model": "qwen3:8b",
                "streaming": True,
                "success": True,
                "prompt_character_count": 1040,
                "estimated_prompt_tokens": 260,
                "duration_ms": 210.0,
                "time_to_first_event_ms": 90.0,
                "time_to_first_token_ms": 100.0,
                "generation_ms": 110.0,
                "generated_token_count": 40,
                "prompt_evaluation_ms": 80.0,
                "model_generation_ms": 110.0,
                "generation_tokens_per_second": 363.6,
                "error_type": None,
            }
        ],
    }


def test_six_reports_are_generated_without_document_passages(tmp_path) -> None:
    records = [make_record(turn) for turn in range(1, 6)]

    paths = write_latency_reports(
        tmp_path,
        records=records,
        initial_loading_ms=42.0,
        warmup={"enabled": True, "total_ms": 5.0},
    )

    assert len(paths) == 6
    assert all(path.is_file() for path in paths)
    optimized = json.loads(
        (tmp_path / "latency_optimized.json").read_text(encoding="utf-8")
    )
    assert len(optimized["turns"]) == 5
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in paths
    )
    assert "texte complet du chunk" not in combined
    assert "gold" not in combined.casefold()


def test_report_requires_exactly_five_turns(tmp_path) -> None:
    try:
        write_latency_reports(
            tmp_path,
            records=[make_record(1)],
            initial_loading_ms=0.0,
            warmup={},
        )
    except ValueError as error:
        assert "exactement cinq tours" in str(error)
    else:
        raise AssertionError("Le rapport aurait dû refuser un profil incomplet.")
