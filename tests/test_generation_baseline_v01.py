"""Safety gates for the Phase-10 generation baseline harness."""

from __future__ import annotations

import inspect
import json

from phosprocess.evaluation.generation_baseline_v01 import (
    FOLLOWUP_CONVERSATIONS,
    PRODUCTION_BASELINE_FILES,
    _existing_ids,
    _load_question_snapshot,
    _protocol,
    _repair_display_text,
    _run_one,
)


def test_question_snapshot_preserves_frozen_primary_splits() -> None:
    questions = _load_question_snapshot()
    primary = [item for item in questions if item["dataset_scope"] == "phase8_primary"]
    assert len(primary) == 64
    assert sum(item["split"] == "dev" for item in primary) == 45
    assert sum(item["split"] == "final_holdout" for item in primary) == 19
    language_counts = {
        language: sum(item["language"] == language for item in primary)
        for language in {"fr", "en", "ar"}
    }
    assert language_counts == {"fr": 46, "en": 12, "ar": 6}


def test_supplemental_set_contains_three_absent_questions_and_process_case() -> None:
    supplemental = [
        item
        for item in _load_question_snapshot()
        if item["dataset_scope"] != "phase8_primary"
    ]
    assert len(supplemental) == 4
    assert sum(item["answerability"] == "unanswerable" for item in supplemental) == 3
    assert sum(item["dataset_scope"] == "process_flow_acceptance" for item in supplemental) == 1


def test_evaluation_protocol_adds_no_production_judge_or_threshold() -> None:
    protocol = _protocol()
    assert protocol["semantic_judge_used"] is False
    assert protocol["production_thresholds_added"] is False
    assert protocol["gold_visibility"] == "evaluation_after_generation_only"


def test_mojibake_repair_restores_french_and_arabic_inputs() -> None:
    assert _repair_display_text("DÃ©cris lâ€™acide") == "Décris l’acide"
    assert _repair_display_text("Ù…Ø§ Ù‡Ùˆ") == "ما هو"


def test_production_hash_scope_contains_no_evaluation_module() -> None:
    assert all(
        not path.startswith("src/phosprocess/evaluation/")
        for path in PRODUCTION_BASELINE_FILES
    )


def test_generation_record_does_not_copy_gold_into_runtime_payload() -> None:
    source = inspect.getsource(_run_one)
    assert "valid_evidence_sets" not in source
    assert "historical_gold" not in source
    assert "expected_concepts" not in source
    assert '"hidden_chain_of_thought_stored": False' in source


def test_production_does_not_import_phase10_evaluation() -> None:
    import phosprocess.rag.orchestrator as orchestrator

    assert "generation_baseline_v01" not in inspect.getsource(orchestrator)


def test_followup_conversations_cover_dev_and_holdout_without_hidden_facts() -> None:
    assert [item["id"] for item in FOLLOWUP_CONVERSATIONS] == [
        "CONV02",
        "CONV04",
        "CONV06",
        "CONV01",
        "CONV03",
        "CONV05",
    ]
    assert sum(item["split"] == "dev" for item in FOLLOWUP_CONVERSATIONS) == 3
    assert sum(item["split"] == "final_holdout" for item in FOLLOWUP_CONVERSATIONS) == 3

def test_existing_ids_ignore_failed_records(tmp_path) -> None:
    result_path = tmp_path / "results.jsonl"
    rows = [
        {"id": "OK", "status": "completed"},
        {"id": "ERR", "status": "error"},
    ]
    result_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    assert _existing_ids(result_path) == {"OK"}


def test_production_hash_scope_freezes_conversation_resolver_and_catalog() -> None:
    assert "configs/knowledge_base_catalog.yaml" in PRODUCTION_BASELINE_FILES
    assert "src/phosprocess/rag/followup_resolver.py" in PRODUCTION_BASELINE_FILES
    assert "src/phosprocess/rag/conversation_state.py" in PRODUCTION_BASELINE_FILES

