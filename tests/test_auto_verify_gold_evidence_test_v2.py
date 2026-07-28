"""Tests des garde-fous de l'adjudication automatique TEST v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from scripts import auto_verify_gold_evidence_test_v2 as auto_verify


def make_questions() -> list[dict[str, Any]]:
    """Créer les 28 questions minimales attendues par la validation."""

    return [
        {
            "query_id": f"Q{number:03d}",
            "split": "test",
            "category": (
                "unanswerable"
                if number >= 45
                else "process_description"
            ),
            "answerable": number < 45,
        }
        for number in range(21, 49)
    ]


def make_complete_gold() -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """Créer un gold final valide et son petit corpus."""

    records: list[dict[str, Any]] = []
    chunks: dict[str, dict[str, Any]] = {}

    for number in range(21, 49):
        question_id = f"Q{number:03d}"
        chunk_ids: list[str] = []

        if number < 45:
            chunk_id = f"document_{number:03d}_chunk"
            chunk_ids = [chunk_id]
            chunks[chunk_id] = {
                "chunk_id": chunk_id,
                "document_id": "expected_document",
                "text": "Preuve complète.",
            }

        records.append(
            {
                "query_id": question_id,
                "gold_chunk_ids": chunk_ids,
            }
        )

    return records, chunks


def test_validate_llm_decision_rejects_unknown_chunk_id() -> None:
    raw = {
        "question_id": "Q029",
        "selected_chunk_ids": ["chunk_invented"],
        "confidence": 0.99,
        "justification": "Le passage couvre la réponse.",
        "missing_claims": [],
    }

    with pytest.raises(
        auto_verify.GoldValidationError,
        match="absent",
    ):
        auto_verify.validate_llm_decision(
            raw,
            question_id="Q029",
            allowed_chunk_ids={"chunk_authorized"},
        )


def test_safety_requires_complete_expected_document_evidence() -> None:
    decision = {
        "question_id": "Q035",
        "selected_chunk_ids": ["wrong_document_chunk"],
        "confidence": 0.90,
        "justification": "Passage proche.",
        "missing_claims": ["Le mécanisme exact."],
    }
    question = {
        "query_id": "Q035",
        "answerable": True,
        "reference_documents": ["expected_document"],
    }
    chunks = {
        "wrong_document_chunk": {
            "document_id": "other_document",
            "text": "Contexte seulement.",
        }
    }

    issues = auto_verify.decision_safety_issues(
        decision,
        question=question,
        chunks_by_id=chunks,
        confidence_threshold=0.85,
    )

    assert any("non couvertes" in issue for issue in issues)
    assert any("hors document" in issue for issue in issues)


def test_safety_rejects_missing_numeric_claim() -> None:
    decision = {
        "question_id": "Q041",
        "selected_chunk_ids": ["context_chunk"],
        "confidence": 0.95,
        "justification": "Le tableau est annoncé, mais absent.",
        "missing_claims": [],
    }
    question = {
        "query_id": "Q041",
        "answerable": True,
        "question": "Quelles productivités sont indiquées ?",
        "expected_answer": "The values are 1.3 and 12.",
        "reference_documents": ["expected_document"],
    }
    chunks = {
        "context_chunk": {
            "document_id": "expected_document",
            "text": "The systems are compared in the following table.",
        }
    }

    issues = auto_verify.decision_safety_issues(
        decision,
        question=question,
        chunks_by_id=chunks,
        confidence_threshold=0.85,
    )

    assert any("1.3" in issue and "12" in issue for issue in issues)


def test_safety_rejects_opposite_solids_condition() -> None:
    decision = {
        "question_id": "Q038",
        "selected_chunk_ids": ["low_solids_chunk"],
        "confidence": 0.95,
        "justification": "Le passage parle des solides.",
        "missing_claims": [],
    }
    question = {
        "query_id": "Q038",
        "answerable": True,
        "question": "How can high solids cause small crystals?",
        "expected_answer": "High solids make mixing difficult.",
        "reference_documents": ["expected_document"],
    }
    chunks = {
        "low_solids_chunk": {
            "document_id": "expected_document",
            "text": (
                "Decreasing solids content reduces crystal surface and "
                "produces smaller crystals."
            ),
        }
    }

    issues = auto_verify.decision_safety_issues(
        decision,
        question=question,
        chunks_by_id=chunks,
        confidence_threshold=0.85,
    )

    assert any("high solids" in issue for issue in issues)


def test_safety_requires_chemical_entities_from_question() -> None:
    decision = {
        "question_id": "Q043",
        "selected_chunk_ids": ["aluminium_only"],
        "confidence": 0.95,
        "justification": "Le passage explique les boues d'aluminium.",
        "missing_claims": [],
    }
    question = {
        "query_id": "Q043",
        "answerable": True,
        "question": "Quel est l'effet de Al2O3 et K2O ?",
        "expected_answer": "Ces impuretés causent des pertes de P2O5.",
        "reference_documents": ["expected_document"],
    }
    chunks = {
        "aluminium_only": {
            "document_id": "expected_document",
            "text": "Al2O3 peut précipiter avec P2O5 dans les boues.",
        }
    }

    issues = auto_verify.decision_safety_issues(
        decision,
        question=question,
        chunks_by_id=chunks,
        confidence_threshold=0.85,
    )

    assert any("k2o" in issue.casefold() for issue in issues)


def test_validate_final_dataset_accepts_28_valid_questions() -> None:
    records, chunks = make_complete_gold()
    human_records = records[:3]

    auto_verify.validate_final_dataset(
        records,
        test_questions=make_questions(),
        chunks_by_id=chunks,
        human_records=human_records,
    )


def test_validate_final_dataset_rejects_nonempty_unanswerable() -> None:
    records, chunks = make_complete_gold()
    chunks["forbidden_chunk"] = {
        "chunk_id": "forbidden_chunk",
        "document_id": "expected_document",
        "text": "Ne devrait pas être sélectionné.",
    }
    records[-1]["gold_chunk_ids"] = ["forbidden_chunk"]

    with pytest.raises(
        auto_verify.GoldValidationError,
        match="non répondable",
    ):
        auto_verify.validate_final_dataset(
            records,
            test_questions=make_questions(),
            chunks_by_id=chunks,
            human_records=[],
        )


def test_validate_final_dataset_preserves_human_records() -> None:
    records, chunks = make_complete_gold()
    human_record = dict(records[0])
    human_record["assessor_id"] = "human"
    records[0] = dict(records[0])
    records[0]["assessor_id"] = "auto"

    with pytest.raises(
        auto_verify.GoldValidationError,
        match="humain Q021 a été modifié",
    ):
        auto_verify.validate_final_dataset(
            records,
            test_questions=make_questions(),
            chunks_by_id=chunks,
            human_records=[human_record],
        )


def test_atomic_writer_falls_back_after_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "records.jsonl"

    def deny_replace(source: Path, destination: Path) -> None:
        del source, destination
        raise PermissionError("Windows rename denied")

    monkeypatch.setattr(auto_verify.os, "replace", deny_replace)

    auto_verify.write_jsonl_atomic(
        [{"question_id": "Q029", "selected_chunk_ids": ["chunk_1"]}],
        output_path,
    )

    assert [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ] == [
        {
            "question_id": "Q029",
            "selected_chunk_ids": ["chunk_1"],
        }
    ]
    assert list(tmp_path.glob("*.tmp")) == []
