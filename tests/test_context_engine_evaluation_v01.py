"""Deterministic guards for the evaluation-only Phase-5 harness."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from phosprocess.evaluation.context_engine_v01 import (
    ACTIVE_VERSION,
    _coverage,
    _rank,
    _recall,
    build_dataset,
    normalize_text,
)
from phosprocess.ingestion.chunk_serialization import read_child_chunks
from phosprocess.retrieval.evidence_bundle import EvidenceBundle

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def bundle(text: str, supporting: tuple[str, ...]) -> EvidenceBundle:
    return EvidenceBundle(
        source_number=1,
        document_id="document",
        document_title="Document",
        filename="document.pdf",
        chapter="Chapter",
        section="Section",
        page_start=4,
        page_end=4,
        parent_id="parent",
        anchor_chunk_ids=(supporting[0],),
        supporting_chunk_ids=supporting,
        display_text=text,
        token_count=30,
        documentary_token_count=10,
        metadata_token_count=20,
        anchor_token_count=5,
        context_token_count=5,
        best_anchor_score=0.9,
        context_scope="partial_parent",
        selection_provenance="reranker",
    )


@pytest.mark.requires_local_data
def test_phase5_dataset_covers_documents_languages_and_requested_types() -> None:
    chunks_path = (
        PROJECT_ROOT
        / "data/knowledge_base/indexes/versions"
        / ACTIVE_VERSION
        / "chunks.jsonl"
    )
    records = build_dataset(read_child_chunks(chunks_path))
    languages = Counter(record["language"] for record in records)
    types = {record["question_type"] for record in records}
    documents = {
        document
        for record in records
        for document in record["expected_document_ids"]
    }

    assert 50 <= len(records) <= 80
    assert set(languages) == {"fr", "en", "ar"}
    assert len(documents) == 8
    assert {
        "definition",
        "process_flow",
        "equipment_operation",
        "equation_explanation",
        "material_balance",
        "energy_balance",
        "plant_numerical_data",
        "troubleshooting",
        "comparison",
        "explicit_source",
        "follow_up",
        "absent_from_corpus",
    }.issubset(types)


def test_normalization_is_accent_and_punctuation_insensitive() -> None:
    assert normalize_text("Capacité évaporatoire—P₂O₅") == (
        normalize_text("capacite evaporatoire P2O5")
    )


def test_rank_and_recall_are_deterministic() -> None:
    ranked = ["noise", "gold_a", "noise_2", "gold_b"]
    expected = {"gold_a", "gold_b"}

    assert _rank(ranked, expected) == 2
    assert _recall(ranked, expected, 2) == 0.5
    assert _recall(ranked, expected, 4) == 1.0
    assert _recall(ranked, set(), 4) is None


def test_coverage_uses_only_evaluation_truth() -> None:
    record = {
        "answerable": True,
        "expected_evidence_chunk_ids": ["gold"],
        "expected_concepts": ["latent heat"],
        "expected_pages": [4],
        "expected_sections": ["Section"],
    }

    assert _coverage(record, [bundle("Latent heat is supplied.", ("gold",))]) == "full"
    assert _coverage(record, [bundle("Unrelated evidence.", ("gold",))]) == "partial"
    assert _coverage(record, [bundle("Unrelated evidence.", ("other",))]) == "partial"


def test_production_modules_do_not_import_phase5_evaluation() -> None:
    production_roots = [
        PROJECT_ROOT / "src/phosprocess/rag",
        PROJECT_ROOT / "src/phosprocess/retrieval",
    ]
    offenders = []
    for root in production_roots:
        for path in root.glob("*.py"):
            if "context_engine_v01" in path.read_text(encoding="utf-8"):
                offenders.append(path)

    assert offenders == []
