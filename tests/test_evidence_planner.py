"""Strict structured-evidence planner tests."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from phosprocess.evaluation.evidence_planner import (
    EvidencePlan,
    EvidencePlanMode,
    EvidencePlanner,
)
from phosprocess.retrieval.evidence_bundle import EvidenceBundle, EvidenceContextScope


def _bundle(number: int = 1) -> EvidenceBundle:
    return EvidenceBundle(
        source_number=number,
        document_id=f"doc_{number}",
        document_title=f"Document {number}",
        filename=f"doc_{number}.pdf",
        chapter="Chapter",
        section="Section",
        page_start=1,
        page_end=1,
        parent_id=f"parent_{number}",
        anchor_chunk_ids=(f"chunk_{number}",),
        supporting_chunk_ids=(f"chunk_{number}",),
        display_text="Fluid enters A and then leaves B.",
        token_count=20,
        documentary_token_count=10,
        metadata_token_count=10,
        anchor_token_count=10,
        best_anchor_score=1.0,
        context_scope=EvidenceContextScope.ANCHOR_ONLY,
        selection_provenance="test",
    )


class _FakeLLM:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0

    def chat_json_with_raw(self, **kwargs: Any) -> tuple[Any, str]:
        self.calls += 1
        raw = json.dumps(self.payload)
        return kwargs["response_model"].model_validate(self.payload), raw


def test_sequence_plan_requires_contiguous_source_backed_items() -> None:
    plan = EvidencePlan.model_validate(
        {
            "mode": "sequence",
            "question_focus": "fluid path",
            "insufficient_evidence": False,
            "items": [
                {
                    "item_id": "P1",
                    "instruction": "Fluid enters A.",
                    "source_numbers": [1],
                    "sequence_index": 1,
                    "case_id": None,
                    "comparison_side": None,
                },
                {
                    "item_id": "P2",
                    "instruction": "Fluid leaves B.",
                    "source_numbers": [1],
                    "sequence_index": 2,
                    "case_id": None,
                    "comparison_side": None,
                },
            ],
        }
    )
    assert plan.mode is EvidencePlanMode.SEQUENCE
    assert [item.sequence_index for item in plan.items] == [1, 2]


@pytest.mark.parametrize(
    ("mode", "items"),
    [
        (
            "comparison",
            [
                {
                    "item_id": "P1",
                    "instruction": "A fact",
                    "source_numbers": [1],
                    "sequence_index": None,
                    "case_id": None,
                    "comparison_side": "A",
                },
                {
                    "item_id": "P2",
                    "instruction": "B fact",
                    "source_numbers": [2],
                    "sequence_index": None,
                    "case_id": None,
                    "comparison_side": "B",
                },
            ],
        ),
        (
            "multiple_cases",
            [
                {
                    "item_id": "P1",
                    "instruction": "First case",
                    "source_numbers": [1],
                    "sequence_index": None,
                    "case_id": "case_a",
                    "comparison_side": None,
                },
                {
                    "item_id": "P2",
                    "instruction": "Second case",
                    "source_numbers": [2],
                    "sequence_index": None,
                    "case_id": "case_b",
                    "comparison_side": None,
                },
            ],
        ),
        (
            "simple",
            [
                {
                    "item_id": "P1",
                    "instruction": "Direct fact",
                    "source_numbers": [1],
                    "sequence_index": None,
                    "case_id": None,
                    "comparison_side": None,
                },
            ],
        ),
    ],
)
def test_planner_accepts_each_non_sequence_mode(
    mode: str,
    items: list[dict[str, Any]],
) -> None:
    assert EvidencePlan.model_validate(
        {
            "mode": mode,
            "question_focus": "focus",
            "insufficient_evidence": False,
            "items": items,
        }
    ).mode.value == mode


def test_invalid_mode_structure_is_rejected_by_strict_schema() -> None:
    with pytest.raises(ValidationError, match="comparison plans require"):
        EvidencePlan.model_validate(
            {
                "mode": "comparison",
                "question_focus": "compare",
                "insufficient_evidence": False,
                "items": [
                    {
                        "item_id": "P1",
                        "instruction": "Only one side",
                        "source_numbers": [1],
                        "sequence_index": None,
                        "case_id": None,
                        "comparison_side": "A",
                    }
                ],
            }
        )


def test_planner_makes_one_call_and_rejects_unknown_source_without_repair() -> None:
    payload = {
        "mode": "simple",
        "question_focus": "fact",
        "insufficient_evidence": False,
        "items": [
            {
                "item_id": "P1",
                "instruction": "Use an unavailable source.",
                "source_numbers": [2],
                "sequence_index": None,
                "case_id": None,
                "comparison_side": None,
            }
        ],
    }
    llm = _FakeLLM(payload)
    with pytest.raises(ValueError, match="unavailable sources"):
        EvidencePlanner(llm).plan(question="What?", evidence_bundles=[_bundle(1)])
    assert llm.calls == 1
