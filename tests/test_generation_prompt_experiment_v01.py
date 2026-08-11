"""Phase-11 prompt replay and decision-boundary tests."""

from __future__ import annotations

from phosprocess.evaluation.generation_prompt_experiment_v01 import (
    VARIANT,
    _parse_requirement_plan,
)
from phosprocess.rag.language import ResponseLanguage
from phosprocess.rag.prompts import build_quality_prompt_package
from phosprocess.rag.question_classifier import classify_question
from phosprocess.retrieval.evidence_bundle import EvidenceBundle, EvidenceContextScope


def _bundle() -> EvidenceBundle:
    return EvidenceBundle(
        source_number=1,
        document_id="doc",
        document_title="Document",
        filename="doc.pdf",
        chapter="Chapter",
        section="Section",
        page_start=1,
        page_end=1,
        parent_id="parent",
        anchor_chunk_ids=("chunk",),
        supporting_chunk_ids=("chunk",),
        display_text="A fluid enters A and then leaves B.",
        token_count=20,
        documentary_token_count=10,
        metadata_token_count=10,
        anchor_token_count=10,
        best_anchor_score=1.0,
        context_scope=EvidenceContextScope.ANCHOR_ONLY,
        selection_provenance="test",
    )


def test_requirement_plan_parser_reconstructs_frozen_plan() -> None:
    prompt = (
        "QUESTION\nWhat happens?\n\n"
        "PRECOMPUTED EVIDENCE COVERAGE PLAN\n"
        "Focus: fluid path\n"
        "R1 | sources=1 | order=1 | fluid enters A\n"
        "R2 | sources=1 | order=2 | fluid leaves B\n\n"
        "Use this plan only as a coverage guide."
    )
    plan = _parse_requirement_plan(prompt)
    assert plan is not None
    assert plan.focus == "fluid path"
    assert [item.sequence_index for item in plan.requirements] == [1, 2]
    assert [item.source_numbers for item in plan.requirements] == [[1], [1]]


def test_phase11_variant_changes_only_system_side_of_frozen_replay() -> None:
    kwargs = {
        "response_language": ResponseLanguage.ENGLISH,
        "classification": classify_question("Describe the path."),
        "json_output": False,
    }
    baseline_system, baseline = build_quality_prompt_package(
        "Describe the path.", [_bundle()], **kwargs
    )
    variant_system, variant = build_quality_prompt_package(
        "Describe the path.",
        [_bundle()],
        **kwargs,
        prompt_variant=VARIANT,
    )
    assert baseline.user_prompt == variant.user_prompt
    assert baseline_system != variant_system
    assert "never concatenate distinct documentary sequences" in variant_system
