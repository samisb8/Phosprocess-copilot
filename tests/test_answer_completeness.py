from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from phosprocess.evaluation.legacy_answer_validation_service import (
    AnswerValidationService,
    _AnswerAuditPayload,
    _EvidenceRequirement,
    _EvidenceRequirementPlan,
)
from phosprocess.rag.citations import CitationValidationError


class _FakeLLM:
    def __init__(self) -> None:
        self.planner_focus = "Requested entity path"
        self.planner_requirements = [
            {
                "description": "First material transition",
                "source_numbers": [1],
                "sequence_index": 1,
            },
            {
                "description": "Second material transition",
                "source_numbers": [2],
                "sequence_index": 2,
            },
        ]

        self.audit_grounded = True
        self.audit_complete = True
        self.audit_missing: list[str] = []
        self.audit_unsupported: list[str] = []

        self.last_system_prompt = ""
        self.last_user_prompt = ""

    def chat_json_with_raw(
        self,
        **kwargs: Any,
    ) -> tuple[Any, str]:
        self.last_system_prompt = kwargs["system_prompt"]
        self.last_user_prompt = kwargs["user_prompt"]

        model = kwargs["response_model"]

        if model is _EvidenceRequirementPlan:
            return (
                model(
                    focus=self.planner_focus,
                    requirements=self.planner_requirements,
                ),
                "{}",
            )

        if model is _AnswerAuditPayload:
            return (
                model(
                    grounded=self.audit_grounded,
                    complete=self.audit_complete,
                    missing_requirement_ids=self.audit_missing,
                    unsupported_claims=self.audit_unsupported,
                ),
                "{}",
            )

        raise AssertionError(f"Unexpected model: {model}")


class _Harness(AnswerValidationService):
    def __init__(self) -> None:
        self.llm = _FakeLLM()


def _bundle(number: int, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        source_number=number,
        display_text=text,
    )


def _fixed_plan() -> _EvidenceRequirementPlan:
    return _EvidenceRequirementPlan(
        focus="Requested entity path",
        requirements=[
            _EvidenceRequirement(
                description="First material transition",
                source_numbers=[1],
                sequence_index=1,
            ),
            _EvidenceRequirement(
                description="Second material transition",
                source_numbers=[2],
                sequence_index=2,
            ),
        ],
    )


def test_requirements_are_planned_before_answer() -> None:
    validator = _Harness()

    plan = validator._plan_answer_requirements(
        question="Describe the complete documented path.",
        evidence_bundles=[
            _bundle(1, "Evidence for the first transition."),
            _bundle(2, "Evidence for the second transition."),
        ],
    )

    assert plan is not None
    assert len(plan.requirements) == 2

    prompt = validator.llm.last_user_prompt

    assert "ANSWER" not in prompt
    assert "QUESTION" in prompt
    assert "EVIDENCE" in prompt


def test_requirement_plan_is_added_to_generation_prompt() -> None:
    validator = _Harness()

    augmented = validator._append_requirement_plan(
        "ORIGINAL PROMPT",
        _fixed_plan(),
    )

    assert "PRECOMPUTED EVIDENCE COVERAGE PLAN" in augmented
    assert "R1" in augmented
    assert "R2" in augmented
    assert "First material transition" in augmented
    assert "Second material transition" in augmented


def test_semantic_audit_accepts_fixed_plan_coverage() -> None:
    validator = _Harness()

    validator._validate_answer_semantics(
        question="Describe the documented path.",
        answer=(
            "The first transition occurs [Source 1]. "
            "The second transition occurs [Source 2]."
        ),
        evidence_bundles=[
            _bundle(1, "Evidence for the first transition."),
            _bundle(2, "Evidence for the second transition."),
        ],
        requirement_plan=_fixed_plan(),
    )

    assert "PRECOMPUTED REQUIREMENTS" in (
        validator.llm.last_user_prompt
    )


def test_semantic_audit_rejects_missing_fixed_requirement() -> None:
    validator = _Harness()
    validator.llm.audit_complete = False
    validator.llm.audit_missing = ["R2"]

    with pytest.raises(
        CitationValidationError,
        match="R2: Second material transition",
    ):
        validator._validate_answer_semantics(
            question="Describe the documented path.",
            answer="Only the first transition [Source 1].",
            evidence_bundles=[
                _bundle(1, "Evidence for the first transition."),
                _bundle(2, "Evidence for the second transition."),
            ],
            requirement_plan=_fixed_plan(),
        )


def test_semantic_audit_overrides_lexical_false_negative() -> None:
    validator = _Harness()

    validator._validate_answer_semantics(
        question="Describe the documented path.",
        answer="A faithful paraphrase [Source 1].",
        evidence_bundles=[
            _bundle(
                1,
                "The same fact expressed using different wording.",
            )
        ],
        lexical_rejection="Low lexical overlap.",
        requirement_plan=_EvidenceRequirementPlan(
            focus="Requested fact",
            requirements=[
                _EvidenceRequirement(
                    description="The supported fact",
                    source_numbers=[1],
                )
            ],
        ),
    )


def test_planner_prompt_contains_no_domain_answer() -> None:
    validator = _Harness()

    validator._plan_answer_requirements(
        question="Describe a generic sequence.",
        evidence_bundles=[
            _bundle(
                1,
                "Generic evidence describing a transition.",
            )
        ],
    )

    prompt = (
        validator.llm.last_system_prompt
        + "\n"
        + validator.llm.last_user_prompt
    ).casefold()

    forbidden = (
        "conical bottom",
        "feed inlet",
        "product outlet",
        "phosphoric acid",
        "vapor body",
    )

    assert not any(term in prompt for term in forbidden)



def test_planner_requires_cross_source_focus() -> None:
    validator = _Harness()

    validator._plan_answer_requirements(
        question="Describe the path of the tracked material.",
        evidence_bundles=[
            _bundle(
                1,
                "One source describes one material transition.",
            ),
            _bundle(
                2,
                "Another source describes a different material transition.",
            ),
        ],
    )

    system = validator.llm.last_system_prompt

    assert (
        "Do not let a coherent narrative from one source suppress"
        in system
    )

    assert (
        "keep the requirements centered on that entity"
        in system
    )

    assert (
        "secondary stream"
        in system
    )

    assert (
        "plant-specific operating values"
        in system
    )

    assert (
        "Compare all evidence bundles"
        in system
    )
