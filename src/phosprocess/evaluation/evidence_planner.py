"""Strict LLM evidence planning before grounded answer generation."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from phosprocess.retrieval.evidence_bundle import EvidenceBundle


class EvidencePlanMode(StrEnum):
    """Question/evidence organization selected by the planner."""

    SEQUENCE = "sequence"
    COMPARISON = "comparison"
    MULTIPLE_CASES = "multiple_cases"
    SIMPLE = "simple"


class EvidencePlanItem(BaseModel):
    """One evidence-backed point that the answer must cover."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    source_numbers: list[int] = Field(min_length=1)
    sequence_index: int | None = Field(default=None, ge=1)
    case_id: str | None = None
    comparison_side: str | None = None

    @field_validator("item_id", "instruction", "case_id", "comparison_side")
    @classmethod
    def _strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("plan text fields cannot be blank")
        return stripped

    @field_validator("source_numbers")
    @classmethod
    def _normalize_sources(cls, values: list[int]) -> list[int]:
        if any(value < 1 for value in values):
            raise ValueError("source numbers must be positive")
        return sorted(set(values))


class EvidencePlan(BaseModel):
    """Strict, source-addressable answer plan produced before generation."""

    model_config = ConfigDict(extra="forbid")

    mode: EvidencePlanMode
    question_focus: str = Field(min_length=1)
    insufficient_evidence: bool
    items: list[EvidencePlanItem] = Field(min_length=1)

    @field_validator("question_focus")
    @classmethod
    def _strip_focus(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _validate_mode_contract(self) -> EvidencePlan:
        item_ids = [item.item_id for item in self.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("plan item IDs must be unique")

        if self.insufficient_evidence and self.mode is not EvidencePlanMode.SIMPLE:
            raise ValueError("insufficient evidence requires simple mode")

        if self.mode is EvidencePlanMode.SEQUENCE:
            indexes = [item.sequence_index for item in self.items]
            if any(index is None for index in indexes):
                raise ValueError("sequence items require sequence_index")
            expected = list(range(1, len(self.items) + 1))
            if sorted(indexes) != expected:
                raise ValueError("sequence indexes must be unique and contiguous")

        if self.mode is EvidencePlanMode.COMPARISON:
            sides = {
                item.comparison_side
                for item in self.items
                if item.comparison_side is not None
            }
            if len(sides) < 2:
                raise ValueError("comparison plans require at least two comparison sides")

        if self.mode is EvidencePlanMode.MULTIPLE_CASES:
            cases = {item.case_id for item in self.items if item.case_id is not None}
            if len(cases) < 2 or any(item.case_id is None for item in self.items):
                raise ValueError("multiple-case plans require at least two explicit cases")

        return self


class _OllamaEvidencePlanItem(BaseModel):
    """Grammar-compatible transport shape; strict validation follows parsing."""

    item_id: str
    instruction: str
    source_numbers: list[int]
    sequence_index: int | None = None
    case_id: str | None = None
    comparison_side: str | None = None


class _OllamaEvidencePlan(BaseModel):
    """Ollama transport schema without unsupported grammar constraints."""

    mode: str
    question_focus: str
    insufficient_evidence: bool
    items: list[_OllamaEvidencePlanItem]


def evidence_plan_transport_schema() -> dict[str, Any]:
    """Return the JSON schema actually supplied to Ollama's grammar engine."""

    return _OllamaEvidencePlan.model_json_schema()


PLANNER_SYSTEM_PROMPT = "\n".join(
    (
        "You are the EvidencePlanner for a grounded multilingual RAG system.",
        "Plan an answer from QUESTION and EVIDENCE BUNDLES only. Never answer the question.",
        "Never use outside knowledge and never turn a heading, index term, or keyword into a fact.",
        "",
        "Select exactly one organization mode:",
        (
            "- sequence: the question requests a path, procedure, chronology, or ordered "
            "mechanism and the evidence establishes an order;"
        ),
        (
            "- comparison: the question requests similarities, differences, or a comparison "
            "between alternatives;"
        ),
        (
            "- multiple_cases: relevant evidence describes distinct documentary cases, "
            "examples, equipment variants, or operating conditions that must remain separate;"
        ),
        "- simple: one ordinary definition, explanation, list, equation, or direct factual answer.",
        "",
        (
            "The user's requested focus determines relevance. Include every distinct "
            "supported point needed for that focus and exclude unrelated documentary material."
        ),
        "Every plan item must cite one or more supplied source numbers in source_numbers.",
        "Write item instructions in the language of the user's question.",
        (
            "If no supplied evidence supports the requested focus, set insufficient_evidence "
            "to true, select simple mode, and make the item instruct a controlled refusal."
        ),
        (
            "For sequence mode, assign contiguous sequence_index values starting at 1 and "
            "do not merge distinct routes."
        ),
        "For comparison mode, label the compared side in comparison_side.",
        (
            "For multiple_cases mode, assign a concise stable case_id to every item and "
            "never fabricate a relationship between cases."
        ),
        "Use null for structural fields that do not apply.",
        "Return only one JSON object matching the supplied schema. Do not expose reasoning.",
    )
)


@dataclass(frozen=True, slots=True)
class EvidencePlanExecution:
    """Planner output and reproducibility telemetry for one question."""

    plan: EvidencePlan
    raw_output: str
    latency_ms: float
    system_prompt_sha256: str
    user_prompt_sha256: str


def build_evidence_planner_user_prompt(
    question: str,
    bundles: list[EvidenceBundle],
) -> str:
    """Serialize the exact planner input without adding domain knowledge."""

    evidence = "\n\n".join(bundle.render_prompt_block() for bundle in bundles)
    return f"QUESTION\n{question.strip()}\n\nEVIDENCE BUNDLES\n{evidence}"


class EvidencePlanner:
    """Single-call strict-JSON LLM planner with no verifier or repair."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def plan(
        self,
        *,
        question: str,
        evidence_bundles: list[EvidenceBundle],
    ) -> EvidencePlanExecution:
        if not question.strip():
            raise ValueError("EvidencePlanner requires a non-empty question")
        if not evidence_bundles:
            raise ValueError("EvidencePlanner requires at least one EvidenceBundle")

        user_prompt = build_evidence_planner_user_prompt(question, evidence_bundles)
        started = time.perf_counter()
        _transport, raw_output = self.llm.chat_json_with_raw(
            user_prompt=user_prompt,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            response_model=_OllamaEvidencePlan,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        decoded = json.loads(raw_output)
        plan = EvidencePlan.model_validate(decoded)

        available_sources = {bundle.source_number for bundle in evidence_bundles}
        invalid_sources = sorted(
            {
                source_number
                for item in plan.items
                for source_number in item.source_numbers
                if source_number not in available_sources
            }
        )
        if invalid_sources:
            raise ValueError(
                "EvidencePlanner referenced unavailable sources: "
                + ", ".join(str(value) for value in invalid_sources)
            )

        return EvidencePlanExecution(
            plan=plan,
            raw_output=raw_output,
            latency_ms=latency_ms,
            system_prompt_sha256=hashlib.sha256(
                PLANNER_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            user_prompt_sha256=hashlib.sha256(user_prompt.encode("utf-8")).hexdigest(),
        )
