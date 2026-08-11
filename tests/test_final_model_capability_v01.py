"""Isolation tests for the final single-model capability replay."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from phosprocess.evaluation.context_engine_v01 import read_jsonl
from phosprocess.evaluation.final_model_capability_v01 import (
    CANDIDATE_MODEL,
    PHASE10_OUTPUT,
    _initial_prompt_call,
    _run_one,
)


class _FakeLLM:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] | None = None
        self.calls = 0

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        **_kwargs: Any,
    ) -> Any:
        self.calls += 1
        self.messages = messages
        yield "Réponse documentaire [Source 1]."


@pytest.mark.requires_local_data
def test_final_model_replay_changes_only_model() -> None:
    baseline = read_jsonl(PHASE10_OUTPUT / "dev_generation_results.jsonl")[0]
    llm = _FakeLLM()
    runtime = SimpleNamespace(llm=llm)

    row = _run_one(runtime, baseline)

    assert row["status"] == "completed"
    assert llm.calls == 1
    assert llm.messages == _initial_prompt_call(baseline)["messages"]
    assert row["response"]["model_name"] == CANDIDATE_MODEL
    assert row["response"]["cited_source_numbers"] == [1]
    isolation = row["final_model_experiment"]
    assert isolation["same_question"] is True
    assert isolation["same_evidence"] is True
    assert isolation["same_messages"] is True
    assert isolation["generation_calls"] == 1
    assert isolation["retrieval_calls"] == 0
    assert isolation["planner_calls"] == 0
    assert isolation["evidence_judge_calls"] == 0
    assert isolation["semantic_verifier_calls"] == 0
    assert isolation["repair_calls"] == 0
