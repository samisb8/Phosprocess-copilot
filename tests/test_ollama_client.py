"""Unit tests for the blocking local Ollama client."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from httpx import TimeoutException

from phosprocess.llm.ollama_client import (
    OllamaConfig,
    OllamaConnectionError,
    OllamaLLM,
    OllamaResponseValidationError,
    OllamaTimeoutError,
)
from phosprocess.rag.schemas import GroundedAnswerPayload


class FakeOllamaClient:
    """Record calls and return a configurable Ollama-like response."""

    def __init__(
        self,
        content: str = "",
        error: Exception | None = None,
    ) -> None:
        self.content = content
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)

        if self.error is not None:
            raise self.error

        return SimpleNamespace(
            message=SimpleNamespace(content=self.content)
        )


def make_config() -> OllamaConfig:
    """Create deterministic inference settings for tests."""

    return OllamaConfig(
        model="qwen-test",
        temperature=0.05,
        context_size=4096,
        max_output_tokens=256,
        timeout_seconds=3.0,
        keep_alive="1m",
        seed=17,
    )


def test_negative_gpu_offload_is_rejected() -> None:
    with pytest.raises(ValueError, match="num_gpu"):
        OllamaConfig(num_gpu=-1)


def test_chat_json_uses_answer_only_schema_and_options() -> None:
    content = json.dumps(
        {"answer": "Réponse fondée [Source 1]."}
    )
    fake = FakeOllamaClient(content)
    llm = OllamaLLM(make_config(), client=fake)

    payload, raw = llm.chat_json_with_raw(
        user_prompt="Question",
        system_prompt="Système",
        response_model=GroundedAnswerPayload,
    )

    assert payload.answer == "Réponse fondée [Source 1]."
    assert raw == content
    call = fake.calls[0]
    assert call["model"] == "qwen-test"
    assert call["think"] is False
    assert call["format"] == GroundedAnswerPayload.model_json_schema()
    assert call["options"] == {
        "temperature": 0.05,
        "num_ctx": 4096,
        "num_predict": 256,
        "seed": 17,
        "num_gpu": 12,
    }


@pytest.mark.parametrize(
    "content",
    [
        "pas du JSON",
        "[]",
        "{}",
        '{"answer": "x", "cited_sources": [1]}',
    ],
)
def test_chat_json_rejects_invalid_payload_and_keeps_raw(
    content: str,
) -> None:
    llm = OllamaLLM(
        make_config(),
        client=FakeOllamaClient(content),
    )

    with pytest.raises(
        OllamaResponseValidationError
    ) as captured:
        llm.chat_json(
            user_prompt="Question",
            system_prompt="Système",
            response_model=GroundedAnswerPayload,
        )

    assert captured.value.raw_response == content


def test_timeout_is_normalized() -> None:
    llm = OllamaLLM(
        make_config(),
        client=FakeOllamaClient(
            error=TimeoutException("timeout"),
        ),
    )

    with pytest.raises(OllamaTimeoutError):
        llm.chat_json(
            user_prompt="Question",
            system_prompt="Système",
            response_model=GroundedAnswerPayload,
        )


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("offline"),
        OSError("socket unavailable"),
    ],
)
def test_connection_failures_are_normalized(error: Exception) -> None:
    llm = OllamaLLM(
        make_config(),
        client=FakeOllamaClient(error=error),
    )

    with pytest.raises(OllamaConnectionError):
        llm.chat_json(
            user_prompt="Question",
            system_prompt="Système",
            response_model=GroundedAnswerPayload,
        )
