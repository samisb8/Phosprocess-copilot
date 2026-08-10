"""Unit tests for the blocking local Ollama client."""

from __future__ import annotations

import json
from pathlib import Path
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
    load_ollama_config,
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


def test_ollama_host_environment_variable_overrides_yaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Allow Docker to override the local Ollama endpoint."""
    config_path = tmp_path / "rag_production.yaml"
    config_path.write_text(
        """
ollama:
  host: http://localhost:11434
  model: qwen-test
  temperature: 0.1
  context_size: 4096
  max_output_tokens: 256
  timeout_seconds: 3.0
  keep_alive: 1m
  seed: 0
  num_gpu: 0
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    local_config = load_ollama_config(config_path)
    assert local_config.host == "http://localhost:11434"

    monkeypatch.setenv(
        "OLLAMA_HOST",
        "http://host.docker.internal:11434",
    )
    docker_config = load_ollama_config(config_path)

    assert docker_config.host == (
        "http://host.docker.internal:11434"
    )
