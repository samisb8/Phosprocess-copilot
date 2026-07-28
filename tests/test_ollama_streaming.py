"""Tests for real Ollama JSONL streaming semantics."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from phosprocess.llm.ollama_client import (
    OllamaConfig,
    OllamaHTTPError,
    OllamaLLM,
    OllamaStreamInterruptedError,
    OllamaTimeoutError,
    parse_ollama_jsonl,
)
from phosprocess.observability.latency import OllamaCallMetrics


class FakeStreamingResponse:
    """Minimal context-managed streaming HTTP response."""

    def __init__(
        self,
        chunks: list[bytes | Exception],
        *,
        status_code: int = 200,
    ) -> None:
        self.chunks = chunks
        self.status_code = status_code
        self.request = httpx.Request(
            "POST",
            "http://localhost:11434/api/chat",
        )

    def __enter__(self) -> FakeStreamingResponse:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            response = httpx.Response(
                self.status_code,
                request=self.request,
            )
            raise httpx.HTTPStatusError(
                "HTTP failure",
                request=self.request,
                response=response,
            )

    def iter_bytes(self) -> Iterator[bytes]:
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk

            yield chunk


class FakeStreamingClient:
    """Record the direct /api/chat request."""

    def __init__(self, response: FakeStreamingResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def stream(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> FakeStreamingResponse:
        self.calls.append((method, url, kwargs))
        return self.response


def make_llm(
    response: FakeStreamingResponse,
) -> tuple[OllamaLLM, FakeStreamingClient]:
    """Create an Ollama client with an injected streaming HTTP client."""

    client = FakeStreamingClient(response)
    llm = OllamaLLM(
        OllamaConfig(
            model="qwen-test",
            temperature=0.1,
            timeout_seconds=2.0,
        ),
        stream_http_client=client,
    )
    return llm, client


def test_fragmented_jsonl_normal_stream_ignores_empty_events() -> None:
    chunks = [
        b'{"message":{"content":"Bon',
        b'jour "}}\n\n{}\n{"message":{"content":""}}\n',
        '{"message":{"content":"procédé"}}\n'.encode(),
        b'{"done":true}\n',
    ]
    llm, client = make_llm(FakeStreamingResponse(chunks))

    tokens = list(
        llm.stream_chat(
            [{"role": "user", "content": "Question"}]
        )
    )

    assert tokens == ["Bonjour ", "procédé"]
    method, url, call = client.calls[0]
    assert method == "POST"
    assert url == "http://localhost:11434/api/chat"
    assert call["json"]["stream"] is True
    assert call["json"]["model"] == "qwen-test"
    assert call["json"]["options"]["temperature"] == 0.1
    assert call["json"]["options"]["num_predict"] == 300
    assert call["json"]["options"]["num_gpu"] == 12
    assert call["json"]["keep_alive"] == "30m"
    assert call["json"]["think"] is False


def test_telemetry_measures_first_event_first_token_and_counts() -> None:
    llm, _ = make_llm(
        FakeStreamingResponse(
            [
                b"{}\n",
                b'{"message":{"content":""}}\n',
                b'{"message":{"content":"R\xc3\xa9ponse"}}\n',
                (
                    b'{"done":true,"eval_count":3,'
                    b'"prompt_eval_count":17,'
                    b'"load_duration":1000000,'
                    b'"prompt_eval_duration":2000000,'
                    b'"eval_duration":3000000}\n'
                ),
            ]
        )
    )
    telemetry = OllamaCallMetrics(
        call_type="generation_main",
        model="qwen-test",
        streaming=True,
    )

    assert list(
        llm.stream_chat(
            [{"role": "user", "content": "Question"}],
            telemetry=telemetry,
        )
    ) == ["Réponse"]
    assert telemetry.success is True
    assert telemetry.error_type is None
    assert telemetry.duration_ms > 0
    assert telemetry.time_to_first_event_ms > 0
    assert (
        telemetry.time_to_first_token_ms
        >= telemetry.time_to_first_event_ms
    )
    assert telemetry.generated_character_count == len("Réponse")
    assert telemetry.generated_chunk_count == 1
    assert telemetry.generated_token_count == 3
    assert telemetry.prompt_token_count == 17
    assert telemetry.model_load_ms == 1.0
    assert telemetry.prompt_evaluation_ms == 2.0
    assert telemetry.model_generation_ms == 3.0
    assert telemetry.generation_tokens_per_second == 1000.0


def test_failed_stream_finalizes_diagnostic_telemetry() -> None:
    llm, _ = make_llm(
        FakeStreamingResponse([httpx.ReadTimeout("timeout")])
    )
    telemetry = OllamaCallMetrics(
        call_type="generation_main",
        model="qwen-test",
        streaming=True,
    )

    with pytest.raises(OllamaTimeoutError):
        list(
            llm.stream_chat(
                [{"role": "user", "content": "Question"}],
                telemetry=telemetry,
            )
        )

    assert telemetry.success is False
    assert telemetry.error_type == "ReadTimeout"
    assert telemetry.duration_ms > 0


def test_done_true_stops_before_later_events() -> None:
    llm, _ = make_llm(
        FakeStreamingResponse(
            [
                b'{"message":{"content":"A"}}\n',
                b'{"done":true}\n',
                b'{"message":{"content":"B"}}\n',
            ]
        )
    )

    assert list(
        llm.stream_chat(
            [{"role": "user", "content": "Question"}]
        )
    ) == ["A"]


def test_timeout_during_stream_is_normalized() -> None:
    llm, _ = make_llm(
        FakeStreamingResponse(
            [httpx.ReadTimeout("timeout")]
        )
    )

    with pytest.raises(OllamaTimeoutError):
        list(
            llm.stream_chat(
                [{"role": "user", "content": "Question"}]
            )
        )


def test_http_error_is_normalized() -> None:
    llm, _ = make_llm(
        FakeStreamingResponse([], status_code=503)
    )

    with pytest.raises(OllamaHTTPError, match="503"):
        list(
            llm.stream_chat(
                [{"role": "user", "content": "Question"}]
            )
        )


def test_stream_without_done_is_rejected_as_interrupted() -> None:
    llm, _ = make_llm(
        FakeStreamingResponse(
            [b'{"message":{"content":"partiel"}}\n']
        )
    )

    with pytest.raises(OllamaStreamInterruptedError):
        list(
            llm.stream_chat(
                [{"role": "user", "content": "Question"}]
            )
        )


def test_jsonl_parser_handles_fragmented_multibyte_character() -> None:
    payload = (
        '{"message":{"content":"réacteur"}}\n'
        '{"done":true}\n'
    ).encode()
    split_index = payload.index("é".encode()) + 1

    events = list(
        parse_ollama_jsonl(
            [payload[:split_index], payload[split_index:]]
        )
    )

    assert events[0]["message"]["content"] == "réacteur"
    assert events[1]["done"] is True
