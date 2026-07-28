"""Validated local Ollama client used by the production RAG pipeline."""

from __future__ import annotations

import codecs
import json
import logging
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import httpx
import yaml
from ollama import Client, ResponseError
from pydantic import BaseModel, ValidationError

from phosprocess.observability.latency import (
    OllamaCallMetrics,
    estimate_tokens,
)

LOGGER = logging.getLogger(__name__)
ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class OllamaError(RuntimeError):
    """Base class for local generation failures."""


class OllamaConnectionError(OllamaError):
    """Raised when the local Ollama server cannot be reached."""


class OllamaTimeoutError(OllamaError):
    """Raised when local generation exceeds the configured timeout."""


class OllamaResponseValidationError(OllamaError):
    """Raised when Qwen does not return the required JSON payload."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: str | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response


class OllamaHTTPError(OllamaError):
    """Raised when the Ollama streaming endpoint returns an HTTP error."""


class OllamaStreamInterruptedError(OllamaError):
    """Raised when a JSONL stream ends before Ollama emits done=true."""


@dataclass(frozen=True, slots=True)
class OllamaConfig:
    """Runtime configuration for the local Qwen model."""

    host: str = "http://localhost:11434"
    model: str = "qwen3:8b"
    temperature: float = 0.1
    context_size: int = 8192
    max_output_tokens: int = 300
    timeout_seconds: float = 120.0
    keep_alive: str = "30m"
    seed: int = 0
    num_gpu: int | None = 12

    def __post_init__(self) -> None:
        """Validate generation settings before creating a client."""

        if not self.host.strip():
            raise ValueError("L'hôte Ollama ne peut pas être vide.")

        if not self.model.strip():
            raise ValueError("Le nom du modèle Qwen ne peut pas être vide.")

        if not 0 <= self.temperature <= 1:
            raise ValueError(
                "La température doit être comprise entre 0 et 1."
            )

        if self.context_size <= 0:
            raise ValueError(
                "context_size doit être strictement positif."
            )

        if self.max_output_tokens <= 0:
            raise ValueError(
                "max_output_tokens doit être strictement positif."
            )

        if self.timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds doit être strictement positif."
            )

        if self.num_gpu is not None and self.num_gpu < 0:
            raise ValueError("num_gpu doit être positif ou nul.")


def load_ollama_config(path: Path) -> OllamaConfig:
    """Load the Ollama section of the production RAG configuration."""

    if not path.is_file():
        raise FileNotFoundError(path)

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ValueError("Configuration RAG invalide.")

    ollama = raw.get("ollama")

    if not isinstance(ollama, dict):
        raise ValueError("Section ollama absente ou invalide.")

    return OllamaConfig(
        host=str(ollama["host"]),
        model=str(ollama["model"]),
        temperature=float(ollama["temperature"]),
        context_size=int(ollama["context_size"]),
        max_output_tokens=int(ollama["max_output_tokens"]),
        timeout_seconds=float(ollama["timeout_seconds"]),
        keep_alive=str(ollama.get("keep_alive", "10m")),
        seed=int(ollama.get("seed", 0)),
        num_gpu=(
            int(ollama["num_gpu"])
            if ollama.get("num_gpu") is not None
            else None
        ),
    )


class OllamaLLM:
    """Centralized, timeout-aware interface to a local Ollama server."""

    def __init__(
        self,
        config: OllamaConfig | None = None,
        *,
        client: Any | None = None,
        stream_http_client: Any | None = None,
    ) -> None:
        self.config = config or OllamaConfig()
        self.client = client or Client(
            host=self.config.host,
            timeout=self.config.timeout_seconds,
        )
        self.stream_http_client = (
            stream_http_client
            if stream_http_client is not None
            else httpx.Client(
                timeout=httpx.Timeout(
                    self.config.timeout_seconds
                )
            )
        )
        self._owns_stream_http_client = (
            stream_http_client is None
        )

    @property
    def model_name(self) -> str:
        """Return the configured local model name."""

        return self.config.model

    def chat(self, question: str, system_prompt: str) -> str:
        """Send a plain-text request while preserving the legacy API."""

        content = self._chat(
            user_prompt=question,
            system_prompt=system_prompt,
            response_format=None,
        )
        return content.strip()

    def chat_json(
        self,
        *,
        user_prompt: str,
        system_prompt: str,
        response_model: type[ResponseModel],
    ) -> ResponseModel:
        """Request strict JSON and validate it with a Pydantic model."""

        payload, _ = self.chat_json_with_raw(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            response_model=response_model,
        )
        return payload

    def chat_json_with_raw(
        self,
        *,
        user_prompt: str,
        system_prompt: str,
        response_model: type[ResponseModel],
    ) -> tuple[ResponseModel, str]:
        """Return a validated JSON payload together with the raw model output."""

        content = self._chat(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            response_format=response_model.model_json_schema(),
        )

        try:
            decoded = json.loads(content)
        except json.JSONDecodeError as error:
            raise OllamaResponseValidationError(
                "Qwen n'a pas retourné un JSON valide.",
                raw_response=content,
            ) from error

        if not isinstance(decoded, dict):
            raise OllamaResponseValidationError(
                "La réponse JSON de Qwen doit être un objet.",
                raw_response=content,
            )

        try:
            payload = response_model.model_validate(decoded)
        except ValidationError as error:
            raise OllamaResponseValidationError(
                "La réponse JSON de Qwen ne respecte pas le schéma.",
                raw_response=content,
            ) from error

        return payload, content

    def stream_chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str | None = None,
        timeout_seconds: float | None = None,
        max_output_tokens: int | None = None,
        call_type: str = "generation",
        telemetry: OllamaCallMetrics | None = None,
    ) -> Iterator[str]:
        """Yield real token fragments from Ollama's streaming JSONL endpoint."""

        if not messages:
            raise ValueError("Le flux Ollama exige au moins un message.")

        for message in messages:
            if (
                message.get("role") not in {"system", "user", "assistant"}
                or not str(message.get("content", "")).strip()
            ):
                raise ValueError("Message Ollama de streaming invalide.")

        effective_model = model or self.config.model
        effective_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self.config.timeout_seconds
        )

        if effective_timeout <= 0:
            raise ValueError("Le timeout de streaming doit être positif.")

        effective_max_output = (
            max_output_tokens
            if max_output_tokens is not None
            else self.config.max_output_tokens
        )

        if effective_max_output <= 0:
            raise ValueError(
                "max_output_tokens de streaming doit être positif."
            )

        prompt_characters = sum(
            len(message["content"])
            for message in messages
        )
        call_metrics = telemetry or OllamaCallMetrics(
            call_type=call_type,
            model=effective_model,
            streaming=True,
        )
        call_metrics.call_type = call_type
        call_metrics.model = effective_model
        call_metrics.streaming = True
        call_metrics.prompt_character_count = prompt_characters
        call_metrics.estimated_prompt_tokens = estimate_tokens(
            "\n".join(message["content"] for message in messages)
        )
        payload = {
            "model": effective_model,
            "messages": list(messages),
            "stream": True,
            "think": False,
            "keep_alive": self.config.keep_alive,
            "options": {
                "temperature": self.config.temperature,
                "num_ctx": self.config.context_size,
                "num_predict": effective_max_output,
                "seed": self.config.seed,
            },
        }

        if self.config.num_gpu is not None:
            payload["options"]["num_gpu"] = self.config.num_gpu
        endpoint = self.config.host.rstrip("/") + "/api/chat"
        timeout = httpx.Timeout(effective_timeout)
        client = self.stream_http_client
        completed = False
        accumulated: list[str] = []
        started = time.perf_counter()
        first_token_at: float | None = None
        metrics_finalized = False

        def finalize_metrics(
            *,
            success: bool,
            error_type: str | None = None,
        ) -> None:
            """Finalize telemetry on success, failure, or interruption."""

            nonlocal metrics_finalized

            if metrics_finalized:
                return

            ended = time.perf_counter()
            call_metrics.duration_ms = (ended - started) * 1000.0
            call_metrics.generated_character_count = len(
                "".join(accumulated)
            )
            call_metrics.generation_ms = (
                (ended - first_token_at) * 1000.0
                if first_token_at is not None
                else 0.0
            )
            call_metrics.success = success

            if error_type is not None:
                call_metrics.error_type = error_type

            metrics_finalized = True

        LOGGER.info(
            "Streaming Ollama call_type=%s model=%s timeout=%.1fs "
            "prompt_chars=%d",
            call_type,
            effective_model,
            effective_timeout,
            prompt_characters,
        )

        try:
            connection_started = time.perf_counter()

            with client.stream(
                "POST",
                endpoint,
                json=payload,
                timeout=timeout,
            ) as response:
                response.raise_for_status()
                call_metrics.connection_ms = (
                    time.perf_counter() - connection_started
                ) * 1000.0

                for event in parse_ollama_jsonl(response.iter_bytes()):
                    event_time = time.perf_counter()

                    if call_metrics.time_to_first_event_ms == 0.0:
                        call_metrics.time_to_first_event_ms = (
                            event_time - started
                        ) * 1000.0

                    if event.get("error"):
                        raise OllamaError(
                            f"Erreur de flux Ollama: {event['error']}"
                        )

                    message = event.get("message")
                    content = (
                        message.get("content")
                        if isinstance(message, dict)
                        else None
                    )

                    if isinstance(content, str) and content:
                        if (
                            first_token_at is None
                            and content.strip()
                        ):
                            first_token_at = event_time
                            call_metrics.time_to_first_token_ms = (
                                event_time - started
                            ) * 1000.0

                        accumulated.append(content)
                        call_metrics.generated_chunk_count += 1
                        yield content

                    if event.get("done") is True:
                        if isinstance(event.get("eval_count"), int):
                            call_metrics.generated_token_count = int(
                                event["eval_count"]
                            )

                        if isinstance(
                            event.get("prompt_eval_count"),
                            int,
                        ):
                            call_metrics.prompt_token_count = int(
                                event["prompt_eval_count"]
                            )

                        for event_field, metric_field in (
                            ("load_duration", "model_load_ms"),
                            (
                                "prompt_eval_duration",
                                "prompt_evaluation_ms",
                            ),
                            (
                                "eval_duration",
                                "model_generation_ms",
                            ),
                        ):
                            duration = event.get(event_field)

                            if isinstance(duration, int | float):
                                setattr(
                                    call_metrics,
                                    metric_field,
                                    float(duration) / 1_000_000.0,
                                )

                        if (
                            call_metrics.generated_token_count is not None
                            and call_metrics.model_generation_ms > 0
                        ):
                            call_metrics.generation_tokens_per_second = (
                                call_metrics.generated_token_count
                                / (
                                    call_metrics.model_generation_ms
                                    / 1000.0
                                )
                            )

                        completed = True
                        break
        except httpx.TimeoutException as error:
            finalize_metrics(
                success=False,
                error_type=type(error).__name__,
            )
            raise OllamaTimeoutError(
                "Le délai maximal du flux Ollama a été dépassé."
            ) from error
        except httpx.HTTPStatusError as error:
            finalize_metrics(
                success=False,
                error_type=type(error).__name__,
            )
            raise OllamaHTTPError(
                f"Ollama a retourné le statut HTTP "
                f"{error.response.status_code}."
            ) from error
        except (httpx.HTTPError, OSError) as error:
            finalize_metrics(
                success=False,
                error_type=type(error).__name__,
            )
            raise OllamaStreamInterruptedError(
                "Le flux Ollama a été interrompu."
            ) from error
        except (OllamaError, OllamaResponseValidationError) as error:
            finalize_metrics(
                success=False,
                error_type=type(error).__name__,
            )
            raise
        except (GeneratorExit, KeyboardInterrupt) as error:
            finalize_metrics(
                success=False,
                error_type=type(error).__name__,
            )
            raise
        except Exception as error:
            finalize_metrics(
                success=False,
                error_type=type(error).__name__,
            )
            raise OllamaStreamInterruptedError(
                "Le flux Ollama a échoué de manière inattendue."
            ) from error

        if not completed:
            finalize_metrics(
                success=False,
                error_type="missing_done_event",
            )
            raise OllamaStreamInterruptedError(
                "Le flux Ollama s'est terminé sans événement done=true."
            )

        finalize_metrics(success=True)
        LOGGER.info(
            "Ollama stream completed call_type=%s model=%s chars=%d "
            "duration_ms=%.1f",
            call_type,
            effective_model,
            len("".join(accumulated)),
            call_metrics.duration_ms,
        )

    def close(self) -> None:
        """Close the persistent HTTP streaming client owned by this wrapper."""

        if self._owns_stream_http_client:
            self.stream_http_client.close()

    def _chat(
        self,
        *,
        user_prompt: str,
        system_prompt: str,
        response_format: dict[str, Any] | None,
    ) -> str:
        """Execute one local chat request with normalized failures."""

        normalized_prompt = user_prompt.strip()
        normalized_system = system_prompt.strip()

        if not normalized_prompt:
            raise ValueError("Le prompt utilisateur ne peut pas être vide.")

        if not normalized_system:
            raise ValueError("Le prompt système ne peut pas être vide.")

        LOGGER.info(
            "Calling Ollama model=%s timeout=%.1fs json=%s",
            self.config.model,
            self.config.timeout_seconds,
            response_format is not None,
        )
        started = time.perf_counter()

        try:
            response = self.client.chat(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": normalized_system,
                    },
                    {
                        "role": "user",
                        "content": normalized_prompt,
                    },
                ],
                think=False,
                format=response_format,
                keep_alive=self.config.keep_alive,
                options={
                    "temperature": self.config.temperature,
                    "num_ctx": self.config.context_size,
                    "num_predict": self.config.max_output_tokens,
                    "seed": self.config.seed,
                    **(
                        {"num_gpu": self.config.num_gpu}
                        if self.config.num_gpu is not None
                        else {}
                    ),
                },
            )
        except httpx.TimeoutException as error:
            raise OllamaTimeoutError(
                "Le délai maximal de génération Ollama a été dépassé."
            ) from error
        except (ConnectionError, OSError) as error:
            raise OllamaConnectionError(
                "Impossible de joindre Ollama. Lancez `ollama serve`."
            ) from error
        except ResponseError as error:
            raise OllamaError(
                f"Erreur Ollama {error.status_code}: {error.error}"
            ) from error

        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000.0
        LOGGER.info(
            "Ollama generation completed model=%s duration_ms=%.1f",
            self.config.model,
            elapsed_ms,
        )
        message = getattr(response, "message", None)
        content = getattr(message, "content", None)

        if content is None and isinstance(response, dict):
            raw_message = response.get("message", {})

            if isinstance(raw_message, dict):
                content = raw_message.get("content")

        if not isinstance(content, str) or not content.strip():
            raise OllamaResponseValidationError(
                "Ollama a retourné une réponse vide ou invalide."
            )

        return content.strip()


def parse_ollama_jsonl(
    chunks: Iterable[bytes | str],
) -> Iterator[dict[str, Any]]:
    """Decode fragmented UTF-8 JSONL events from Ollama."""

    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""

    for chunk in chunks:
        if not chunk:
            continue

        if isinstance(chunk, bytes):
            buffer += decoder.decode(chunk)
        else:
            buffer += chunk

        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            event = _decode_stream_line(line)

            if event is not None:
                yield event

    buffer += decoder.decode(b"", final=True)
    event = _decode_stream_line(buffer)

    if event is not None:
        yield event


def _decode_stream_line(line: str) -> dict[str, Any] | None:
    """Decode one non-empty Ollama JSONL line."""

    normalized = line.strip()

    if not normalized:
        return None

    try:
        event = json.loads(normalized)
    except json.JSONDecodeError as error:
        raise OllamaResponseValidationError(
            "Le flux Ollama contient un événement JSONL invalide.",
            raw_response=normalized,
        ) from error

    if not isinstance(event, dict):
        raise OllamaResponseValidationError(
            "Un événement du flux Ollama n'est pas un objet JSON.",
            raw_response=normalized,
        )

    return event
