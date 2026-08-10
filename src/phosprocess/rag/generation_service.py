"""Prompt execution and local Qwen/Ollama generation service."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator

from phosprocess.llm.ollama_client import (
    OllamaError,
    OllamaResponseValidationError,
)
from phosprocess.observability.latency import OllamaCallMetrics, RAGLatencyMetrics
from phosprocess.rag.adaptive_router import AdaptiveRouteDecision, RequestPath
from phosprocess.rag.citations import CitationValidationError
from phosprocess.rag.language import detect_response_language
from phosprocess.rag.prompts import (
    SYSTEM_PROMPT,
    build_direct_prompt_package,
)
from phosprocess.rag.retrieval_service import (
    RAGGenerationError,
    RAGResponseValidationError,
)
from phosprocess.rag.schemas import (
    GroundedAnswerPayload,
    RAGResponse,
    RAGStreamEvent,
    RAGTimings,
)

LOGGER = logging.getLogger("phosprocess.rag.pipeline")


def _controlled_fallback_for_language(language: str) -> str:
    normalized = language.strip().lower()
    if normalized.startswith("en"):
        return (
            "The retrieved passages do not provide enough information to "
            "answer this question precisely."
        )
    if normalized.startswith("ar"):
        return "لا توفر المقاطع المسترجعة معلومات كافية للإجابة عن هذا السؤال بدقة."
    return "Les passages retrouvés ne permettent pas de répondre précisément à cette question."


class GenerationService:
    @staticmethod
    def _direct_language_code(
        decision: AdaptiveRouteDecision,
        question: str,
        *,
        language_mode: str,
        last_explicit_language: str | None = None,
    ) -> str:
        """Resolve direct-output language without document-language leakage."""

        if decision.requested_output_language is not None:
            return decision.requested_output_language

        detected = detect_response_language(
            question,
            last_explicit_language=last_explicit_language,
            mode=language_mode,
        )
        return detected.language.value

    def _build_direct_response(
        self,
        *,
        question: str,
        answer: str,
        decision: AdaptiveRouteDecision,
        response_language: str,
        generation_ms: float,
        total_ms: float,
        first_token_ms: float | None = None,
        latency: RAGLatencyMetrics | None = None,
    ) -> RAGResponse:
        """Build a source-free response for an intentional retrieval bypass."""

        model_name = getattr(
            self.llm,
            "model_name",
            self.runtime_config.ollama.model,
        )
        return RAGResponse(
            question=question,
            answer=answer,
            sources=[],
            cited_source_numbers=[],
            insufficient_context=False,
            model_name=str(model_name),
            selected_variant=self.frozen_config.selected_variant,
            snapshot_sha256=self.frozen_config.snapshot_sha256,
            candidate_count=0,
            selected_count=0,
            source_policy_route=RequestPath.DIRECT_LLM.value,
            source_policy_mode="bypass",
            source_policy_primary=None,
            source_policy_fallback_used=False,
            source_policy_forced=False,
            response_language=response_language,
            standalone_query=question,
            question_type=(
                decision.direct_intent.value if decision.direct_intent is not None else "direct"
            ),
            detected_domains=[],
            timings=RAGTimings(
                hybrid_ms=0.0,
                reranking_ms=0.0,
                generation_ms=generation_ms,
                total_ms=total_ms,
                first_token_ms=first_token_ms,
            ),
            latency=latency.to_dict() if latency is not None else {},
        )

    def _answer_direct(
        self,
        question: str,
        decision: AdaptiveRouteDecision,
        *,
        language_mode: str,
        started: float,
    ) -> RAGResponse:
        """Execute one self-contained request without touching retrieval."""

        response_language = self._direct_language_code(
            decision,
            question,
            language_mode=language_mode,
        )
        system_prompt, package = build_direct_prompt_package(
            question,
            decision,
            json_output=True,
        )
        generation_started = time.perf_counter()

        try:
            payload, _raw = self.llm.chat_json_with_raw(
                user_prompt=package.user_prompt,
                system_prompt=system_prompt,
                response_model=GroundedAnswerPayload,
            )
        except OllamaError as error:
            raise RAGGenerationError(str(error)) from error
        except Exception as error:
            raise RAGGenerationError("La génération directe locale Qwen a échoué.") from error

        generation_ms = (time.perf_counter() - generation_started) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0
        return self._build_direct_response(
            question=question,
            answer=payload.answer,
            decision=decision,
            response_language=response_language,
            generation_ms=generation_ms,
            total_ms=total_ms,
        )

    def _stream_direct_request(
        self,
        question: str,
        decision: AdaptiveRouteDecision,
        *,
        language_mode: str,
        last_explicit_language: str | None,
        metrics: RAGLatencyMetrics,
        turn_started: float,
    ) -> Iterator[RAGStreamEvent]:
        """Stream a direct answer while explicitly reporting retrieval bypass."""

        response_language = self._direct_language_code(
            decision,
            question,
            language_mode=language_mode,
            last_explicit_language=last_explicit_language,
        )
        metrics.retrieval_query = question
        metrics.source_policy_route = RequestPath.DIRECT_LLM.value
        metrics.source_policy_mode = "bypass"
        metrics.source_policy_primary = "Aucune"
        metrics.source_policy_attempt_count = 0
        direct_type = (
            decision.direct_intent.value if decision.direct_intent is not None else "direct"
        )

        yield RAGStreamEvent(
            event_type="retrieval_started",
            metadata={
                "question_id": metrics.question_id,
                "follow_up": False,
                "reformulated": False,
                "reformulation_method": "adaptive_router",
                "standalone_query": question,
                "language": response_language,
                "question_type": direct_type,
                "source_policy_route": RequestPath.DIRECT_LLM.value,
                "source_policy_primary": "Aucune",
                "retrieval_skipped": True,
                "routing_reason": decision.reason,
            },
        )
        yield RAGStreamEvent(
            event_type="retrieval_completed",
            sources=[],
            metadata={
                "candidate_count": 0,
                "selected_count": 0,
                "hybrid_ms": 0.0,
                "reranking_ms": 0.0,
                "source_policy_route": RequestPath.DIRECT_LLM.value,
                "source_policy_primary": "Aucune",
                "source_policy_fallback_used": False,
                "query_expansion": [],
                "hierarchical_sections": [],
                "section_retrieval_ms": 0.0,
                "retrieval_skipped": True,
                "routing_reason": decision.reason,
            },
        )

        prompt_started = time.perf_counter()
        system_prompt, package = build_direct_prompt_package(
            question,
            decision,
            json_output=False,
        )
        metrics.prompt_build_ms = (time.perf_counter() - prompt_started) * 1000.0
        metrics.prompt_character_count = package.size.total_characters
        metrics.estimated_prompt_tokens = package.size.total_tokens
        metrics.system_prompt_token_count = package.size.system_tokens
        metrics.question_token_count = package.size.question_tokens
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": package.user_prompt},
        ]
        call = OllamaCallMetrics(
            call_type="generation_direct",
            model=self.runtime_config.ollama.model,
            streaming=True,
        )
        fragments: list[str] = []
        generation_started = time.perf_counter()
        first_turn_token_ms: float | None = None

        try:
            for fragment in self.llm.stream_chat(
                messages,
                call_type=call.call_type,
                telemetry=call,
            ):
                if first_turn_token_ms is None and fragment.strip():
                    first_turn_token_ms = (time.perf_counter() - turn_started) * 1000.0
                    metrics.turn_time_to_first_token_ms = first_turn_token_ms
                fragments.append(fragment)
                yield RAGStreamEvent(
                    event_type="token",
                    content=fragment,
                    metadata={
                        "attempt": "direct",
                        "retrieval_skipped": True,
                    },
                )
        finally:
            metrics.absorb_ollama_call(call)

        answer = "".join(fragments).strip()
        if not answer:
            raise RAGGenerationError("Qwen a retourné une réponse directe vide.")

        yield RAGStreamEvent(
            event_type="validation_started",
            metadata={
                "attempt": "direct",
                "citation_validation_skipped": True,
            },
        )
        generation_ms = (time.perf_counter() - generation_started) * 1000.0
        metrics.total_ms = (time.perf_counter() - turn_started) * 1000.0
        response = self._build_direct_response(
            question=question,
            answer=answer,
            decision=decision,
            response_language=response_language,
            generation_ms=generation_ms,
            total_ms=metrics.total_ms,
            first_token_ms=first_turn_token_ms,
            latency=metrics,
        )
        LOGGER.info(
            "ADAPTIVE_RAG bypass intent=%s reason=%s %s",
            direct_type,
            decision.reason,
            metrics.concise_log_fields(),
        )
        yield RAGStreamEvent(
            event_type="sources",
            sources=[],
            metadata={"citations": [], "retrieval_skipped": True},
        )
        yield RAGStreamEvent(
            event_type="completed",
            response=response,
            metadata={"latency": metrics.to_dict()},
        )

    def _reject_likely_truncation(
        self,
        answer: str,
        *,
        generated_token_count: int | None,
    ) -> None:
        """Reject an output that exhausted its budget mid-sentence."""

        if (
            generated_token_count is None
            or generated_token_count < self.runtime_config.ollama.max_output_tokens
        ):
            return

        if re.search(
            r"(?:[.!?…]|(?:\[Source [1-9]\d*\]))\s*$",
            answer,
        ):
            return

        raise CitationValidationError(
            "La sortie a atteint la limite de tokens au milieu d'une phrase."
        )

    def _finalize_likely_truncation(
        self,
        answer: str,
        *,
        generated_token_count: int | None,
        response_language: str,
    ) -> tuple[str, bool]:
        """Keep a completed cited prefix when generation stops mid-sentence.

        This is an objective, single-call fallback: it never invents text, asks the
        model again, or treats conversation history as evidence. If no completed
        citation boundary exists, the original fail-closed behavior is preserved.
        """

        try:
            self._reject_likely_truncation(
                answer,
                generated_token_count=generated_token_count,
            )
        except CitationValidationError:
            citation_matches = list(
                re.finditer(r"\[Source [1-9]\d*\]", answer)
            )
            if not citation_matches:
                raise
            completed_prefix = answer[: citation_matches[-1].end()].rstrip()
            fallback = _controlled_fallback_for_language(response_language)
            LOGGER.warning(
                "Truncated generation reduced to last cited boundary; no retry performed"
            )
            return f"{completed_prefix}\n\n{fallback}", True
        return answer, False

    def _generate_json_answer(
        self,
        *,
        user_prompt: str,
        available_source_count: int,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> tuple[GroundedAnswerPayload, list[int], bool]:
        """Generate once, then enforce only JSON and citation invariants."""

        try:
            payload, _raw_output = self.llm.chat_json_with_raw(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                response_model=GroundedAnswerPayload,
            )
        except OllamaResponseValidationError as error:
            raise RAGResponseValidationError(
                f"Réponse Qwen JSON invalide : {error}"
            ) from error
        except OllamaError as error:
            raise RAGGenerationError(str(error)) from error
        except Exception as error:
            raise RAGGenerationError("La génération locale Qwen a échoué.") from error

        try:
            citations, insufficient = self._validate_answer(
                answer=payload.answer,
                available_source_count=available_source_count,
                attempt="initial",
            )
        except CitationValidationError as error:
            raise RAGResponseValidationError(f"Réponse Qwen invalide : {error}") from error
        return payload, citations, insufficient
