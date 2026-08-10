"""Fine-grained, content-safe latency metrics for production RAG."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def estimate_tokens(text: str) -> int:
    """Estimate tokens without loading another tokenizer into memory."""

    if not text:
        return 0

    return max(1, math.ceil(len(text) / 4))


@dataclass(slots=True)
class OllamaCallMetrics:
    """Telemetry for one Ollama request, excluding prompt contents."""

    call_type: str
    model: str
    streaming: bool
    started_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    prompt_character_count: int = 0
    estimated_prompt_tokens: int = 0
    connection_ms: float = 0.0
    time_to_first_event_ms: float = 0.0
    time_to_first_token_ms: float = 0.0
    generation_ms: float = 0.0
    duration_ms: float = 0.0
    generated_character_count: int = 0
    generated_chunk_count: int = 0
    generated_token_count: int | None = None
    prompt_token_count: int | None = None
    model_load_ms: float = 0.0
    prompt_evaluation_ms: float = 0.0
    model_generation_ms: float = 0.0
    generation_tokens_per_second: float | None = None
    success: bool = False
    error_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record."""

        return asdict(self)


@dataclass(slots=True)
class WarmupMetrics:
    """One-time model warm-up durations."""

    enabled: bool = True
    embedding_ms: float = 0.0
    reranker_ms: float = 0.0
    ollama_ms: float = 0.0
    total_ms: float = 0.0
    ollama_call_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record."""

        return asdict(self)


@dataclass(slots=True)
class RAGLatencyMetrics:
    """All measured phases for one session-local RAG turn."""

    question_id: str
    question_validation_ms: float = 0.0
    followup_detection_ms: float = 0.0
    reformulation_ms: float = 0.0
    resolver_llm_ms: float = 0.0
    embedding_ms: float = 0.0
    dense_search_ms: float = 0.0
    bm25_search_ms: float = 0.0
    query_expansion_ms: float = 0.0
    hybrid_fusion_ms: float = 0.0
    candidate_preparation_ms: float = 0.0
    reranker_tokenization_ms: float = 0.0
    reranker_scoring_ms: float = 0.0
    reranking_ms: float = 0.0
    lexical_selection_ms: float = 0.0
    source_loading_ms: float = 0.0
    excerpt_preparation_ms: float = 0.0
    memory_build_ms: float = 0.0
    prompt_build_ms: float = 0.0
    json_validation_ms: float = 0.0
    citation_extraction_ms: float = 0.0
    citation_validation_ms: float = 0.0
    repair_ms: float = 0.0
    ollama_connection_ms: float = 0.0
    ollama_time_to_first_event_ms: float = 0.0
    ollama_time_to_first_token_ms: float = 0.0
    ollama_generation_ms: float = 0.0
    turn_time_to_first_token_ms: float = 0.0
    total_ms: float = 0.0

    ollama_call_count: int = 0
    resolver_llm_call_count: int = 0
    prompt_character_count: int = 0
    estimated_prompt_tokens: int = 0
    generated_character_count: int = 0
    generated_token_count: int | None = None
    generated_chunk_count: int = 0
    history_turn_count: int = 0
    summary_token_count: int = 0
    recent_history_token_count: int = 0
    document_context_token_count: int = 0
    system_prompt_token_count: int = 0
    question_token_count: int = 0
    baseline_equivalent_prompt_characters: int = 0
    baseline_equivalent_prompt_tokens: int = 0
    repair_attempted: bool = False
    repair_reason: str | None = None
    truncation_salvaged: bool = False
    reformulation_attempted: bool = False
    reformulation_method: str = "none"
    retrieval_query: str = ""
    citations: list[int] = field(default_factory=list)
    displayed_source_count: int = 0
    source_policy_route: str = "disabled"
    source_policy_mode: str = "automatic"
    source_policy_primary: str = "Aucune"
    source_policy_fallback_used: bool = False
    source_policy_forced: bool = False
    source_policy_attempt_count: int = 0
    source_policy_sufficient_preferred_chunks: int = 0
    ollama_calls: list[dict[str, Any]] = field(default_factory=list)

    def absorb_ollama_call(self, call: OllamaCallMetrics) -> None:
        """Aggregate one model call into the current turn."""

        self.ollama_call_count += 1
        self.ollama_calls.append(call.to_dict())
        self.ollama_connection_ms += call.connection_ms
        self.ollama_time_to_first_event_ms = call.time_to_first_event_ms
        self.ollama_time_to_first_token_ms = call.time_to_first_token_ms
        self.ollama_generation_ms += call.generation_ms
        self.generated_character_count = call.generated_character_count
        self.generated_chunk_count = call.generated_chunk_count
        self.generated_token_count = call.generated_token_count

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record."""

        return asdict(self)

    def concise_log_fields(self) -> str:
        """Build a one-line diagnostic without prompts or document text."""

        retrieval_ms = self.dense_search_ms + self.bm25_search_ms + self.hybrid_fusion_ms
        return (
            f"question_id={self.question_id} "
            f"retrieval_ms={retrieval_ms:.1f} "
            f"reranking_ms={self.reranking_ms:.1f} "
            f"prompt_tokens={self.estimated_prompt_tokens} "
            f"ollama_calls={self.ollama_call_count} "
            f"source_policy={self.source_policy_route} "
            f"source_fallback={self.source_policy_fallback_used} "
            f"ttft_ms={self.ollama_time_to_first_token_ms:.1f} "
            f"generation_ms={self.ollama_generation_ms:.1f} "
            f"total_ms={self.total_ms:.1f}"
        )
