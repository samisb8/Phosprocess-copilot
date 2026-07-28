"""Unit tests for blocking and streaming frozen-v3 RAG orchestration."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from phosprocess.llm.ollama_client import (
    OllamaConfig,
    OllamaResponseValidationError,
)
from phosprocess.observability.latency import RAGLatencyMetrics
from phosprocess.preprocessing.chunk_schemas import DocumentChunk
from phosprocess.rag.citations import INSUFFICIENT_CONTEXT_ANSWER
from phosprocess.rag.pipeline import (
    EXPECTED_SELECTED_VARIANT,
    ConversationRuntimeConfig,
    FrozenV3Config,
    GenerationRuntimeConfig,
    PhosProcessRAG,
    RAGConfigurationError,
    RAGResponseValidationError,
    RAGRetrievalError,
    RAGRuntimeConfig,
    load_frozen_v3_config,
)
from phosprocess.rag.prompts import SYSTEM_PROMPT
from phosprocess.rag.schemas import ChatMessage, GroundedAnswerPayload
from phosprocess.rag.source_policy import (
    AppliedSourcePolicy,
    SourcePolicyConfig,
)
from phosprocess.reranking.reranker import RerankedSearchResult
from phosprocess.retrieval.evidence_bundle import EvidenceBundle
from phosprocess.retrieval.hybrid import HybridSearchResult


class FakeRetriever:
    """Return controlled hybrid candidates and record the retrieval query."""

    def __init__(self, candidates: list[HybridSearchResult]) -> None:
        self.candidates = candidates
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def search(self, question: str, **kwargs: Any) -> SimpleNamespace:
        self.calls.append((question, kwargs))
        return SimpleNamespace(
            results=self.candidates,
            total_duration_ms=12.5,
        )


def test_applied_quality_policy_populates_latency_labels() -> None:
    retrieved = SimpleNamespace(source_policy=None)
    policy = AppliedSourcePolicy(
        route="equipment,heat_transfer",
        mode="auto",
        primary_source="03_fundamentals_heat_mass_transfer.pdf",
        preferred_sources=("03_fundamentals_heat_mass_transfer.pdf",),
        selected_scope=("03_fundamentals_heat_mass_transfer.pdf",),
        fallback_used=False,
        forced=False,
        attempt_count=1,
        sufficient_preferred_chunks=4,
    )
    metrics = RAGLatencyMetrics(question_id="unit")

    result = PhosProcessRAG._attach_source_policy(
        retrieved,  # type: ignore[arg-type]
        policy,
        metrics=metrics,
    )

    assert result is retrieved
    assert metrics.source_policy_route == "equipment,heat_transfer"
    assert metrics.source_policy_mode == "auto"
    assert metrics.source_policy_primary == "Incropera"
    assert metrics.source_policy_fallback_used is False


class FakeReranker:
    """Return a complete deterministic reranker ordering."""

    def __init__(self, results: list[RerankedSearchResult]) -> None:
        self.results = results
        self.calls: list[
            tuple[str, list[HybridSearchResult], int]
        ] = []

    def rerank(
        self,
        question: str,
        candidates: list[HybridSearchResult],
        *,
        top_k: int,
    ) -> SimpleNamespace:
        self.calls.append((question, candidates, top_k))
        return SimpleNamespace(
            results=self.results,
            reranking_duration_ms=7.5,
        )


class FakeLLM:
    """Simulate strict JSON and real-token interfaces without Ollama."""

    model_name = "qwen-mock"

    def __init__(
        self,
        *,
        json_outputs: list[str | Exception] | None = None,
        streams: list[list[str] | Exception] | None = None,
        stream_generated_tokens: list[int | None] | None = None,
    ) -> None:
        self.json_outputs = json_outputs or [
            json.dumps(
                {
                    "answer": (
                        "Conduite documentée [Source 1] et "
                        "[Source 5]."
                    )
                }
            )
        ]
        self.streams = streams or [
            ["Conduite ", "documentée [Source 2]."]
        ]
        self.stream_generated_tokens = stream_generated_tokens or []
        self.json_calls: list[dict[str, Any]] = []
        self.stream_calls: list[list[dict[str, str]]] = []

    def chat_json_with_raw(
        self,
        **kwargs: Any,
    ) -> tuple[GroundedAnswerPayload, str]:
        self.json_calls.append(kwargs)
        output = self.json_outputs.pop(0)

        if isinstance(output, Exception):
            raise output

        try:
            decoded = json.loads(output)
            payload = GroundedAnswerPayload.model_validate(decoded)
        except Exception as error:
            raise OllamaResponseValidationError(
                "JSON simulé invalide.",
                raw_response=output,
            ) from error

        return payload, output

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        self.stream_calls.append(messages)
        output = self.streams.pop(0)

        if isinstance(output, Exception):
            raise output

        telemetry = kwargs.get("telemetry")

        if telemetry is not None:
            telemetry.success = True

            if self.stream_generated_tokens:
                telemetry.generated_token_count = (
                    self.stream_generated_tokens.pop(0)
                )

        yield from output


def make_chunk(number: int) -> DocumentChunk:
    """Create one complete chunk with source metadata."""

    text = (
        f"Passage industriel complet numéro {number}. "
        + ("Détail opératoire documenté. " * 40)
    )
    return DocumentChunk(
        chunk_id=f"production_chunk_{number:02d}",
        document_id=f"document_{number % 3}",
        source_file=f"guide_procede_{number % 3}.pdf",
        chunk_index=number,
        heading_path=["Procédé humide", f"Section {number}"],
        source_pages=[number, number + 1],
        page_start=number,
        page_end=number + 1,
        content_types=["paragraph"],
        text=text,
        embedding_text=text,
        body_token_count=80,
        token_count=80,
        source_chunk_ids=[],
        postprocessing_actions=[],
    )


def make_candidates() -> list[HybridSearchResult]:
    """Make 20 candidates with candidate 20 as BM25 rank one."""

    candidates: list[HybridSearchResult] = []

    for number in range(1, 21):
        bm25_rank = 1 if number == 20 else number + 1
        candidates.append(
            HybridSearchResult(
                rank=number,
                rrf_score=1.0 / (60 + number),
                matched_retrievers=("dense", "bm25"),
                dense_rank=number,
                dense_score=1.0 / number,
                dense_rrf_contribution=0.0,
                bm25_rank=bm25_rank,
                bm25_score=1.0 / bm25_rank,
                bm25_rrf_contribution=0.0,
                chunk=make_chunk(number),
            )
        )

    return candidates


def make_reranked(
    candidates: list[HybridSearchResult],
) -> list[RerankedSearchResult]:
    """Keep the hybrid order as the reranker order."""

    return [
        RerankedSearchResult(
            rank=rank,
            reranker_score=1.0 - rank / 100.0,
            original_hybrid_rank=candidate.rank,
            original_rrf_score=candidate.rrf_score,
            matched_retrievers=candidate.matched_retrievers,
            dense_rank=candidate.dense_rank,
            dense_score=candidate.dense_score,
            bm25_rank=candidate.bm25_rank,
            bm25_score=candidate.bm25_score,
            chunk=candidate.chunk,
        )
        for rank, candidate in enumerate(candidates, start=1)
    ]


def make_frozen_config() -> FrozenV3Config:
    """Create exact frozen parameters without reading evaluation data."""

    placeholder = Path("unused.yaml")
    return FrozenV3Config(
        snapshot_directory=Path("unused"),
        snapshot_sha256="A" * 64,
        selected_variant=EXPECTED_SELECTED_VARIANT,
        candidate_k=20,
        dense_candidates=20,
        bm25_candidates=20,
        query_expansion=True,
        top_k=5,
        lexical_slots=1,
        reranker_leading_slots=4,
        lexical_source="bm25",
        duplicate_policy="skip",
        fallback="next_reranker_result",
        retrieval_config_path=placeholder,
        reranking_config_path=placeholder,
    )


def make_runtime_config() -> RAGRuntimeConfig:
    """Create small fake-driven runtime settings."""

    return RAGRuntimeConfig(
        ollama=OllamaConfig(model="qwen-mock"),
        maximum_question_characters=500,
        source_excerpt_characters=120,
        conversation=ConversationRuntimeConfig(
            recent_turns=2,
            summary_max_tokens=100,
            recent_history_max_tokens=200,
            total_history_max_tokens=300,
        ),
        generation=GenerationRuntimeConfig(
            max_context_tokens_per_source=80,
            max_total_document_context_tokens=400,
        ),
        source_policy=SourcePolicyConfig(
            enabled=False,
            default_priority=(
                "01_becker_phosphates_and_phosphoric_acid.pdf",
                "03_unido_phosphate_process_technologies.pdf",
                "02_jacobs_largest_phosphoric_acid_plant.pdf",
                "04_rapport_atelier_acide_phosphorique.pdf",
            ),
            domain_routes={
                "general": (
                    "01_becker_phosphates_and_phosphoric_acid.pdf",
                ),
                "jacobs": (
                    "02_jacobs_largest_phosphoric_acid_plant.pdf",
                    "01_becker_phosphates_and_phosphoric_acid.pdf",
                ),
                "ocp_atelier": (
                    "04_rapport_atelier_acide_phosphorique.pdf",
                    "01_becker_phosphates_and_phosphoric_acid.pdf",
                ),
            },
            minimum_preferred_chunks=2,
            allow_fallback=True,
        ),
    )


def make_service(
    *,
    candidates: list[HybridSearchResult] | None = None,
    llm: FakeLLM | None = None,
) -> tuple[PhosProcessRAG, FakeRetriever, FakeReranker, FakeLLM]:
    """Build a service with all expensive or external work simulated."""

    effective_candidates = (
        candidates
        if candidates is not None
        else make_candidates()
    )
    retriever = FakeRetriever(effective_candidates)
    reranker = FakeReranker(make_reranked(effective_candidates))
    effective_llm = llm or FakeLLM()
    service = PhosProcessRAG(
        frozen_config=make_frozen_config(),
        runtime_config=make_runtime_config(),
        retriever=retriever,
        reranker=reranker,
        llm=effective_llm,
        verify_snapshot=False,
    )
    return service, retriever, reranker, effective_llm


def test_answer_uses_exact_v3_and_returns_only_cited_sources() -> None:
    service, retriever, reranker, llm = make_service()

    response = service.answer(
        "  Comment   stabiliser la filtration industrielle ?  "
    )

    assert response.question == (
        "Comment stabiliser la filtration industrielle ?"
    )
    assert response.candidate_count == 20
    assert response.selected_count == 5
    assert response.cited_source_numbers == [1, 5]
    assert [source.source_number for source in response.sources] == [1, 5]
    assert [source.chunk_id for source in response.sources] == [
        "production_chunk_01",
        "production_chunk_20",
    ]
    assert response.sources[-1].selection_source == "bm25_safeguard"

    _, retrieval_kwargs = retriever.calls[0]
    assert retrieval_kwargs == {
        "top_k": 20,
        "dense_candidate_k": 20,
        "bm25_candidate_k": 20,
        "document_ids": None,
        "use_query_expansion": True,
    }
    assert reranker.calls[0][2] == 20
    assert llm.json_calls[0]["system_prompt"] == SYSTEM_PROMPT
    assert (
        llm.json_calls[0]["response_model"]
        is GroundedAnswerPayload
    )
    assert "Passage industriel complet numéro 20." in (
        llm.json_calls[0]["user_prompt"]
    )


@pytest.mark.parametrize("question", ["", "   ", "???", "x" * 501])
def test_invalid_questions_are_rejected_before_retrieval(
    question: str,
) -> None:
    service, retriever, _, _ = make_service()

    with pytest.raises((ValueError, TypeError)):
        service.answer(question)

    assert retriever.calls == []


def test_exactly_twenty_candidates_are_required() -> None:
    service, _, _, _ = make_service(
        candidates=make_candidates()[:19],
    )

    with pytest.raises(RAGRetrievalError, match="exactement 20"):
        service.answer("Question métier valide")


def test_blocking_repair_succeeds_once() -> None:
    llm = FakeLLM(
        json_outputs=[
            '{"answer": "Affirmation sans citation."}',
            '{"answer": "Affirmation [Source 3]."}',
        ]
    )
    service, _, _, _ = make_service(llm=llm)

    response = service.answer("Question métier valide")

    assert response.cited_source_numbers == [3]
    assert [source.source_number for source in response.sources] == [3]
    assert len(llm.json_calls) == 2


def test_blocking_repair_fails_after_second_invalid_output() -> None:
    llm = FakeLLM(
        json_outputs=[
            '{"answer": "Affirmation [Source 6]."}',
            '{"answer": "Toujours sans citation."}',
        ]
    )
    service, _, _, _ = make_service(llm=llm)

    with pytest.raises(
        RAGResponseValidationError,
        match="après une réparation",
    ):
        service.answer("Question métier valide")

    assert len(llm.json_calls) == 2


def test_invalid_json_is_repaired_once() -> None:
    llm = FakeLLM(
        json_outputs=[
            "pas du JSON",
            '{"answer": "Réponse corrigée [Source 1]."}',
        ]
    )
    service, _, _, _ = make_service(llm=llm)

    response = service.answer("Question métier valide")

    assert response.cited_source_numbers == [1]
    assert len(llm.json_calls) == 2


def test_controlled_insufficient_answer_has_no_source() -> None:
    llm = FakeLLM(
        json_outputs=[
            json.dumps(
                {"answer": INSUFFICIENT_CONTEXT_ANSWER},
                ensure_ascii=False,
            )
        ]
    )
    service, _, _, _ = make_service(llm=llm)

    response = service.answer("Question métier hors contexte")

    assert response.insufficient_context is True
    assert response.cited_source_numbers == []
    assert response.sources == []


def test_stream_emits_tokens_typed_events_and_final_cited_source() -> None:
    llm = FakeLLM(
        streams=[
            ["Réponse ", "progressive [Source 2]."],
        ]
    )
    service, _, _, _ = make_service(llm=llm)

    events = list(service.stream_answer("Question métier valide"))
    event_types = [event.event_type for event in events]

    assert event_types == [
        "retrieval_started",
        "retrieval_completed",
        "token",
        "token",
        "validation_started",
        "sources",
        "completed",
    ]
    retrieval_event = events[1]
    assert len(retrieval_event.sources) == 5
    assert len({item.chunk_id for item in retrieval_event.sources}) == 5
    completed = events[-1].response
    assert completed is not None
    assert completed.answer == "Réponse progressive [Source 2]."
    assert completed.cited_source_numbers == [2]
    assert [source.source_number for source in completed.sources] == [2]
    assert completed.timings.first_token_ms is not None


def test_stream_repair_succeeds_and_is_streamed() -> None:
    llm = FakeLLM(
        streams=[
            ["Réponse sans citation."],
            ["Réponse réparée ", "[Source 4]."],
        ]
    )
    service, _, _, _ = make_service(llm=llm)

    events = list(service.stream_answer("Question métier valide"))
    token_attempts = [
        event.metadata["attempt"]
        for event in events
        if event.event_type == "token"
    ]

    assert token_attempts == ["initial", "repair", "repair"]
    assert events[-1].event_type == "completed"
    assert events[-1].response is not None
    assert events[-1].response.cited_source_numbers == [4]
    assert len(llm.stream_calls) == 2


def test_stream_repair_failure_emits_error_without_completed() -> None:
    llm = FakeLLM(
        streams=[
            ["Réponse sans citation."],
            ["Réponse toujours invalide."],
        ]
    )
    service, _, _, _ = make_service(llm=llm)

    events = list(service.stream_answer("Question métier valide"))

    assert events[-1].event_type == "error"
    assert "après une réparation" in (events[-1].content or "")
    assert all(event.event_type != "completed" for event in events)


def test_truncated_stream_is_repaired_once_before_return() -> None:
    llm = FakeLLM(
        streams=[
            ["Réponse interrompue sans fin"],
            ["Réponse complète [Source 1]."],
        ],
        stream_generated_tokens=[300, 20],
    )
    service, _, _, _ = make_service(llm=llm)

    events = list(service.stream_answer("Question métier valide"))
    completed = events[-1].response

    assert completed is not None
    assert completed.answer == "Réponse complète [Source 1]."
    assert completed.latency["repair_attempted"] is True
    assert completed.latency["ollama_call_count"] == 2


def test_follow_up_uses_bounded_autonomous_retrieval_query() -> None:
    service, retriever, _, llm = make_service()
    history = [
        ChatMessage(
            role="user",
            content="Quel est le rôle de la recirculation Jacobs ?",
        ),
        ChatMessage(
            role="assistant",
            content="Elle stabilise le milieu [Source 1].",
        ),
    ]

    list(
        service.stream_answer(
            "Et pourquoi est-ce important ?",
            history,
        )
    )

    retrieval_query = retriever.calls[0][0]
    assert "recirculation Jacobs" in retrieval_query
    assert "Question :" in retrieval_query
    sent_messages = llm.stream_calls[0]
    assert len(sent_messages) <= 4
    assert all(
        "[Source 1]" not in message["content"]
        for message in sent_messages[1:-1]
    )


def test_long_lived_pipeline_reuses_retrieval_models_and_llm() -> None:
    llm = FakeLLM(
        streams=[
            ["Première réponse [Source 1]."],
            ["Seconde réponse [Source 2]."],
        ]
    )
    service, _, _, _ = make_service(llm=llm)
    before = service.lifecycle_debug()

    first = list(service.stream_answer("Première question métier"))
    second = list(service.stream_answer("Deuxième question métier"))
    after = service.lifecycle_debug()

    assert first[-1].event_type == "completed"
    assert second[-1].event_type == "completed"
    assert before["ids"] == after["ids"]
    assert after["counts"] == {
        "pipeline": 1,
        "retriever": 1,
        "embedding_model": 1,
        "bm25_index": 1,
        "reranker": 1,
        "ollama_client": 1,
    }


def test_warmup_call_is_not_counted_in_first_real_turn() -> None:
    llm = FakeLLM(
        streams=[
            ["OK"],
            ["Réponse principale [Source 1]."],
        ]
    )
    service, _, _, _ = make_service(llm=llm)

    warmup = service.warmup()
    repeated_warmup = service.warmup()
    events = list(service.stream_answer("Question métier réelle"))
    completed = events[-1].response

    assert warmup.ollama_call_count == 1
    assert repeated_warmup is warmup
    assert len(llm.stream_calls) == 2
    assert completed is not None
    assert completed.latency["ollama_call_count"] == 1
    assert completed.latency["ollama_calls"][0]["call_type"] == (
        "generation_main"
    )


def test_turn_metrics_preserve_five_unique_sources_and_prompt_budget() -> None:
    service, _, _, _ = make_service()

    events = list(service.stream_answer("Question sur la recirculation"))
    retrieval = next(
        event
        for event in events
        if event.event_type == "retrieval_completed"
    )
    completed = events[-1].response

    assert len(retrieval.sources) == 5
    assert len({source.chunk_id for source in retrieval.sources}) == 5
    assert completed is not None
    assert completed.selected_count == 5
    assert completed.latency["document_context_token_count"] <= 400
    assert completed.latency["ollama_call_count"] == 1
    assert completed.latency["total_ms"] >= 0


def test_inference_has_no_benchmark_or_gold_inputs() -> None:
    signature = inspect.signature(PhosProcessRAG.answer)
    source = inspect.getsource(PhosProcessRAG.answer).casefold()

    assert set(signature.parameters) == {
        "self",
        "question",
        "source_mode",
        "language_mode",
    }

    for forbidden_name in (
        "gold",
        "reference_answer",
        "expected_answer",
        "query_id",
    ):
        assert forbidden_name not in source


def test_frozen_snapshot_integrity_and_exact_parameters() -> None:
    frozen = load_frozen_v3_config(
        verify_integrity=True,
        verify_runtime_sources=False,
    )

    assert frozen.selected_variant == "lexical_safeguard_001"
    assert (
        frozen.candidate_k,
        frozen.dense_candidates,
        frozen.bm25_candidates,
        frozen.top_k,
    ) == (20, 20, 20, 5)
    assert frozen.query_expansion is True
    assert frozen.reranker_leading_slots == 4
    assert frozen.lexical_slots == 1
    assert frozen.lexical_source == "bm25"
    assert frozen.duplicate_policy == "skip"
    assert frozen.fallback == "next_reranker_result"


def test_frozen_v3_runtime_hash_check_is_explicit_legacy_mode() -> None:
    with pytest.raises(
        RAGConfigurationError,
        match="Le code runtime diffère de dev_best_v3",
    ):
        load_frozen_v3_config(
            verify_integrity=True,
            verify_runtime_sources=True,
        )


def test_blocking_translation_bypasses_retrieval_and_sources() -> None:
    llm = FakeLLM(json_outputs=['{"answer": "Hello"}'])
    service, retriever, reranker, _ = make_service(llm=llm)

    response = service.answer('Traduis "Bonjour" en anglais.')

    assert response.answer == "Hello"
    assert response.candidate_count == 0
    assert response.selected_count == 0
    assert response.sources == []
    assert response.cited_source_numbers == []
    assert response.source_policy_route == "direct_no_retrieval"
    assert response.question_type == "translation"
    assert response.response_language == "en"
    assert retriever.calls == []
    assert reranker.calls == []
    assert "phosphoric" not in llm.json_calls[0]["user_prompt"].casefold()


def test_streaming_translation_bypasses_followup_and_retrieval() -> None:
    llm = FakeLLM(streams=[["Hello"]])
    service, retriever, reranker, _ = make_service(llm=llm)

    events = list(
        service.stream_answer(
            'Traduis "Bonjour" en anglais.',
            [
                ChatMessage(
                    role="user",
                    content="Parle-moi de l’évaporateur phosphorique.",
                ),
                ChatMessage(
                    role="assistant",
                    content="Le procédé est documenté [Source 1].",
                ),
            ],
        )
    )

    assert [event.event_type for event in events] == [
        "retrieval_started",
        "retrieval_completed",
        "token",
        "validation_started",
        "sources",
        "completed",
    ]
    assert events[0].metadata["retrieval_skipped"] is True
    assert events[1].metadata["candidate_count"] == 0
    assert events[1].sources == []
    completed = events[-1].response
    assert completed is not None
    assert completed.answer == "Hello"
    assert completed.source_policy_route == "direct_no_retrieval"
    assert completed.latency["dense_search_ms"] == 0.0
    assert completed.latency["bm25_search_ms"] == 0.0
    assert completed.latency["reranking_ms"] == 0.0
    assert completed.latency["ollama_call_count"] == 1
    assert retriever.calls == []
    assert reranker.calls == []


def test_domain_question_still_uses_retrieval_after_adaptive_router() -> None:
    service, retriever, reranker, _ = make_service()

    response = service.answer(
        "Décris la circulation dans un évaporateur phosphorique."
    )

    assert response.candidate_count == 20
    assert response.selected_count == 5
    assert len(retriever.calls) == 1
    assert len(reranker.calls) == 1


def test_json_generation_normalizes_supported_duplicate_process_flow() -> None:
    duplicate_answer = "\n".join(
        [
            (
                "1. The liquid phase is fed by the inlet acid pipe coming "
                "from the heat exchanger [Source 1]."
            ),
            (
                "2. The pump withdraws liquor from the flash chamber and "
                "forces it through the heating element back to the flash "
                "chamber [Source 2]."
            ),
            "3. The liquor returns to the flash chamber [Source 2].",
            (
                "4. The concentrated finished product acid is withdrawn from "
                "the vapor body at the product outlet [Source 1]."
            ),
            (
                "5. The product outlet withdraws the concentrated finished "
                "product acid from the vapor body [Source 1]."
            ),
        ]
    )
    llm = FakeLLM(
        json_outputs=[json.dumps({"answer": duplicate_answer})]
    )
    service, _retriever, _reranker, _llm = make_service(llm=llm)
    evidence = [
        EvidenceBundle(
            source_number=1,
            document_id="becker",
            document_title="Becker",
            filename="becker.pdf",
            chapter="Acid Concentration Systems",
            section="Vapor Body",
            page_start=219,
            page_end=220,
            anchor_chunk_id="becker_vapor_body",
            expanded_chunk_ids=("becker_vapor_body",),
            display_text=(
                "The vapor body achieves vapor/liquid separation. The liquid "
                "phase is fed by the inlet acid pipe coming from the heat "
                "exchanger. The cycling acid leaves the vapor body through a "
                "conical bottom. The concentrated finished product acid is "
                "withdrawn from the vapor body at an outlet below the feed "
                "level."
            ),
            token_count=120,
            anchor_score=0.9,
            selection_provenance="reranker",
        ),
        EvidenceBundle(
            source_number=2,
            document_id="perry",
            document_title="Perry",
            filename="perry.pdf",
            chapter="Evaporators",
            section="Forced Circulation",
            page_start=1034,
            page_end=1035,
            anchor_chunk_id="perry_fc",
            expanded_chunk_ids=("perry_fc",),
            display_text=(
                "A pump ensures circulation past the heating surface. The "
                "pump withdraws liquor from the flash chamber and forces it "
                "through the heating element back to the flash chamber."
            ),
            token_count=100,
            anchor_score=0.8,
            selection_provenance="reranker",
        ),
    ]

    payload, citations, insufficient = service._generate_json_answer(
        user_prompt="Describe the path.",
        available_source_count=2,
        evidence_bundles=evidence,
        question_type="process_flow",
    )

    lines = payload.answer.splitlines()
    assert insufficient is False
    assert citations == [1, 2]
    assert len(lines) == 5
    assert [line[:2] for line in lines] == ["1.", "2.", "3.", "4.", "5."]
    assert "conical bottom" in lines[1]
    assert "; vapor-liquid separation" in lines[3]
    assert payload.answer.count("product outlet") == 1
    assert "heat is applied to evaporate water" not in payload.answer
