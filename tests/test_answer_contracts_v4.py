"""Public response, prompt and source-scope invariants."""

from __future__ import annotations

import inspect

from phosprocess.rag.conversation_state import ConversationState
from phosprocess.rag.pipeline import PhosProcessRAG
from phosprocess.rag.schemas import RAGResponse, RAGTimings


def response_with_candidate_count(candidate_count: int) -> RAGResponse:
    return RAGResponse(
        question="Question",
        answer="Answer",
        sources=[],
        cited_source_numbers=[],
        insufficient_context=True,
        model_name="qwen-test",
        selected_variant="retriever_v4",
        snapshot_sha256="A" * 64,
        candidate_count=candidate_count,
        selected_count=5,
        timings=RAGTimings(
            hybrid_ms=1.0,
            reranking_ms=1.0,
            generation_ms=1.0,
            total_ms=3.0,
        ),
    )


def test_rag_response_records_runtime_candidate_counts_without_semantic_cap() -> None:
    assert response_with_candidate_count(30).candidate_count == 30
    assert response_with_candidate_count(31).candidate_count == 31


def test_quality_stream_validates_before_emitting_buffered_answer() -> None:
    source = inspect.getsource(PhosProcessRAG.stream_answer)
    validation_position = source.index('event_type="validation_started"')
    response_position = source.index("response = self._build_response(")
    emission_position = source.index('event_type="token"', response_position)
    assert validation_position < response_position < emission_position


def test_explicit_source_scope_persists_only_across_followups() -> None:
    rag = object.__new__(PhosProcessRAG)
    rag.quality_engine = object()
    state = ConversationState()

    first = rag._resolve_turn_source_mode(
        "auto",
        question="Selon Becker, définis cet équipement.",
        follow_up=False,
        state=state,
    )
    inherited = rag._resolve_turn_source_mode(
        "auto",
        question="Et son fonctionnement ?",
        follow_up=True,
        state=state,
    )
    reset = rag._resolve_turn_source_mode(
        "auto",
        question="Explique un autre sujet.",
        follow_up=False,
        state=state,
    )

    assert first == "becker"
    assert inherited == "becker"
    assert reset == "auto"


def test_explicit_all_sources_request_releases_source_lock() -> None:
    rag = object.__new__(PhosProcessRAG)
    rag.quality_engine = object()
    state = ConversationState()
    state.record_source_scope("becker", explicit=True, origin="user_question")

    mode = rag._resolve_turn_source_mode(
        "auto",
        question="Cherche maintenant dans toutes les sources.",
        follow_up=True,
        state=state,
    )
    assert mode == "auto"
    assert state.source_scope_explicit is False
