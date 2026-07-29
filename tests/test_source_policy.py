"""Production-only document routing and scoped retrieval tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from phosprocess.llm.ollama_client import OllamaConfig
from phosprocess.preprocessing.chunk_schemas import DocumentChunk
from phosprocess.rag.pipeline import (
    EXPECTED_SELECTED_VARIANT,
    ConversationRuntimeConfig,
    FrozenV3Config,
    GenerationRuntimeConfig,
    PhosProcessRAG,
    RAGRuntimeConfig,
    load_runtime_config,
)
from phosprocess.rag.schemas import GroundedAnswerPayload
from phosprocess.rag.source_policy import (
    ATELIER_SOURCE,
    BECKER_SOURCE,
    CONTROL_SOURCE,
    CRYSTALLIZATION_SOURCE,
    HEAT_TRANSFER_SOURCE,
    PERRY_SOURCE,
    THERMODYNAMICS_SOURCE,
    TRANSPORT_SOURCE,
    SourcePolicyConfig,
    decide_source_policy,
    detect_explicit_active_source,
    document_id_from_source,
)
from phosprocess.reranking.reranker import RerankedSearchResult
from phosprocess.retrieval.hybrid import HybridSearchResult

ALL_SOURCES = (
    BECKER_SOURCE,
    THERMODYNAMICS_SOURCE,
    HEAT_TRANSFER_SOURCE,
    ATELIER_SOURCE,
    PERRY_SOURCE,
    CRYSTALLIZATION_SOURCE,
    CONTROL_SOURCE,
    TRANSPORT_SOURCE,
)


def make_policy_config() -> SourcePolicyConfig:
    """Return the exact eight-document rollback policy from YAML."""

    return SourcePolicyConfig(
        enabled=True,
        default_priority=ALL_SOURCES,
        domain_routes={
            "general": (PERRY_SOURCE, BECKER_SOURCE),
            "phosphoric_acid": (
                BECKER_SOURCE,
                ATELIER_SOURCE,
                PERRY_SOURCE,
            ),
            "plant_specific": (
                ATELIER_SOURCE,
                BECKER_SOURCE,
                PERRY_SOURCE,
            ),
            "thermodynamics": (
                THERMODYNAMICS_SOURCE,
                PERRY_SOURCE,
                ATELIER_SOURCE,
            ),
            "heat_transfer": (
                HEAT_TRANSFER_SOURCE,
                PERRY_SOURCE,
                ATELIER_SOURCE,
            ),
            "equipment": (
                PERRY_SOURCE,
                BECKER_SOURCE,
                ATELIER_SOURCE,
            ),
            "crystallization": (
                CRYSTALLIZATION_SOURCE,
                BECKER_SOURCE,
            ),
            "control": (CONTROL_SOURCE, ATELIER_SOURCE),
            "transport": (TRANSPORT_SOURCE, PERRY_SOURCE),
        },
        minimum_preferred_chunks=2,
        allow_fallback=True,
    )


@pytest.mark.parametrize(
    ("question", "route", "primary_source"),
    [
        (
            "Comment améliorer la filtration de l'acide phosphorique ?",
            "phosphoric_acid",
            BECKER_SOURCE,
        ),
        (
            "Quelle relation relie l'enthalpie et la pression de vapeur ?",
            "thermodynamics",
            THERMODYNAMICS_SOURCE,
        ),
        (
            "Comment fonctionne l'atelier OCP JFC4 ?",
            "plant_specific",
            ATELIER_SOURCE,
        ),
    ],
)
def test_deterministic_route_prioritizes_expected_document(
    question: str,
    route: str,
    primary_source: str,
) -> None:
    decision = decide_source_policy(
        question,
        config=make_policy_config(),
    )

    assert decision.route == route
    assert decision.primary_source == primary_source


def test_yaml_contains_exact_production_source_policy() -> None:
    runtime = load_runtime_config()

    assert runtime.source_policy == replace(
        make_policy_config(),
        enabled=False,
    )


def test_explicit_active_document_match_uses_distinctive_name_only() -> None:
    sources = (
        "07_process_dynamics_control_seborg_4e.pdf",
        "08_transport_phenomena_bird_2e.pdf",
    )

    assert (
        detect_explicit_active_source(
            "Selon Seborg, pourquoi utilise-t-on le feedback ?",
            sources,
        )
        == sources[0]
    )
    assert (
        detect_explicit_active_source(
            "Explique un procédé d'ingénierie chimique.",
            sources,
        )
        is None
    )


def make_chunk(
    number: int,
    source_file: str,
) -> DocumentChunk:
    """Create one indexed-looking production chunk."""

    text = (
        f"Passage industriel {number}. "
        + ("Données opératoires documentées. " * 25)
    )
    return DocumentChunk(
        chunk_id=(
            f"{document_id_from_source(source_file)}_"
            f"{number:06d}_policy"
        ),
        document_id=document_id_from_source(source_file),
        source_file=source_file,
        chunk_index=number,
        heading_path=["Procédé phosphorique", f"Section {number}"],
        source_pages=[number],
        page_start=number,
        page_end=number,
        content_types=["paragraph"],
        text=text,
        embedding_text=text,
        body_token_count=60,
        token_count=60,
        source_chunk_ids=[],
        postprocessing_actions=[],
    )


def make_candidates(
    source_file: str,
    *,
    strong_count: int = 20,
) -> list[HybridSearchResult]:
    """Create 20 candidates with controlled cross-retriever support."""

    candidates: list[HybridSearchResult] = []

    for number in range(1, 21):
        strong = number <= strong_count
        candidates.append(
            HybridSearchResult(
                rank=number,
                rrf_score=1.0 / (60 + number),
                matched_retrievers=(
                    ("dense", "bm25")
                    if strong
                    else ("dense",)
                ),
                dense_rank=number,
                dense_score=1.0 / number,
                dense_rrf_contribution=0.0,
                bm25_rank=number if strong else None,
                bm25_score=(1.0 / number if strong else None),
                bm25_rrf_contribution=0.0,
                chunk=make_chunk(number, source_file),
            )
        )

    return candidates


class ScopedFakeRetriever:
    """Return candidates for the exact application-provided document scope."""

    def __init__(
        self,
        datasets: dict[frozenset[str], list[HybridSearchResult]],
    ) -> None:
        self.datasets = datasets
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def search(self, question: str, **kwargs: Any) -> SimpleNamespace:
        self.calls.append((question, kwargs))
        scope = frozenset(kwargs.get("document_ids") or ())
        return SimpleNamespace(
            results=self.datasets.get(scope, []),
            dense_duration_ms=1.0,
            bm25_duration_ms=1.0,
            total_duration_ms=3.0,
        )


class DynamicFakeReranker:
    """Preserve the candidates supplied by each scoped retrieval attempt."""

    def rerank(
        self,
        question: str,
        candidates: list[HybridSearchResult],
        *,
        top_k: int,
    ) -> SimpleNamespace:
        del question
        return SimpleNamespace(
            results=[
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
                for rank, candidate in enumerate(
                    candidates[:top_k],
                    start=1,
                )
            ],
            reranking_duration_ms=1.0,
        )


class CitationFakeLLM:
    """Cite every selected source without calling a real Ollama server."""

    model_name = "qwen-policy-mock"

    def __init__(self) -> None:
        self.call_count = 0

    def chat_json_with_raw(
        self,
        **kwargs: Any,
    ) -> tuple[GroundedAnswerPayload, str]:
        del kwargs
        self.call_count += 1
        answer = (
            "Réponse [Source 1] [Source 2] [Source 3] "
            "[Source 4] [Source 5]."
        )
        raw = json.dumps({"answer": answer})
        return GroundedAnswerPayload(answer=answer), raw


def make_frozen_config() -> FrozenV3Config:
    """Create frozen-v3 values without reading or changing the snapshot."""

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


def make_service(
    datasets: dict[frozenset[str], list[HybridSearchResult]],
) -> tuple[PhosProcessRAG, ScopedFakeRetriever]:
    """Build a complete fake-driven policy service."""

    retriever = ScopedFakeRetriever(datasets)
    service = PhosProcessRAG(
        frozen_config=make_frozen_config(),
        runtime_config=RAGRuntimeConfig(
            ollama=OllamaConfig(model="qwen-policy-mock"),
            maximum_question_characters=500,
            source_excerpt_characters=120,
            conversation=ConversationRuntimeConfig(),
            generation=GenerationRuntimeConfig(
                max_context_tokens_per_source=80,
                max_total_document_context_tokens=400,
            ),
            source_policy=make_policy_config(),
        ),
        retriever=retriever,
        reranker=DynamicFakeReranker(),
        llm=CitationFakeLLM(),
        verify_snapshot=False,
    )
    return service, retriever


@pytest.mark.parametrize(
    ("question", "route", "source_file"),
    [
        (
            "Pourquoi concentrer l'acide phosphorique ?",
            "phosphoric_acid",
            BECKER_SOURCE,
        ),
        (
            "Comment fonctionne un échangeur thermique ?",
            "heat_transfer",
            HEAT_TRANSFER_SOURCE,
        ),
        (
            "Quels équipements utilise l'atelier OCP JFC4 ?",
            "plant_specific",
            ATELIER_SOURCE,
        ),
    ],
)
def test_sufficient_preferred_document_supplies_five_cited_chunks(
    question: str,
    route: str,
    source_file: str,
) -> None:
    scope = frozenset({document_id_from_source(source_file)})
    service, retriever = make_service(
        {scope: make_candidates(source_file)}
    )

    response = service.answer(question)

    assert response.source_policy_route == route
    assert response.source_policy_fallback_used is False
    assert response.selected_count == 5
    assert len(response.sources) == 5
    assert len({source.chunk_id for source in response.sources}) == 5
    assert {
        source.document_name
        for source in response.sources
    } == {source_file}
    assert retriever.calls[0][1]["document_ids"] == set(scope)


def test_phosphoric_route_falls_back_when_becker_is_insufficient() -> None:
    default_scope = frozenset(
        document_id_from_source(source)
        for source in ALL_SOURCES
    )
    service, retriever = make_service(
        {
            default_scope: make_candidates(PERRY_SOURCE),
        }
    )

    response = service.answer(
        "Pourquoi la filtration de l'acide phosphorique est-elle importante ?"
    )

    assert response.source_policy_route == "phosphoric_acid"
    assert response.source_policy_fallback_used is True
    assert len(retriever.calls) == 3
    assert service.llm.call_count == 1
    assert {
        source.document_name
        for source in response.sources
    } == {PERRY_SOURCE}


def test_forced_becker_excludes_every_other_document() -> None:
    becker_scope = frozenset(
        {document_id_from_source(BECKER_SOURCE)}
    )
    service, retriever = make_service(
        {becker_scope: make_candidates(BECKER_SOURCE)}
    )

    response = service.answer(
        "Question générale sur les équipements",
        source_mode="becker",
    )

    assert response.source_policy_forced is True
    assert response.source_policy_fallback_used is False
    assert len(retriever.calls) == 1
    assert retriever.calls[0][1]["document_ids"] == set(becker_scope)
    assert {
        source.document_name
        for source in response.sources
    } == {BECKER_SOURCE}


def test_explicit_new_active_document_is_retrieved_and_cited() -> None:
    source = "07_process_dynamics_control_seborg_4e.pdf"
    scope = frozenset({document_id_from_source(source)})
    service, retriever = make_service(
        {scope: make_candidates(source)}
    )
    service.active_knowledge_base = SimpleNamespace(
        documents=({"filename": source},)
    )

    response = service.answer(
        "Selon Seborg, quel est le rôle du feedback ?"
    )

    assert response.source_policy_route == "explicit_document"
    assert response.source_policy_forced is True
    assert retriever.calls[0][1]["document_ids"] == set(scope)
    assert len(response.sources) == 5
    assert {
        item.document_name
        for item in response.sources
    } == {source}
