"""Architecture guards for the behavior-preserving RAG refactor."""

from __future__ import annotations

from phosprocess.rag.answer_validation_service import AnswerValidationService
from phosprocess.rag.generation_service import GenerationService
from phosprocess.rag.orchestrator import PhosProcessRAG
from phosprocess.rag.pipeline import (
    FrozenV3Config,
    RAGRuntimeConfig,
    load_frozen_v3_config,
)
from phosprocess.rag.retrieval_service import RetrievalService


def test_pipeline_facade_keeps_public_runtime_imports() -> None:
    assert FrozenV3Config.__module__ == "phosprocess.rag.orchestrator"
    assert RAGRuntimeConfig.__module__ == "phosprocess.rag.orchestrator"
    assert load_frozen_v3_config.__module__ == "phosprocess.rag.orchestrator"


def test_orchestrator_composes_the_three_runtime_services() -> None:
    assert issubclass(PhosProcessRAG, RetrievalService)
    assert issubclass(PhosProcessRAG, GenerationService)
    assert issubclass(PhosProcessRAG, AnswerValidationService)


def test_pipeline_methods_live_in_expected_modules() -> None:
    assert PhosProcessRAG.stream_answer.__module__ == "phosprocess.rag.orchestrator"
    assert PhosProcessRAG._retrieve.__module__ == "phosprocess.rag.retrieval_service"
    assert (
        PhosProcessRAG._generate_json_answer.__module__
        == "phosprocess.rag.generation_service"
    )
    assert (
        PhosProcessRAG._validate_answer.__module__
        == "phosprocess.rag.answer_validation_service"
    )


def test_objective_validation_and_claim_parsing_have_direct_owners() -> None:
    from phosprocess.rag.citation_binding import iter_answer_claims
    from phosprocess.rag.claim_support import validate_claim_support

    assert iter_answer_claims.__module__ == "phosprocess.rag.citation_binding"
    assert validate_claim_support.__module__ == "phosprocess.rag.claim_support"
