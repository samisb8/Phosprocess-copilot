"""Deterministic tests for adaptive retrieval bypass routing."""

from phosprocess.rag.adaptive_router import (
    DirectIntent,
    RequestPath,
    decide_request_path,
)


def test_translation_is_routed_directly_before_domain_injection() -> None:
    decision = decide_request_path(
        'Traduis "Bonjour" en anglais.',
        source_mode="auto",
    )

    assert decision.path is RequestPath.DIRECT_LLM
    assert decision.direct_intent is DirectIntent.TRANSLATION
    assert decision.requested_output_language == "en"


def test_phosphoric_process_question_keeps_grounded_rag() -> None:
    decision = decide_request_path(
        "Décris le trajet de l’acide dans l’évaporateur.",
        source_mode="auto",
    )

    assert decision.path is RequestPath.DOMAIN_RAG
    assert decision.retrieval_required is True


def test_explicit_document_scope_forces_rag_even_for_translation() -> None:
    decision = decide_request_path(
        'Traduis "Bonjour" en anglais.',
        source_mode="becker",
    )

    assert decision.path is RequestPath.DOMAIN_RAG
    assert decision.reason == "explicit_document_scope"


def test_empty_anaphoric_translation_does_not_bypass_retrieval() -> None:
    decision = decide_request_path("Traduis ça.", source_mode="auto")

    assert decision.path is RequestPath.DOMAIN_RAG


def test_factual_question_uses_fail_closed_documentary_rag() -> None:
    decision = decide_request_path(
        "Who is Victor Hugo?",
        source_mode="auto",
    )

    assert decision.path is RequestPath.DOMAIN_RAG
    assert decision.direct_intent is None


def test_technical_definition_does_not_use_general_bypass() -> None:
    decision = decide_request_path(
        "What is a forced-circulation evaporator?",
        source_mode="auto",
    )

    assert decision.path is RequestPath.DOMAIN_RAG
