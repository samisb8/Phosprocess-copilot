"""Grounded prompts and domain-neutral objective claim checks."""

from __future__ import annotations

import pytest

from phosprocess.rag.citation_binding import iter_answer_claims
from phosprocess.rag.citations import CitationValidationError
from phosprocess.rag.claim_support import validate_claim_support
from phosprocess.rag.language import ResponseLanguage
from phosprocess.rag.prompts import build_quality_prompt_package
from phosprocess.rag.question_classifier import classify_question
from phosprocess.retrieval.evidence_bundle import EvidenceBundle, EvidenceContextScope


def bundle(number: int, text: str) -> EvidenceBundle:
    return EvidenceBundle(
        source_number=number,
        document_id=f"document_{number}",
        document_title=f"Document {number}",
        filename=f"document_{number}.pdf",
        chapter="Generic chapter",
        section="Generic section",
        page_start=1,
        page_end=1,
        parent_id=f"parent_{number}",
        anchor_chunk_ids=(f"chunk_{number}",),
        supporting_chunk_ids=(f"chunk_{number}",),
        display_text=text,
        token_count=40,
        documentary_token_count=20,
        metadata_token_count=20,
        anchor_token_count=20,
        best_anchor_score=0.9,
        context_scope=EvidenceContextScope.ANCHOR_ONLY,
        selection_provenance="reranker",
    )


def test_quality_prompt_is_grounded_and_has_no_semantic_size_limit() -> None:
    system, package = build_quality_prompt_package(
        "Describe the sequence.",
        [bundle(1, "Fluid enters unit A, crosses unit B, and exits unit C.")],
        response_language=ResponseLanguage.ENGLISH,
        classification=classify_question("Describe the path step by step."),
        json_output=True,
    )
    combined = f"{system}\n{package.user_prompt}".casefold()
    assert "supplied evidence bundles" in combined
    assert "include each supported fact only once" in combined
    assert "omit unrelated context" in combined
    assert "stop when the question is answered" in combined
    assert "when the question asks for a sequence" in combined
    assert "exactly five" not in combined
    assert "maximum answer length" not in combined
    assert "conical" not in combined
    assert "product outlet" not in combined


def test_phase11_prompt_variant_is_question_focused_and_context_safe() -> None:
    system, package = build_quality_prompt_package(
        "Describe the sequence.",
        [bundle(1, "Fluid enters unit A, crosses unit B, and exits unit C.")],
        response_language=ResponseLanguage.ENGLISH,
        classification=classify_question("Describe the path step by step."),
        json_output=False,
        prompt_variant="grounded_evidence_utilization_v1",
    )
    combined = f"{system}\n{package.user_prompt}".casefold()
    assert "user's exact question" in combined
    assert "every distinct supported fact necessary" in combined
    assert "unrelated to the question" in combined
    assert "never concatenate distinct documentary sequences" in combined
    assert "immediately after each substantive factual claim" in combined
    assert "copy documentary numbers and units faithfully" in combined
    assert "phosphoric" not in combined
    assert "evaporator" not in combined


def test_unknown_generation_prompt_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown generation prompt variant"):
        build_quality_prompt_package(
            "What is documented?",
            [bundle(1, "A documentary fact.")],
            response_language=ResponseLanguage.ENGLISH,
            classification=classify_question("What is documented?"),
            json_output=False,
            prompt_variant="unknown",  # type: ignore[arg-type]
        )


def test_objective_validator_defers_non_numeric_entailment_to_llm() -> None:
    validate_claim_support(
        "An unrelated semantic assertion [Source 1].",
        [bundle(1, "Different documentary wording.")],
    )


def test_objective_validator_accepts_measurement_present_in_cited_evidence() -> None:
    validate_claim_support(
        "The documented flow is 12 kg/h [Source 1].",
        [bundle(1, "The measured flow rate equals 12 kg per hour.")],
    )


def test_objective_validator_rejects_measurement_absent_from_cited_evidence() -> None:
    with pytest.raises(CitationValidationError, match="Valeur ou unité"):
        validate_claim_support(
            "The documented flow is 15 kg/h [Source 1].",
            [bundle(1, "The measured flow rate equals 12 kg per hour.")],
        )


def test_claim_parser_preserves_numbered_items_and_independent_citations() -> None:
    claims = iter_answer_claims(
        "1. Fluid enters unit A [Source 1].\n"
        "2. It crosses unit B [Source 1]; it exits unit C [Source 2]."
    )
    assert len(claims) == 3
    assert claims[0].startswith("1. ")
