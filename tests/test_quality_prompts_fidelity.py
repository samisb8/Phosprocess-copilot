"""Quality prompt budgets, language instructions and offline support tests."""

from __future__ import annotations

import pytest

from phosprocess.rag.citations import CitationValidationError
from phosprocess.rag.fidelity import (
    ClaimSupportStatus,
    build_atomic_process_flow_answer,
    evaluate_claim_support,
    prune_unsupported_claims,
    validate_claim_support,
)
from phosprocess.rag.language import ResponseLanguage
from phosprocess.rag.prompts import build_quality_prompt_package
from phosprocess.rag.question_classifier import classify_question
from phosprocess.retrieval.evidence_bundle import EvidenceBundle


def bundle(number: int, text: str) -> EvidenceBundle:
    return EvidenceBundle(
        source_number=number,
        document_id=f"document_{number}",
        document_title=f"Document {number}",
        filename=f"document_{number}.pdf",
        chapter="Heat Transfer",
        section="Forced Circulation",
        page_start=10,
        page_end=11,
        anchor_chunk_id=f"chunk_{number}",
        expanded_chunk_ids=(f"chunk_{number}",),
        display_text=text,
        token_count=100,
        anchor_score=0.9,
        selection_provenance="reranker",
    )


def test_quality_prompt_enforces_language_evidence_and_dynamic_length() -> None:
    evidence = [
        bundle(
            1,
            "The heat exchanger transfers heat from condensing steam "
            "to circulating phosphoric acid.",
        )
    ]
    classification = classify_question(
        "What is the role of the heat exchanger?"
    )
    system, package = build_quality_prompt_package(
        "What is the role of the heat exchanger?",
        evidence,
        response_language=ResponseLanguage.ENGLISH,
        classification=classification,
        json_output=True,
    )

    assert "Answer exclusively in English." in system
    assert "Use only the supplied evidence bundles." in system
    assert "Maximum answer length: 140 words" in package.user_prompt
    assert "Document: Document 1" in package.user_prompt
    assert package.size.total_tokens < 3500


def test_process_prompt_requests_numbered_steps() -> None:
    evidence = [bundle(1, "Acid enters the pump and then the exchanger.")]
    classification = classify_question(
        "Décris le trajet étape par étape."
    )
    _system, package = build_quality_prompt_package(
        "Décris le trajet étape par étape.",
        evidence,
        response_language=ResponseLanguage.FRENCH,
        classification=classification,
        json_output=False,
    )

    assert "Use numbered steps" in package.user_prompt
    assert "do not invent connectors" in package.user_prompt
    assert "Output exactly five numbered steps" in package.user_prompt
    assert "conical-bottom withdrawal" in package.user_prompt
    assert "two independently cited clauses" in package.user_prompt
    assert "Do not repeat" in package.user_prompt
    assert "Maximum answer length: 320 words" in package.user_prompt


def test_offline_fidelity_reports_supported_and_missing_claims() -> None:
    evidence = [
        bundle(
            1,
            "The heat exchanger transfers thermal energy from condensing "
            "steam to circulating acid.",
        )
    ]
    results = evaluate_claim_support(
        (
            "The heat exchanger transfers thermal energy to the acid "
            "[Source 1]. An invented catalyst is required."
        ),
        evidence,
    )

    assert results[0].status is ClaimSupportStatus.SUPPORTED
    assert results[1].status is ClaimSupportStatus.CITATION_MISSING


def test_offline_fidelity_flags_unrelated_citation() -> None:
    evidence = [bundle(1, "Heat transfer by steam condensation.")]
    result = evaluate_claim_support(
        "A platinum catalyst produces ammonia [Source 1].",
        evidence,
    )[0]

    assert result.status is ClaimSupportStatus.UNSUPPORTED


def test_production_fidelity_rejects_unstated_process_transition() -> None:
    evidence = [
        bundle(
            1,
            "The unit concentrates phosphoric acid under vacuum by evaporation.",
        )
    ]

    with pytest.raises(CitationValidationError):
        validate_claim_support(
            "L’acide entre dans la chambre de flash [Source 1].",
            evidence,
        )


def test_production_fidelity_accepts_explicit_cross_language_transition() -> None:
    evidence = [
        bundle(
            1,
            "The pump withdraws liquor from the flash chamber and forces it "
            "through the heat exchanger before it is returned.",
        )
    ]

    validate_claim_support(
        "La pompe retire le liquide de la chambre de flash, le pousse dans "
        "l’échangeur puis il est renvoyé [Source 1].",
        evidence,
    )


def test_process_flow_equivalent_inlet_and_outlet_relations_are_supported() -> None:
    evidence = [
        bundle(
            1,
            "The liquid phase is fed near the liquid level by the inlet acid "
            "pipe. The concentrated product acid is withdrawn from the vapor "
            "body at an outlet below the feed level.",
        )
    ]

    validate_claim_support(
        "The acid enters through the feed inlet [Source 1]. "
        "The concentrated product acid exits through the product outlet "
        "[Source 1].",
        evidence,
    )


def test_process_flow_pruning_trims_tail_keeps_coverage_and_renumbers() -> None:
    evidence = [
        bundle(
            1,
            "The vapor body separates vapor from liquid. The liquid phase is "
            "fed by the inlet acid pipe coming from the heat exchanger. The "
            "concentrated finished product acid is withdrawn from the vapor "
            "body at an outlet located below the feed level.",
        ),
        bundle(
            2,
            "The pump withdraws liquor from the flash chamber and forces it "
            "through the heating element back to the flash chamber.",
        ),
    ]
    answer = "\n".join(
        [
            "2. Acid enters through the feed inlet [Source 1].",
            (
                "3. The pump forces the acid through the heating element back "
                "to the flash chamber [Source 2]."
            ),
            "4. Vapor and liquid separate in the vapor body [Source 1].",
            (
                "5. The concentrated product acid exits through the product "
                "outlet, completing its path through the evaporator "
                "[Source 1]."
            ),
        ]
    )

    result = prune_unsupported_claims(
        answer,
        evidence,
        fallback_language="en",
        question_type="process_flow",
    )

    assert result.fallback_used is False
    assert result.missing_required_concepts == ()
    assert result.answer.splitlines()[0].startswith("1. ")
    assert result.answer.splitlines()[-1].startswith("4. ")
    assert "completing its path" not in result.answer
    assert "product outlet [Source 1]" in result.answer
    validate_claim_support(result.answer, evidence)


def test_process_flow_pruning_rejects_incomplete_final_coverage() -> None:
    evidence = [
        bundle(
            1,
            "The liquid is fed by the inlet pipe. The vapor body separates "
            "vapor and liquid. Product is withdrawn at the outlet.",
        ),
        bundle(
            2,
            "The pump forces liquor through the heating element.",
        ),
    ]
    answer = "\n".join(
        [
            "1. Liquid enters through the feed inlet [Source 1].",
            "2. The pump forces liquor through the heating element [Source 2].",
            "3. Vapor and liquid separate in the vapor body [Source 1].",
            "4. Product exits through the outlet [Source 1].",
        ]
    )

    result = prune_unsupported_claims(
        answer,
        evidence,
        fallback_language="en",
        question_type="process_flow",
    )

    assert result.fallback_used is True
    assert result.missing_required_concepts == ("recirculation",)


def test_previous_process_flow_output_is_repaired_without_losing_endpoints() -> None:
    evidence = [
        bundle(
            1,
            "The vapor body achieves vapor/liquid separation between evaporated "
            "water and recirculating acid. The liquid phase is fed near the top "
            "of the liquid level by the inlet acid pipe coming from the heat "
            "exchanger. Foaming aggravates entrainment. The concentrated "
            "finished product acid is withdrawn from the vapor body at an outlet "
            "located slightly below the feed level.",
        ),
        bundle(
            2,
            "The pump withdraws liquor from the flash chamber and forces it "
            "through the heating element back to the flash chamber.",
        ),
    ]
    previous_output = "\n".join(
        [
            (
                "1. Phosphoric acid enters the forced-circulation evaporator "
                "through the feed inlet, which is positioned near the top of "
                "the liquid level in the vapor body [Source 1]."
            ),
            (
                "2. The acid is circulated through the heating element, where "
                "heat is applied to evaporate water, and the liquid is forced "
                "back to the flash chamber by a pump [Source 2]."
            ),
            (
                "3. The vaporized water is separated from the liquid in the "
                "vapor body [Source 1]."
            ),
            (
                "4. The concentrated product acid exits through the product "
                "outlet, completing its path through the forced-circulation "
                "evaporator [Source 1]."
            ),
        ]
    )

    result = prune_unsupported_claims(
        previous_output,
        evidence,
        fallback_language="en",
        question_type="process_flow",
    )

    assert result.fallback_used is False
    assert result.missing_required_concepts == ()
    assert "feed inlet" in result.answer
    assert "back to the flash chamber" in result.answer
    assert "product outlet" in result.answer
    assert "heat is applied to evaporate water" not in result.answer
    assert "completing its path" not in result.answer
    assert [line[:2] for line in result.answer.splitlines()] == [
        "1.",
        "2.",
        "3.",
        "4.",
    ]
    validate_claim_support(result.answer, evidence)


def test_atomic_process_flow_planner_builds_five_source_local_steps() -> None:
    evidence = [
        bundle(
            1,
            "The vapor body achieves vapor/liquid separation between the "
            "evaporated water and the recirculating acid. The liquid phase is "
            "fed near the top of the liquid level by the inlet acid pipe coming "
            "from the heat exchanger. The cycling acid leaves the vapor body "
            "through a conical bottom. The concentrated finished product acid "
            "is withdrawn from the vapor body at an outlet located slightly "
            "below the feed level.",
        ),
        bundle(
            2,
            "The use of a pump ensures circulation past the heating surface. "
            "The pump withdraws liquor from the flash chamber and forces it "
            "through the heating element back to the flash chamber.",
        ),
    ]

    answer = build_atomic_process_flow_answer(evidence, language="en")

    assert answer is not None
    lines = answer.splitlines()
    assert len(lines) == 5
    assert [line[:2] for line in lines] == ["1.", "2.", "3.", "4.", "5."]
    assert "inlet acid pipe" in lines[0]
    assert "conical bottom [Source 1]" in lines[1]
    assert "heating element [Source 2]" in lines[2]
    assert "returns to the flash chamber [Source 2]" in lines[3]
    assert "; vapor-liquid separation" in lines[3]
    assert "product outlet [Source 1]" in lines[4]
    assert answer.count("product outlet") == 1
    validate_claim_support(answer, evidence)


def test_latest_compound_process_output_uses_atomic_evidence_recovery() -> None:
    evidence = [
        bundle(
            1,
            "The function of the vapor body is to achieve vapor/liquid "
            "separation between evaporated water and recirculating acid. The "
            "liquid phase is fed near the top of the liquid level by the inlet "
            "acid pipe coming from the heat exchanger. The cycling acid leaves "
            "the vapor body through a conical bottom. The concentrated finished "
            "product acid is withdrawn from the vapor body at an outlet located "
            "slightly below the feed level.",
        ),
        bundle(
            2,
            "The use of a pump to ensure circulation past the heating surface "
            "separates heat transfer from vapor-liquid separation. The pump "
            "withdraws liquor from the flash chamber and forces it through the "
            "heating element back to the flash chamber.",
        ),
    ]
    generated = "\n".join(
        [
            "1. The acid enters through the feed inlet [Source 1].",
            (
                "2. The acid is then circulated through the heating element, "
                "where heat is applied to evaporate water from the solution "
                "[Source 2]."
            ),
            (
                "3. The liquid phase, after being heated and partially "
                "evaporated, leaves the vapor body through a conical bottom and "
                "is recirculated back to the heating element [Source 1]."
            ),
            (
                "4. Vapor-liquid separation takes place in the vapor body "
                "[Source 1]."
            ),
            (
                "5. The concentrated product acid exits through the product "
                "outlet [Source 1]."
            ),
        ]
    )

    result = prune_unsupported_claims(
        generated,
        evidence,
        fallback_language="en",
        question_type="process_flow",
    )

    assert result.fallback_used is False
    assert result.atomic_plan_used is True
    assert result.reconstructed_claim_count == 5
    assert result.missing_required_concepts == ()
    assert "heat is applied to evaporate water" not in result.answer
    assert len(result.answer.splitlines()) == 5
    validate_claim_support(result.answer, evidence)


def test_supported_duplicate_process_flow_is_normalized() -> None:
    evidence = [
        bundle(
            1,
            "The vapor body achieves vapor/liquid separation. The liquid "
            "phase is fed by the inlet acid pipe coming from the heat "
            "exchanger. The cycling acid leaves the vapor body through a "
            "conical bottom. The concentrated finished product acid is "
            "withdrawn from the vapor body at an outlet below the feed level.",
        ),
        bundle(
            2,
            "The pump withdraws liquor from the flash chamber and forces it "
            "through the heating element back to the flash chamber.",
        ),
    ]
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
            (
                "3. The liquor returns to the flash chamber [Source 2]."
            ),
            (
                "4. The concentrated finished product acid is withdrawn "
                "from the vapor body at the product outlet [Source 1]."
            ),
            (
                "5. The product outlet withdraws the concentrated finished "
                "product acid from the vapor body [Source 1]."
            ),
        ]
    )

    result = prune_unsupported_claims(
        duplicate_answer,
        evidence,
        fallback_language="en",
        question_type="process_flow",
    )

    lines = result.answer.splitlines()
    assert result.atomic_plan_used is True
    assert result.reconstructed_claim_count == 5
    assert len(lines) == 5
    assert "conical bottom" in lines[1]
    assert "; vapor-liquid separation" in lines[3]
    assert result.answer.count("product outlet") == 1
    validate_claim_support(result.answer, evidence)


def test_explicitly_cited_semicolon_clauses_validate_independently() -> None:
    evidence = [
        bundle(1, "The vapor body separates vapor from liquid."),
        bundle(2, "The liquor is returned to the flash chamber."),
    ]

    validate_claim_support(
        "The liquor returns to the flash chamber [Source 2]; "
        "vapor-liquid separation takes place in the vapor body [Source 1].",
        evidence,
    )


def test_atomic_process_flow_planner_refuses_incomplete_evidence() -> None:
    evidence = [
        bundle(1, "The acid is fed by the inlet pipe."),
        bundle(2, "The pump forces liquor through the heating element."),
    ]

    assert build_atomic_process_flow_answer(evidence, language="en") is None
