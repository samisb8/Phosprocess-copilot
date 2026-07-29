"""End-to-end answer contracts introduced by Retriever v4 integration."""

from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from phosprocess.rag.conversation_state import ConversationState
from phosprocess.rag.fidelity import (
    build_atomic_process_flow_answer,
    build_deterministic_balance_answer,
    enforce_answer_contract,
    prune_unsupported_claims,
    validate_claim_support,
)
from phosprocess.rag.language import ResponseLanguage
from phosprocess.rag.pipeline import PhosProcessRAG
from phosprocess.rag.prompts import build_quality_prompt_package
from phosprocess.rag.question_classifier import (
    QuestionType,
    classify_question,
)
from phosprocess.rag.schemas import RAGResponse, RAGTimings
from phosprocess.retrieval.evidence_bundle import EvidenceBundle


def bundle(
    number: int,
    text: str,
    *,
    provenance: str = "reranker",
) -> EvidenceBundle:
    return EvidenceBundle(
        source_number=number,
        document_id=f"document_{number}",
        document_title=f"Document {number}",
        filename=f"document_{number}.pdf",
        chapter="Evaporation",
        section="Forced circulation",
        page_start=1,
        page_end=2,
        anchor_chunk_id=f"chunk_{number}",
        expanded_chunk_ids=(f"chunk_{number}",),
        display_text=text,
        token_count=120,
        anchor_score=0.9,
        selection_provenance=provenance,
    )


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


def test_rag_response_accepts_thirty_candidates_but_not_thirty_one() -> None:
    assert response_with_candidate_count(30).candidate_count == 30

    with pytest.raises(ValidationError):
        response_with_candidate_count(31)


def test_french_contracted_definition_is_classified_as_definition() -> None:
    classification = classify_question(
        "Qu’est-ce qu’un évaporateur à circulation forcée ?"
    )

    assert classification.question_type is QuestionType.DEFINITION


def test_multilingual_definition_keeps_definition_mechanism_and_function() -> None:
    evidence = [
        bundle(
            1,
            "A forced-circulation evaporator is an evaporator in which a "
            "pump forces the liquid through a heating element and returns it "
            "to the vapor body. This separates heat transfer, vapor-liquid "
            "separation, and crystallization functions.",
        )
    ]
    answer = (
        "Un évaporateur à circulation forcée est un type d’évaporateur où "
        "le liquide est pompé à travers un échangeur de chaleur et "
        "réintroduit dans le corps de l’évaporateur [Source 1]. "
        "Ce type d’évaporateur permet de séparer les fonctions de transfert "
        "de chaleur, de séparation vapeur-liquide et de cristallisation "
        "[Source 1]."
    )

    validate_claim_support(answer, evidence)
    result = enforce_answer_contract(
        answer,
        evidence,
        question_type="definition",
        language="fr",
    )

    assert result.fallback_used is False
    assert result.missing_roles == ()
    assert "pompe" in result.answer.casefold()
    assert "séparation vapeur-liquide" in result.answer.casefold()
    validate_claim_support(result.answer, evidence)


def test_process_flow_contract_always_reconstructs_five_atomic_steps() -> None:
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

    result = enforce_answer_contract(
        "A badly ordered answer [Source 1].",
        evidence,
        question_type="process_flow",
        language="en",
    )

    lines = result.answer.splitlines()
    assert result.atomic_plan_used is True
    assert result.fallback_used is False
    assert len(lines) == 5
    assert "fed" in lines[0].casefold()
    assert "conical bottom" in lines[1].casefold()
    assert "pump" in lines[2].casefold()
    assert "returns to the flash chamber" in lines[3].casefold()
    assert "product outlet" in lines[4].casefold()
    validate_claim_support(result.answer, evidence)


def test_process_flow_composite_stage_uses_retrieval_roles() -> None:
    evidence = [
        bundle(
            1,
            "The acid is fed through the inlet acid pipe. The cycling acid "
            "leaves the vapor body through a conical bottom.",
            provenance="evidence_role:feed_inlet",
        ),
        bundle(
            2,
            "The pump withdraws liquor from the flash chamber and forces it "
            "through the heating element.",
            provenance="evidence_role:pump_heat_exchanger",
        ),
        bundle(
            3,
            "The liquor returns to the flash chamber.",
            provenance="evidence_role:recirculation",
        ),
        bundle(
            4,
            "Vapor-liquid separation takes place in the vapor body.",
            provenance="evidence_role:vapor_body",
        ),
        bundle(
            5,
            "The concentrated finished product acid is withdrawn from the "
            "vapor body at the product outlet.",
            provenance="evidence_role:product_outlet",
        ),
    ]

    answer = build_atomic_process_flow_answer(evidence, language="en")

    assert answer is not None
    assert len(answer.splitlines()) == 5
    assert "returns to the flash chamber" in answer.casefold()
    assert "vapor-liquid separation" in answer.casefold()
    validate_claim_support(answer, evidence)


def test_process_flow_builder_reads_coverage_guard_multi_role_provenance() -> None:
    evidence = [
        bundle(
            1,
            "The liquid phase is fed by the inlet acid pipe. The cycling "
            "acid leaves the vapor body through a conical bottom. The "
            "concentrated product acid is withdrawn at the product outlet.",
            provenance=(
                "coverage_guard;evidence_roles:feed_inlet,conical_bottom,"
                "product_outlet"
            ),
        ),
        bundle(
            2,
            "The pump withdraws liquor from the flash chamber and forces it "
            "through the heating element back to the flash chamber. "
            "Vapor-liquid separation takes place in the vapor body.",
            provenance=(
                "coverage_guard;evidence_roles:pump_heat_exchanger,"
                "vapor_body,recirculation"
            ),
        ),
    ]

    answer = build_atomic_process_flow_answer(evidence, language="en")

    assert answer is not None
    assert len(answer.splitlines()) == 5
    assert "returns to the flash chamber" in answer.casefold()
    assert "vapor-liquid separation" in answer.casefold()
    validate_claim_support(answer, evidence)


def test_process_flow_pruning_builds_from_pre_pruning_evidence() -> None:
    evidence = [
        bundle(
            1,
            "The vapor body achieves vapor/liquid separation. The liquid "
            "phase is fed by the inlet acid pipe coming from the heat "
            "exchanger. The cycling acid leaves through a conical bottom. "
            "The concentrated product is withdrawn at the product outlet.",
            provenance="evidence_role:feed_inlet",
        ),
        bundle(
            2,
            "The pump withdraws liquor from the flash chamber and forces it "
            "through the heating element back to the flash chamber. "
            "Vapor-liquid separation takes place in the vapor body.",
            provenance="evidence_role:recirculation_vapor_body",
        ),
    ]

    result = prune_unsupported_claims(
        "Unsupported generated wording without citations.",
        evidence,
        fallback_language="en",
        question_type="process_flow",
    )

    assert result.atomic_plan_used is True
    assert result.fallback_used is False
    assert result.reconstructed_claim_count == 5
    assert len(result.answer.splitlines()) == 5
    validate_claim_support(result.answer, evidence)


def test_comparison_contract_removes_grounded_but_off_task_claim() -> None:
    answer = "\n".join(
        [
            (
                "Falling-film evaporators are less suited for liquids "
                "containing solids or prone to scaling because feed "
                "distributors may plug [Source 1]."
            ),
            (
                "Forced-circulation evaporators provide positive circulation "
                "and higher heat-transfer coefficients [Source 2]."
            ),
            (
                "Falling-film evaporators operate with a smaller pressure "
                "drop through the tubes [Source 1]."
            ),
            (
                "The choice between the two systems may depend on sludge "
                "treatment at 40% P2O5 [Source 3]."
            ),
        ]
    )

    result = enforce_answer_contract(
        answer,
        [],
        question_type="comparison",
        language="en",
        comparison_subjects=(
            "a forced-circulation evaporator",
            "a falling-film evaporator",
        ),
    )

    assert result.fallback_used is False
    assert "sludge" not in result.answer.casefold()
    assert "40%" not in result.answer
    assert len(result.answer.splitlines()) == 3
    assert result.missing_roles == ()


def test_troubleshooting_contract_keeps_problem_specific_chain_only() -> None:
    answer = "\n".join(
        [
            (
                "Fouling can reduce steam economy because it reduces "
                "heat-transfer efficiency [Source 1]."
            ),
            (
                "Fouling is the formation of deposits due to corrosion or "
                "solid matter entering with the feed [Source 2]."
            ),
            (
                "These deposits cause a rapid decrease in heat-transfer "
                "coefficients and may require shutdown and washing [Source 3]."
            ),
            "Defoamers reduce carryover losses [Source 4].",
            "Lower reactor strengths improve filtration rates [Source 4].",
        ]
    )

    result = enforce_answer_contract(
        answer,
        [],
        question_type="troubleshooting",
        language="en",
    )

    assert result.fallback_used is False
    assert "defoamer" not in result.answer.casefold()
    assert "filtration" not in result.answer.casefold()
    lines = result.answer.splitlines()
    assert "due to corrosion" in lines[0].casefold()
    assert "washing" in " ".join(lines).casefold()
    assert result.missing_roles == ()


def test_quality_prompts_state_task_contracts() -> None:
    evidence = [bundle(1, "Documented evidence.")]

    definition = classify_question(
        "Qu’est-ce qu’un évaporateur à circulation forcée ?"
    )
    _system, definition_prompt = build_quality_prompt_package(
        "Qu’est-ce qu’un évaporateur à circulation forcée ?",
        evidence,
        response_language=ResponseLanguage.FRENCH,
        classification=definition,
        json_output=False,
    )
    assert "all three supported parts" in definition_prompt.user_prompt

    comparison = classify_question("Compare A versus B.")
    _system, comparison_prompt = build_quality_prompt_package(
        "Compare A versus B.",
        evidence,
        response_language=ResponseLanguage.ENGLISH,
        classification=comparison,
        json_output=False,
    )
    assert "explicitly name equipment A" in comparison_prompt.user_prompt
    assert "Exclude unrelated process facts" in comparison_prompt.user_prompt

    troubleshooting = classify_question(
        "Explain fouling causes and remedies."
    )
    _system, troubleshooting_prompt = build_quality_prompt_package(
        "Explain fouling causes and remedies.",
        evidence,
        response_language=ResponseLanguage.ENGLISH,
        classification=troubleshooting,
        json_output=False,
    )
    assert "cause; physical mechanism; operational effect" in (
        troubleshooting_prompt.user_prompt
    )


def test_quality_stream_builds_public_response_before_emitting_answer() -> None:
    source = inspect.getsource(PhosProcessRAG.stream_answer)

    response_position = source.index("response = self._build_response(")
    buffered_token_position = source.index(
        "if buffer_until_validated:",
        response_position,
    )
    buffered_token_position = source.index(
        "yield RAGStreamEvent(", buffered_token_position
    )

    assert response_position < buffered_token_position


def test_process_flow_contract_recovers_after_pruning_fallback() -> None:
    source = inspect.getsource(PhosProcessRAG.stream_answer)

    pruning_position = source.index("prune_unsupported_claims(")
    contract_position = source.index(
        "contract = enforce_answer_contract(",
        pruning_position,
    )
    assert pruning_position < contract_position
    assert "if quality_bundles is not None:" in source[
        pruning_position:contract_position
    ]


def test_overall_mass_balance_builder_produces_equation_and_definitions() -> None:
    evidence = [
        bundle(
            1,
            "At steady state, conservation of mass requires total mass in "
            "to equal total mass out.",
            provenance="evidence_role:overall_conservation",
        ),
        bundle(
            2,
            "The feed stream enters the evaporator as dilute phosphoric acid.",
            provenance="evidence_role:feed_stream",
        ),
        bundle(
            3,
            "The evaporator has a concentrated liquid product and an "
            "evaporated-water vapor outlet.",
            provenance="evidence_role:product_and_vapor",
        ),
    ]

    answer = build_deterministic_balance_answer(
        evidence,
        balance_kind="overall_mass",
        language="en",
    )

    assert answer is not None
    assert "F = P + V" in answer
    assert "F is" in answer
    assert "P is" in answer
    validate_claim_support(answer, evidence)


def test_p2o5_balance_builder_produces_component_equation() -> None:
    evidence = [
        bundle(
            1,
            "A component balance follows conservation of mass at steady state.",
            provenance="evidence_role:species_conservation",
        ),
        bundle(
            2,
            "The P2O5 feed flow and feed concentration enter the evaporator.",
            provenance="evidence_role:species_feed",
        ),
        bundle(
            3,
            "The concentrated product outlet contains P2O5.",
            provenance="evidence_role:species_product",
        ),
        bundle(
            4,
            "P2O5 losses may occur through entrainment or carryover.",
            provenance="evidence_role:species_losses",
        ),
    ]

    result = enforce_answer_contract(
        "Incomplete answer [Source 1].",
        evidence,
        question_type="balance",
        language="fr",
        question="Établis le bilan de P2O5 en régime permanent.",
        balance_kind="species",
    )

    assert result.fallback_used is False
    assert "F x_F = P x_P + L_P2O5" in result.answer
    assert "L_P2O5 = 0" not in result.answer
    validate_claim_support(result.answer, evidence)


def test_energy_balance_builder_produces_all_stream_terms() -> None:
    evidence = [
        bundle(
            1,
            "The steady-state energy balance includes heat, work, and "
            "enthalpy in and out of the control volume.",
            provenance="evidence_role:energy_conservation",
        ),
        bundle(
            2,
            "Heating steam supplies the heat input and the pump supplies "
            "shaft work.",
            provenance="evidence_role:heat_input",
        ),
        bundle(
            3,
            "Feed enthalpy and concentrated-product enthalpy are liquid "
            "enthalpy terms.",
            provenance="evidence_role:feed_product_enthalpy",
        ),
        bundle(
            4,
            "Generated vapor carries vapor enthalpy and latent heat.",
            provenance="evidence_role:vapor_enthalpy",
        ),
    ]

    result = enforce_answer_contract(
        "Generic energy statement [Source 1].",
        evidence,
        question_type="balance",
        language="en",
        question="Establish the steady-state energy balance.",
        balance_kind="energy",
    )

    assert result.fallback_used is False
    assert "Qdot + F h_F + Wdot_s" in result.answer
    assert "V h_V" in result.answer
    validate_claim_support(result.answer, evidence)


def test_pump_followups_are_limited_to_the_requested_mechanism() -> None:
    evidence = [
        bundle(
            1,
            "The pump withdraws liquor from the flash chamber and forces it "
            "through the heating element back to the flash chamber. "
            "Circulation past the heating surface is maintained independently "
            "of the evaporation rate. This separates heat transfer, "
            "vapor-liquid separation, and crystallization functions.",
        )
    ]

    necessity = enforce_answer_contract(
        "The pump consumes electrical energy [Source 1].",
        evidence,
        question_type="explanation",
        language="fr",
        question="Pourquoi la pompe de circulation est-elle nécessaire ?",
    )
    assert necessity.fallback_used is False
    assert "indépendamment du taux d’évaporation" in necessity.answer
    assert "corrosion" not in necessity.answer.casefold()

    returned = enforce_answer_contract(
        "Long answer [Source 1].",
        evidence,
        question_type="explanation",
        language="en",
        question=(
            "How does the circulation pump send the liquid back to the "
            "flash chamber?"
        ),
    )
    assert returned.answer.count("\n") == 0
    assert "back to the flash chamber" in returned.answer
    validate_claim_support(returned.answer, evidence)


def test_arabic_vapor_body_answer_is_canonically_validated() -> None:
    evidence = [
        bundle(
            1,
            "The vapor body provides vapor-liquid separation after the heated "
            "acid returns from the heating element.",
        )
    ]

    result = enforce_answer_contract(
        "إجابة غير مكتملة [Source 1].",
        evidence,
        question_type="explanation",
        language="ar",
        question="ما هو دور غرفة التبخير في فصل البخار عن الحمض؟",
    )

    assert result.fallback_used is False
    assert "فصل البخار" in result.answer
    validate_claim_support(result.answer, evidence)


def test_pump_necessity_contract_uses_resolved_question_and_roles() -> None:
    evidence = [
        bundle(
            1,
            "A pump ensures circulation past the heating surface. "
            "Circulation is maintained independently of evaporation rate.",
            provenance="evidence_role:pump_circulation",
        ),
        bundle(
            2,
            "The pump forces the liquid through the heating element.",
            provenance="evidence_role:pump_heating_path",
        ),
        bundle(
            3,
            "This separates heat transfer from vapor-liquid separation and "
            "crystallization.",
            provenance="evidence_role:pump_process_function",
        ),
    ]

    result = enforce_answer_contract(
        "La pompe consomme de l'électricité et subit la corrosion [Source 1].",
        evidence,
        question_type="explanation",
        language="fr",
        question="Pourquoi la pompe de circulation est-elle nécessaire ?",
    )

    assert result.fallback_used is False
    assert "maintient une circulation positive" in result.answer
    assert "corrosion" not in result.answer.casefold()
    validate_claim_support(result.answer, evidence)


def test_explicit_source_scope_persists_only_across_followups() -> None:
    rag = object.__new__(PhosProcessRAG)
    rag.quality_engine = object()
    state = ConversationState()

    first = rag._resolve_turn_source_mode(
        "auto",
        question="Selon Becker, définis l'évaporateur à circulation forcée.",
        follow_up=False,
        state=state,
    )
    inherited = rag._resolve_turn_source_mode(
        "auto",
        question="Et pourquoi sa pompe est-elle nécessaire ?",
        follow_up=True,
        state=state,
    )
    reset = rag._resolve_turn_source_mode(
        "auto",
        question="Explique le coefficient global d'un échangeur.",
        follow_up=False,
        state=state,
    )

    assert first == "becker"
    assert inherited == "becker"
    assert reset == "auto"
    assert state.source_scope_explicit is False


def test_jfc4_p2o5_balance_builder_uses_report_values_and_loss() -> None:
    evidence = [
        bundle(
            1,
            "Le bilan de matière global de l'échelon J inclut le P2O5 "
            "entrant, le produit et le P2O5 entraîné à la sortie du bouilleur.",
            provenance="evidence_role:p2o5_conservation",
        ),
        bundle(
            2,
            "Le débit de P2O5 à l'entrée de la ligne 1 est 18,03 T/h.",
            provenance="evidence_role:p2o5_feed",
        ),
        bundle(
            3,
            "Le débit de P2O5 à la sortie produit de la ligne 5 est 18 T/h.",
            provenance="evidence_role:p2o5_product",
        ),
        bundle(
            4,
            "Le P2O5 entraîné à la sortie du bouilleur est égal à 30 kg/h.",
            provenance="evidence_role:p2o5_entrainment",
        ),
    ]

    result = enforce_answer_contract(
        "Réponse incomplète [Source 1].",
        evidence,
        question_type="balance",
        language="fr",
        question=(
            "Établis le bilan de P2O5 de l’échelon J de JFC4 selon le rapport OCP."
        ),
        balance_kind="p2o5_plant",
    )

    assert result.fallback_used is False
    assert "18,03" in result.answer
    assert "18,00" in result.answer
    assert "30 kg/h" in result.answer
    assert "18,03 = 18,00 + 0,03 t/h" not in result.answer
    validate_claim_support(result.answer, evidence)


def test_momentum_diffusion_contract_excludes_fick_law() -> None:
    evidence = [
        bundle(
            1,
            "Molecular transport of momentum is described as a momentum flux "
            "between adjacent fluid layers.",
            provenance="evidence_role:momentum_transport",
        ),
        bundle(
            2,
            "The momentum flux is associated with a velocity gradient and "
            "appears mechanically as shear stress.",
            provenance="evidence_role:velocity_gradient",
        ),
        bundle(
            3,
            "Newton's law of viscosity relates shear stress to the velocity "
            "gradient through the dynamic viscosity.",
            provenance="evidence_role:newton_viscosity_law",
        ),
    ]

    result = enforce_answer_contract(
        "Fick's law uses a concentration gradient [Source 1].",
        evidence,
        question_type="momentum_diffusion",
        language="en",
        question="Explain momentum diffusion according to Bird.",
    )

    assert result.fallback_used is False
    assert "velocity gradient" in result.answer.casefold()
    assert "shear stress" in result.answer.casefold()
    assert "dynamic viscosity" in result.answer.casefold()
    assert "fick" not in result.answer.casefold()
    assert "concentration gradient" not in result.answer.casefold()
    validate_claim_support(result.answer, evidence)


def test_explicit_all_sources_request_releases_source_lock() -> None:
    rag = object.__new__(PhosProcessRAG)
    rag.quality_engine = object()
    state = ConversationState()
    state.record_source_scope(
        "becker",
        explicit=True,
        origin="user_question",
    )

    mode = rag._resolve_turn_source_mode(
        "auto",
        question="Cherche maintenant dans toutes les sources.",
        follow_up=True,
        state=state,
    )

    assert mode == "auto"
    assert state.current_source_mode == "auto"
    assert state.source_scope_explicit is False
    assert state.source_scope_origin is None


def test_p2o5_role_tags_cannot_inject_values_absent_from_evidence() -> None:
    evidence = [
        bundle(
            1,
            "Le bilan de matière global relie l'alimentation, le produit et "
            "le P2O5 entraîné.",
            provenance="evidence_role:p2o5_conservation",
        ),
        bundle(
            2,
            "Liste des figures : variation du débit d'acide d'entrée.",
            provenance="evidence_role:p2o5_feed",
        ),
        bundle(
            3,
            "La productivité cible est 418 T P2O5/J.",
            provenance="evidence_role:p2o5_product",
        ),
        bundle(
            4,
            "Le P2O5 entraîné à la sortie du bouilleur est 30 kg/h.",
            provenance="evidence_role:p2o5_entrainment",
        ),
    ]

    result = enforce_answer_contract(
        "Réponse incomplète [Source 1].",
        evidence,
        question_type="balance",
        language="fr",
        question=(
            "Établis le bilan de P2O5 de l’échelon J de JFC4 selon le rapport OCP."
        ),
        balance_kind="p2o5_plant",
    )

    assert result.fallback_used is True
    assert "18,03" not in result.answer
    assert "18,00" not in result.answer


def test_momentum_builder_ignores_mass_diffusion_role_mistag() -> None:
    evidence = [
        bundle(
            1,
            "Momentum is transmitted between adjacent fluid layers as a "
            "molecular momentum flux.",
            provenance="evidence_role:momentum_transport",
        ),
        bundle(
            2,
            "The velocity gradient is the driving force for momentum "
            "transport and is expressed mechanically by shear stress.",
            provenance="evidence_role:velocity_gradient",
        ),
        bundle(
            3,
            "This mass-transport chapter mentions Newton's law of viscosity, "
            "then presents Fick's law, concentration gradient, and molecular "
            "diffusivity.",
            provenance="evidence_role:newton_viscosity_law",
        ),
        bundle(
            4,
            "Newton's law of viscosity relates shear stress to the velocity "
            "gradient through dynamic viscosity.",
            provenance="reranker_fill",
        ),
    ]

    result = enforce_answer_contract(
        "Fick's law explains diffusion [Source 3].",
        evidence,
        question_type="momentum_diffusion",
        language="en",
        question="Explain momentum diffusion according to Bird.",
    )

    assert result.fallback_used is False
    assert "[Source 4]" in result.answer.splitlines()[-1]
    assert "[Source 3]" not in result.answer.splitlines()[-1]
    assert "Fick" not in result.answer
    validate_claim_support(result.answer, evidence)


def test_definition_builder_binds_becker_claims_to_semantic_sources() -> None:
    evidence = [
        bundle(
            1,
            "Two or three evaporators may operate in parallel or as a "
            "multistage concentration system at several concentration levels.",
            provenance="evidence_role:definition_nature",
        ),
        bundle(
            2,
            "The type of acid circulation pump depends on the pressure drop "
            "of the heat exchanger. Large-diameter exchangers use pumps with "
            "a large flow capacity.",
            provenance="evidence_role:definition_mechanism",
        ),
        bundle(
            3,
            "The vapor body receives heated acid from the heat exchanger and "
            "provides vapor-liquid separation between evaporated water and "
            "recirculating acid.",
            provenance="evidence_role:definition_function",
        ),
    ]

    result = enforce_answer_contract(
        "Réponse générique [Source 1].",
        evidence,
        question_type="definition",
        language="fr",
        question="C'est quoi un évaporateur à circulation forcée selon Becker ?",
    )

    assert result.fallback_used is False
    assert "[Source 1]" not in result.answer
    assert "[Source 2]" in result.answer.splitlines()[0]
    assert "[Source 3]" in result.answer.splitlines()[1]
    assert "perte de charge" in result.answer.casefold()
    assert "séparation vapeur-liquide" in result.answer.casefold()
    validate_claim_support(result.answer, evidence)


def test_pump_builder_ignores_hazard_mistag_and_binds_flow_path_source() -> None:
    evidence = [
        bundle(
            1,
            "The circulation pump consumes electrical energy and its impeller "
            "is exposed to chloride and fluoride corrosion.",
            provenance="evidence_role:pump_heating_path",
        ),
        bundle(
            2,
            "The pump withdraws liquor from the flash chamber and forces it "
            "through the heating element.",
            provenance="reranker_fill",
        ),
        bundle(
            3,
            "This separates heat transfer, vapor-liquid separation, and "
            "crystallization functions.",
            provenance="evidence_role:pump_process_function",
        ),
    ]

    result = enforce_answer_contract(
        "La pompe consomme de l'électricité [Source 1].",
        evidence,
        question_type="explanation",
        language="fr",
        question="Quel est le rôle de la pompe de circulation ?",
    )

    assert result.fallback_used is False
    assert "[Source 2]" in result.answer.splitlines()[0]
    assert "[Source 1]" not in result.answer
    assert "corrosion" not in result.answer.casefold()
    validate_claim_support(result.answer, evidence)


def test_p2o5_composite_balance_binds_each_value_to_its_own_source() -> None:
    evidence = [
        bundle(
            1,
            "For P2O5, m6 entrained equals m1 feed minus m5 product. "
            "The P2O5 entrained at the boiler outlet is 30 kg/h.",
            provenance=(
                "evidence_roles:p2o5_conservation,p2o5_entrainment"
            ),
        ),
        bundle(
            2,
            "The stage-J feed, line 1, contains 18.03 t/h of P2O5.",
            provenance="evidence_role:p2o5_feed",
        ),
        bundle(
            3,
            "The concentrated product at line 5 contains 18.00 t/h of P2O5.",
            provenance="evidence_role:p2o5_product",
        ),
    ]

    result = enforce_answer_contract(
        "Réponse incomplète [Source 1].",
        evidence,
        question_type="balance",
        language="fr",
        question=(
            "Établis le bilan de P2O5 de l’échelon J de JFC4 selon le rapport OCP."
        ),
        balance_kind="p2o5_plant",
    )

    assert result.fallback_used is False
    lines = result.answer.splitlines()
    assert "[Source 2]" in next(line for line in lines if "18,03" in line)
    assert "[Source 3]" in next(line for line in lines if "18,00" in line)
    assert "[Source 1]" in next(line for line in lines if "30 kg/h" in line)
    validate_claim_support(result.answer, evidence)
