"""Retriever-v4 planning and focus-resolution invariants."""

from __future__ import annotations

from phosprocess.rag.conversation_state import ConversationState
from phosprocess.rag.followup_resolver import resolve_standalone_query
from phosprocess.rag.question_classifier import QuestionType, classify_question
from phosprocess.retrieval.retrieval_planner import build_retrieval_plan


def test_p2o5_balance_has_independent_species_roles() -> None:
    question = (
        "Établis le bilan de P2O5 autour d’un évaporateur "
        "d’acide phosphorique en régime permanent."
    )
    classification = classify_question(question)
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type=classification.question_type.value,
    )

    assert classification.question_type is QuestionType.BALANCE
    assert plan.balance_kind == "species"
    assert {role.name for role in plan.roles} == {
        "species_conservation",
        "species_feed",
        "species_product",
        "species_losses",
    }


def test_energy_balance_does_not_request_mass_or_p2o5_roles() -> None:
    question = (
        "Establish the steady-state energy balance of a forced-circulation "
        "evaporator and define every term used."
    )
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="balance",
    )

    assert plan.balance_kind == "energy"
    assert {role.name for role in plan.roles} == {
        "energy_conservation",
        "heat_input",
        "feed_product_enthalpy",
        "vapor_enthalpy",
    }
    assert all("p2o5" not in role.name for role in plan.roles)


def test_comparison_reserves_both_equipment_sides() -> None:
    question = (
        "Compare a forced-circulation evaporator with a falling-film "
        "evaporator for phosphoric acid concentration."
    )
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="comparison",
    )

    assert plan.comparison_subjects is not None
    assert {role.name for role in plan.roles} == {
        "equipment_a",
        "equipment_b",
        "comparison_criteria",
    }
    assert "forced-circulation" in (plan.roles[0].subject or "")
    assert "falling-film" in (plan.roles[1].subject or "")


def test_focus_entity_resolves_french_and_english_pronouns() -> None:
    state = ConversationState()
    state.observe_question(
        "Quel est le rôle de la pompe de circulation dans cet évaporateur ?"
    )

    assert state.focus_entity == "pompe de circulation"

    french = resolve_standalone_query(
        "Et pourquoi est-elle nécessaire ?",
        state=state,
    )
    english = resolve_standalone_query(
        "How does it send the liquid back to the flash chamber?",
        state=state,
    )

    assert french.standalone_query == (
        "Pourquoi la pompe de circulation est-elle nécessaire ?"
    )
    assert english.standalone_query == (
        "How does the circulation pump send the liquid back to the flash chamber?"
    )
    assert french.resolver_type == "focus_entity"
    assert english.resolver_type == "focus_entity"


def test_process_flow_plan_queries_conical_bottom_explicitly() -> None:
    question = (
        "Describe step by step the path of phosphoric acid through a "
        "forced-circulation evaporator."
    )
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="process_flow",
    )

    assert "conical_bottom" in {role.name for role in plan.roles}
    conical = next(
        role for role in plan.roles if role.name == "conical_bottom"
    )
    assert "conical bottom" in conical.query


def test_pump_necessity_has_independent_functional_roles() -> None:
    question = "Pourquoi la pompe de circulation est-elle nécessaire ?"
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="explanation",
    )

    assert plan.answer_intent == "pump_necessity"
    assert {role.name for role in plan.roles} == {
        "pump_circulation",
        "pump_heating_path",
        "pump_process_function",
    }


def test_definition_plan_requires_nature_mechanism_and_function() -> None:
    question = (
        "C'est quoi un évaporateur à circulation forcée en wet process ?"
    )
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="definition",
    )

    assert {role.name for role in plan.roles} == {
        "definition_nature",
        "definition_mechanism",
        "definition_function",
    }


def test_jfc4_p2o5_balance_uses_plant_specific_roles() -> None:
    question = "Établis le bilan de P2O5 de l’échelon J de JFC4 selon le rapport OCP."
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="balance",
    )

    assert plan.balance_kind == "p2o5_plant"
    assert {role.name for role in plan.roles} == {
        "p2o5_conservation",
        "p2o5_feed",
        "p2o5_product",
        "p2o5_entrainment",
    }


def test_momentum_diffusion_plan_excludes_mass_diffusion_roles() -> None:
    question = "Explain momentum diffusion in a fluid according to Bird."
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="momentum_diffusion",
    )

    assert plan.answer_intent == "momentum_diffusion"
    assert {role.name for role in plan.roles} == {
        "momentum_transport",
        "velocity_gradient",
        "newton_viscosity_law",
    }
    assert all("concentration" not in role.query.casefold() for role in plan.roles)


def test_pump_role_keeps_withdrawal_as_optional_becker_detail() -> None:
    question = "Quel est le rôle de la pompe de circulation ?"
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="explanation",
    )

    roles = {role.name: role for role in plan.roles}
    assert plan.answer_intent == "pump_role"
    assert roles["pump_withdrawal"].required is False
    assert roles["pump_heating_path"].required is True
    assert roles["pump_process_function"].required is False


def test_definition_and_p2o5_composite_roles_only_require_atomic_evidence() -> None:
    definition = build_retrieval_plan(
        "C'est quoi un évaporateur à circulation forcée ?",
        standalone_query="C'est quoi un évaporateur à circulation forcée ?",
        question_type="definition",
    )
    definition_roles = {role.name: role for role in definition.roles}
    assert definition_roles["definition_nature"].required is False
    assert definition_roles["definition_mechanism"].required is True
    assert definition_roles["definition_function"].required is True

    question = (
        "Établis le bilan de P2O5 de l’échelon J de JFC4 selon le rapport OCP."
    )
    balance = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="balance",
    )
    balance_roles = {role.name: role for role in balance.roles}
    assert balance_roles["p2o5_conservation"].required is False
    assert balance_roles["p2o5_feed"].required is True
    assert balance_roles["p2o5_product"].required is True
    assert balance_roles["p2o5_entrainment"].required is True
