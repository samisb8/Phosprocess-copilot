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
