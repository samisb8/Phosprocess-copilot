"""Domain-neutral fallback retrieval planning invariants."""

from __future__ import annotations

from phosprocess.retrieval.retrieval_planner import build_retrieval_plan


def test_process_flow_roles_describe_information_structure_only() -> None:
    question = "Describe step by step the path of the material through the unit."
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="process_flow",
    )
    assert tuple(role.name for role in plan.roles) == (
        "sequence_overview",
        "entry_context",
        "transitions",
        "exit_context",
    )
    joined = " ".join(role.query for role in plan.roles).casefold()
    for hidden_answer_term in ("conical", "circulation pump", "heat exchanger", "line 1"):
        assert hidden_answer_term not in joined


def test_balance_roles_are_generic_for_any_conserved_quantity() -> None:
    question = "Establish the balance of species X2Y5."
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="balance",
    )
    assert tuple(role.name for role in plan.roles) == (
        "governing_relation",
        "inputs",
        "outputs",
        "assumptions_units",
    )
    assert plan.balance_kind == "species"


def test_comparison_plans_both_user_supplied_subjects() -> None:
    question = "Compare unit A with unit B for this service."
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="comparison",
    )
    assert plan.comparison_subjects == ("unit A", "unit B")
    assert {role.name for role in plan.roles} == {
        "subject_a",
        "subject_b",
        "comparison_criteria",
    }
