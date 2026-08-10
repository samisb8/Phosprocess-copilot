"""Architecture guard: production RAG must not contain domain answer writers."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PRODUCTION_FILES = (
    "src/phosprocess/rag/answer_validation_service.py",
    "src/phosprocess/rag/generation_service.py",
    "src/phosprocess/rag/orchestrator.py",
    "src/phosprocess/rag/claim_support.py",
    "src/phosprocess/rag/citation_binding.py",
)


def test_production_rag_has_no_deterministic_answer_builders() -> None:
    forbidden = (
        "phosprocess.rag.deterministic_builders",
        "build_atomic_process_flow_answer",
        "build_deterministic_",
    )

    for relative_path in PRODUCTION_FILES:
        text = (ROOT / relative_path).read_text(encoding="utf-8")

        for marker in forbidden:
            assert marker not in text, (
                f"{relative_path} contient encore un answer writer interdit: "
                f"{marker}"
            )


def test_generation_paths_do_not_prune_into_new_answers() -> None:
    for relative_path in (
        "src/phosprocess/rag/generation_service.py",
        "src/phosprocess/rag/orchestrator.py",
    ):
        text = (ROOT / relative_path).read_text(encoding="utf-8")

        assert "prune_unsupported_claims(" not in text, (
            f"{relative_path} ne doit pas reconstruire ou pruner "
            "la réponse à la place du LLM."
        )


def test_active_generation_path_has_no_planner_verifier_or_repair_loop() -> None:
    forbidden = (
        "_plan_answer_requirements(",
        "_validate_answer_semantics(",
        "citation_repair",
        "build_repair_prompt(",
    )
    for relative_path in (
        "src/phosprocess/rag/answer_validation_service.py",
        "src/phosprocess/rag/generation_service.py",
        "src/phosprocess/rag/orchestrator.py",
        "src/phosprocess/rag/prompts.py",
    ):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        for marker in forbidden:
            assert marker not in text, f"Forbidden component active in {relative_path}: {marker}"


def test_production_does_not_import_research_evaluation_modules() -> None:
    for path in (ROOT / "src/phosprocess").rglob("*.py"):
        if "evaluation" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        assert "phosprocess.evaluation" not in text, f"Research import active in {path}"


def test_legacy_domain_answer_writers_are_physically_removed() -> None:
    obsolete_modules = (
        "src/phosprocess/rag/deterministic_builders.py",
        "src/phosprocess/rag/answer_contracts.py",
        "src/phosprocess/rag/fidelity.py",
        "src/phosprocess/retrieval/evidence_coverage.py",
    )
    for relative_path in obsolete_modules:
        assert not (ROOT / relative_path).exists(), f"Module obsolète encore actif: {relative_path}"

    citation_binding = (
        ROOT / "src/phosprocess/rag/citation_binding.py"
    ).read_text(encoding="utf-8")

    forbidden = (
        "_PROCESS_FLOW_ATOMIC_TEMPLATES",
        "_PROCESS_FLOW_STAGE_ORDER",
        "_ATOMIC_STAGE_MARKERS",
        "_ATOMIC_ROLE_BY_STAGE",
        "_atomic_template_stage",
        "_bundle_supports_atomic_stage",
        "_best_bundle_for_atomic_claim",
        "_best_bundle_for_atomic_stage",
        "build_atomic_process_flow_answer",
    )

    for marker in forbidden:
        assert marker not in citation_binding, (
            f"citation_binding.py contient encore du code métier "
            f"de génération déterministe: {marker}"
        )



def test_generation_policy_has_no_arbitrary_length_or_domain_answer_facts() -> None:
    """Generation policy may format answers but must not know domain answers."""

    files = (
        "src/phosprocess/rag/question_classifier.py",
        "src/phosprocess/rag/prompts.py",
        "configs/quality_pipeline.yaml",
    )

    forbidden = (
        "max_words",
        "18.03 t/h",
        "18030 kg/h",
        "18000 kg/h",
        "30 kg/h",
        "feed at line 1",
        "product at line 5",
        "entrainment at line 6",
        "exactly five numbered steps",
    )

    for relative_path in files:
        text = (
            ROOT / relative_path
        ).read_text(
            encoding="utf-8"
        ).casefold()

        for marker in forbidden:
            assert marker.casefold() not in text, (
                f"{relative_path} contient encore une contrainte "
                f"ou un fait m?tier interdit: {marker}"
            )


def test_retrieval_layer_does_not_encode_expected_answer_facts() -> None:
    """Retrieval may encode structure and terminology, never the answer."""

    target_files = (
        "src/phosprocess/retrieval/retrieval_planner.py",
        "src/phosprocess/retrieval/query_expansion.py",
        "src/phosprocess/retrieval/evidence_roles.py",
        "src/phosprocess/rag/quality_retrieval.py",
    )

    forbidden = (
        "18.03 t/h",
        "18030 kg/h",
        "18 t/h",
        "18000 kg/h",
        "30 kg/h",
        "ligne 1",
        "ligne 5",
        "ligne 6",
        "conical_bottom",
        "conical bottom",
        "pump_heat_exchanger",
        "p2o5_plant",
        "p2o5_conservation",
        "p2o5_feed",
        "p2o5_product",
        "p2o5_entrainment",
        "pump withdraws liquor",
        "pump forces liquid",
        "cycling acid leaves vapor body",
    )

    for relative_path in target_files:
        text = (
            ROOT / relative_path
        ).read_text(
            encoding="utf-8"
        ).casefold()

        for marker in forbidden:
            assert marker.casefold() not in text, (
                f"{relative_path} contient encore "
                f"un fait de r?ponse cod?: {marker}"
            )


def test_process_flow_retrieval_roles_are_structural_only() -> None:
    from phosprocess.retrieval.retrieval_planner import (
        build_retrieval_plan,
    )

    question = (
        "D?cris ?tape par ?tape le trajet du fluide "
        "dans cet ?quipement."
    )

    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="process_flow",
    )

    assert tuple(
        role.name
        for role in plan.roles
    ) == (
        "sequence_overview",
        "entry_context",
        "transitions",
        "exit_context",
    )

    assert all(not hasattr(role, "required") for role in plan.roles)



def test_validation_precheck_is_domain_neutral() -> None:
    """Deterministic validation must not know the expected domain answer."""

    claim_support = (
        ROOT
        / "src/phosprocess/rag/claim_support.py"
    ).read_text(
        encoding="utf-8"
    ).casefold()

    citation_binding = (
        ROOT
        / "src/phosprocess/rag/citation_binding.py"
    ).read_text(
        encoding="utf-8"
    ).casefold()

    forbidden_support = (
        "_relation_concepts",
        "_strict_relation_concepts",
        "_all_concepts",
        "conical_bottom",
        "conical bottom",
        "fond conique",
    )

    forbidden_citation = (
        "_max_answer_claims",
        "_missing_process_flow_concepts",
        "pump_heat_exchanger",
    )

    for marker in forbidden_support:
        assert marker not in claim_support, (
            "claim_support.py contient encore "
            f"une r?gle m?tier d?terministe: {marker}"
        )

    for marker in forbidden_citation:
        assert marker not in citation_binding, (
            "citation_binding.py contient encore "
            f"un contrat de r?ponse m?tier: {marker}"
        )
