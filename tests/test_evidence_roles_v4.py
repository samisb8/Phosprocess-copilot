"""Role-aware final evidence-selection invariants."""

from __future__ import annotations

from phosprocess.preprocessing.chunk_schemas import DocumentChunk
from phosprocess.reranking.reranker import RerankedSearchResult
from phosprocess.retrieval.evidence_roles import select_role_aware_evidence
from phosprocess.retrieval.hybrid import HybridSearchResult
from phosprocess.retrieval.retrieval_planner import build_retrieval_plan


def _chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc",
        source_file="doc.pdf",
        chunk_index=0,
        source_pages=[1],
        page_start=1,
        page_end=1,
        text=text,
        embedding_text=f"Document: Doc\n{text}",
        body_token_count=30,
        token_count=30,
        document_title="Doc",
        active=True,
    )


def _hybrid(
    rank: int,
    chunk: DocumentChunk,
    roles: tuple[str, ...],
) -> HybridSearchResult:
    return HybridSearchResult(
        rank=rank,
        rrf_score=1.0 / rank,
        matched_retrievers=("dense", "bge_sparse", "bm25"),
        dense_rank=rank,
        dense_score=0.8,
        dense_rrf_contribution=0.01,
        bm25_rank=rank,
        bm25_score=2.0,
        bm25_rrf_contribution=0.01,
        chunk=chunk,
        sparse_rank=rank,
        sparse_score=1.0,
        sparse_rrf_contribution=0.01,
        role_matches=roles,
    )


def _reranked(rank: int, candidate: HybridSearchResult) -> RerankedSearchResult:
    return RerankedSearchResult(
        rank=rank,
        reranker_score=1.0 - rank / 10,
        original_hybrid_rank=candidate.rank,
        original_rrf_score=candidate.rrf_score,
        matched_retrievers=candidate.matched_retrievers,
        dense_rank=candidate.dense_rank,
        dense_score=candidate.dense_score,
        bm25_rank=candidate.bm25_rank,
        bm25_score=candidate.bm25_score,
        chunk=candidate.chunk,
        sparse_rank=candidate.sparse_rank,
        sparse_score=candidate.sparse_score,
        role_matches=candidate.role_matches,
    )


def test_comparison_requires_explicit_evidence_for_both_equipment() -> None:
    question = (
        "Compare a forced-circulation evaporator with a falling-film "
        "evaporator for phosphoric acid concentration."
    )
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="comparison",
    )
    a = _hybrid(
        1,
        _chunk("a", "Forced-circulation evaporators use a circulation pump."),
        ("equipment_a", "comparison_criteria"),
    )
    b = _hybrid(
        2,
        _chunk("b", "A falling-film evaporator forms a thin falling liquid film."),
        ("equipment_b", "comparison_criteria"),
    )
    candidates = [a, b]

    selection = select_role_aware_evidence(
        plan,
        candidates,
        [_reranked(1, a), _reranked(2, b)],
        top_k=2,
    )

    assert selection.complete
    assert {item.chunk_id for item in selection.selected} == {"a", "b"}
    assert {"equipment_a", "equipment_b"}.issubset(selection.covered_roles)


def test_comparison_reports_missing_second_side_instead_of_inventing_it() -> None:
    question = (
        "Compare a forced-circulation evaporator with a falling-film "
        "evaporator for phosphoric acid concentration."
    )
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="comparison",
    )
    a = _hybrid(
        1,
        _chunk("a", "Forced-circulation evaporators use a circulation pump."),
        ("equipment_a", "equipment_b", "comparison_criteria"),
    )

    selection = select_role_aware_evidence(
        plan,
        [a],
        [_reranked(1, a)],
        top_k=1,
    )

    assert "equipment_b" in selection.missing_roles


def test_required_balance_roles_are_promoted_into_evidence_window() -> None:
    from phosprocess.retrieval.evidence_roles import (
        promote_required_roles_in_reranking,
    )

    question = "Establish the steady-state energy balance of an evaporator."
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="balance",
    )
    generic = _hybrid(
        1,
        _chunk("generic", "General evaporator design passage."),
        tuple(role.name for role in plan.roles),
    )
    conservation = _hybrid(
        2,
        _chunk("conservation", "Energy balance: energy in equals energy out."),
        ("energy_conservation",),
    )
    heat = _hybrid(
        3,
        _chunk("heat", "The heating medium supplies steam heat input."),
        ("heat_input",),
    )
    enthalpy = _hybrid(
        4,
        _chunk("enthalpy", "Feed enthalpy and product enthalpy are included."),
        ("feed_product_enthalpy",),
    )
    vapor = _hybrid(
        5,
        _chunk("vapor", "The vapor enthalpy includes latent heat."),
        ("vapor_enthalpy",),
    )
    candidates = [generic, conservation, heat, enthalpy, vapor]
    original = [
        _reranked(1, generic),
        _reranked(2, conservation),
        _reranked(3, heat),
        _reranked(4, enthalpy),
        _reranked(5, vapor),
    ]

    promoted = promote_required_roles_in_reranking(
        plan,
        candidates,
        original,
    )

    first_four = {item.chunk.chunk_id for item in promoted[:4]}
    assert first_four == {"conservation", "heat", "enthalpy", "vapor"}


def test_pump_necessity_reserves_functional_evidence_roles() -> None:
    question = "Pourquoi la pompe de circulation est-elle nécessaire ?"
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="explanation",
    )
    circulation = _hybrid(
        1,
        _chunk(
            "circulation",
            "The pump maintains circulation independently of evaporation rate.",
        ),
        ("pump_circulation",),
    )
    heating = _hybrid(
        2,
        _chunk(
            "heating",
            "The pump forces the liquor through the heating surface.",
        ),
        ("pump_heating_path",),
    )
    function = _hybrid(
        3,
        _chunk(
            "function",
            "Heat transfer and vapor liquid separation are separate functions.",
        ),
        ("pump_process_function",),
    )
    candidates = [circulation, heating, function]

    selection = select_role_aware_evidence(
        plan,
        candidates,
        [_reranked(index, item) for index, item in enumerate(candidates, 1)],
        top_k=3,
    )

    assert selection.complete
    assert set(selection.covered_roles) == {
        "pump_circulation",
        "pump_heating_path",
        "pump_process_function",
    }


def test_jfc4_p2o5_roles_accept_report_wording() -> None:
    question = "Établis le bilan de P2O5 de l’échelon J de JFC4 selon le rapport OCP."
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="balance",
    )
    conservation = _hybrid(
        1,
        _chunk(
            "conservation",
            "Le bilan de matière global relie la ligne 1, la ligne 5 et la "
            "ligne 6, avec le P2O5 entraîné à la sortie du bouilleur.",
        ),
        ("p2o5_conservation",),
    )
    feed = _hybrid(
        2,
        _chunk(
            "feed",
            "La ligne 1 est l'entrée acide et le débit massique de P2O5 "
            "dans l'alimentation est 18,03 T/h, soit 18030 kg/h.",
        ),
        ("p2o5_feed",),
    )
    product = _hybrid(
        3,
        _chunk(
            "product",
            "La ligne 5 est la sortie acide produit et contient 18 T/h, "
            "soit 18000 kg/h de P2O5 dans le produit concentré.",
        ),
        ("p2o5_product",),
    )
    loss = _hybrid(
        4,
        _chunk(
            "loss",
            "La ligne 6, sortie bouilleur, contient 30 kg/h de P2O5 "
            "entraîné avec les gaz.",
        ),
        ("p2o5_entrainment",),
    )
    candidates = [conservation, feed, product, loss]

    selection = select_role_aware_evidence(
        plan,
        candidates,
        [
            _reranked(1, conservation),
            _reranked(2, feed),
            _reranked(3, product),
            _reranked(4, loss),
        ],
        top_k=4,
    )

    assert selection.complete
    assert set(selection.covered_roles) == {
        "p2o5_conservation",
        "p2o5_feed",
        "p2o5_product",
        "p2o5_entrainment",
    }


def test_momentum_roles_reject_mass_diffusion_only_passage() -> None:
    question = "Explain momentum diffusion in a fluid according to Bird."
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="momentum_diffusion",
    )
    mass_only = _hybrid(
        1,
        _chunk(
            "mass",
            "Fick's law relates species mass flux to a concentration gradient "
            "and a molecular diffusivity.",
        ),
        tuple(role.name for role in plan.roles),
    )

    selection = select_role_aware_evidence(
        plan,
        [mass_only],
        [_reranked(1, mass_only)],
        top_k=1,
    )

    assert set(selection.missing_roles) == {
        "momentum_transport",
        "velocity_gradient",
        "newton_viscosity_law",
    }


def test_p2o5_numeric_roles_reject_toc_and_unrelated_productivity() -> None:
    question = "Établis le bilan de P2O5 de l’échelon J de JFC4 selon le rapport OCP."
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="balance",
    )
    toc = _hybrid(
        1,
        _chunk(
            "toc",
            "Liste des figures : variation du débit d'acide d'entrée et de sortie.",
        ),
        ("p2o5_feed",),
    )
    toc.chunk.section = "Liste des figures"
    productivity = _hybrid(
        2,
        _chunk(
            "productivity",
            "Pour 54 m3/h, la productivité atteint 418 T P2O5/J avec 50,5 % "
            "à la sortie.",
        ),
        ("p2o5_product",),
    )
    loss = _hybrid(
        3,
        _chunk(
            "loss",
            "La ligne 6 à la sortie du bouilleur contient 30 kg/h de P2O5 "
            "entraîné.",
        ),
        ("p2o5_entrainment",),
    )
    candidates = [toc, productivity, loss]

    selection = select_role_aware_evidence(
        plan,
        candidates,
        [_reranked(index, item) for index, item in enumerate(candidates, 1)],
        top_k=3,
    )

    assert "p2o5_feed" in selection.missing_roles
    assert "p2o5_product" in selection.missing_roles
    assert "p2o5_entrainment" in selection.covered_roles


def test_newton_viscosity_role_rejects_mass_chapter_cross_reference() -> None:
    question = "Explain momentum diffusion in a fluid according to Bird."
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="momentum_diffusion",
    )
    cross_reference = _hybrid(
        1,
        _chunk(
            "mass_chapter",
            "In Chapter 1 we stated Newton's law of viscosity. This chapter "
            "gives Fick's law for mass transport caused by a concentration "
            "gradient and defines molecular diffusivity.",
        ),
        ("newton_viscosity_law",),
    )

    selection = select_role_aware_evidence(
        plan,
        [cross_reference],
        [_reranked(1, cross_reference)],
        top_k=1,
    )

    assert "newton_viscosity_law" in selection.missing_roles


def test_definition_roles_reject_generic_multistage_evaporator_passage() -> None:
    question = "C'est quoi un évaporateur à circulation forcée ?"
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="definition",
    )
    generic = _hybrid(
        1,
        _chunk(
            "generic_definition",
            "Two or three evaporators may operate in parallel or as a "
            "multistage concentration system at 34, 44, and 54 percent.",
        ),
        ("definition_nature", "definition_mechanism"),
    )

    selection = select_role_aware_evidence(
        plan,
        [generic],
        [_reranked(1, generic)],
        top_k=1,
    )

    assert "definition_mechanism" in selection.missing_roles
    assert "definition_nature" not in selection.covered_roles


def test_pump_heating_path_rejects_corrosion_and_electricity_only() -> None:
    question = "Quel est le rôle de la pompe de circulation ?"
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="explanation",
    )
    hazard_only = _hybrid(
        1,
        _chunk(
            "hazard_only",
            "The circulation pump consumes most of the electrical energy, "
            "and its metal impeller is exposed to chloride and fluoride "
            "corrosion.",
        ),
        ("pump_heating_path",),
    )

    selection = select_role_aware_evidence(
        plan,
        [hazard_only],
        [_reranked(1, hazard_only)],
        top_k=1,
    )

    assert "pump_heating_path" in selection.missing_roles


def test_role_selection_preserves_all_roles_bound_to_same_chunk() -> None:
    question = "Quel est le rôle de la pompe de circulation ?"
    plan = build_retrieval_plan(
        question,
        standalone_query=question,
        question_type="explanation",
    )
    path = _hybrid(
        1,
        _chunk(
            "path",
            "The circulation pump withdraws liquor from the flash chamber "
            "and forces it through the heating element.",
        ),
        ("pump_withdrawal", "pump_heating_path"),
    )
    function = _hybrid(
        2,
        _chunk(
            "function",
            "This separates heat transfer, vapor-liquid separation, and "
            "crystallization functions.",
        ),
        ("pump_process_function",),
    )

    selection = select_role_aware_evidence(
        plan,
        [path, function],
        [_reranked(1, path), _reranked(2, function)],
        top_k=2,
    )

    path_selection = next(
        item for item in selection.selected if item.chunk_id == "path"
    )
    assert path_selection.source == (
        "evidence_roles:pump_withdrawal,pump_heating_path"
    )
