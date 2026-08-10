"""Catalog-driven explicit source resolution and legacy source-lock tests."""

from __future__ import annotations

import pytest

from phosprocess.knowledge_base.catalog import load_document_catalog
from phosprocess.knowledge_base.source_resolution import (
    resolve_explicit_source,
    resolve_source_mode,
)
from phosprocess.rag.source_policy import (
    SourcePolicyConfig,
    decide_source_policy,
    detect_explicit_active_source,
)


@pytest.mark.parametrize(
    ("question", "document_id"),
    [
        ("according to Becker, explain the equipment", "becker_phosphates_and_phosphoric_acid"),
        ("selon Becker, explique l'équipement", "becker_phosphates_and_phosphoric_acid"),
        ("dans Perry, que dit le manuel ?", "perrys_chemical_engineers_handbook"),
        ("selon le rapport OCP, donne le bilan", "ocp_phosphoric_acid_workshop_report"),
        ("cherche dans le rapport", "ocp_phosphoric_acid_workshop_report"),
        (
            "Dans l'atelier OCP, quels équipements sont décrits ?",
            "ocp_phosphoric_acid_workshop_report",
        ),
        (
            "Décris le trajet indiqué par le rapport d'atelier.",
            "ocp_phosphoric_acid_workshop_report",
        ),
    ],
)
def test_explicit_source_is_resolved_from_catalog_aliases(
    question: str,
    document_id: str,
) -> None:
    resolution = resolve_explicit_source(question, catalog=load_document_catalog())
    assert resolution is not None
    assert resolution.document_id == document_id


@pytest.mark.parametrize(
    "question",
    [
        "Explique un évaporateur.",
        "Explain crystallization kinetics.",
        "How does process control work?",
        "Describe momentum transport.",
    ],
)
def test_domain_terms_are_not_explicit_source_requests(question: str) -> None:
    assert resolve_explicit_source(question, catalog=load_document_catalog()) is None


def test_query_expansion_alias_is_not_a_user_source_request() -> None:
    catalog = load_document_catalog()
    raw_question = (
        "Comment le coefficient global de transfert thermique intervient-il "
        "dans le dimensionnement ?"
    )
    assert resolve_explicit_source(raw_question, catalog=catalog) is None


def test_manual_source_mode_uses_the_same_catalog_metadata() -> None:
    catalog = load_document_catalog()
    resolution = resolve_source_mode("perry", catalog=catalog)
    assert resolution is not None
    assert resolution.document_id == "perrys_chemical_engineers_handbook"


def test_legacy_policy_stays_global_without_explicit_source() -> None:
    decision = decide_source_policy(
        "Explique un évaporateur.",
        config=SourcePolicyConfig(enabled=False),
    )
    assert decision.forced is False
    assert decision.primary_source is None
    assert decision.preferred_sources == ()


def test_legacy_policy_locks_an_explicit_catalog_source() -> None:
    decision = decide_source_policy(
        "Que dit Perry sur cet équipement ?",
        config=SourcePolicyConfig(enabled=False),
    )
    assert decision.forced is True
    assert decision.mode == "perry"
    assert decision.primary_source == "05_perrys_chemical_engineers_handbook_9e.pdf"


def test_explicit_active_source_requires_the_catalog_file_to_be_active() -> None:
    active = ("04_rapport_atelier_acide_phosphorique.pdf",)
    assert detect_explicit_active_source("cherche dans le rapport", active) == active[0]
    assert detect_explicit_active_source("selon Becker", active) is None
