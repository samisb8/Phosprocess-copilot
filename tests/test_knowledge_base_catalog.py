"""Tests for explicit production document catalogue metadata."""

from __future__ import annotations

import pytest

from phosprocess.knowledge_base.catalog import (
    load_document_catalog,
    verify_catalogue_sources,
)
from phosprocess.knowledge_base.domains import KnowledgeDomain
from phosprocess.knowledge_base.schemas import (
    ExtractionStatus,
    KnowledgeBaseCatalog,
)


def test_catalog_contains_exactly_eight_unique_documents() -> None:
    catalog = load_document_catalog()

    assert len(catalog.documents) == 8
    assert len({document.document_id for document in catalog.documents}) == 8
    assert len({document.canonical_filename for document in catalog.documents}) == 8


def test_catalog_assigns_domains_explicitly() -> None:
    catalog = load_document_catalog()
    documents = {document.document_id: document for document in catalog.documents}

    assert KnowledgeDomain.THERMODYNAMICS in (
        documents["smith_van_ness_chemical_engineering_thermodynamics"].domains
    )
    assert KnowledgeDomain.CRYSTALLIZATION in (
        documents["mullin_crystallization"].domains
    )
    assert KnowledgeDomain.PROCESS_CONTROL in (
        documents["seborg_process_dynamics_control"].domains
    )
    assert documents["ocp_phosphoric_acid_workshop_report"].plant_specific is True


def test_catalog_matches_all_observed_pdf_hashes_and_page_counts() -> None:
    catalog = load_document_catalog()
    results = verify_catalogue_sources(catalog)

    assert set(results) == {document.document_id for document in catalog.documents}
    assert all(valid for valid, _reason in results.values())
    mullin = next(
        document
        for document in catalog.documents
        if document.document_id == "mullin_crystallization"
    )
    assert mullin.active is True
    assert (
        mullin.extraction_status
        is ExtractionStatus.EXTRACTED_SUCCESSFULLY
    )


def test_catalog_rejects_duplicate_document_content() -> None:
    catalog = load_document_catalog()
    documents = list(catalog.documents)
    documents[1] = documents[1].model_copy(
        update={"sha256": documents[0].sha256}
    )

    with pytest.raises(ValueError, match="sha256"):
        KnowledgeBaseCatalog(
            catalog_version=catalog.catalog_version,
            documents=tuple(documents),
        )
