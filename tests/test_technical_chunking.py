"""Parent–child technical chunk schemas, links and boundary tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from docling_core.types.doc import DocItemLabel
from transformers import AutoTokenizer

from phosprocess.embeddings.embedder import resolve_cached_model_source
from phosprocess.ingestion.chunk_serialization import (
    TechnicalChunkType,
    read_child_chunks,
    read_parent_chunks,
    write_jsonl,
)
from phosprocess.ingestion.chunk_validation import validate_chunk_hierarchy
from phosprocess.ingestion.technical_chunker import (
    TechnicalDocumentChunker,
)
from phosprocess.knowledge_base.catalog import load_document_catalog


def make_chunker() -> TechnicalDocumentChunker:
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_cached_model_source("BAAI/bge-m3"),
        use_fast=True,
    )
    return TechnicalDocumentChunker(tokenizer=tokenizer)


def test_heading_fields_keep_section_for_shallow_hierarchies() -> None:
    assert TechnicalDocumentChunker._heading_fields(()) == (
        None,
        None,
        None,
    )
    assert TechnicalDocumentChunker._heading_fields(("Evaporation",)) == (
        None,
        "Evaporation",
        None,
    )
    assert TechnicalDocumentChunker._heading_fields(
        ("Heat Transfer", "Forced Convection")
    ) == (
        "Heat Transfer",
        "Forced Convection",
        None,
    )
    assert TechnicalDocumentChunker._heading_fields(
        ("Heat Transfer", "Convection", "Internal Flow")
    ) == (
        "Heat Transfer",
        "Convection",
        "Internal Flow",
    )


def fallback_payload(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_name": "pymupdf_fallback_v1",
                "pages": [
                    {
                        "page_number": 10,
                        "text": (
                            "HEAT EXCHANGER\n"
                            "A heat exchanger transfers energy between fluids. "
                            "It is defined as industrial equipment. "
                            + "Convection and conduction determine performance. " * 20
                        ),
                        "markdown": "",
                    },
                    {
                        "page_number": 11,
                        "text": (
                            "Step 1. Start the circulation pump. "
                            "Step 2. Admit heating steam. "
                            "Step 3. Verify the outlet temperature. "
                        )
                        * 20,
                        "markdown": "",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_fallback_chunking_builds_valid_parent_and_neighbor_links(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "document.json"
    fallback_payload(document_path)
    entry = load_document_catalog().documents[2]
    chunker = make_chunker()

    result = chunker.chunk(document_path=document_path, entry=entry)
    summary = validate_chunk_hierarchy(
        list(result.children),
        list(result.parents),
        maximum_child_tokens=560,
        maximum_parent_tokens=1700,
    )

    assert summary.child_count >= 2
    assert result.children[0].previous_chunk_id is None
    assert result.children[-1].next_chunk_id is None
    assert result.children[0].next_chunk_id == result.children[1].chunk_id
    assert result.children[1].previous_chunk_id == result.children[0].chunk_id
    assert all(child.parent_id != "pending" for child in result.children)
    assert all(child.embedding_text != child.display_text for child in result.children)
    assert all(child.bm25_text != child.embedding_text for child in result.children)


def test_chunk_ids_are_stable_and_procedures_are_classified(tmp_path: Path) -> None:
    document_path = tmp_path / "document.json"
    fallback_payload(document_path)
    entry = load_document_catalog().documents[2]
    chunker = make_chunker()

    first = chunker.chunk(document_path=document_path, entry=entry)
    second = chunker.chunk(document_path=document_path, entry=entry)

    assert [child.chunk_id for child in first.children] == [
        child.chunk_id for child in second.children
    ]
    assert TechnicalChunkType.PROCEDURE in {
        child.chunk_type for child in first.children
    }


def test_jsonl_round_trip_preserves_children_and_parents(tmp_path: Path) -> None:
    document_path = tmp_path / "document.json"
    fallback_payload(document_path)
    entry = load_document_catalog().documents[2]
    result = make_chunker().chunk(document_path=document_path, entry=entry)
    child_path = tmp_path / "children.jsonl"
    parent_path = tmp_path / "parents.jsonl"

    write_jsonl(child_path, list(result.children))
    write_jsonl(parent_path, list(result.parents))

    assert read_child_chunks(child_path) == list(result.children)
    assert read_parent_chunks(parent_path) == list(result.parents)


def test_invalid_neighbor_link_is_rejected(tmp_path: Path) -> None:
    document_path = tmp_path / "document.json"
    fallback_payload(document_path)
    entry = load_document_catalog().documents[2]
    result = make_chunker().chunk(document_path=document_path, entry=entry)
    children = list(result.children)
    children[0] = children[0].model_copy(
        update={"next_chunk_id": "invented"}
    )

    with pytest.raises(ValueError, match="Lien next invalide"):
        validate_chunk_hierarchy(
            children,
            list(result.parents),
            maximum_child_tokens=560,
            maximum_parent_tokens=1700,
        )


@pytest.mark.parametrize(
    ("labels", "text", "expected"),
    [
        (
            (DocItemLabel.TABLE.value,),
            "Temperature | Vapor pressure",
            TechnicalChunkType.TABLE,
        ),
        (
            (DocItemLabel.FORMULA.value,),
            "q = U A ΔT, where U is the overall coefficient.",
            TechnicalChunkType.EQUATION,
        ),
        (
            (DocItemLabel.CAPTION.value,),
            "Figure 2. Forced-circulation evaporator.",
            TechnicalChunkType.FIGURE_CAPTION,
        ),
    ],
)
def test_structured_labels_preserve_technical_item_types(
    labels: tuple[str, ...],
    text: str,
    expected: TechnicalChunkType,
) -> None:
    assert (
        TechnicalDocumentChunker._classify(text, ("Heat transfer",), labels)
        is expected
    )


def test_docling_chunker_repeats_table_headers() -> None:
    assert make_chunker().hybrid_chunker.repeat_table_header is True
