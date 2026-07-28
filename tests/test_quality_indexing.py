"""Quality-index records and manifest compatibility tests."""

from __future__ import annotations

from phosprocess.ingestion.chunk_serialization import (
    TechnicalChildChunk,
    TechnicalChunkType,
)
from phosprocess.knowledge_base.quality_indexing import child_to_runtime_record


def test_quality_child_maps_to_runtime_without_losing_metadata() -> None:
    text = "Heat transfer in a forced-circulation evaporator."
    child = TechnicalChildChunk(
        chunk_id="child",
        parent_id="parent",
        document_id="book",
        document_title="Technical Book",
        source_file="book.pdf",
        domains=("heat_transfer",),
        chapter="Chapter 1",
        section="Evaporation",
        chunk_type=TechnicalChunkType.EQUIPMENT_DESCRIPTION,
        page_start=10,
        page_end=11,
        text=text,
        display_text=text,
        embedding_text=f"Document: Technical Book\n\n{text}",
        bm25_text=f"Technical Book\nEvaporation\n{text}",
        token_count=20,
        sha256="a" * 64,
    )

    record = child_to_runtime_record(
        child,
        chunk_index=0,
        document_sha256="b" * 64,
    )

    assert record["text"] == child.display_text
    assert record["embedding_text"] == child.embedding_text
    assert record["bm25_text"] == child.bm25_text
    assert record["parent_id"] == "parent"
    assert record["source_pages"] == [10, 11]
    assert record["chunk_type"] == "equipment_description"
