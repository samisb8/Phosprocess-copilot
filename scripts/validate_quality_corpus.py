"""Validate structured extraction and chunk samples without loading retrieval."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime

from phosprocess.ingestion.chunk_serialization import (
    read_child_chunks,
    read_parent_chunks,
    read_sections,
)
from phosprocess.ingestion.chunk_validation import validate_chunk_hierarchy
from phosprocess.knowledge_base.catalog import load_document_catalog
from phosprocess.knowledge_base.runtime import (
    DEFAULT_KNOWLEDGE_BASE_ROOT,
    PROJECT_ROOT,
    load_active_knowledge_base,
)


def main() -> int:
    catalog = load_document_catalog()
    active = load_active_knowledge_base()
    active_index = json.loads(active.pointer_path.read_text(encoding="utf-8"))
    reports = []
    total_children = 0
    total_parents = 0
    total_sections = 0

    for entry in catalog.documents:
        parsed = (
            DEFAULT_KNOWLEDGE_BASE_ROOT
            / "parsed"
            / entry.document_id
            / entry.sha256
        )
        corpus = (
            DEFAULT_KNOWLEDGE_BASE_ROOT
            / "corpus"
            / entry.document_id
            / entry.sha256
        )
        extraction = json.loads(
            (parsed / "extraction_report.json").read_text(encoding="utf-8")
        )
        children = read_child_chunks(corpus / "children.jsonl")
        parents = read_parent_chunks(corpus / "parents.jsonl")
        sections = read_sections(corpus / "sections.jsonl")
        hierarchy = validate_chunk_hierarchy(
            children,
            parents,
            maximum_child_tokens=560,
            maximum_parent_tokens=1700,
            sections=sections,
        )
        type_counts = Counter(child.chunk_type.value for child in children)
        total_children += hierarchy.child_count
        total_parents += hierarchy.parent_count
        total_sections += hierarchy.section_count
        reports.append(
            {
                "document_id": entry.document_id,
                "filename": entry.source_filename,
                "parser": extraction["parser"],
                "fallback_used": extraction["fallback_used"],
                "activation_allowed": extraction["activation_allowed"],
                "table_items_extracted": extraction["tables"],
                "formula_items_extracted": extraction["formulas"],
                "section_items_extracted": extraction["sections_detected"],
                "child_count": hierarchy.child_count,
                "parent_count": hierarchy.parent_count,
                "section_count": hierarchy.section_count,
                "chunk_type_counts": dict(sorted(type_counts.items())),
                "maximum_child_tokens": hierarchy.maximum_child_tokens,
                "maximum_parent_tokens": hierarchy.maximum_parent_tokens,
                "children_with_section": sum(
                    bool(child.section_id and child.hierarchy_path)
                    for child in children
                ),
            }
        )

    if len(reports) != 8:
        raise ValueError("Huit rapports structurés sont requis.")

    if any(not report["activation_allowed"] for report in reports):
        raise ValueError("Un document ne passe pas le quality gate.")

    if active_index["document_count"] != len(reports):
        raise ValueError(
            "Le catalogue structuré ne correspond pas à l'index actif."
        )

    if active_index["chunk_count"] != total_children:
        raise ValueError(
            "Les child chunks structurés ne correspondent pas à l'index actif."
        )

    index_manifest = json.loads(
        (active.version_directory / "index_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if index_manifest.get("section_count") != total_sections:
        raise ValueError(
            "Les sections structurées ne correspondent pas à l'index actif."
        )

    output = (
        PROJECT_ROOT
        / "data"
        / "observability"
        / "quality"
        / "structured_corpus_validation.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "pipeline_version": active_index["pipeline_version"],
                "active_index_version": active_index["version"],
                "document_count": len(reports),
                "child_count": total_children,
                "parent_count": total_parents,
                "section_count": total_sections,
                "historical_test_used": False,
                "documents": reports,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Documents validés : {len(reports)}")
    print(f"Child chunks : {total_children}")
    print(f"Parent chunks : {total_parents}")
    print(f"Sections : {total_sections}")
    print(f"Rapport : {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
