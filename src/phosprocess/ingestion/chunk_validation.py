"""Cross-record validation for technical chunk, parent and section layers."""

from __future__ import annotations

from dataclasses import dataclass

from phosprocess.ingestion.chunk_serialization import (
    TechnicalChildChunk,
    TechnicalParentChunk,
    TechnicalSection,
)


@dataclass(frozen=True, slots=True)
class ChunkValidationSummary:
    """Successful hierarchy validation counts."""

    child_count: int
    parent_count: int
    maximum_child_tokens: int
    maximum_parent_tokens: int
    section_count: int = 0


def _validate_sections(
    children: list[TechnicalChildChunk],
    sections: list[TechnicalSection],
) -> None:
    if not sections:
        raise ValueError("La hiérarchie de sections ne peut pas être vide.")

    child_by_id = {child.chunk_id: child for child in children}
    section_by_id = {section.section_id: section for section in sections}
    if len(section_by_id) != len(sections):
        raise ValueError("section_id dupliqué.")

    assigned_children: set[str] = set()
    for section in sections:
        section_children: list[TechnicalChildChunk] = []
        for chunk_id in section.child_chunk_ids:
            child = child_by_id.get(chunk_id)
            if child is None:
                raise ValueError(f"Chunk absent de la section {section.section_id}: {chunk_id}.")
            if child.section_id != section.section_id:
                raise ValueError(f"section_id incohérent pour {child.chunk_id}.")
            if child.document_id != section.document_id:
                raise ValueError(f"Document incohérent pour la section {section.section_id}.")
            if child.hierarchy_path != section.hierarchy_path:
                raise ValueError(f"Chemin hiérarchique incohérent pour {child.chunk_id}.")
            section_children.append(child)
            assigned_children.add(chunk_id)

        expected_types = {child.chunk_type for child in section_children}
        if set(section.chunk_types) != expected_types:
            raise ValueError(f"Types de chunks incohérents pour {section.section_id}.")
        if section.page_start != min(child.page_start for child in section_children):
            raise ValueError(f"page_start incohérente pour {section.section_id}.")
        if section.page_end != max(child.page_end for child in section_children):
            raise ValueError(f"page_end incohérente pour {section.section_id}.")
        if not section.embedding_text.strip() or not section.bm25_text.strip():
            raise ValueError(f"Représentation de section vide : {section.section_id}.")

    if assigned_children != set(child_by_id):
        raise ValueError("Chaque chunk doit appartenir exactement à une section.")


def validate_chunk_hierarchy(
    children: list[TechnicalChildChunk],
    parents: list[TechnicalParentChunk],
    *,
    maximum_child_tokens: int,
    maximum_parent_tokens: int,
    sections: list[TechnicalSection] | None = None,
) -> ChunkValidationSummary:
    """Reject broken IDs, links, pages, representations or budgets."""

    if not children or not parents:
        raise ValueError("La hiérarchie doit contenir parents et enfants.")

    child_by_id = {child.chunk_id: child for child in children}
    parent_by_id = {parent.parent_id: parent for parent in parents}

    if len(child_by_id) != len(children):
        raise ValueError("chunk_id enfant dupliqué.")

    if len(parent_by_id) != len(parents):
        raise ValueError("parent_id dupliqué.")

    for index, child in enumerate(children):
        if child.parent_id not in parent_by_id:
            raise ValueError(f"Parent absent pour {child.chunk_id}.")

        previous = children[index - 1] if index else None
        following = children[index + 1] if index + 1 < len(children) else None
        scope = (child.document_id, child.chapter, child.section)
        expected_previous = (
            previous.chunk_id
            if previous is not None
            and (previous.document_id, previous.chapter, previous.section) == scope
            else None
        )
        expected_next = (
            following.chunk_id
            if following is not None
            and (following.document_id, following.chapter, following.section) == scope
            else None
        )

        if child.previous_chunk_id != expected_previous:
            raise ValueError(f"Lien previous invalide pour {child.chunk_id}.")

        if child.next_chunk_id != expected_next:
            raise ValueError(f"Lien next invalide pour {child.chunk_id}.")

        if child.token_count > maximum_child_tokens:
            raise ValueError(f"Chunk trop long : {child.chunk_id}.")

        if not child.display_text.strip() or not child.bm25_text.strip():
            raise ValueError(f"Représentation vide : {child.chunk_id}.")

    parent_children: set[str] = set()

    for parent in parents:
        if parent.token_count > maximum_parent_tokens:
            raise ValueError(f"Parent trop long : {parent.parent_id}.")

        for child_id in parent.child_chunk_ids:
            child = child_by_id.get(child_id)

            if child is None or child.parent_id != parent.parent_id:
                raise ValueError(f"Relation parent/enfant invalide : {child_id}.")

            parent_children.add(child_id)

    if parent_children != set(child_by_id):
        raise ValueError("Tous les enfants ne sont pas reliés une fois.")

    if sections is not None:
        _validate_sections(children, sections)

    return ChunkValidationSummary(
        child_count=len(children),
        parent_count=len(parents),
        maximum_child_tokens=max(child.token_count for child in children),
        maximum_parent_tokens=max(parent.token_count for parent in parents),
        section_count=len(sections or ()),
    )
