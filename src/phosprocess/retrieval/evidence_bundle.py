"""Validated parent-grouped documentary evidence supplied as ``Source N``."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceContextScope(StrEnum):
    """Exact structural scope represented by one evidence bundle."""

    FULL_PARENT = "full_parent"
    PARTIAL_PARENT = "partial_parent"
    ANCHOR_WITH_NEIGHBORS = "anchor_with_neighbors"
    ANCHOR_ONLY = "anchor_only"


def render_evidence_block(
    *,
    source_number: int,
    document_title: str,
    filename: str,
    chapter: str | None,
    section: str | None,
    page_start: int,
    page_end: int,
    display_text: str,
) -> str:
    """Serialize exactly the documentary block later supplied to Qwen."""

    header = "\n".join(
        [
            f"[Source {source_number}]",
            f"Document: {document_title}",
            f"Filename: {filename}",
            f"Chapter: {chapter or 'Not specified'}",
            f"Section: {section or 'Not specified'}",
            f"Pages: {page_start}-{page_end}",
            "Evidence:",
        ]
    )
    cleaned = display_text.strip()
    return f"{header}\n{cleaned}" if cleaned else header


class EvidenceBundle(BaseModel):
    """One packed parent candidate with complete child provenance."""

    model_config = ConfigDict(extra="forbid")

    source_number: int = Field(gt=0)
    document_id: str
    document_title: str
    filename: str
    chapter: str | None = None
    section: str | None = None
    subsection: str | None = None
    hierarchy_path: str = ""
    page_start: int = Field(gt=0)
    page_end: int = Field(gt=0)
    parent_id: str
    anchor_chunk_ids: tuple[str, ...] = Field(min_length=1)
    supporting_chunk_ids: tuple[str, ...] = Field(min_length=1)
    display_text: str = Field(min_length=1)
    token_count: int = Field(gt=0)
    documentary_token_count: int = Field(gt=0)
    metadata_token_count: int = Field(ge=0)
    anchor_token_count: int = Field(gt=0)
    context_token_count: int = Field(default=0, ge=0)
    best_anchor_score: float
    context_scope: EvidenceContextScope
    selection_provenance: str
    context_truncated: bool = False

    @model_validator(mode="after")
    def validate_bundle(self) -> EvidenceBundle:
        if self.page_end < self.page_start:
            raise ValueError("Pages du bundle invalides.")
        if len(self.anchor_chunk_ids) != len(set(self.anchor_chunk_ids)):
            raise ValueError("Le bundle contient une ancre dupliquee.")
        if len(self.supporting_chunk_ids) != len(set(self.supporting_chunk_ids)):
            raise ValueError("Le bundle contient un chunk documentaire duplique.")
        if not set(self.anchor_chunk_ids).issubset(self.supporting_chunk_ids):
            raise ValueError("Chaque ancre doit appartenir aux chunks documentaires.")
        if self.context_token_count > self.documentary_token_count:
            raise ValueError("Le contexte ajoute depasse le texte documentaire.")
        if self.context_scope is EvidenceContextScope.ANCHOR_ONLY and (
            set(self.supporting_chunk_ids) != set(self.anchor_chunk_ids)
        ):
            raise ValueError("Un bundle anchor_only ne peut contenir de contexte ajoute.")
        return self

    @property
    def anchor_chunk_id(self) -> str:
        """Primary anchor retained for compatibility with response telemetry."""

        return self.anchor_chunk_ids[0]

    @property
    def expanded_chunk_ids(self) -> tuple[str, ...]:
        """Compatibility name for all faithful supporting chunks."""

        return self.supporting_chunk_ids

    @property
    def anchor_score(self) -> float:
        """Compatibility name for the strongest grouped anchor score."""

        return self.best_anchor_score

    @property
    def parent_included(self) -> bool:
        """Whether documentary content beyond the anchors came from the parent."""

        return self.context_scope in {
            EvidenceContextScope.FULL_PARENT,
            EvidenceContextScope.PARTIAL_PARENT,
        }

    def render_prompt_block(self) -> str:
        """Return the exact source block used by the generation prompt."""

        return render_evidence_block(
            source_number=self.source_number,
            document_title=self.document_title,
            filename=self.filename,
            chapter=self.chapter,
            section=self.section,
            page_start=self.page_start,
            page_end=self.page_end,
            display_text=self.display_text,
        )
