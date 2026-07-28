"""Validated expanded evidence supplied as one indivisible Source N."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceBundle(BaseModel):
    """One anchor and its same-document contextual expansion."""

    model_config = ConfigDict(extra="forbid")

    source_number: int = Field(gt=0, le=5)
    document_id: str
    document_title: str
    filename: str
    chapter: str | None = None
    section: str | None = None
    page_start: int = Field(gt=0)
    page_end: int = Field(gt=0)
    anchor_chunk_id: str
    expanded_chunk_ids: tuple[str, ...] = Field(min_length=1)
    display_text: str = Field(min_length=1)
    token_count: int = Field(gt=0)
    anchor_token_count: int | None = Field(default=None, gt=0)
    context_token_count: int = Field(default=0, ge=0)
    anchor_score: float
    selection_provenance: str
    parent_included: bool = False
    context_truncated: bool = False

    @model_validator(mode="after")
    def validate_bundle(self) -> EvidenceBundle:
        if self.page_end < self.page_start:
            raise ValueError("Pages du bundle invalides.")

        if self.anchor_chunk_id not in self.expanded_chunk_ids:
            raise ValueError("Le bundle doit inclure son anchor.")

        if len(self.expanded_chunk_ids) != len(set(self.expanded_chunk_ids)):
            raise ValueError("Le bundle contient un chunk dupliqué.")

        anchor_tokens = self.anchor_token_count or self.token_count

        if self.context_token_count > self.token_count:
            raise ValueError("Le contexte ajouté dépasse le budget du bundle.")

        if anchor_tokens + self.context_token_count < self.token_count:
            raise ValueError("La télémétrie de tokens du bundle est incohérente.")

        return self
