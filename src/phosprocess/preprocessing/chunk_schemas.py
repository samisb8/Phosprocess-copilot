"""Schémas validés des chunks documentaires."""

from pydantic import BaseModel, ConfigDict, Field


class DocumentChunk(BaseModel):
    """Passage structuré prêt pour les embeddings et le retrieval."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    document_id: str
    source_file: str

    chunk_index: int = Field(ge=0)

    heading_path: list[str] = Field(default_factory=list)
    source_pages: list[int] = Field(min_length=1)

    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)

    content_types: list[str] = Field(default_factory=list)

    text: str = Field(min_length=1)
    embedding_text: str = Field(min_length=1)

    body_token_count: int = Field(ge=1)
    token_count: int = Field(ge=1)

    source_chunk_ids: list[str] = Field(default_factory=list)
    postprocessing_actions: list[str] = Field(default_factory=list)

    # Optional structure-aware fields used by kb_quality_* indexes. Keeping
    # them optional preserves compatibility with the legacy production index.
    parent_id: str | None = None
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    document_title: str | None = None
    domains: list[str] = Field(default_factory=list)
    chapter: str | None = None
    section: str | None = None
    subsection: str | None = None
    hierarchy_path: str | None = None
    section_id: str | None = None
    chunk_type: str | None = None
    display_text: str | None = None
    bm25_text: str | None = None
    active: bool = True
    retrieval_weight: float = Field(default=1.0, gt=0.0)
    source_item_labels: list[str] = Field(default_factory=list)
