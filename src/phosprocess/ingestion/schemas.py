"""Schémas validés pour les pages extraites des documents."""

from typing import Literal

from pydantic import BaseModel, Field

ParserName = Literal["pymupdf", "pymupdf4llm", "docling", "ocr"]


class PageContent(BaseModel):
    """Contenu exploitable extrait d'une page."""

    plain_text: str = ""
    markdown: str = ""
    tables: list[str] = Field(default_factory=list)
    figures: list[str] = Field(default_factory=list)


class PageProvenance(BaseModel):
    """Informations permettant de retrouver la source."""

    source_file: str
    document_id: str
    page_number: int = Field(ge=1)
    parser: ParserName
    ocr_used: bool = False


class PageQuality(BaseModel):
    """Mesures utilisées pour évaluer l'extraction."""

    character_count: int = Field(ge=0)
    word_count: int = Field(ge=0)
    is_empty: bool
    needs_review: bool = False
    warnings: list[str] = Field(default_factory=list)


class ParsedPage(BaseModel):
    """Format commun produit par tous les parseurs."""

    content: PageContent
    provenance: PageProvenance
    quality: PageQuality