"""Validated schemas for the production document catalogue."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from phosprocess.knowledge_base.domains import KnowledgeDomain

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DocumentType(StrEnum):
    """Supported technical source categories."""

    PROCESS_REFERENCE = "process_reference"
    PLANT_REPORT = "plant_report"
    THERMODYNAMICS_TEXTBOOK = "thermodynamics_textbook"
    HEAT_TRANSFER_TEXTBOOK = "heat_transfer_textbook"
    HANDBOOK = "handbook"
    CRYSTALLIZATION_TEXTBOOK = "crystallization_textbook"
    PROCESS_CONTROL_TEXTBOOK = "process_control_textbook"
    TRANSPORT_TEXTBOOK = "transport_textbook"


class ExtractionStatus(StrEnum):
    """Document-level extraction readiness."""

    NATIVE_TEXT = "native_text"
    EXTRACTED_SUCCESSFULLY = "extracted_successfully"
    DOCLING_REQUIRED = "docling_required"
    OCR_REQUIRED = "ocr_required"
    CORRUPTED_TEXT = "corrupted_text"
    INCOMPLETE_SOURCE = "incomplete_source"
    INVALID = "invalid"


class DocumentCatalogEntry(BaseModel):
    """One explicit source description independent of its filename tokens."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    canonical_filename: str = Field(pattern=r"(?i)^.+\.pdf$")
    source_filename: str = Field(pattern=r"(?i)^.+\.pdf$")
    display_title: str = Field(min_length=1)
    authors: tuple[str, ...] = Field(min_length=1)
    edition: str = Field(min_length=1)
    language: str = Field(pattern=r"^(?:en|fr|ar|multilingual)$")
    document_type: DocumentType
    domains: tuple[KnowledgeDomain, ...] = Field(min_length=1)
    subdomains: tuple[str, ...] = Field(min_length=1)
    priority: int = Field(ge=0, le=100)
    plant_specific: bool
    sha256: str
    page_count: int = Field(gt=0)
    ingestion_version: str = Field(min_length=1)
    active: bool
    extraction_status: ExtractionStatus

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = value.casefold()

        if not _SHA256.fullmatch(normalized):
            raise ValueError("sha256 doit contenir exactement 64 hexadécimaux.")

        return normalized

    @field_validator("authors", "subdomains")
    @classmethod
    def validate_unique_non_empty_values(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)

        if any(not value for value in normalized):
            raise ValueError("Les valeurs du catalogue ne peuvent pas être vides.")

        if len(normalized) != len(set(normalized)):
            raise ValueError("Le catalogue contient une valeur dupliquée.")

        return normalized

    @field_validator("domains")
    @classmethod
    def validate_unique_domains(
        cls,
        values: tuple[KnowledgeDomain, ...],
    ) -> tuple[KnowledgeDomain, ...]:
        if len(values) != len(set(values)):
            raise ValueError("Les domaines du document sont dupliqués.")

        return values


class KnowledgeBaseCatalog(BaseModel):
    """Complete catalogue used by production ingestion and routing."""

    model_config = ConfigDict(extra="forbid")

    catalog_version: str = Field(min_length=1)
    documents: tuple[DocumentCatalogEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_documents(self) -> KnowledgeBaseCatalog:
        if len(self.documents) != 8:
            raise ValueError("Le catalogue de production doit contenir huit documents.")

        for attribute in (
            "document_id",
            "canonical_filename",
            "source_filename",
            "sha256",
        ):
            values = [getattr(document, attribute).casefold() for document in self.documents]

            if len(values) != len(set(values)):
                raise ValueError(f"{attribute} doit être unique dans le catalogue.")

        return self
