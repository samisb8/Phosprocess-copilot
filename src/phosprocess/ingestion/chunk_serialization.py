"""Schemas and JSONL serialization for technical parent–child chunks."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TechnicalChunkType(StrEnum):
    """Retrieval-aware technical content classes."""

    DEFINITION = "definition"
    NARRATIVE = "narrative"
    PROCESS_DESCRIPTION = "process_description"
    PROCEDURE = "procedure"
    EQUIPMENT_DESCRIPTION = "equipment_description"
    EQUATION = "equation"
    EQUATION_EXPLANATION = "equation_explanation"
    TABLE = "table"
    FIGURE_CAPTION = "figure_caption"
    WORKED_EXAMPLE = "worked_example"
    TROUBLESHOOTING = "troubleshooting"
    OPERATING_PROBLEM = "operating_problem"
    BALANCE = "balance"
    SIMULATION_RESULTS = "simulation_results"
    ABBREVIATIONS = "abbreviations"
    SAFETY = "safety"
    CONTROL_STRATEGY = "control_strategy"
    EXERCISE = "exercise"
    BIBLIOGRAPHY = "bibliography"
    TABLE_OF_CONTENTS = "table_of_contents"
    INDEX = "index"


NON_PRIMARY_CHUNK_TYPES = frozenset(
    {
        TechnicalChunkType.BIBLIOGRAPHY,
        TechnicalChunkType.TABLE_OF_CONTENTS,
        TechnicalChunkType.INDEX,
        TechnicalChunkType.ABBREVIATIONS,
    }
)


class TechnicalChildChunk(BaseModel):
    """Child passage used for retrieval and reranking."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    parent_id: str
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    document_id: str
    document_title: str
    source_file: str
    domains: tuple[str, ...] = Field(min_length=1)
    chapter: str | None = None
    section: str | None = None
    subsection: str | None = None
    hierarchy_path: str = ""
    section_id: str = ""
    chunk_type: TechnicalChunkType
    page_start: int = Field(gt=0)
    page_end: int = Field(gt=0)
    text: str = Field(min_length=1)
    display_text: str = Field(min_length=1)
    embedding_text: str = Field(min_length=1)
    bm25_text: str = Field(min_length=1)
    token_count: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    active: bool = True
    retrieval_weight: float = Field(default=1.0, gt=0.0, le=1.0)
    source_item_labels: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_child(self) -> TechnicalChildChunk:
        if self.page_end < self.page_start:
            raise ValueError("page_end doit être supérieur ou égal à page_start.")

        if self.text != self.display_text:
            raise ValueError("text et display_text doivent être identiques.")

        if self.embedding_text == self.display_text:
            raise ValueError("embedding_text doit contextualiser display_text.")

        return self


class TechnicalParentChunk(BaseModel):
    """Larger section context used only after anchor selection."""

    model_config = ConfigDict(extra="forbid")

    parent_id: str
    document_id: str
    document_title: str
    source_file: str
    chapter: str | None = None
    section: str | None = None
    subsection: str | None = None
    hierarchy_path: str = ""
    section_id: str = ""
    page_start: int = Field(gt=0)
    page_end: int = Field(gt=0)
    child_chunk_ids: tuple[str, ...] = Field(min_length=1)
    display_text: str = Field(min_length=1)
    token_count: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    active: bool = True

    @model_validator(mode="after")
    def validate_parent(self) -> TechnicalParentChunk:
        if self.page_end < self.page_start:
            raise ValueError("Plage de pages parent invalide.")

        if len(self.child_chunk_ids) != len(set(self.child_chunk_ids)):
            raise ValueError("Un parent contient un child_chunk_id dupliqué.")

        return self


class TechnicalSection(BaseModel):
    """Searchable chapter/section/subsection unit for stage-one retrieval."""

    model_config = ConfigDict(extra="forbid")

    section_id: str
    document_id: str
    document_title: str
    source_file: str
    domains: tuple[str, ...] = Field(min_length=1)
    chapter: str | None = None
    section: str | None = None
    subsection: str | None = None
    hierarchy_path: str = Field(min_length=1)
    page_start: int = Field(gt=0)
    page_end: int = Field(gt=0)
    child_chunk_ids: tuple[str, ...] = Field(min_length=1)
    chunk_types: tuple[TechnicalChunkType, ...] = Field(min_length=1)
    display_text: str = Field(min_length=1)
    embedding_text: str = Field(min_length=1)
    bm25_text: str = Field(min_length=1)
    token_count: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    active: bool = True

    @model_validator(mode="after")
    def validate_section(self) -> TechnicalSection:
        if self.page_end < self.page_start:
            raise ValueError("Plage de pages de section invalide.")
        if len(self.child_chunk_ids) != len(set(self.child_chunk_ids)):
            raise ValueError("Une section contient un chunk duplique.")
        return self


def write_jsonl(
    path: Path,
    records: list[BaseModel],
) -> None:
    """Write validated models as deterministic UTF-8 JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            json.dumps(
                record.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
            )
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )


def read_child_chunks(path: Path) -> list[TechnicalChildChunk]:
    """Read validated child JSONL."""

    return [
        TechnicalChildChunk.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_parent_chunks(path: Path) -> list[TechnicalParentChunk]:
    """Read validated parent JSONL."""

    return [
        TechnicalParentChunk.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_sections(path: Path) -> list[TechnicalSection]:
    """Read validated hierarchical section JSONL."""

    return [
        TechnicalSection.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
