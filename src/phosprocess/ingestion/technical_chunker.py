"""Docling HybridChunker adapter producing technical parent–child evidence."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docling_core.transforms.chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from docling_core.types.doc import DocItemLabel, DoclingDocument
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from phosprocess.embeddings.embedder import resolve_cached_model_source
from phosprocess.ingestion.chunk_serialization import (
    NON_PRIMARY_CHUNK_TYPES,
    TechnicalChildChunk,
    TechnicalChunkType,
    TechnicalParentChunk,
    TechnicalSection,
)
from phosprocess.knowledge_base.schemas import DocumentCatalogEntry

_EXERCISE = re.compile(r"\b(?:exercise|problem|question for discussion)\b", re.I)
_BIBLIOGRAPHY = re.compile(r"\b(?:bibliography|references)\b", re.I)
_CONTENTS = re.compile(r"\b(?:table of contents|contents)\b", re.I)
_INDEX = re.compile(r"^\s*(?:subject |author )?index\b", re.I)
_DEFINITION = re.compile(r"\b(?:is defined as|refers to|means that)\b", re.I)
_PROCEDURE = re.compile(
    r"(?m)^\s*(?:step\s+\d+|\d+[.)]\s+|first,|second,|finally,)",
    re.I,
)
_PROCESS = re.compile(
    r"\b(?:process flow|flows? through|is fed|is recycled|recirculat|"
    r"process description|enters? the|leaves? the)\b",
    re.I,
)
_EQUIPMENT = re.compile(
    r"\b(?:pump|evaporator|reactor|filter|heat exchanger|vessel|"
    r"compressor|column)\b",
    re.I,
)
_TROUBLESHOOTING = re.compile(
    r"\b(?:troubleshoot|failure|fault|symptom|remedial|corrective action)\b",
    re.I,
)
_OPERATING_PROBLEM = re.compile(
    r"\b(?:fouling|scaling|salting|plugging|corrosion|erosion|entrainment|"
    r"foaming|deposit|encrassement|entartrage|d[eé]p[oô]t|colmatage|"
    r"reduces?|decreases?|increases?|affects?|causes?|results? in|"
    r"diminue|augmente|provoque|entra[iî]ne|affecte)\b",
    re.I,
)
_BALANCE = re.compile(
    r"\b(?:mass|material|component|energy|heat|enthalpy|p2o5|mati[eè]re|"
    r"massique|[eé]nerg[eé]tique|thermique)\s+balance\b|"
    r"\bbilan\s+(?:global|mati[eè]re|massique|p2o5|[eé]nerg[eé]tique|thermique)",
    re.I,
)
_SIMULATION = re.compile(
    r"\b(?:simulation|matlab|simulink|generated curves?|courbes? g[eé]n[eé]r[eé]es?|"
    r"simulation results?|r[eé]sultats? de simulation)\b",
    re.I,
)
_ABBREVIATIONS = re.compile(
    r"\b(?:list of abbreviations|abbreviations|liste des abr[eé]viations)\b",
    re.I,
)
_SAFETY = re.compile(r"\b(?:hazard|safety|explosion|toxic|corrosive|ppe)\b", re.I)
_CONTROL = re.compile(
    r"\b(?:pid|feedback|feedforward|controller|control loop|mpc|"
    r"model predictive control)\b",
    re.I,
)
_EXAMPLE = re.compile(r"\b(?:worked example|example\s+\d+)\b", re.I)
_FORMULA_PLACEHOLDER = re.compile(r"<!--\s*formula-not-decoded\s*-->", re.I)
_LEGAL = re.compile(
    r"\b(?:all rights reserved|isbn|copyright|library of congress)\b",
    re.I,
)


@dataclass(frozen=True, slots=True)
class TechnicalChunkingConfig:
    """Initial, explicitly non-optimal parent–child parameters."""

    tokenizer_name: str = "BAAI/bge-m3"
    child_target_tokens: int = 420
    child_minimum_tokens: int = 160
    child_maximum_tokens: int = 560
    overlap_tokens: int = 70
    parent_target_tokens: int = 1250
    parent_maximum_tokens: int = 1700

    def __post_init__(self) -> None:
        if not (
            0
            < self.child_minimum_tokens
            <= self.child_target_tokens
            <= self.child_maximum_tokens
            < self.parent_target_tokens
            <= self.parent_maximum_tokens
        ):
            raise ValueError("Budgets parent–child incohérents.")

        if not 0 <= self.overlap_tokens < self.child_minimum_tokens:
            raise ValueError("overlap_tokens est incohérent.")


@dataclass(frozen=True, slots=True)
class TechnicalChunkingResult:
    """One complete document chunk hierarchy."""

    children: tuple[TechnicalChildChunk, ...]
    parents: tuple[TechnicalParentChunk, ...]
    sections: tuple[TechnicalSection, ...] = ()
    excluded_chunk_count: int = 0


@dataclass(frozen=True, slots=True)
class _Draft:
    text: str
    headings: tuple[str, ...]
    pages: tuple[int, ...]
    labels: tuple[str, ...]
    chunk_type: TechnicalChunkType


class TechnicalDocumentChunker:
    """Convert Docling structured items into retrieval and context layers."""

    def __init__(
        self,
        config: TechnicalChunkingConfig | None = None,
        *,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        self.config = config or TechnicalChunkingConfig()
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            resolve_cached_model_source(self.config.tokenizer_name),
            use_fast=True,
        )
        self.docling_tokenizer = HuggingFaceTokenizer(
            tokenizer=self.tokenizer,
            max_tokens=self.config.child_maximum_tokens,
        )
        self.hybrid_chunker = HybridChunker(
            tokenizer=self.docling_tokenizer,
            repeat_table_header=True,
            merge_peers=True,
            always_emit_headings=False,
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _tail(self, text: str) -> str:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return self.tokenizer.decode(
            token_ids[-self.config.overlap_tokens :],
            skip_special_tokens=True,
        ).strip()

    @staticmethod
    def _item_metadata(chunk: Any) -> tuple[tuple[int, ...], tuple[str, ...]]:
        pages: set[int] = set()
        labels: set[str] = set()

        for item in chunk.meta.doc_items:
            label = getattr(item, "label", None)
            labels.add(str(label.value) if hasattr(label, "value") else str(label or "unknown"))

            for provenance in getattr(item, "prov", None) or ():
                pages.add(int(provenance.page_no))

        return tuple(sorted(pages)), tuple(sorted(labels))

    @staticmethod
    def _classify(
        text: str,
        headings: tuple[str, ...],
        labels: tuple[str, ...],
    ) -> TechnicalChunkType:
        context = " ".join([*headings, text[:2000]])
        label_set = set(labels)

        if _CONTENTS.search(context):
            return TechnicalChunkType.TABLE_OF_CONTENTS

        if _ABBREVIATIONS.search(context):
            return TechnicalChunkType.ABBREVIATIONS

        if _BIBLIOGRAPHY.search(" ".join(headings)):
            return TechnicalChunkType.BIBLIOGRAPHY

        if _INDEX.search(" ".join(headings)):
            return TechnicalChunkType.INDEX

        if DocItemLabel.TABLE.value in label_set:
            return TechnicalChunkType.TABLE

        if DocItemLabel.FORMULA.value in label_set or _FORMULA_PLACEHOLDER.search(text):
            return TechnicalChunkType.EQUATION

        if DocItemLabel.CAPTION.value in label_set or DocItemLabel.PICTURE.value in label_set:
            return TechnicalChunkType.FIGURE_CAPTION

        if _EXERCISE.search(context):
            return TechnicalChunkType.EXERCISE

        if _EXAMPLE.search(context):
            return TechnicalChunkType.WORKED_EXAMPLE

        if _TROUBLESHOOTING.search(context):
            return TechnicalChunkType.TROUBLESHOOTING

        if _BALANCE.search(context):
            return TechnicalChunkType.BALANCE

        if _SIMULATION.search(context):
            return TechnicalChunkType.SIMULATION_RESULTS

        if _OPERATING_PROBLEM.search(context):
            return TechnicalChunkType.OPERATING_PROBLEM

        if _SAFETY.search(context):
            return TechnicalChunkType.SAFETY

        if _CONTROL.search(context):
            return TechnicalChunkType.CONTROL_STRATEGY

        if _PROCEDURE.search(text):
            return TechnicalChunkType.PROCEDURE

        if _DEFINITION.search(text):
            return TechnicalChunkType.DEFINITION

        if _PROCESS.search(context):
            return TechnicalChunkType.PROCESS_DESCRIPTION

        if _EQUIPMENT.search(context):
            return TechnicalChunkType.EQUIPMENT_DESCRIPTION

        return TechnicalChunkType.NARRATIVE

    @staticmethod
    def _headings(values: list[str] | None) -> tuple[str, ...]:
        return tuple(value.strip() for value in (values or ()) if value.strip())

    def _docling_drafts(self, document: DoclingDocument) -> list[_Draft]:
        drafts: list[_Draft] = []

        for chunk in self.hybrid_chunker.chunk(document):
            text = chunk.text.strip()

            if not text:
                continue

            pages, labels = self._item_metadata(chunk)

            if not pages:
                continue

            headings = self._headings(chunk.meta.headings)
            chunk_type = self._classify(text, headings, labels)
            drafts.append(
                _Draft(
                    text=text,
                    headings=headings,
                    pages=pages,
                    labels=labels,
                    chunk_type=chunk_type,
                )
            )

        return drafts

    def _fallback_drafts(self, payload: dict[str, Any]) -> list[_Draft]:
        drafts: list[_Draft] = []

        for page in payload.get("pages", []):
            text = str(page.get("text", "")).strip()

            if not text:
                continue

            page_number = int(page["page_number"])
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)

            step = self.config.child_target_tokens - self.config.overlap_tokens

            for offset in range(0, len(token_ids), step):
                piece_ids = token_ids[offset : offset + self.config.child_target_tokens]
                piece = self.tokenizer.decode(
                    piece_ids,
                    skip_special_tokens=True,
                ).strip()

                if piece:
                    drafts.append(
                        _Draft(
                            text=piece,
                            headings=(),
                            pages=(page_number,),
                            labels=(DocItemLabel.TEXT.value,),
                            chunk_type=self._classify(piece, (), ("text",)),
                        )
                    )

        return drafts

    def _load_drafts(self, document_path: Path) -> list[_Draft]:
        payload = json.loads(document_path.read_text(encoding="utf-8"))

        if payload.get("schema_name") == "pymupdf_fallback_v1":
            return self._fallback_drafts(payload)

        document = DoclingDocument.model_validate(payload)
        return self._docling_drafts(document)

    @staticmethod
    def _heading_fields(
        headings: tuple[str, ...],
    ) -> tuple[str | None, str | None, str | None]:
        values = [*headings[-3:]]

        if not values:
            return None, None, None

        if len(values) == 1:
            return None, values[0], None

        if len(values) == 2:
            return values[0], values[1], None

        return values[0], values[1], values[2]

    @staticmethod
    def _stable_digest(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    @staticmethod
    def _hierarchy_path(
        *,
        entry: DocumentCatalogEntry,
        chapter: str | None,
        section: str | None,
        subsection: str | None,
    ) -> str:
        values = [
            entry.display_title,
            *(value for value in (chapter, section, subsection) if value),
        ]
        return " > ".join(values)

    def _embedding_text(
        self,
        *,
        entry: DocumentCatalogEntry,
        draft: _Draft,
        previous_text: str | None,
    ) -> str:
        chapter, section, subsection = self._heading_fields(draft.headings)
        hierarchy_path = self._hierarchy_path(
            entry=entry,
            chapter=chapter,
            section=section,
            subsection=subsection,
        )
        lines = [
            f"Document: {entry.display_title}",
            f"Domains: {', '.join(domain.value for domain in entry.domains)}",
            f"Chapter: {chapter or 'Not specified'}",
            f"Section: {section or 'Not specified'}",
            f"Subsection: {subsection or 'Not specified'}",
            f"Hierarchy: {hierarchy_path}",
            f"Content type: {draft.chunk_type.value}",
        ]

        if previous_text:
            lines.extend(["Previous context:", self._tail(previous_text)])

        return "\n".join(lines) + "\n\n" + draft.text

    @staticmethod
    def _bm25_text(
        *,
        entry: DocumentCatalogEntry,
        draft: _Draft,
    ) -> str:
        chapter, section, subsection = TechnicalDocumentChunker._heading_fields(draft.headings)
        hierarchy_path = TechnicalDocumentChunker._hierarchy_path(
            entry=entry,
            chapter=chapter,
            section=section,
            subsection=subsection,
        )
        return "\n".join(
            [
                f"Document: {entry.display_title}",
                f"Chapter: {chapter or ''}",
                f"Section: {section or ''}",
                f"Subsection: {subsection or ''}",
                f"Hierarchy: {hierarchy_path}",
                f"Content type: {draft.chunk_type.value}",
                hierarchy_path,
                hierarchy_path,
                draft.text,
                "Technical terms: " + ", ".join(entry.subdomains),
            ]
        )

    def _draft_children(
        self,
        drafts: list[_Draft],
        entry: DocumentCatalogEntry,
    ) -> tuple[list[TechnicalChildChunk], int]:
        children: list[TechnicalChildChunk] = []
        excluded = 0
        previous_text: str | None = None
        previous_headings: tuple[str, ...] | None = None

        for draft in drafts:
            if draft.chunk_type in NON_PRIMARY_CHUNK_TYPES:
                excluded += 1
                continue

            if _LEGAL.search(draft.text[:500]) and draft.pages[0] <= 5:
                excluded += 1
                continue

            chapter, section, subsection = self._heading_fields(draft.headings)
            hierarchy_path = self._hierarchy_path(
                entry=entry,
                chapter=chapter,
                section=section,
                subsection=subsection,
            )
            if not any((chapter, section, subsection)):
                block_start = ((draft.pages[0] - 1) // 5) * 5 + 1
                block_end = block_start + 4
                hierarchy_path = f"{hierarchy_path} > Unlabelled pages {block_start}-{block_end}"
            section_digest = self._stable_digest(
                entry.document_id,
                hierarchy_path,
            )
            section_id = f"{entry.document_id}_section_{section_digest[:16]}"
            digest = self._stable_digest(
                entry.document_id,
                str(draft.pages[0]),
                str(draft.pages[-1]),
                " > ".join(draft.headings),
                draft.text,
            )
            chunk_id = f"{entry.document_id}_{digest[:16]}"
            weight = 0.65 if draft.chunk_type is TechnicalChunkType.EXERCISE else 1.0
            children.append(
                TechnicalChildChunk(
                    chunk_id=chunk_id,
                    parent_id="pending",
                    document_id=entry.document_id,
                    document_title=entry.display_title,
                    source_file=entry.canonical_filename,
                    domains=tuple(domain.value for domain in entry.domains),
                    chapter=chapter,
                    section=section,
                    subsection=subsection,
                    hierarchy_path=hierarchy_path,
                    section_id=section_id,
                    chunk_type=draft.chunk_type,
                    page_start=draft.pages[0],
                    page_end=draft.pages[-1],
                    text=draft.text,
                    display_text=draft.text,
                    embedding_text=self._embedding_text(
                        entry=entry,
                        draft=draft,
                        previous_text=(
                            previous_text if previous_headings == draft.headings else None
                        ),
                    ),
                    bm25_text=self._bm25_text(entry=entry, draft=draft),
                    token_count=self.count_tokens(draft.text),
                    sha256=self._stable_digest(draft.text),
                    retrieval_weight=weight,
                    source_item_labels=draft.labels,
                )
            )
            previous_text = draft.text
            previous_headings = draft.headings

        return children, excluded

    def _parent_groups(
        self,
        children: list[TechnicalChildChunk],
    ) -> list[list[TechnicalChildChunk]]:
        groups: list[list[TechnicalChildChunk]] = []
        current: list[TechnicalChildChunk] = []
        current_key: tuple[str | None, str | None, str | None] | None = None

        for child in children:
            key = (child.chapter, child.section, child.subsection)
            candidate = [*current, child]
            candidate_tokens = self.count_tokens(
                "\n\n".join(item.display_text for item in candidate)
            )

            if current and (
                key != current_key
                or candidate_tokens > self.config.parent_maximum_tokens
                or self.count_tokens("\n\n".join(item.display_text for item in current))
                >= self.config.parent_target_tokens
            ):
                groups.append(current)
                current = []

            if not current:
                current_key = key

            current.append(child)

        if current:
            groups.append(current)

        return groups

    def _section_representation(
        self,
        children: list[TechnicalChildChunk],
        *,
        maximum_tokens: int = 900,
    ) -> str:
        preferred = {
            TechnicalChunkType.PROCESS_DESCRIPTION,
            TechnicalChunkType.EQUIPMENT_DESCRIPTION,
            TechnicalChunkType.BALANCE,
            TechnicalChunkType.EQUATION,
            TechnicalChunkType.TABLE,
            TechnicalChunkType.TROUBLESHOOTING,
            TechnicalChunkType.OPERATING_PROBLEM,
        }
        ordered: list[TechnicalChildChunk] = []

        for candidate in (
            children[0],
            *(item for item in children if item.chunk_type in preferred),
            children[-1],
        ):
            if candidate.chunk_id not in {item.chunk_id for item in ordered}:
                ordered.append(candidate)

        for candidate in children:
            if candidate.chunk_id not in {item.chunk_id for item in ordered}:
                ordered.append(candidate)

        pieces: list[str] = []
        used = 0

        for child in ordered:
            remaining = maximum_tokens - used
            if remaining <= 0:
                break
            token_ids = self.tokenizer.encode(
                child.display_text,
                add_special_tokens=False,
            )[:remaining]
            piece = self.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
            ).strip()
            if not piece:
                continue
            pieces.append(piece)
            used += len(token_ids)

        return "\n\n".join(pieces)

    def _build_sections(
        self,
        children: list[TechnicalChildChunk],
        entry: DocumentCatalogEntry,
    ) -> list[TechnicalSection]:
        grouped: dict[str, list[TechnicalChildChunk]] = {}
        for child in children:
            grouped.setdefault(child.section_id, []).append(child)

        sections: list[TechnicalSection] = []
        for section_id, group in grouped.items():
            representation = self._section_representation(group)
            hierarchy_path = group[0].hierarchy_path
            chunk_types = tuple(
                sorted(
                    {item.chunk_type for item in group},
                    key=lambda item: item.value,
                )
            )
            type_text = ", ".join(item.value for item in chunk_types)
            embedding_text = "\n".join(
                [
                    f"Document: {entry.display_title}",
                    f"Hierarchy: {hierarchy_path}",
                    f"Chapter: {group[0].chapter or 'Not specified'}",
                    f"Section: {group[0].section or 'Not specified'}",
                    f"Subsection: {group[0].subsection or 'Not specified'}",
                    f"Content types: {type_text}",
                    "",
                    representation,
                ]
            )
            bm25_text = "\n".join(
                [
                    hierarchy_path,
                    hierarchy_path,
                    hierarchy_path,
                    f"Content types: {type_text}",
                    representation,
                    "Technical terms: " + ", ".join(entry.subdomains),
                ]
            )
            sections.append(
                TechnicalSection(
                    section_id=section_id,
                    document_id=entry.document_id,
                    document_title=entry.display_title,
                    source_file=entry.canonical_filename,
                    domains=tuple(domain.value for domain in entry.domains),
                    chapter=group[0].chapter,
                    section=group[0].section,
                    subsection=group[0].subsection,
                    hierarchy_path=hierarchy_path,
                    page_start=min(item.page_start for item in group),
                    page_end=max(item.page_end for item in group),
                    child_chunk_ids=tuple(item.chunk_id for item in group),
                    chunk_types=chunk_types,
                    display_text=representation,
                    embedding_text=embedding_text,
                    bm25_text=bm25_text,
                    token_count=self.count_tokens(representation),
                    sha256=self._stable_digest(
                        section_id,
                        representation,
                    ),
                )
            )

        return sorted(
            sections,
            key=lambda item: (
                item.document_id,
                item.page_start,
                item.page_end,
                item.section_id,
            ),
        )

    def chunk(
        self,
        *,
        document_path: Path,
        entry: DocumentCatalogEntry,
    ) -> TechnicalChunkingResult:
        """Create and link active children and section-sized parents."""

        drafts = self._load_drafts(document_path)
        initial_children, excluded = self._draft_children(drafts, entry)
        parents: list[TechnicalParentChunk] = []
        parent_by_child: dict[str, str] = {}

        for group in self._parent_groups(initial_children):
            display_text = "\n\n".join(child.display_text for child in group)
            digest = self._stable_digest(
                entry.document_id,
                group[0].chunk_id,
                group[-1].chunk_id,
                display_text,
            )
            parent_id = f"{entry.document_id}_parent_{digest[:16]}"
            parent = TechnicalParentChunk(
                parent_id=parent_id,
                document_id=entry.document_id,
                document_title=entry.display_title,
                source_file=entry.canonical_filename,
                chapter=group[0].chapter,
                section=group[0].section,
                subsection=group[0].subsection,
                hierarchy_path=group[0].hierarchy_path,
                section_id=group[0].section_id,
                page_start=min(child.page_start for child in group),
                page_end=max(child.page_end for child in group),
                child_chunk_ids=tuple(child.chunk_id for child in group),
                display_text=display_text,
                token_count=self.count_tokens(display_text),
                sha256=self._stable_digest(display_text),
            )
            parents.append(parent)

            for child in group:
                parent_by_child[child.chunk_id] = parent_id

        children = [
            child.model_copy(
                update={
                    "parent_id": parent_by_child[child.chunk_id],
                    "previous_chunk_id": (
                        initial_children[index - 1].chunk_id
                        if (
                            index > 0
                            and (
                                initial_children[index - 1].chapter,
                                initial_children[index - 1].section,
                            )
                            == (child.chapter, child.section)
                        )
                        else None
                    ),
                    "next_chunk_id": (
                        initial_children[index + 1].chunk_id
                        if (
                            index + 1 < len(initial_children)
                            and (
                                initial_children[index + 1].chapter,
                                initial_children[index + 1].section,
                            )
                            == (child.chapter, child.section)
                        )
                        else None
                    ),
                }
            )
            for index, child in enumerate(initial_children)
        ]
        sections = self._build_sections(children, entry)
        return TechnicalChunkingResult(
            children=tuple(children),
            parents=tuple(parents),
            sections=tuple(sections),
            excluded_chunk_count=excluded,
        )
