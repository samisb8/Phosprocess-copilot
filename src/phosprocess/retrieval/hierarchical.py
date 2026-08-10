"""Two-stage hierarchical retrieval over sections then child passages."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bm25s
import faiss
import numpy as np

from phosprocess.ingestion.chunk_serialization import (
    TechnicalChunkType,
    TechnicalSection,
)
from phosprocess.retrieval.bm25 import technical_tokenize
from phosprocess.retrieval.domain_router import DomainRoutingDecision
from phosprocess.retrieval.query_expansion import ExpandedTechnicalQuery

_LOW_VALUE_HEADING = re.compile(
    r"\b(?:table of contents|contents|list of figures|liste des figures|"
    r"list of abbreviations|abbreviations|liste des abr[e?]viations|"
    r"bibliography|references|subject index|author index)\b",
    re.I,
)
_SIMULATION_QUERY = re.compile(
    r"\b(?:simulation|matlab|simulink|curve|courbe|model result|r[eé]sultat)\b",
    re.I,
)


@dataclass(frozen=True, slots=True)
class RetrievalProfile:
    preferred_types: frozenset[TechnicalChunkType]
    penalized_types: frozenset[TechnicalChunkType]
    title_terms: tuple[str, ...]


_PROFILES: dict[str, RetrievalProfile] = {
    "process_flow": RetrievalProfile(
        preferred_types=frozenset(
            {
                TechnicalChunkType.PROCESS_DESCRIPTION,
                TechnicalChunkType.EQUIPMENT_DESCRIPTION,
                TechnicalChunkType.PROCEDURE,
                TechnicalChunkType.NARRATIVE,
            }
        ),
        penalized_types=frozenset(
            {
                TechnicalChunkType.EQUATION,
                TechnicalChunkType.WORKED_EXAMPLE,
                TechnicalChunkType.SIMULATION_RESULTS,
            }
        ),
        title_terms=(
            "process",
            "flow",
            "sequence",
            "operation",
            "fonctionnement",
            "entry",
            "inlet",
            "transition",
            "exit",
            "outlet",
        ),
    ),
    "balance": RetrievalProfile(
        preferred_types=frozenset(
            {
                TechnicalChunkType.BALANCE,
                TechnicalChunkType.EQUATION,
                TechnicalChunkType.EQUATION_EXPLANATION,
                TechnicalChunkType.TABLE,
            }
        ),
        penalized_types=frozenset(
            {
                TechnicalChunkType.SIMULATION_RESULTS,
                TechnicalChunkType.FIGURE_CAPTION,
            }
        ),
        title_terms=(
            "balance",
            "bilan",
            "mass",
            "material",
            "component",
            "energy",
            "equation",
            "table",
        ),
    ),
    "equation_explanation": RetrievalProfile(
        preferred_types=frozenset(
            {
                TechnicalChunkType.BALANCE,
                TechnicalChunkType.EQUATION,
                TechnicalChunkType.EQUATION_EXPLANATION,
                TechnicalChunkType.TABLE,
            }
        ),
        penalized_types=frozenset({TechnicalChunkType.SIMULATION_RESULTS}),
        title_terms=("equation", "formula", "balance", "relation", "table"),
    ),
    "calculation": RetrievalProfile(
        preferred_types=frozenset(
            {
                TechnicalChunkType.BALANCE,
                TechnicalChunkType.EQUATION,
                TechnicalChunkType.EQUATION_EXPLANATION,
                TechnicalChunkType.WORKED_EXAMPLE,
                TechnicalChunkType.TABLE,
            }
        ),
        penalized_types=frozenset(),
        title_terms=("calculation", "example", "balance", "equation", "table"),
    ),
    "troubleshooting": RetrievalProfile(
        preferred_types=frozenset(
            {
                TechnicalChunkType.TROUBLESHOOTING,
                TechnicalChunkType.OPERATING_PROBLEM,
                TechnicalChunkType.EQUIPMENT_DESCRIPTION,
                TechnicalChunkType.NARRATIVE,
            }
        ),
        penalized_types=frozenset(
            {
                TechnicalChunkType.SIMULATION_RESULTS,
                TechnicalChunkType.WORKED_EXAMPLE,
            }
        ),
        title_terms=(
            "operating",
            "problem",
            "difficulty",
            "troubleshooting",
            "cause",
            "effect",
            "action",
        ),
    ),
    "definition": RetrievalProfile(
        preferred_types=frozenset(
            {
                TechnicalChunkType.DEFINITION,
                TechnicalChunkType.EQUIPMENT_DESCRIPTION,
                TechnicalChunkType.NARRATIVE,
            }
        ),
        penalized_types=frozenset(
            {
                TechnicalChunkType.SIMULATION_RESULTS,
                TechnicalChunkType.WORKED_EXAMPLE,
            }
        ),
        title_terms=("definition", "types and applications", "description"),
    ),
}
_DEFAULT_PROFILE = RetrievalProfile(
    preferred_types=frozenset(
        {
            TechnicalChunkType.NARRATIVE,
            TechnicalChunkType.PROCESS_DESCRIPTION,
            TechnicalChunkType.EQUIPMENT_DESCRIPTION,
        }
    ),
    penalized_types=frozenset({TechnicalChunkType.SIMULATION_RESULTS}),
    title_terms=(),
)


@dataclass(frozen=True, slots=True)
class SectionSearchResult:
    rank: int
    section: TechnicalSection
    final_score: float
    rrf_score: float
    dense_rank: int | None
    dense_score: float | None
    bm25_rank: int | None
    bm25_score: float | None
    source_boost: float
    profile_boost: float


@dataclass(frozen=True, slots=True)
class SectionSearchResponse:
    query: str
    question_type: str
    duration_ms: float
    candidates_considered: int
    results: tuple[SectionSearchResult, ...]
    allowed_chunk_ids: frozenset[str]


@dataclass(slots=True)
class _Candidate:
    section: TechnicalSection
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None


class HierarchicalSectionRetriever:
    """Search hierarchy titles and representative section content first."""

    def __init__(
        self,
        *,
        version_directory: Path,
        query_embedder: Any,
        rrf_k: int = 60,
    ) -> None:
        self.version_directory = version_directory.resolve()
        self.root = self.version_directory / "sections"
        self.dense_directory = self.root / "dense"
        self.bm25_directory = self.root / "bm25"
        self.rrf_k = rrf_k
        self.query_embedder = query_embedder
        self.dense_index = faiss.read_index(str(self.dense_directory / "index.faiss"))
        self.dense_sections = self._load_metadata(
            self.dense_directory / "metadata.jsonl",
            id_field="vector_id",
        )
        self.bm25_sections = self._load_metadata(
            self.bm25_directory / "metadata.jsonl",
            id_field="lexical_id",
        )
        self.bm25_model: Any = bm25s.BM25.load(
            str(self.bm25_directory),
            load_corpus=False,
            mmap=False,
            show_progress=False,
        )
        dense_ids = [item.section_id for item in self.dense_sections]
        bm25_ids = [item.section_id for item in self.bm25_sections]
        if dense_ids != bm25_ids or int(self.dense_index.ntotal) != len(dense_ids):
            raise ValueError("Les index hiérarchiques dense et BM25 sont désalignés.")

    @staticmethod
    def _load_metadata(path: Path, *, id_field: str) -> list[TechnicalSection]:
        sections: list[TechnicalSection] = []
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise ValueError(f"{path.name}, ligne {line_number} vide.")
                payload = json.loads(line)
                payload.pop(id_field, None)
                sections.append(TechnicalSection.model_validate(payload))
        return sections

    @classmethod
    def is_available(cls, version_directory: Path) -> bool:
        root = version_directory / "sections"
        return all(
            path.is_file()
            for path in (
                root / "dense" / "index.faiss",
                root / "dense" / "metadata.jsonl",
                root / "bm25" / "metadata.jsonl",
                root / "bm25" / "params.index.json",
            )
        )

    def _dense_search(
        self,
        query: str,
        *,
        top_k: int,
        document_ids: set[str] | None,
    ) -> list[tuple[int, float, TechnicalSection]]:
        vector = self.query_embedder.embed_query(query)
        matrix = np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32)
        retrieval_k = (
            len(self.dense_sections) if document_ids else min(top_k, len(self.dense_sections))
        )
        scores, ids = self.dense_index.search(matrix, retrieval_k)
        results: list[tuple[int, float, TechnicalSection]] = []
        for score, index in zip(scores[0], ids[0], strict=True):
            position = int(index)
            if position < 0:
                continue
            section = self.dense_sections[position]
            if document_ids and section.document_id not in document_ids:
                continue
            results.append((len(results) + 1, float(score), section))
            if len(results) >= top_k:
                break
        return results

    def _bm25_search(
        self,
        query: str,
        *,
        top_k: int,
        document_ids: set[str] | None,
    ) -> list[tuple[int, float, TechnicalSection]]:
        tokens = technical_tokenize(query)
        if not tokens:
            return []
        retrieval_k = (
            len(self.bm25_sections) if document_ids else min(top_k, len(self.bm25_sections))
        )
        raw = self.bm25_model.retrieve(
            [tokens],
            k=retrieval_k,
            sorted=True,
            return_as="tuple",
            show_progress=False,
        )
        results: list[tuple[int, float, TechnicalSection]] = []
        for index, score in zip(raw.documents[0], raw.scores[0], strict=True):
            position = int(index)
            value = float(score)
            if position < 0 or value <= 0:
                continue
            section = self.bm25_sections[position]
            if document_ids and section.document_id not in document_ids:
                continue
            results.append((len(results) + 1, value, section))
            if len(results) >= top_k:
                break
        return results

    @staticmethod
    def _profile_adjustment(
        section: TechnicalSection,
        *,
        profile: RetrievalProfile,
        query: str,
    ) -> float:
        types = set(section.chunk_types)
        score = 0.0
        if types & profile.preferred_types:
            score += 0.045
        if types & profile.penalized_types:
            score -= 0.06
        normalized_path = section.hierarchy_path.casefold()
        title_matches = sum(term in normalized_path for term in profile.title_terms)
        score += min(0.06, title_matches * 0.02)
        if TechnicalChunkType.SIMULATION_RESULTS in types and not _SIMULATION_QUERY.search(query):
            score -= 0.12
        return score

    def search(
        self,
        expanded_query: ExpandedTechnicalQuery,
        *,
        question_type: str,
        routing: DomainRoutingDecision,
        top_k: int = 8,
        candidate_k: int = 30,
    ) -> SectionSearchResponse:
        started = time.perf_counter()
        profile = _PROFILES.get(question_type, _DEFAULT_PROFILE)
        document_ids = set(routing.hard_filter) if routing.hard_filter else None
        dense = self._dense_search(
            expanded_query.dense_query,
            top_k=candidate_k,
            document_ids=document_ids,
        )
        lexical = self._bm25_search(
            expanded_query.bm25_expanded_query,
            top_k=candidate_k,
            document_ids=document_ids,
        )
        candidates: dict[str, _Candidate] = {}
        for rank, score, section in dense:
            item = candidates.setdefault(section.section_id, _Candidate(section))
            item.dense_rank = rank
            item.dense_score = score
        for rank, score, section in lexical:
            item = candidates.setdefault(section.section_id, _Candidate(section))
            item.bm25_rank = rank
            item.bm25_score = score

        scored: list[tuple[_Candidate, float, float, float, float]] = []
        for item in candidates.values():
            if _LOW_VALUE_HEADING.search(item.section.hierarchy_path):
                continue
            rrf = 0.0
            if item.dense_rank is not None:
                rrf += 1.0 / (self.rrf_k + item.dense_rank)
            if item.bm25_rank is not None:
                rrf += 1.0 / (self.rrf_k + item.bm25_rank)
            source_boost = routing.soft_boosts.get(item.section.document_id, 0.0)
            profile_boost = self._profile_adjustment(
                item.section,
                profile=profile,
                query=expanded_query.standalone_query,
            )
            scored.append(
                (
                    item,
                    rrf + source_boost + profile_boost,
                    rrf,
                    source_boost,
                    profile_boost,
                )
            )
        scored.sort(
            key=lambda value: (
                -value[1],
                -(value[0].dense_score or -1.0),
                value[0].section.document_id,
                value[0].section.page_start,
            )
        )

        selected: list[SectionSearchResult] = []
        per_document: dict[str, int] = {}
        for item, final, rrf, source_boost, profile_boost in scored:
            if len(selected) >= top_k:
                break
            document_count = per_document.get(item.section.document_id, 0)
            if document_count >= 3:
                continue
            selected.append(
                SectionSearchResult(
                    rank=len(selected) + 1,
                    section=item.section,
                    final_score=final,
                    rrf_score=rrf,
                    dense_rank=item.dense_rank,
                    dense_score=item.dense_score,
                    bm25_rank=item.bm25_rank,
                    bm25_score=item.bm25_score,
                    source_boost=source_boost,
                    profile_boost=profile_boost,
                )
            )
            per_document[item.section.document_id] = document_count + 1

        allowed = frozenset(
            chunk_id for result in selected for chunk_id in result.section.child_chunk_ids
        )
        if not selected or not allowed:
            raise ValueError("La recherche hiérarchique n'a sélectionné aucune section.")
        return SectionSearchResponse(
            query=expanded_query.standalone_query,
            question_type=question_type,
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            candidates_considered=len(candidates),
            results=tuple(selected),
            allowed_chunk_ids=allowed,
        )
