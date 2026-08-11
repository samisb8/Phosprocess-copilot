"""Generic, label-blind structural candidate expansion for Phase 9 evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum

from phosprocess.ingestion.chunk_serialization import (
    TechnicalChildChunk,
    TechnicalParentChunk,
)
from phosprocess.retrieval.context_expander import ContextExpander


class RegionVariant(StrEnum):
    """Bounded structural relationships evaluated before reranking."""

    SAME_PARENT = "same_parent"
    PARENT_AND_NEIGHBORS = "same_parent_previous_next"
    PARENT_NEIGHBORS_SECTION_2 = "same_parent_previous_next_section_distance_2"


@dataclass(frozen=True, slots=True)
class RegionCandidate:
    """A candidate admitted by a structural relationship, without semantic score."""

    chunk_id: str
    provenance: str
    anchor_chunk_id: str


@dataclass(frozen=True, slots=True)
class RegionComposition:
    """Bounded candidate pool and auditable structural provenance."""

    candidate_ids: tuple[str, ...]
    anchor_ids: tuple[str, ...]
    structural_candidates: tuple[RegionCandidate, ...]
    lookup_latency_ms: float
    composition_latency_ms: float


class RegionCandidateExpander:
    """Expand strong anchors through local document structure only."""

    def __init__(
        self,
        *,
        children: list[TechnicalChildChunk],
        parents: list[TechnicalParentChunk],
    ) -> None:
        self.child_by_id = {child.chunk_id: child for child in children}
        self.parent_by_id = {parent.parent_id: parent for parent in parents}
        self.child_order = {
            child.chunk_id: position for position, child in enumerate(children)
        }
        self.document_children: dict[str, list[TechnicalChildChunk]] = {}
        for child in children:
            self.document_children.setdefault(child.document_id, []).append(child)
        self.document_position = {
            child.chunk_id: position
            for document_children in self.document_children.values()
            for position, child in enumerate(document_children)
        }

    @staticmethod
    def _compatible(
        anchor: TechnicalChildChunk,
        candidate: TechnicalChildChunk,
    ) -> bool:
        return ContextExpander._structurally_compatible(anchor, candidate)

    def _same_parent(self, anchor: TechnicalChildChunk) -> list[RegionCandidate]:
        parent = self.parent_by_id.get(anchor.parent_id)
        if parent is None or parent.document_id != anchor.document_id:
            return []
        return [
            RegionCandidate(chunk_id, "same_parent", anchor.chunk_id)
            for chunk_id in parent.child_chunk_ids
            if chunk_id != anchor.chunk_id
            and (candidate := self.child_by_id.get(chunk_id)) is not None
            and self._compatible(anchor, candidate)
        ]

    def _immediate_neighbors(
        self, anchor: TechnicalChildChunk
    ) -> list[RegionCandidate]:
        output: list[RegionCandidate] = []
        for field, provenance in (
            ("previous_chunk_id", "previous_neighbor"),
            ("next_chunk_id", "next_neighbor"),
        ):
            chunk_id = getattr(anchor, field)
            candidate = self.child_by_id.get(chunk_id) if chunk_id else None
            if candidate is not None and self._compatible(anchor, candidate):
                output.append(RegionCandidate(candidate.chunk_id, provenance, anchor.chunk_id))
        return output

    def _section_neighbors(
        self,
        anchor: TechnicalChildChunk,
        *,
        maximum_distance: int = 2,
    ) -> list[RegionCandidate]:
        if not anchor.section_id:
            return []
        document_children = self.document_children.get(anchor.document_id, [])
        position = self.document_position.get(anchor.chunk_id)
        if position is None:
            return []
        output: list[RegionCandidate] = []
        for distance in range(1, maximum_distance + 1):
            for candidate_position in (position - distance, position + distance):
                if not 0 <= candidate_position < len(document_children):
                    continue
                candidate = document_children[candidate_position]
                if (
                    candidate.section_id == anchor.section_id
                    and self._compatible(anchor, candidate)
                ):
                    output.append(
                        RegionCandidate(
                            candidate.chunk_id,
                            "same_section_neighbor",
                            anchor.chunk_id,
                        )
                    )
        return output

    def related_candidates(
        self,
        anchor_id: str,
        variant: RegionVariant,
    ) -> list[RegionCandidate]:
        """Return generic structural candidates in deterministic priority order."""

        anchor = self.child_by_id[anchor_id]
        related = self._same_parent(anchor)
        if variant in {
            RegionVariant.PARENT_AND_NEIGHBORS,
            RegionVariant.PARENT_NEIGHBORS_SECTION_2,
        }:
            related.extend(self._immediate_neighbors(anchor))
        if variant == RegionVariant.PARENT_NEIGHBORS_SECTION_2:
            related.extend(self._section_neighbors(anchor))
        deduplicated: list[RegionCandidate] = []
        seen = {anchor_id}
        for candidate in related:
            if candidate.chunk_id in seen:
                continue
            seen.add(candidate.chunk_id)
            deduplicated.append(candidate)
        return deduplicated

    def compose(
        self,
        baseline_ids: list[str],
        *,
        locked_document: str,
        variant: RegionVariant,
        anchor_k: int,
        candidate_budget: int,
    ) -> RegionComposition:
        """Preserve anchors, then structural candidates, then baseline fill."""

        started = time.perf_counter()
        if anchor_k <= 0 or candidate_budget <= 0:
            raise ValueError("anchor_k and candidate_budget must be positive")
        unique_baseline = list(dict.fromkeys(baseline_ids))
        for chunk_id in unique_baseline:
            chunk = self.child_by_id.get(chunk_id)
            if chunk is None or chunk.document_id != locked_document:
                raise ValueError("baseline violates the exact source lock")
        anchors = unique_baseline[: min(anchor_k, candidate_budget)]
        lookup_started = time.perf_counter()
        structural: list[RegionCandidate] = []
        seen_structural = set(anchors)
        for anchor_id in anchors:
            for candidate in self.related_candidates(anchor_id, variant):
                if candidate.chunk_id in seen_structural:
                    continue
                candidate_chunk = self.child_by_id[candidate.chunk_id]
                if candidate_chunk.document_id != locked_document:
                    continue
                seen_structural.add(candidate.chunk_id)
                structural.append(candidate)
        lookup_latency_ms = (time.perf_counter() - lookup_started) * 1000.0

        ordered = list(anchors)
        ordered.extend(candidate.chunk_id for candidate in structural)
        ordered.extend(unique_baseline)
        candidate_ids = tuple(dict.fromkeys(ordered))[:candidate_budget]
        retained = tuple(
            candidate for candidate in structural if candidate.chunk_id in candidate_ids
        )
        if not set(anchors).issubset(candidate_ids):
            raise AssertionError("anchor preservation invariant violated")
        if len(candidate_ids) > candidate_budget:
            raise AssertionError("candidate budget invariant violated")
        if any(
            self.child_by_id[chunk_id].document_id != locked_document
            for chunk_id in candidate_ids
        ):
            raise AssertionError("source lock invariant violated")
        return RegionComposition(
            candidate_ids=candidate_ids,
            anchor_ids=tuple(anchors),
            structural_candidates=retained,
            lookup_latency_ms=lookup_latency_ms,
            composition_latency_ms=(time.perf_counter() - started) * 1000.0,
        )
