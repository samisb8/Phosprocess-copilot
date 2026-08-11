"""Parent-first evidence reconstruction and dynamic token-budget packing."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from phosprocess.ingestion.chunk_serialization import (
    TechnicalChildChunk,
    TechnicalParentChunk,
)
from phosprocess.observability.latency import estimate_tokens
from phosprocess.retrieval.evidence_bundle import (
    EvidenceBundle,
    EvidenceContextScope,
    render_evidence_block,
)

TokenCounter = Callable[[str], int]

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class EvidenceAnchor:
    """Reranked child eligible for documentary context packing."""

    child: TechnicalChildChunk
    score: float
    provenance: str


@dataclass(frozen=True, slots=True)
class ParentCandidate:
    """All eligible anchors that resolve to one documentary parent."""

    parent_id: str
    anchors: tuple[EvidenceAnchor, ...]
    best_anchor_score: float
    selection_provenance: tuple[str, ...]
    first_anchor_position: int

    @property
    def anchor_chunk_ids(self) -> tuple[str, ...]:
        return tuple(anchor.child.chunk_id for anchor in self.anchors)


@dataclass(frozen=True, slots=True)
class ContextExpansionConfig:
    """Technical context budgets; bundle cardinality is intentionally dynamic."""

    neighbor_window: int = 1
    max_tokens_per_bundle: int = 650
    max_total_context_tokens: int = 2600
    maximum_bundles: int | None = None

    def __post_init__(self) -> None:
        if self.neighbor_window < 0:
            raise ValueError("neighbor_window doit etre positif ou nul.")
        if self.max_tokens_per_bundle <= 0 or self.max_total_context_tokens <= 0:
            raise ValueError("Les budgets de contexte doivent etre positifs.")
        if self.maximum_bundles is not None and self.maximum_bundles <= 0:
            raise ValueError("maximum_bundles doit etre positif lorsqu'il est defini.")


class ContextExpander:
    """Group anchors by parent, reconstruct faithful context, then pack it."""

    def __init__(
        self,
        *,
        children: list[TechnicalChildChunk],
        parents: list[TechnicalParentChunk],
        config: ContextExpansionConfig | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.config = config or ContextExpansionConfig()
        self.token_counter = token_counter or estimate_tokens
        self.child_by_id = {child.chunk_id: child for child in children}
        self.child_order = {child.chunk_id: position for position, child in enumerate(children)}
        self.parent_by_id = {parent.parent_id: parent for parent in parents}

    @staticmethod
    def _normalize_documentary_text(text: str) -> str:
        return _WHITESPACE.sub(" ", text).strip().casefold()

    def group_anchors_by_parent(
        self,
        anchors: list[EvidenceAnchor],
    ) -> list[ParentCandidate]:
        """Collapse duplicate anchors and create one ranked candidate per parent."""

        grouped: dict[str, list[tuple[int, EvidenceAnchor]]] = {}
        seen_chunk_ids: set[str] = set()
        for position, anchor in enumerate(anchors):
            chunk_id = anchor.child.chunk_id
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            grouped.setdefault(anchor.child.parent_id, []).append((position, anchor))

        candidates: list[ParentCandidate] = []
        for parent_id, positioned in grouped.items():
            ordered = tuple(
                anchor
                for _position, anchor in sorted(
                    positioned,
                    key=lambda item: (-item[1].score, item[0]),
                )
            )
            provenance = tuple(dict.fromkeys(anchor.provenance for anchor in ordered))
            candidates.append(
                ParentCandidate(
                    parent_id=parent_id,
                    anchors=ordered,
                    best_anchor_score=max(anchor.score for anchor in ordered),
                    selection_provenance=provenance,
                    first_anchor_position=min(position for position, _anchor in positioned),
                )
            )

        return sorted(
            candidates,
            key=lambda item: (-item.best_anchor_score, item.first_anchor_position),
        )

    @staticmethod
    def _structurally_compatible(
        anchor: TechnicalChildChunk,
        candidate: TechnicalChildChunk,
    ) -> bool:
        """Prevent neighbor expansion across document or hierarchy boundaries."""

        if anchor.document_id != candidate.document_id:
            return False
        if anchor.section_id and candidate.section_id:
            return anchor.section_id == candidate.section_id
        return (
            anchor.chapter,
            anchor.section,
            anchor.subsection,
            anchor.hierarchy_path,
        ) == (
            candidate.chapter,
            candidate.section,
            candidate.subsection,
            candidate.hierarchy_path,
        )

    def _parent_children(
        self,
        candidate: ParentCandidate,
    ) -> tuple[TechnicalParentChunk | None, list[TechnicalChildChunk]]:
        parent = self.parent_by_id.get(candidate.parent_id)
        if parent is None:
            return None, []
        primary = candidate.anchors[0].child
        if parent.document_id != primary.document_id:
            return None, []
        children = [
            child
            for chunk_id in parent.child_chunk_ids
            if (child := self.child_by_id.get(chunk_id)) is not None
            and child.document_id == primary.document_id
        ]
        return parent, children

    @staticmethod
    def _provenance_priority(candidate: ParentCandidate) -> int:
        """Break score ties using generic retrieval provenance only."""

        provenance = " ".join(candidate.selection_provenance).casefold()
        if "evidence_role" in provenance or "role_evidence" in provenance:
            return 0
        if "reranker" in provenance:
            return 1
        if "bm25" in provenance:
            return 2
        return 3

    def _compatible_neighbors(
        self,
        anchors: tuple[EvidenceAnchor, ...],
    ) -> list[TechnicalChildChunk]:
        neighbors: dict[str, TechnicalChildChunk] = {}
        for evidence_anchor in anchors:
            anchor = evidence_anchor.child
            for direction in ("previous_chunk_id", "next_chunk_id"):
                current = anchor
                for _ in range(self.config.neighbor_window):
                    chunk_id = getattr(current, direction)
                    if not chunk_id:
                        break
                    neighbor = self.child_by_id.get(chunk_id)
                    if neighbor is None or not self._structurally_compatible(anchor, neighbor):
                        break
                    neighbors.setdefault(neighbor.chunk_id, neighbor)
                    current = neighbor
        return sorted(
            neighbors.values(),
            key=lambda child: self.child_order.get(child.chunk_id, 10**12),
        )

    @staticmethod
    def _page_range(children: list[TechnicalChildChunk]) -> tuple[int, int]:
        return (
            min(child.page_start for child in children),
            max(child.page_end for child in children),
        )

    def _rendered_tokens(
        self,
        *,
        source_number: int,
        anchor: TechnicalChildChunk,
        page_start: int,
        page_end: int,
        display_text: str,
    ) -> int:
        return self.token_counter(
            render_evidence_block(
                source_number=source_number,
                document_title=anchor.document_title,
                filename=anchor.source_file,
                chapter=anchor.chapter,
                section=anchor.section,
                page_start=page_start,
                page_end=page_end,
                display_text=display_text,
            )
        )

    def _build_bundle(
        self,
        candidate: ParentCandidate,
        *,
        source_number: int,
        maximum_tokens: int,
        reserved_anchor_ids: set[str],
        used_chunk_ids: set[str],
        seen_documentary_texts: set[str],
    ) -> EvidenceBundle | None:
        primary = candidate.anchors[0].child
        parent, parent_children = self._parent_children(candidate)
        available_anchors = [
            anchor
            for anchor in candidate.anchors
            if anchor.child.chunk_id not in used_chunk_ids
        ]
        if not available_anchors:
            return None

        parent_ids = {child.chunk_id for child in parent_children}
        parent_is_available = (
            parent is not None
            and parent_children
            and not any(child.chunk_id in used_chunk_ids for child in parent_children)
            and all(anchor.child.chunk_id in parent_ids for anchor in available_anchors)
        )
        if parent_is_available:
            normalized_parent = self._normalize_documentary_text(parent.display_text)
            parent_tokens = self._rendered_tokens(
                source_number=source_number,
                anchor=primary,
                page_start=parent.page_start,
                page_end=parent.page_end,
                display_text=parent.display_text,
            )
            if normalized_parent not in seen_documentary_texts and parent_tokens <= maximum_tokens:
                anchor_text = "\n\n".join(
                    anchor.child.display_text for anchor in available_anchors
                )
                documentary_tokens = self.token_counter(parent.display_text)
                metadata_tokens = self._rendered_tokens(
                    source_number=source_number,
                    anchor=primary,
                    page_start=parent.page_start,
                    page_end=parent.page_end,
                    display_text="",
                )
                return EvidenceBundle(
                    source_number=source_number,
                    document_id=primary.document_id,
                    document_title=primary.document_title,
                    filename=primary.source_file,
                    chapter=primary.chapter,
                    section=primary.section,
                    subsection=primary.subsection,
                    hierarchy_path=primary.hierarchy_path,
                    page_start=parent.page_start,
                    page_end=parent.page_end,
                    parent_id=candidate.parent_id,
                    anchor_chunk_ids=tuple(
                        anchor.child.chunk_id for anchor in available_anchors
                    ),
                    supporting_chunk_ids=tuple(child.chunk_id for child in parent_children),
                    display_text=parent.display_text,
                    token_count=parent_tokens,
                    documentary_token_count=documentary_tokens,
                    metadata_token_count=metadata_tokens,
                    anchor_token_count=self.token_counter(anchor_text),
                    context_token_count=max(
                        0,
                        documentary_tokens - self.token_counter(anchor_text),
                    ),
                    best_anchor_score=candidate.best_anchor_score,
                    context_scope=EvidenceContextScope.FULL_PARENT,
                    selection_provenance=" | ".join(candidate.selection_provenance),
                )

        selected: dict[str, TechnicalChildChunk] = {}
        selected_anchor_ids: list[str] = []

        def try_add(child: TechnicalChildChunk, *, anchor: bool) -> bool:
            if child.chunk_id in selected or child.chunk_id in used_chunk_ids:
                return False
            normalized = self._normalize_documentary_text(child.display_text)
            if not normalized or normalized in seen_documentary_texts:
                return False
            tentative = [*selected.values(), child]
            tentative.sort(key=lambda item: self.child_order.get(item.chunk_id, 10**12))
            page_start, page_end = self._page_range(tentative)
            display_text = "\n\n".join(item.display_text for item in tentative)
            if self._rendered_tokens(
                source_number=source_number,
                anchor=primary,
                page_start=page_start,
                page_end=page_end,
                display_text=display_text,
            ) > maximum_tokens:
                return False
            selected[child.chunk_id] = child
            if anchor:
                selected_anchor_ids.append(child.chunk_id)
            return True

        for evidence_anchor in available_anchors:
            try_add(evidence_anchor.child, anchor=True)
        if not selected_anchor_ids:
            return None

        anchor_positions = [self.child_order.get(chunk_id, 10**12) for chunk_id in selected]
        complementary_parent_children = sorted(
            (
                child
                for child in parent_children
                if child.chunk_id not in selected
                and child.chunk_id not in reserved_anchor_ids
            ),
            key=lambda child: (
                min(
                    abs(self.child_order.get(child.chunk_id, 10**12) - position)
                    for position in anchor_positions
                ),
                self.child_order.get(child.chunk_id, 10**12),
            ),
        )
        parent_context_added = False
        for child in complementary_parent_children:
            parent_context_added = try_add(child, anchor=False) or parent_context_added

        neighbor_context_added = False
        for child in self._compatible_neighbors(tuple(available_anchors)):
            if child.chunk_id in parent_ids or child.chunk_id in reserved_anchor_ids:
                continue
            neighbor_context_added = try_add(child, anchor=False) or neighbor_context_added

        ordered = sorted(
            selected.values(),
            key=lambda child: self.child_order.get(child.chunk_id, 10**12),
        )
        display_text = "\n\n".join(child.display_text for child in ordered)
        page_start, page_end = self._page_range(ordered)
        anchor_text = "\n\n".join(
            self.child_by_id[chunk_id].display_text for chunk_id in selected_anchor_ids
        )
        documentary_tokens = self.token_counter(display_text)
        anchor_tokens = self.token_counter(anchor_text)
        total_tokens = self._rendered_tokens(
            source_number=source_number,
            anchor=primary,
            page_start=page_start,
            page_end=page_end,
            display_text=display_text,
        )
        metadata_tokens = self._rendered_tokens(
            source_number=source_number,
            anchor=primary,
            page_start=page_start,
            page_end=page_end,
            display_text="",
        )
        if parent_context_added:
            scope = EvidenceContextScope.PARTIAL_PARENT
        elif neighbor_context_added:
            scope = EvidenceContextScope.ANCHOR_WITH_NEIGHBORS
        else:
            scope = EvidenceContextScope.ANCHOR_ONLY
        return EvidenceBundle(
            source_number=source_number,
            document_id=primary.document_id,
            document_title=primary.document_title,
            filename=primary.source_file,
            chapter=primary.chapter,
            section=primary.section,
            subsection=primary.subsection,
            hierarchy_path=primary.hierarchy_path,
            page_start=page_start,
            page_end=page_end,
            parent_id=candidate.parent_id,
            anchor_chunk_ids=tuple(selected_anchor_ids),
            supporting_chunk_ids=tuple(child.chunk_id for child in ordered),
            display_text=display_text,
            token_count=total_tokens,
            documentary_token_count=documentary_tokens,
            metadata_token_count=metadata_tokens,
            anchor_token_count=anchor_tokens,
            context_token_count=max(0, documentary_tokens - anchor_tokens),
            best_anchor_score=candidate.best_anchor_score,
            context_scope=scope,
            selection_provenance=" | ".join(candidate.selection_provenance),
        )

    def expand(
        self,
        anchors: list[EvidenceAnchor],
        *,
        question_type: str,
    ) -> list[EvidenceBundle]:
        """Pack a dynamic number of deduplicated parent evidence bundles."""

        del question_type
        candidates = self.group_anchors_by_parent(anchors)
        reserved_anchor_ids = {
            anchor.child.chunk_id for candidate in candidates for anchor in candidate.anchors
        }
        bundles: list[EvidenceBundle] = []
        used_chunk_ids: set[str] = set()
        seen_documentary_texts: set[str] = set()
        used_sections: set[str] = set()
        remaining_total = self.config.max_total_context_tokens

        pending = list(candidates)
        while pending and remaining_total > 0:
            if (
                self.config.maximum_bundles is not None
                and len(bundles) >= self.config.maximum_bundles
            ):
                break
            pending.sort(
                key=lambda item: (
                    -item.best_anchor_score,
                    self._provenance_priority(item),
                    (
                        item.anchors[0].child.section_id in used_sections
                        if item.anchors[0].child.section_id
                        else False
                    ),
                    item.first_anchor_position,
                )
            )
            packed: EvidenceBundle | None = None
            packed_candidate: ParentCandidate | None = None
            bundle_budget = min(self.config.max_tokens_per_bundle, remaining_total)
            for candidate in pending:
                packed = self._build_bundle(
                    candidate,
                    source_number=len(bundles) + 1,
                    maximum_tokens=bundle_budget,
                    reserved_anchor_ids=reserved_anchor_ids,
                    used_chunk_ids=used_chunk_ids,
                    seen_documentary_texts=seen_documentary_texts,
                )
                if packed is not None:
                    packed_candidate = candidate
                    break
            if packed is None or packed_candidate is None:
                break
            bundles.append(packed)
            pending.remove(packed_candidate)
            remaining_total -= packed.token_count
            used_chunk_ids.update(packed.supporting_chunk_ids)
            seen_documentary_texts.add(
                self._normalize_documentary_text(packed.display_text)
            )
            for chunk_id in packed.supporting_chunk_ids:
                child = self.child_by_id.get(chunk_id)
                if child is not None:
                    seen_documentary_texts.add(
                        self._normalize_documentary_text(child.display_text)
                    )
            section_id = packed_candidate.anchors[0].child.section_id
            if section_id:
                used_sections.add(section_id)

        return bundles
