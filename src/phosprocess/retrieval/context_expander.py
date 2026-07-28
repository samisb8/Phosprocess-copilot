"""Parent and neighbor expansion under a strict global context budget."""

from __future__ import annotations

from dataclasses import dataclass

from phosprocess.ingestion.chunk_serialization import (
    TechnicalChildChunk,
    TechnicalParentChunk,
)
from phosprocess.retrieval.evidence_bundle import EvidenceBundle


@dataclass(frozen=True, slots=True)
class EvidenceAnchor:
    """Reranked child selected for contextual expansion."""

    child: TechnicalChildChunk
    score: float
    provenance: str


@dataclass(frozen=True, slots=True)
class ContextExpansionConfig:
    """Initial evidence-window budgets."""

    neighbor_window: int = 1
    include_parent: str = "conditional"
    max_tokens_per_bundle: int = 650
    max_total_context_tokens: int = 2600
    maximum_bundles: int = 5
    minimum_context_slice_tokens: int = 32


@dataclass(frozen=True, slots=True)
class _SelectedPiece:
    child: TechnicalChildChunk
    text: str
    token_count: int
    partial: bool


class ContextExpander:
    """Expand anchors without crossing document boundaries or losing anchors.

    A whole neighboring child often contains 400-560 tokens. With five bundles
    sharing a 2,600-token budget, the old all-or-nothing policy usually had no
    room for a neighbor and therefore reported ``context_tokens=0``. This
    implementation keeps complete neighbors when they fit and otherwise adds a
    bounded tail/head slice, preserving the most useful local continuity.
    """

    def __init__(
        self,
        *,
        children: list[TechnicalChildChunk],
        parents: list[TechnicalParentChunk],
        config: ContextExpansionConfig | None = None,
    ) -> None:
        self.config = config or ContextExpansionConfig()
        self.child_by_id = {child.chunk_id: child for child in children}
        self.child_order = {
            child.chunk_id: position
            for position, child in enumerate(children)
        }
        self.parent_by_id = {parent.parent_id: parent for parent in parents}

    @staticmethod
    def _requires_neighbors(question_type: str) -> bool:
        return question_type in {
            "process_flow",
            "procedure",
            "equation_explanation",
            "troubleshooting",
        }

    @staticmethod
    def _requires_parent(question_type: str) -> bool:
        return question_type in {
            "process_flow",
            "procedure",
            "equation_explanation",
            "table_question",
        }

    def _neighbor_ids(
        self,
        anchor: TechnicalChildChunk,
        *,
        question_type: str,
    ) -> list[str]:
        if not self._requires_neighbors(question_type):
            return [anchor.chunk_id]

        previous_ids: list[str] = []
        current = anchor

        for _ in range(self.config.neighbor_window):
            if not current.previous_chunk_id:
                break

            previous = self.child_by_id.get(current.previous_chunk_id)

            if previous is None or previous.document_id != anchor.document_id:
                break

            previous_ids.append(previous.chunk_id)
            current = previous

        next_ids: list[str] = []
        current = anchor

        for _ in range(self.config.neighbor_window):
            if not current.next_chunk_id:
                break

            following = self.child_by_id.get(current.next_chunk_id)

            if following is None or following.document_id != anchor.document_id:
                break

            next_ids.append(following.chunk_id)
            current = following

        return [*reversed(previous_ids), anchor.chunk_id, *next_ids]

    def _candidate_ids(
        self,
        anchor: TechnicalChildChunk,
        *,
        question_type: str,
        used_parents: set[str],
    ) -> tuple[list[str], bool]:
        ids = self._neighbor_ids(anchor, question_type=question_type)
        parent_available = False

        if (
            self.config.include_parent == "conditional"
            and self._requires_parent(question_type)
            and anchor.parent_id not in used_parents
        ):
            parent = self.parent_by_id.get(anchor.parent_id)

            if parent is not None:
                ids = list(parent.child_chunk_ids)
                parent_available = True

        return list(dict.fromkeys(ids)), parent_available

    @staticmethod
    def _slice_context(
        child: TechnicalChildChunk,
        *,
        maximum_tokens: int,
        use_tail: bool,
    ) -> tuple[str, int]:
        """Take the tail of previous context or head of following context."""

        if maximum_tokens <= 0:
            return "", 0

        if child.token_count <= maximum_tokens:
            return child.display_text, child.token_count

        ratio = maximum_tokens / child.token_count
        character_budget = max(1, int(len(child.display_text) * ratio))
        raw = (
            child.display_text[-character_budget:]
            if use_tail
            else child.display_text[:character_budget]
        ).strip()

        if use_tail:
            boundaries = (". ", "\n")

            for boundary in boundaries:
                position = raw.find(boundary)

                if 0 <= position <= len(raw) // 2:
                    raw = raw[position + len(boundary) :].strip()
                    break
        else:
            boundaries = (". ", "\n")

            for boundary in boundaries:
                position = raw.rfind(boundary)

                if position >= len(raw) // 2:
                    raw = raw[: position + 1].strip()
                    break

        effective_tokens = max(
            1,
            min(
                maximum_tokens,
                round(child.token_count * len(raw) / max(1, len(child.display_text))),
            ),
        )
        return raw, effective_tokens

    def expand(
        self,
        anchors: list[EvidenceAnchor],
        *,
        question_type: str,
    ) -> list[EvidenceBundle]:
        """Build at most five same-document bundles within 2,600 tokens."""

        bundles: list[EvidenceBundle] = []
        remaining_total = self.config.max_total_context_tokens
        used_parents: set[str] = set()
        retained_anchors = anchors[: self.config.maximum_bundles]

        for anchor_index, anchor in enumerate(retained_anchors):
            remaining_slots = len(retained_anchors) - anchor_index
            fair_bundle_budget = max(1, remaining_total // remaining_slots)
            bundle_budget = min(
                self.config.max_tokens_per_bundle,
                fair_bundle_budget,
            )
            candidate_ids, parent_available = self._candidate_ids(
                anchor.child,
                question_type=question_type,
                used_parents=used_parents,
            )
            source_positions = {
                chunk_id: position
                for position, chunk_id in enumerate(candidate_ids)
            }
            anchor_position = source_positions.get(anchor.child.chunk_id, 0)
            candidate_ids.sort(
                key=lambda chunk_id: (
                    chunk_id != anchor.child.chunk_id,
                    abs(source_positions[chunk_id] - anchor_position),
                )
            )

            anchor_tokens = min(anchor.child.token_count, bundle_budget)
            anchor_truncated = anchor.child.token_count > bundle_budget
            anchor_text = (
                self._truncate_text(
                    anchor.child.display_text,
                    source_tokens=anchor.child.token_count,
                    maximum_tokens=bundle_budget,
                )
                if anchor_truncated
                else anchor.child.display_text
            )
            selected: dict[str, _SelectedPiece] = {
                anchor.child.chunk_id: _SelectedPiece(
                    child=anchor.child,
                    text=anchor_text,
                    token_count=anchor_tokens,
                    partial=anchor_truncated,
                )
            }
            used_tokens = anchor_tokens

            expansion_ids = [
                chunk_id
                for chunk_id in candidate_ids
                if chunk_id != anchor.child.chunk_id
            ][: max(2, self.config.neighbor_window * 2)]

            for candidate_index, chunk_id in enumerate(expansion_ids):
                remaining = bundle_budget - used_tokens
                remaining_candidates = len(expansion_ids) - candidate_index

                if remaining < self.config.minimum_context_slice_tokens:
                    break

                child = self.child_by_id.get(chunk_id)

                if child is None or child.document_id != anchor.child.document_id:
                    continue

                fair_slice_budget = max(
                    self.config.minimum_context_slice_tokens,
                    remaining // max(1, remaining_candidates),
                )
                allowed_tokens = min(remaining, fair_slice_budget)

                if child.token_count <= allowed_tokens:
                    text = child.display_text
                    token_count = child.token_count
                    partial = False
                else:
                    text, token_count = self._slice_context(
                        child,
                        maximum_tokens=allowed_tokens,
                        use_tail=(
                            self.child_order[child.chunk_id]
                            < self.child_order[anchor.child.chunk_id]
                        ),
                    )
                    partial = True

                if not text or token_count <= 0:
                    continue

                selected[child.chunk_id] = _SelectedPiece(
                    child=child,
                    text=text,
                    token_count=token_count,
                    partial=partial,
                )
                used_tokens += token_count

            ordered = sorted(
                selected.values(),
                key=lambda piece: self.child_order[piece.child.chunk_id],
            )
            full_text = "\n\n".join(piece.text for piece in ordered)
            context_tokens = sum(
                piece.token_count
                for piece in ordered
                if piece.child.chunk_id != anchor.child.chunk_id
            )
            parent_included = parent_available and len(ordered) > 1
            context_truncated = any(piece.partial for piece in ordered)
            bundles.append(
                EvidenceBundle(
                    source_number=len(bundles) + 1,
                    document_id=anchor.child.document_id,
                    document_title=anchor.child.document_title,
                    filename=anchor.child.source_file,
                    chapter=anchor.child.chapter,
                    section=anchor.child.section,
                    page_start=min(piece.child.page_start for piece in ordered),
                    page_end=max(piece.child.page_end for piece in ordered),
                    anchor_chunk_id=anchor.child.chunk_id,
                    expanded_chunk_ids=tuple(
                        piece.child.chunk_id for piece in ordered
                    ),
                    display_text=full_text,
                    token_count=used_tokens,
                    anchor_token_count=anchor_tokens,
                    context_token_count=context_tokens,
                    anchor_score=anchor.score,
                    selection_provenance=anchor.provenance,
                    parent_included=parent_included,
                    context_truncated=context_truncated,
                )
            )
            remaining_total -= used_tokens

            if parent_included:
                used_parents.add(anchor.child.parent_id)

            if remaining_total <= 0:
                break

        return bundles

    @staticmethod
    def _truncate_text(
        text: str,
        *,
        source_tokens: int,
        maximum_tokens: int,
    ) -> str:
        """Bound a rare oversized anchor without dropping its provenance."""

        if source_tokens <= maximum_tokens:
            return text

        character_budget = max(
            1,
            int(len(text) * maximum_tokens / source_tokens),
        )
        truncated = text[:character_budget].rstrip()

        for boundary in (". ", "\n"):
            position = truncated.rfind(boundary)

            if position >= character_budget // 2:
                truncated = truncated[: position + 1].rstrip()
                break

        return truncated + "\n[Context truncated to token budget]"
