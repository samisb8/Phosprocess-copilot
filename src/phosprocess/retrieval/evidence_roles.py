"""Generic role-aware evidence selection.

This module may reserve evidence for information-structure roles produced by
the retrieval planner.  It deliberately contains no domain answer facts,
expected numeric values, equipment paths, or document-specific rules.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from phosprocess.reranking.reranker import RerankedSearchResult
from phosprocess.retrieval.hybrid import HybridSearchResult
from phosprocess.retrieval.retrieval_planner import (
    EvidenceRole,
    RetrievalPlan,
)
from phosprocess.retrieval.v3_selection import V3SelectedResult


@dataclass(frozen=True, slots=True)
class RoleEvidenceSelection:
    """Selection plus structural-role telemetry."""

    selected: tuple[V3SelectedResult, ...]
    covered_roles: tuple[str, ...]


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize(
        "NFKD",
        value.casefold(),
    )
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return re.sub(
        r"[^a-z0-9%]+",
        " ",
        without_marks,
    ).strip()


def _chunk_text(
    result: HybridSearchResult,
) -> str:
    chunk = result.chunk

    return _normalize(
        " ".join(
            str(part)
            for part in (
                getattr(chunk, "document_title", ""),
                getattr(chunk, "hierarchy_path", ""),
                getattr(chunk, "section", ""),
                getattr(chunk, "subsection", ""),
                getattr(chunk, "text", ""),
                getattr(chunk, "embedding_text", ""),
            )
            if part
        )
    )


def _is_non_evidentiary_candidate(
    candidate: HybridSearchResult,
) -> bool:
    """Reject only generic navigation/reference noise."""

    text = _chunk_text(candidate)

    if not text:
        return True

    low_value_markers = (
        "table of contents",
        "list of figures",
        "list of tables",
        "bibliography",
        "references",
        "index",
    )

    return any(marker in text for marker in low_value_markers)


def _supports_role(
    candidate: HybridSearchResult,
    role: EvidenceRole,
    plan: RetrievalPlan,
) -> bool:
    """Use retrieval provenance only; semantic truth is checked by the LLM."""

    del plan

    if role.name not in candidate.role_matches:
        return False

    return not _is_non_evidentiary_candidate(candidate)


def supported_role_names(
    plan: RetrievalPlan,
    candidates: Sequence[HybridSearchResult],
) -> tuple[str, ...]:
    """Return structural roles reached by at least one candidate."""

    return tuple(
        role.name
        for role in plan.roles
        if any(
            _supports_role(
                candidate,
                role,
                plan,
            )
            for candidate in candidates
        )
    )


def select_role_aware_evidence(
    plan: RetrievalPlan,
    candidates: Sequence[HybridSearchResult],
    reranked_results: Sequence[RerankedSearchResult],
    *,
    top_k: int,
) -> RoleEvidenceSelection:
    """Diversify by generic retrieval roles, then fill by reranker."""

    if top_k <= 0:
        raise ValueError("top_k doit être strictement positif.")

    candidate_by_id = {item.chunk.chunk_id: item for item in candidates}
    reranked_by_id = {item.chunk.chunk_id: item for item in reranked_results}

    ordered_candidates = [
        candidate_by_id[item.chunk.chunk_id]
        for item in reranked_results
        if item.chunk.chunk_id in candidate_by_id
    ]

    selected_ids: list[str] = []
    source_by_id: dict[str, str] = {}
    roles_by_id: dict[str, list[str]] = {}
    covered: list[str] = []
    for role in plan.roles:
        matching = [
            item
            for item in ordered_candidates
            if _supports_role(
                item,
                role,
                plan,
            )
        ]

        if not matching:
            continue

        covered.append(role.name)

        chosen = next(
            (item for item in matching if item.chunk.chunk_id not in selected_ids),
            matching[0],
        )

        chunk_id = chosen.chunk.chunk_id
        bound = roles_by_id.setdefault(
            chunk_id,
            [],
        )

        if role.name not in bound:
            bound.append(role.name)

        if chunk_id not in selected_ids and len(selected_ids) < top_k:
            selected_ids.append(chunk_id)
            source_by_id[chunk_id] = "role_evidence"

    for item in reranked_results:
        if len(selected_ids) >= top_k:
            break

        chunk_id = item.chunk.chunk_id

        if chunk_id in selected_ids or chunk_id not in candidate_by_id:
            continue

        selected_ids.append(chunk_id)
        source_by_id[chunk_id] = "reranker_fill"

    def provenance(
        chunk_id: str,
    ) -> str:
        roles = roles_by_id.get(
            chunk_id,
            [],
        )

        if len(roles) == 1:
            return f"evidence_role:{roles[0]}"

        if roles:
            return "evidence_roles:" + ",".join(roles)

        return source_by_id[chunk_id]

    selected = tuple(
        V3SelectedResult(
            rank=rank,
            chunk_id=chunk_id,
            source=provenance(chunk_id),
            reranker_rank=(reranked_by_id[chunk_id].rank if chunk_id in reranked_by_id else None),
            hybrid_rank=candidate_by_id[chunk_id].rank,
            bm25_rank=candidate_by_id[chunk_id].bm25_rank,
        )
        for rank, chunk_id in enumerate(
            selected_ids,
            start=1,
        )
    )

    return RoleEvidenceSelection(
        selected=selected,
        covered_roles=tuple(dict.fromkeys(covered)),
    )
