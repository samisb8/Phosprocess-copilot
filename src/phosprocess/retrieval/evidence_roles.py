"""Role-aware evidence selection for comparison, balance and troubleshooting."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from phosprocess.reranking.reranker import RerankedSearchResult
from phosprocess.retrieval.hybrid import HybridSearchResult
from phosprocess.retrieval.retrieval_planner import EvidenceRole, RetrievalPlan
from phosprocess.retrieval.v3_selection import V3SelectedResult


@dataclass(frozen=True, slots=True)
class RoleEvidenceSelection:
    """Final selection plus explicit required-role coverage."""

    selected: tuple[V3SelectedResult, ...]
    covered_roles: tuple[str, ...]
    missing_roles: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_roles


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return re.sub(r"[^a-z0-9%]+", " ", without_marks).strip()


def _chunk_text(result: HybridSearchResult) -> str:
    chunk = result.chunk
    return _normalize(
        " ".join(
            part
            for part in (
                chunk.document_title or "",
                chunk.hierarchy_path or "",
                chunk.section or "",
                chunk.subsection or "",
                chunk.text,
                chunk.embedding_text,
            )
            if part
        )
    )


def _subject_aliases(subject: str) -> tuple[str, ...]:
    normalized = _normalize(subject)
    aliases = {normalized}
    if "forced circulation" in normalized or "circulation forcee" in normalized:
        aliases.update(
            {
                "forced circulation evaporator",
                "forced circulation",
                "fc evaporator",
                "evaporateur a circulation forcee",
            }
        )
    if "falling film" in normalized or "film tombant" in normalized:
        aliases.update(
            {
                "falling film evaporator",
                "falling film",
                "evaporateur a film tombant",
                "film tombant",
            }
        )
    return tuple(alias for alias in aliases if alias)


def _supports_subject(
    candidate: HybridSearchResult,
    subject: str | None,
) -> bool:
    if not subject:
        return True
    if (candidate.chunk.chunk_type or "") in {
        "figure_caption",
        "table_of_contents",
        "index",
        "bibliography",
    }:
        return False

    text = _chunk_text(candidate)
    aliases = _subject_aliases(subject)
    if any(alias in text for alias in aliases):
        return True
    tokens = [
        token
        for token in _normalize(subject).split()
        if token not in {"a", "an", "the", "with", "and", "evaporator"}
    ]
    return bool(tokens) and all(token in text for token in tokens)


def _supports_balance_role(text: str, role_name: str) -> bool:
    marker_groups = {
        "species_conservation": (
            ("mass in", "mass out"),
            ("component balance",),
            ("species balance",),
            ("conservation law for mass",),
            ("conservation of mass",),
            ("mass balance",),
            ("material balance",),
            ("bilan de matiere",),
            ("conservation de la masse",),
            ("accumulation", "generation"),
        ),
        "species_feed": (
            ("p2o5", "feed"),
            ("p2o5", "inlet"),
            ("component", "mass in"),
        ),
        "species_product": (
            ("p2o5", "product"),
            ("p2o5", "outlet"),
            ("concentrated", "product"),
        ),
        "species_losses": (
            ("p2o5", "loss"),
            ("entrainment",),
            ("carryover",),
            ("droplet", "evaporation"),
        ),
        "energy_conservation": (
            ("energy", "in", "out"),
            ("energy balance",),
            ("enthalpy balance",),
            ("conservation of energy",),
            ("first law", "control volume"),
            ("bilan energetique",),
            ("conservation de l energie",),
            ("heat", "work", "enthalpy"),
        ),
        "heat_input": (
            ("steam", "heat"),
            ("heating medium",),
            ("heat input",),
            ("steam duty",),
        ),
        "feed_product_enthalpy": (
            ("feed", "enthalpy"),
            ("product", "enthalpy"),
            ("liquid", "enthalpy"),
        ),
        "vapor_enthalpy": (
            ("vapor", "enthalpy"),
            ("latent heat",),
            ("water evaporated", "heat"),
        ),
        "overall_conservation": (
            ("mass in", "mass out"),
            ("overall mass balance",),
            ("material balance", "accumulation"),
            ("conservation of mass",),
            ("mass balance",),
            ("bilan de matiere",),
            ("conservation de la masse",),
        ),
        "feed_stream": (("feed", "mass"), ("mass in",)),
        "product_and_vapor": (
            ("product", "vapor"),
            ("mass out",),
            ("evaporated water",),
        ),
    }
    groups = marker_groups.get(role_name)
    if not groups:
        return True
    return any(all(marker in text for marker in group) for group in groups)


def _supports_troubleshooting_role(text: str, role_name: str) -> bool:
    marker_groups = {
        "cause": (
            "cause",
            "caused by",
            "due to",
            "may be due",
            "results from",
            "provoque",
            "cause par",
            "du a",
        ),
        "mechanism": (
            "deposit",
            "accumulation",
            "precipitation",
            "corrosion",
            "resistance",
            "crystallization",
            "depot",
            "entartrage",
        ),
        "effect": (
            "decrease",
            "reduce",
            "increase",
            "performance",
            "heat transfer",
            "pressure drop",
            "capacity",
            "efficiency",
            "diminue",
            "augmente",
            "perte",
        ),
        "action": (
            "clean",
            "remove",
            "reduce",
            "eliminate",
            "wash",
            "control",
            "maintain",
            "mitigation",
            "nettoyage",
            "eliminer",
            "lavage",
        ),
    }
    markers = marker_groups.get(role_name)
    return True if markers is None else any(marker in text for marker in markers)


def _supports_role(
    candidate: HybridSearchResult,
    role: EvidenceRole,
    plan: RetrievalPlan,
) -> bool:
    if role.name not in candidate.role_matches:
        return False
    text = _chunk_text(candidate)
    if role.name in {"equipment_a", "equipment_b"}:
        return _supports_subject(candidate, role.subject)
    if plan.question_type == "balance":
        return _supports_balance_role(text, role.name)
    if plan.question_type == "troubleshooting":
        return _supports_troubleshooting_role(text, role.name)
    if role.name == "comparison_criteria":
        return any(
            marker in text
            for marker in (
                "heat transfer",
                "fouling",
                "scaling",
                "viscosity",
                "residence time",
                "circulation",
                "application",
            )
        )
    if role.name == "conical_bottom":
        return any(
            marker in text
            for marker in (
                "conical bottom",
                "cone bottom",
                "fond conique",
                "القاع المخروطي",
            )
        )
    return True


def supported_role_names(
    plan: RetrievalPlan,
    candidates: Sequence[HybridSearchResult],
) -> tuple[str, ...]:
    """Return roles explicitly supported by at least one candidate."""

    return tuple(
        role.name
        for role in plan.roles
        if any(_supports_role(candidate, role, plan) for candidate in candidates)
    )



def promote_required_roles_in_reranking(
    plan: RetrievalPlan,
    candidates: Sequence[HybridSearchResult],
    reranked_results: Sequence[RerankedSearchResult],
) -> list[RerankedSearchResult]:
    """Move the best explicit evidence for every required role to the front.

    The cross-encoder still scores every candidate once against the complete
    question. This deterministic normalization only prevents a generic passage
    from pushing an explicit equation or stream definition below the evidence
    window. It does not create evidence and it preserves the original order
    inside both the reserved and remaining groups.
    """

    candidate_by_id = {item.chunk.chunk_id: item for item in candidates}
    reserved: list[RerankedSearchResult] = []
    reserved_ids: set[str] = set()

    for role in plan.roles:
        if not role.required:
            continue
        matching = [
            item
            for item in reranked_results
            if item.chunk.chunk_id in candidate_by_id
            and _supports_role(
                candidate_by_id[item.chunk.chunk_id],
                role,
                plan,
            )
        ]
        if not matching:
            continue
        chosen = next(
            (
                item
                for item in matching
                if item.chunk.chunk_id not in reserved_ids
            ),
            matching[0],
        )
        chunk_id = chosen.chunk.chunk_id
        if chunk_id in reserved_ids:
            continue
        reserved.append(chosen)
        reserved_ids.add(chunk_id)

    ordered = [
        *reserved,
        *(
            item
            for item in reranked_results
            if item.chunk.chunk_id not in reserved_ids
        ),
    ]
    return [
        RerankedSearchResult(
            rank=rank,
            reranker_score=item.reranker_score,
            original_hybrid_rank=item.original_hybrid_rank,
            original_rrf_score=item.original_rrf_score,
            matched_retrievers=item.matched_retrievers,
            dense_rank=item.dense_rank,
            dense_score=item.dense_score,
            bm25_rank=item.bm25_rank,
            bm25_score=item.bm25_score,
            chunk=item.chunk,
            sparse_rank=item.sparse_rank,
            sparse_score=item.sparse_score,
            colbert_score=item.colbert_score,
            section_bonus=item.section_bonus,
            role_matches=item.role_matches,
        )
        for rank, item in enumerate(ordered, start=1)
    ]

def select_role_aware_evidence(
    plan: RetrievalPlan,
    candidates: Sequence[HybridSearchResult],
    reranked_results: Sequence[RerankedSearchResult],
    *,
    top_k: int,
) -> RoleEvidenceSelection:
    """Reserve evidence for every required role, then fill by reranker score."""

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
    covered: list[str] = []
    missing: list[str] = []

    for role in plan.roles:
        matching = [
            item
            for item in ordered_candidates
            if _supports_role(item, role, plan)
        ]
        if not matching:
            if role.required:
                missing.append(role.name)
            continue
        covered.append(role.name)
        chosen = next(
            (item for item in matching if item.chunk.chunk_id not in selected_ids),
            matching[0],
        )
        chunk_id = chosen.chunk.chunk_id
        if chunk_id not in selected_ids and len(selected_ids) < top_k:
            selected_ids.append(chunk_id)
            source_by_id[chunk_id] = f"evidence_role:{role.name}"

    for item in reranked_results:
        if len(selected_ids) >= top_k:
            break
        chunk_id = item.chunk.chunk_id
        if chunk_id in selected_ids:
            continue
        selected_ids.append(chunk_id)
        source_by_id[chunk_id] = "reranker_fill"

    selected = tuple(
        V3SelectedResult(
            rank=rank,
            chunk_id=chunk_id,
            source=source_by_id[chunk_id],
            reranker_rank=(
                reranked_by_id[chunk_id].rank if chunk_id in reranked_by_id else None
            ),
            hybrid_rank=candidate_by_id[chunk_id].rank,
            bm25_rank=candidate_by_id[chunk_id].bm25_rank,
        )
        for rank, chunk_id in enumerate(selected_ids, start=1)
    )
    return RoleEvidenceSelection(
        selected=selected,
        covered_roles=tuple(dict.fromkeys(covered)),
        missing_roles=tuple(dict.fromkeys(missing)),
    )
