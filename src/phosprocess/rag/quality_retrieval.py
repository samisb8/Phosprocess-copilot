"""Structure-aware production retrieval layered on the frozen v3 policy."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from phosprocess.ingestion.chunk_serialization import (
    TechnicalChunkType,
    read_child_chunks,
    read_parent_chunks,
)
from phosprocess.knowledge_base.catalog import load_document_catalog
from phosprocess.knowledge_base.schemas import KnowledgeBaseCatalog
from phosprocess.reranking.reranker import (
    RerankedSearchResult,
    RerankingResponse,
)
from phosprocess.retrieval.bge_sparse import BGESparseRetriever
from phosprocess.retrieval.context_expander import (
    ContextExpander,
    EvidenceAnchor,
)
from phosprocess.retrieval.domain_router import (
    DomainRoutingDecision,
    route_query,
)
from phosprocess.retrieval.evidence_bundle import EvidenceBundle
from phosprocess.retrieval.evidence_coverage import (
    EvidenceCoverage,
    EvidenceCoverageError,
    coverage_keys_for_text,
    evaluate_evidence_coverage_texts,
    required_evidence_keys,
    select_coverage_aware_evidence,
)
from phosprocess.retrieval.evidence_roles import (
    promote_required_roles_in_reranking,
    select_role_aware_evidence,
    supported_role_names,
)
from phosprocess.retrieval.hierarchical import (
    HierarchicalSectionRetriever,
    SectionSearchResponse,
)
from phosprocess.retrieval.hybrid import (
    HybridSearchResponse,
    HybridSearchResult,
)
from phosprocess.retrieval.quality_hybrid import (
    search_expanded_hybrid,
    search_planned_hybrid,
)
from phosprocess.retrieval.query_expansion import (
    ExpandedTechnicalQuery,
    expand_technical_query,
)
from phosprocess.retrieval.retrieval_planner import (
    RetrievalPlan,
    build_retrieval_plan,
)
from phosprocess.retrieval.source_boosting import DEFAULT_CHUNK_TYPE_BOOSTS
from phosprocess.retrieval.v3_selection import (
    V3SelectedResult,
    select_with_lexical_safeguard,
)

LOGGER = logging.getLogger(__name__)

_REQUIREMENT_QUERY_HINTS: dict[str, tuple[str, ...]] = {
    "feed_inlet": (
        "weak phosphoric acid feed enters evaporator",
        "feed inlet introduced into circulation loop",
    ),
    "conical_bottom": (
        "cycling acid leaves vapor body through conical bottom",
        "evaporator vapor body conical bottom recirculating acid outlet",
    ),
    "pump_heat_exchanger": (
        "circulation pump sends acid through heat exchanger",
        "forced circulation heater loop",
    ),
    "vapor_body": (
        "acid enters vapor body flash chamber",
        "vapor liquid separation evaporator body",
    ),
    "recirculation": (
        "liquid returns through recirculation line",
        "acid recycled back to flash chamber",
    ),
    "product_outlet": (
        "concentrated phosphoric acid product is withdrawn from evaporator",
        "product acid outlet draw-off sent to storage",
    ),
    "overall_mass_balance": (
        "overall material balance feed product accumulation steady state",
    ),
    "p2o5_balance": (
        "P2O5 component balance feed product steady state",
    ),
    "energy_balance": (
        "evaporator energy heat enthalpy balance steam",
    ),
    "p2o5_conservation": (
        "JFC4 bilan matière P2O5 ligne 1 ligne 5 ligne 6",
    ),
    "p2o5_feed": (
        "ligne 1 entrée acide débit P2O5 alimentation",
    ),
    "p2o5_product": (
        "ligne 5 sortie acide débit P2O5 produit",
    ),
    "p2o5_entrainment": (
        "ligne 6 sortie bouilleur P2O5 entraîné gaz",
    ),
}

_QUESTION_TYPE_CHUNKS: dict[str, frozenset[TechnicalChunkType]] = {
    "definition": frozenset({TechnicalChunkType.DEFINITION}),
    "process_flow": frozenset(
        {
            TechnicalChunkType.PROCESS_DESCRIPTION,
            TechnicalChunkType.EQUIPMENT_DESCRIPTION,
        }
    ),
    "procedure": frozenset({TechnicalChunkType.PROCEDURE}),
    "balance": frozenset(
        {
            TechnicalChunkType.BALANCE,
            TechnicalChunkType.EQUATION,
            TechnicalChunkType.EQUATION_EXPLANATION,
            TechnicalChunkType.TABLE,
        }
    ),
    "equation_explanation": frozenset(
        {
            TechnicalChunkType.EQUATION,
            TechnicalChunkType.EQUATION_EXPLANATION,
        }
    ),
    "thermodynamic_relation": frozenset(
        {
            TechnicalChunkType.EQUATION,
            TechnicalChunkType.EQUATION_EXPLANATION,
        }
    ),
    "calculation": frozenset(
        {
            TechnicalChunkType.EQUATION,
            TechnicalChunkType.WORKED_EXAMPLE,
        }
    ),
    "table_question": frozenset({TechnicalChunkType.TABLE}),
    "troubleshooting": frozenset(
        {
            TechnicalChunkType.TROUBLESHOOTING,
            TechnicalChunkType.OPERATING_PROBLEM,
            TechnicalChunkType.EQUIPMENT_DESCRIPTION,
            TechnicalChunkType.NARRATIVE,
        }
    ),
    "control_strategy": frozenset(
        {TechnicalChunkType.CONTROL_STRATEGY}
    ),
    "momentum_diffusion": frozenset(
        {
            TechnicalChunkType.DEFINITION,
            TechnicalChunkType.EQUATION,
            TechnicalChunkType.EQUATION_EXPLANATION,
            TechnicalChunkType.NARRATIVE,
        }
    ),
    "safety": frozenset({TechnicalChunkType.SAFETY}),
}


@dataclass(frozen=True, slots=True)
class QualityRetrievalResult:
    """Complete trace of one quality retrieval turn."""

    query: ExpandedTechnicalQuery
    routing: DomainRoutingDecision
    hybrid: HybridSearchResponse
    reranking: RerankingResponse
    selected: tuple[V3SelectedResult, ...]
    bundles: tuple[EvidenceBundle, ...]
    source_boosts: dict[str, float]
    coverage: EvidenceCoverage
    section_search: SectionSearchResponse | None = None
    retrieval_plan: RetrievalPlan | None = None
    covered_roles: tuple[str, ...] = ()
    missing_roles: tuple[str, ...] = ()


class QualityRetrievalEngine:
    """Retrieve globally, apply small boosts, then expand final evidence."""

    def __init__(
        self,
        *,
        version_directory: Path,
        retriever: Any,
        reranker: Any,
        catalog: KnowledgeBaseCatalog | None = None,
        require_sparse_index: bool = False,
    ) -> None:
        self.version_directory = version_directory.resolve()
        self.retriever = retriever
        self.reranker = reranker
        self.catalog = catalog or load_document_catalog()
        self.require_sparse_index = require_sparse_index
        self.children = read_child_chunks(self.version_directory / "chunks.jsonl")
        self.parents = read_parent_chunks(self.version_directory / "parents.jsonl")
        self.child_by_id = {child.chunk_id: child for child in self.children}
        self.expander = ContextExpander(
            children=self.children,
            parents=self.parents,
        )
        self.section_retriever = (
            HierarchicalSectionRetriever(
                version_directory=self.version_directory,
                query_embedder=self.retriever.dense_retriever.embedder,
                rrf_k=self.retriever.config.rrf_k,
            )
            if HierarchicalSectionRetriever.is_available(self.version_directory)
            else None
        )
        self.sparse_retriever = (
            BGESparseRetriever(
                version_directory=self.version_directory,
                embedder=self.retriever.dense_retriever.embedder,
                metadata=self.retriever.dense_retriever.metadata,
            )
            if BGESparseRetriever.is_available(self.version_directory)
            else None
        )
        if self.sparse_retriever is None:
            message = (
                "Index BGE sparse absent. Exécutez "
                "scripts/build_bge_sparse_index.py avant le retriever v4."
            )
            if self.require_sparse_index:
                raise FileNotFoundError(message)
            LOGGER.warning("%s Version=%s", message, self.version_directory.name)

    @staticmethod
    def is_quality_index(version_directory: Path) -> bool:
        return (
            version_directory.name.startswith("kb_quality_")
            and (version_directory / "chunks.jsonl").is_file()
            and (version_directory / "parents.jsonl").is_file()
        )

    @staticmethod
    def _adjust_reranking(
        response: RerankingResponse,
        *,
        routing: DomainRoutingDecision,
        question_type: str | None = None,
    ) -> tuple[RerankingResponse, dict[str, float]]:
        adjusted: list[tuple[RerankedSearchResult, float, float]] = []
        source_boost_by_id: dict[str, float] = {}

        for result in response.results:
            raw_chunk_type = result.chunk.chunk_type or "narrative"

            try:
                chunk_type = TechnicalChunkType(raw_chunk_type)
            except ValueError:
                chunk_type = TechnicalChunkType.NARRATIVE

            source_boost = routing.soft_boosts.get(result.chunk.document_id, 0.0)
            type_boost = DEFAULT_CHUNK_TYPE_BOOSTS.get(chunk_type, 0.0)
            heading = " ".join(
                str(value)
                for value in (
                    getattr(result.chunk, "chapter", ""),
                    getattr(result.chunk, "section", ""),
                    getattr(result.chunk, "subsection", ""),
                    getattr(result.chunk, "hierarchy_path", ""),
                )
                if value
            )
            decomposed_heading = unicodedata.normalize(
                "NFKD",
                heading.casefold(),
            )
            normalized_heading = re.sub(
                r"[^a-z0-9%]+",
                " ",
                "".join(
                    character
                    for character in decomposed_heading
                    if not unicodedata.combining(character)
                ),
            ).strip()
            section_boost = (
                0.025
                if normalized_heading
                and any(
                    term in normalized_heading
                    for term in routing.section_affinity_terms
                )
                else 0.0
            )
            preferred = _QUESTION_TYPE_CHUNKS.get(
                question_type or "",
                frozenset(),
            )
            intent_boost = 0.06 if chunk_type in preferred else 0.0
            penalty_types = {
                "process_flow": {
                    TechnicalChunkType.EQUATION,
                    TechnicalChunkType.WORKED_EXAMPLE,
                    TechnicalChunkType.SIMULATION_RESULTS,
                    TechnicalChunkType.TABLE,
                },
                "procedure": {
                    TechnicalChunkType.EQUATION,
                    TechnicalChunkType.WORKED_EXAMPLE,
                    TechnicalChunkType.SIMULATION_RESULTS,
                },
                "balance": {
                    TechnicalChunkType.SIMULATION_RESULTS,
                    TechnicalChunkType.FIGURE_CAPTION,
                },
                "troubleshooting": {
                    TechnicalChunkType.SIMULATION_RESULTS,
                    TechnicalChunkType.WORKED_EXAMPLE,
                },
                "definition": {
                    TechnicalChunkType.SIMULATION_RESULTS,
                    TechnicalChunkType.WORKED_EXAMPLE,
                },
            }
            intent_penalty = (
                -0.08
                if chunk_type in penalty_types.get(question_type or "", set())
                else 0.0
            )
            total_boost = (
                source_boost
                + type_boost
                + section_boost
                + intent_boost
                + intent_penalty
            )
            source_boost_by_id[result.chunk.chunk_id] = source_boost
            adjusted.append(
                (
                    result,
                    float(result.reranker_score) + total_boost,
                    total_boost,
                )
            )

        adjusted.sort(
            key=lambda item: (
                -item[1],
                item[0].rank,
                item[0].chunk.chunk_id,
            )
        )
        results = [
            replace(
                result,
                rank=rank,
                reranker_score=adjusted_score,
            )
            for rank, (result, adjusted_score, _boost) in enumerate(
                adjusted,
                start=1,
            )
        ]
        return replace(response, results=results), source_boost_by_id

    @staticmethod
    def _filter_process_scope_incompatibilities(
        response: RerankingResponse,
        *,
        plan: RetrievalPlan,
    ) -> RerankingResponse:
        """Remove passages that explicitly belong to refrigeration service.

        Chemical-process evaporators and refrigeration evaporators share the
        same equipment name. When the resolved query is clearly about an acid
        concentrator, a passage dominated by refrigerant, air-cooler or vapor-
        compression terminology is scope-incompatible unless the user asked
        about refrigeration.
        """

        query = re.sub(r"\s+", " ", plan.base_query.casefold())
        refrigeration_query_markers = (
            "refrigerant",
            "refrigeration",
            "chiller",
            "air cooler",
            "vapor compression system",
        )
        process_query_markers = (
            "phosphoric acid",
            "forced circulation",
            "circulation pump",
            "flash chamber",
            "vapor body",
            "acid concentration",
        )
        if any(marker in query for marker in refrigeration_query_markers):
            return response
        if not any(marker in query for marker in process_query_markers):
            return response

        strong_section_markers = (
            "mechanical refrigeration",
            "refrigeration evaporator",
            "air cooling",
            "air-cooling",
            "liquid chiller",
            "vapor compression system",
        )
        process_passage_markers = (
            "phosphoric acid",
            "forced-circulation evaporator",
            "forced circulation evaporator",
            "flash chamber",
            "circulating liquor",
            "vapor body",
        )

        retained: list[RerankedSearchResult] = []
        for item in response.results:
            chunk = item.chunk
            heading = " ".join(
                part
                for part in (
                    chunk.section or "",
                    chunk.subsection or "",
                    chunk.hierarchy_path or "",
                )
                if part
            ).casefold()
            passage = re.sub(
                r"\s+",
                " ",
                f"{heading} {chunk.text}".casefold(),
            )
            strong_scope_mismatch = any(
                marker in heading for marker in strong_section_markers
            )
            refrigerant_dominated = (
                passage.count("refrigerant") >= 2
                or "halocarbon" in passage
                or "compressor crankcase" in passage
            )
            explicit_process_context = any(
                marker in passage for marker in process_passage_markers
            )
            if (strong_scope_mismatch or refrigerant_dominated) and not (
                explicit_process_context
            ):
                continue
            retained.append(item)

        return replace(
            response,
            results=[
                replace(item, rank=rank)
                for rank, item in enumerate(retained, start=1)
            ],
        )

    @staticmethod
    def _process_flow_marker_count(text: str) -> int:
        normalized = re.sub(r"\s+", " ", text.casefold())
        marker_groups = (
            ("feed", "inlet", "alimentation", "entrée"),
            ("pump", "pomp", "circulation"),
            ("heat exchanger", "heating element", "échangeur", "chauffage"),
            ("flash chamber", "vapor body", "chambre de flash", "bouilleur"),
            ("return", "returned", "recirculation", "recyclage", "retour"),
            ("outlet", "product", "sortie", "soutirage"),
        )
        return sum(
            any(term in normalized for term in group)
            for group in marker_groups
        )

    @classmethod
    def _repair_weak_lexical_selection(
        cls,
        selected: list[V3SelectedResult],
        *,
        candidates: tuple[Any, ...] | list[Any],
        reranked_results: list[RerankedSearchResult],
        question_type: str,
        top_k: int,
    ) -> list[V3SelectedResult]:
        """Keep the frozen selection, but replace clearly unusable lexical slots.

        The frozen v3 safeguard intentionally reserves one BM25 slot. In a much
        larger eight-book corpus, a rank-1 lexical hit can still be semantically
        unrelated. We preserve the safeguard unless its reranker score is very
        low, or a process-flow lexical hit contains no actual flow evidence.
        """

        candidate_by_id = {item.chunk.chunk_id: item for item in candidates}
        reranked_by_id = {
            item.chunk.chunk_id: item
            for item in reranked_results
        }
        best_score = max(
            (float(item.reranker_score) for item in reranked_results),
            default=0.0,
        )
        minimum_score = max(0.12, best_score * 0.15)
        retained: list[V3SelectedResult] = []

        for item in selected:
            reranked = reranked_by_id.get(item.chunk_id)
            weak_score = (
                item.source == "bm25_safeguard"
                and (reranked is None or reranked.reranker_score < minimum_score)
            )
            weak_flow_evidence = False

            if (
                item.source == "bm25_safeguard"
                and question_type == "process_flow"
            ):
                candidate = candidate_by_id[item.chunk_id]
                weak_flow_evidence = (
                    cls._process_flow_marker_count(candidate.chunk.text) < 2
                )

            if not weak_score and not weak_flow_evidence:
                retained.append(item)

        retained_ids = {item.chunk_id for item in retained}

        for reranked in reranked_results:
            if len(retained) >= top_k:
                break

            chunk_id = reranked.chunk.chunk_id

            if chunk_id in retained_ids:
                continue

            candidate = candidate_by_id[chunk_id]
            retained.append(
                V3SelectedResult(
                    rank=len(retained) + 1,
                    chunk_id=chunk_id,
                    source="reranker_fill_quality",
                    reranker_rank=reranked.rank,
                    hybrid_rank=candidate.rank,
                    bm25_rank=candidate.bm25_rank,
                )
            )
            retained_ids.add(chunk_id)

        return [replace(item, rank=rank) for rank, item in enumerate(retained, 1)]

    def _diversify_selected(
        self,
        selected: list[V3SelectedResult],
        *,
        candidates: tuple[Any, ...] | list[Any],
        reranked_results: list[RerankedSearchResult],
        top_k: int,
    ) -> list[V3SelectedResult]:
        """Prefer complementary sections and avoid five near-duplicate anchors."""

        if not all(self.child_by_id[item.chunk_id].section_id for item in selected):
            return selected

        candidate_by_id = {item.chunk.chunk_id: item for item in candidates}
        retained: list[V3SelectedResult] = []
        retained_ids: set[str] = set()
        section_counts: dict[str, int] = {}
        document_counts: dict[str, int] = {}

        def can_add(chunk_id: str) -> bool:
            child = self.child_by_id[chunk_id]
            return (
                section_counts.get(child.section_id, 0) < 2
                and document_counts.get(child.document_id, 0) < 3
            )

        def add(item: V3SelectedResult) -> None:
            child = self.child_by_id[item.chunk_id]
            retained.append(item)
            retained_ids.add(item.chunk_id)
            section_counts[child.section_id] = section_counts.get(child.section_id, 0) + 1
            document_counts[child.document_id] = (
                document_counts.get(child.document_id, 0) + 1
            )

        for item in selected:
            if can_add(item.chunk_id):
                add(item)

        for reranked in reranked_results:
            if len(retained) >= top_k:
                break
            chunk_id = reranked.chunk.chunk_id
            if chunk_id in retained_ids or not can_add(chunk_id):
                continue
            candidate = candidate_by_id[chunk_id]
            add(
                V3SelectedResult(
                    rank=len(retained) + 1,
                    chunk_id=chunk_id,
                    source="hierarchical_diversity_fill",
                    reranker_rank=reranked.rank,
                    hybrid_rank=candidate.rank,
                    bm25_rank=candidate.bm25_rank,
                )
            )

        if len(retained) < top_k:
            for reranked in reranked_results:
                if len(retained) >= top_k:
                    break
                chunk_id = reranked.chunk.chunk_id
                if chunk_id in retained_ids:
                    continue
                candidate = candidate_by_id[chunk_id]
                add(
                    V3SelectedResult(
                        rank=len(retained) + 1,
                        chunk_id=chunk_id,
                        source="hierarchical_fill",
                        reranker_rank=reranked.rank,
                        hybrid_rank=candidate.rank,
                        bm25_rank=candidate.bm25_rank,
                    )
                )

        return [replace(item, rank=rank) for rank, item in enumerate(retained, 1)]

    def _coverage_text_for_chunk(
        self,
        chunk_id: str,
        *,
        question_type: str,
    ) -> str:
        """Build the text window used only by the deterministic coverage guard."""

        child = self.child_by_id[chunk_id]
        pieces = [child.hierarchy_path, child.display_text]

        if question_type in {"process_flow", "procedure", "balance"}:
            for neighbor_id in (
                child.previous_chunk_id,
                child.next_chunk_id,
            ):
                if not neighbor_id:
                    continue

                neighbor = self.child_by_id.get(neighbor_id)

                if neighbor is not None and neighbor.document_id == child.document_id:
                    pieces.append(neighbor.display_text)

        return "\n".join(dict.fromkeys(piece for piece in pieces if piece))

    def _coverage_texts_for_results(
        self,
        results: list[HybridSearchResult],
        *,
        question_type: str,
    ) -> dict[str, str]:
        return {
            result.chunk.chunk_id: self._coverage_text_for_chunk(
                result.chunk.chunk_id,
                question_type=question_type,
            )
            for result in results
        }

    @staticmethod
    def _requirement_query(
        expanded: ExpandedTechnicalQuery,
        requirement: str,
    ) -> ExpandedTechnicalQuery:
        hints = _REQUIREMENT_QUERY_HINTS.get(requirement, (requirement,))
        dense_hint = hints[0]
        lexical_hints = " ".join(hints)

        return ExpandedTechnicalQuery(
            original_query=expanded.original_query,
            standalone_query=expanded.standalone_query,
            dense_query=f"{expanded.standalone_query} {dense_hint}",
            bm25_expanded_query=(
                f"{expanded.bm25_expanded_query} {lexical_hints}"
            ),
            added_terms=tuple(dict.fromkeys((*expanded.added_terms, *hints))),
        )

    @staticmethod
    def _merge_coverage_candidates(
        primary: HybridSearchResponse,
        recovered: list[HybridSearchResult],
        *,
        candidate_k: int,
        extra_duration_ms: float,
        recovery_tag: str = "coverage_recovery",
    ) -> HybridSearchResponse:
        """Reserve candidate slots for proven missing-aspect passages."""

        recovered_by_id = {
            result.chunk.chunk_id: replace(
                result,
                matched_retrievers=tuple(
                    dict.fromkeys(
                        (*result.matched_retrievers, recovery_tag)
                    )
                ),
            )
            for result in recovered
        }
        reserved = list(recovered_by_id.values())[: min(6, candidate_k)]
        reserved_ids = {result.chunk.chunk_id for result in reserved}
        retained_primary = [
            result
            for result in primary.results
            if result.chunk.chunk_id not in reserved_ids
        ][: max(0, candidate_k - len(reserved))]
        merged = [*retained_primary, *reserved]

        if len(merged) < candidate_k:
            existing = {result.chunk.chunk_id for result in merged}

            for result in primary.results:
                if result.chunk.chunk_id in existing:
                    continue

                merged.append(result)
                existing.add(result.chunk.chunk_id)

                if len(merged) >= candidate_k:
                    break

        ranked = [
            replace(result, rank=rank)
            for rank, result in enumerate(merged[:candidate_k], start=1)
        ]

        return replace(
            primary,
            fusion_candidates=max(primary.fusion_candidates, len(ranked)),
            total_duration_ms=round(
                primary.total_duration_ms + extra_duration_ms,
                3,
            ),
            results=ranked,
        )

    def _recover_missing_coverage_candidates(
        self,
        primary: HybridSearchResponse,
        *,
        expanded: ExpandedTechnicalQuery,
        routing: DomainRoutingDecision,
        question_type: str,
        candidate_k: int,
    ) -> HybridSearchResponse:
        """Run targeted global retrieval only for aspects absent from candidates."""

        required = required_evidence_keys(question_type)

        if not required:
            return primary

        coverage_texts = self._coverage_texts_for_results(
            primary.results,
            question_type=question_type,
        )
        coverage = evaluate_evidence_coverage_texts(
            list(coverage_texts.items()),
            question_type=question_type,
        )

        if coverage.complete:
            return primary

        recovered: list[HybridSearchResult] = []
        recovered_ids: set[str] = set()
        extra_duration_ms = 0.0
        document_ids = (
            set(routing.hard_filter)
            if routing.hard_filter is not None
            else None
        )

        for requirement in coverage.missing:
            targeted_query = self._requirement_query(expanded, requirement)
            targeted = search_expanded_hybrid(
                self.retriever,
                targeted_query,
                top_k=min(8, candidate_k),
                dense_candidate_k=min(12, candidate_k),
                bm25_candidate_k=min(12, candidate_k),
                sparse_retriever=self.sparse_retriever,
                sparse_candidate_k=min(12, candidate_k),
                document_ids=document_ids,
                chunk_ids=None,
            )
            extra_duration_ms += targeted.total_duration_ms
            accepted = 0

            for result in targeted.results:
                chunk_id = result.chunk.chunk_id

                if chunk_id in recovered_ids or chunk_id not in self.child_by_id:
                    continue

                text = self._coverage_text_for_chunk(
                    chunk_id,
                    question_type=question_type,
                )

                if requirement not in coverage_keys_for_text(
                    text,
                    question_type,
                ):
                    continue

                recovered.append(result)
                recovered_ids.add(chunk_id)
                accepted += 1

                if accepted >= 2:
                    break

        LOGGER.info(
            "Coverage recovery question_type=%s missing=%s recovered=%s",
            question_type,
            ",".join(coverage.missing),
            ",".join(result.chunk.chunk_id for result in recovered) or "none",
        )

        if not recovered:
            return primary

        return self._merge_coverage_candidates(
            primary,
            recovered,
            candidate_k=candidate_k,
            extra_duration_ms=extra_duration_ms,
        )

    def _recover_missing_role_candidates(
        self,
        primary: HybridSearchResponse,
        *,
        plan: RetrievalPlan,
        routing: DomainRoutingDecision,
        candidate_k: int,
    ) -> HybridSearchResponse:
        """Recover explicit role evidence before reranking and generation.

        Multi-query fusion may rank a generic passage above a narrow
        conservation equation.  This recovery searches only roles that are
        still unsupported by the current cross-encoder window and reserves a
        small number of explicitly verified candidates for those roles.
        """

        supported = set(supported_role_names(plan, primary.results))
        missing_roles = [
            role
            for role in plan.roles
            if role.required and role.name not in supported
        ]
        if not missing_roles:
            return primary

        document_ids = (
            set(routing.hard_filter)
            if routing.hard_filter is not None
            else None
        )
        recovered: list[HybridSearchResult] = []
        recovered_ids: set[str] = set()
        extra_duration_ms = 0.0

        for role in missing_roles:
            expanded = expand_technical_query(
                role.query,
                standalone_query=role.query,
                question_type=plan.question_type,
            )
            targeted = search_expanded_hybrid(
                self.retriever,
                expanded,
                top_k=min(12, candidate_k),
                dense_candidate_k=max(20, min(50, candidate_k * 2)),
                bm25_candidate_k=max(20, min(50, candidate_k * 2)),
                sparse_retriever=self.sparse_retriever,
                sparse_candidate_k=max(20, min(50, candidate_k * 2)),
                document_ids=document_ids,
                chunk_ids=None,
                role_name=role.name,
            )
            extra_duration_ms += targeted.total_duration_ms
            accepted = 0

            for result in targeted.results:
                chunk_id = result.chunk.chunk_id
                if chunk_id in recovered_ids:
                    continue
                if role.name not in supported_role_names(plan, [result]):
                    continue
                recovered.append(result)
                recovered_ids.add(chunk_id)
                accepted += 1
                if accepted >= 2:
                    break

        LOGGER.info(
            "Role recovery question_type=%s missing=%s recovered=%s",
            plan.question_type,
            ",".join(role.name for role in missing_roles),
            ",".join(result.chunk.chunk_id for result in recovered) or "none",
        )
        if not recovered:
            return primary
        return self._merge_coverage_candidates(
            primary,
            recovered,
            candidate_k=candidate_k,
            extra_duration_ms=extra_duration_ms,
            recovery_tag="role_recovery",
        )

    @staticmethod
    def _section_bonus_by_chunk(
        response: SectionSearchResponse | None,
    ) -> dict[str, float]:
        """Convert hierarchy results into a small rank bonus, never a filter."""

        if response is None:
            return {}
        bonuses: dict[str, float] = {}
        for result in response.results:
            rank_bonus = max(0.0005, 0.006 - 0.0005 * (result.rank - 1))
            for chunk_id in result.section.child_chunk_ids:
                bonuses[chunk_id] = max(bonuses.get(chunk_id, 0.0), rank_bonus)
        return bonuses

    def retrieve(
        self,
        original_question: str,
        *,
        standalone_query: str,
        question_type: str,
        source_mode: str = "auto",
        candidate_k: int = 20,
        dense_candidate_k: int = 20,
        bm25_candidate_k: int = 20,
        top_k: int = 5,
        lexical_slots: int = 1,
    ) -> QualityRetrievalResult:
        """Run retriever v4 without hard section gating.

        Hierarchy contributes only a small score bonus. Every evidence role is
        searched globally with dense, BGE sparse and BM25 retrieval before
        dynamic ColBERT scoring and cross-encoder reranking.
        """

        if self.require_sparse_index and self.sparse_retriever is None:
            raise FileNotFoundError(
                "Retriever v4 incomplet : index BGE sparse absent. "
                "Exécutez scripts/build_bge_sparse_index.py avant la recherche."
            )

        plan = build_retrieval_plan(
            original_question,
            standalone_query=standalone_query,
            question_type=question_type,
        )
        routing = route_query(
            plan.base_query,
            catalog=self.catalog,
            source_mode=source_mode,
            question_type=question_type,
        )
        expanded = expand_technical_query(
            original_question,
            standalone_query=plan.base_query,
            question_type=question_type,
        )
        section_search = (
            self.section_retriever.search(
                expanded,
                question_type=question_type,
                routing=routing,
                top_k=12,
                candidate_k=40,
            )
            if self.section_retriever is not None
            else None
        )
        section_bonus_by_chunk = self._section_bonus_by_chunk(section_search)
        document_ids = (
            set(routing.hard_filter)
            if routing.hard_filter is not None
            else None
        )
        cross_encoder_k = max(30, candidate_k, top_k)
        structured_roles = (
            question_type
            in {
                "comparison",
                "balance",
                "troubleshooting",
                "momentum_diffusion",
            }
            or (
                question_type in {"definition", "explanation"}
                and len(plan.roles) > 1
            )
        )
        hybrid = search_planned_hybrid(
            self.retriever,
            plan,
            sparse_retriever=self.sparse_retriever,
            top_k=cross_encoder_k,
            dense_candidate_k=max(50, dense_candidate_k),
            sparse_candidate_k=50,
            bm25_candidate_k=max(50, bm25_candidate_k),
            fusion_k=80,
            colbert_candidate_k=80,
            document_ids=document_ids,
            section_bonus_by_chunk=section_bonus_by_chunk,
        )
        if question_type == "process_flow":
            hybrid = self._recover_missing_coverage_candidates(
                hybrid,
                expanded=expanded,
                routing=routing,
                question_type=question_type,
                candidate_k=cross_encoder_k,
            )
        if structured_roles:
            hybrid = self._recover_missing_role_candidates(
                hybrid,
                plan=plan,
                routing=routing,
                candidate_k=cross_encoder_k,
            )

        if len(hybrid.results) < top_k:
            raise ValueError(
                "Le retriever v4 n'a pas produit assez de candidats globaux "
                f"({len(hybrid.results)} < {top_k})."
            )

        raw_reranking = self.reranker.rerank(
            plan.base_query,
            list(hybrid.results),
            top_k=min(cross_encoder_k, len(hybrid.results)),
        )
        reranking, source_boosts = self._adjust_reranking(
            raw_reranking,
            routing=routing,
            question_type=question_type,
        )
        reranking = self._filter_process_scope_incompatibilities(
            reranking,
            plan=plan,
        )
        if structured_roles:
            reranking = replace(
                reranking,
                results=promote_required_roles_in_reranking(
                    plan,
                    hybrid.results,
                    reranking.results,
                ),
            )
        covered_roles: tuple[str, ...] = ()
        missing_roles: tuple[str, ...] = ()

        if structured_roles:
            role_selection = select_role_aware_evidence(
                plan,
                hybrid.results,
                reranking.results,
                top_k=top_k,
            )
            selected = list(role_selection.selected)
            covered_roles = role_selection.covered_roles
            missing_roles = role_selection.missing_roles
            if missing_roles:
                raise EvidenceCoverageError(
                    "Preuves par rôle incomplètes avant génération : "
                    + ", ".join(missing_roles)
                )
            coverage = EvidenceCoverage(
                question_type=question_type,
                required=tuple(role.name for role in plan.roles if role.required),
                covered=covered_roles,
                missing=missing_roles,
                chunk_ids_by_requirement={},
            )
        else:
            selected = select_with_lexical_safeguard(
                hybrid.results,
                reranking.results,
                top_k=top_k,
                lexical_slots=lexical_slots,
            )
            selected = self._repair_weak_lexical_selection(
                selected,
                candidates=hybrid.results,
                reranked_results=reranking.results,
                question_type=question_type,
                top_k=top_k,
            )
            if question_type == "process_flow":
                coverage_text_by_id = self._coverage_texts_for_results(
                    hybrid.results,
                    question_type=question_type,
                )
                selected, coverage = select_coverage_aware_evidence(
                    selected,
                    candidates=hybrid.results,
                    reranked_results=reranking.results,
                    child_by_id=self.child_by_id,
                    question_type=question_type,
                    top_k=top_k,
                    coverage_text_by_id=coverage_text_by_id,
                )
            else:
                coverage = EvidenceCoverage(
                    question_type=question_type,
                    required=(),
                    covered=(),
                    missing=(),
                    chunk_ids_by_requirement={},
                )

        reranked_by_id = {
            result.chunk.chunk_id: result
            for result in reranking.results
        }
        anchors: list[EvidenceAnchor] = []
        for selection in selected:
            child = self.child_by_id.get(selection.chunk_id)
            if child is None:
                raise ValueError(
                    f"Chunk qualité introuvable : {selection.chunk_id}"
                )
            reranked = reranked_by_id.get(selection.chunk_id)
            anchors.append(
                EvidenceAnchor(
                    child=child,
                    score=(
                        reranked.reranker_score
                        if reranked is not None
                        else 0.0
                    ),
                    provenance=selection.source,
                )
            )

        bundles = self.expander.expand(
            anchors,
            question_type=question_type,
        )
        if len(bundles) != top_k:
            raise ValueError(
                f"L'expansion qualité doit produire {top_k} evidence bundles."
            )

        LOGGER.info(
            "RETRIEVER_V4 type=%s roles=%s missing_roles=%s "
            "dense_found=%s sparse_found=%s bm25_found=%s fusion=%s reranked=%s",
            question_type,
            ",".join(role.name for role in plan.roles),
            ",".join(missing_roles) or "none",
            hybrid.dense_results_found,
            hybrid.sparse_results_found,
            hybrid.bm25_results_found,
            hybrid.fusion_candidates,
            len(reranking.results),
        )
        return QualityRetrievalResult(
            query=expanded,
            routing=routing,
            hybrid=hybrid,
            reranking=reranking,
            selected=tuple(selected),
            bundles=tuple(bundles),
            source_boosts=source_boosts,
            coverage=coverage,
            section_search=section_search,
            retrieval_plan=plan,
            covered_roles=covered_roles,
            missing_roles=missing_roles,
        )
