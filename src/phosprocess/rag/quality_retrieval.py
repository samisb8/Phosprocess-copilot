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
from phosprocess.retrieval.document_selector import RankedDocument, rank_documents
from phosprocess.retrieval.domain_router import (
    DomainRoutingDecision,
    route_query,
)
from phosprocess.retrieval.evidence_bundle import EvidenceBundle
from phosprocess.retrieval.evidence_roles import select_role_aware_evidence
from phosprocess.retrieval.hierarchical import (
    HierarchicalSectionRetriever,
    SectionSearchResponse,
)
from phosprocess.retrieval.hybrid import (
    HybridSearchResponse,
)
from phosprocess.retrieval.quality_hybrid import (
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


_REQUIREMENT_QUERY_HINTS: dict[str, tuple[str, ...]] = {}

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
    "control_strategy": frozenset({TechnicalChunkType.CONTROL_STRATEGY}),
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
    section_search: SectionSearchResponse | None = None
    retrieval_plan: RetrievalPlan | None = None
    covered_roles: tuple[str, ...] = ()


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
        dense_embedder = getattr(self.retriever.dense_retriever, "embedder", None)
        self.expander = ContextExpander(
            children=self.children,
            parents=self.parents,
            token_counter=getattr(dense_embedder, "count_tokens", None),
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
                and any(term in normalized_heading for term in routing.section_affinity_terms)
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
                -0.08 if chunk_type in penalty_types.get(question_type or "", set()) else 0.0
            )
            total_boost = source_boost + type_boost + section_boost + intent_boost + intent_penalty
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
        """Replace only a clearly weak lexical safeguard.

        The decision depends on retrieval scores, never on expected
        domain-specific answer content.
        """

        del cls
        del question_type

        candidate_by_id = {item.chunk.chunk_id: item for item in candidates}
        reranked_by_id = {item.chunk.chunk_id: item for item in reranked_results}

        best_score = max(
            (float(item.reranker_score) for item in reranked_results),
            default=0.0,
        )

        minimum_score = max(
            0.12,
            best_score * 0.15,
        )

        retained: list[V3SelectedResult] = []

        for item in selected:
            reranked = reranked_by_id.get(item.chunk_id)

            weak = item.source == "bm25_safeguard" and (
                reranked is None or reranked.reranker_score < minimum_score
            )

            if not weak:
                retained.append(item)

        retained_ids = {item.chunk_id for item in retained}

        for reranked in reranked_results:
            if len(retained) >= top_k:
                break

            chunk_id = reranked.chunk.chunk_id

            if chunk_id in retained_ids or chunk_id not in candidate_by_id:
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

        return [
            replace(
                item,
                rank=rank,
            )
            for rank, item in enumerate(
                retained,
                start=1,
            )
        ]

    def _diversify_selected(
        self,
        selected: list[V3SelectedResult],
        *,
        candidates: tuple[Any, ...] | list[Any],
        reranked_results: list[RerankedSearchResult],
        top_k: int,
    ) -> list[V3SelectedResult]:
        """Prefer complementary sections and avoid near-duplicate anchors."""

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
            document_counts[child.document_id] = document_counts.get(child.document_id, 0) + 1

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

    def discover_documents(
        self,
        original_question: str,
        *,
        standalone_query: str,
        question_type: str,
        candidate_k: int = 20,
        dense_candidate_k: int = 20,
        bm25_candidate_k: int = 20,
    ) -> tuple[RankedDocument, ...]:
        """Run global evidence discovery before any automatic source lock."""

        if self.require_sparse_index and self.sparse_retriever is None:
            raise FileNotFoundError("Retriever v4 incomplet : index BGE sparse absent.")

        plan = build_retrieval_plan(
            original_question,
            standalone_query=standalone_query,
            question_type=question_type,
        )

        # Domain understanding is retained for telemetry and query semantics,
        # but automatic routing is forbidden from selecting a document.
        # Source identity is a user-control signal.  Query planning may append
        # generic technical aliases that happen to match a catalog alias, so
        # explicit-source detection must inspect the original user wording.
        routing = route_query(
            original_question,
            catalog=self.catalog,
            source_mode="auto",
            question_type=question_type,
        )

        discovery_k = max(
            40,
            candidate_k,
        )

        hybrid = search_planned_hybrid(
            self.retriever,
            plan,
            sparse_retriever=self.sparse_retriever,
            top_k=discovery_k,
            dense_candidate_k=max(
                50,
                dense_candidate_k,
            ),
            sparse_candidate_k=50,
            bm25_candidate_k=max(
                50,
                bm25_candidate_k,
            ),
            fusion_k=80,
            colbert_candidate_k=80,
            document_ids=None,
            section_bonus_by_chunk=None,
        )

        if not hybrid.results:
            raise ValueError(
                "La découverte documentaire globale n'a produit aucun candidat."
            )

        reranking = self.reranker.rerank(
            plan.base_query,
            list(hybrid.results),
            top_k=min(
                discovery_k,
                len(hybrid.results),
            ),
        )

        ranking = rank_documents(
            reranked_results=reranking.results,
            hybrid_results=hybrid.results,
        )

        if not ranking:
            raise ValueError(
                "Aucun document n'a pu être classé après la recherche globale."
            )

        LOGGER.info(
            "DOCUMENT_DISCOVERY type=%s domains=%s ranking=%s",
            question_type,
            ",".join(domain.value for domain, _score in routing.detected_domains),
            " | ".join(
                (
                    f"{item.document_id}:"
                    f"rr={item.reranker_reciprocal_rank_sum:.4f},"
                    f"best={item.best_reranker_rank},"
                    f"hits={item.reranker_hits}"
                )
                for item in ranking
            ),
        )

        return ranking

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
        document_ids: set[str] | None = None,
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
            original_question,
            catalog=self.catalog,
            source_mode=source_mode,
            question_type=question_type,
        )

        if document_ids is not None:
            locked_ids = frozenset(str(document_id) for document_id in document_ids)

            if not locked_ids:
                raise ValueError("Le source lock ne peut pas être vide.")

            active_ids = {
                document.document_id for document in self.catalog.documents if document.active
            }

            unknown = locked_ids - active_ids

            if unknown:
                raise ValueError("Source lock inconnu : " + ", ".join(sorted(unknown)))

            if routing.hard_filter is not None and routing.hard_filter != locked_ids:
                raise ValueError(
                    "Le source lock runtime contredit la source explicitement demandée."
                )

            routing = replace(
                routing,
                preferred_documents=tuple(sorted(locked_ids)),
                soft_boosts={},
                hard_filter=locked_ids,
                explanation=("Runtime evidence-based source lock."),
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
        document_ids = set(routing.hard_filter) if routing.hard_filter is not None else None
        cross_encoder_k = max(30, candidate_k, top_k)
        structured_roles = len(plan.roles) > 1
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
        if not hybrid.results:
            raise ValueError("Le retriever v4 n'a produit aucun candidat global.")

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
        covered_roles: tuple[str, ...] = ()
        selection_limit = len(reranking.results)
        if selection_limit == 0:
            raise ValueError("Le reranker v4 n'a produit aucun candidat.")

        if structured_roles:
            role_selection = select_role_aware_evidence(
                plan,
                hybrid.results,
                reranking.results,
                top_k=selection_limit,
            )
            selected = list(role_selection.selected)
            covered_roles = role_selection.covered_roles
        else:
            selected = select_with_lexical_safeguard(
                hybrid.results,
                reranking.results,
                top_k=selection_limit,
                lexical_slots=min(lexical_slots, max(0, selection_limit - 1)),
            )
            selected = self._repair_weak_lexical_selection(
                selected,
                candidates=hybrid.results,
                reranked_results=reranking.results,
                question_type=question_type,
                top_k=selection_limit,
            )

        reranked_by_id = {result.chunk.chunk_id: result for result in reranking.results}
        anchors: list[EvidenceAnchor] = []
        for selection in selected:
            child = self.child_by_id.get(selection.chunk_id)
            if child is None:
                raise ValueError(f"Chunk qualité introuvable : {selection.chunk_id}")
            reranked = reranked_by_id.get(selection.chunk_id)
            anchors.append(
                EvidenceAnchor(
                    child=child,
                    score=(reranked.reranker_score if reranked is not None else 0.0),
                    provenance=selection.source,
                )
            )

        bundles = self.expander.expand(
            anchors,
            question_type=question_type,
        )
        if not bundles:
            raise ValueError("L'expansion qualité n'a produit aucun evidence bundle.")

        packed_anchor_ids = {
            chunk_id for bundle in bundles for chunk_id in bundle.anchor_chunk_ids
        }
        selected = [item for item in selected if item.chunk_id in packed_anchor_ids]

        if document_ids is not None:
            allowed_document_ids = set(document_ids)

            leaked_candidates = {
                result.chunk.document_id
                for result in hybrid.results
                if result.chunk.document_id not in allowed_document_ids
            }

            leaked_bundles = {
                bundle.document_id
                for bundle in bundles
                if bundle.document_id not in allowed_document_ids
            }

            if leaked_candidates or leaked_bundles:
                raise ValueError(
                    "Violation du source lock : "
                    f"candidates={sorted(leaked_candidates)} "
                    f"bundles={sorted(leaked_bundles)}"
                )

        LOGGER.info(
            "RETRIEVER_V4 type=%s roles=%s "
            "dense_found=%s sparse_found=%s bm25_found=%s fusion=%s reranked=%s",
            question_type,
            ",".join(role.name for role in plan.roles),
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
            section_search=section_search,
            retrieval_plan=plan,
            covered_roles=covered_roles,
        )
