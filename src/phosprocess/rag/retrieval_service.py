"""Retrieval, source-policy and context preparation service."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from phosprocess.observability.latency import RAGLatencyMetrics
from phosprocess.rag.context_window import (
    PreparedDocumentContext,
    prepare_document_context,
)
from phosprocess.rag.quality_retrieval import QualityRetrievalResult
from phosprocess.rag.question_classifier import classify_question
from phosprocess.rag.schemas import RAGSource
from phosprocess.rag.source_policy import (
    AppliedSourcePolicy,
    SourcePolicyDecision,
    decide_source_policy,
    detect_explicit_active_source,
    document_id_from_source,
)
from phosprocess.retrieval.hybrid import expand_lexical_query
from phosprocess.retrieval.v3_selection import select_with_lexical_safeguard

LOGGER = logging.getLogger("phosprocess.rag.pipeline")


class RAGError(RuntimeError):
    """Base class for production RAG failures."""


class RAGConfigurationError(RAGError):
    """Raised when the frozen runtime configuration is invalid."""


class RAGRetrievalError(RAGError):
    """Raised when retrieval cannot produce grounded evidence."""


class RAGGenerationError(RAGError):
    """Raised when local Qwen generation fails."""


class RAGResponseValidationError(RAGError):
    """Raised when citations or structured output are invalid."""


@dataclass(slots=True)
class _RetrievedContext:
    """Internal retrieval result shared by blocking and streaming calls."""

    candidates: list[Any]
    selected: list[Any]
    sources: list[RAGSource]
    source_texts: list[str]
    hybrid_response: Any
    reranked_response: Any
    source_policy: AppliedSourcePolicy | None = None
    quality_result: QualityRetrievalResult | None = None
    response_language: str | None = None
    question_type: str | None = None


class RetrievalService:
    def _retrieve_quality(
        self,
        query: str,
        *,
        original_question: str,
        question_type: str,
        source_mode: str,
        metrics: RAGLatencyMetrics | None,
    ) -> _RetrievedContext:
        """Run global quality retrieval with soft routing and evidence bundles."""

        engine = self.quality_engine

        if engine is None:
            raise RAGRetrievalError("L'index qualité n'est pas actif.")

        quality_mode = self._quality_source_mode(source_mode)

        attempt_count = 1
        source_fallback_used = False

        try:
            discover = getattr(
                engine,
                "discover_documents",
                None,
            )

            if quality_mode == "auto" and callable(discover):
                ranking = discover(
                    original_question,
                    standalone_query=query,
                    question_type=question_type,
                    candidate_k=self.frozen_config.candidate_k,
                    dense_candidate_k=(self.frozen_config.dense_candidates),
                    bm25_candidate_k=(self.frozen_config.bm25_candidates),
                )

                result = None
                last_document_error: Exception | None = None

                for (
                    attempt_count,
                    ranked_document,
                ) in enumerate(
                    ranking,
                    start=1,
                ):
                    try:
                        result = engine.retrieve(
                            original_question,
                            standalone_query=query,
                            question_type=question_type,
                            source_mode="auto",
                            candidate_k=(self.frozen_config.candidate_k),
                            dense_candidate_k=(self.frozen_config.dense_candidates),
                            bm25_candidate_k=(self.frozen_config.bm25_candidates),
                            top_k=self.frozen_config.top_k,
                            lexical_slots=(self.frozen_config.lexical_slots),
                            document_ids={ranked_document.document_id},
                        )

                        bundle_documents = {bundle.document_id for bundle in result.bundles}

                        if bundle_documents != {ranked_document.document_id}:
                            raise ValueError("Le deep retrieval n'a pas respect? le source lock.")

                        source_fallback_used = attempt_count > 1

                        break

                    except Exception as document_error:
                        last_document_error = document_error

                        LOGGER.warning(
                            "DOCUMENT_LOCK_ATTEMPT_FAILED attempt=%d document=%s reason=%s",
                            attempt_count,
                            ranked_document.document_id,
                            document_error,
                        )

                        result = None

                if result is None:
                    if last_document_error is not None:
                        raise last_document_error

                    raise ValueError("Aucun document class? n'a fourni un contexte suffisant.")

                LOGGER.info(
                    "DOCUMENT_LOCK_SELECTED document=%s attempt=%d fallback=%s ranking=%s",
                    result.routing.preferred_documents[0],
                    attempt_count,
                    source_fallback_used,
                    " | ".join(item.document_id for item in ranking),
                )

            else:
                result = engine.retrieve(
                    original_question,
                    standalone_query=query,
                    question_type=question_type,
                    source_mode=quality_mode,
                    candidate_k=self.frozen_config.candidate_k,
                    dense_candidate_k=(self.frozen_config.dense_candidates),
                    bm25_candidate_k=(self.frozen_config.bm25_candidates),
                    top_k=self.frozen_config.top_k,
                    lexical_slots=(self.frozen_config.lexical_slots),
                )

        except Exception as error:
            detail = str(error).strip() or repr(error)
            LOGGER.error(
                "Structured quality retrieval failed "
                "error_type=%s reason=%s question_type=%s "
                "source_mode=%s original_question=%r standalone_query=%r",
                type(error).__name__,
                detail,
                question_type,
                source_mode,
                original_question,
                query,
            )
            raise RAGRetrievalError(
                f"La recherche structurée qualité a échoué : {type(error).__name__}: {detail}"
            ) from error

        LOGGER.info(
            "ROUTING_DECISION intent=%s source_mode=%s explicit_source=%s "
            "temporal_scope=%s domains=%s primary=%s hard_filter=%s "
            "section_hints=%s",
            result.routing.question_type,
            result.routing.source_mode,
            result.routing.explicit_source or "none",
            result.routing.temporal_scope,
            ",".join(domain.value for domain, _confidence in result.routing.detected_domains)
            or "none",
            (
                result.routing.preferred_documents[0]
                if result.routing.preferred_documents
                else "none"
            ),
            ",".join(sorted(result.routing.hard_filter or ())) or "none",
            "|".join(result.routing.section_affinity_terms) or "none",
        )

        LOGGER.debug(
            "RAG_QUALITY_QUERY original=%r standalone=%r dense=%r bm25=%r "
            "added_terms=%s domains=%s preferred_documents=%s hard_filter=%s",
            original_question,
            result.query.standalone_query,
            result.query.dense_query,
            result.query.bm25_expanded_query,
            result.query.added_terms,
            tuple(domain.value for domain, _confidence in result.routing.detected_domains),
            result.routing.preferred_documents,
            result.routing.hard_filter,
        )

        candidate_by_id = {
            candidate.chunk.chunk_id: candidate for candidate in result.hybrid.results
        }
        reranked_by_id = {
            reranked.chunk.chunk_id: reranked for reranked in result.reranking.results
        }
        selection_by_id = {selection.chunk_id: selection for selection in result.selected}
        sources: list[RAGSource] = []

        for bundle in result.bundles:
            candidate = candidate_by_id[bundle.anchor_chunk_id]
            reranked = reranked_by_id.get(bundle.anchor_chunk_id)
            selection = selection_by_id[bundle.anchor_chunk_id]
            child = engine.child_by_id[bundle.anchor_chunk_id]
            sources.append(
                RAGSource(
                    source_number=bundle.source_number,
                    chunk_id=bundle.anchor_chunk_id,
                    document_name=bundle.filename,
                    pages=list(range(bundle.page_start, bundle.page_end + 1)),
                    section=bundle.section,
                    excerpt=self._excerpt(bundle.display_text),
                    document_title=bundle.document_title,
                    filename=bundle.filename,
                    chapter=bundle.chapter,
                    page_start=bundle.page_start,
                    page_end=bundle.page_end,
                    anchor_chunk_id=bundle.anchor_chunk_id,
                    anchor_chunk_ids=list(bundle.anchor_chunk_ids),
                    expanded_chunk_ids=list(bundle.expanded_chunk_ids),
                    supporting_chunk_ids=list(bundle.supporting_chunk_ids),
                    display_text=bundle.display_text,
                    anchor_text=child.display_text,
                    domain=", ".join(child.domains),
                    chunk_type=child.chunk_type.value,
                    parent_id=child.parent_id,
                    context_scope=bundle.context_scope.value,
                    best_anchor_score=bundle.best_anchor_score,
                    source_boost=result.source_boosts.get(
                        bundle.anchor_chunk_id,
                        0.0,
                    ),
                    context_added_tokens=bundle.context_token_count,
                    context_truncated=bundle.context_truncated,
                    selection_source=selection.source,
                    hybrid_rank=candidate.rank,
                    rrf_score=float(candidate.rrf_score),
                    dense_rank=candidate.dense_rank,
                    dense_score=candidate.dense_score,
                    bm25_rank=candidate.bm25_rank,
                    bm25_score=candidate.bm25_score,
                    reranker_rank=(reranked.rank if reranked is not None else None),
                    reranker_score=(reranked.reranker_score if reranked is not None else None),
                )
            )

        detected_domains = tuple(
            domain.value for domain, _confidence in result.routing.detected_domains
        )
        catalog_by_id = {entry.document_id: entry for entry in engine.catalog.documents}
        preferred_files = tuple(
            catalog_by_id[document_id].canonical_filename
            for document_id in result.routing.preferred_documents
        )
        primary = preferred_files[0] if preferred_files else None
        application = AppliedSourcePolicy(
            route=",".join(detected_domains) or "general_chemical_engineering",
            mode=result.routing.source_mode,
            primary_source=primary,
            preferred_sources=preferred_files,
            selected_scope=tuple(dict.fromkeys(bundle.filename for bundle in result.bundles)),
            fallback_used=source_fallback_used,
            forced=(quality_mode != "auto" and result.routing.hard_filter is not None),
            attempt_count=attempt_count,
            sufficient_preferred_chunks=sum(
                bundle.document_id in result.routing.preferred_documents
                for bundle in result.bundles
            ),
        )

        if metrics is not None:
            metrics.retrieval_query = result.query.standalone_query
            metrics.query_expansion_ms = 0.0
            metrics.dense_search_ms += result.hybrid.dense_duration_ms
            metrics.bm25_search_ms += result.hybrid.bm25_duration_ms
            metrics.hybrid_fusion_ms += max(
                0.0,
                result.hybrid.total_duration_ms
                - result.hybrid.dense_duration_ms
                - result.hybrid.bm25_duration_ms,
            )
            metrics.reranking_ms += result.reranking.reranking_duration_ms
            metrics.document_context_token_count = sum(
                bundle.token_count for bundle in result.bundles
            )

        retrieved = _RetrievedContext(
            candidates=list(result.hybrid.results),
            selected=list(result.selected),
            sources=sources,
            source_texts=[bundle.display_text for bundle in result.bundles],
            hybrid_response=result.hybrid,
            reranked_response=result.reranking,
            quality_result=result,
        )
        return self._attach_source_policy(
            retrieved,
            application,
            metrics=metrics,
        )

    def _retrieve_with_source_policy(
        self,
        query: str,
        *,
        policy_question: str,
        source_mode: str,
        metrics: RAGLatencyMetrics | None = None,
    ) -> _RetrievedContext:
        """Retrieve globally unless the user explicitly locks one source."""

        if self.quality_engine is not None:
            classification = classify_question(query)
            return self._retrieve_quality(
                query,
                original_question=policy_question,
                question_type=classification.question_type.value,
                source_mode=source_mode,
                metrics=metrics,
            )

        decision = self._decide_source_policy(
            policy_question,
            mode=source_mode,
        )

        if metrics is not None:
            metrics.source_policy_route = decision.route
            metrics.source_policy_mode = decision.mode
            metrics.source_policy_primary = decision.primary_label
            metrics.source_policy_forced = decision.forced

        if decision.forced and decision.primary_source is not None:
            scope = (decision.primary_source,)
            retrieved = self._retrieve(
                query,
                metrics=metrics,
                document_ids=self._document_ids(scope),
            )
            application = AppliedSourcePolicy(
                route=decision.route,
                mode=decision.mode,
                primary_source=decision.primary_source,
                preferred_sources=scope,
                selected_scope=scope,
                fallback_used=False,
                forced=True,
                attempt_count=1,
                sufficient_preferred_chunks=len(retrieved.selected),
            )
            return self._attach_source_policy(
                retrieved,
                application,
                metrics=metrics,
            )

        retrieved = self._retrieve(
            query,
            metrics=metrics,
        )
        application = AppliedSourcePolicy(
            route="automatic_global",
            mode="automatic",
            primary_source=None,
            preferred_sources=(),
            selected_scope=self._active_source_filenames(),
            fallback_used=False,
            forced=False,
            attempt_count=1,
            sufficient_preferred_chunks=0,
        )
        return self._attach_source_policy(
            retrieved,
            application,
            metrics=metrics,
        )

    def _decide_source_policy(
        self,
        question: str,
        *,
        mode: str,
    ) -> SourcePolicyDecision:
        """Honor explicit source requests; automatic mode stays global."""

        normalized_mode = mode.strip().casefold()
        if normalized_mode == "auto":
            normalized_mode = "automatic"

        if normalized_mode == "automatic":
            explicit_source = detect_explicit_active_source(
                question,
                self._active_source_filenames(),
            )
            if explicit_source is not None:
                return SourcePolicyDecision(
                    route="explicit_document",
                    mode="automatic",
                    preferred_sources=(explicit_source,),
                    primary_source=explicit_source,
                    forced=True,
                    allow_fallback=False,
                )

        return decide_source_policy(
            question,
            config=self.runtime_config.source_policy,
            mode=mode,
        )

    def _active_source_filenames(self) -> tuple[str, ...]:
        active = self.active_knowledge_base

        if active is None:
            return ()

        return tuple(
            str(document["filename"])
            for document in active.documents
            if isinstance(document.get("filename"), str)
        )

    @staticmethod
    def _document_ids(sources: Sequence[str]) -> set[str]:
        """Build exact indexed document IDs from configured filenames."""

        return {document_id_from_source(source) for source in sources}

    @staticmethod
    def _attach_source_policy(
        retrieved: _RetrievedContext,
        application: AppliedSourcePolicy,
        *,
        metrics: RAGLatencyMetrics | None,
    ) -> _RetrievedContext:
        """Attach, log and expose one content-safe policy outcome."""

        retrieved.source_policy = application

        if metrics is not None:
            metrics.source_policy_route = application.route
            metrics.source_policy_mode = application.mode
            metrics.source_policy_primary = application.primary_label
            metrics.source_policy_fallback_used = application.fallback_used
            metrics.source_policy_attempt_count = application.attempt_count
            metrics.source_policy_sufficient_preferred_chunks = (
                application.sufficient_preferred_chunks
            )

        LOGGER.info(
            "RAG_SOURCE_POLICY route=%s mode=%s primary=%s "
            "fallback=%s attempts=%d selected_scope=%s",
            application.route,
            application.mode,
            application.primary_label,
            application.fallback_used,
            application.attempt_count,
            ",".join(application.selected_scope),
        )
        return retrieved

    def _retrieve(
        self,
        query: str,
        *,
        metrics: RAGLatencyMetrics | None = None,
        document_ids: set[str] | None = None,
    ) -> _RetrievedContext:
        """Run exact frozen hybrid → reranker → safeguard sequence."""

        self._active_metrics = metrics
        embedding_before = metrics.embedding_ms if metrics is not None else 0.0

        if metrics is not None:
            expansion_started = time.perf_counter()
            hybrid_config = getattr(self.retriever, "config", None)
            expansion_version = getattr(
                hybrid_config,
                "query_expansion_version",
                "phosphoric_v2",
            )
            expand_lexical_query(
                query,
                version=expansion_version,
            )
            metrics.query_expansion_ms += (time.perf_counter() - expansion_started) * 1000.0

        try:
            hybrid_response = self.retriever.search(
                query,
                top_k=self.frozen_config.candidate_k,
                dense_candidate_k=self.frozen_config.dense_candidates,
                bm25_candidate_k=self.frozen_config.bm25_candidates,
                document_ids=document_ids,
                use_query_expansion=self.frozen_config.query_expansion,
            )
        except Exception as error:
            raise RAGRetrievalError("La recherche hybride a échoué.") from error

        if metrics is not None:
            dense_duration_ms = float(getattr(hybrid_response, "dense_duration_ms", 0.0))
            bm25_duration_ms = float(getattr(hybrid_response, "bm25_duration_ms", 0.0))
            hybrid_total_ms = float(getattr(hybrid_response, "total_duration_ms", 0.0))
            attempt_embedding_ms = metrics.embedding_ms - embedding_before
            metrics.dense_search_ms += max(
                0.0,
                dense_duration_ms - attempt_embedding_ms,
            )
            metrics.bm25_search_ms += bm25_duration_ms
            metrics.hybrid_fusion_ms += max(
                0.0,
                hybrid_total_ms - dense_duration_ms - bm25_duration_ms,
            )

        phase_started = time.perf_counter()
        candidates = list(hybrid_response.results)
        candidate_ids = [result.chunk.chunk_id for result in candidates]

        if len(candidates) != self.frozen_config.candidate_k:
            raise RAGRetrievalError("Le retriever doit conserver exactement 20 candidats.")

        if len(candidate_ids) != len(set(candidate_ids)):
            raise RAGRetrievalError("Le retriever a retourné des chunks dupliqués.")

        if metrics is not None:
            metrics.candidate_preparation_ms += (time.perf_counter() - phase_started) * 1000.0

        phase_started = time.perf_counter()
        tokenization_before = metrics.reranker_tokenization_ms if metrics is not None else 0.0

        try:
            reranked_response = self.reranker.rerank(
                query,
                candidates,
                top_k=self.frozen_config.candidate_k,
            )
        except Exception as error:
            raise RAGRetrievalError("Le reranking BGE a échoué.") from error

        reranking_total_ms = (time.perf_counter() - phase_started) * 1000.0

        if metrics is not None:
            metrics.reranking_ms += reranking_total_ms
            reranker_internal_ms = float(
                getattr(
                    reranked_response,
                    "reranking_duration_ms",
                    reranking_total_ms,
                )
            )
            attempt_tokenization_ms = metrics.reranker_tokenization_ms - tokenization_before
            metrics.reranker_scoring_ms += max(
                0.0,
                reranker_internal_ms - attempt_tokenization_ms,
            )

        reranked_results = list(reranked_response.results)

        if len(reranked_results) != len(candidates):
            raise RAGRetrievalError("Le reranker doit conserver les 20 candidats.")

        phase_started = time.perf_counter()

        try:
            selected = select_with_lexical_safeguard(
                candidates,
                reranked_results,
                top_k=self.frozen_config.top_k,
                lexical_slots=self.frozen_config.lexical_slots,
            )
        except ValueError as error:
            raise RAGRetrievalError("La sélection lexical_safeguard_001 a échoué.") from error

        if metrics is not None:
            metrics.lexical_selection_ms += (time.perf_counter() - phase_started) * 1000.0

        if (
            len(selected) != self.frozen_config.top_k
            or len({result.chunk_id for result in selected}) != self.frozen_config.top_k
        ):
            raise RAGRetrievalError(
                "La sélection finale doit respecter top_k avec des chunks uniques."
            )

        phase_started = time.perf_counter()
        sources, source_texts = self._build_sources(
            candidates=candidates,
            reranked_results=reranked_results,
            selected=selected,
        )

        if metrics is not None:
            metrics.source_loading_ms += (time.perf_counter() - phase_started) * 1000.0

        return _RetrievedContext(
            candidates=candidates,
            selected=selected,
            sources=sources,
            source_texts=source_texts,
            hybrid_response=hybrid_response,
            reranked_response=reranked_response,
        )

    def _prepare_context(
        self,
        source_texts: Sequence[str],
        question: str,
    ) -> PreparedDocumentContext:
        """Select bounded relevant excerpts for all retrieved sources."""

        config = self.runtime_config.generation
        return prepare_document_context(
            source_texts,
            question,
            maximum_tokens_per_source=(config.max_context_tokens_per_source),
            maximum_total_tokens=(config.max_total_document_context_tokens),
        )
