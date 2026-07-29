"""Citation validation, public response assembly and source binding."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from phosprocess.observability.latency import RAGLatencyMetrics
from phosprocess.rag.citations import (
    CitationValidationError,
    extract_citations,
    is_controlled_insufficient_answer,
)
from phosprocess.rag.fidelity import validate_claim_support
from phosprocess.rag.quality_retrieval import QualityRetrievalResult
from phosprocess.rag.retrieval_service import (
    RAGRetrievalError,
    _RetrievedContext,
)
from phosprocess.rag.schemas import RAGResponse, RAGSource, RAGTimings
from phosprocess.reranking.reranker import clean_passage_text
from phosprocess.retrieval.evidence_bundle import EvidenceBundle

LOGGER = logging.getLogger("phosprocess.rag.pipeline")


class AnswerValidationService:
    def _validate_answer(
        self,
        *,
        answer: str,
        available_source_count: int,
        attempt: str,
        evidence_bundles: Sequence[EvidenceBundle] | None = None,
    ) -> tuple[list[int], bool]:
        """Validate citations with answer as the sole source of truth."""

        citations = extract_citations(
            answer,
            available_source_count=available_source_count,
        )
        insufficient = is_controlled_insufficient_answer(answer)

        if not citations and not insufficient:
            raise CitationValidationError(
                "Une réponse affirmative exige une citation [Source N]."
            )

        if evidence_bundles is not None and not insufficient:
            validate_claim_support(answer, list(evidence_bundles))

        LOGGER.info(
            "RAG citations validated attempt=%s citations=%s "
            "available_sources=%d",
            attempt,
            citations,
            available_source_count,
        )
        return citations, insufficient

    def _validate_answer_with_metrics(
        self,
        *,
        answer: str,
        available_source_count: int,
        attempt: str,
        metrics: RAGLatencyMetrics,
        evidence_bundles: Sequence[EvidenceBundle] | None = None,
    ) -> tuple[list[int], bool]:
        """Measure citation extraction separately from policy validation."""

        phase_started = time.perf_counter()
        citations = extract_citations(
            answer,
            available_source_count=available_source_count,
        )
        metrics.citation_extraction_ms += (
            time.perf_counter() - phase_started
        ) * 1000.0
        phase_started = time.perf_counter()
        insufficient = is_controlled_insufficient_answer(answer)

        if not citations and not insufficient:
            metrics.citation_validation_ms += (
                time.perf_counter() - phase_started
            ) * 1000.0
            raise CitationValidationError(
                "Une réponse affirmative exige une citation [Source N]."
            )

        if evidence_bundles is not None and not insufficient:
            validate_claim_support(answer, list(evidence_bundles))

        metrics.citation_validation_ms += (
            time.perf_counter() - phase_started
        ) * 1000.0
        LOGGER.info(
            "RAG citations validated attempt=%s citations=%s "
            "available_sources=%d",
            attempt,
            citations,
            available_source_count,
        )
        return citations, insufficient

    @staticmethod
    def _log_validation_rejection(
        *,
        attempt: str,
        error: CitationValidationError,
        available_source_count: int,
        raw_output: str,
        final: bool,
    ) -> None:
        """Log safe diagnostics; raw output appears only after final failure."""

        LOGGER.warning(
            "RAG citations rejected attempt=%s detected_citations=%s "
            "available_sources=%d reason=%s",
            attempt,
            error.detected_citations,
            available_source_count,
            error,
        )

        if final:
            LOGGER.error(
                "RAG streaming invalid after repair raw_output=%r",
                raw_output,
            )

    @staticmethod
    def _comparison_subjects(
        quality_result: QualityRetrievalResult | None,
    ) -> tuple[str, ...]:
        """Return explicit A/B subjects from the retrieval plan."""

        if quality_result is None or quality_result.retrieval_plan is None:
            return ()
        return tuple(
            role.subject
            for role in quality_result.retrieval_plan.roles
            if role.name in {"equipment_a", "equipment_b"}
            and role.subject
        )

    @staticmethod
    def _cited_sources(
        sources: Sequence[RAGSource],
        cited_source_numbers: Sequence[int],
    ) -> list[RAGSource]:
        """Return only cited source objects in first-appearance order."""

        source_by_number = {
            source.source_number: source
            for source in sources
        }
        return [
            source_by_number[source_number]
            for source_number in cited_source_numbers
        ]

    def _build_response(
        self,
        *,
        question: str,
        answer: str,
        cited_sources: list[RAGSource],
        cited_source_numbers: list[int],
        insufficient_context: bool,
        retrieved: _RetrievedContext,
        generation_ms: float,
        total_ms: float,
        first_token_ms: float | None = None,
        latency: RAGLatencyMetrics | None = None,
    ) -> RAGResponse:
        """Build the public response for blocking or streaming calls."""

        model_name = getattr(
            self.llm,
            "model_name",
            self.runtime_config.ollama.model,
        )
        source_policy = retrieved.source_policy
        return RAGResponse(
            question=question,
            answer=answer,
            sources=cited_sources,
            cited_source_numbers=cited_source_numbers,
            insufficient_context=insufficient_context,
            model_name=str(model_name),
            selected_variant=self.frozen_config.selected_variant,
            snapshot_sha256=self.frozen_config.snapshot_sha256,
            candidate_count=len(retrieved.candidates),
            selected_count=len(retrieved.selected),
            source_policy_route=(
                source_policy.route
                if source_policy is not None
                else "disabled"
            ),
            source_policy_mode=(
                source_policy.mode
                if source_policy is not None
                else "automatic"
            ),
            source_policy_primary=(
                source_policy.primary_label
                if source_policy is not None
                else None
            ),
            source_policy_fallback_used=(
                source_policy.fallback_used
                if source_policy is not None
                else False
            ),
            source_policy_forced=(
                source_policy.forced
                if source_policy is not None
                else False
            ),
            response_language=retrieved.response_language,
            standalone_query=(
                retrieved.quality_result.query.standalone_query
                if retrieved.quality_result is not None
                else None
            ),
            question_type=retrieved.question_type,
            detected_domains=(
                [
                    domain.value
                    for domain, _confidence in (
                        retrieved.quality_result.routing.detected_domains
                    )
                ]
                if retrieved.quality_result is not None
                else []
            ),
            timings=RAGTimings(
                hybrid_ms=float(
                    retrieved.hybrid_response.total_duration_ms
                ),
                reranking_ms=float(
                    retrieved.reranked_response.reranking_duration_ms
                ),
                generation_ms=generation_ms,
                total_ms=total_ms,
                first_token_ms=first_token_ms,
            ),
            latency=(
                latency.to_dict()
                if latency is not None
                else {}
            ),
        )

    def _build_sources(
        self,
        *,
        candidates: list[Any],
        reranked_results: list[Any],
        selected: list[Any],
    ) -> tuple[list[RAGSource], list[str]]:
        """Rehydrate full text and provenance for five selections."""

        candidate_by_id = {
            result.chunk.chunk_id: result
            for result in candidates
        }
        reranked_by_id = {
            result.chunk.chunk_id: result
            for result in reranked_results
        }
        sources: list[RAGSource] = []
        full_texts: list[str] = []

        for source_number, selection in enumerate(selected, start=1):
            candidate = candidate_by_id[selection.chunk_id]
            reranked = reranked_by_id.get(selection.chunk_id)
            chunk = candidate.chunk
            full_text = clean_passage_text(chunk.text)

            if not full_text:
                raise RAGRetrievalError(
                    f"Le chunk {chunk.chunk_id} possède un texte vide."
                )

            section = (
                " > ".join(
                    heading
                    for heading in chunk.heading_path
                    if heading
                )
                or None
            )
            sources.append(
                RAGSource(
                    source_number=source_number,
                    chunk_id=chunk.chunk_id,
                    document_name=Path(chunk.source_file).name,
                    pages=list(chunk.source_pages),
                    section=section,
                    excerpt=self._excerpt(full_text),
                    selection_source=selection.source,
                    hybrid_rank=candidate.rank,
                    rrf_score=float(candidate.rrf_score),
                    dense_rank=candidate.dense_rank,
                    dense_score=candidate.dense_score,
                    bm25_rank=candidate.bm25_rank,
                    bm25_score=candidate.bm25_score,
                    reranker_rank=(
                        reranked.rank
                        if reranked is not None
                        else None
                    ),
                    reranker_score=(
                        float(reranked.reranker_score)
                        if reranked is not None
                        else None
                    ),
                )
            )
            full_texts.append(full_text)

        return sources, full_texts

    def _excerpt(self, text: str) -> str:
        """Create a bounded source display excerpt."""

        limit = self.runtime_config.source_excerpt_characters

        if len(text) <= limit:
            return text

        return text[: limit - 1].rstrip() + "…"
