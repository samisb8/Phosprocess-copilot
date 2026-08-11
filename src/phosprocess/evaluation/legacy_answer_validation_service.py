"""Research-only Phase-10 requirement planning and semantic audit helpers."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from phosprocess.llm.ollama_client import OllamaResponseValidationError
from phosprocess.observability.latency import RAGLatencyMetrics
from phosprocess.rag.citations import (
    CitationValidationError,
    extract_citations,
    is_controlled_insufficient_answer,
)
from phosprocess.rag.retrieval_service import (
    RAGRetrievalError,
    _RetrievedContext,
)
from phosprocess.rag.schemas import RAGResponse, RAGSource, RAGTimings
from phosprocess.reranking.reranker import clean_passage_text
from phosprocess.retrieval.evidence_bundle import EvidenceBundle

LOGGER = logging.getLogger("phosprocess.rag.pipeline")


class _EvidenceRequirement(BaseModel):
    """One atomic evidence-backed answer requirement."""

    description: str
    source_numbers: list[int] = Field(default_factory=list)
    sequence_index: int | None = None


class _EvidenceRequirementPlan(BaseModel):
    """Question-focused coverage plan produced before answer generation."""

    focus: str
    requirements: list[_EvidenceRequirement] = Field(default_factory=list)


class _AnswerAuditPayload(BaseModel):
    """Semantic grounding and fixed-plan coverage verdict."""

    grounded: bool
    complete: bool
    missing_requirement_ids: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)


class AnswerValidationService:
    def _plan_answer_requirements(
        self,
        *,
        question: str,
        evidence_bundles: Sequence[EvidenceBundle],
    ) -> _EvidenceRequirementPlan | None:
        """Derive question-focused requirements before seeing an answer."""

        if not question.strip() or not evidence_bundles:
            return None

        evidence_text = "\n\n".join(
            (f"[Source {bundle.source_number}]\n{bundle.display_text.strip()}")
            for bundle in evidence_bundles
        )

        system_prompt = "\n".join(
            [
                "You are an evidence requirement planner for a RAG system.",
                "Use only QUESTION and EVIDENCE.",
                "There is no candidate answer yet.",
                "Never use outside knowledge.",
                "",
                "Identify the exact information focus requested by the user.",
                (
                    "Before creating requirements, inspect every Source N "
                    "independently for information that directly contributes "
                    "to the requested focus."
                ),
                (
                    "Do not let a coherent narrative from one source suppress "
                    "a distinct relevant contribution explicitly supported by "
                    "another source."
                ),
                (
                    "For a question asking for the path, evolution, sequence "
                    "or movement of an entity, identify the entity being tracked "
                    "and keep the requirements centered on that entity."
                ),
                (
                    "For such tracking questions, requirements should describe "
                    "material movements, locations, states or direct "
                    "transformations of the tracked entity."
                ),
                (
                    "Do not switch the coverage focus to a secondary stream, "
                    "by-product, utility, control action or downstream treatment "
                    "of another stream unless the user explicitly asks for it."
                ),
                (
                    "A detail is not a requirement merely because it occurs "
                    "after another fact in the same source."
                ),
                (
                    "When the question is general and does not request a "
                    "specific plant, operating case or numerical condition, "
                    "do not promote plant-specific operating values to required "
                    "coverage merely because they appear in the evidence."
                ),
                (
                    "Prefer the relations necessary to answer the requested "
                    "task over incidental temperatures, pressures, flow rates, "
                    "capacities or other operating values unless the user asks "
                    "for those values or they are indispensable to the answer."
                ),
                (
                    "Compare all evidence bundles for distinct relevant "
                    "contributions, then merge genuinely duplicate information."
                ),
                (
                    "Do not combine incompatible equipment variants, operating "
                    "cases or scopes into one artificial sequence. Preserve an "
                    "explicit scope distinction when it matters."
                ),
                (
                    "Create atomic requirements containing only material "
                    "evidence-backed information needed for a complete answer."
                ),
                (
                    "A retrieved fact is not automatically a requirement merely "
                    "because it appears in the evidence."
                ),
                (
                    "Exclude peripheral operating values, examples, side streams "
                    "or auxiliary details unless they are necessary to satisfy "
                    "the actual question."
                ),
                (
                    "If the question asks for the path, evolution or sequence of "
                    "a particular entity, focus requirements on that entity and "
                    "its material transitions."
                ),
                (
                    "For sequence questions, preserve ordering only when the "
                    "evidence explicitly supports that order."
                ),
                (
                    "Merge overlapping evidence into one requirement when sources "
                    "describe the same material fact."
                ),
                (
                    "Every requirement must list one or more Source N numbers "
                    "that explicitly support it."
                ),
                ("Do not invent missing stages, mechanisms or relationships."),
                (
                    "Return exactly one compact JSON object. "
                    "Do not use Markdown, code fences or explanatory text."
                ),
                (
                    "Required JSON shape: "
                    '{"focus":"...",'
                    '"requirements":['
                    '{"description":"...",'
                    '"source_numbers":[1],'
                    '"sequence_index":1}'
                    "]}"
                ),
                (
                    "Use null for sequence_index when the evidence does not "
                    "establish an explicit order."
                ),
                (
                    "Before returning the plan, verify that each requirement "
                    "directly serves the QUESTION rather than merely summarizing "
                    "a retrieved document."
                ),
                (
                    "For tracking or sequence questions, verify that you have "
                    "not omitted a distinct evidence-backed transition of the "
                    "tracked entity solely because another source provides a "
                    "more detailed narrative."
                ),
            ]
        )

        user_prompt = "\n\n".join(
            [
                f"QUESTION\n{question.strip()}",
                f"EVIDENCE\n{evidence_text}",
                (
                    "Return focus and requirements. Each requirement must contain "
                    "description, source_numbers, and sequence_index. "
                    "Use sequence_index=null when no explicit order is supported."
                ),
            ]
        )

        try:
            payload, _raw = self.llm.chat_json_with_raw(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                response_model=_EvidenceRequirementPlan,
            )

        except OllamaResponseValidationError as error:
            raw_output = error.raw_response or ""

            LOGGER.warning(
                "RAG evidence requirement planner JSON invalid; "
                "attempting one structure repair reason=%s raw_chars=%d",
                error,
                len(raw_output),
            )

            repair_system_prompt = "\n".join(
                [
                    "You repair structured JSON output.",
                    "Do not answer the original question.",
                    "Do not add new factual information.",
                    ("Use only the QUESTION, EVIDENCE and previous planner draft supplied below."),
                    ("Return exactly one valid compact JSON object matching the required schema."),
                    "No Markdown.",
                    "No code fences.",
                    "No prose outside JSON.",
                ]
            )

            repair_user_prompt = "\n\n".join(
                [
                    user_prompt,
                    "PREVIOUS INVALID PLANNER OUTPUT\n" + raw_output,
                    (
                        "REQUIRED OUTPUT SHAPE\n"
                        '{"focus":"...",'
                        '"requirements":['
                        '{"description":"...",'
                        '"source_numbers":[1],'
                        '"sequence_index":1}'
                        "]}\n"
                        "Use null for sequence_index when no explicit "
                        "order is supported."
                    ),
                ]
            )

            try:
                payload, _raw = self.llm.chat_json_with_raw(
                    user_prompt=repair_user_prompt,
                    system_prompt=repair_system_prompt,
                    response_model=_EvidenceRequirementPlan,
                )
            except Exception as repair_error:
                LOGGER.warning(
                    "RAG evidence requirement planner unavailable after JSON repair reason=%s",
                    repair_error,
                )
                return None

            LOGGER.info("RAG evidence requirement planner JSON repair succeeded")

        except Exception as error:
            LOGGER.warning(
                "RAG evidence requirement planner unavailable reason=%s",
                error,
            )
            return None

        valid_sources = {bundle.source_number for bundle in evidence_bundles}

        requirements: list[_EvidenceRequirement] = []

        for requirement in payload.requirements:
            description = requirement.description.strip()

            source_numbers = sorted(
                {
                    source_number
                    for source_number in requirement.source_numbers
                    if source_number in valid_sources
                }
            )

            if not description or not source_numbers:
                continue

            requirements.append(
                requirement.model_copy(
                    update={
                        "description": description,
                        "source_numbers": source_numbers,
                    }
                )
            )

        if not requirements:
            LOGGER.warning("RAG evidence requirement planner returned no valid requirements")
            return None

        plan = payload.model_copy(
            update={
                "focus": payload.focus.strip() or question.strip(),
                "requirements": requirements,
            }
        )

        LOGGER.info(
            "RAG evidence requirements planned focus=%r requirements=%s",
            plan.focus,
            tuple(
                (
                    index,
                    requirement.description,
                    tuple(requirement.source_numbers),
                    requirement.sequence_index,
                )
                for index, requirement in enumerate(
                    plan.requirements,
                    start=1,
                )
            ),
        )

        return plan

    @staticmethod
    def _format_requirement_plan(
        plan: _EvidenceRequirementPlan,
    ) -> str:
        """Serialize a fixed evidence plan without creating new facts."""

        lines = [
            f"Focus: {plan.focus}",
        ]

        for index, requirement in enumerate(
            plan.requirements,
            start=1,
        ):
            sources = ",".join(str(source_number) for source_number in requirement.source_numbers)

            ordering = (
                f" | order={requirement.sequence_index}"
                if requirement.sequence_index is not None
                else ""
            )

            lines.append(f"R{index} | sources={sources}{ordering} | {requirement.description}")

        return "\n".join(lines)

    def _append_requirement_plan(
        self,
        prompt: str,
        plan: _EvidenceRequirementPlan | None,
    ) -> str:
        """Add the evidence-derived plan to the answer-generation prompt."""

        if plan is None:
            return prompt

        plan_text = self._format_requirement_plan(plan)

        return (
            prompt.rstrip()
            + "\n\nPRECOMPUTED EVIDENCE COVERAGE PLAN\n"
            + plan_text
            + "\n\n"
            + (
                "Use this plan only as a coverage guide. Cover every material "
                "requirement while writing the answer naturally. Preserve an "
                "order only when the plan provides one. The original evidence "
                "remains the factual authority, and every factual statement "
                "must cite its supporting Source N."
            )
        )

    def _validate_answer_semantics(
        self,
        *,
        question: str,
        answer: str,
        evidence_bundles: Sequence[EvidenceBundle],
        lexical_rejection: str = "",
        requirement_plan: _EvidenceRequirementPlan | None = None,
    ) -> None:
        """Audit semantic grounding and fixed-plan completeness."""

        if not question.strip() or not evidence_bundles:
            return

        evidence_text = "\n\n".join(
            (f"[Source {bundle.source_number}]\n{bundle.display_text.strip()}")
            for bundle in evidence_bundles
        )

        if requirement_plan is not None:
            plan_text = self._format_requirement_plan(requirement_plan)
            coverage_rules = [
                "COMPLETENESS:",
                ("Use PRECOMPUTED REQUIREMENTS as the fixed coverage specification."),
                ("Do not create, remove, broaden or reinterpret the requirements."),
                (
                    "For each R identifier, determine whether the ANSWER "
                    "semantically covers that requirement."
                ),
                (
                    "complete=true only when every material requirement "
                    "in the fixed plan is covered."
                ),
                (
                    "Return missing requirement identifiers such as R2 "
                    "or R4 in missing_requirement_ids."
                ),
            ]
        else:
            plan_text = "No precomputed requirement plan is available."
            coverage_rules = [
                "COMPLETENESS:",
                ("No precomputed requirement plan is available. Audit grounding only."),
                (
                    "Set complete=true and missing_requirement_ids=[] "
                    "rather than inventing requirements after seeing the answer."
                ),
            ]

        system_prompt = "\n".join(
            [
                "You are a strict semantic auditor for a RAG system.",
                "Use only QUESTION, ANSWER, EVIDENCE and the fixed plan.",
                "Never use outside knowledge.",
                "",
                "GROUNDING:",
                (
                    "Evaluate every factual statement against the exact "
                    "Source N cited by that statement."
                ),
                (
                    "Accept faithful semantic paraphrases and translations; "
                    "wording does not need to match the evidence."
                ),
                (
                    "Reject a claim when its cited evidence does not entail it, "
                    "contradicts it, reverses a relation or direction, changes "
                    "an actor/object, or changes a technical value."
                ),
                ("Topical similarity alone is not sufficient support."),
                (
                    "Numbers, units, directions and explicit relationships "
                    "must remain faithful to the cited evidence."
                ),
                "",
                *coverage_rules,
                "",
                "VERDICT:",
                (
                    "grounded=true only when every factual claim is supported "
                    "by at least one source that the claim actually cites."
                ),
                (
                    "unsupported_claims must contain only genuinely unsupported "
                    "claims, not valid paraphrases."
                ),
                "Return JSON only.",
            ]
        )

        user_prompt = "\n\n".join(
            [
                f"QUESTION\n{question.strip()}",
                ("PRECOMPUTED REQUIREMENTS\n" + plan_text),
                f"ANSWER\n{answer.strip()}",
                f"EVIDENCE\n{evidence_text}",
                (
                    "Return exactly grounded, complete, "
                    "missing_requirement_ids and unsupported_claims."
                ),
            ]
        )

        try:
            payload, _raw = self.llm.chat_json_with_raw(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                response_model=_AnswerAuditPayload,
            )
        except Exception as error:
            LOGGER.warning(
                "RAG semantic auditor unavailable reason=%s",
                error,
            )

            if lexical_rejection:
                raise CitationValidationError(lexical_rejection) from error

            return

        missing_ids = tuple(
            identifier.strip().upper()
            for identifier in payload.missing_requirement_ids
            if identifier.strip()
        )

        unsupported = tuple(claim.strip() for claim in payload.unsupported_claims if claim.strip())

        LOGGER.info(
            "RAG semantic audit grounded=%s complete=%s "
            "missing_requirement_ids=%s unsupported_claims=%s "
            "lexical_precheck_failed=%s fixed_plan=%s",
            payload.grounded,
            payload.complete,
            missing_ids,
            unsupported,
            bool(lexical_rejection),
            requirement_plan is not None,
        )

        if payload.grounded and payload.complete:
            return

        reasons: list[str] = []

        if not payload.grounded:
            detail = "; ".join(unsupported[:5])

            if not detail:
                detail = "one or more factual claims are not entailed"

            reasons.append("affirmations non soutenues : " + detail)

        if not payload.complete:
            requirement_by_id = {}

            if requirement_plan is not None:
                requirement_by_id = {
                    f"R{index}": requirement.description
                    for index, requirement in enumerate(
                        requirement_plan.requirements,
                        start=1,
                    )
                }

            missing_details = []

            for identifier in missing_ids:
                description = requirement_by_id.get(identifier)

                if description is not None:
                    missing_details.append(f"{identifier}: {description}")
                else:
                    missing_details.append(identifier)

            detail = "; ".join(missing_details[:8])

            if not detail:
                detail = "material planned requirements are missing"

            reasons.append("éléments importants manquants : " + detail)

        raise CitationValidationError(
            "Audit sémantique de la réponse rejeté : " + " | ".join(reasons)
        )

    def _validate_answer(
        self,
        *,
        answer: str,
        available_source_count: int,
        attempt: str,
    ) -> tuple[list[int], bool]:
        """Validate only objective citation and insufficiency invariants."""

        citations = extract_citations(
            answer,
            available_source_count=available_source_count,
        )
        insufficient = is_controlled_insufficient_answer(answer)

        if not citations and not insufficient:
            raise CitationValidationError("Une réponse affirmative exige une citation [Source N].")

        LOGGER.info(
            "RAG citations validated attempt=%s citations=%s available_sources=%d",
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
    ) -> tuple[list[int], bool]:
        """Measure citation extraction separately from policy validation."""

        phase_started = time.perf_counter()
        citations = extract_citations(
            answer,
            available_source_count=available_source_count,
        )
        metrics.citation_extraction_ms += (time.perf_counter() - phase_started) * 1000.0
        phase_started = time.perf_counter()
        insufficient = is_controlled_insufficient_answer(answer)

        if not citations and not insufficient:
            metrics.citation_validation_ms += (time.perf_counter() - phase_started) * 1000.0
            raise CitationValidationError("Une réponse affirmative exige une citation [Source N].")

        metrics.citation_validation_ms += (time.perf_counter() - phase_started) * 1000.0
        LOGGER.info(
            "RAG citations validated attempt=%s citations=%s available_sources=%d",
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
                "RAG streaming invalid objective output raw_output=%r",
                raw_output,
            )

    @staticmethod
    def _cited_sources(
        sources: Sequence[RAGSource],
        cited_source_numbers: Sequence[int],
    ) -> list[RAGSource]:
        """Return only cited source objects in first-appearance order."""

        source_by_number = {source.source_number: source for source in sources}
        return [source_by_number[source_number] for source_number in cited_source_numbers]

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
            source_policy_route=(source_policy.route if source_policy is not None else "disabled"),
            source_policy_mode=(source_policy.mode if source_policy is not None else "automatic"),
            source_policy_primary=(
                source_policy.primary_label if source_policy is not None else None
            ),
            source_policy_fallback_used=(
                source_policy.fallback_used if source_policy is not None else False
            ),
            source_policy_forced=(source_policy.forced if source_policy is not None else False),
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
                    for domain, _confidence in (retrieved.quality_result.routing.detected_domains)
                ]
                if retrieved.quality_result is not None
                else []
            ),
            timings=RAGTimings(
                hybrid_ms=float(retrieved.hybrid_response.total_duration_ms),
                reranking_ms=float(retrieved.reranked_response.reranking_duration_ms),
                generation_ms=generation_ms,
                total_ms=total_ms,
                first_token_ms=first_token_ms,
            ),
            latency=(latency.to_dict() if latency is not None else {}),
        )

    def _build_sources(
        self,
        *,
        candidates: list[Any],
        reranked_results: list[Any],
        selected: list[Any],
    ) -> tuple[list[RAGSource], list[str]]:
        """Rehydrate full text and provenance for the selected evidence."""

        candidate_by_id = {result.chunk.chunk_id: result for result in candidates}
        reranked_by_id = {result.chunk.chunk_id: result for result in reranked_results}
        sources: list[RAGSource] = []
        full_texts: list[str] = []

        for source_number, selection in enumerate(selected, start=1):
            candidate = candidate_by_id[selection.chunk_id]
            reranked = reranked_by_id.get(selection.chunk_id)
            chunk = candidate.chunk
            full_text = clean_passage_text(chunk.text)

            if not full_text:
                raise RAGRetrievalError(f"Le chunk {chunk.chunk_id} possède un texte vide.")

            section = " > ".join(heading for heading in chunk.heading_path if heading) or None
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
                    reranker_rank=(reranked.rank if reranked is not None else None),
                    reranker_score=(
                        float(reranked.reranker_score) if reranked is not None else None
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
