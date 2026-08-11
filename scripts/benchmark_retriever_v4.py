"""Retrieval-only acceptance benchmark for Patch 3.8 (no Ollama/Qwen call)."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from phosprocess.embeddings.embedder import resolve_cached_model_source
from phosprocess.knowledge_base.runtime import load_active_knowledge_base
from phosprocess.rag.conversation_state import ConversationState
from phosprocess.rag.followup_resolver import resolve_standalone_query
from phosprocess.rag.pipeline import (
    DEFAULT_EMBEDDING_CONFIG_PATH,
    load_frozen_v3_config,
)
from phosprocess.rag.quality_retrieval import QualityRetrievalEngine
from phosprocess.rag.question_classifier import classify_question
from phosprocess.reranking.reranker import (
    BGEReranker,
    load_reranking_config,
)
from phosprocess.retrieval.evidence_coverage import (
    coverage_keys_for_text,
    evaluate_evidence_coverage_texts,
    required_evidence_keys,
)
from phosprocess.retrieval.evidence_roles import supported_role_names
from phosprocess.retrieval.hybrid import HybridRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CASES = (
    ("definition_fc", "Qu’est-ce qu’un évaporateur à circulation forcée ?"),
    ("explanation_fc", "Why is forced circulation used in an industrial evaporator?"),
    (
        "process_flow",
        "Describe step by step the path of phosphoric acid through a "
        "forced-circulation evaporator, from the feed inlet to the concentrated "
        "product outlet.",
    ),
    (
        "comparison_fc_falling_film",
        "Compare a forced-circulation evaporator with a falling-film evaporator "
        "for phosphoric acid concentration.",
    ),
    (
        "fouling",
        "Explain how fouling can affect the operation of a phosphoric acid "
        "evaporator and give documented actions.",
    ),
    (
        "thermodynamic_pressure_boiling",
        "Quelle relation existe entre la pression de fonctionnement et la "
        "température d’ébullition dans un évaporateur ?",
    ),
    (
        "overall_mass_balance",
        "Establish the symbolic overall mass balance of a forced-circulation "
        "evaporator at steady state.",
    ),
    (
        "p2o5_balance",
        "Établis le bilan de P2O5 autour d’un évaporateur d’acide phosphorique "
        "en régime permanent.",
    ),
    (
        "energy_balance",
        "Establish the steady-state energy balance of a forced-circulation "
        "evaporator and define every term used.",
    ),
    ("pump_context", "Quel est le rôle de la pompe de circulation dans cet évaporateur ?"),
    ("pump_followup_fr", "Et pourquoi est-elle nécessaire ?"),
    ("pump_followup_en", "How does it send the liquid back to the flash chamber?"),
    (
        "arabic_vapor_body",
        "ما هو دور غرفة التبخير في فصل البخار عن الحمض؟",
    ),
)


def _normalized_source_texts(result: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        " ".join(bundle.display_text for bundle in result.bundles).casefold(),
    )


def _retrieval_recalls(
    engine: QualityRetrievalEngine,
    result: Any,
) -> tuple[
    float,
    float,
    list[str],
    list[str],
    dict[str, int],
    dict[str, int],
]:
    plan = result.retrieval_plan
    if plan is None:
        return 0.0, 0.0, [], [], {}, {}

    candidate_results = list(result.hybrid.results[:50])
    candidate_by_id = {item.chunk.chunk_id: item for item in candidate_results}
    evidence_results = [
        candidate_by_id[item.chunk.chunk_id]
        for item in result.reranking.results[:10]
        if item.chunk.chunk_id in candidate_by_id
    ]

    if plan.question_type == "process_flow":
        required = list(required_evidence_keys("process_flow"))
        candidate_texts = engine._coverage_texts_for_results(  # noqa: SLF001
            candidate_results,
            question_type="process_flow",
        )
        evidence_texts = engine._coverage_texts_for_results(  # noqa: SLF001
            evidence_results,
            question_type="process_flow",
        )
        candidate_coverage = evaluate_evidence_coverage_texts(
            list(candidate_texts.items()),
            question_type="process_flow",
        )
        evidence_coverage = evaluate_evidence_coverage_texts(
            list(evidence_texts.items()),
            question_type="process_flow",
        )
        candidate_supported = list(candidate_coverage.covered)
        evidence_supported = list(evidence_coverage.covered)
        candidate_role_ranks: dict[str, int] = {}
        for candidate in candidate_results:
            text = candidate_texts.get(candidate.chunk.chunk_id, "")
            for role_name in coverage_keys_for_text(text, "process_flow"):
                candidate_role_ranks.setdefault(role_name, candidate.rank)
        reranker_rank_by_id = {
            item.chunk.chunk_id: item.rank
            for item in result.reranking.results[:10]
        }
        evidence_role_ranks: dict[str, int] = {}
        for candidate in evidence_results:
            text = evidence_texts.get(candidate.chunk.chunk_id, "")
            for role_name in coverage_keys_for_text(text, "process_flow"):
                evidence_role_ranks.setdefault(
                    role_name,
                    reranker_rank_by_id[candidate.chunk.chunk_id],
                )
    else:
        required = [role.name for role in plan.roles if role.required]
        candidate_supported = list(
            supported_role_names(plan, candidate_results)
        )
        evidence_supported = list(
            supported_role_names(plan, evidence_results)
        )
        candidate_role_ranks = {}
        for candidate in candidate_results:
            for role_name in supported_role_names(plan, [candidate]):
                candidate_role_ranks.setdefault(role_name, candidate.rank)
        reranker_rank_by_id = {
            item.chunk.chunk_id: item.rank
            for item in result.reranking.results[:10]
        }
        evidence_role_ranks = {}
        for candidate in evidence_results:
            for role_name in supported_role_names(plan, [candidate]):
                evidence_role_ranks.setdefault(
                    role_name,
                    reranker_rank_by_id[candidate.chunk.chunk_id],
                )

    denominator = max(1, len(required))
    candidate_recall = len(set(required) & set(candidate_supported)) / denominator
    evidence_recall = len(set(required) & set(evidence_supported)) / denominator
    return (
        candidate_recall,
        evidence_recall,
        candidate_supported,
        evidence_supported,
        candidate_role_ranks,
        evidence_role_ranks,
    )


def _case_checks(
    case_id: str,
    resolution: Any,
    result: Any,
    *,
    candidate_recall_at_50: float,
    evidence_recall_at_10: float,
) -> list[str]:
    errors: list[str] = []
    text = _normalized_source_texts(result)

    if len(result.hybrid.results) < 5:
        errors.append("candidate_count_below_5")
    if candidate_recall_at_50 < 1.0:
        errors.append(
            f"candidate_recall_at_50={candidate_recall_at_50:.3f}"
        )
    if evidence_recall_at_10 < 1.0:
        errors.append(
            f"evidence_recall_at_10={evidence_recall_at_10:.3f}"
        )
    if result.hybrid.sparse_results_found <= 0:
        errors.append("bge_sparse_not_used")
    if len(result.reranking.results) < 5:
        errors.append("reranked_count_below_5")

    if case_id == "process_flow" and not result.coverage.complete:
        errors.append("process_flow_coverage_incomplete")
    if case_id == "thermodynamic_pressure_boiling":
        pressure_present = "pressure" in text or "pression" in text
        boiling_present = (
            "boiling" in text
            or "ébullition" in text
            or "ebullition" in text
            or "saturation temperature" in text
        )
        if not pressure_present or not boiling_present:
            errors.append("thermodynamic_relation_evidence_missing")
    if case_id == "comparison_fc_falling_film":
        if "forced circulation" not in text and "forced-circulation" not in text:
            errors.append("comparison_missing_equipment_a")
        if "falling film" not in text and "falling-film" not in text:
            errors.append("comparison_missing_equipment_b")
        if result.missing_roles:
            errors.append("comparison_missing_roles:" + ",".join(result.missing_roles))
    if case_id.endswith("balance") or case_id == "p2o5_balance":
        if result.missing_roles:
            errors.append("balance_missing_roles:" + ",".join(result.missing_roles))
        if result.retrieval_plan is None or result.retrieval_plan.balance_kind is None:
            errors.append("balance_kind_missing")
    if case_id in {"pump_followup_fr", "pump_followup_en"}:
        if "pompe de circulation" not in resolution.standalone_query.casefold() and (
            "circulation pump" not in resolution.standalone_query.casefold()
        ):
            errors.append("followup_focus_entity_not_resolved")
        if "filtration system" in text or "refrigerant" in text:
            errors.append("followup_off_topic_chunk")
    if case_id == "pump_context" and resolution.focus_entity != "pompe de circulation":
        errors.append("pump_not_saved_as_focus_entity")
    if case_id == "arabic_vapor_body":
        domains = {domain.value for domain, _score in result.routing.detected_domains}
        if "equipment" not in domains:
            errors.append("arabic_equipment_route_missing")
        if "vapor body" not in text and "evaporation chamber" not in text:
            errors.append("arabic_vapor_body_evidence_missing")
    if "refrigeration evaporator" in text or "refrigerant" in text:
        errors.append("refrigeration_chunk_in_final_evidence")
    if case_id == "fouling" and result.missing_roles:
        errors.append(
            "troubleshooting_missing_roles:" + ",".join(result.missing_roles)
        )
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "retrieval_v4_benchmark.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    active = load_active_knowledge_base()
    frozen = load_frozen_v3_config(
        verify_integrity=True,
        verify_runtime_sources=False,
    )
    retriever = HybridRetriever(
        dense_index_directory=active.dense_index_directory,
        bm25_index_directory=active.bm25_index_directory,
        embedding_config_path=DEFAULT_EMBEDDING_CONFIG_PATH,
        retrieval_config_path=frozen.retrieval_config_path,
    )
    reranking_config = load_reranking_config(frozen.reranking_config_path)
    reranker = BGEReranker(
        replace(
            reranking_config,
            model_name=resolve_cached_model_source(reranking_config.model_name),
        )
    )
    engine = QualityRetrievalEngine(
        version_directory=active.version_directory,
        retriever=retriever,
        reranker=reranker,
        require_sparse_index=True,
    )
    if engine.sparse_retriever is None:
        raise SystemExit(
            "Index BGE sparse absent. Lancez scripts/build_bge_sparse_index.py d'abord."
        )

    state = ConversationState()
    rows: list[dict[str, Any]] = []
    total_errors = 0
    for case_id, question in CASES:
        resolution = resolve_standalone_query(question, state=state)
        classification = classify_question(resolution.standalone_query)
        try:
            result = engine.retrieve(
                question,
                standalone_query=resolution.standalone_query,
                question_type=classification.question_type.value,
                candidate_k=50,
                dense_candidate_k=50,
                bm25_candidate_k=50,
                top_k=5,
                lexical_slots=1,
            )
            (
                candidate_recall_at_50,
                evidence_recall_at_10,
                candidate_supported_roles,
                evidence_supported_roles,
                candidate_role_ranks,
                evidence_role_ranks,
            ) = _retrieval_recalls(engine, result)
            errors = _case_checks(
                case_id,
                resolution,
                result,
                candidate_recall_at_50=candidate_recall_at_50,
                evidence_recall_at_10=evidence_recall_at_10,
            )
            row = {
                "id": case_id,
                "question": question,
                "standalone_query": resolution.standalone_query,
                "focus_entity": resolution.focus_entity,
                "question_type": classification.question_type.value,
                "plan_roles": (
                    [role.name for role in result.retrieval_plan.roles]
                    if result.retrieval_plan is not None
                    else []
                ),
                "plan_queries": (
                    [
                        {
                            "role": role.name,
                            "query": role.query,
                            "subject": role.subject,
                        }
                        for role in result.retrieval_plan.roles
                    ]
                    if result.retrieval_plan is not None
                    else []
                ),
                "covered_roles": list(result.covered_roles),
                "missing_roles": list(result.missing_roles),
                "candidate_recall_at_50": candidate_recall_at_50,
                "evidence_recall_at_10": evidence_recall_at_10,
                "candidate_supported_roles": candidate_supported_roles,
                "evidence_supported_roles": evidence_supported_roles,
                "candidate_role_ranks": candidate_role_ranks,
                "evidence_role_ranks": evidence_role_ranks,
                "top50_candidates": [
                    {
                        "rank": item.rank,
                        "chunk_id": item.chunk.chunk_id,
                        "document": item.chunk.source_file,
                        "section": item.chunk.section,
                        "roles": list(item.role_matches),
                        "dense_rank": item.dense_rank,
                        "sparse_rank": item.sparse_rank,
                        "bm25_rank": item.bm25_rank,
                        "colbert_score": item.colbert_score,
                    }
                    for item in result.hybrid.results
                ],
                "top10_reranked": [
                    {
                        "rank": item.rank,
                        "chunk_id": item.chunk.chunk_id,
                        "document": item.chunk.source_file,
                        "section": item.chunk.section,
                        "roles": list(item.role_matches),
                        "reranker_score": item.reranker_score,
                    }
                    for item in result.reranking.results[:10]
                ],
                "top5": [
                    {
                        "rank": selection.rank,
                        "chunk_id": selection.chunk_id,
                        "selection": selection.source,
                    }
                    for selection in result.selected
                ],
                "errors": errors,
                "passed": not errors,
            }
        except Exception as error:
            row = {
                "id": case_id,
                "question": question,
                "standalone_query": resolution.standalone_query,
                "focus_entity": resolution.focus_entity,
                "question_type": classification.question_type.value,
                "passed": False,
                "errors": [f"{type(error).__name__}: {error}"],
            }
        total_errors += len(row["errors"])
        rows.append(row)
        print(f"{case_id}: {'PASS' if row['passed'] else 'FAIL'}")
        for error in row["errors"]:
            print(f"  - {error}")

    payload = {
        "created_at": datetime.now().isoformat(),
        "knowledge_base": active.version,
        "case_count": len(rows),
        "passed": sum(bool(row["passed"]) for row in rows),
        "failed": sum(not bool(row["passed"]) for row in rows),
        "error_count": total_errors,
        "candidate_recall_at_50": (
            sum(float(row.get("candidate_recall_at_50", 0.0)) for row in rows)
            / len(rows)
        ),
        "evidence_recall_at_10": (
            sum(float(row.get("evidence_recall_at_10", 0.0)) for row in rows)
            / len(rows)
        ),
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Rapport : {args.output}")
    if payload["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
