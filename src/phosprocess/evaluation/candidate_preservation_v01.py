"""Phase-7 candidate-preservation evaluation.

The module is deliberately evaluation-only.  Gold chunk identifiers are used
only for measurement and are never included in retrieval queries, fusion, or
Qwen search-planning prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from phosprocess.evaluation.context_engine_v01 import (
    ACTIVE_DIRECTORY,
    read_jsonl,
    write_jsonl,
)
from phosprocess.evaluation.context_engine_v01 import (
    DEFAULT_OUTPUT as PHASE5_OUTPUT,
)
from phosprocess.evaluation.retriever_forensics_v01 import (
    QuerySpec,
    _locked_routing,
    aggregate_retriever_ids,
    fuse_raw,
    query_formulations,
    ranking_metrics,
    run_raw_retrieval,
)
from phosprocess.ingestion.chunk_serialization import read_child_chunks
from phosprocess.llm.ollama_client import OllamaLLM, load_ollama_config
from phosprocess.preprocessing.chunk_schemas import DocumentChunk
from phosprocess.rag.orchestrator import PhosProcessRAG
from phosprocess.rag.quality_retrieval import QualityRetrievalEngine
from phosprocess.reranking.reranker import build_reranking_passage, clean_passage_text
from phosprocess.retrieval.hybrid import HybridSearchResult
from phosprocess.retrieval.quality_hybrid import search_planned_hybrid
from phosprocess.retrieval.query_expansion import expand_technical_query

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "data/evaluation/candidate_preservation/v0.1"
DATASET_SIZE = 64
HOLDOUT_SIZE = 19
RAW_DEPTH = 200
RETRIEVERS = ("dense", "sparse", "bm25")
BUDGETS = (30, 40, 50, 60)
RESERVATIONS = (5, 10, 15)


# These assignments were made by reading the cited passages in the frozen
# active index.  Keeping them explicit prevents a retrieval model from
# manufacturing its own labels.
MANUAL_GOLDS: dict[str, tuple[str, ...]] = {
    "DQ004": (
        "ocp_phosphoric_acid_workshop_report_90de48de23c95efe",
        "ocp_phosphoric_acid_workshop_report_6f494915c239e09e",
    ),
    "DQ005": ("becker_phosphates_and_phosphoric_acid_bb3f6674a834bab6",),
    "DQ006": ("ocp_phosphoric_acid_workshop_report_6f494915c239e09e",),
    "DQ007": ("becker_phosphates_and_phosphoric_acid_c313bd326b0125bf",),
    "DQ008": ("becker_phosphates_and_phosphoric_acid_51e1c0227420da6c",),
    "DQ010": ("becker_phosphates_and_phosphoric_acid_f4f98a13827a21e3",),
    "DQ012": ("ocp_phosphoric_acid_workshop_report_94490160409585f8",),
    "DQ014": ("perrys_chemical_engineers_handbook_1606c2d550d62f92",),
    "DQ017": ("perrys_chemical_engineers_handbook_a7759841815ac9b4",),
    "DQ019": ("ocp_phosphoric_acid_workshop_report_dfd10f5ad2613f96",),
    "DQ024": ("ocp_phosphoric_acid_workshop_report_4dd00f7a95aa37d9",),
    "DQ026": ("perrys_chemical_engineers_handbook_30719e4f4d0445e0",),
    "DQ028": ("perrys_chemical_engineers_handbook_0a98c7afbb8e9ea8",),
    "DQ030": ("incropera_fundamentals_heat_mass_transfer_4e662122c49d1ed3",),
    "DQ032": ("mullin_crystallization_27134c0c5707f3e6",),
    "DQ033": ("becker_phosphates_and_phosphoric_acid_70e1c6804c7447cd",),
    "DQ040": ("seborg_process_dynamics_control_5120e39eda3e0722",),
    "DQ041": ("seborg_process_dynamics_control_e29d550f1dd8eec0",),
    "DQ042": ("seborg_process_dynamics_control_e41c99aa806354f6",),
    "DQ043": ("seborg_process_dynamics_control_af33ff56bb9192f4",),
    "DQ044": ("seborg_process_dynamics_control_d7620d288dba471a",),
    "DQ048": ("perrys_chemical_engineers_handbook_c2a7999ed5a4fc53",),
    "DQ049": ("incropera_fundamentals_heat_mass_transfer_b69399e466a31d43",),
    "CE055": ("perrys_chemical_engineers_handbook_0a98c7afbb8e9ea8",),
}


NEW_QUESTIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "CE062",
        "question": "What does the no-slip condition require at a solid-fluid interface?",
        "standalone_question": (
            "What does the no-slip condition require at a solid-fluid interface?"
        ),
        "language": "en",
        "question_type": "transport_phenomena",
        "gold_chunk_ids": ["bird_transport_phenomena_e0e4015c12e2e199"],
        "expected_concepts": ["no-slip condition", "solid-fluid interface", "velocity"],
    },
    {
        "id": "CE063",
        "question": (
            "What three properties must be equal for coexisting pure liquid and vapor "
            "phases to be in equilibrium?"
        ),
        "standalone_question": (
            "What three properties must be equal for coexisting pure liquid and vapor "
            "phases to be in equilibrium?"
        ),
        "language": "en",
        "question_type": "thermodynamics",
        "gold_chunk_ids": [
            "smith_van_ness_chemical_engineering_thermodynamics_9b668ccb438b81c0"
        ],
        "expected_concepts": ["temperature", "pressure", "fugacity"],
    },
    {
        "id": "CE064",
        "question": (
            "What process benefits result from properly operated calcium sulfate "
            "crystallization in phosphoric acid production?"
        ),
        "standalone_question": (
            "What process benefits result from properly operated calcium sulfate "
            "crystallization in phosphoric acid production?"
        ),
        "language": "en",
        "question_type": "crystallization",
        "gold_chunk_ids": ["becker_phosphates_and_phosphoric_acid_c313bd326b0125bf"],
        "expected_concepts": ["P2O5 recovery", "concentration", "filtration"],
    },
    {
        "id": "CE065",
        "question": (
            "What is the induction period in crystallization, and which operating "
            "factors influence it?"
        ),
        "standalone_question": (
            "What is the induction period in crystallization, and which operating "
            "factors influence it?"
        ),
        "language": "en",
        "question_type": "crystallization",
        "gold_chunk_ids": ["mullin_crystallization_27134c0c5707f3e6"],
        "expected_concepts": ["induction period", "supersaturation", "agitation"],
    },
    {
        "id": "CE066",
        "question": "إلى أي تركيز من P2O5 ترفع وحدة التركيز الحمض الفوسفوري، وكيف تزيل الماء؟",
        "standalone_question": (
            "إلى أي تركيز من P2O5 ترفع وحدة التركيز الحمض الفوسفوري، وكيف تزيل الماء؟"
        ),
        "language": "ar",
        "question_type": "process_flow",
        "gold_chunk_ids": ["ocp_phosphoric_acid_workshop_report_8802e71c2937ec58"],
        "expected_concepts": ["54% P2O5", "29% P2O5", "vacuum evaporation"],
    },
    {
        "id": "CE067",
        "question": "Why must liquid level and pressure be controlled in an evaporator?",
        "standalone_question": (
            "Why must liquid level and pressure be controlled in an evaporator?"
        ),
        "language": "en",
        "question_type": "control",
        "gold_chunk_ids": ["seborg_process_dynamics_control_e99ec9b53deeb4d4"],
        "expected_concepts": ["liquid level", "pressure", "operating safety"],
    },
)


class SearchPlanPayload(BaseModel):
    """Strict label-free Qwen output for hard-miss query planning."""

    model_config = ConfigDict(extra="forbid")

    standalone_query: str = Field(min_length=1)
    search_queries: list[str] = Field(min_length=1, max_length=4)


def _document_id(chunk_id: str) -> str:
    return chunk_id.rsplit("_", 1)[0]


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def freeze_splits(question_ids: Sequence[str]) -> dict[str, str]:
    """Freeze an exact 70/30 split from IDs alone, before any measurements."""

    if len(question_ids) != DATASET_SIZE or len(set(question_ids)) != DATASET_SIZE:
        raise ValueError(f"Expected {DATASET_SIZE} unique question IDs")
    ordered = sorted(question_ids, key=lambda item: (_sha256(item), item))
    holdout = set(ordered[:HOLDOUT_SIZE])
    return {item: ("final_holdout" if item in holdout else "dev") for item in question_ids}


def _choose_existing_gold(
    question: dict[str, Any], phase5_row: dict[str, Any]
) -> tuple[list[str], str]:
    gold = list(question.get("expected_evidence_chunk_ids") or [])
    if not gold:
        gold = list(MANUAL_GOLDS[question["id"]])
        provenance = "phase7_manual_passage_review"
    else:
        provenance = "phase5_curated"
    selected_document = str(phase5_row.get("selected_document") or "")
    selected = [item for item in gold if _document_id(item) == selected_document]
    if not selected:
        selected_document = _document_id(gold[0])
        selected = [item for item in gold if _document_id(item) == selected_document]
    return selected, provenance


def build_verified_dataset() -> list[dict[str, Any]]:
    """Create the static 64-question set and verify every ID against the index."""

    children = {
        child.chunk_id: child
        for child in read_child_chunks(ACTIVE_DIRECTORY / "chunks.jsonl")
    }
    phase5_rows = {
        row["id"]: row for row in read_jsonl(PHASE5_OUTPUT / "per_question_results.jsonl")
    }
    rows: list[dict[str, Any]] = []
    for question in read_jsonl(PHASE5_OUTPUT / "questions.jsonl"):
        if not question.get("answerable"):
            continue
        gold, provenance = _choose_existing_gold(question, phase5_rows[question["id"]])
        rows.append(
            {
                "id": question["id"],
                "question": question["question"],
                "standalone_question": question["standalone_question"],
                "language": question["language"],
                "question_type": phase5_rows[question["id"]].get(
                    "classified_question_type", question["question_type"]
                ),
                "locked_document": _document_id(gold[0]),
                "gold_chunk_ids": gold,
                "expected_concepts": list(question.get("expected_concepts") or []),
                "gold_provenance": provenance,
            }
        )
    for question in NEW_QUESTIONS:
        gold = list(question["gold_chunk_ids"])
        rows.append(
            {
                **question,
                "locked_document": _document_id(gold[0]),
                "gold_provenance": "phase7_manual_passage_review",
            }
        )
    if len(rows) != DATASET_SIZE:
        raise ValueError(f"Expected {DATASET_SIZE} verified rows, got {len(rows)}")
    splits = freeze_splits([row["id"] for row in rows])
    for row in rows:
        missing = [chunk_id for chunk_id in row["gold_chunk_ids"] if chunk_id not in children]
        if missing:
            raise ValueError(f"Missing gold chunks for {row['id']}: {missing}")
        documents = {children[chunk_id].document_id for chunk_id in row["gold_chunk_ids"]}
        if documents != {row["locked_document"]}:
            raise ValueError(f"Cross-document gold set for {row['id']}: {documents}")
        row["split"] = splits[row["id"]]
        row["gold_verification"] = [
            {
                "chunk_id": chunk_id,
                "page_start": children[chunk_id].page_start,
                "page_end": children[chunk_id].page_end,
                "section": children[chunk_id].section,
                "text_sha256": children[chunk_id].sha256,
                "reviewed_excerpt": children[chunk_id].text[:240],
            }
            for chunk_id in row["gold_chunk_ids"]
        ]
    return sorted(rows, key=lambda item: item["id"])


def dataset_manifest(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    serialized = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    return {
        "version": "candidate_preservation_v0.1",
        "active_index": ACTIVE_DIRECTORY.name,
        "frozen_before_tuning": True,
        "holdout_policy": "open once after selecting one configuration on DEV",
        "question_count": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "language_counts": dict(Counter(row["language"] for row in rows)),
        "document_counts": dict(Counter(row["locked_document"] for row in rows)),
        "question_type_counts": dict(Counter(row["question_type"] for row in rows)),
        "gold_provenance_counts": dict(Counter(row["gold_provenance"] for row in rows)),
        "questions_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }


def prepare_dataset(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    rows = build_verified_dataset()
    output_directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_directory / "questions.jsonl", rows)
    manifest = dataset_manifest(rows)
    (output_directory / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def round_robin_union(rankings: Sequence[Sequence[str]], limit: int) -> list[str]:
    """Fairly interleave rankings while removing duplicates."""

    output: list[str] = []
    seen: set[str] = set()
    depth = max((len(ranking) for ranking in rankings), default=0)
    for rank in range(depth):
        for ranking in rankings:
            if rank >= len(ranking):
                continue
            chunk_id = ranking[rank]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            output.append(chunk_id)
            if len(output) == limit:
                return output
    return output


def compose_candidates(
    retriever_rankings: dict[str, Sequence[str]],
    fused_ranking: Sequence[str],
    *,
    budget: int,
    reserve_k: int = 0,
    union_first_k: int | None = None,
) -> list[str]:
    """Reserve modality candidates before fusion truncation, then fill by fusion."""

    if budget <= 0 or reserve_k < 0:
        raise ValueError("budget must be positive and reserve_k non-negative")
    rankings = [retriever_rankings[name] for name in RETRIEVERS]
    if union_first_k is not None:
        output = round_robin_union([ranking[:union_first_k] for ranking in rankings], budget)
    elif reserve_k:
        output = round_robin_union([ranking[:reserve_k] for ranking in rankings], budget)
    else:
        output = []
    seen = set(output)
    for chunk_id in fused_ranking:
        if chunk_id not in seen:
            seen.add(chunk_id)
            output.append(chunk_id)
            if len(output) == budget:
                break
    return output


def contribution_class(
    chunk_id: str,
    *,
    retriever_rankings: dict[str, Sequence[str]],
    fused_ranking: Sequence[str],
    budget: int,
    reserve_k: int,
) -> str:
    """Explain which reservation rescued a chunk absent from fused top-budget."""

    sources = [
        name
        for name in RETRIEVERS
        if chunk_id in set(retriever_rankings[name][:reserve_k])
    ]
    if len(sources) > 1:
        return "multiple"
    if sources:
        return f"{sources[0]}_rescue"
    if chunk_id in set(fused_ranking):
        return "fusion_rescue"
    return "not_preserved"


def _best_raw_values(
    raw: dict[str, Any], retriever: str
) -> tuple[dict[str, int], dict[str, float]]:
    best_rank: dict[str, int] = {}
    best_score: dict[str, float] = {}
    for run in raw["runs"][retriever]:
        for rank, (chunk_id, score) in enumerate(zip(run["ids"], run["scores"], strict=True), 1):
            if rank < best_rank.get(chunk_id, 10**9):
                best_rank[chunk_id] = rank
                best_score[chunk_id] = float(score)
    return best_rank, best_score


def make_hybrid_candidates(
    ids: Sequence[str], raw: dict[str, Any], chunks: dict[str, DocumentChunk]
) -> list[HybridSearchResult]:
    """Adapt composed IDs to the production reranker's public candidate type."""

    values = {name: _best_raw_values(raw, name) for name in RETRIEVERS}
    output: list[HybridSearchResult] = []
    for rank, chunk_id in enumerate(ids, 1):
        matched = tuple(name for name in RETRIEVERS if chunk_id in values[name][0])
        output.append(
            HybridSearchResult(
                rank=rank,
                rrf_score=1.0 / rank,
                matched_retrievers=matched,
                dense_rank=values["dense"][0].get(chunk_id),
                dense_score=values["dense"][1].get(chunk_id),
                dense_rrf_contribution=0.0,
                sparse_rank=values["sparse"][0].get(chunk_id),
                sparse_score=values["sparse"][1].get(chunk_id),
                sparse_rrf_contribution=0.0,
                bm25_rank=values["bm25"][0].get(chunk_id),
                bm25_score=values["bm25"][1].get(chunk_id),
                bm25_rrf_contribution=0.0,
                chunk=chunks[chunk_id],
            )
        )
    return output


def _score_candidate_union(
    engine: QualityRetrievalEngine,
    trace: dict[str, Any],
    query: str,
    candidates: list[HybridSearchResult],
) -> tuple[dict[str, float], float]:
    response = engine.reranker.rerank(query, candidates, top_k=len(candidates))
    routing = _locked_routing(engine, trace)
    adjusted, _ = engine._adjust_reranking(
        response, routing=routing, question_type=trace["question_type"]
    )
    return (
        {item.chunk.chunk_id: float(item.reranker_score) for item in adjusted.results},
        float(response.reranking_duration_ms),
    )


def rank_by_scores(ids: Sequence[str], scores: dict[str, float]) -> list[str]:
    original = {chunk_id: rank for rank, chunk_id in enumerate(ids, 1)}
    return sorted(ids, key=lambda item: (-scores[item], original[item], item))


def candidate_configurations(raw: dict[str, Any]) -> dict[str, list[str]]:
    retriever_rankings = {
        name: aggregate_retriever_ids(raw, name) for name in RETRIEVERS
    }
    fusions = {
        "pure_rrf": fuse_raw(raw),
        "phase6_weighted_rrf": fuse_raw(
            raw, weights={"dense": 0.5, "sparse": 2.0, "bm25": 1.0}
        ),
        "normalized": fuse_raw(raw, method="normalized"),
    }
    output: dict[str, list[str]] = {}
    for fusion_name, fused in fusions.items():
        for budget in BUDGETS:
            output[f"{fusion_name}__fused__b{budget}"] = compose_candidates(
                retriever_rankings, fused, budget=budget
            )
            for reserve in RESERVATIONS:
                output[f"{fusion_name}__reserve{reserve}__b{budget}"] = compose_candidates(
                    retriever_rankings, fused, budget=budget, reserve_k=reserve
                )
    for budget in BUDGETS:
        output[f"union_first50__b{budget}"] = compose_candidates(
            retriever_rankings,
            fusions["pure_rrf"],
            budget=budget,
            union_first_k=50,
        )
    return output


def evaluate_trace_no_colbert(
    engine: QualityRetrievalEngine, trace: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate every preservation policy with one shared, label-free score pass."""

    started = time.perf_counter()
    compatible_trace = {
        **trace,
        "classified_question_type": trace["question_type"],
    }
    plan, formulations = query_formulations(compatible_trace, engine.catalog)
    raw = run_raw_retrieval(
        engine,
        formulations["current"],
        document_ids={trace["locked_document"]},
        depth=RAW_DEPTH,
    )
    configurations = candidate_configurations(raw)
    union_ids = list(dict.fromkeys(item for ids in configurations.values() for item in ids))
    indexed_chunks = {
        chunk.chunk_id: chunk for chunk in engine.retriever.dense_retriever.metadata
    }
    candidates = make_hybrid_candidates(union_ids, raw, indexed_chunks)
    scores, score_latency = _score_candidate_union(
        engine, compatible_trace, plan.base_query, candidates
    )
    rankings = {name: rank_by_scores(ids, scores) for name, ids in configurations.items()}
    retriever_rankings = {
        name: aggregate_retriever_ids(raw, name) for name in RETRIEVERS
    }
    gold = set(trace["gold_chunk_ids"])
    rescue_events: list[dict[str, str]] = []
    for config_name, ids in configurations.items():
        if "__reserve" not in config_name:
            continue
        fusion_name, reservation, budget_value = config_name.split("__")
        reserve_k = int(reservation.removeprefix("reserve"))
        budget = int(budget_value.removeprefix("b"))
        fused = configurations[f"{fusion_name}__fused__b{budget}"]
        for chunk_id in sorted((set(ids) & gold) - (set(fused) & gold)):
            rescue_events.append(
                {
                    "configuration": config_name,
                    "chunk_id": chunk_id,
                    "contribution": contribution_class(
                        chunk_id,
                        retriever_rankings=retriever_rankings,
                        fused_ranking=fuse_raw(
                            raw,
                            method="normalized" if fusion_name == "normalized" else "rrf",
                            weights=(
                                {"dense": 0.5, "sparse": 2.0, "bm25": 1.0}
                                if fusion_name == "phase6_weighted_rrf"
                                else None
                            ),
                        ),
                        budget=budget,
                        reserve_k=reserve_k,
                    ),
                }
            )
    return {
        "id": trace["id"],
        "question": trace["question"],
        "language": trace["language"],
        "question_type": trace["question_type"],
        "split": trace["split"],
        "locked_document": trace["locked_document"],
        "gold_chunk_ids": trace["gold_chunk_ids"],
        "query": plan.base_query,
        "raw_latency_ms": raw["latency_ms"],
        "raw_rankings": retriever_rankings,
        "candidate_ids": configurations,
        "reranked_ids": rankings,
        "candidate_scores": scores,
        "shared_reranker_latency_ms": score_latency,
        "shared_reranker_candidate_count": len(candidates),
        "rescue_events": rescue_events,
        "total_latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def evaluate_current_colbert(
    engine: QualityRetrievalEngine, trace: dict[str, Any]
) -> dict[str, Any]:
    """Run the current production candidate path (RRF + ColBERT, pool 30)."""

    compatible_trace = {
        **trace,
        "classified_question_type": trace["question_type"],
    }
    plan, _formulations = query_formulations(compatible_trace, engine.catalog)
    routing = _locked_routing(engine, compatible_trace)
    expanded = expand_technical_query(
        trace["question"],
        standalone_query=plan.base_query,
        question_type=trace["question_type"],
    )
    section_response = (
        engine.section_retriever.search(
            expanded,
            question_type=trace["question_type"],
            routing=routing,
            top_k=12,
            candidate_k=40,
        )
        if engine.section_retriever is not None
        else None
    )
    hybrid = search_planned_hybrid(
        engine.retriever,
        plan,
        sparse_retriever=engine.sparse_retriever,
        top_k=30,
        dense_candidate_k=50,
        sparse_candidate_k=50,
        bm25_candidate_k=50,
        fusion_k=80,
        colbert_candidate_k=80,
        document_ids={trace["locked_document"]},
        section_bonus_by_chunk=engine._section_bonus_by_chunk(section_response),
    )
    response = engine.reranker.rerank(
        plan.base_query, list(hybrid.results), top_k=len(hybrid.results)
    )
    adjusted, _boosts = engine._adjust_reranking(
        response, routing=routing, question_type=trace["question_type"]
    )
    return {
        "candidate_ids": [item.chunk.chunk_id for item in hybrid.results],
        "reranked_ids": [item.chunk.chunk_id for item in adjusted.results],
        "first_stage_latency_ms": {
            "dense": float(hybrid.dense_duration_ms),
            "sparse": float(hybrid.sparse_duration_ms),
            "bm25": float(hybrid.bm25_duration_ms),
            "total": float(hybrid.total_duration_ms),
        },
        "reranker_latency_ms": float(response.reranking_duration_ms),
    }


def evaluate_selected_trace(
    engine: QualityRetrievalEngine,
    trace: dict[str, Any],
    selected_configuration: str,
) -> dict[str, Any]:
    """Evaluate only the DEV-locked candidate policy on one holdout row."""

    started = time.perf_counter()
    compatible_trace = {
        **trace,
        "classified_question_type": trace["question_type"],
    }
    plan, formulations = query_formulations(compatible_trace, engine.catalog)
    raw = run_raw_retrieval(
        engine,
        formulations["current"],
        document_ids={trace["locked_document"]},
        depth=RAW_DEPTH,
    )
    ids = candidate_configurations(raw)[selected_configuration]
    indexed_chunks = {
        chunk.chunk_id: chunk for chunk in engine.retriever.dense_retriever.metadata
    }
    candidates = make_hybrid_candidates(ids, raw, indexed_chunks)
    scores, reranker_latency = _score_candidate_union(
        engine, compatible_trace, plan.base_query, candidates
    )
    current = evaluate_current_colbert(engine, trace)
    return {
        "id": trace["id"],
        "question": trace["question"],
        "language": trace["language"],
        "question_type": trace["question_type"],
        "split": trace["split"],
        "locked_document": trace["locked_document"],
        "gold_chunk_ids": trace["gold_chunk_ids"],
        "selected_configuration": selected_configuration,
        "candidate_ids": ids,
        "reranked_ids": rank_by_scores(ids, scores),
        "raw_latency_ms": raw["latency_ms"],
        "reranker_latency_ms": reranker_latency,
        "current_colbert": current,
        "total_latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def reranker_input_variants(chunk: DocumentChunk, engine: QualityRetrievalEngine) -> dict[str, str]:
    """Build the four controlled passage representations requested by Phase 7."""

    passage = clean_passage_text(chunk.text)
    section = " > ".join(chunk.heading_path) or (chunk.section or "")
    document = chunk.document_title or Path(chunk.source_file).stem.replace("_", " ")
    section_passage = (
        f"Section: {section}\n\nPassage:\n{passage}" if section else passage
    )
    return {
        "passage_only": passage,
        "section_passage": section_passage,
        "document_section_passage": (
            f"Document: {document}\n{section_passage}" if document else section_passage
        ),
        "current": build_reranking_passage(chunk, engine.reranker.config),
    }


def run_reranker_diagnostics(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Inspect exact DQ003/DQ039 inputs and score-local competitors."""

    dev_summary = json.loads(
        (output_directory / "dev_summary.json").read_text(encoding="utf-8")
    )
    selected = str(dev_summary["selected_configuration"])
    dev_rows = {row["id"]: row for row in read_jsonl(output_directory / "dev_results.jsonl")}
    questions = {row["id"]: row for row in read_jsonl(output_directory / "questions.jsonl")}
    rag = PhosProcessRAG()
    engine = rag.quality_engine
    if engine is None:
        raise RuntimeError("Production quality engine is required")
    chunks = {chunk.chunk_id: chunk for chunk in engine.retriever.dense_retriever.metadata}
    output: dict[str, Any] = {}
    for question_id in ("DQ003", "DQ039"):
        row = dev_rows[question_id]
        truth = questions[question_id]
        candidate_ids = list(row["candidate_ids"][selected])
        diagnostic_ids = list(
            dict.fromkeys([*candidate_ids, *truth["gold_chunk_ids"]])
        )
        variants: dict[str, Any] = {}
        for variant_name in reranker_input_variants(chunks[diagnostic_ids[0]], engine):
            passages = {
                chunk_id: reranker_input_variants(chunks[chunk_id], engine)[variant_name]
                for chunk_id in diagnostic_ids
            }
            pairs = [[row["query"], passages[chunk_id]] for chunk_id in diagnostic_ids]
            started = time.perf_counter()
            scores = engine.reranker._compute_scores(pairs=pairs)
            latency = (time.perf_counter() - started) * 1000.0
            scored = sorted(
                zip(diagnostic_ids, scores, strict=True),
                key=lambda item: (-float(item[1]), diagnostic_ids.index(item[0]), item[0]),
            )
            ranked_ids = [item[0] for item in scored]
            score_by_id = {chunk_id: float(score) for chunk_id, score in scored}
            gold_details = []
            neighbor_ids: set[str] = set()
            for gold_id in truth["gold_chunk_ids"]:
                rank = ranked_ids.index(gold_id) + 1
                neighbor_ids.update(ranked_ids[max(0, rank - 3) : rank + 2])
                gold_details.append(
                    {
                        "chunk_id": gold_id,
                        "was_in_candidate_pool": gold_id in candidate_ids,
                        "rank_in_diagnostic_pool": rank,
                        "score": score_by_id[gold_id],
                        "exact_reranker_input": passages[gold_id],
                    }
                )
            variants[variant_name] = {
                "latency_ms": round(latency, 3),
                "gold": gold_details,
                "top_5": [
                    {
                        "rank": rank,
                        "chunk_id": chunk_id,
                        "score": float(score),
                        "section": chunks[chunk_id].section,
                        "excerpt": chunks[chunk_id].text[:300],
                    }
                    for rank, (chunk_id, score) in enumerate(scored[:5], 1)
                ],
                "gold_score_neighbors": [
                    {
                        "rank": ranked_ids.index(chunk_id) + 1,
                        "chunk_id": chunk_id,
                        "score": score_by_id[chunk_id],
                        "section": chunks[chunk_id].section,
                        "exact_reranker_input": passages[chunk_id],
                    }
                    for chunk_id in ranked_ids
                    if chunk_id in neighbor_ids
                ],
            }
        output[question_id] = {
            "question": truth["question"],
            "reranker_query": row["query"],
            "selected_candidate_configuration": selected,
            "candidate_count": len(candidate_ids),
            "variants": variants,
        }
    path = output_directory / "reranker_input_diagnostics.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


SEARCH_PLANNER_IDS = ("DQ027", "CE051", "DQ011", "CE056")
SEARCH_PLANNER_SYSTEM = """You are a multilingual search-query planner.
Return only JSON matching the supplied schema: standalone_query plus
search_queries. Create 1 to 4 concise retrieval queries from the user's question
alone. Keep the original question verbatim as standalone_query. You may
decompose or paraphrase technical concepts, but do not invent
document titles, authors, pages, sections, expected answers, or hidden facts.
If the user explicitly names a source, that wording may be retained. No prose."""


def _freeze_planner_queries(original: str, proposed: Sequence[str]) -> list[str]:
    output = [original]
    seen = {original.casefold().strip()}
    for query in proposed:
        cleaned = query.strip()
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
        if len(output) == 4:
            break
    return output


def _evaluate_search_plan(
    engine: QualityRetrievalEngine,
    trace: dict[str, Any],
    queries: Sequence[str],
    selected_configuration: str,
) -> dict[str, Any]:
    compatible = {**trace, "classified_question_type": trace["question_type"]}
    specs = [QuerySpec(f"qwen_{index}", query, query, query) for index, query in enumerate(queries)]
    raw = run_raw_retrieval(
        engine, specs, document_ids={trace["locked_document"]}, depth=RAW_DEPTH
    )
    ids = candidate_configurations(raw)[selected_configuration]
    chunks = {chunk.chunk_id: chunk for chunk in engine.retriever.dense_retriever.metadata}
    candidates = make_hybrid_candidates(ids, raw, chunks)
    plan, _formulations = query_formulations(compatible, engine.catalog)
    scores, latency = _score_candidate_union(engine, compatible, plan.base_query, candidates)
    raw_rankings = {name: aggregate_retriever_ids(raw, name) for name in RETRIEVERS}
    gold = set(trace["gold_chunk_ids"])
    return {
        "raw_gold_ranks": {
            name: next(
                (rank for rank, chunk_id in enumerate(ranking, 1) if chunk_id in gold), None
            )
            for name, ranking in raw_rankings.items()
        },
        "candidate_ids": ids,
        "reranked_ids": rank_by_scores(ids, scores),
        "candidate_gold_rank": next(
            (rank for rank, chunk_id in enumerate(ids, 1) if chunk_id in gold), None
        ),
        "final_gold_rank": next(
            (
                rank
                for rank, chunk_id in enumerate(rank_by_scores(ids, scores), 1)
                if chunk_id in gold
            ),
            None,
        ),
        "raw_latency_ms": raw["latency_ms"],
        "reranker_latency_ms": latency,
    }


def run_search_planner(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Freeze label-free Qwen plans, then measure DEV hard misses only."""

    questions = {row["id"]: row for row in read_jsonl(output_directory / "questions.jsonl")}
    llm = OllamaLLM(load_ollama_config(PROJECT_ROOT / "configs/rag_production.yaml"))
    plans: dict[str, Any] = {}
    for question_id in SEARCH_PLANNER_IDS:
        truth = questions[question_id]
        payload, raw_response = llm.chat_json_with_raw(
            user_prompt=truth["question"],
            system_prompt=SEARCH_PLANNER_SYSTEM,
            response_model=SearchPlanPayload,
        )
        plans[question_id] = {
            "question": truth["question"],
            "split": truth["split"],
            "queries": _freeze_planner_queries(
                truth["question"],
                [payload.standalone_query, *payload.search_queries],
            ),
            "raw_response": raw_response,
            "prompt_received_gold": False,
            "prompt_received_locked_document": False,
        }
    frozen_payload = json.dumps(plans, ensure_ascii=False, sort_keys=True)
    plan_file = {
        "system_prompt": SEARCH_PLANNER_SYSTEM,
        "plans": plans,
        "plans_sha256": hashlib.sha256(frozen_payload.encode("utf-8")).hexdigest(),
        "frozen_before_holdout": True,
    }
    (output_directory / "qwen_search_plans.json").write_text(
        json.dumps(plan_file, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dev_summary = json.loads(
        (output_directory / "dev_summary.json").read_text(encoding="utf-8")
    )
    selected = str(dev_summary["selected_configuration"])
    rag = PhosProcessRAG()
    engine = rag.quality_engine
    if engine is None or engine.sparse_retriever is None:
        raise RuntimeError("Production quality engine with BGE sparse is required")
    evaluated: dict[str, Any] = {}
    for question_id, plan in plans.items():
        trace = questions[question_id]
        if trace["split"] != "dev":
            evaluated[question_id] = {"status": "frozen_for_single_holdout_opening"}
            continue
        evaluated[question_id] = _evaluate_search_plan(
            engine, trace, plan["queries"], selected
        )
    output = {"plans_sha256": plan_file["plans_sha256"], "dev_evaluation": evaluated}
    (output_directory / "qwen_search_plan_dev_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output


def run_planner_rank_details(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Report each already-frozen planner query separately without retuning."""

    frozen = json.loads(
        (output_directory / "qwen_search_plans.json").read_text(encoding="utf-8")
    )
    questions = {row["id"]: row for row in read_jsonl(output_directory / "questions.jsonl")}
    rag = PhosProcessRAG()
    engine = rag.quality_engine
    if engine is None or engine.sparse_retriever is None:
        raise RuntimeError("Production quality engine with BGE sparse is required")
    output: dict[str, Any] = {"plans_sha256": frozen["plans_sha256"], "questions": {}}
    for question_id in SEARCH_PLANNER_IDS:
        trace = questions[question_id]
        queries = frozen["plans"][question_id]["queries"]
        specs = [
            QuerySpec(f"query_{index}", query, query, query)
            for index, query in enumerate(queries)
        ]
        raw = run_raw_retrieval(
            engine, specs, document_ids={trace["locked_document"]}, depth=RAW_DEPTH
        )
        gold = set(trace["gold_chunk_ids"])
        per_query: list[dict[str, Any]] = []
        for index, query in enumerate(queries):
            ranks: dict[str, int | None] = {}
            for retriever in RETRIEVERS:
                run = raw["runs"][retriever][index]
                ranks[retriever] = next(
                    (
                        rank
                        for rank, chunk_id in enumerate(run["ids"], 1)
                        if chunk_id in gold
                    ),
                    None,
                )
            per_query.append(
                {
                    "kind": "original" if index == 0 else f"llm_query_{index}",
                    "query": query,
                    "ranks": ranks,
                }
            )
        aggregated_ranks = {
            retriever: next(
                (
                    rank
                    for rank, chunk_id in enumerate(
                        aggregate_retriever_ids(raw, retriever), 1
                    )
                    if chunk_id in gold
                ),
                None,
            )
            for retriever in RETRIEVERS
        }
        union_ranks = {
            retriever: min(
                (
                    item["ranks"][retriever]
                    for item in per_query
                    if item["ranks"][retriever] is not None
                ),
                default=None,
            )
            for retriever in RETRIEVERS
        }
        best = min((rank for rank in union_ranks.values() if rank is not None), default=None)
        output["questions"][question_id] = {
            "split": trace["split"],
            "per_query": per_query,
            "union_best_ranks": union_ranks,
            "aggregated_multi_query_ranks": aggregated_ranks,
            "best_union_rank": best,
            "converted_to_under_30": best is not None and best < 30,
            "converted_to_under_100": best is not None and best < 100,
        }
    (output_directory / "qwen_search_plan_rank_details.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output


def _metrics(rows: Sequence[dict[str, Any]], field: str, config: str) -> dict[str, float]:
    return ranking_metrics(
        [row[field][config] for row in rows],
        [set(row["gold_chunk_ids"]) for row in rows],
    )


def summarize_dev(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Select exactly one configuration using DEV aggregate metrics only."""

    if not rows or any(row["split"] != "dev" for row in rows):
        raise ValueError("DEV selection accepts DEV rows only")
    configs = sorted(rows[0]["candidate_ids"])
    table: dict[str, Any] = {}
    scored: list[tuple[tuple[float, ...], str]] = []
    for config in configs:
        access = _metrics(rows, "candidate_ids", config)
        final = _metrics(rows, "reranked_ids", config)
        budget = int(config.rsplit("b", 1)[1])
        table[config] = {"access": access, "final": final, "budget": budget}
        objective = (
            access["recall_at_50"],
            access["recall_at_20"],
            final["recall_at_20"],
            final["recall_at_10"],
            final["mrr"],
            -float(budget),
        )
        scored.append((objective, config))
    _objective, winner = max(scored, key=lambda item: (item[0], item[1]))
    return {
        "dev_question_count": len(rows),
        "selection_policy": (
            "max Access R@50, Access R@20, final R@20, final R@10, MRR, then lower budget"
        ),
        "selected_configuration": winner,
        "selected_metrics": table[winner],
        "configuration_metrics": table,
        "holdout_opened": False,
    }


def run_dev(output_directory: Path = DEFAULT_OUTPUT, *, limit: int | None = None) -> dict[str, Any]:
    dataset_path = output_directory / "questions.jsonl"
    if not dataset_path.is_file():
        prepare_dataset(output_directory)
    traces = [row for row in read_jsonl(dataset_path) if row["split"] == "dev"]
    if limit is not None:
        traces = traces[:limit]
    rag = PhosProcessRAG()
    engine = rag.quality_engine
    if engine is None or engine.sparse_retriever is None:
        raise RuntimeError("Production quality engine with BGE sparse is required")
    results_path = output_directory / "dev_results.jsonl"
    persisted = read_jsonl(results_path) if results_path.is_file() else []
    wanted_ids = {trace["id"] for trace in traces}
    rows = [row for row in persisted if row["id"] in wanted_ids]
    completed_ids = {row["id"] for row in rows}
    pending = [trace for trace in traces if trace["id"] not in completed_ids]
    for index, trace in enumerate(pending, len(rows) + 1):
        print(f"[DEV {index}/{len(traces)}] {trace['id']}", flush=True)
        rows.append(evaluate_trace_no_colbert(engine, trace))
        rows.sort(key=lambda item: item["id"])
        write_jsonl(results_path, rows)
    summary = summarize_dev(rows)
    (output_directory / "dev_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run_dev_colbert_baseline(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Measure the unchanged current candidate path on the already frozen DEV rows."""

    traces = [
        row
        for row in read_jsonl(output_directory / "questions.jsonl")
        if row["split"] == "dev"
    ]
    rag = PhosProcessRAG()
    engine = rag.quality_engine
    if engine is None or engine.sparse_retriever is None:
        raise RuntimeError("Production quality engine with BGE sparse is required")
    results_path = output_directory / "dev_current_colbert_results.jsonl"
    rows = read_jsonl(results_path) if results_path.is_file() else []
    completed = {row["id"] for row in rows}
    for index, trace in enumerate(
        (row for row in traces if row["id"] not in completed), len(rows) + 1
    ):
        print(f"[DEV ColBERT {index}/{len(traces)}] {trace['id']}", flush=True)
        rows.append(
            {
                "id": trace["id"],
                "gold_chunk_ids": trace["gold_chunk_ids"],
                **evaluate_current_colbert(engine, trace),
            }
        )
        rows.sort(key=lambda item: item["id"])
        write_jsonl(results_path, rows)
    summary = {
        "question_count": len(rows),
        "access": ranking_metrics(
            [row["candidate_ids"] for row in rows],
            [set(row["gold_chunk_ids"]) for row in rows],
        ),
        "final": ranking_metrics(
            [row["reranked_ids"] for row in rows],
            [set(row["gold_chunk_ids"]) for row in rows],
        ),
        "average_first_stage_latency_ms": mean(
            row["first_stage_latency_ms"]["total"] for row in rows
        ),
        "average_reranker_latency_ms": mean(row["reranker_latency_ms"] for row in rows),
    }
    (output_directory / "dev_current_colbert_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _paired_recall_at_20(
    rows: Sequence[dict[str, Any]], candidate_field: str, baseline_path: tuple[str, ...]
) -> dict[str, float]:
    candidate_values: list[float] = []
    baseline_values: list[float] = []
    for row in rows:
        gold = set(row["gold_chunk_ids"])
        candidate_values.append(len(set(row[candidate_field][:20]) & gold) / len(gold))
        baseline: Any = row
        for part in baseline_path:
            baseline = baseline[part]
        baseline_values.append(len(set(baseline[:20]) & gold) / len(gold))
    differences = [
        candidate - baseline
        for candidate, baseline in zip(candidate_values, baseline_values, strict=True)
    ]
    generator = random.Random(7)
    bootstrap = sorted(
        mean(generator.choice(differences) for _index in differences)
        for _sample in range(10_000)
    )
    return {
        "candidate": mean(candidate_values),
        "baseline": mean(baseline_values),
        "absolute_difference": mean(differences),
        "wins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "losses": sum(value < 0 for value in differences),
        "bootstrap_95pct_ci": [bootstrap[249], bootstrap[9749]],
        "statistically_stronger": bootstrap[249] > 0.0,
    }


def run_holdout(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Open FINAL HOLDOUT once and evaluate only the DEV-selected policy."""

    dev_summary = json.loads(
        (output_directory / "dev_summary.json").read_text(encoding="utf-8")
    )
    selected = str(dev_summary["selected_configuration"])
    manifest = json.loads(
        (output_directory / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    open_path = output_directory / "holdout_open_manifest.json"
    open_record = {
        "dataset_sha256": manifest["questions_sha256"],
        "selected_configuration": selected,
        "policy": "single opening; resume is allowed only for incomplete persisted rows",
    }
    if open_path.is_file():
        existing = json.loads(open_path.read_text(encoding="utf-8"))
        if existing != open_record:
            raise RuntimeError("Holdout was already opened with a different frozen policy")
    else:
        open_path.write_text(
            json.dumps(open_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    traces = [
        row
        for row in read_jsonl(output_directory / "questions.jsonl")
        if row["split"] == "final_holdout"
    ]
    rag = PhosProcessRAG()
    engine = rag.quality_engine
    if engine is None or engine.sparse_retriever is None:
        raise RuntimeError("Production quality engine with BGE sparse is required")
    results_path = output_directory / "holdout_results.jsonl"
    rows = read_jsonl(results_path) if results_path.is_file() else []
    completed = {row["id"] for row in rows}
    for index, trace in enumerate(
        (row for row in traces if row["id"] not in completed), len(rows) + 1
    ):
        print(f"[HOLDOUT {index}/{len(traces)}] {trace['id']}", flush=True)
        rows.append(evaluate_selected_trace(engine, trace, selected))
        rows.sort(key=lambda item: item["id"])
        write_jsonl(results_path, rows)
    gold_sets = [set(row["gold_chunk_ids"]) for row in rows]
    qwen_holdout: dict[str, Any] | None = None
    planner_path = output_directory / "qwen_search_plans.json"
    if planner_path.is_file():
        planner = json.loads(planner_path.read_text(encoding="utf-8"))
        dq027 = next(row for row in traces if row["id"] == "DQ027")
        qwen_holdout = {
            "plans_sha256": planner["plans_sha256"],
            "DQ027": _evaluate_search_plan(
                engine,
                dq027,
                planner["plans"]["DQ027"]["queries"],
                selected,
            ),
        }
        (output_directory / "qwen_search_plan_holdout_result.json").write_text(
            json.dumps(qwen_holdout, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    summary = {
        "holdout_opened": True,
        "question_count": len(rows),
        "selected_configuration": selected,
        "candidate_access": ranking_metrics(
            [row["candidate_ids"] for row in rows], gold_sets
        ),
        "candidate_final": ranking_metrics(
            [row["reranked_ids"] for row in rows], gold_sets
        ),
        "current_colbert_access": ranking_metrics(
            [row["current_colbert"]["candidate_ids"] for row in rows], gold_sets
        ),
        "current_colbert_final": ranking_metrics(
            [row["current_colbert"]["reranked_ids"] for row in rows], gold_sets
        ),
        "paired_final_recall_at_20": _paired_recall_at_20(
            rows, "reranked_ids", ("current_colbert", "reranked_ids")
        ),
        "average_latency_ms": {
            "candidate_first_stage": mean(
                sum(row["raw_latency_ms"].values()) for row in rows
            ),
            "candidate_reranker": mean(row["reranker_latency_ms"] for row in rows),
            "current_colbert_first_stage": mean(
                row["current_colbert"]["first_stage_latency_ms"]["total"] for row in rows
            ),
            "current_colbert_reranker": mean(
                row["current_colbert"]["reranker_latency_ms"] for row in rows
            ),
        },
        "qwen_frozen_hard_miss_diagnostic": qwen_holdout,
    }
    (output_directory / "holdout_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dev_summary["holdout_opened"] = True
    (output_directory / "dev_summary.json").write_text(
        json.dumps(dev_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _minimal_candidates(
    ids: Sequence[str], chunks: dict[str, DocumentChunk]
) -> list[HybridSearchResult]:
    return [
        HybridSearchResult(
            rank=rank,
            rrf_score=1.0 / rank,
            matched_retrievers=(),
            dense_rank=None,
            dense_score=None,
            dense_rrf_contribution=0.0,
            bm25_rank=None,
            bm25_score=None,
            bm25_rrf_contribution=0.0,
            chunk=chunks[chunk_id],
        )
        for rank, chunk_id in enumerate(ids, 1)
    ]


def run_latency_benchmark(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Measure actual reranker latency at each budget on a frozen DEV sample."""

    rows = read_jsonl(output_directory / "dev_results.jsonl")
    sample = sorted(rows, key=lambda row: (_sha256(row["id"]), row["id"]))[:12]
    rag = PhosProcessRAG()
    engine = rag.quality_engine
    if engine is None:
        raise RuntimeError("Production quality engine is required")
    chunks = {chunk.chunk_id: chunk for chunk in engine.retriever.dense_retriever.metadata}
    measurements: list[dict[str, Any]] = []
    for index, row in enumerate(sample, 1):
        print(f"[LATENCY {index}/{len(sample)}] {row['id']}", flush=True)
        for budget in BUDGETS:
            config = f"pure_rrf__reserve10__b{budget}"
            ids = row["candidate_ids"][config]
            response = engine.reranker.rerank(
                row["query"], _minimal_candidates(ids, chunks), top_k=len(ids)
            )
            measurements.append(
                {
                    "id": row["id"],
                    "budget": budget,
                    "candidate_count": len(ids),
                    "reranker_latency_ms": float(response.reranking_duration_ms),
                }
            )
    output = {
        "sample_policy": "12 lowest SHA-256 DEV question IDs; label-free",
        "sample_ids": [row["id"] for row in sample],
        "sample_size": len(sample),
        "average_first_stage_latency_ms": mean(
            sum(row["raw_latency_ms"].values()) for row in sample
        ),
        "budgets": {
            str(budget): {
                "average_candidate_count": mean(
                    item["candidate_count"]
                    for item in measurements
                    if item["budget"] == budget
                ),
                "average_reranker_latency_ms": mean(
                    item["reranker_latency_ms"]
                    for item in measurements
                    if item["budget"] == budget
                ),
            }
            for budget in BUDGETS
        },
        "measurements": measurements,
    }
    (output_directory / "latency_by_budget.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output


def _evaluation_records(output_directory: Path) -> list[dict[str, Any]]:
    questions = {row["id"]: row for row in read_jsonl(output_directory / "questions.jsonl")}
    dev_summary = json.loads(
        (output_directory / "dev_summary.json").read_text(encoding="utf-8")
    )
    selected = str(dev_summary["selected_configuration"])
    dev_candidate = {
        row["id"]: row for row in read_jsonl(output_directory / "dev_results.jsonl")
    }
    dev_current = {
        row["id"]: row
        for row in read_jsonl(output_directory / "dev_current_colbert_results.jsonl")
    }
    holdout = {
        row["id"]: row for row in read_jsonl(output_directory / "holdout_results.jsonl")
    }
    output: list[dict[str, Any]] = []
    for question_id, truth in questions.items():
        if truth["split"] == "dev":
            candidate = dev_candidate[question_id]
            current = dev_current[question_id]
            candidate_ids = candidate["candidate_ids"][selected]
            reranked_ids = candidate["reranked_ids"][selected]
            current_candidates = current["candidate_ids"]
            current_reranked = current["reranked_ids"]
        else:
            row = holdout[question_id]
            candidate_ids = row["candidate_ids"]
            reranked_ids = row["reranked_ids"]
            current_candidates = row["current_colbert"]["candidate_ids"]
            current_reranked = row["current_colbert"]["reranked_ids"]
        output.append(
            {
                **truth,
                "candidate_ids": candidate_ids,
                "reranked_ids": reranked_ids,
                "current_candidate_ids": current_candidates,
                "current_reranked_ids": current_reranked,
            }
        )
    return sorted(output, key=lambda row: row["id"])


def _record_metrics(rows: Sequence[dict[str, Any]], prefix: str) -> dict[str, float]:
    return ranking_metrics(
        [row[f"{prefix}ids"] for row in rows],
        [set(row["gold_chunk_ids"]) for row in rows],
    )


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def build_phase7_summary(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    records = _evaluation_records(output_directory)
    manifest = json.loads(
        (output_directory / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    dev = [row for row in records if row["split"] == "dev"]
    holdout = [row for row in records if row["split"] == "final_holdout"]
    language = {
        value: {
            "questions": len(group),
            "candidate_final": _record_metrics(group, "reranked_"),
            "current_final": _record_metrics(group, "current_reranked_"),
        }
        for value in ("fr", "en", "ar")
        if (group := [row for row in records if row["language"] == value])
    }
    access_failures = []
    reranker_failures = []
    for row in records:
        gold = set(row["gold_chunk_ids"])
        missing_access = sorted(gold - set(row["candidate_ids"]))
        missing_final = sorted(
            (gold & set(row["candidate_ids"])) - set(row["reranked_ids"][:20])
        )
        if missing_access:
            access_failures.append({"id": row["id"], "gold": missing_access})
        if missing_final:
            reranker_failures.append({"id": row["id"], "gold": missing_final})
    dev_rows = read_jsonl(output_directory / "dev_results.jsonl")
    selected = "pure_rrf__reserve10__b50"
    contribution_rows: list[dict[str, str]] = []
    for row in dev_rows:
        gold = set(row["gold_chunk_ids"])
        for chunk_id in sorted(gold & set(row["candidate_ids"][selected])):
            sources = [
                name
                for name in RETRIEVERS
                if chunk_id in set(row["raw_rankings"][name][:10])
            ]
            contribution = (
                "multiple"
                if len(sources) > 1
                else f"{sources[0]}_rescue"
                if sources
                else "fusion_rescue"
            )
            contribution_rows.append(
                {"id": row["id"], "chunk_id": chunk_id, "contribution": contribution}
            )
    contribution_counts = dict(Counter(row["contribution"] for row in contribution_rows))
    return {
        "dataset": manifest,
        "selected_configuration": selected,
        "all_64": {
            "candidate_access": _record_metrics(records, "candidate_"),
            "candidate_final": _record_metrics(records, "reranked_"),
            "current_access": _record_metrics(records, "current_candidate_"),
            "current_final": _record_metrics(records, "current_reranked_"),
        },
        "dev": {
            "candidate_access": _record_metrics(dev, "candidate_"),
            "candidate_final": _record_metrics(dev, "reranked_"),
            "current_access": _record_metrics(dev, "current_candidate_"),
            "current_final": _record_metrics(dev, "current_reranked_"),
        },
        "holdout": {
            "candidate_access": _record_metrics(holdout, "candidate_"),
            "candidate_final": _record_metrics(holdout, "reranked_"),
            "current_access": _record_metrics(holdout, "current_candidate_"),
            "current_final": _record_metrics(holdout, "current_reranked_"),
        },
        "multilingual": language,
        "contributions_dev": {
            "counts": contribution_counts,
            "rows": contribution_rows,
        },
        "access_failures": access_failures,
        "reranker_failures": reranker_failures,
        "production_change": {
            "selected": "none",
            "target_reached": False,
            "statistically_stronger_on_holdout": False,
            "reason": (
                "Holdout final Recall@20 is 73.68% for both candidate preservation "
                "and current ColBERT; target 85% is not reached."
            ),
        },
    }


def render_phase7_report(output_directory: Path = DEFAULT_OUTPUT) -> str:
    summary = build_phase7_summary(output_directory)
    dataset = summary["dataset"]
    dev_grid = json.loads(
        (output_directory / "dev_summary.json").read_text(encoding="utf-8")
    )
    current_dev = json.loads(
        (output_directory / "dev_current_colbert_summary.json").read_text(encoding="utf-8")
    )
    holdout = json.loads(
        (output_directory / "holdout_summary.json").read_text(encoding="utf-8")
    )
    latency = json.loads(
        (output_directory / "latency_by_budget.json").read_text(encoding="utf-8")
    )
    diagnostics = json.loads(
        (output_directory / "reranker_input_diagnostics.json").read_text(encoding="utf-8")
    )
    planner = json.loads(
        (output_directory / "qwen_search_plan_rank_details.json").read_text(encoding="utf-8")
    )
    gates_path = output_directory / "test_gates.json"
    gates = (
        json.loads(gates_path.read_text(encoding="utf-8"))
        if gates_path.is_file()
        else {"status": "pending final gate run"}
    )
    gold_assignment_count = sum(
        len(row["gold_chunk_ids"])
        for row in read_jsonl(output_directory / "questions.jsonl")
    )
    holdout_candidate_total = (
        holdout["average_latency_ms"]["candidate_first_stage"]
        + holdout["average_latency_ms"]["candidate_reranker"]
    )
    holdout_current_total = (
        holdout["average_latency_ms"]["current_colbert_first_stage"]
        + holdout["average_latency_ms"]["current_colbert_reranker"]
    )
    matrix_names = [
        "pure_rrf__fused__b30",
        "pure_rrf__reserve5__b30",
        "pure_rrf__reserve10__b30",
        "pure_rrf__reserve15__b30",
        "union_first50__b50",
        "pure_rrf__fused__b50",
        "pure_rrf__reserve5__b50",
        "pure_rrf__reserve10__b50",
        "pure_rrf__reserve15__b50",
        "phase6_weighted_rrf__reserve10__b50",
        "normalized__reserve10__b50",
    ]
    lines = [
        "# Phase 7 — Candidate preservation and query access",
        "",
        "## 1. EXPANDED EVALUATION DATASET",
        "",
        (
            f"{dataset['question_count']} verified questions and "
            f"{gold_assignment_count} "
            "gold assignments across all 8 active documents."
        ),
        "",
        f"DEV: {dataset['split_counts']['dev']}; "
        f"FINAL HOLDOUT: {dataset['split_counts']['final_holdout']}. "
        f"Languages: `{json.dumps(dataset['language_counts'], ensure_ascii=False)}`.",
        "",
        f"Frozen dataset SHA-256: `{dataset['questions_sha256']}`. Gold provenance: "
        f"`{json.dumps(dataset['gold_provenance_counts'])}`.",
        "",
        "## 2. RERANKER ACCESS RECALL",
        "",
        "| Split | Current ColBERT access@30 | Selected access@50 |",
        "|---|---:|---:|",
    ]
    for split_name in ("dev", "holdout", "all_64"):
        values = summary[split_name]
        lines.append(
            f"| {split_name} | {_pct(values['current_access']['recall_at_50'])} | "
            f"{_pct(values['candidate_access']['recall_at_50'])} |"
        )
    lines += [
        "",
        "Selected policy: `pure RRF + reserve 10 per modality + budget 50`, chosen on DEV only.",
        "",
        "## 3. CANDIDATE COMPOSITION ABLATION",
        "",
        "| DEV configuration | Access (whole pool) | Final R@20 | MRR |",
        "|---|---:|---:|---:|",
    ]
    for name in matrix_names:
        values = dev_grid["configuration_metrics"][name]
        lines.append(
            f"| `{name}` | {_pct(values['access']['recall_at_50'])} | "
            f"{_pct(values['final']['recall_at_20'])} | {values['final']['mrr']:.3f} |"
        )
    lines += [
        "",
        "Per-retriever attribution on accessible DEV gold: "
        f"`{json.dumps(summary['contributions_dev']['counts'], ensure_ascii=False)}`. "
        "All contribution classes are computed from rank/identity signals only.",
        "",
        "## 4. COLBERT ABLATION",
        "",
        "| DEV pipeline | R@5 | R@10 | R@20 | MRR | First-stage ms |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Current RRF + ColBERT, pool 30 | {_pct(current_dev['final']['recall_at_5'])} | "
            f"{_pct(current_dev['final']['recall_at_10'])} | "
            f"{_pct(current_dev['final']['recall_at_20'])} | {current_dev['final']['mrr']:.3f} | "
            f"{current_dev['average_first_stage_latency_ms']:.1f} |"
        ),
        (
            f"| No ColBERT + selected pool 50 | "
            f"{_pct(summary['dev']['candidate_final']['recall_at_5'])} | "
            f"{_pct(summary['dev']['candidate_final']['recall_at_10'])} | "
            f"{_pct(summary['dev']['candidate_final']['recall_at_20'])} | "
            f"{summary['dev']['candidate_final']['mrr']:.3f} | "
            f"{holdout['average_latency_ms']['candidate_first_stage']:.1f} (holdout) |"
        ),
        "",
        "DEV improves, but frozen holdout final R@20 is exactly tied; "
        "ColBERT removal is not deployed.",
        "",
        "## 5. RERANKER INPUT ABLATION",
        "",
        "| Question | Passage | Section+passage | Document+section+passage | Current |",
        "|---|---:|---:|---:|---:|",
    ]
    for question_id in ("DQ003", "DQ039"):
        variants = diagnostics[question_id]["variants"]
        ranks = [
            variants[name]["gold"][0]["rank_in_diagnostic_pool"]
            for name in (
                "passage_only",
                "section_passage",
                "document_section_passage",
                "current",
            )
        ]
        lines.append(f"| {question_id} | {ranks[0]} | {ranks[1]} | {ranks[2]} | {ranks[3]} |")
    lines += [
        "",
        "DQ003 is recovered at rank 2 in every representation. DQ039 remains rank 29–32; "
        "metadata alone does not fix the semantic mismatch. Exact inputs and neighboring scores "
        "are in `reranker_input_diagnostics.json`.",
        "",
        "## 6. HARD QUERY MISSES",
        "",
        f"Candidate-access failures: {len(summary['access_failures'])}; candidates that entered "
        f"but still missed final top-20: {len(summary['reranker_failures'])}.",
        "",
        "DQ027 and CE051 remain absent from all three retriever top-200 lists under the frozen "
        "Qwen plans. DQ011 is deep; CE056 is reachable but remains outside final top-20.",
        "",
        "## 7. LLM QUERY PLANNER ABLATION",
        "",
        "| ID | Best Dense | Best Sparse | Best BM25 | <100 | <30 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for question_id in SEARCH_PLANNER_IDS:
        value = planner["questions"][question_id]
        ranks = value["union_best_ranks"]
        lines.append(
            f"| {question_id} | {ranks['dense'] or '-'} | {ranks['sparse'] or '-'} | "
            f"{ranks['bm25'] or '-'} | {str(value['converted_to_under_100']).lower()} | "
            f"{str(value['converted_to_under_30']).lower()} |"
        )
    lines += [
        "",
        "The original query was retained for every case. CE056 improves to Dense rank 14 on one "
        "LLM formulation; DQ011 reaches Sparse rank 39, but multi-query candidate "
        "composition still "
        "does not put it in the top-50 reranker pool. No generic query planner is deployed.",
        "",
        "## 8. MULTILINGUAL RESULTS",
        "",
        "| Language | n | Selected R@20 | Current R@20 |",
        "|---|---:|---:|---:|",
    ]
    for language, values in summary["multilingual"].items():
        lines.append(
            f"| {language} | {values['questions']} | "
            f"{_pct(values['candidate_final']['recall_at_20'])} | "
            f"{_pct(values['current_final']['recall_at_20'])} |"
        )
    lines += [
        "",
        "## 9. DEV RESULTS",
        "",
        "| Pipeline | R@5 | R@10 | R@20 | MRR |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Current | {_pct(summary['dev']['current_final']['recall_at_5'])} | "
            f"{_pct(summary['dev']['current_final']['recall_at_10'])} | "
            f"{_pct(summary['dev']['current_final']['recall_at_20'])} | "
            f"{summary['dev']['current_final']['mrr']:.3f} |"
        ),
        (
            f"| Selected | {_pct(summary['dev']['candidate_final']['recall_at_5'])} | "
            f"{_pct(summary['dev']['candidate_final']['recall_at_10'])} | "
            f"{_pct(summary['dev']['candidate_final']['recall_at_20'])} | "
            f"{summary['dev']['candidate_final']['mrr']:.3f} |"
        ),
        "",
        "## 10. FROZEN HOLDOUT RESULTS",
        "",
        "| Pipeline | Access | R@5 | R@10 | R@20 | MRR |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Current | {_pct(summary['holdout']['current_access']['recall_at_50'])} | "
            f"{_pct(summary['holdout']['current_final']['recall_at_5'])} | "
            f"{_pct(summary['holdout']['current_final']['recall_at_10'])} | "
            f"{_pct(summary['holdout']['current_final']['recall_at_20'])} | "
            f"{summary['holdout']['current_final']['mrr']:.3f} |"
        ),
        (
            f"| Selected | {_pct(summary['holdout']['candidate_access']['recall_at_50'])} | "
            f"{_pct(summary['holdout']['candidate_final']['recall_at_5'])} | "
            f"{_pct(summary['holdout']['candidate_final']['recall_at_10'])} | "
            f"{_pct(summary['holdout']['candidate_final']['recall_at_20'])} | "
            f"{summary['holdout']['candidate_final']['mrr']:.3f} |"
        ),
        "",
        "Paired R@20 difference = 0.00 points; bootstrap 95% CI = [0.00, 0.00]. "
        "The candidate policy is not statistically stronger.",
        "",
        "## 11. LATENCY",
        "",
        "| Candidate budget | First-stage ms | Reranker ms | Total ms |",
        "|---:|---:|---:|---:|",
    ]
    for budget, values in latency["budgets"].items():
        first = latency["average_first_stage_latency_ms"]
        reranker = values["average_reranker_latency_ms"]
        lines.append(f"| {budget} | {first:.1f} | {reranker:.1f} | {first + reranker:.1f} |")
    lines += [
        "",
        f"Frozen holdout, selected total: {holdout_candidate_total:.1f} ms; "
        f"current total: {holdout_current_total:.1f} ms.",
        "",
        "## 12. BEST ARCHITECTURE",
        "",
        "Research winner: no ColBERT, pure RRF, reserve 10 per retriever, total candidate budget "
        "50, then the unchanged BGE reranker. It is faster and improves DEV access, but is not a "
        "validated production winner because holdout quality is tied at R@20 and worse at R@5/MRR.",
        "",
        "## 13. PRODUCTION CHANGE",
        "",
        "**None.** Production retrieval remains unchanged. The 85% target was missed and the "
        "expanded holdout does not show statistical superiority. Source locks were exact in every "
        "experiment; no evaluation label entered a query or composer decision.",
        "",
        "## 14. TEST GATES",
        "",
        "```json",
        json.dumps(gates, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 15. PHASE 8 RECOMMENDATION",
        "",
        "Investigate index/chunk representations for DQ027/CE051 and generic multilingual query "
        "aggregation for CE056/DQ011. Separately study reranker semantics for DQ039. Keep absent-"
        "corpus score distributions as observational signals only; do not add a hard threshold. "
        "Do not start an Evidence Judge, answer verifier, repair loop, or agentic search loop yet.",
        "",
        "Phase 7 stops here.",
        "",
    ]
    return "\n".join(lines)


def write_phase7_report(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    summary = build_phase7_summary(output_directory)
    (output_directory / "phase7_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_directory / "summary.md").write_text(
        render_phase7_report(output_directory), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare-dataset", action="store_true")
    parser.add_argument("--run-dev", action="store_true")
    parser.add_argument("--run-dev-colbert", action="store_true")
    parser.add_argument("--reranker-diagnostics", action="store_true")
    parser.add_argument("--search-planner", action="store_true")
    parser.add_argument("--planner-rank-details", action="store_true")
    parser.add_argument("--latency-benchmark", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--open-holdout", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.prepare_dataset:
        value = prepare_dataset(args.output)
    elif args.run_dev:
        value = run_dev(args.output, limit=args.limit)
    elif args.run_dev_colbert:
        value = run_dev_colbert_baseline(args.output)
    elif args.reranker_diagnostics:
        value = run_reranker_diagnostics(args.output)
    elif args.search_planner:
        value = run_search_planner(args.output)
    elif args.planner_rank_details:
        value = run_planner_rank_details(args.output)
    elif args.latency_benchmark:
        value = run_latency_benchmark(args.output)
    elif args.report:
        value = write_phase7_report(args.output)
    elif args.open_holdout:
        value = run_holdout(args.output)
    else:
        parser.error(
            "choose a Phase-7 dataset, DEV, diagnostic, planner, or holdout command"
        )
    print(json.dumps(value, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
