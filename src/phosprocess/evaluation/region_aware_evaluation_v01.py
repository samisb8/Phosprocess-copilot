# ruff: noqa: E501
"""Phase-9 DEV-first evaluation of label-blind region-aware candidate access."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from phosprocess.evaluation.candidate_preservation_v01 import (
    ACTIVE_DIRECTORY,
    _score_candidate_union,
)
from phosprocess.evaluation.candidate_preservation_v01 import (
    DEFAULT_OUTPUT as PHASE7_OUTPUT,
)
from phosprocess.evaluation.context_engine_v01 import read_jsonl, write_jsonl
from phosprocess.evaluation.evidence_ground_truth_audit_v01 import (
    DEFAULT_OUTPUT as PHASE8_OUTPUT,
)
from phosprocess.evaluation.evidence_ground_truth_audit_v01 import (
    _access_metrics,
    _evidence_best_rank,
    _evidence_ids,
    _ranking_metrics,
    build_manual_annotations,
)
from phosprocess.evaluation.region_candidate_expansion_v01 import (
    RegionCandidateExpander,
    RegionComposition,
    RegionVariant,
)
from phosprocess.ingestion.chunk_serialization import (
    read_child_chunks,
    read_parent_chunks,
)
from phosprocess.rag.orchestrator import PhosProcessRAG
from phosprocess.retrieval.hybrid import HybridSearchResult

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "data/evaluation/region_aware_candidate_access/v0.1"
ANCHOR_VALUES = (5, 10, 20)
CANDIDATE_BUDGETS = (30, 40, 50, 60)
BASELINES = {
    "current": "current_candidate_ids",
    "phase7_candidate": "candidate_ids",
}


def _phase7_holdout_summary() -> dict[str, Any]:
    return json.loads(
        (PHASE7_OUTPUT / "holdout_summary.json").read_text(encoding="utf-8")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repair_display_text(value: str | None) -> str | None:
    """Repair legacy UTF-8/CP1252 mojibake in human-readable audit artifacts."""

    if value is None:
        return None
    repaired = value
    for _attempt in range(2):
        if not any(marker in repaired for marker in ("Ã", "â", "Â")):
            break
        try:
            repaired = repaired.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
    return repaired


def _configuration_name(
    baseline: str,
    variant: RegionVariant,
    anchor_k: int,
    budget: int,
) -> str:
    return f"{baseline}__{variant.value}__a{anchor_k}__b{budget}"


def _configuration_grid() -> list[dict[str, Any]]:
    return [
        {
            "name": _configuration_name(baseline, variant, anchor_k, budget),
            "baseline": baseline,
            "variant": variant,
            "anchor_k": anchor_k,
            "candidate_budget": budget,
        }
        for baseline in BASELINES
        for variant in RegionVariant
        for anchor_k in ANCHOR_VALUES
        for budget in CANDIDATE_BUDGETS
    ]


def _dummy_candidates(
    ids: list[str],
    chunks: dict[str, Any],
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


def _score_ids(
    engine: Any,
    record: dict[str, Any],
    ids: list[str],
    chunks: dict[str, Any],
) -> tuple[dict[str, float], float]:
    if not ids:
        return {}, 0.0
    compatible = {
        **record,
        "classified_question_type": record["question_type"],
    }
    return _score_candidate_union(
        engine,
        compatible,
        record["question"],
        _dummy_candidates(ids, chunks),
    )


def _rank_by_score(ids: list[str], scores: dict[str, float]) -> list[str]:
    original = {chunk_id: rank for rank, chunk_id in enumerate(ids, 1)}
    missing = [chunk_id for chunk_id in ids if chunk_id not in scores]
    if missing:
        raise RuntimeError(f"missing reranker scores for {missing[:3]}")
    return sorted(ids, key=lambda item: (-scores[item], original[item], item))


def _metric_payload(
    records: list[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    candidate_ids: dict[str, list[str]],
    reranked_ids: dict[str, list[str]],
) -> dict[str, float]:
    synthetic = [
        {
            **record,
            "phase9_candidate_ids": candidate_ids[record["id"]],
            "phase9_reranked_ids": reranked_ids[record["id"]],
        }
        for record in records
    ]
    return {
        **_ranking_metrics(synthetic, annotations, "phase9_reranked_ids"),
        **_access_metrics(synthetic, annotations, "phase9_candidate_ids"),
    }


def _load_dev_scores() -> dict[str, dict[str, float]]:
    return {
        row["id"]: {key: float(value) for key, value in row["candidate_scores"].items()}
        for row in read_jsonl(PHASE7_OUTPUT / "dev_results.jsonl")
    }


def _compose_grid(
    records: list[dict[str, Any]],
    expander: RegionCandidateExpander,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, RegionComposition]],
    dict[str, set[str]],
]:
    grid = _configuration_grid()
    compositions: dict[str, dict[str, RegionComposition]] = {
        config["name"]: {} for config in grid
    }
    required_ids = {record["id"]: set() for record in records}
    for config in grid:
        for record in records:
            composition = expander.compose(
                list(record[BASELINES[config["baseline"]]]),
                locked_document=record["locked_document"],
                variant=config["variant"],
                anchor_k=config["anchor_k"],
                candidate_budget=config["candidate_budget"],
            )
            compositions[config["name"]][record["id"]] = composition
            required_ids[record["id"]].update(composition.candidate_ids)
    return grid, compositions, required_ids


def _ensure_dev_scores(
    records: list[dict[str, Any]],
    scores: dict[str, dict[str, float]],
    required_ids: dict[str, set[str]],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    missing = {
        record["id"]: sorted(required_ids[record["id"]] - set(scores[record["id"]]))
        for record in records
    }
    missing = {question_id: ids for question_id, ids in missing.items() if ids}
    if not missing:
        return scores, {"questions_scored": 0, "passages_scored": 0, "latency_ms": 0.0}
    rag = PhosProcessRAG()
    engine = rag.quality_engine
    if engine is None:
        raise RuntimeError("production quality engine is required")
    chunks = {
        chunk.chunk_id: chunk for chunk in engine.retriever.dense_retriever.metadata
    }
    by_id = {record["id"]: record for record in records}
    latency = 0.0
    for question_id, ids in missing.items():
        additions, elapsed = _score_ids(engine, by_id[question_id], ids, chunks)
        scores[question_id].update(additions)
        latency += elapsed
    return scores, {
        "questions_scored": len(missing),
        "passages_scored": sum(len(ids) for ids in missing.values()),
        "latency_ms": latency,
    }


def _evaluate_grid(
    records: list[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    grid: list[dict[str, Any]],
    compositions: dict[str, dict[str, RegionComposition]],
    scores: dict[str, dict[str, float]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[str]]]]:
    rows: list[dict[str, Any]] = []
    rankings: dict[str, dict[str, list[str]]] = {}
    for config in grid:
        name = config["name"]
        candidate_ids = {
            question_id: list(composition.candidate_ids)
            for question_id, composition in compositions[name].items()
        }
        reranked_ids = {
            question_id: _rank_by_score(ids, scores[question_id])
            for question_id, ids in candidate_ids.items()
        }
        rankings[name] = reranked_ids
        metrics = _metric_payload(records, annotations, candidate_ids, reranked_ids)
        config_compositions = list(compositions[name].values())
        rows.append(
            {
                **config,
                "variant": config["variant"].value,
                "metrics": metrics,
                "candidate_counts": {
                    "average_anchors_expanded": mean(
                        len(item.anchor_ids) for item in config_compositions
                    ),
                    "average_structural_candidates_added": mean(
                        len(item.structural_candidates) for item in config_compositions
                    ),
                    "average_unique_reranker_candidates": mean(
                        len(item.candidate_ids) for item in config_compositions
                    ),
                },
                "latency_ms": {
                    "average_region_lookup": mean(
                        item.lookup_latency_ms for item in config_compositions
                    ),
                    "average_candidate_composition": mean(
                        item.composition_latency_ms for item in config_compositions
                    ),
                },
            }
        )
    return rows, rankings


def _baseline_metrics() -> dict[str, Any]:
    phase8 = json.loads((PHASE8_OUTPUT / "metrics.json").read_text(encoding="utf-8"))
    return phase8


def _eligible_for_selection(row: dict[str, Any], baselines: dict[str, Any]) -> bool:
    baseline = baselines["dev"][row["baseline"]]
    metrics = row["metrics"]
    return (
        metrics["evidence_set_recall_at_5"]
        >= baseline["evidence_set_recall_at_5"] - 0.02
        and metrics["evidence_set_recall_at_20"]
        >= baseline["evidence_set_recall_at_20"]
        and metrics["question_evidence_coverage_at_20"]
        >= baseline["question_evidence_coverage_at_20"]
    )


def _select_policy(rows: list[dict[str, Any]], baselines: dict[str, Any]) -> dict[str, Any]:
    eligible = [row for row in rows if _eligible_for_selection(row, baselines)]
    if not eligible:
        raise RuntimeError("no region configuration passes DEV safety constraints")
    variant_cost = {
        RegionVariant.SAME_PARENT.value: 0,
        RegionVariant.PARENT_AND_NEIGHBORS.value: 1,
        RegionVariant.PARENT_NEIGHBORS_SECTION_2.value: 2,
    }
    return max(
        eligible,
        key=lambda row: (
            row["metrics"]["evidence_set_recall_at_20"],
            row["metrics"]["question_evidence_coverage_at_20"],
            row["metrics"]["evidence_set_access_recall"],
            row["metrics"]["evidence_set_recall_at_5"],
            row["metrics"]["evidence_set_mrr"],
            -row["candidate_counts"]["average_unique_reranker_candidates"],
            -variant_cost[row["variant"]],
            -row["anchor_k"],
            -row["candidate_budget"],
        ),
    )


def _trace_question(
    record: dict[str, Any],
    annotation: dict[str, Any],
    composition: RegionComposition,
    reranked_ids: list[str],
    expander: RegionCandidateExpander,
) -> dict[str, Any]:
    evidence_sets = annotation["valid_evidence_sets"]
    return {
        "question_id": record["id"],
        "question": record["question"],
        "top_anchors": [
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "parent_id": expander.child_by_id[chunk_id].parent_id,
                "previous_chunk_id": expander.child_by_id[chunk_id].previous_chunk_id,
                "next_chunk_id": expander.child_by_id[chunk_id].next_chunk_id,
                "section_id": expander.child_by_id[chunk_id].section_id,
            }
            for rank, chunk_id in enumerate(composition.anchor_ids, 1)
        ],
        "structural_candidates": [
            {
                "chunk_id": item.chunk_id,
                "provenance": item.provenance,
                "anchor_chunk_id": item.anchor_chunk_id,
            }
            for item in composition.structural_candidates
        ],
        "valid_evidence_ids": sorted(_evidence_ids(evidence_sets)),
        "evidence_accessible": _evidence_best_rank(
            evidence_sets, composition.candidate_ids
        )
        is not None,
        "complete_evidence_rank_before_reranker": _evidence_best_rank(
            evidence_sets, composition.candidate_ids
        ),
        "complete_evidence_rank_after_reranker": _evidence_best_rank(
            evidence_sets, reranked_ids
        ),
    }


def run_dev(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Evaluate the full structural grid on DEV and freeze exactly one policy."""

    if (output_directory / "frozen_policy.json").exists():
        raise RuntimeError("Phase-9 DEV policy is already frozen; refusing to retune")
    output_directory.mkdir(parents=True, exist_ok=True)
    records, annotations = build_manual_annotations()
    records = [record for record in records if record["split"] == "dev"]
    children = read_child_chunks(ACTIVE_DIRECTORY / "chunks.jsonl")
    parents = read_parent_chunks(ACTIVE_DIRECTORY / "parents.jsonl")
    expander = RegionCandidateExpander(children=children, parents=parents)
    grid, compositions, required_ids = _compose_grid(records, expander)
    scores, backfill = _ensure_dev_scores(records, _load_dev_scores(), required_ids)
    rows, rankings = _evaluate_grid(
        records, annotations, grid, compositions, scores
    )
    baselines = _baseline_metrics()
    selected = _select_policy(rows, baselines)
    selected_name = selected["name"]
    selected_candidate_ids = {
        question_id: list(item.candidate_ids)
        for question_id, item in compositions[selected_name].items()
    }
    selected_rankings = rankings[selected_name]
    selected_metrics = _metric_payload(
        records, annotations, selected_candidate_ids, selected_rankings
    )
    manifest = json.loads(
        (PHASE7_OUTPUT / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    frozen = {
        "phase": 9,
        "selected_on": "dev_only",
        "frozen_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": manifest["questions_sha256"],
        "evidence_sets_sha256": _sha256(PHASE8_OUTPUT / "evidence_sets.json"),
        "baseline": selected["baseline"],
        "variant": selected["variant"],
        "anchor_policy": f"top_{selected['anchor_k']}_baseline_anchors",
        "anchor_k": selected["anchor_k"],
        "candidate_budget": selected["candidate_budget"],
        "selection_metrics": selected_metrics,
        "selection_rule": "max DEV evidence R@20, coverage@20, access, R@5, evidence MRR; then smaller pool/region/anchor/budget",
        "holdout_metrics_inspected_during_selection": False,
        "production_changed": False,
    }
    frozen["policy_sha256"] = hashlib.sha256(
        json.dumps(frozen, sort_keys=True).encode("utf-8")
    ).hexdigest()
    per_question = []
    by_id = {record["id"]: record for record in records}
    for question_id, candidate_ids in selected_candidate_ids.items():
        per_question.append(
            {
                "id": question_id,
                "candidate_ids": candidate_ids,
                "reranked_ids": selected_rankings[question_id],
                "composition": {
                    "anchor_ids": list(
                        compositions[selected_name][question_id].anchor_ids
                    ),
                    "structural_candidates": [
                        {
                            "chunk_id": item.chunk_id,
                            "provenance": item.provenance,
                            "anchor_chunk_id": item.anchor_chunk_id,
                        }
                        for item in compositions[selected_name][
                            question_id
                        ].structural_candidates
                    ],
                },
                "source_lock_exact": all(
                    expander.child_by_id[chunk_id].document_id
                    == by_id[question_id]["locked_document"]
                    for chunk_id in candidate_ids
                ),
            }
        )
    write_jsonl(output_directory / "dev_selected_results.jsonl", per_question)
    (output_directory / "dev_ablation.json").write_text(
        json.dumps(
            {
                "question_count": len(records),
                "baseline_metrics": baselines["dev"],
                "configurations": rows,
                "selected_configuration": selected_name,
                "selected_metrics": selected_metrics,
                "score_backfill": backfill,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_directory / "frozen_policy.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    ce051 = next(record for record in records if record["id"] == "CE051")
    (output_directory / "ce051_trace.json").write_text(
        json.dumps(
            _trace_question(
                ce051,
                annotations["CE051"],
                compositions[selected_name]["CE051"],
                selected_rankings["CE051"],
                expander,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "dev_questions": len(records),
        "configurations": len(rows),
        "selected_configuration": selected_name,
        "selected_metrics": selected_metrics,
        "holdout_opened": False,
    }


def _load_frozen_policy(output_directory: Path) -> dict[str, Any]:
    policy_path = output_directory / "frozen_policy.json"
    if not policy_path.exists():
        raise RuntimeError("run and freeze DEV before opening holdout")
    return json.loads(policy_path.read_text(encoding="utf-8"))


def run_holdout(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Open FINAL HOLDOUT exactly once with the frozen DEV-selected policy."""

    result_path = output_directory / "holdout_result.json"
    if result_path.exists():
        raise RuntimeError("Phase-9 holdout was already opened; refusing a second run")
    frozen = _load_frozen_policy(output_directory)
    records, annotations = build_manual_annotations()
    records = [record for record in records if record["split"] == "final_holdout"]
    children = read_child_chunks(ACTIVE_DIRECTORY / "chunks.jsonl")
    parents = read_parent_chunks(ACTIVE_DIRECTORY / "parents.jsonl")
    expander = RegionCandidateExpander(children=children, parents=parents)
    variant = RegionVariant(frozen["variant"])
    baseline_key = BASELINES[frozen["baseline"]]
    compositions: dict[str, RegionComposition] = {}
    for record in records:
        compositions[record["id"]] = expander.compose(
            list(record[baseline_key]),
            locked_document=record["locked_document"],
            variant=variant,
            anchor_k=int(frozen["anchor_k"]),
            candidate_budget=int(frozen["candidate_budget"]),
        )

    rag = PhosProcessRAG()
    engine = rag.quality_engine
    if engine is None:
        raise RuntimeError("production quality engine is required")
    chunks = {
        chunk.chunk_id: chunk for chunk in engine.retriever.dense_retriever.metadata
    }
    scores: dict[str, dict[str, float]] = {}
    reranker_latency: dict[str, float] = {}
    rankings: dict[str, list[str]] = {}
    for record in records:
        question_id = record["id"]
        ids = list(compositions[question_id].candidate_ids)
        scores[question_id], reranker_latency[question_id] = _score_ids(
            engine, record, ids, chunks
        )
        rankings[question_id] = _rank_by_score(ids, scores[question_id])
    candidate_ids = {
        question_id: list(item.candidate_ids)
        for question_id, item in compositions.items()
    }
    metrics = _metric_payload(records, annotations, candidate_ids, rankings)
    source_lock_exact = all(
        expander.child_by_id[chunk_id].document_id == record["locked_document"]
        for record in records
        for chunk_id in candidate_ids[record["id"]]
    )
    dq027 = next(record for record in records if record["id"] == "DQ027")
    dq027_trace = _trace_question(
        dq027,
        annotations["DQ027"],
        compositions["DQ027"],
        rankings["DQ027"],
        expander,
    )
    baseline_metrics = _baseline_metrics()["final_holdout"]
    strongest_r20 = max(
        item["evidence_set_recall_at_20"] for item in baseline_metrics.values()
    )
    strongest_coverage = max(
        item["question_evidence_coverage_at_20"] for item in baseline_metrics.values()
    )
    strongest_r5 = max(
        item["evidence_set_recall_at_5"] for item in baseline_metrics.values()
    )
    r5_regression = (
        metrics["evidence_set_recall_at_5"] - strongest_r5
    )
    production_consideration = (
        metrics["evidence_set_recall_at_20"] > strongest_r20
        and metrics["question_evidence_coverage_at_20"] > strongest_coverage
        and r5_regression >= -0.02
        and source_lock_exact
    )
    historical_latency = _phase7_holdout_summary()["average_latency_ms"]
    baseline_first_stage = historical_latency[
        "current_colbert_first_stage"
        if frozen["baseline"] == "current"
        else "candidate_first_stage"
    ]
    average_region_lookup = mean(
        item.lookup_latency_ms for item in compositions.values()
    )
    average_candidate_composition = mean(
        item.composition_latency_ms for item in compositions.values()
    )
    average_reranker = mean(reranker_latency.values())
    payload = {
        "opened_at": datetime.now(UTC).isoformat(),
        "policy_sha256": frozen["policy_sha256"],
        "dataset_sha256": frozen["dataset_sha256"],
        "question_count": len(records),
        "frozen_policy": frozen,
        "baseline_metrics": baseline_metrics,
        "metrics": metrics,
        "candidate_counts": {
            "average_anchors_expanded": mean(
                len(item.anchor_ids) for item in compositions.values()
            ),
            "average_structural_candidates_added": mean(
                len(item.structural_candidates) for item in compositions.values()
            ),
            "average_unique_reranker_candidates": mean(
                len(item.candidate_ids) for item in compositions.values()
            ),
        },
        "latency_ms": {
            "average_baseline_first_stage": baseline_first_stage,
            "average_region_lookup": average_region_lookup,
            "average_candidate_composition": average_candidate_composition,
            "average_reranker": average_reranker,
            "average_total_retrieval": (
                baseline_first_stage
                + average_region_lookup
                + average_candidate_composition
                + average_reranker
            ),
        },
        "source_lock_exact": source_lock_exact,
        "production_consideration": production_consideration,
        "production_changed": False,
        "per_question": [
            {
                "id": record["id"],
                "candidate_ids": candidate_ids[record["id"]],
                "reranked_ids": rankings[record["id"]],
                "reranker_scores": scores[record["id"]],
            }
            for record in records
        ],
    }
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_directory / "dq027_trace.json").write_text(
        json.dumps(dq027_trace, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_directory / "holdout_open_manifest.json").write_text(
        json.dumps(
            {
                "opened_at": payload["opened_at"],
                "policy_sha256": frozen["policy_sha256"],
                "dataset_sha256": frozen["dataset_sha256"],
                "policy_frozen_before_open": True,
                "opening_count": 1,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "holdout_questions": len(records),
        "metrics": metrics,
        "dq027_evidence_accessible": dq027_trace["evidence_accessible"],
        "production_consideration": production_consideration,
    }


def _dq028_read_only_diagnostic(
    output_directory: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    source = json.loads(
        (PHASE8_OUTPUT / "dq028_reranker_score_backfill.json").read_text(
            encoding="utf-8"
        )
    )
    chunks = {
        chunk.chunk_id: chunk
        for chunk in read_child_chunks(ACTIVE_DIRECTORY / "chunks.jsonl")
    }

    def describe(chunk_id: str, *, rank: int, score: float | None) -> dict[str, Any]:
        chunk = chunks[chunk_id]
        return {
            "rank": rank,
            "chunk_id": chunk_id,
            "reranker_score": score,
            "document_id": chunk.document_id,
            "parent_id": chunk.parent_id,
            "section_metadata": {
                "chapter": _repair_display_text(chunk.chapter),
                "section": _repair_display_text(chunk.section),
                "subsection": _repair_display_text(chunk.subsection),
                "section_id": chunk.section_id,
                "hierarchy_path": _repair_display_text(chunk.hierarchy_path),
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
            },
            "documentary_wording": _repair_display_text(chunk.display_text),
        }

    score_by_id = {
        item["chunk_id"]: float(item["score"]) for item in source["top_10"]
    }
    phase9_path = output_directory / "holdout_result.json"
    phase9_record: dict[str, Any] | None = None
    if phase9_path.exists():
        phase9 = json.loads(phase9_path.read_text(encoding="utf-8"))
        phase9_record = next(
            item for item in phase9["per_question"] if item["id"] == "DQ028"
        )
    target_ids = list(source["historical_gold"])
    target_passages = []
    for chunk_id in target_ids:
        phase9_rank = None
        phase9_score = None
        if phase9_record is not None:
            if chunk_id in phase9_record["reranked_ids"]:
                phase9_rank = phase9_record["reranked_ids"].index(chunk_id) + 1
            phase9_score = phase9_record["reranker_scores"].get(chunk_id)
        target = describe(
            chunk_id,
            rank=int(source["historical_gold_rank"]),
            score=(
                float(phase9_score)
                if phase9_score is not None
                else score_by_id.get(chunk_id)
            ),
        )
        target["phase8_rank"] = int(source["historical_gold_rank"])
        target["phase9_rank"] = phase9_rank
        target["score_source"] = (
            "phase9_frozen_holdout" if phase9_score is not None else "phase8_backfill"
        )
        target_passages.append(target)
    return {
        "question_id": source["question_id"],
        "question": _repair_display_text(source["question"]),
        "classification": "reranker_failure_already_candidate_accessible",
        "candidate_passages": target_passages,
        "top_competing_passages": [
            describe(
                item["chunk_id"],
                rank=int(item["rank"]),
                score=float(item["score"]),
            )
            for item in source["top_10"]
        ],
        "candidate_ids_unchanged": source["candidate_ids_unchanged"],
        "reranker_model_unchanged": source["reranker_model_unchanged"],
        "phase9_changes": [],
    }


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_report(output_directory: Path = DEFAULT_OUTPUT) -> str:
    dev = json.loads((output_directory / "dev_ablation.json").read_text(encoding="utf-8"))
    frozen = _load_frozen_policy(output_directory)
    holdout = json.loads(
        (output_directory / "holdout_result.json").read_text(encoding="utf-8")
    )
    ce051 = json.loads((output_directory / "ce051_trace.json").read_text(encoding="utf-8"))
    dq027 = json.loads((output_directory / "dq027_trace.json").read_text(encoding="utf-8"))
    dq028_path = output_directory / "dq028_read_only_diagnostic.json"
    dq028 = (
        json.loads(dq028_path.read_text(encoding="utf-8"))
        if dq028_path.exists()
        else _dq028_read_only_diagnostic(output_directory)
    )
    gates_path = output_directory / "test_gates.json"
    gates = (
        json.loads(gates_path.read_text(encoding="utf-8"))
        if gates_path.exists()
        else {"status": "pending"}
    )
    selected = next(
        row
        for row in dev["configurations"]
        if row["name"] == dev["selected_configuration"]
    )
    baselines = dev["baseline_metrics"]
    holdout_baselines = holdout["baseline_metrics"]

    def metric_row(label: str, item: dict[str, float]) -> str:
        return (
            f"| {label} | {_pct(item['evidence_set_recall_at_5'])} | "
            f"{_pct(item['evidence_set_recall_at_10'])} | "
            f"{_pct(item['evidence_set_recall_at_20'])} | "
            f"{_pct(item['question_evidence_coverage_at_20'])} | "
            f"{item['evidence_set_mrr']:.3f} | "
            f"{_pct(item['region_recall_at_20'])} | "
            f"{_pct(item['exact_recall_at_20'])} |"
        )

    lines = [
        "# 1. REGION POLICY",
        "",
        f"Selected on DEV only: `{frozen['variant']}`, `{frozen['anchor_policy']}`, candidate budget `{frozen['candidate_budget']}`, baseline `{frozen['baseline']}`. Anchors are preserved first, then structural candidates, then baseline fill. Structural candidates receive provenance only and are reranked normally by unchanged BGE.",
        "",
        "# 2. DEV ABLATION",
        "",
        "| Architecture | Evidence R@20 | Coverage@20 | Evidence R@5 | Access |",
        "|---|---:|---:|---:|---:|",
        f"| Current | {_pct(baselines['current']['evidence_set_recall_at_20'])} | {_pct(baselines['current']['question_evidence_coverage_at_20'])} | {_pct(baselines['current']['evidence_set_recall_at_5'])} | {_pct(baselines['current']['evidence_set_access_recall'])} |",
        f"| Phase-7 | {_pct(baselines['phase7_candidate']['evidence_set_recall_at_20'])} | {_pct(baselines['phase7_candidate']['question_evidence_coverage_at_20'])} | {_pct(baselines['phase7_candidate']['evidence_set_recall_at_5'])} | {_pct(baselines['phase7_candidate']['evidence_set_access_recall'])} |",
    ]
    for baseline in BASELINES:
        best = max(
            (row for row in dev["configurations"] if row["baseline"] == baseline),
            key=lambda row: (
                row["metrics"]["evidence_set_recall_at_20"],
                row["metrics"]["question_evidence_coverage_at_20"],
                row["metrics"]["evidence_set_recall_at_5"],
            ),
        )
        lines.append(
            f"| {baseline} + region (`{best['variant']}`, a{best['anchor_k']}, b{best['candidate_budget']}) | {_pct(best['metrics']['evidence_set_recall_at_20'])} | {_pct(best['metrics']['question_evidence_coverage_at_20'])} | {_pct(best['metrics']['evidence_set_recall_at_5'])} | {_pct(best['metrics']['evidence_set_access_recall'])} |"
        )
    lines += [
        "",
        f"72 generic configurations were evaluated. Frozen winner: `{dev['selected_configuration']}`.",
        "",
        "# 3. EVIDENCE ACCESS",
        "",
        "| Split / architecture | Evidence access recall | Evidence access coverage |",
        "|---|---:|---:|",
        f"| DEV current | {_pct(baselines['current']['evidence_set_access_recall'])} | {_pct(baselines['current']['question_evidence_access_coverage'])} |",
        f"| DEV Phase-7 | {_pct(baselines['phase7_candidate']['evidence_set_access_recall'])} | {_pct(baselines['phase7_candidate']['question_evidence_access_coverage'])} |",
        f"| DEV selected region | {_pct(selected['metrics']['evidence_set_access_recall'])} | {_pct(selected['metrics']['question_evidence_access_coverage'])} |",
        f"| HOLDOUT current | {_pct(holdout_baselines['current']['evidence_set_access_recall'])} | {_pct(holdout_baselines['current']['question_evidence_access_coverage'])} |",
        f"| HOLDOUT Phase-7 | {_pct(holdout_baselines['phase7_candidate']['evidence_set_access_recall'])} | {_pct(holdout_baselines['phase7_candidate']['question_evidence_access_coverage'])} |",
        f"| HOLDOUT selected region | {_pct(holdout['metrics']['evidence_set_access_recall'])} | {_pct(holdout['metrics']['question_evidence_access_coverage'])} |",
        "",
        f"Source lock remained exact: `{holdout['source_lock_exact']}`.",
        "",
        "# 4. EVIDENCE-SET METRICS",
        "",
        "| Split / architecture | R@5 | R@10 | R@20 | Coverage@20 | MRR | Region R@20 | Exact R@20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        metric_row("DEV current", baselines["current"]),
        metric_row("DEV Phase-7", baselines["phase7_candidate"]),
        metric_row("DEV selected region", selected["metrics"]),
        metric_row("HOLDOUT current", holdout_baselines["current"]),
        metric_row("HOLDOUT Phase-7", holdout_baselines["phase7_candidate"]),
        metric_row("HOLDOUT selected region", holdout["metrics"]),
        "",
        "# 5. CE051 TRACE",
        "",
        f"Expanded anchors: `{[item['chunk_id'] for item in ce051['top_anchors']]}`. Anchor parent IDs: `{list(dict.fromkeys(item['parent_id'] for item in ce051['top_anchors']))}`. Structural candidates: `{[item['chunk_id'] for item in ce051['structural_candidates']]}`. Evidence accessible before reranker: `{ce051['evidence_accessible']}`; complete evidence rank before/after: `{ce051['complete_evidence_rank_before_reranker']}` / `{ce051['complete_evidence_rank_after_reranker']}`. Per-anchor previous/next links are recorded in `ce051_trace.json`; the frozen same-parent policy did not admit them. No content terms or expected pages were used.",
        "",
        "# 6. DQ027 TRACE",
        "",
        f"Expanded anchors: `{[item['chunk_id'] for item in dq027['top_anchors']]}`. Anchor parent IDs: `{list(dict.fromkeys(item['parent_id'] for item in dq027['top_anchors']))}`. Structural candidates: `{[item['chunk_id'] for item in dq027['structural_candidates']]}`. Evidence accessible before reranker: `{dq027['evidence_accessible']}`; complete evidence rank before/after: `{dq027['complete_evidence_rank_before_reranker']}` / `{dq027['complete_evidence_rank_after_reranker']}`. Per-anchor previous/next links are recorded in `dq027_trace.json`; the frozen same-parent policy did not admit them.",
        "",
        "# 7. DQ028 READ-ONLY DIAGNOSTIC",
        "",
        f"Question: `{dq028['question']}`. The direct candidate `{dq028['candidate_passages'][0]['chunk_id']}` scored `{dq028['candidate_passages'][0]['reranker_score']:.6f}` and remains a reranker failure at Phase-8 rank `{dq028['candidate_passages'][0]['phase8_rank']}` / Phase-9 rank `{dq028['candidate_passages'][0]['phase9_rank']}` in section `{dq028['candidate_passages'][0]['section_metadata']['hierarchy_path']}`. Top competitor `{dq028['top_competing_passages'][0]['chunk_id']}` scored `{dq028['top_competing_passages'][0]['reranker_score']:.6f}`. Passage wording and section metadata are preserved in `dq028_read_only_diagnostic.json`. No reranker model, prompt, input representation or score adjustment was changed.",
        "",
        "# 8. CANDIDATE COUNTS",
        "",
        f"DEV selected averages: anchors `{selected['candidate_counts']['average_anchors_expanded']:.2f}`, structural additions `{selected['candidate_counts']['average_structural_candidates_added']:.2f}`, unique reranker candidates `{selected['candidate_counts']['average_unique_reranker_candidates']:.2f}`. HOLDOUT: anchors `{holdout['candidate_counts']['average_anchors_expanded']:.2f}`, structural additions `{holdout['candidate_counts']['average_structural_candidates_added']:.2f}`, unique candidates `{holdout['candidate_counts']['average_unique_reranker_candidates']:.2f}`.",
        "",
        "# 9. LATENCY",
        "",
        f"HOLDOUT averages: baseline first-stage `{holdout['latency_ms']['average_baseline_first_stage']:.1f} ms`, region lookup `{holdout['latency_ms']['average_region_lookup']:.3f} ms`, composition `{holdout['latency_ms']['average_candidate_composition']:.3f} ms`, BGE reranker `{holdout['latency_ms']['average_reranker']:.1f} ms`, total retrieval `{holdout['latency_ms']['average_total_retrieval']:.1f} ms`. The first-stage figure is the frozen Phase-7 holdout measurement for the selected baseline; region and reranker figures are Phase-9 measurements.",
        "",
        "# 10. HOLDOUT RESULT",
        "",
        f"Policy was frozen at `{frozen['frozen_at']}` with dataset hash `{frozen['dataset_sha256']}` and opened once on `{holdout['opened_at']}`. Evidence R@20: {_pct(holdout['metrics']['evidence_set_recall_at_20'])}; Coverage@20: {_pct(holdout['metrics']['question_evidence_coverage_at_20'])}; R@5: {_pct(holdout['metrics']['evidence_set_recall_at_5'])}.",
        "",
        "# 11. PRODUCTION DECISION",
        "",
        ("**DEPLOY (recommended, not applied).** The frozen policy beats both holdout baselines on Evidence R@20 and Coverage@20 without material R@5 regression, while respecting the source lock and bounded pool." if holdout["production_consideration"] else "**DO NOT DEPLOY.** The frozen region policy did not demonstrate a convincing holdout improvement over the strongest baseline under the success criteria. Production retrieval remains unchanged."),
        "",
        "# 12. TEST GATES",
        "",
        "```json",
        json.dumps(gates, ensure_ascii=False, indent=2),
        "```",
        "",
        "# 13. PHASE 10 RECOMMENDATION",
        "",
        ("Prepare a narrowly scoped production-candidate review of this frozen structural composer; do not deploy automatically." if holdout["production_consideration"] else "Stop region-expansion optimization. Keep CE051/DQ027 as representation/query misses and move next to a read-only reranker/generation-quality investigation, without starting Evidence Judge automatically."),
        "",
        "Phase 9 stops here. No production patch, reindex, reranker change, Evidence Judge, answer repair or iterative RAG loop was implemented.",
    ]
    return "\n".join(lines) + "\n"


def write_report(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    (output_directory / "dq028_read_only_diagnostic.json").write_text(
        json.dumps(
            _dq028_read_only_diagnostic(output_directory),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = render_report(output_directory)
    (output_directory / "summary.md").write_text(report, encoding="utf-8")
    return {
        "sections": 13,
        "summary": str(output_directory / "summary.md"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-dev", action="store_true")
    parser.add_argument("--open-holdout", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.run_dev:
        value = run_dev(args.output)
    elif args.open_holdout:
        value = run_holdout(args.output)
    elif args.report:
        value = write_report(args.output)
    else:
        parser.error("choose --run-dev, --open-holdout or --report")
    print(json.dumps(value, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
