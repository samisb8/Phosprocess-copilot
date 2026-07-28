"""Validate lexical-safeguard robustness and sensitivity on DEV only."""

from __future__ import annotations

import ast
import inspect
import re
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_hybrid_reranked_dev_v3 import (  # noqa: E402
    BM25_CANDIDATES,
    CANDIDATE_K,
    DENSE_CANDIDATES,
    DENSE_INDEX_DIRECTORY,
    EMBEDDING_CONFIG_PATH,
    RERANKING_CONFIG_PATH,
    RETRIEVAL_CONFIG_PATH,
    TOP_K,
    atomic_write_csv,
    atomic_write_json,
    build_metric_fields,
    first_rank,
    load_dev_cases,
    resolve_project_path,
    sha256_file,
    validate_reproduced_baseline,
    verify_v2_starting_point,
)

from phosprocess.reranking.reranker import (  # noqa: E402
    BGEReranker,
    load_reranking_config,
)
from phosprocess.retrieval.bm25 import load_bm25_config  # noqa: E402
from phosprocess.retrieval.hybrid import HybridRetriever  # noqa: E402
from phosprocess.retrieval.v3_selection import (  # noqa: E402
    select_with_lexical_safeguard,
)

ROBUSTNESS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval"
    / "v0.1"
    / "v3"
    / "robustness"
)
SUMMARY_PATH = ROBUSTNESS_DIRECTORY / "robustness_summary.json"
PER_QUERY_PATH = ROBUSTNESS_DIRECTORY / "robustness_per_query.csv"
SENSITIVITY_PATH = ROBUSTNESS_DIRECTORY / "parameter_sensitivity.csv"
REPORT_PATH = ROBUSTNESS_DIRECTORY / "robustness_report.md"
SELECTION_SOURCE_PATH = (
    PROJECT_ROOT
    / "src"
    / "phosprocess"
    / "retrieval"
    / "v3_selection.py"
)
REPETITIONS = 3


@dataclass(frozen=True, slots=True)
class Variant:
    """One value of the sole lexical-protection parameter."""

    variant_id: str
    label: str
    lexical_slots: int
    complexity: int


VARIANTS = (
    Variant(
        variant_id="strict_lexical_slots_0",
        label="plus_stricte",
        lexical_slots=0,
        complexity=0,
    ),
    Variant(
        variant_id="lexical_safeguard_001",
        label="actuelle",
        lexical_slots=1,
        complexity=1,
    ),
    Variant(
        variant_id="permissive_lexical_slots_2",
        label="plus_permissive",
        lexical_slots=2,
        complexity=2,
    ),
)


def audit_selection_policy() -> dict[str, Any]:
    """Statically prove that the safeguard contains no gold-specific rule."""

    function_source = inspect.getsource(
        select_with_lexical_safeguard
    )
    syntax_tree = ast.parse(function_source)
    hardcoded_query_ids = sorted(
        set(re.findall(r"\bQ\d{3}\b", function_source))
    )
    hardcoded_chunk_ids = sorted(
        set(
            re.findall(
                r"\b[0-9]{2}_[a-z0-9_]+_[0-9]{6}_[0-9a-f]{12}\b",
                function_source,
            )
        )
    )
    forbidden_terms = (
        "gold",
        "reference_answer",
        "expected_answer",
        "query_id",
    )
    present_forbidden_terms = [
        term
        for term in forbidden_terms
        if term in function_source.casefold()
    ]
    parameter_names = list(
        inspect.signature(
            select_with_lexical_safeguard
        ).parameters
    )
    attribute_names = sorted(
        {
            node.attr
            for node in ast.walk(syntax_tree)
            if isinstance(node, ast.Attribute)
        }
    )
    allowed_signal_attributes = {
        "append",
        "bm25_rank",
        "chunk",
        "chunk_id",
        "join",
        "rank",
    }
    unexpected_signal_attributes = sorted(
        set(attribute_names) - allowed_signal_attributes
    )
    checks = {
        "generic_policy": (
            not hardcoded_query_ids
            and not hardcoded_chunk_ids
        ),
        "no_query_id_literal": not hardcoded_query_ids,
        "no_chunk_id_literal": not hardcoded_chunk_ids,
        "no_gold_or_reference_answer_term": (
            not present_forbidden_terms
        ),
        "inference_signature_is_generic": (
            parameter_names
            == [
                "candidates",
                "reranked_results",
                "top_k",
                "lexical_slots",
            ]
        ),
        "only_retrieval_rank_signals_used": (
            not unexpected_signal_attributes
        ),
    }

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "function_parameters": parameter_names,
        "retrieval_signal_attributes": attribute_names,
        "hardcoded_query_ids": hardcoded_query_ids,
        "hardcoded_chunk_ids": hardcoded_chunk_ids,
        "forbidden_terms_found": present_forbidden_terms,
        "unexpected_signal_attributes": unexpected_signal_attributes,
        "inference_boundary": {
            "question_usage": (
                "La question est utilisée uniquement par le retriever et "
                "le reranker en amont."
            ),
            "safeguard_inputs": (
                "Candidats hybrides, rangs BM25/hybrides et ordre du reranker."
            ),
            "evaluation_only": (
                "Les gold DEV sont consultés après sélection uniquement "
                "pour calculer les métriques."
            ),
        },
    }


def compute_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    """Aggregate metrics for one variant and one repetition."""

    total = len(rows)

    return {
        "candidate_recall_at_20": (
            sum(row["candidate_hit"] for row in rows) / total
        ),
        "evidence_recall_at_5": statistics.fmean(
            row["evidence_recall_at_5"]
            for row in rows
        ),
        "hit_at_5": (
            sum(row["hit_at_5"] for row in rows) / total
        ),
        "mrr_at_5": statistics.fmean(
            row["reciprocal_rank_at_5"]
            for row in rows
        ),
        "hit_at_1": (
            sum(row["hit_at_1"] for row in rows) / total
        ),
    }


def metric_signature(metrics: dict[str, float]) -> tuple[float, ...]:
    """Create an exact comparison signature in selection-rule order."""

    return (
        metrics["candidate_recall_at_20"],
        metrics["evidence_recall_at_5"],
        metrics["hit_at_5"],
        metrics["mrr_at_5"],
        metrics["hit_at_1"],
    )


def choose_variant(
    sensitivity_rows: list[dict[str, Any]],
    *,
    baseline_candidate_recall: float,
) -> dict[str, Any]:
    """Apply the predefined lexicographic DEV selection rule."""

    eligible = [
        row
        for row in sensitivity_rows
        if (
            row["candidate_recall_at_20"]
            >= baseline_candidate_recall - 1e-12
        )
    ]

    if not eligible:
        raise ValueError(
            "Aucune variante ne conserve Candidate Recall@20."
        )

    ranked = sorted(
        eligible,
        key=lambda row: (
            -row["evidence_recall_at_5"],
            -row["hit_at_5"],
            -row["mrr_at_5"],
            -row["hit_at_1"],
            row["complexity"],
            row["median_policy_latency_us"],
            row["variant_id"],
        ),
    )

    for position, row in enumerate(ranked, start=1):
        row["selection_rank"] = position
        row["selected"] = position == 1

    return ranked[0]


def build_per_query_rows(
    raw_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse three repetitions into one row per query and variant."""

    grouped: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in raw_rows:
        grouped[(row["variant_id"], row["query_id"])].append(row)

    output_rows: list[dict[str, Any]] = []

    for variant in VARIANTS:
        variant_rows = [
            (key, rows)
            for key, rows in grouped.items()
            if key[0] == variant.variant_id
        ]

        for (_, query_id), rows in sorted(variant_rows):
            rows = sorted(rows, key=lambda row: row["repetition"])
            selected_lists = [
                row["selected_chunk_ids"]
                for row in rows
            ]
            baseline_lists = [
                row["baseline_chunk_ids"]
                for row in rows
            ]
            first = rows[0]
            baseline_rr = first["baseline_reciprocal_rank_at_5"]
            variant_rr = first["reciprocal_rank_at_5"]

            if variant_rr > baseline_rr:
                outcome = "improved"
            elif variant_rr < baseline_rr:
                outcome = "regressed"
            elif (
                first["selected_chunk_ids"]
                != first["baseline_chunk_ids"]
            ):
                outcome = "selection_changed_metric_equal"
            else:
                outcome = "unchanged"

            output_rows.append(
                {
                    "variant_id": variant.variant_id,
                    "label": variant.label,
                    "lexical_slots": variant.lexical_slots,
                    "query_id": query_id,
                    "category": first["category"],
                    "candidate_gold_rank": first["candidate_gold_rank"],
                    "baseline_gold_rank": first["baseline_gold_rank"],
                    "variant_gold_rank": first["gold_rank"],
                    "outcome_vs_baseline": outcome,
                    "selection_changed": int(
                        first["selected_chunk_ids"]
                        != first["baseline_chunk_ids"]
                    ),
                    "same_selection_across_runs": int(
                        len(set(selected_lists)) == 1
                    ),
                    "same_baseline_top5_across_runs": int(
                        len(set(baseline_lists)) == 1
                    ),
                    "baseline_chunk_ids": first["baseline_chunk_ids"],
                    "run_1_selected_chunk_ids": selected_lists[0],
                    "run_2_selected_chunk_ids": selected_lists[1],
                    "run_3_selected_chunk_ids": selected_lists[2],
                    "selection_sources": first["selection_sources"],
                    "evidence_recall_at_5": first[
                        "evidence_recall_at_5"
                    ],
                    "hit_at_5": first["hit_at_5"],
                    "reciprocal_rank_at_5": first[
                        "reciprocal_rank_at_5"
                    ],
                    "hit_at_1": first["hit_at_1"],
                }
            )

    return output_rows


def build_report(
    *,
    audit: dict[str, Any],
    determinism: dict[str, Any],
    sensitivity_rows: list[dict[str, Any]],
    per_query_rows: list[dict[str, Any]],
    recommendation: str,
) -> str:
    """Render the concise DEV-only robustness report."""

    lines = [
        "# Robustesse DEV du candidat v3 lexical_safeguard_001",
        "",
        "## Périmètre",
        "",
        "- Split utilisé : DEV uniquement.",
        "- Répétitions complètes : 3.",
        "- Artefact TEST lu ou exécuté : non.",
        "- Gold DEV utilisé dans l'inférence : non ; métriques uniquement.",
        "- Gel automatique de v3 : non.",
        "",
        "## Audit de la politique",
        "",
        f"- Audit statique réussi : {str(audit['passed']).lower()}.",
        "- Aucun query_id, chunk précis, gold ou texte de réponse codé en dur.",
        "- Entrées du safeguard : candidats, rangs BM25/hybrides et ordre du reranker.",
        "",
        "## Déterminisme",
        "",
        (
            "- Top-5 courant identique sur les 3 runs : "
            f"{str(determinism['current_same_top5']).lower()}."
        ),
        (
            "- Métriques courantes identiques : "
            f"{str(determinism['current_same_metrics']).lower()}."
        ),
        (
            "- Toutes les sélections de toutes les variantes sont stables : "
            f"{str(determinism['all_variant_selections_stable']).lower()}."
        ),
        "",
        "## Sensibilité du paramètre lexical_slots",
        "",
        (
            "| Variante | slots | Candidate Recall@20 | Evidence Recall@5 | "
            "Hit@5 | MRR@5 | Hit@1 | Rang |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in sorted(
        sensitivity_rows,
        key=lambda item: item["lexical_slots"],
    ):
        lines.append(
            f"| {row['variant_id']} | {row['lexical_slots']} | "
            f"{row['candidate_recall_at_20']:.4f} | "
            f"{row['evidence_recall_at_5']:.4f} | "
            f"{row['hit_at_5']:.4f} | "
            f"{row['mrr_at_5']:.4f} | "
            f"{row['hit_at_1']:.4f} | "
            f"{row['selection_rank']} |"
        )

    differences = [
        row
        for row in per_query_rows
        if row["outcome_vs_baseline"] != "unchanged"
    ]
    lines.extend(
        [
            "",
            "## Différences par question",
            "",
            "| Variante | Question | Rang v2 | Rang variante | Résultat |",
            "|---|---|---:|---:|---|",
        ]
    )

    for row in differences:
        baseline_rank = row["baseline_gold_rank"] or "MISS"
        variant_rank = row["variant_gold_rank"] or "MISS"
        lines.append(
            f"| {row['variant_id']} | {row['query_id']} | "
            f"{baseline_rank} | {variant_rank} | "
            f"{row['outcome_vs_baseline']} |"
        )

    lines.extend(
        [
            "",
            "## Recommandation",
            "",
            f"**{recommendation}**",
            "",
            (
                "Cette recommandation est fondée uniquement sur DEV. "
                "Aucun snapshot dev_best_v3 n'a été créé."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    """Run three full DEV repetitions and produce robustness artifacts."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    audit = audit_selection_policy()

    if not audit["passed"]:
        raise ValueError(
            "L'audit statique de la politique v3 a échoué."
        )

    source_hashes = verify_v2_starting_point()
    cases = load_dev_cases()
    bm25_config = load_bm25_config(RETRIEVAL_CONFIG_PATH)

    print("Audit statique: OK")
    print(f"DEV-only queries: {len(cases)}")
    print(f"Full repetitions: {REPETITIONS}")
    print("Loading hybrid retriever once...")

    retriever = HybridRetriever(
        dense_index_directory=DENSE_INDEX_DIRECTORY,
        bm25_index_directory=resolve_project_path(
            bm25_config.output_directory
        ),
        embedding_config_path=EMBEDDING_CONFIG_PATH,
        retrieval_config_path=RETRIEVAL_CONFIG_PATH,
    )
    print("Loading reranker once...")
    reranker = BGEReranker(
        load_reranking_config(RERANKING_CONFIG_PATH)
    )
    raw_rows: list[dict[str, Any]] = []
    policy_latencies: dict[str, list[float]] = defaultdict(list)

    for repetition in range(1, REPETITIONS + 1):
        print(f"\n=== DEV repetition {repetition}/{REPETITIONS} ===")

        for position, case in enumerate(cases, start=1):
            hybrid_response = retriever.search(
                case["question"],
                top_k=CANDIDATE_K,
                dense_candidate_k=DENSE_CANDIDATES,
                bm25_candidate_k=BM25_CANDIDATES,
                use_query_expansion=True,
            )
            reranked_response = reranker.rerank(
                case["question"],
                hybrid_response.results,
                top_k=CANDIDATE_K,
            )
            candidate_ids = [
                result.chunk.chunk_id
                for result in hybrid_response.results
            ]
            baseline_ids = [
                result.chunk.chunk_id
                for result in reranked_response.results[:TOP_K]
            ]
            gold_ids = set(case["gold_chunk_ids"])
            candidate_gold_rank = first_rank(
                candidate_ids,
                gold_ids,
            )
            baseline_fields = build_metric_fields(
                chunk_ids=baseline_ids,
                gold_ids=gold_ids,
            )

            for variant in VARIANTS:
                selection_started = time.perf_counter_ns()
                selected = select_with_lexical_safeguard(
                    hybrid_response.results,
                    reranked_response.results,
                    top_k=TOP_K,
                    lexical_slots=variant.lexical_slots,
                )
                selection_latency_us = (
                    time.perf_counter_ns() - selection_started
                ) / 1000.0
                policy_latencies[variant.variant_id].append(
                    selection_latency_us
                )
                selected_ids = [
                    result.chunk_id
                    for result in selected
                ]

                if len(selected_ids) != TOP_K:
                    raise ValueError(
                        f"{variant.variant_id}: top-5 incomplet."
                    )

                if len(selected_ids) != len(set(selected_ids)):
                    raise ValueError(
                        f"{variant.variant_id}: chunk dupliqué."
                    )

                if (
                    variant.lexical_slots == 0
                    and selected_ids != baseline_ids
                ):
                    raise ValueError(
                        "La variante stricte ne reproduit pas le top-5 v2."
                    )

                metric_fields = build_metric_fields(
                    chunk_ids=selected_ids,
                    gold_ids=gold_ids,
                )
                raw_rows.append(
                    {
                        "repetition": repetition,
                        "variant_id": variant.variant_id,
                        "label": variant.label,
                        "lexical_slots": variant.lexical_slots,
                        "query_id": case["query_id"],
                        "category": case["category"],
                        "candidate_chunk_ids": "|".join(candidate_ids),
                        "candidate_gold_rank": (
                            candidate_gold_rank
                            if candidate_gold_rank is not None
                            else ""
                        ),
                        "candidate_hit": int(
                            candidate_gold_rank is not None
                        ),
                        "baseline_chunk_ids": "|".join(baseline_ids),
                        "baseline_gold_rank": (
                            baseline_fields["gold_rank"]
                            if baseline_fields["gold_rank"] is not None
                            else ""
                        ),
                        "baseline_reciprocal_rank_at_5": (
                            baseline_fields[
                                "reciprocal_rank_at_5"
                            ]
                        ),
                        "selected_chunk_ids": "|".join(selected_ids),
                        "selection_sources": "|".join(
                            result.source
                            for result in selected
                        ),
                        "gold_rank": (
                            metric_fields["gold_rank"]
                            if metric_fields["gold_rank"] is not None
                            else ""
                        ),
                        "evidence_recall_at_5": metric_fields[
                            "evidence_recall_at_5"
                        ],
                        "hit_at_5": metric_fields["hit_at_5"],
                        "reciprocal_rank_at_5": metric_fields[
                            "reciprocal_rank_at_5"
                        ],
                        "hit_at_1": metric_fields["hit_at_1"],
                        "policy_latency_us": selection_latency_us,
                    }
                )

            print(
                f"[{position:02d}/{len(cases):02d}] "
                f"{case['query_id']}"
            )

    rows_by_variant_run: dict[
        tuple[str, int],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in raw_rows:
        rows_by_variant_run[
            (row["variant_id"], row["repetition"])
        ].append(row)

    metrics_by_variant_run = {
        key: compute_metrics(rows)
        for key, rows in rows_by_variant_run.items()
    }

    for repetition in range(1, REPETITIONS + 1):
        strict_metrics = metrics_by_variant_run[
            ("strict_lexical_slots_0", repetition)
        ]
        validate_reproduced_baseline(
            {
                "evidence_recall_at_5": strict_metrics[
                    "evidence_recall_at_5"
                ],
                "hit_at_5": strict_metrics["hit_at_5"],
                "mrr_at_5": strict_metrics["mrr_at_5"],
                "hit_at_1": strict_metrics["hit_at_1"],
            }
        )

    per_query_rows = build_per_query_rows(raw_rows)
    current_rows = [
        row
        for row in per_query_rows
        if row["variant_id"] == "lexical_safeguard_001"
    ]
    current_metric_signatures = {
        metric_signature(
            metrics_by_variant_run[
                ("lexical_safeguard_001", repetition)
            ]
        )
        for repetition in range(1, REPETITIONS + 1)
    }
    determinism = {
        "repetitions": REPETITIONS,
        "current_same_top5": all(
            row["same_selection_across_runs"] == 1
            for row in current_rows
        ),
        "current_same_metrics": (
            len(current_metric_signatures) == 1
        ),
        "baseline_same_top5": all(
            row["same_baseline_top5_across_runs"] == 1
            for row in current_rows
        ),
        "all_variant_selections_stable": all(
            row["same_selection_across_runs"] == 1
            for row in per_query_rows
        ),
        "stable_current_queries": sum(
            row["same_selection_across_runs"]
            for row in current_rows
        ),
        "total_current_queries": len(current_rows),
    }
    baseline_metrics = metrics_by_variant_run[
        ("strict_lexical_slots_0", 1)
    ]
    sensitivity_rows: list[dict[str, Any]] = []

    for variant in VARIANTS:
        run_metrics = [
            metrics_by_variant_run[
                (variant.variant_id, repetition)
            ]
            for repetition in range(1, REPETITIONS + 1)
        ]
        metrics = {
            metric_name: statistics.fmean(
                run[metric_name]
                for run in run_metrics
            )
            for metric_name in run_metrics[0]
        }
        deterministic = (
            len(
                {
                    metric_signature(run)
                    for run in run_metrics
                }
            )
            == 1
        )
        sensitivity_rows.append(
            {
                "variant_id": variant.variant_id,
                "label": variant.label,
                "lexical_slots": variant.lexical_slots,
                "complexity": variant.complexity,
                **metrics,
                "delta_candidate_recall_at_20": (
                    metrics["candidate_recall_at_20"]
                    - baseline_metrics["candidate_recall_at_20"]
                ),
                "delta_evidence_recall_at_5": (
                    metrics["evidence_recall_at_5"]
                    - baseline_metrics["evidence_recall_at_5"]
                ),
                "delta_hit_at_5": (
                    metrics["hit_at_5"]
                    - baseline_metrics["hit_at_5"]
                ),
                "delta_mrr_at_5": (
                    metrics["mrr_at_5"]
                    - baseline_metrics["mrr_at_5"]
                ),
                "delta_hit_at_1": (
                    metrics["hit_at_1"]
                    - baseline_metrics["hit_at_1"]
                ),
                "median_policy_latency_us": statistics.median(
                    policy_latencies[variant.variant_id]
                ),
                "metrics_identical_across_runs": deterministic,
                "eligible_candidate_recall": (
                    metrics["candidate_recall_at_20"]
                    >= baseline_metrics["candidate_recall_at_20"]
                    - 1e-12
                ),
                "selection_rank": "",
                "selected": False,
            }
        )

    selected_variant = choose_variant(
        sensitivity_rows,
        baseline_candidate_recall=baseline_metrics[
            "candidate_recall_at_20"
        ],
    )
    robustness_passed = (
        audit["passed"]
        and determinism["current_same_top5"]
        and determinism["current_same_metrics"]
        and determinism["all_variant_selections_stable"]
    )

    if (
        selected_variant["variant_id"]
        == "lexical_safeguard_001"
        and robustness_passed
    ):
        recommendation = "freeze lexical_safeguard_001"
    elif robustness_passed:
        recommendation = (
            "conserver une autre variante : "
            + selected_variant["variant_id"]
        )
    else:
        recommendation = "poursuivre les expériences DEV"

    summary = {
        "validation_id": "v3_lexical_safeguard_robustness_001",
        "status": "completed_dev_only",
        "split": "dev",
        "test_artifacts_read": False,
        "test_evaluation_run": False,
        "reference_answers_used_for_inference": False,
        "gold_used_for_inference": False,
        "gold_usage": "post_selection_metrics_only",
        "v3_frozen": False,
        "audit": audit,
        "repetitions": REPETITIONS,
        "determinism": determinism,
        "selection_rule": [
            "no_candidate_recall_at_20_regression",
            "maximize_evidence_recall_at_5",
            "maximize_hit_at_5",
            "maximize_mrr_at_5",
            "maximize_hit_at_1",
            "minimize_complexity",
            "minimize_policy_latency",
        ],
        "baseline_v2_metrics": baseline_metrics,
        "variants": sensitivity_rows,
        "selected_variant": selected_variant["variant_id"],
        "robustness_passed": robustness_passed,
        "recommendation": recommendation,
        "per_question_differences": {
            variant.variant_id: [
                {
                    "query_id": row["query_id"],
                    "outcome": row["outcome_vs_baseline"],
                    "baseline_gold_rank": row["baseline_gold_rank"],
                    "variant_gold_rank": row["variant_gold_rank"],
                }
                for row in per_query_rows
                if (
                    row["variant_id"] == variant.variant_id
                    and row["outcome_vs_baseline"] != "unchanged"
                )
            ]
            for variant in VARIANTS
        },
        "source_sha256": {
            **source_hashes,
            "v3_selection.py": sha256_file(
                SELECTION_SOURCE_PATH
            ),
            "validate_retrieval_dev_v3_robustness.py": sha256_file(
                Path(__file__)
            ),
        },
        "outputs": {
            "summary": str(
                SUMMARY_PATH.relative_to(PROJECT_ROOT)
            ).replace("\\", "/"),
            "per_query": str(
                PER_QUERY_PATH.relative_to(PROJECT_ROOT)
            ).replace("\\", "/"),
            "parameter_sensitivity": str(
                SENSITIVITY_PATH.relative_to(PROJECT_ROOT)
            ).replace("\\", "/"),
            "report": str(
                REPORT_PATH.relative_to(PROJECT_ROOT)
            ).replace("\\", "/"),
        },
    }
    report = build_report(
        audit=audit,
        determinism=determinism,
        sensitivity_rows=sensitivity_rows,
        per_query_rows=per_query_rows,
        recommendation=recommendation,
    )

    atomic_write_json(SUMMARY_PATH, summary)
    atomic_write_csv(PER_QUERY_PATH, per_query_rows)
    atomic_write_csv(SENSITIVITY_PATH, sensitivity_rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("\n=== Robustness result ===")
    print(f"Audit passed: {audit['passed']}")
    print(
        "Current top-5 stable: "
        f"{determinism['stable_current_queries']}/"
        f"{determinism['total_current_queries']}"
    )

    for row in sorted(
        sensitivity_rows,
        key=lambda item: item["lexical_slots"],
    ):
        print(
            f"{row['variant_id']} | slots={row['lexical_slots']} | "
            f"EvidenceRecall@5={row['evidence_recall_at_5']:.4f} | "
            f"Hit@5={row['hit_at_5']:.4f} | "
            f"MRR@5={row['mrr_at_5']:.4f} | "
            f"Hit@1={row['hit_at_1']:.4f}"
        )

    print(f"Recommendation: {recommendation}")
    print("v3 frozen: False")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Per query: {PER_QUERY_PATH}")
    print(f"Sensitivity: {SENSITIVITY_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
