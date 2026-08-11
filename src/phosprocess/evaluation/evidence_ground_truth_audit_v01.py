# ruff: noqa: E501
"""Phase-8 manual evidence ground-truth audit and evidence-set evaluation.

This module is evaluation-only.  It never mutates production retrieval or the
active indexes, and it never uses an LLM to decide whether evidence is valid.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from phosprocess.evaluation.candidate_preservation_v01 import (
    ACTIVE_DIRECTORY,
    _evaluation_records,
)
from phosprocess.evaluation.candidate_preservation_v01 import (
    DEFAULT_OUTPUT as PHASE7_OUTPUT,
)
from phosprocess.ingestion.chunk_serialization import (
    TechnicalChildChunk,
    read_child_chunks,
    read_parent_chunks,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "data/evaluation/evidence_ground_truth_audit/v0.1"
FAILURE_IDS = frozenset(
    {
        "CE051",
        "CE056",
        "DQ009",
        "DQ011",
        "DQ013",
        "DQ015",
        "DQ027",
        "DQ037",
        "DQ038",
        "DQ028",
        "DQ035",
        "DQ039",
    }
)

CATEGORY_LABELS = {
    1: "one uniquely necessary passage",
    2: "multiple equivalent valid passages",
    3: "complementary passages required together",
    4: "historical gold relevant but insufficient",
    5: "questionable historical gold assignment",
}

# Every ID was reviewed against the active child text, page, section, historical
# gold and retrieved passages. IDs omitted from the four exceptional sets are
# category 1.  These labels describe the *historical* annotation, before the
# evidence-set correction below.
CATEGORY_BY_QUESTION = {
    **{
        question_id: 2
        for question_id in (
            "CE051",
            "CE055",
            "CE056",
            "DQ003",
            "DQ024",
            "DQ025",
            "DQ027",
        )
    },
    **{question_id: 3 for question_id in ("DQ002", "DQ004", "DQ045")},
    **{question_id: 4 for question_id in ("DQ013", "DQ035", "DQ039")},
    **{
        question_id: 5
        for question_id in (
            "DQ009",
            "DQ011",
            "DQ015",
            "DQ018",
            "DQ020",
            "DQ037",
            "DQ038",
        )
    },
}

CORRECTED_IDS = frozenset(
    {
        "DQ009",
        "DQ011",
        "DQ013",
        "DQ015",
        "DQ018",
        "DQ020",
        "DQ035",
        "DQ037",
        "DQ038",
    }
)
ADDED_EVIDENCE_IDS = frozenset(
    {"CE051", "CE055", "CE056", "DQ003", "DQ024", "DQ025", "DQ027", "DQ039"}
)

# Manually verified against the active corpus. An ``alternative`` set is
# satisfied by any listed chunk. A ``complementary`` set is satisfied only when
# at least one chunk from every group is present. Multiple entries are OR paths.
EVIDENCE_OVERRIDES: dict[str, list[dict[str, Any]]] = {
    "CE051": [
        {
            "type": "alternative",
            "chunk_ids": [
                "becker_phosphates_and_phosphoric_acid_bd5716159148dec0",
                "becker_phosphates_and_phosphoric_acid_2f2a160a376652cc",
                "becker_phosphates_and_phosphoric_acid_7ee6f8193c5857af",
            ],
        }
    ],
    "CE055": [
        {
            "type": "alternative",
            "chunk_ids": [
                "perrys_chemical_engineers_handbook_0a98c7afbb8e9ea8",
                "perrys_chemical_engineers_handbook_c2db5314ed1b1baf",
                "perrys_chemical_engineers_handbook_ed8229451d371966",
            ],
        }
    ],
    "CE056": [
        {
            "type": "alternative",
            "chunk_ids": [
                "mullin_crystallization_e23198f092452faf",
                "mullin_crystallization_47011a3420e91355",
            ],
        }
    ],
    "DQ003": [
        {
            "type": "alternative",
            "chunk_ids": [
                "perrys_chemical_engineers_handbook_1666e4fbdc480b49",
                "perrys_chemical_engineers_handbook_b7abdbd81bf56814",
            ],
        }
    ],
    "DQ009": [
        {
            "type": "alternative",
            "chunk_ids": [
                "becker_phosphates_and_phosphoric_acid_a9556fe0825de3c3",
                "becker_phosphates_and_phosphoric_acid_db86aaf551221b81",
            ],
        }
    ],
    "DQ011": [
        {
            "type": "alternative",
            "chunk_ids": [
                "smith_van_ness_chemical_engineering_thermodynamics_22b016db38977aed",
                "smith_van_ness_chemical_engineering_thermodynamics_828e7ca876a6e585",
            ],
        }
    ],
    "DQ013": [
        {
            "type": "alternative",
            "chunk_ids": [
                "smith_van_ness_chemical_engineering_thermodynamics_ac5b80df7ff72fc5",
                "smith_van_ness_chemical_engineering_thermodynamics_b7bd3deb5d25cf2e",
            ],
        }
    ],
    "DQ015": [
        {
            "type": "alternative",
            "chunk_ids": [
                "smith_van_ness_chemical_engineering_thermodynamics_59380007ee763c66",
                "smith_van_ness_chemical_engineering_thermodynamics_f74d91dc17ddf930",
            ],
        }
    ],
    "DQ018": [
        {
            "type": "alternative",
            "chunk_ids": [
                "incropera_fundamentals_heat_mass_transfer_9c14d395d096356a",
            ],
        }
    ],
    "DQ020": [
        {
            "type": "alternative",
            "chunk_ids": [
                "incropera_fundamentals_heat_mass_transfer_9c14d395d096356a",
            ],
        }
    ],
    "DQ024": [
        {
            "type": "alternative",
            "chunk_ids": [
                "ocp_phosphoric_acid_workshop_report_4dd00f7a95aa37d9",
                "ocp_phosphoric_acid_workshop_report_1d67a78d647c6289",
                "ocp_phosphoric_acid_workshop_report_d87f0d8bc2b03e4c",
            ],
        }
    ],
    "DQ025": [
        {
            "type": "alternative",
            "chunk_ids": [
                "perrys_chemical_engineers_handbook_1666e4fbdc480b49",
                "perrys_chemical_engineers_handbook_2594c93f8834e0d2",
            ],
        }
    ],
    "DQ027": [
        {
            "type": "alternative",
            "chunk_ids": ["bird_transport_phenomena_64e49a695bbec9de"],
        },
        {
            "type": "complementary",
            "groups": [
                ["bird_transport_phenomena_61f9e0c7a11d29e3"],
                ["bird_transport_phenomena_0f0156da49f6a3c1"],
            ],
        },
    ],
    "DQ035": [
        {
            "type": "alternative",
            "chunk_ids": [
                "mullin_crystallization_16f0e8b4ae22773d",
                "mullin_crystallization_4f5e049378e93d66",
            ],
        }
    ],
    "DQ037": [
        {
            "type": "alternative",
            "chunk_ids": [
                "mullin_crystallization_d14d86855c1f5521",
                "mullin_crystallization_8f888a2b16fd329a",
            ],
        }
    ],
    "DQ038": [
        {
            "type": "alternative",
            "chunk_ids": ["seborg_process_dynamics_control_d7620d288dba471a"],
        }
    ],
    "DQ039": [
        {
            "type": "complementary",
            "groups": [
                [
                    "seborg_process_dynamics_control_d7620d288dba471a",
                    "seborg_process_dynamics_control_e99ec9b53deeb4d4",
                ],
                [
                    "seborg_process_dynamics_control_83b1283c6579e4fe",
                    "seborg_process_dynamics_control_9095fc8fd57338b4",
                ],
            ],
        }
    ],
}

RATIONALES = {
    "CE051": "Becker p.11 states altered heat transfer and pressure drop; pp.102-103 independently document deposits, plugged pipework and equipment cleaning.",
    "CE056": "Mullin p.201 directly links crossing critical supersaturation to a rapid nucleation-rate increase.",
    "DQ009": "The old p.12 passage is historical context; pp.29 and 39 actually compare process strength, equipment, cost and operating conditions.",
    "DQ011": "The old gold is an exercise prompt. Page 262 states the vapor-pressure/temperature relation and Clapeyron connection.",
    "DQ013": "The old phase-rule passage is generic; pp.451-452 describe VLE coexistence and pressure reduction at fixed temperature/composition.",
    "DQ015": "The old heat-effects overview lacks a steady-flow balance; pp.71 and 441 explicitly use enthalpy in steady flow and an evaporator.",
    "DQ018": "The old passage discusses U, not the exchanger role; p.734 defines exchange between fluids separated by a wall.",
    "DQ020": "The old passage discusses U, not the exchanger role; p.734 directly defines the device and its purpose.",
    "DQ027": "Bird p.56 is sufficient, while the circular-tube setup on p.63 plus the shell balance on p.64 form a second complete route.",
    "DQ035": "The old theory passage only alludes to two effects; pp.252 and 325 directly list temperature, supersaturation/concentration, agitation/turbulence and related factors.",
    "DQ037": "The old p.265 passage omits residence time; pp.422 and 430 explicitly connect residence time, grown size and CSD.",
    "DQ038": "The old control-design overview does not identify evaporator MVs; p.246 explicitly lists steam pressure, product flow and vapor flow.",
    "DQ039": "No single audited passage applies MPC to an evaporator; valid support requires an evaporator-variable passage plus an MPC design/calculation passage.",
}

REGION_EXTRA_IDS = {
    "DQ027": ["bird_transport_phenomena_de3ab1178d32562c"],
    "DQ036": ["mullin_crystallization_a4b7c1477593a5fa"],
}


def _excerpt(text: str, limit: int) -> str:
    return " ".join(text.split())[:limit]


def _rank(ids: list[str], chunk_id: str) -> int | None:
    return ids.index(chunk_id) + 1 if chunk_id in ids else None


def _chunk_line(chunk: TechnicalChildChunk, *, rank: int | None = None) -> str:
    prefix = f"rank={rank} " if rank is not None else ""
    return (
        f"- {prefix}`{chunk.chunk_id}` p.{chunk.page_start}-{chunk.page_end} "
        f"section={chunk.section!r} parent={chunk.parent_id!r}\n"
        f"  {_excerpt(chunk.text, 700)}"
    )


def build_review_packet(
    phase7_directory: Path = PHASE7_OUTPUT,
) -> tuple[str, str, dict[str, Any]]:
    """Create compact all-question and deep-failure packets for manual review."""

    records = _evaluation_records(phase7_directory)
    children = read_child_chunks(ACTIVE_DIRECTORY / "chunks.jsonl")
    child_by_id = {child.chunk_id: child for child in children}
    parents = {
        parent.parent_id: parent
        for parent in read_parent_chunks(ACTIVE_DIRECTORY / "parents.jsonl")
    }
    all_lines = ["# Phase 8 manual gold review packet", ""]
    failure_lines = ["# Phase 8 deep failure review packet", ""]
    machine: dict[str, Any] = {}
    for record in records:
        question_id = record["id"]
        gold_ids = list(record["gold_chunk_ids"])
        current_final = list(record["current_reranked_ids"])
        candidate_final = list(record["reranked_ids"])
        current_pool = list(record["current_candidate_ids"])
        candidate_pool = list(record["candidate_ids"])
        header = (
            f"## {question_id} [{record['split']}] [{record['language']}] "
            f"{record['locked_document']}"
        )
        lines = [
            header,
            "",
            f"Question: {record['question']}",
            "",
            f"Type: `{record['question_type']}`",
            "",
            "Gold:",
            "",
        ]
        region_ids: set[str] = set()
        gold_details = []
        for gold_id in gold_ids:
            gold = child_by_id[gold_id]
            lines.append(_chunk_line(gold))
            parent = parents[gold.parent_id]
            related = list(parent.child_chunk_ids)
            region_ids.update(related)
            gold_details.append(
                {
                    "chunk_id": gold_id,
                    "page_start": gold.page_start,
                    "page_end": gold.page_end,
                    "section": gold.section,
                    "parent_id": gold.parent_id,
                    "previous_chunk_id": gold.previous_chunk_id,
                    "next_chunk_id": gold.next_chunk_id,
                    "parent_child_ids": related,
                    "current_pool_rank": _rank(current_pool, gold_id),
                    "current_final_rank": _rank(current_final, gold_id),
                    "candidate_pool_rank": _rank(candidate_pool, gold_id),
                    "candidate_final_rank": _rank(candidate_final, gold_id),
                }
            )
        current_region = [item for item in current_final if item in region_ids]
        candidate_region = [item for item in candidate_final if item in region_ids]
        lines += [
            "",
            "Top current production evidence:",
            "",
        ]
        for rank, chunk_id in enumerate(current_final[:5], 1):
            lines.append(_chunk_line(child_by_id[chunk_id], rank=rank))
        lines += ["", "Top Phase-7 candidate evidence:", ""]
        for rank, chunk_id in enumerate(candidate_final[:5], 1):
            lines.append(_chunk_line(child_by_id[chunk_id], rank=rank))
        lines += [
            "",
            f"Current same-parent hits: `{current_region}`",
            f"Candidate same-parent hits: `{candidate_region}`",
            "",
        ]
        all_lines.extend(lines)
        if question_id in FAILURE_IDS:
            deep = list(lines)
            deep += ["Current ranks 6-30:", ""]
            for rank, chunk_id in enumerate(current_final[5:30], 6):
                deep.append(_chunk_line(child_by_id[chunk_id], rank=rank))
            deep += ["", "Candidate ranks 6-50:", ""]
            for rank, chunk_id in enumerate(candidate_final[5:50], 6):
                deep.append(_chunk_line(child_by_id[chunk_id], rank=rank))
            deep += ["", "Gold parent regions:", ""]
            for chunk_id in sorted(region_ids):
                deep.append(_chunk_line(child_by_id[chunk_id]))
            deep.append("")
            failure_lines.extend(deep)
        machine[question_id] = {
            "question": record["question"],
            "split": record["split"],
            "language": record["language"],
            "question_type": record["question_type"],
            "locked_document": record["locked_document"],
            "gold": gold_details,
            "current_candidate_ids": current_pool,
            "current_reranked_ids": current_final,
            "candidate_ids": candidate_pool,
            "candidate_reranked_ids": candidate_final,
            "current_same_parent_hits": current_region,
            "candidate_same_parent_hits": candidate_region,
        }
    return "\n".join(all_lines), "\n".join(failure_lines), machine


def write_review_packet(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    all_packet, failure_packet, machine = build_review_packet()
    (output_directory / "review_packet_all.md").write_text(all_packet, encoding="utf-8")
    (output_directory / "review_packet_failures.md").write_text(
        failure_packet, encoding="utf-8"
    )
    (output_directory / "review_packet.json").write_text(
        json.dumps(machine, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "question_count": len(machine),
        "failure_count": sum(question_id in FAILURE_IDS for question_id in machine),
        "active_index": ACTIVE_DIRECTORY.name,
    }


def _default_evidence_sets(gold_ids: Sequence[str]) -> list[dict[str, Any]]:
    if len(gold_ids) == 1:
        return [{"type": "alternative", "chunk_ids": list(gold_ids)}]
    return [
        {
            "type": "complementary",
            "groups": [[chunk_id] for chunk_id in gold_ids],
        }
    ]


def _evidence_ids(evidence_sets: Sequence[dict[str, Any]]) -> set[str]:
    output: set[str] = set()
    for evidence_set in evidence_sets:
        if evidence_set["type"] == "alternative":
            output.update(evidence_set["chunk_ids"])
        else:
            for group in evidence_set["groups"]:
                output.update(group)
    return output


def _solution_coverage(evidence_set: dict[str, Any], retrieved: set[str]) -> float:
    if evidence_set["type"] == "alternative":
        return float(bool(retrieved & set(evidence_set["chunk_ids"])))
    groups = evidence_set["groups"]
    return sum(bool(retrieved & set(group)) for group in groups) / len(groups)


def _evidence_coverage(evidence_sets: Sequence[dict[str, Any]], ranking: Sequence[str]) -> float:
    retrieved = set(ranking)
    return max(_solution_coverage(solution, retrieved) for solution in evidence_sets)


def _evidence_best_rank(
    evidence_sets: Sequence[dict[str, Any]], ranking: Sequence[str]
) -> int | None:
    ranks = {chunk_id: rank for rank, chunk_id in enumerate(ranking, 1)}
    completed_at: list[int] = []
    for solution in evidence_sets:
        if solution["type"] == "alternative":
            hits = [ranks[chunk_id] for chunk_id in solution["chunk_ids"] if chunk_id in ranks]
            if hits:
                completed_at.append(min(hits))
            continue
        group_ranks = [
            min((ranks[item] for item in group if item in ranks), default=None)
            for group in solution["groups"]
        ]
        if all(rank is not None for rank in group_ranks):
            completed_at.append(max(int(rank) for rank in group_ranks if rank is not None))
    return min(completed_at, default=None)


def _ranking_metrics(
    records: Sequence[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    ranking_key: str,
) -> dict[str, float]:
    exact_ranks: list[int | None] = []
    evidence_ranks: list[int | None] = []
    region_ranks: list[int | None] = []
    exact_coverages: dict[int, list[float]] = {5: [], 10: [], 20: []}
    coverages: dict[int, list[float]] = {5: [], 10: [], 20: []}
    for record in records:
        ranking = list(record[ranking_key])
        exact_ranks.append(
            min(
                (_rank(ranking, chunk_id) for chunk_id in record["gold_chunk_ids"]),
                key=lambda value: value if value is not None else 10**9,
            )
        )
        annotation = annotations[record["id"]]
        evidence_ranks.append(_evidence_best_rank(annotation["valid_evidence_sets"], ranking))
        region_ranks.append(
            min(
                (_rank(ranking, chunk_id) for chunk_id in annotation["region_chunk_ids"]),
                key=lambda value: value if value is not None else 10**9,
            )
        )
        for cutoff in coverages:
            exact_coverages[cutoff].append(
                len(set(record["gold_chunk_ids"]) & set(ranking[:cutoff]))
                / len(record["gold_chunk_ids"])
            )
            coverages[cutoff].append(
                _evidence_coverage(annotation["valid_evidence_sets"], ranking[:cutoff])
            )

    def recall_at(ranks: Sequence[int | None], cutoff: int) -> float:
        return sum(rank is not None and rank <= cutoff for rank in ranks) / len(ranks)

    def reciprocal_rank(ranks: Sequence[int | None]) -> float:
        return sum(1.0 / rank for rank in ranks if rank is not None) / len(ranks)

    output: dict[str, float] = {}
    for cutoff in (5, 10, 20):
        output[f"exact_recall_at_{cutoff}"] = sum(exact_coverages[cutoff]) / len(records)
        output[f"evidence_set_recall_at_{cutoff}"] = recall_at(evidence_ranks, cutoff)
        output[f"question_evidence_coverage_at_{cutoff}"] = sum(coverages[cutoff]) / len(
            records
        )
        output[f"region_recall_at_{cutoff}"] = recall_at(region_ranks, cutoff)
    output["exact_mrr"] = reciprocal_rank(exact_ranks)
    output["evidence_set_mrr"] = reciprocal_rank(evidence_ranks)
    return output


def _access_metrics(
    records: Sequence[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    ranking_key: str,
) -> dict[str, float]:
    exact = 0
    evidence = 0
    coverage = 0.0
    for record in records:
        ranking = list(record[ranking_key])
        exact += len(set(record["gold_chunk_ids"]) & set(ranking)) / len(
            record["gold_chunk_ids"]
        )
        annotation = annotations[record["id"]]
        value = _evidence_coverage(annotation["valid_evidence_sets"], ranking)
        evidence += value == 1.0
        coverage += value
    return {
        "exact_access_recall": exact / len(records),
        "evidence_set_access_recall": evidence / len(records),
        "question_evidence_access_coverage": coverage / len(records),
    }


def build_manual_annotations(
    phase7_directory: Path = PHASE7_OUTPUT,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records = _evaluation_records(phase7_directory)
    annotations: dict[str, dict[str, Any]] = {}
    for record in records:
        question_id = record["id"]
        evidence_sets = EVIDENCE_OVERRIDES.get(
            question_id, _default_evidence_sets(record["gold_chunk_ids"])
        )
        status = (
            "corrected"
            if question_id in CORRECTED_IDS
            else "added_alternative_evidence"
            if question_id in ADDED_EVIDENCE_IDS
            else "unchanged"
        )
        region_ids = _evidence_ids(evidence_sets)
        region_ids.update(REGION_EXTRA_IDS.get(question_id, []))
        category = CATEGORY_BY_QUESTION.get(question_id, 1)
        annotations[question_id] = {
            "question_id": question_id,
            "question": record["question"],
            "split": record["split"],
            "language": record["language"],
            "locked_document": record["locked_document"],
            "historical_gold_chunk_ids": list(record["gold_chunk_ids"]),
            "historical_gold_category": category,
            "historical_gold_category_label": CATEGORY_LABELS[category],
            "audit_status": status,
            "valid_evidence_sets": evidence_sets,
            "region_chunk_ids": sorted(region_ids),
            "documentary_justification": RATIONALES.get(
                question_id,
                "Historical gold text was manually checked against its question, page, section and retrieved evidence and remains sufficient.",
            ),
        }
    return records, annotations


def _failure_audit(
    records: Sequence[dict[str, Any]], annotations: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    by_id = {record["id"]: record for record in records}
    answer_status = {
        "CE051": "NO",
        "CE056": "YES",
        "DQ009": "YES",
        "DQ011": "YES",
        "DQ013": "YES",
        "DQ015": "PARTIALLY_CURRENT_YES_CANDIDATE",
        "DQ027": "NO",
        "DQ037": "YES",
        "DQ038": "YES",
        "DQ028": "NO",
        "DQ035": "YES",
        "DQ039": "YES_CURRENT_COMPLEMENTARY",
    }
    output: dict[str, Any] = {}
    for question_id in sorted(FAILURE_IDS):
        record = by_id[question_id]
        annotation = annotations[question_id]
        evidence_ids = _evidence_ids(annotation["valid_evidence_sets"])
        output[question_id] = {
            "question": record["question"],
            "current_gold": list(record["gold_chunk_ids"]),
            "top_current_chunk_ids": list(record["current_reranked_ids"][:10]),
            "top_candidate_chunk_ids": list(record["reranked_ids"][:10]),
            "top_retrieved_answers": answer_status[question_id],
            "valid_alternative_evidence_found": bool(
                evidence_ids - set(record["gold_chunk_ids"])
            ),
            "gold_change_justified": annotation["audit_status"] != "unchanged",
            "documentary_justification": annotation["documentary_justification"],
        }
    return output


def _holdout_changes(annotations: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "question_id": annotation["question_id"],
            "question": annotation["question"],
            "change_type": annotation["audit_status"],
            "old_gold": annotation["historical_gold_chunk_ids"],
            "new_or_alternative_gold": annotation["valid_evidence_sets"],
            "documentary_justification": annotation["documentary_justification"],
        }
        for annotation in annotations.values()
        if annotation["split"] == "final_holdout"
        and annotation["audit_status"] != "unchanged"
    ]


def _hard_misses(
    records: Sequence[dict[str, Any]], annotations: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {record["id"]: record for record in records}
    return [
        {
            "question_id": "CE051",
            "query": by_id["CE051"]["question"],
            "valid_evidence": sorted(
                _evidence_ids(annotations["CE051"]["valid_evidence_sets"])
            ),
            "dense_top_200_best_relevant_rank": 169,
            "sparse_best_relevant_rank": 245,
            "bm25_best_relevant_rank": None,
            "pure_rrf_fusion_best_relevant_rank": 350,
            "candidate_access": False,
        },
        {
            "question_id": "DQ027",
            "query": by_id["DQ027"]["question"],
            "valid_evidence": sorted(
                _evidence_ids(annotations["DQ027"]["valid_evidence_sets"])
            ),
            "dense_top_200_best_relevant_rank": None,
            "sparse_best_relevant_rank": 52,
            "bm25_best_relevant_rank": 115,
            "pure_rrf_fusion_best_relevant_rank": 242,
            "candidate_access": False,
        },
    ]


def _representation_diagnostics() -> dict[str, Any]:
    chunks = {
        child.chunk_id: child
        for child in read_child_chunks(ACTIVE_DIRECTORY / "chunks.jsonl")
    }
    definitions = {
        "CE051": {
            "question_concepts": [
                "entartrage/scaling",
                "effets sur les équipements",
                "transfert thermique",
                "perte de charge",
            ],
            "relevant_ids": [
                "becker_phosphates_and_phosphoric_acid_bd5716159148dec0",
                "becker_phosphates_and_phosphoric_acid_2f2a160a376652cc",
            ],
            "diagnosis": "French query uses entartrage and asks for effects; English chunks use scaling/deposits and often describe consequences indirectly.",
        },
        "DQ027": {
            "question_concepts": [
                "bilan de quantité de mouvement",
                "conduite",
                "flux convectif et moléculaire",
                "pression/viscosité/gravité",
            ],
            "relevant_ids": [
                "bird_transport_phenomena_64e49a695bbec9de",
                "bird_transport_phenomena_61f9e0c7a11d29e3",
                "bird_transport_phenomena_0f0156da49f6a3c1",
            ],
            "diagnosis": "The French abstraction maps to English shell momentum balance; tube context and the useful equation are split across adjacent children.",
        },
    }
    output: dict[str, Any] = {}
    for question_id, definition in definitions.items():
        representations = []
        for chunk_id in definition["relevant_ids"]:
            chunk = chunks[chunk_id]
            representations.append(
                {
                    "chunk_id": chunk_id,
                    "display_text": _excerpt(chunk.display_text, 700),
                    "embedding_text": _excerpt(chunk.embedding_text, 700),
                    "bm25_text": _excerpt(chunk.bm25_text, 700),
                    "section": chunk.section,
                    "hierarchy": list(chunk.hierarchy_path),
                    "previous_chunk_id": chunk.previous_chunk_id,
                    "next_chunk_id": chunk.next_chunk_id,
                    "parent_id": chunk.parent_id,
                }
            )
        output[question_id] = {**definition, "relevant_representations": representations}
    return output


def build_phase8_results(
    phase7_directory: Path = PHASE7_OUTPUT,
) -> dict[str, Any]:
    records, annotations = build_manual_annotations(phase7_directory)
    splits = {
        "all": records,
        "dev": [record for record in records if record["split"] == "dev"],
        "final_holdout": [
            record for record in records if record["split"] == "final_holdout"
        ],
    }
    metrics: dict[str, Any] = {}
    for split_name, split_records in splits.items():
        metrics[split_name] = {
            "current": {
                **_ranking_metrics(
                    split_records, annotations, "current_reranked_ids"
                ),
                **_access_metrics(
                    split_records, annotations, "current_candidate_ids"
                ),
            },
            "phase7_candidate": {
                **_ranking_metrics(split_records, annotations, "reranked_ids"),
                **_access_metrics(split_records, annotations, "candidate_ids"),
            },
        }
    category_counts = {
        str(category): sum(
            annotation["historical_gold_category"] == category
            for annotation in annotations.values()
        )
        for category in CATEGORY_LABELS
    }
    planner_details = json.loads(
        (phase7_directory / "qwen_search_plan_rank_details.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "dataset": {
            "question_count": len(records),
            "unchanged": sum(
                annotation["audit_status"] == "unchanged"
                for annotation in annotations.values()
            ),
            "added_alternative_evidence": len(ADDED_EVIDENCE_IDS),
            "corrected": len(CORRECTED_IDS),
            "questionable_historical_gold": category_counts["5"],
            "category_counts": category_counts,
        },
        "annotations": annotations,
        "metrics": metrics,
        "failure_audit": _failure_audit(records, annotations),
        "true_hard_misses": _hard_misses(records, annotations),
        "representation_mismatch_taxonomy": {
            "CE051": [1, 2, 3],
            "DQ027": [1, 2, 3, 4, 5, 6, 7],
        },
        "representation_diagnostics": _representation_diagnostics(),
        "query_planner_diagnostic": {
            question_id: planner_details["questions"][question_id]
            for question_id in ("CE051", "DQ027")
        },
        "holdout_gold_changes": _holdout_changes(annotations),
    }


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _metric_table(values: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| Pipeline | Exact R@5 | Exact R@10 | Exact R@20 | Evidence-set R@5 | Evidence-set R@10 | Evidence-set R@20 | Coverage@20 | Exact MRR | Evidence MRR | Access |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("Current", "current"), ("Phase-7 candidate", "phase7_candidate")):
        item = values[key]
        lines.append(
            f"| {label} | {_pct(item['exact_recall_at_5'])} | "
            f"{_pct(item['exact_recall_at_10'])} | {_pct(item['exact_recall_at_20'])} | "
            f"{_pct(item['evidence_set_recall_at_5'])} | "
            f"{_pct(item['evidence_set_recall_at_10'])} | "
            f"{_pct(item['evidence_set_recall_at_20'])} | "
            f"{_pct(item['question_evidence_coverage_at_20'])} | "
            f"{item['exact_mrr']:.3f} | {item['evidence_set_mrr']:.3f} | "
            f"{_pct(item['evidence_set_access_recall'])} |"
        )
    return lines


def render_phase8_report(results: dict[str, Any], gates: dict[str, Any]) -> str:
    dataset = results["dataset"]
    annotations = results["annotations"]
    metrics = results["metrics"]
    failures = results["failure_audit"]
    lines = [
        "# 1. GOLD AUDIT",
        "",
        (
            f"64 questions audited manually: {dataset['unchanged']} unchanged, "
            f"{dataset['added_alternative_evidence']} with added alternative evidence, "
            f"{dataset['corrected']} corrected, and "
            f"{dataset['questionable_historical_gold']} historical assignments classified "
            "as questionable. Category counts (1→5): "
            f"`{json.dumps(dataset['category_counts'], sort_keys=True)}`."
        ),
        "",
        "| ID | Split | Historical class | Audit action |",
        "|---|---|---|---|",
    ]
    for question_id in sorted(annotations):
        annotation = annotations[question_id]
        lines.append(
            f"| {question_id} | {annotation['split']} | "
            f"{annotation['historical_gold_category']}. "
            f"{annotation['historical_gold_category_label']} | "
            f"{annotation['audit_status']} |"
        )
    lines += [
        "",
        "# 2. EXACT VS EVIDENCE-SET METRICS",
        "",
        "All 64 questions; exact recall preserves the historical chunk IDs, while evidence-set recall uses only the manually verified sets.",
        "",
        *_metric_table(metrics["all"]),
        "",
        "# 3. REGION RECALL",
        "",
        "A region hit is the historical chunk or a manually verified supporting adjacent/parent passage; section membership alone never counts.",
        "",
        "| Pipeline | Region R@5 | Region R@10 | Region R@20 |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (("Current", "current"), ("Phase-7 candidate", "phase7_candidate")):
        item = metrics["all"][key]
        lines.append(
            f"| {label} | {_pct(item['region_recall_at_5'])} | "
            f"{_pct(item['region_recall_at_10'])} | {_pct(item['region_recall_at_20'])} |"
        )
    lines += [
        "",
        "Neighbor/parent finding: DQ027 retrieves the chapter-introduction child only deeply, while the circular-tube setup and balance children remain outside the candidate pool. DQ036 retrieves its verified adjacent curve explanation at rank 4, but its exact gold is already rank 1.",
        "",
        "# 4. ACCESS FAILURES RECLASSIFIED",
        "",
        "| ID | Top evidence answers? | Valid alternative? | Gold change? | Result |",
        "|---|---|---:|---:|---|",
    ]
    for question_id in (
        "CE051",
        "CE056",
        "DQ009",
        "DQ011",
        "DQ013",
        "DQ015",
        "DQ027",
        "DQ037",
        "DQ038",
    ):
        item = failures[question_id]
        lines.append(
            f"| {question_id} | {item['top_retrieved_answers']} | "
            f"{'YES' if item['valid_alternative_evidence_found'] else 'NO'} | "
            f"{'YES' if item['gold_change_justified'] else 'NO'} | "
            f"{item['documentary_justification']} |"
        )
    lines += [
        "",
        "# 5. RERANKER FAILURES RECLASSIFIED",
        "",
        "| ID | Manual verdict | Gold verdict | Why competitors win |",
        "|---|---|---|---|",
        "| DQ028 | No valid answer in current top 20; exact Perry passage reaches the Phase-7 pool at 20 but is demoted to 44. | A — gold is the best direct evidence. | Generic index/pump/mixing passages share conduit/flow vocabulary; the direct laminar/turbulent passage has weaker query-language overlap. |",
        "| DQ035 | Current rank 1 is a valid, more direct answer; the old gold is rank 17/24. | C — old gold is indirect. | Rank 1 explicitly lists temperature, concentration and agitation; the gold mainly defines face-growth velocity. |",
        "| DQ039 | Current top 20 jointly contains evaporator evidence (rank 18) and MPC method evidence (rank 9); exact gold alone ranks 24/37. | C — old gold alone is insufficient. | MPC-specific distillation/implementation passages score higher; the query requires a complementary evaporator passage that the old label omitted. |",
        "",
        "Frozen Phase-7 candidate scores are preserved in `reranker_failure_details.json`. DQ028's omitted holdout scores were reproduced once with the unchanged candidate IDs and unchanged BGE reranker and recorded in `dq028_reranker_score_backfill.json`.",
        "",
        "# 6. TRUE HARD MISSES",
        "",
        "Only CE051 and DQ027 have no complete manually verified evidence path in either reranker candidate pool.",
        "",
        "| ID | Dense best | Sparse best | BM25 best | Pure-RRF fusion best | Candidate access |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for miss in results["true_hard_misses"]:
        fmt = lambda value: str(value) if value is not None else ">200 / absent"  # noqa: E731
        lines.append(
            f"| {miss['question_id']} | {fmt(miss['dense_top_200_best_relevant_rank'])} | "
            f"{fmt(miss['sparse_best_relevant_rank'])} | "
            f"{fmt(miss['bm25_best_relevant_rank'])} | "
            f"{fmt(miss['pure_rrf_fusion_best_relevant_rank'])} | NO |"
        )
    lines += [
        "",
        "- CE051 query: `Selon Becker, quels effets l’entartrage produit-il sur les équipements ?` Valid evidence: Becker p.11 (`...bd571615...`), p.102 (`...7ee6f819...`) or p.103 (`...2f2a160a...`).",
        "- DQ027 query: `Explique le bilan de quantité de mouvement dans une conduite.` Valid evidence: Bird p.56 (`...64e49a69...`) OR the complementary circular-tube setup p.63 (`...61f9e0c7...`) plus balance equation p.64 (`...0f0156da...`).",
    ]
    lines += [
        "",
        "# 7. REPRESENTATION MISMATCH TAXONOMY",
        "",
        "| Class | CE051 | DQ027 |",
        "|---|---:|---:|",
    ]
    labels = {
        1: "terminology mismatch",
        2: "cross-language mismatch",
        3: "answer described indirectly",
        4: "relevant term only in neighbor",
        5: "heading carries essential semantics",
        6: "chunk lacks context",
        7: "boundary splits relation",
        8: "query too broad",
        9: "application context absent from passage",
    }
    taxonomy = results["representation_mismatch_taxonomy"]
    for number, label in labels.items():
        lines.append(
            f"| {number}. {label} | {'YES' if number in taxonomy['CE051'] else '—'} | "
            f"{'YES' if number in taxonomy['DQ027'] else '—'} |"
        )
    lines += [
        "",
        "CE051 is chiefly French `entartrage/équipements` versus English `scaling/deposits/pipework`. DQ027 is French abstract terminology versus English `shell momentum balance`, with tube setup and balance equation split across children.",
        "",
        "# 8. QUERY-PLANNER DIAGNOSTIC",
        "",
        "The already frozen, label-free Qwen plans retained the original query and received no gold, page, section, document lock or expected answer. Neither true hard miss improved: all Dense/Sparse/BM25 gold ranks remained absent for CE051 and DQ027, and candidate access stayed false.",
        "",
        "| ID | Original primary-gold best | Planner primary-gold best | Candidate access |",
        "|---|---|---|---:|",
        "| CE051 | Dense 169; Sparse/BM25 absent | Dense/Sparse/BM25 absent | NO |",
        "| DQ027 | Dense/Sparse/BM25 absent | Dense/Sparse/BM25 absent | NO |",
        "",
        "# 9. DEV RESULTS",
        "",
        *_metric_table(metrics["dev"]),
        "",
        "# 10. HOLDOUT RESULTS",
        "",
        "Frozen holdout was evaluated once with the locked Phase-7 configuration; no retuning followed label review.",
        "",
        *_metric_table(metrics["final_holdout"]),
        "",
        "# 11. HOLDOUT LABEL CHANGES",
        "",
        f"{len(results['holdout_gold_changes'])} explicit changes; full old/new IDs and documentary justifications are in `holdout_gold_changes.json`.",
        "",
        "| ID | Change | Justification |",
        "|---|---|---|",
    ]
    for change in results["holdout_gold_changes"]:
        lines.append(
            f"| {change['question_id']} | {change['change_type']} | "
            f"{change['documentary_justification']} |"
        )
    evidence_r20 = metrics["all"]["current"]["evidence_set_recall_at_20"]
    lines += [
        "",
        "# 12. ARCHITECTURAL CONCLUSION",
        "",
        (
            f"Case A dominates: current evidence-set R@20 is {_pct(evidence_r20)}, "
            "materially above exact recall. The benchmark was understating retrieval quality. "
            "Two genuine access misses remain and DQ028 remains a genuine reranker demotion, "
            "so the system also has smaller Case C/D/E components. Production remains unchanged."
        ),
        "",
        "# 13. ONE PHASE-9 RECOMMENDATION",
        "",
        "Run one DEV-only region-aware candidate-access experiment that promotes a strongly retrieved child into its verified adjacent/parent region, targeting CE051/DQ027 without rebuilding indexes; keep the frozen holdout sealed until the policy is fixed.",
        "",
        "# 14. TEST GATES",
        "",
        "```json",
        json.dumps(gates, ensure_ascii=False, indent=2),
        "```",
        "",
        "Phase 8 stops here: no production-ranking edit, model change, Evidence Judge, answer repair, Query Planner deployment, rechunking, reindexing or index rebuild was performed.",
    ]
    return "\n".join(lines) + "\n"


def _reranker_failure_details(
    phase7_directory: Path, records: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    selected_scores: dict[str, dict[str, float]] = {}
    for line in (phase7_directory / "dev_results.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        row = json.loads(line)
        if row["id"] in {"DQ035", "DQ039"}:
            selected_scores[row["id"]] = row["candidate_scores"]
    dq028_path = DEFAULT_OUTPUT / "dq028_reranker_score_backfill.json"
    if dq028_path.exists():
        dq028 = json.loads(dq028_path.read_text(encoding="utf-8"))
        selected_scores["DQ028"] = {
            item["chunk_id"]: item["score"] for item in dq028["top_10"]
        }
    output: dict[str, Any] = {}
    for record in records:
        if record["id"] not in {"DQ028", "DQ035", "DQ039"}:
            continue
        scores = selected_scores.get(record["id"], {})
        output[record["id"]] = {
            "question": record["question"],
            "historical_gold": record["gold_chunk_ids"],
            "phase7_candidate_top_10": [
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "score": scores.get(chunk_id),
                }
                for rank, chunk_id in enumerate(record["reranked_ids"][:10], 1)
            ],
            "score_note": (
                "unchanged BGE score persisted by Phase 7"
                if scores
                else "score not persisted in frozen holdout artifact; ranking retained"
            ),
        }
    return output


def write_phase8_outputs(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    results = build_phase8_results()
    gates_path = output_directory / "test_gates.json"
    gates = (
        json.loads(gates_path.read_text(encoding="utf-8"))
        if gates_path.exists()
        else {"status": "pending"}
    )
    records, _annotations = build_manual_annotations()
    (output_directory / "evidence_sets.json").write_text(
        json.dumps(results["annotations"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_directory / "metrics.json").write_text(
        json.dumps(results["metrics"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_directory / "failure_audit.json").write_text(
        json.dumps(results["failure_audit"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_directory / "true_hard_misses.json").write_text(
        json.dumps(results["true_hard_misses"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_directory / "representation_diagnostics.json").write_text(
        json.dumps(results["representation_diagnostics"], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (output_directory / "holdout_gold_changes.json").write_text(
        json.dumps(results["holdout_gold_changes"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_directory / "reranker_failure_details.json").write_text(
        json.dumps(
            _reranker_failure_details(PHASE7_OUTPUT, records),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_directory / "phase8_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_directory / "summary.md").write_text(
        render_phase8_report(results, gates), encoding="utf-8"
    )
    return {
        "question_count": results["dataset"]["question_count"],
        "corrected": results["dataset"]["corrected"],
        "added_alternative_evidence": results["dataset"][
            "added_alternative_evidence"
        ],
        "true_hard_misses": [
            item["question_id"] for item in results["true_hard_misses"]
        ],
    }


def backfill_holdout_reranker_scores(
    output_directory: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Persist missing DQ028 scores without changing candidates or reranker."""

    from phosprocess.evaluation.candidate_preservation_v01 import (  # noqa: PLC0415
        _locked_routing,
    )
    from phosprocess.rag.orchestrator import PhosProcessRAG  # noqa: PLC0415
    from phosprocess.retrieval.hybrid import HybridSearchResult  # noqa: PLC0415

    record = next(row for row in _evaluation_records(PHASE7_OUTPUT) if row["id"] == "DQ028")
    rag = PhosProcessRAG()
    engine = rag.quality_engine
    if engine is None:
        raise RuntimeError("Production quality engine is required")
    chunks = {
        chunk.chunk_id: chunk for chunk in engine.retriever.dense_retriever.metadata
    }
    candidates = [
        HybridSearchResult(
            rank=rank,
            rrf_score=0.0,
            matched_retrievers=(),
            dense_rank=None,
            dense_score=None,
            dense_rrf_contribution=0.0,
            bm25_rank=None,
            bm25_score=None,
            bm25_rrf_contribution=0.0,
            chunk=chunks[chunk_id],
        )
        for rank, chunk_id in enumerate(record["candidate_ids"], 1)
    ]
    response = engine.reranker.rerank(
        record["question"], candidates, top_k=len(candidates)
    )
    adjusted, _diagnostics = engine._adjust_reranking(
        response,
        routing=_locked_routing(
            engine,
            {**record, "classified_question_type": record["question_type"]},
        ),
        question_type=record["question_type"],
    )
    result = {
        "question_id": "DQ028",
        "question": record["question"],
        "candidate_ids_unchanged": True,
        "reranker_model_unchanged": True,
        "top_10": [
            {
                "rank": rank,
                "chunk_id": item.chunk.chunk_id,
                "score": float(item.reranker_score),
            }
            for rank, item in enumerate(adjusted.results[:10], 1)
        ],
        "historical_gold": record["gold_chunk_ids"],
        "historical_gold_rank": next(
            (
                rank
                for rank, item in enumerate(adjusted.results, 1)
                if item.chunk.chunk_id in set(record["gold_chunk_ids"])
            ),
            None,
        ),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "dq028_reranker_score_backfill.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-packet", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--reranker-score-backfill", action="store_true")
    args = parser.parse_args()
    if args.review_packet:
        value = write_review_packet(args.output)
    elif args.report:
        value = write_phase8_outputs(args.output)
    elif args.reranker_score_backfill:
        value = backfill_holdout_reranker_scores(args.output)
    else:
        parser.error("choose --review-packet, --report or --reranker-score-backfill")
    print(json.dumps(value, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
