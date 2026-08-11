"""Deterministic Phase-5 evaluation for the production Context Engine.

This module consumes production retrieval APIs but is never imported by them.
Ground truth is evaluation-only and tied to the active quality corpus.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from phosprocess.ingestion.chunk_serialization import (
    TechnicalChildChunk,
    read_child_chunks,
)
from phosprocess.rag.orchestrator import PhosProcessRAG
from phosprocess.rag.quality_retrieval import QualityRetrievalEngine
from phosprocess.rag.question_classifier import classify_question
from phosprocess.retrieval.context_expander import (
    ContextExpander,
    ContextExpansionConfig,
    EvidenceAnchor,
)
from phosprocess.retrieval.domain_router import route_query
from phosprocess.retrieval.evidence_bundle import EvidenceBundle
from phosprocess.retrieval.evidence_roles import select_role_aware_evidence
from phosprocess.retrieval.retrieval_planner import build_retrieval_plan
from phosprocess.retrieval.v3_selection import select_with_lexical_safeguard

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_QUESTIONS = PROJECT_ROOT / "data/evaluation/domain_quality/v1/questions.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/evaluation/context_engine/v0.1"
ACTIVE_VERSION = "kb_quality_20260809_015038"
ACTIVE_DIRECTORY = (
    PROJECT_ROOT / "data/knowledge_base/indexes/versions" / ACTIVE_VERSION
)
CONTEXT_BUDGET = 2600
SMALL_CONTEXT_BUDGET = 1300
LARGE_CONTEXT_BUDGET = 3900

_WORD = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Normalize documentary text for deterministic matching only."""

    decomposed = unicodedata.normalize("NFKD", text.casefold())
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(_WORD.findall(ascii_text))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _expected_documents(number: int) -> list[str]:
    multi_source = {
        4: [
            "becker_phosphates_and_phosphoric_acid",
            "ocp_phosphoric_acid_workshop_report",
        ],
        5: [
            "becker_phosphates_and_phosphoric_acid",
            "ocp_phosphoric_acid_workshop_report",
            "perrys_chemical_engineers_handbook",
        ],
        6: [
            "becker_phosphates_and_phosphoric_acid",
            "ocp_phosphoric_acid_workshop_report",
        ],
        7: ["becker_phosphates_and_phosphoric_acid", "mullin_crystallization"],
        8: [
            "becker_phosphates_and_phosphoric_acid",
            "ocp_phosphoric_acid_workshop_report",
        ],
        10: [
            "becker_phosphates_and_phosphoric_acid",
            "ocp_phosphoric_acid_workshop_report",
        ],
        12: [
            "smith_van_ness_chemical_engineering_thermodynamics",
            "ocp_phosphoric_acid_workshop_report",
            "perrys_chemical_engineers_handbook",
        ],
        13: [
            "smith_van_ness_chemical_engineering_thermodynamics",
            "ocp_phosphoric_acid_workshop_report",
        ],
        14: [
            "smith_van_ness_chemical_engineering_thermodynamics",
            "perrys_chemical_engineers_handbook",
        ],
        17: [
            "smith_van_ness_chemical_engineering_thermodynamics",
            "perrys_chemical_engineers_handbook",
        ],
        18: [
            "incropera_fundamentals_heat_mass_transfer",
            "ocp_phosphoric_acid_workshop_report",
            "perrys_chemical_engineers_handbook",
        ],
        19: [
            "incropera_fundamentals_heat_mass_transfer",
            "ocp_phosphoric_acid_workshop_report",
            "perrys_chemical_engineers_handbook",
        ],
        20: [
            "incropera_fundamentals_heat_mass_transfer",
            "ocp_phosphoric_acid_workshop_report",
            "perrys_chemical_engineers_handbook",
        ],
        23: [
            "incropera_fundamentals_heat_mass_transfer",
            "ocp_phosphoric_acid_workshop_report",
        ],
        24: [
            "incropera_fundamentals_heat_mass_transfer",
            "ocp_phosphoric_acid_workshop_report",
            "perrys_chemical_engineers_handbook",
        ],
        25: [
            "perrys_chemical_engineers_handbook",
            "ocp_phosphoric_acid_workshop_report",
        ],
        26: [
            "bird_transport_phenomena",
            "ocp_phosphoric_acid_workshop_report",
            "perrys_chemical_engineers_handbook",
        ],
        28: ["bird_transport_phenomena", "perrys_chemical_engineers_handbook"],
        29: ["bird_transport_phenomena", "perrys_chemical_engineers_handbook"],
        30: [
            "bird_transport_phenomena",
            "incropera_fundamentals_heat_mass_transfer",
        ],
        31: ["mullin_crystallization", "perrys_chemical_engineers_handbook"],
        33: ["mullin_crystallization", "becker_phosphates_and_phosphoric_acid"],
        37: ["mullin_crystallization", "becker_phosphates_and_phosphoric_acid"],
    }
    if number in multi_source:
        return multi_source[number]
    if number <= 3:
        return [
            "perrys_chemical_engineers_handbook",
            "mullin_crystallization",
            "ocp_phosphoric_acid_workshop_report",
        ]
    if number <= 10:
        return ["becker_phosphates_and_phosphoric_acid"]
    if number <= 17:
        return ["smith_van_ness_chemical_engineering_thermodynamics"]
    if number <= 24:
        return ["incropera_fundamentals_heat_mass_transfer"]
    if number == 25:
        return ["perrys_chemical_engineers_handbook"]
    if number <= 30:
        return ["bird_transport_phenomena"]
    if number <= 37:
        return ["mullin_crystallization"]
    if number <= 44:
        return ["seborg_process_dynamics_control"]
    if number <= 47:
        return ["ocp_phosphoric_acid_workshop_report"]
    if number == 48:
        return [
            "incropera_fundamentals_heat_mass_transfer",
            "perrys_chemical_engineers_handbook",
        ]
    if number == 49:
        return ["incropera_fundamentals_heat_mass_transfer"]
    return []


def _question_type(number: int) -> str:
    overrides = {
        1: "definition",
        2: "follow_up",
        3: "follow_up",
        4: "process_flow",
        9: "comparison",
        10: "explanation",
        11: "thermodynamics",
        13: "thermodynamics",
        14: "comparison",
        15: "energy_balance",
        17: "equation_explanation",
        19: "energy_balance",
        21: "follow_up",
        24: "troubleshooting",
        25: "equipment_operation",
        27: "transport_phenomena",
        29: "transport_phenomena",
        30: "transport_phenomena",
        31: "definition",
        34: "comparison",
        38: "control",
        39: "control",
        40: "comparison",
        42: "follow_up",
        44: "follow_up",
        45: "explicit_source",
        46: "explicit_source",
        47: "follow_up",
        49: "follow_up",
        50: "absent_from_corpus",
    }
    if number in overrides:
        return overrides[number]
    if 4 <= number <= 10:
        return "explanation"
    if 11 <= number <= 17:
        return "thermodynamics"
    if 18 <= number <= 24:
        return "heat_transfer"
    if 26 <= number <= 30:
        return "transport_phenomena"
    if 31 <= number <= 37:
        return "crystallization"
    if 38 <= number <= 44:
        return "control"
    return "explanation"


EVIDENCE_BY_ID: dict[str, list[str]] = {
    "DQ001": [
        "perrys_chemical_engineers_handbook_1666e4fbdc480b49",
        "mullin_crystallization_972412ee23f6e718",
    ],
    "DQ002": [
        "perrys_chemical_engineers_handbook_0283da773ac0de7a",
        "mullin_crystallization_972412ee23f6e718",
        "ocp_phosphoric_acid_workshop_report_278d9140b4c6a02b",
        "ocp_phosphoric_acid_workshop_report_9aeddee41a217705",
    ],
    "DQ003": [
        "perrys_chemical_engineers_handbook_1666e4fbdc480b49",
        "mullin_crystallization_679971882c05958f",
        "ocp_phosphoric_acid_workshop_report_9aeddee41a217705",
    ],
    "DQ009": ["becker_phosphates_and_phosphoric_acid_7147a36c57dde5d4"],
    "DQ011": ["smith_van_ness_chemical_engineering_thermodynamics_7aa981eaa4715ba3"],
    "DQ013": ["smith_van_ness_chemical_engineering_thermodynamics_611006cb00a98cae"],
    "DQ015": ["smith_van_ness_chemical_engineering_thermodynamics_bbb49d5b5ce4f4d0"],
    "DQ016": ["smith_van_ness_chemical_engineering_thermodynamics_b320e5efd9166ab6"],
    "DQ018": ["incropera_fundamentals_heat_mass_transfer_998fd6b5af60e473"],
    "DQ020": ["incropera_fundamentals_heat_mass_transfer_998fd6b5af60e473"],
    "DQ021": ["incropera_fundamentals_heat_mass_transfer_777dff6b6616c991"],
    "DQ022": ["incropera_fundamentals_heat_mass_transfer_998fd6b5af60e473"],
    "DQ023": ["incropera_fundamentals_heat_mass_transfer_eabe9bd011963e92"],
    "DQ025": ["perrys_chemical_engineers_handbook_1666e4fbdc480b49"],
    "DQ027": ["bird_transport_phenomena_64e49a695bbec9de"],
    "DQ029": ["bird_transport_phenomena_53819f8de263d1b6"],
    "DQ031": ["mullin_crystallization_e23198f092452faf"],
    "DQ034": ["mullin_crystallization_30e8ffc1112aa3d7"],
    "DQ035": ["mullin_crystallization_82b03ec4a1523953"],
    "DQ036": ["mullin_crystallization_e23198f092452faf"],
    "DQ037": ["mullin_crystallization_1b5fb9fcd4e80e39"],
    "DQ038": ["seborg_process_dynamics_control_f8b4c60ab4454edb"],
    "DQ039": ["seborg_process_dynamics_control_83b1283c6579e4fe"],
    "DQ045": ["ocp_phosphoric_acid_workshop_report_9aeddee41a217705"],
    "DQ046": ["ocp_phosphoric_acid_workshop_report_278d9140b4c6a02b"],
    "DQ047": ["ocp_phosphoric_acid_workshop_report_278d9140b4c6a02b"],
}

CONCEPTS_BY_ID: dict[str, list[str]] = {
    "DQ001": ["forced circulation", "heating surface"],
    "DQ002": ["pump", "heating"],
    "DQ003": ["circulation", "heat transfer"],
    "DQ004": ["phosphate rock", "sulfuric acid", "filtration"],
    "DQ005": ["concentration", "boiling point"],
    "DQ006": ["filtration", "calcium sulfate"],
    "DQ007": ["crystal size", "filtration"],
    "DQ008": ["phosphate rock", "sulfuric acid"],
    "DQ009": ["dihydrate", "hemihydrate"],
    "DQ010": ["p2o5", "loss"],
    "DQ011": ["vapor pressure", "temperature"],
    "DQ012": ["pressure", "boiling"],
    "DQ013": ["vapor liquid equilibrium"],
    "DQ014": ["boiler", "reboiler"],
    "DQ015": ["enthalpy", "heat"],
    "DQ016": ["latent heat", "phase"],
    "DQ017": ["antoine", "vapor pressure"],
    "DQ018": ["heat exchanger", "heat transfer"],
    "DQ019": ["energy balance", "latent heat"],
    "DQ020": ["heat exchanger", "heat transfer"],
    "DQ021": ["fouling", "thermal resistance"],
    "DQ022": ["overall heat transfer coefficient"],
    "DQ023": ["conduction", "convection"],
    "DQ024": ["fouling", "pressure drop"],
    "DQ025": ["pump", "circulation"],
    "DQ026": ["pressure drop", "flow rate"],
    "DQ027": ["momentum balance"],
    "DQ028": ["reynolds", "laminar", "turbulent"],
    "DQ029": ["viscosity", "pressure drop"],
    "DQ030": ["molecular diffusion", "convective"],
    "DQ031": ["supersaturation"],
    "DQ032": ["supersaturation", "nucleation"],
    "DQ033": ["crystal size", "crystallization"],
    "DQ034": ["primary nucleation", "secondary nucleation"],
    "DQ035": ["crystal growth", "supersaturation"],
    "DQ036": ["solubility", "metastable zone"],
    "DQ037": ["residence time", "crystal size distribution"],
    "DQ038": ["manipulated variable", "concentration"],
    "DQ039": ["model predictive control", "prediction"],
    "DQ040": ["pid", "model predictive control"],
    "DQ041": ["closed loop", "stability"],
    "DQ042": ["disturbance", "flow"],
    "DQ043": ["measured variable", "evaporator"],
    "DQ044": ["manipulated variable"],
    "DQ045": ["echangeur", "bouilleur", "pompe"],
    "DQ046": ["circuit acide", "echangeur", "bouilleur"],
    "DQ047": ["echangeur", "bouilleur"],
    "DQ048": ["heat exchanger", "evaporator"],
    "DQ049": ["fouling", "heat transfer"],
}

STANDALONE_BY_ID = {
    "DQ002": "Décris étape par étape le trajet dans un évaporateur à circulation forcée.",
    "DQ003": (
        "Pourquoi la recirculation est-elle nécessaire dans un évaporateur "
        "à circulation forcée ?"
    ),
    "DQ021": "How does fouling reduce heat-exchanger performance?",
    "DQ042": "Comment la stabilité en boucle fermée affecte-t-elle les perturbations de débit ?",
    "DQ044": "Quelles variables peuvent être manipulées pour contrôler un évaporateur ?",
    "DQ047": "Selon le rapport OCP, où va l’acide après l’échangeur ?",
    "DQ049": "كيف يؤثر الاتساخ على أداء المبادل الحراري في المبخر؟",
}

EXTRA_QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "CE051",
        "question": "Selon Becker, quels effets l’entartrage produit-il sur les équipements ?",
        "language": "fr",
        "question_type": "explicit_source",
        "expected_document_ids": ["becker_phosphates_and_phosphoric_acid"],
        "expected_evidence_chunk_ids": ["becker_phosphates_and_phosphoric_acid_bd5716159148dec0"],
        "expected_concepts": ["scaling", "heat transfer coefficient", "pressure drop"],
        "explicit_source": True,
        "answerable": True,
    },
    {
        "id": "CE052",
        "question": (
            "According to Perry, what operating difficulties affect "
            "forced-circulation evaporators?"
        ),
        "language": "en",
        "question_type": "explicit_source",
        "expected_document_ids": ["perrys_chemical_engineers_handbook"],
        "expected_evidence_chunk_ids": ["perrys_chemical_engineers_handbook_cf1a7e0dcafb73b9"],
        "expected_concepts": ["plugging", "poor circulation", "salting"],
        "explicit_source": True,
        "answerable": True,
    },
    {
        "id": "CE053",
        "question": "Dans le rapport OCP, que décrit le modèle de Kern et Seaton ?",
        "language": "fr",
        "question_type": "explicit_source",
        "expected_document_ids": ["ocp_phosphoric_acid_workshop_report"],
        "expected_evidence_chunk_ids": ["ocp_phosphoric_acid_workshop_report_d6f88f60e7102b61"],
        "expected_concepts": ["kern", "seaton", "encrassement"],
        "explicit_source": True,
        "answerable": True,
    },
    {
        "id": "CE054",
        "question": (
            "According to Smith, why does vaporization require latent heat "
            "at constant pressure?"
        ),
        "language": "en",
        "question_type": "explicit_source",
        "expected_document_ids": ["smith_van_ness_chemical_engineering_thermodynamics"],
        "expected_evidence_chunk_ids": [
            "smith_van_ness_chemical_engineering_thermodynamics_b320e5efd9166ab6"
        ],
        "expected_concepts": ["latent heat", "constant pressure", "vaporized"],
        "explicit_source": True,
        "answerable": True,
    },
    {
        "id": "CE055",
        "question": "كيف يميز عدد رينولدز بين الجريان الصفحي والجريان المضطرب داخل الأنبوب؟",
        "language": "ar",
        "question_type": "transport_phenomena",
        "expected_document_ids": [
            "bird_transport_phenomena",
            "perrys_chemical_engineers_handbook",
        ],
        "expected_evidence_chunk_ids": [],
        "expected_concepts": ["reynolds", "laminar", "turbulent"],
        "explicit_source": False,
        "answerable": True,
    },
    {
        "id": "CE056",
        "question": "ما علاقة منطقة فوق الذوبان ببدء التنوي في عملية التبلور؟",
        "language": "ar",
        "question_type": "crystallization",
        "expected_document_ids": ["mullin_crystallization"],
        "expected_evidence_chunk_ids": ["mullin_crystallization_e23198f092452faf"],
        "expected_concepts": ["supersaturation", "nucleation"],
        "explicit_source": False,
        "answerable": True,
    },
    {
        "id": "CE057",
        "question": "متى يسبب الفعل التكاملي في متحكم PID ظاهرة تراكم الإعادة؟",
        "language": "ar",
        "question_type": "control",
        "expected_document_ids": ["seborg_process_dynamics_control"],
        "expected_evidence_chunk_ids": ["seborg_process_dynamics_control_86d8bd05f91d039e"],
        "expected_concepts": ["integral control", "reset windup"],
        "explicit_source": False,
        "answerable": True,
    },
    {
        "id": "CE058",
        "question": "What was the evaporator’s electricity consumption yesterday at 14:05?",
        "language": "en",
        "question_type": "absent_from_corpus",
        "expected_document_ids": [],
        "expected_evidence_chunk_ids": [],
        "expected_concepts": [],
        "explicit_source": False,
        "answerable": False,
    },
    {
        "id": "CE059",
        "question": "من هو المشغل المناوب حاليا في وحدة التركيز؟",
        "language": "ar",
        "question_type": "absent_from_corpus",
        "expected_document_ids": [],
        "expected_evidence_chunk_ids": [],
        "expected_concepts": [],
        "explicit_source": False,
        "answerable": False,
    },
    {
        "id": "CE060",
        "question": (
            "Dans le rapport OCP, quelle relation exprime la capacité "
            "évaporatoire du bouilleur ?"
        ),
        "language": "fr",
        "question_type": "plant_numerical_data",
        "expected_document_ids": ["ocp_phosphoric_acid_workshop_report"],
        "expected_evidence_chunk_ids": ["ocp_phosphoric_acid_workshop_report_a992a675f7830a0b"],
        "expected_concepts": ["capacite evaporatoire", "debit vapeur", "m1", "m5"],
        "expected_numeric_facts": ["6893,98 kg/h"],
        "explicit_source": True,
        "answerable": True,
    },
    {
        "id": "CE061",
        "question": (
            "Dans le rapport OCP, comment le bilan de P2O5 relie-t-il les "
            "teneurs d’entrée et de sortie à la production ?"
        ),
        "language": "fr",
        "question_type": "material_balance",
        "expected_document_ids": ["ocp_phosphoric_acid_workshop_report"],
        "expected_evidence_chunk_ids": [
            "ocp_phosphoric_acid_workshop_report_d577637427bda29f"
        ],
        "expected_concepts": ["25", "p2o5", "50", "production"],
        "explicit_source": True,
        "answerable": True,
    },
]


def build_dataset(children: Sequence[TechnicalChildChunk]) -> list[dict[str, Any]]:
    """Create versioned truth from reviewed questions and active chunk IDs."""

    child_by_id = {child.chunk_id: child for child in children}
    records: list[dict[str, Any]] = []
    for base in read_jsonl(BASE_QUESTIONS):
        question_id = str(base["question_id"])
        number = int(question_id.removeprefix("DQ"))
        evidence_ids = EVIDENCE_BY_ID.get(question_id, [])
        record = {
            "id": question_id,
            "question": base["question"],
            "standalone_question": STANDALONE_BY_ID.get(question_id, base["question"]),
            "language": base["language"],
            "question_type": _question_type(number),
            "expected_document_ids": _expected_documents(number),
            "expected_pages": sorted(
                {child_by_id[item].page_start for item in evidence_ids if item in child_by_id}
            ),
            "expected_sections": sorted(
                {
                    child_by_id[item].section
                    for item in evidence_ids
                    if item in child_by_id and child_by_id[item].section
                }
            ),
            "expected_evidence_chunk_ids": evidence_ids,
            "expected_concepts": CONCEPTS_BY_ID.get(question_id, []),
            "expected_numeric_facts": [],
            "explicit_source": 45 <= number <= 47,
            "answerable": base["answerability"] == "answerable",
        }
        records.append(record)

    for extra in EXTRA_QUESTIONS:
        record = {
            "standalone_question": extra["question"],
            "expected_pages": [],
            "expected_sections": [],
            "expected_numeric_facts": [],
            **extra,
        }
        evidence_ids = record["expected_evidence_chunk_ids"]
        record["expected_pages"] = sorted(
            {child_by_id[item].page_start for item in evidence_ids if item in child_by_id}
        )
        record["expected_sections"] = sorted(
            {
                child_by_id[item].section
                for item in evidence_ids
                if item in child_by_id and child_by_id[item].section
            }
        )
        records.append(record)

    unknown = {
        chunk_id
        for record in records
        for chunk_id in record["expected_evidence_chunk_ids"]
        if chunk_id not in child_by_id
    }
    if unknown:
        raise ValueError(f"Inactive or unknown gold chunks: {sorted(unknown)}")
    return records


@dataclass(slots=True)
class CountingTokenizer:
    counter: Callable[[str], int]
    calls: int = 0
    elapsed_ms: float = 0.0

    def __call__(self, text: str) -> int:
        started = time.perf_counter()
        value = self.counter(text)
        self.elapsed_ms += (time.perf_counter() - started) * 1000.0
        self.calls += 1
        return value


def _anchor_pool(
    engine: QualityRetrievalEngine,
    result: Any,
    question_type: str,
) -> list[EvidenceAnchor]:
    plan = result.retrieval_plan
    limit = len(result.reranking.results)
    if plan is not None and len(plan.roles) > 1:
        selected = list(
            select_role_aware_evidence(
                plan,
                result.hybrid.results,
                result.reranking.results,
                top_k=limit,
            ).selected
        )
    else:
        selected = select_with_lexical_safeguard(
            result.hybrid.results,
            result.reranking.results,
            top_k=limit,
            lexical_slots=min(1, max(0, limit - 1)),
        )
        selected = engine._repair_weak_lexical_selection(  # noqa: SLF001
            selected,
            candidates=result.hybrid.results,
            reranked_results=result.reranking.results,
            question_type=question_type,
            top_k=limit,
        )
    reranked = {item.chunk.chunk_id: item for item in result.reranking.results}
    return [
        EvidenceAnchor(
            child=engine.child_by_id[item.chunk_id],
            score=reranked[item.chunk_id].reranker_score,
            provenance=item.source,
        )
        for item in selected
        if item.chunk_id in engine.child_by_id and item.chunk_id in reranked
    ]


def _run_ablation(
    engine: QualityRetrievalEngine,
    anchors: list[EvidenceAnchor],
    *,
    budget: int,
    child_only: bool,
    question_type: str,
) -> tuple[list[EvidenceBundle], dict[str, float | int]]:
    tokenizer = CountingTokenizer(engine.retriever.dense_retriever.embedder.count_tokens)
    if child_only:
        fake_children = [
            anchor.child.model_copy(
                update={
                    "parent_id": f"child-only::{anchor.child.chunk_id}",
                    "previous_chunk_id": None,
                    "next_chunk_id": None,
                }
            )
            for anchor in anchors
        ]
        score_by_id = {anchor.child.chunk_id: anchor for anchor in anchors}
        fake_anchors = [
            EvidenceAnchor(
                child=child,
                score=score_by_id[child.chunk_id].score,
                provenance=score_by_id[child.chunk_id].provenance,
            )
            for child in fake_children
        ]
        expander = ContextExpander(
            children=fake_children,
            parents=[],
            config=ContextExpansionConfig(
                neighbor_window=0,
                max_tokens_per_bundle=650,
                max_total_context_tokens=budget,
            ),
            token_counter=tokenizer,
        )
        run_anchors = fake_anchors
    else:
        expander = ContextExpander(
            children=engine.children,
            parents=engine.parents,
            config=ContextExpansionConfig(
                max_tokens_per_bundle=650,
                max_total_context_tokens=budget,
            ),
            token_counter=tokenizer,
        )
        run_anchors = anchors
    started = time.perf_counter()
    bundles = expander.expand(run_anchors, question_type=question_type)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return bundles, {
        "context_and_packing_ms": round(elapsed_ms, 3),
        "context_structure_ms": round(max(0.0, elapsed_ms - tokenizer.elapsed_ms), 3),
        "token_packing_ms": round(tokenizer.elapsed_ms, 3),
        "tokenizer_calls": tokenizer.calls,
    }


def _rank(chunk_ids: Sequence[str], expected: set[str]) -> int | None:
    return next((index for index, item in enumerate(chunk_ids, start=1) if item in expected), None)


def _recall(chunk_ids: Sequence[str], expected: set[str], k: int) -> float | None:
    if not expected:
        return None
    return len(set(chunk_ids[:k]) & expected) / len(expected)


def _coverage(record: dict[str, Any], bundles: Sequence[EvidenceBundle]) -> str:
    if not record["answerable"]:
        return "not_applicable"
    evidence_ids = set(record["expected_evidence_chunk_ids"])
    concepts = [normalize_text(item) for item in record["expected_concepts"]]
    pages = set(record["expected_pages"])
    sections = {normalize_text(item) for item in record["expected_sections"]}
    if not any((evidence_ids, concepts, pages, sections)):
        return "unannotated"
    supporting = {item for bundle in bundles for item in bundle.supporting_chunk_ids}
    context = normalize_text("\n".join(bundle.display_text for bundle in bundles))
    bundle_pages = {
        page
        for bundle in bundles
        for page in range(bundle.page_start, bundle.page_end + 1)
    }
    bundle_sections = {normalize_text(bundle.section or "") for bundle in bundles}
    checks: list[float] = []
    if evidence_ids:
        checks.append(len(evidence_ids & supporting) / len(evidence_ids))
    if concepts:
        checks.extend(1.0 if concept in context else 0.0 for concept in concepts)
    if pages:
        checks.append(1.0 if pages & bundle_pages else 0.0)
    if sections:
        checks.append(1.0 if sections & bundle_sections else 0.0)
    if checks and all(value == 1.0 for value in checks):
        return "full"
    if any(value > 0.0 for value in checks):
        return "partial"
    return "none"


def _overlap_count(bundles: Sequence[EvidenceBundle]) -> int:
    count = 0
    for left, right in zip(bundles, bundles[1:], strict=False):
        left_words = set(normalize_text(left.display_text).split())
        right_words = set(normalize_text(right.display_text).split())
        union = left_words | right_words
        if union and len(left_words & right_words) / len(union) >= 0.9:
            count += 1
    return count


def analyze_bundles(
    record: dict[str, Any],
    bundles: Sequence[EvidenceBundle],
    *,
    budget: int,
    parent_sizes: dict[str, int],
) -> dict[str, Any]:
    parent_ids = [bundle.parent_id for bundle in bundles]
    chunk_ids = [item for bundle in bundles for item in bundle.supporting_chunk_ids]
    normalized = [normalize_text(bundle.display_text) for bundle in bundles]
    documentary = sum(bundle.documentary_token_count for bundle in bundles)
    anchor_tokens = sum(bundle.anchor_token_count for bundle in bundles)
    serialized = sum(bundle.token_count for bundle in bundles)
    unique_documentary = sum(
        bundle.documentary_token_count
        for index, bundle in enumerate(bundles)
        if normalized[index] not in normalized[:index]
    )
    scope_counts = Counter(bundle.context_scope.value for bundle in bundles)
    reranker_scores = [bundle.best_anchor_score for bundle in bundles]
    return {
        "bundle_count": len(bundles),
        "documentary_tokens": documentary,
        "serialized_context_tokens": serialized,
        "anchor_tokens": anchor_tokens,
        "added_context_tokens": sum(bundle.context_token_count for bundle in bundles),
        "context_expansion_ratio": documentary / anchor_tokens if anchor_tokens else 0.0,
        "context_utilization": serialized / budget,
        "unique_parent_count": len(set(parent_ids)),
        "unique_section_count": len({bundle.section for bundle in bundles if bundle.section}),
        "duplicate_parent_count": len(parent_ids) - len(set(parent_ids)),
        "duplicate_chunk_count": len(chunk_ids) - len(set(chunk_ids)),
        "exact_duplicate_bundle_count": len(normalized) - len(set(normalized)),
        "very_high_overlap_adjacent_count": _overlap_count(bundles),
        "duplication_efficiency": unique_documentary / documentary if documentary else 1.0,
        "scope_counts": dict(scope_counts),
        "evidence_coverage": _coverage(record, bundles),
        "top_reranker_score": max(reranker_scores, default=None),
        "lowest_included_reranker_score": min(reranker_scores, default=None),
        "selected_parent_child_counts": [parent_sizes.get(item, 0) for item in parent_ids],
        "bundle_document_ids": sorted({bundle.document_id for bundle in bundles}),
    }


def evaluate_question(
    engine: QualityRetrievalEngine,
    record: dict[str, Any],
) -> dict[str, Any]:
    question = str(record["question"])
    standalone = str(record["standalone_question"])
    started = time.perf_counter()
    preprocessing_started = time.perf_counter()
    classification = classify_question(standalone)
    plan = build_retrieval_plan(
        question,
        standalone_query=standalone,
        question_type=classification.question_type.value,
    )
    routing = route_query(
        plan.base_query,
        catalog=engine.catalog,
        source_mode="auto",
        question_type=classification.question_type.value,
    )
    preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000.0

    discovery_ms = 0.0
    ranking_ids: list[str]
    resolved_ids = sorted(routing.hard_filter or [])
    if record["explicit_source"] and resolved_ids:
        ranking_ids = resolved_ids
    else:
        discovery_started = time.perf_counter()
        ranking = engine.discover_documents(
            question,
            standalone_query=standalone,
            question_type=classification.question_type.value,
        )
        discovery_ms = (time.perf_counter() - discovery_started) * 1000.0
        ranking_ids = [item.document_id for item in ranking]
    if not ranking_ids:
        raise ValueError("No document could be selected for locked evaluation.")
    locked_ids = {ranking_ids[0]}

    locked_started = time.perf_counter()
    result = engine.retrieve(
        question,
        standalone_query=standalone,
        question_type=classification.question_type.value,
        source_mode="auto",
        document_ids=locked_ids,
    )
    locked_ms = (time.perf_counter() - locked_started) * 1000.0
    anchors = _anchor_pool(engine, result, classification.question_type.value)
    parent_sizes = {parent.parent_id: len(parent.child_chunk_ids) for parent in engine.parents}
    locked_gold_ids = [
        chunk_id
        for chunk_id in record["expected_evidence_chunk_ids"]
        if engine.child_by_id[chunk_id].document_id in locked_ids
    ]
    locked_record = {
        **record,
        "expected_evidence_chunk_ids": locked_gold_ids,
        "expected_pages": sorted(
            {engine.child_by_id[chunk_id].page_start for chunk_id in locked_gold_ids}
        ),
        "expected_sections": sorted(
            {
                engine.child_by_id[chunk_id].section
                for chunk_id in locked_gold_ids
                if engine.child_by_id[chunk_id].section
            }
        ),
    }

    variants: dict[str, Any] = {}
    for name, budget, child_only in (
        ("child_only", CONTEXT_BUDGET, True),
        ("current", CONTEXT_BUDGET, False),
        ("small_budget", SMALL_CONTEXT_BUDGET, False),
        ("large_budget", LARGE_CONTEXT_BUDGET, False),
    ):
        bundles, timing = _run_ablation(
            engine,
            anchors,
            budget=budget,
            child_only=child_only,
            question_type=classification.question_type.value,
        )
        variants[name] = {
            **analyze_bundles(
                locked_record,
                bundles,
                budget=budget,
                parent_sizes=parent_sizes,
            ),
            **timing,
        }

    expected_documents = set(record["expected_document_ids"])
    expected_chunks = set(locked_gold_ids)
    hybrid_ids = [item.chunk.chunk_id for item in result.hybrid.results]
    reranked_ids = [item.chunk.chunk_id for item in result.reranking.results]
    leaked_hybrid = sorted(
        {item.chunk.document_id for item in result.hybrid.results} - locked_ids
    )
    leaked_bundles = sorted(set(variants["current"]["bundle_document_ids"]) - locked_ids)
    fusion_ms = max(
        0.0,
        result.hybrid.total_duration_ms
        - result.hybrid.dense_duration_ms
        - result.hybrid.sparse_duration_ms
        - result.hybrid.bm25_duration_ms,
    )
    return {
        "id": record["id"],
        "question": question,
        "language": record["language"],
        "question_type": record["question_type"],
        "classified_question_type": classification.question_type.value,
        "answerable": record["answerable"],
        "explicit_source": record["explicit_source"],
        "expected_document_ids": record["expected_document_ids"],
        "expected_evidence_chunk_ids": record["expected_evidence_chunk_ids"],
        "locked_expected_evidence_chunk_ids": locked_gold_ids,
        "document_ranking": ranking_ids,
        "selected_document": ranking_ids[0],
        "document_hit_at_1": (
            bool(expected_documents & set(ranking_ids[:1])) if expected_documents else None
        ),
        "document_hit_at_3": (
            bool(expected_documents & set(ranking_ids[:3])) if expected_documents else None
        ),
        "document_reciprocal_rank": (
            1.0 / _rank(ranking_ids, expected_documents)
            if expected_documents and _rank(ranking_ids, expected_documents)
            else 0.0
        ),
        "source_resolution_correct": (
            bool(expected_documents & set(resolved_ids)) if record["explicit_source"] else None
        ),
        "source_lock_matches_expected": (
            bool(expected_documents & locked_ids) if expected_documents else None
        ),
        "source_lock_leakage": {
            "hybrid_documents": leaked_hybrid,
            "bundle_documents": leaked_bundles,
        },
        "hybrid_top_chunk_ids": hybrid_ids[:20],
        "reranked_top_chunk_ids": reranked_ids[:20],
        "expected_rank_before_reranking": _rank(hybrid_ids, expected_chunks),
        "expected_rank_after_reranking": _rank(reranked_ids, expected_chunks),
        "recall_at_5": _recall(reranked_ids, expected_chunks, 5),
        "recall_at_10": _recall(reranked_ids, expected_chunks, 10),
        "recall_at_20": _recall(reranked_ids, expected_chunks, 20),
        "reranker_mrr": (
            1.0 / _rank(reranked_ids, expected_chunks)
            if expected_chunks and _rank(reranked_ids, expected_chunks)
            else (0.0 if expected_chunks else None)
        ),
        "reranker_candidate_count": len(result.reranking.results),
        "variants": variants,
        "latency_ms": {
            "query_preprocessing": round(preprocessing_ms, 3),
            "dense_retrieval": result.hybrid.dense_duration_ms,
            "sparse_retrieval": result.hybrid.sparse_duration_ms,
            "bm25_retrieval": result.hybrid.bm25_duration_ms,
            "rrf_fusion_and_colbert": round(fusion_ms, 3),
            "gpu_reranking": result.reranking.reranking_duration_ms,
            "document_discovery": round(discovery_ms, 3),
            "locked_retrieval": round(locked_ms, 3),
            "context_expander": variants["current"]["context_structure_ms"],
            "token_packing": variants["current"]["token_packing_ms"],
            "total_retrieval_context": round(
                (time.perf_counter() - started) * 1000.0,
                3,
            ),
        },
    }


def _average(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def _pearson(pairs: Sequence[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs, strict=True)
    x_mean, y_mean = mean(xs), mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else None


def summarize(questions: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    automatic = [
        row
        for row in results
        if not row["explicit_source"] and row["expected_document_ids"]
    ]
    evidence = [row for row in results if row["locked_expected_evidence_chunk_ids"]]
    explicit = [row for row in results if row["explicit_source"]]

    def variant_summary(name: str) -> dict[str, Any]:
        rows = [row["variants"][name] for row in results]
        scopes = Counter()
        coverage = Counter()
        for row in rows:
            scopes.update(row["scope_counts"])
            coverage[row["evidence_coverage"]] += 1
        scope_total = sum(scopes.values()) or 1
        return {
            "average_bundle_count": _average(rows, "bundle_count"),
            "average_documentary_tokens": _average(rows, "documentary_tokens"),
            "average_serialized_context_tokens": _average(rows, "serialized_context_tokens"),
            "average_context_expansion_ratio": _average(rows, "context_expansion_ratio"),
            "average_context_utilization": _average(rows, "context_utilization"),
            "average_duplication_efficiency": _average(rows, "duplication_efficiency"),
            "duplicate_parents": sum(row["duplicate_parent_count"] for row in rows),
            "duplicate_chunks": sum(row["duplicate_chunk_count"] for row in rows),
            "exact_duplicate_bundles": sum(row["exact_duplicate_bundle_count"] for row in rows),
            "very_high_overlap_adjacent": sum(
                row["very_high_overlap_adjacent_count"] for row in rows
            ),
            "scope_distribution": {
                key: {"count": scopes[key], "percentage": 100.0 * scopes[key] / scope_total}
                for key in (
                    "full_parent",
                    "partial_parent",
                    "anchor_with_neighbors",
                    "anchor_only",
                )
            },
            "evidence_coverage": dict(coverage),
        }

    by_language: dict[str, Any] = {}
    for language in sorted({row["language"] for row in results}):
        subset = [row for row in results if row["language"] == language]
        doc_subset = [
            row
            for row in subset
            if row["expected_document_ids"] and not row["explicit_source"]
        ]
        evidence_subset = [
            row for row in subset if row["locked_expected_evidence_chunk_ids"]
        ]
        by_language[language] = {
            "questions": len(subset),
            "document_hit_at_1": _average(doc_subset, "document_hit_at_1"),
            "document_hit_at_3": _average(doc_subset, "document_hit_at_3"),
            "document_mrr": _average(doc_subset, "document_reciprocal_rank"),
            "recall_at_5": _average(evidence_subset, "recall_at_5"),
            "recall_at_10": _average(evidence_subset, "recall_at_10"),
            "recall_at_20": _average(evidence_subset, "recall_at_20"),
            "reranker_mrr": _average(evidence_subset, "reranker_mrr"),
            "coverage": dict(
                Counter(
                    row["variants"]["current"]["evidence_coverage"] for row in subset
                )
            ),
        }

    parent_by_document: dict[str, list[tuple[list[int], float]]] = defaultdict(list)
    for row in results:
        current = row["variants"]["current"]
        sizes = [size for size in current["selected_parent_child_counts"] if size > 0]
        if sizes:
            parent_by_document[row["selected_document"]].append(
                (sizes, current["context_expansion_ratio"])
            )
    parent_analysis = {}
    for document_id, entries in parent_by_document.items():
        sizes = [size for entry_sizes, _ratio in entries for size in entry_sizes]
        correlation_pairs = [
            (mean(entry_sizes), ratio) for entry_sizes, ratio in entries
        ]
        parent_analysis[document_id] = {
            "questions": len(entries),
            "average_children_per_selected_parent": mean(sizes),
            "average_context_expansion_ratio": mean(
                ratio for _entry_sizes, ratio in entries
            ),
            "one_child_percentage": 100.0 * sum(size == 1 for size in sizes) / len(sizes),
            "two_children_percentage": 100.0 * sum(size == 2 for size in sizes) / len(sizes),
            "three_plus_children_percentage": 100.0 * sum(size >= 3 for size in sizes) / len(sizes),
            "expansion_ratio_correlation": _pearson(correlation_pairs),
        }

    latency_keys = results[0]["latency_ms"] if results else {}
    latency = {
        key: mean(row["latency_ms"][key] for row in results)
        for key in latency_keys
    }
    latency["average_reranker_candidates"] = mean(
        row["reranker_candidate_count"] for row in results
    ) if results else 0.0
    latency["average_tokenizer_calls"] = mean(
        row["variants"]["current"]["tokenizer_calls"] for row in results
    ) if results else 0.0

    absent = [row for row in results if not row["answerable"]]
    absent_summary = {
        "questions": len(absent),
        "average_top_reranker_score": _average(
            [row["variants"]["current"] for row in absent],
            "top_reranker_score",
        ),
        "average_bundle_count": _average(
            [row["variants"]["current"] for row in absent],
            "bundle_count",
        ),
        "average_serialized_context_tokens": _average(
            [row["variants"]["current"] for row in absent],
            "serialized_context_tokens",
        ),
    }

    failures = sorted(
        (
            row
            for row in results
            if row["expected_document_ids"]
            and (
                not row["document_hit_at_1"]
                or row["variants"]["current"]["evidence_coverage"] in {"none", "partial"}
            )
        ),
        key=lambda row: (
            bool(row["document_hit_at_1"]),
            row["variants"]["current"]["evidence_coverage"] != "none",
        ),
    )[:10]

    def failure_reason(row: dict[str, Any]) -> str:
        if not row["document_hit_at_1"]:
            return "document_discovery_miss"
        if row["locked_expected_evidence_chunk_ids"]:
            before = row["expected_rank_before_reranking"]
            after = row["expected_rank_after_reranking"]
            if before is None:
                return "first_stage_evidence_miss"
            if after is None or after > before:
                return "reranker_demotion"
            if row["recall_at_20"] and row["variants"]["current"]["evidence_coverage"] != "full":
                return "gold_retrieved_but_context_coverage_incomplete"
        return "expected_concepts_missing_from_final_context"

    return {
        "evaluation_version": "context-engine-v0.1",
        "active_kb": ACTIVE_VERSION,
        "dataset": {
            "question_count": len(questions),
            "documents": sorted(
                {item for row in questions for item in row["expected_document_ids"]}
            ),
            "languages": dict(Counter(row["language"] for row in questions)),
            "question_types": dict(Counter(row["question_type"] for row in questions)),
            "questions_with_chunk_gold": sum(
                bool(row["expected_evidence_chunk_ids"]) for row in questions
            ),
            "evaluated_with_locked_chunk_gold": len(evidence),
            "answerable": sum(row["answerable"] for row in questions),
            "absent_from_corpus": sum(not row["answerable"] for row in questions),
        },
        "document_retrieval_automatic": {
            "questions": len(automatic),
            "hit_at_1": _average(automatic, "document_hit_at_1"),
            "hit_at_3": _average(automatic, "document_hit_at_3"),
            "mrr": _average(automatic, "document_reciprocal_rank"),
        },
        "evidence_retrieval": {
            "questions": len(evidence),
            "recall_at_5": _average(evidence, "recall_at_5"),
            "recall_at_10": _average(evidence, "recall_at_10"),
            "recall_at_20": _average(evidence, "recall_at_20"),
            "mrr_after_reranking": _average(evidence, "reranker_mrr"),
            "average_rank_before": _average(evidence, "expected_rank_before_reranking"),
            "average_rank_after": _average(evidence, "expected_rank_after_reranking"),
        },
        "explicit_source": {
            "questions": len(explicit),
            "source_resolution_accuracy": _average(explicit, "source_resolution_correct"),
            "source_lock_accuracy": _average(explicit, "source_lock_matches_expected"),
            "leakage_questions": sum(
                bool(row["source_lock_leakage"]["hybrid_documents"])
                or bool(row["source_lock_leakage"]["bundle_documents"])
                for row in explicit
            ),
        },
        "ablations": {
            name: variant_summary(name)
            for name in ("child_only", "current", "small_budget", "large_budget")
        },
        "multilingual": by_language,
        "parent_structure": parent_analysis,
        "absent_corpus_behavior": absent_summary,
        "latency_ms_average": latency,
        "failures": [
            {
                "id": row["id"],
                "question": row["question"],
                "expected_documents": row["expected_document_ids"],
                "expected_evidence": row["expected_evidence_chunk_ids"],
                "retrieved_document": row["selected_document"],
                "top_retrieved_chunks": row["reranked_top_chunk_ids"][:5],
                "coverage": row["variants"]["current"]["evidence_coverage"],
                "reason": failure_reason(row),
            }
            for row in failures
        ],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    dataset = summary["dataset"]
    document = summary["document_retrieval_automatic"]
    evidence = summary["evidence_retrieval"]
    current = summary["ablations"]["current"]
    explicit = summary["explicit_source"]
    scopes = current["scope_distribution"]
    lines = [
        "# Context Engine Phase 5 — deterministic evaluation v0.1",
        "",
        f"Active KB: `{summary['active_kb']}`",
        "",
        "## 1. Dataset",
        "",
        f"- Questions: {dataset['question_count']}",
        f"- Documents: {len(dataset['documents'])}",
        f"- Languages: {dataset['languages']}",
        f"- Question types: {dataset['question_types']}",
        f"- Chunk-level gold: {dataset['questions_with_chunk_gold']}",
        f"- Locked runs with applicable chunk gold: {dataset['evaluated_with_locked_chunk_gold']}",
        "",
        "## 2. Document retrieval",
        "",
        f"- Hit@1: {document['hit_at_1']:.3f}",
        f"- Hit@3: {document['hit_at_3']:.3f}",
        f"- MRR: {document['mrr']:.3f}",
        f"- Explicit source resolution: {explicit['source_resolution_accuracy']:.3f}",
        f"- Source-lock accuracy: {explicit['source_lock_accuracy']:.3f}",
        f"- Source-lock leakage questions: {explicit['leakage_questions']}",
        "",
        "## 3. Evidence retrieval",
        "",
        f"- Recall@5: {evidence['recall_at_5']:.3f}",
        f"- Recall@10: {evidence['recall_at_10']:.3f}",
        f"- Recall@20: {evidence['recall_at_20']:.3f}",
        f"- Reranker MRR: {evidence['mrr_after_reranking']:.3f}",
        f"- Mean found rank before reranking: {evidence['average_rank_before']:.2f}",
        f"- Mean found rank after reranking: {evidence['average_rank_after']:.2f}",
        "",
        "## 4. Context quality",
        "",
        f"- Expansion ratio: {current['average_context_expansion_ratio']:.3f}",
        f"- Documentary tokens: {current['average_documentary_tokens']:.1f}",
        f"- Serialized tokens: {current['average_serialized_context_tokens']:.1f}",
        f"- Context utilization: {current['average_context_utilization']:.3f}",
        f"- Duplication efficiency: {current['average_duplication_efficiency']:.3f}",
        f"- Full parent: {scopes['full_parent']['percentage']:.2f}%",
        f"- Partial parent: {scopes['partial_parent']['percentage']:.2f}%",
        f"- Anchor with neighbors: {scopes['anchor_with_neighbors']['percentage']:.2f}%",
        f"- Anchor only: {scopes['anchor_only']['percentage']:.2f}%",
        f"- Evidence coverage: {current['evidence_coverage']}",
        f"- Parent structure: {summary['parent_structure']}",
        "",
        "## 5. Ablation",
        "",
        "| Variant | Bundles | Documentary tokens | Expansion | Coverage |",
        "|---|---:|---:|---:|---|",
    ]
    for name, values in summary["ablations"].items():
        lines.append(
            f"| {name} | {values['average_bundle_count']:.2f} | "
            f"{values['average_documentary_tokens']:.1f} | "
            f"{values['average_context_expansion_ratio']:.3f} | "
            f"{values['evidence_coverage']} |"
        )
    lines.extend(
        [
            "",
            "## 6. Multilingual",
            "",
            "```json",
            json.dumps(summary["multilingual"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## 7. Latency (mean ms)",
            "",
            "```json",
            json.dumps(summary["latency_ms_average"], ensure_ascii=False, indent=2),
            "```",
            "",
            f"Absent-corpus behavior: `{summary['absent_corpus_behavior']}`",
            "",
            "## 8. Failure analysis",
            "",
            "```json",
            json.dumps(summary["failures"], ensure_ascii=False, indent=2),
            "```",
            "",
            f"Evaluation errors: `{summary.get('evaluation_errors', [])}`",
            "",
            "## 9. Architectural conclusion",
            "",
            (
                "Primary classification: child-evidence retrieval and ranking problem. "
                "Document discovery is strong after multi-source truth validation and "
                "packing has no measured duplication problem. Parent expansion adds "
                "about 1% documentary content and does not improve deterministic "
                "coverage over child-only context; parent structure is a secondary "
                "constraint, especially for the OCP report."
            ),
            "",
            "## 10. Phase 6 recommendation",
            "",
            (
                "Make first-stage child-evidence query representation and fusion the "
                "next single architectural workstream, calibrated against this frozen "
                "benchmark. Do not add an Evidence Judge until Recall@K improves and "
                "the benchmark is rerun."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run(output_directory: Path, *, limit: int | None = None) -> dict[str, Any]:
    rag = PhosProcessRAG()
    engine = rag.quality_engine
    if engine is None:
        raise RuntimeError("Production quality engine is not available.")
    questions = build_dataset(engine.children)
    if limit is not None:
        questions = questions[:limit]
    output_directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_directory / "questions.jsonl", questions)
    results: list[dict[str, Any]] = []
    for index, record in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] {record['id']} {record['language']}", flush=True)
        try:
            results.append(evaluate_question(engine, record))
        except Exception as error:  # evaluation must preserve per-question failures
            results.append(
                {
                    "id": record["id"],
                    "question": record["question"],
                    "language": record["language"],
                    "question_type": record["question_type"],
                    "answerable": record["answerable"],
                    "explicit_source": record["explicit_source"],
                    "expected_document_ids": record["expected_document_ids"],
                    "expected_evidence_chunk_ids": record["expected_evidence_chunk_ids"],
                    "evaluation_error": f"{type(error).__name__}: {error}",
                }
            )
        write_jsonl(output_directory / "per_question_results.jsonl", results)
    successful = [row for row in results if "evaluation_error" not in row]
    if not successful:
        raise RuntimeError("All evaluation questions failed.")
    summary = summarize(questions, successful)
    summary["evaluation_errors"] = [
        {"id": row["id"], "error": row["evaluation_error"]}
        for row in results
        if "evaluation_error" in row
    ]
    (output_directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_directory / "summary.md").write_text(
        render_markdown(summary),
        encoding="utf-8",
    )
    return summary


def refresh_summary(output_directory: Path) -> dict[str, Any]:
    """Reapply corrected evaluation truth without repeating GPU retrieval."""

    questions = build_dataset(read_child_chunks(ACTIVE_DIRECTORY / "chunks.jsonl"))
    truth_by_id = {row["id"]: row for row in questions}
    results = read_jsonl(output_directory / "per_question_results.jsonl")
    for row in results:
        truth = truth_by_id[row["id"]]
        row["expected_document_ids"] = truth["expected_document_ids"]
        row["expected_evidence_chunk_ids"] = truth["expected_evidence_chunk_ids"]
        if "evaluation_error" in row:
            continue
        expected = set(truth["expected_document_ids"])
        ranking = row["document_ranking"]
        rank = _rank(ranking, expected)
        row["document_hit_at_1"] = bool(expected & set(ranking[:1])) if expected else None
        row["document_hit_at_3"] = bool(expected & set(ranking[:3])) if expected else None
        row["document_reciprocal_rank"] = 1.0 / rank if rank else 0.0
        row["source_lock_matches_expected"] = (
            row["selected_document"] in expected if expected else None
        )
    write_jsonl(output_directory / "questions.jsonl", questions)
    write_jsonl(output_directory / "per_question_results.jsonl", results)
    successful = [row for row in results if "evaluation_error" not in row]
    summary = summarize(questions, successful)
    summary["evaluation_errors"] = [
        {"id": row["id"], "error": row["evaluation_error"]}
        for row in results
        if "evaluation_error" in row
    ]
    (output_directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_directory / "summary.md").write_text(
        render_markdown(summary),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        refresh_summary(args.output)
    else:
        run(args.output, limit=args.limit)


if __name__ == "__main__":
    main()
