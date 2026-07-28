"""Deterministic multi-domain routing with soft preferences only."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from phosprocess.knowledge_base.domains import KnowledgeDomain
from phosprocess.knowledge_base.schemas import KnowledgeBaseCatalog

_DOMAIN_TERMS: dict[KnowledgeDomain, tuple[str, ...]] = {
    KnowledgeDomain.PHOSPHORIC_ACID_PROCESS: (
        "phosphoric acid",
        "acide phosphorique",
        "phosphate",
        "gypsum",
        "gypse",
        "p2o5",
        "wet process",
        "procédé humide",
        "حمض الفوسفوريك",
        "الحمض الفوسفوري",
        "الحمض",
    ),
    KnowledgeDomain.PLANT_SPECIFIC: (
        "atelier",
        "installation",
        "ocp",
        "plant",
        "sur site",
    ),
    KnowledgeDomain.THERMODYNAMICS: (
        "thermodynamic",
        "thermodynamique",
        "enthalpy",
        "enthalpie",
        "entropy",
        "entropie",
        "phase equilibrium",
        "équilibre de phase",
        "vapor pressure",
        "pression de vapeur",
        "boiling temperature",
        "température d'ébullition",
    ),
    KnowledgeDomain.HEAT_TRANSFER: (
        "heat transfer",
        "transfert thermique",
        "heat exchanger",
        "échangeur thermique",
        "conduction",
        "convection",
        "boiling",
        "ébullition",
        "condensation",
        "energy balance",
        "heat balance",
        "enthalpy balance",
        "bilan énergétique",
        "bilan thermique",
        "bilan enthalpique",
        "المبادل الحراري",
        "التبادل الحراري",
        "غرفة التبخير",
        "فصل البخار",
    ),
    KnowledgeDomain.MASS_TRANSFER: (
        "mass transfer",
        "transfert de masse",
        "diffusion",
    ),
    KnowledgeDomain.FLUID_MECHANICS: (
        "fluid",
        "fluide",
        "flow",
        "écoulement",
        "pressure drop",
        "perte de charge",
        "pump",
        "pompe",
        "recirculation",
        "momentum",
        "مضخة",
        "تدفق",
    ),
    KnowledgeDomain.CRYSTALLIZATION: (
        "crystallization",
        "cristallisation",
        "supersaturation",
        "sursaturation",
        "nucleation",
        "nucléation",
        "crystal growth",
        "croissance cristalline",
        "precipitation",
        "فرط التشبع",
        "التنوي",
    ),
    KnowledgeDomain.PROCESS_CONTROL: (
        "process control",
        "régulation",
        "control loop",
        "boucle de contrôle",
        "pid",
        "feedback",
        "stability",
        "stabilité",
    ),
    KnowledgeDomain.MPC: ("mpc", "model predictive control"),
    KnowledgeDomain.INSTRUMENTATION: (
        "instrument",
        "sensor",
        "capteur",
        "transmitter",
        "actionneur",
    ),
    KnowledgeDomain.EQUIPMENT: (
        "equipment",
        "équipement",
        "pump",
        "pompe",
        "reactor",
        "réacteur",
        "filter",
        "filtre",
        "exchanger",
        "échangeur",
        "evaporator",
        "évaporateur",
        "المبخر",
        "غرفة التبخير",
        "حجرة التبخير",
        "جسم البخار",
    ),
    KnowledgeDomain.SAFETY: (
        "safety",
        "sécurité",
        "hazard",
        "danger",
        "corrosion",
    ),
    KnowledgeDomain.GENERAL_CHEMICAL_ENGINEERING: (
        "chemical engineering",
        "génie chimique",
        "unit operation",
        "opération unitaire",
        "mass balance",
        "material balance",
        "component balance",
        "species balance",
        "bilan massique",
        "bilan matière",
    ),
}

_DOMAIN_PRIMARY_DOCUMENTS: dict[KnowledgeDomain, str] = {
    KnowledgeDomain.PHOSPHORIC_ACID_PROCESS: (
        "becker_phosphates_and_phosphoric_acid"
    ),
    KnowledgeDomain.PLANT_SPECIFIC: "ocp_phosphoric_acid_workshop_report",
    KnowledgeDomain.THERMODYNAMICS: (
        "smith_van_ness_chemical_engineering_thermodynamics"
    ),
    KnowledgeDomain.HEAT_TRANSFER: (
        "incropera_fundamentals_heat_mass_transfer"
    ),
    KnowledgeDomain.FLUID_MECHANICS: "bird_transport_phenomena",
    KnowledgeDomain.MASS_TRANSFER: "bird_transport_phenomena",
    KnowledgeDomain.CRYSTALLIZATION: "mullin_crystallization",
    KnowledgeDomain.PROCESS_CONTROL: "seborg_process_dynamics_control",
    KnowledgeDomain.MPC: "seborg_process_dynamics_control",
    KnowledgeDomain.EQUIPMENT: "perrys_chemical_engineers_handbook",
    KnowledgeDomain.GENERAL_CHEMICAL_ENGINEERING: (
        "perrys_chemical_engineers_handbook"
    ),
}

SOURCE_MODE_DOCUMENTS = {
    "becker": "becker_phosphates_and_phosphoric_acid",
    "report": "ocp_phosphoric_acid_workshop_report",
    "thermodynamics": "smith_van_ness_chemical_engineering_thermodynamics",
    "heat_transfer": "incropera_fundamentals_heat_mass_transfer",
    "perry": "perrys_chemical_engineers_handbook",
    "crystallization": "mullin_crystallization",
    "control": "seborg_process_dynamics_control",
    "transport": "bird_transport_phenomena",
}
SUPPORTED_SOURCE_MODES = frozenset({"auto", *SOURCE_MODE_DOCUMENTS})


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", without_marks)


@dataclass(frozen=True, slots=True)
class DomainRoutingDecision:
    """Traceable soft-routing decision for one standalone query."""

    detected_domains: tuple[tuple[KnowledgeDomain, float], ...]
    confidence: float
    preferred_documents: tuple[str, ...]
    soft_boosts: dict[str, float]
    explanation: str
    hard_filter: frozenset[str] | None
    source_mode: str


def route_query(
    question: str,
    *,
    catalog: KnowledgeBaseCatalog,
    source_mode: str = "auto",
    maximum_source_boost: float = 0.06,
) -> DomainRoutingDecision:
    """Detect multiple domains and never hard-filter in automatic mode."""

    mode = source_mode.strip().casefold()

    if mode == "automatic":
        mode = "auto"

    if mode not in SUPPORTED_SOURCE_MODES:
        raise ValueError(
            "Mode source invalide : "
            + ", ".join(sorted(SUPPORTED_SOURCE_MODES))
        )

    if mode != "auto":
        document_id = SOURCE_MODE_DOCUMENTS[mode]
        return DomainRoutingDecision(
            detected_domains=(),
            confidence=1.0,
            preferred_documents=(document_id,),
            soft_boosts={},
            explanation=f"Explicit user filter: {mode}",
            hard_filter=frozenset({document_id}),
            source_mode=mode,
        )

    normalized = _normalize(question)
    raw_scores: dict[KnowledgeDomain, float] = {}

    for domain, terms in _DOMAIN_TERMS.items():
        matches = sum(_normalize(term) in normalized for term in terms)

        if matches:
            raw_scores[domain] = min(1.0, 0.45 + 0.2 * matches)

    if not raw_scores:
        raw_scores[KnowledgeDomain.GENERAL_CHEMICAL_ENGINEERING] = 0.4

    detected = tuple(
        sorted(raw_scores.items(), key=lambda item: (-item[1], item[0].value))
    )
    boosts: dict[str, float] = {}

    for document in catalog.documents:
        overlap = sum(raw_scores.get(domain, 0.0) for domain in document.domains)

        if overlap <= 0:
            continue

        priority_factor = 0.75 + 0.25 * document.priority / 100
        primary_bonus = sum(
            0.004 * raw_scores.get(domain, 0.0)
            for domain, primary_document in _DOMAIN_PRIMARY_DOCUMENTS.items()
            if primary_document == document.document_id
        )
        boosts[document.document_id] = round(
            min(
                maximum_source_boost,
                overlap * 0.025 * priority_factor + primary_bonus,
            ),
            6,
        )

    preferred = tuple(
        document_id
        for document_id, _boost in sorted(
            boosts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )
    return DomainRoutingDecision(
        detected_domains=detected,
        confidence=max(score for _domain, score in detected),
        preferred_documents=preferred,
        soft_boosts=boosts,
        explanation=(
            "Matched deterministic technical terms; all active documents "
            "remain searchable."
        ),
        hard_filter=None,
        source_mode="auto",
    )
