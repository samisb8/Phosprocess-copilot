"""Deterministic multi-domain and eight-document routing.

Automatic mode never filters the corpus.  It combines domain overlap with an
intent/entity source profile.  A hard document filter is permitted only when
the user explicitly requests a named source or selects a terminal source mode.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from phosprocess.knowledge_base.domains import KnowledgeDomain
from phosprocess.knowledge_base.schemas import KnowledgeBaseCatalog

BECKER_DOCUMENT = "becker_phosphates_and_phosphoric_acid"
THERMODYNAMICS_DOCUMENT = "smith_van_ness_chemical_engineering_thermodynamics"
HEAT_TRANSFER_DOCUMENT = "incropera_fundamentals_heat_mass_transfer"
REPORT_DOCUMENT = "ocp_phosphoric_acid_workshop_report"
PERRY_DOCUMENT = "perrys_chemical_engineers_handbook"
CRYSTALLIZATION_DOCUMENT = "mullin_crystallization"
CONTROL_DOCUMENT = "seborg_process_dynamics_control"
TRANSPORT_DOCUMENT = "bird_transport_phenomena"

SOURCE_MODE_DOCUMENTS = {
    "becker": BECKER_DOCUMENT,
    "report": REPORT_DOCUMENT,
    "thermodynamics": THERMODYNAMICS_DOCUMENT,
    "heat_transfer": HEAT_TRANSFER_DOCUMENT,
    "perry": PERRY_DOCUMENT,
    "crystallization": CRYSTALLIZATION_DOCUMENT,
    "control": CONTROL_DOCUMENT,
    "transport": TRANSPORT_DOCUMENT,
}
SUPPORTED_SOURCE_MODES = frozenset({"auto", *SOURCE_MODE_DOCUMENTS})

_SOURCE_ALIASES: dict[str, tuple[str, ...]] = {
    "becker": ("becker", "phosphates and phosphoric acid"),
    "report": (
        "rapport",
        "ocp report",
        "jfc4 report",
        "rapport ocp",
        "rapport atelier",
        "rapport de l atelier",
        "rapport jfc4",
        "memoire jfc4",
        "04 rapport",
    ),
    "thermodynamics": (
        "smith",
        "smith van ness",
        "smith-van ness",
        "van ness",
        "chemical engineering thermodynamics",
    ),
    "heat_transfer": ("incropera", "fundamentals of heat and mass transfer"),
    "perry": ("perry", "perry s", "chemical engineers handbook"),
    "crystallization": ("mullin", "crystallization 4e"),
    "control": ("seborg", "process dynamics and control"),
    "transport": ("bird", "transport phenomena"),
}

_EXPLICIT_SOURCE_REQUEST = re.compile(
    r"(?:\bselon\b|\bd apres\b|\bdans\s+(?:le\s+)?|\bcherche\s+(?:dans|sur)\b|"
    r"\bbase(?:e|es|ez)?\s+(?:toi|vous)?\s*sur\b|\buniquement\s+(?:dans|selon)\b|"
    r"\baccording to\b|\bbased on\b|\buse only\b|\bfrom\b|\bin\b|"
    r"\bاستنادا إلى\b|\bحسب\b)",
    re.I,
)

_SOURCE_SCOPE_RELEASE = re.compile(
    r"\b(?:toutes? les sources|tous les documents|cherche partout|"
    r"sans source impos[eé]e|mode automatique|source automatique|"
    r"all sources|all documents|search everywhere|automatic source|"
    r"no source restriction)\b",
    re.I,
)

_DOMAIN_TERMS: dict[KnowledgeDomain, tuple[str, ...]] = {
    KnowledgeDomain.PHOSPHORIC_ACID_PROCESS: (
        "phosphoric acid",
        "acide phosphorique",
        "phosphate",
        "gypsum",
        "gypse",
        "p2o5",
        "wet process",
        "voie humide",
        "procede humide",
        "jacobs",
        "حمض الفوسفوريك",
        "الحمض الفوسفوري",
        "الحمض",
    ),
    KnowledgeDomain.PLANT_SPECIFIC: (
        "atelier",
        "installation",
        "ocp",
        "jfc4",
        "echelon",
        "plant",
        "sur site",
        "design",
        "historique",
        "reel",
    ),
    KnowledgeDomain.THERMODYNAMICS: (
        "thermodynamic",
        "thermodynamique",
        "enthalpy",
        "enthalpie",
        "entropy",
        "entropie",
        "phase equilibrium",
        "equilibre de phase",
        "vapor pressure",
        "pression de vapeur",
        "boiling temperature",
        "temperature d ebullition",
        "saturation",
    ),
    KnowledgeDomain.HEAT_TRANSFER: (
        "heat transfer",
        "transfert thermique",
        "heat exchanger",
        "echangeur thermique",
        "conduction",
        "convection",
        "boiling",
        "ebullition",
        "condensation",
        "energy balance",
        "heat balance",
        "enthalpy balance",
        "bilan energetique",
        "bilan thermique",
        "lmtd",
        "coefficient global",
        "المبادل الحراري",
        "التبادل الحراري",
        "غرفة التبخير",
        "فصل البخار",
    ),
    KnowledgeDomain.MASS_TRANSFER: (
        "mass transfer",
        "transfert de masse",
        "diffusion",
        "sherwood",
        "schmidt",
        "interphase",
    ),
    KnowledgeDomain.FLUID_MECHANICS: (
        "fluid",
        "fluide",
        "flow",
        "ecoulement",
        "pressure drop",
        "perte de charge",
        "pump",
        "pompe",
        "recirculation",
        "momentum",
        "reynolds",
        "مضخة",
        "تدفق",
    ),
    KnowledgeDomain.CRYSTALLIZATION: (
        "crystallization",
        "cristallisation",
        "supersaturation",
        "sursaturation",
        "nucleation",
        "nucleation",
        "crystal growth",
        "croissance cristalline",
        "precipitation",
        "فرط التشبع",
        "التنوي",
    ),
    KnowledgeDomain.PROCESS_CONTROL: (
        "process control",
        "regulation",
        "control loop",
        "boucle de controle",
        "pid",
        "feedback",
        "stability",
        "stabilite",
        "dynamic model",
    ),
    KnowledgeDomain.MPC: ("mpc", "model predictive control"),
    KnowledgeDomain.INSTRUMENTATION: (
        "instrument",
        "sensor",
        "capteur",
        "transmitter",
        "actionneur",
        "debitmetre",
    ),
    KnowledgeDomain.EQUIPMENT: (
        "equipment",
        "equipement",
        "pump",
        "pompe",
        "reactor",
        "reacteur",
        "filter",
        "filtre",
        "exchanger",
        "echangeur",
        "evaporator",
        "evaporateur",
        "bouilleur",
        "condenser",
        "condenseur",
        "separator",
        "separateur",
        "المبخر",
        "غرفة التبخير",
        "حجرة التبخير",
        "جسم البخار",
    ),
    KnowledgeDomain.SAFETY: (
        "safety",
        "securite",
        "hazard",
        "danger",
        "corrosion",
        "risk",
        "risque",
    ),
    KnowledgeDomain.GENERAL_CHEMICAL_ENGINEERING: (
        "chemical engineering",
        "genie chimique",
        "unit operation",
        "operation unitaire",
        "mass balance",
        "material balance",
        "component balance",
        "species balance",
        "bilan massique",
        "bilan matiere",
    ),
}

_DOMAIN_PRIMARY_DOCUMENTS: dict[KnowledgeDomain, str] = {
    KnowledgeDomain.PHOSPHORIC_ACID_PROCESS: BECKER_DOCUMENT,
    KnowledgeDomain.PLANT_SPECIFIC: REPORT_DOCUMENT,
    KnowledgeDomain.THERMODYNAMICS: THERMODYNAMICS_DOCUMENT,
    KnowledgeDomain.HEAT_TRANSFER: HEAT_TRANSFER_DOCUMENT,
    KnowledgeDomain.FLUID_MECHANICS: TRANSPORT_DOCUMENT,
    KnowledgeDomain.MASS_TRANSFER: TRANSPORT_DOCUMENT,
    KnowledgeDomain.CRYSTALLIZATION: CRYSTALLIZATION_DOCUMENT,
    KnowledgeDomain.PROCESS_CONTROL: CONTROL_DOCUMENT,
    KnowledgeDomain.MPC: CONTROL_DOCUMENT,
    KnowledgeDomain.EQUIPMENT: PERRY_DOCUMENT,
    KnowledgeDomain.GENERAL_CHEMICAL_ENGINEERING: PERRY_DOCUMENT,
}


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return re.sub(r"[^a-z0-9%\u0600-\u06ff]+", " ", without_marks).strip()


def detect_explicit_source_mode(question: str) -> str | None:
    """Return a named source only when the wording is an explicit request.

    Merely mentioning ``OCP`` or ``JFC4`` describes plant context and must not
    silently hard-filter the report.  A citation/request marker plus a unique
    source alias is required.
    """

    normalized = _normalize(question)
    if not _EXPLICIT_SOURCE_REQUEST.search(normalized):
        return None
    matched = [
        mode
        for mode, aliases in _SOURCE_ALIASES.items()
        if any(_normalize(alias) in normalized for alias in aliases)
    ]
    return matched[0] if len(matched) == 1 else None


def requests_automatic_source_scope(question: str) -> bool:
    """Return whether the user explicitly releases a prior source lock."""

    return _SOURCE_SCOPE_RELEASE.search(question) is not None


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _infer_question_type(normalized: str) -> str:
    if _contains_any(normalized, ("balance", "bilan")):
        return "balance"
    if _contains_any(
        normalized,
        (
            "momentum diffusion",
            "momentum transport",
            "transport of momentum",
            "diffusion de quantite de mouvement",
            "transport de quantite de mouvement",
            "انتقال الزخم",
        ),
    ):
        return "momentum_diffusion"
    if _contains_any(normalized, ("step by step", "etape par etape", "trajet", "path")):
        return "process_flow"
    if _contains_any(normalized, ("compare", "difference", "versus")):
        return "comparison"
    if _contains_any(normalized, ("fouling", "encrassement", "bouchage", "panne")):
        return "troubleshooting"
    # Role/necessity/mechanism questions may start with a generic form such as
    # Arabic "ما هو دور ...".  Detect the requested relation before the
    # broader definition prefix so that equipment-domain routing remains
    # specific (for example, heat exchangers route to Incropera).
    if _contains_any(normalized, ("role", "rôle", "دور", "why", "pourquoi", "how does")):
        return "explanation"
    if re.search(
        r"(?:^| )(?:what is|c est quoi|cest quoi|c quoi|"
        r"qu est ce que|define|ما هو|ما هي)(?: |$)",
        normalized,
    ):
        return "definition"
    if _contains_any(normalized, ("relation", "vapor pressure", "pression de vapeur")):
        return "thermodynamic_relation"
    if _contains_any(normalized, ("pid", "mpc", "control", "regulation")):
        return "control_strategy"
    return "explanation"


def _profile_scores(normalized: str, question_type: str) -> dict[str, float]:
    """Return document suitability in [0, 1] for intent/entity context."""

    scores: dict[str, float] = {}

    def prefer(document_id: str, score: float) -> None:
        scores[document_id] = max(scores.get(document_id, 0.0), score)

    plant = _contains_any(
        normalized,
        ("ocp", "jfc4", "atelier", "echelon", "sur site", "design", "historique", "reel"),
    )
    phosphoric = _contains_any(
        normalized,
        ("phosphoric acid", "acide phosphorique", "p2o5", "wet process", "voie humide", "jacobs"),
    )
    pump = _contains_any(normalized, ("pump", "pompe", "مضخة"))
    heat = _contains_any(
        normalized,
        (
            "heat exchanger",
            "echangeur",
            "heat transfer",
            "transfert thermique",
            "lmtd",
            "condensation",
            "المبادل الحراري",
            "التبادل الحراري",
        ),
    )
    thermodynamics = _contains_any(
        normalized,
        (
            "enthalpy",
            "enthalpie",
            "vapor pressure",
            "pression de vapeur",
            "boiling",
            "ebullition",
            "thermodynamic",
            "thermodynamique",
        ),
    ) or re.search(r"\bsaturation\b", normalized) is not None
    mass_transfer = _contains_any(
        normalized,
        (
            "mass transfer",
            "transfert de masse",
            "diffusion",
            "sherwood",
            "schmidt",
            "interphase",
        ),
    )
    fluid = _contains_any(
        normalized,
        (
            "pressure drop",
            "perte de charge",
            "fluid flow",
            "fluid",
            "fluide",
            "ecoulement",
            "momentum",
            "reynolds",
            "pipe flow",
        ),
    )
    fouling = _contains_any(
        normalized,
        (
            "fouling",
            "encrassement",
            "scaling",
            "entartrage",
            "bouchage",
        ),
    )
    crystallization = _contains_any(
        normalized,
        ("crystall", "sursaturation", "supersaturation", "nucleation"),
    )
    control = _contains_any(
        normalized,
        ("pid", "mpc", "control", "regulation", "setpoint", "consigne"),
    )

    if question_type == "definition":
        if phosphoric:
            prefer(BECKER_DOCUMENT, 1.0)
            prefer(PERRY_DOCUMENT, 0.95)
            prefer(REPORT_DOCUMENT, 0.45 if not plant else 0.9)
        else:
            prefer(PERRY_DOCUMENT, 1.0)
            prefer(BECKER_DOCUMENT, 0.65)
    elif question_type == "process_flow":
        prefer(BECKER_DOCUMENT, 1.0)
        prefer(REPORT_DOCUMENT, 0.98 if phosphoric or plant else 0.65)
        prefer(PERRY_DOCUMENT, 0.8)
    elif question_type == "balance":
        species = _contains_any(normalized, ("p2o5", "species", "component", "composant"))
        energy = _contains_any(
            normalized,
            ("energy", "heat", "enthalpy", "energet", "thermique", "chaleur"),
        )
        if plant:
            prefer(REPORT_DOCUMENT, 1.0)
        if species:
            prefer(BECKER_DOCUMENT, 1.0)
            prefer(REPORT_DOCUMENT, 0.98)
            prefer(PERRY_DOCUMENT, 0.55)
        elif energy:
            prefer(THERMODYNAMICS_DOCUMENT, 1.0)
            prefer(PERRY_DOCUMENT, 0.95)
            prefer(HEAT_TRANSFER_DOCUMENT, 0.85)
            if phosphoric:
                prefer(REPORT_DOCUMENT, 0.92)
        else:
            prefer(PERRY_DOCUMENT, 1.0)
            prefer(TRANSPORT_DOCUMENT, 0.82)
            if phosphoric:
                prefer(REPORT_DOCUMENT, 0.78)
    elif question_type == "momentum_diffusion":
        prefer(TRANSPORT_DOCUMENT, 1.0)
        prefer(PERRY_DOCUMENT, 0.65)
    elif question_type == "thermodynamic_relation" or thermodynamics:
        prefer(THERMODYNAMICS_DOCUMENT, 1.0)
        prefer(PERRY_DOCUMENT, 0.65)
        if phosphoric or plant:
            prefer(REPORT_DOCUMENT, 0.82)
    elif question_type == "troubleshooting" or fouling:
        prefer(PERRY_DOCUMENT, 1.0)
        prefer(HEAT_TRANSFER_DOCUMENT, 0.9)
        if phosphoric or plant:
            prefer(REPORT_DOCUMENT, 0.98)
            prefer(BECKER_DOCUMENT, 0.72)
    elif question_type == "control_strategy" or control:
        prefer(CONTROL_DOCUMENT, 1.0)
        if plant or phosphoric:
            prefer(REPORT_DOCUMENT, 0.72)
    elif crystallization:
        prefer(CRYSTALLIZATION_DOCUMENT, 1.0)
        prefer(BECKER_DOCUMENT, 0.8)
    elif heat:
        prefer(HEAT_TRANSFER_DOCUMENT, 1.0)
        prefer(PERRY_DOCUMENT, 0.8)
        if plant or phosphoric:
            prefer(REPORT_DOCUMENT, 0.86)
    elif mass_transfer:
        prefer(TRANSPORT_DOCUMENT, 1.0)
        prefer(HEAT_TRANSFER_DOCUMENT, 0.82)
        prefer(PERRY_DOCUMENT, 0.72)
        if plant or phosphoric:
            prefer(REPORT_DOCUMENT, 0.88)
    elif fluid:
        prefer(TRANSPORT_DOCUMENT, 1.0)
        prefer(PERRY_DOCUMENT, 0.78)
    elif pump:
        prefer(PERRY_DOCUMENT, 1.0)
        prefer(TRANSPORT_DOCUMENT, 0.78)
        if phosphoric:
            prefer(BECKER_DOCUMENT, 0.82)
            prefer(REPORT_DOCUMENT, 0.88 if plant else 0.62)
    elif phosphoric:
        prefer(BECKER_DOCUMENT, 1.0)
        prefer(REPORT_DOCUMENT, 0.72 if plant else 0.55)
        prefer(PERRY_DOCUMENT, 0.62)
    else:
        prefer(PERRY_DOCUMENT, 1.0)

    if plant:
        prefer(REPORT_DOCUMENT, 1.0)
    return scores


def _section_affinity_terms(
    normalized: str,
    question_type: str,
) -> tuple[str, ...]:
    """Return heading-level hints, especially for the multi-domain OCP report.

    These hints never filter candidates.  They only provide a small post-rerank
    preference when a retrieved chunk's chapter/section heading matches the
    requested technical subtopic.
    """

    terms: list[str] = []

    def add(*values: str) -> None:
        for value in values:
            normalized_value = _normalize(value)
            if normalized_value and normalized_value not in terms:
                terms.append(normalized_value)

    plant = _contains_any(
        normalized,
        ("ocp", "jfc4", "atelier", "echelon", "rapport", "sur site"),
    )
    species = _contains_any(
        normalized,
        ("p2o5", "species", "component", "composant", "fluor", "water", "eau"),
    )
    energy = _contains_any(
        normalized,
        ("energy", "heat", "enthalpy", "energet", "thermique", "chaleur", "steam", "vapeur"),
    )

    if question_type == "process_flow":
        add(
            "principe de fonctionnement",
            "circuit de concentration",
            "description de l unite de concentration",
            "flowsheet de l echelon",
            "equipements de l unite de concentration",
        )
    elif question_type == "balance":
        if species:
            add(
                "bilan de matiere",
                "debit des constituants",
                "sortie acide",
                "sortie bouilleur",
                "ligne 1 entree acide",
                "ligne 5 sortie acide",
                "ligne 6 sortie bouilleur",
                "p2o5 entraine",
                "tableau 6",
                "tableau 7",
                "tableau 8",
            )
        elif energy:
            add(
                "bilan thermique",
                "flux thermique",
                "debit massique de la vapeur",
                "debit de circulation",
                "debit d eau de bassin",
            )
        else:
            add("bilans theoriques", "bilan de matiere", "bilan thermique")
    elif question_type == "troubleshooting":
        add(
            "diagnostic d etat",
            "diagramme d ishikawa",
            "resistance d encrassement",
            "debouchage de l echangeur",
            "periode de lavage",
            "duree et frequence des arrets",
        )
    elif question_type == "momentum_diffusion":
        add(
            "transport of momentum",
            "momentum flux",
            "newton s law of viscosity",
            "velocity gradient",
            "shear stress",
        )

    if _contains_any(
        normalized,
        (
            "echangeur",
            "heat exchanger",
            "lmtd",
            "coefficient global",
            "kern seaton",
            "kern and seaton",
        ),
    ):
        add(
            "parametre de marche echangeur",
            "coefficient global de transfert thermique",
            "difference de temperature moyenne logarithmique",
            "resistance d encrassement",
        )
    if _contains_any(
        normalized,
        (
            "condenseur",
            "condenser",
            "sherwood",
            "schmidt",
            "mass transfer",
            "transfert de masse",
        ),
    ):
        add(
            "parametre de marche condenseur",
            "colonne de condensation",
            "transfert de masse",
            "chaleur latente de condensation",
        )
    if _contains_any(normalized, ("bouilleur", "flash", "vapor body", "pression du bouilleur")):
        add(
            "parametre de marche bouilleur",
            "debit de gaz",
            "debit d acide produit",
            "simulateur bouilleur",
        )
    if _contains_any(
        normalized,
        (
            "simulation",
            "simulateur",
            "modelisation",
            "optim",
            "point de fonctionnement",
        ),
    ):
        add(
            "modelisation du procede",
            "determination des parametres optimaux",
            "resultats matlab",
            "solutions proposees",
        )
    if plant and _contains_any(
        normalized,
        (
            "performance",
            "productivite",
            "production",
            "capacite evaporatoire",
            "arret",
        ),
    ):
        add(
            "diagnostic d etat des echelons",
            "production reelle et theorique",
            "capacite evaporatoire reelle et theorique",
            "impact des heures d arret",
        )

    return tuple(terms)


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
    question_type: str = "explanation"
    explicit_source: str | None = None
    temporal_scope: str = "static"
    section_affinity_terms: tuple[str, ...] = ()


def route_query(
    question: str,
    *,
    catalog: KnowledgeBaseCatalog,
    source_mode: str = "auto",
    question_type: str | None = None,
    focus_entity: str | None = None,
    maximum_source_boost: float = 0.06,
) -> DomainRoutingDecision:
    """Route across all eight PDFs while preserving global automatic search."""

    mode = source_mode.strip().casefold()
    if mode == "automatic":
        mode = "auto"
    if mode not in SUPPORTED_SOURCE_MODES:
        raise ValueError(
            "Mode source invalide : " + ", ".join(sorted(SUPPORTED_SOURCE_MODES))
        )

    explicit = detect_explicit_source_mode(question) if mode == "auto" else None
    effective_mode = explicit or mode
    resolved_type = (question_type or "").strip().casefold()
    normalized = _normalize(" ".join(part for part in (question, focus_entity or "") if part))
    if not resolved_type:
        resolved_type = _infer_question_type(normalized)

    if effective_mode != "auto":
        document_id = SOURCE_MODE_DOCUMENTS[effective_mode]
        explicit_domains = (
            ((KnowledgeDomain.FLUID_MECHANICS, 1.0),)
            if resolved_type == "momentum_diffusion"
            else ()
        )
        return DomainRoutingDecision(
            detected_domains=explicit_domains,
            confidence=1.0,
            preferred_documents=(document_id,),
            soft_boosts={},
            explanation=f"Explicit user source filter: {effective_mode}",
            hard_filter=frozenset({document_id}),
            source_mode=effective_mode,
            question_type=resolved_type,
            explicit_source=effective_mode,
            section_affinity_terms=_section_affinity_terms(
                normalized,
                resolved_type,
            ),
        )

    raw_scores: dict[KnowledgeDomain, float] = {}
    for domain, terms in _DOMAIN_TERMS.items():
        matches = sum(_normalize(term) in normalized for term in terms)
        if matches:
            raw_scores[domain] = min(1.0, 0.45 + 0.2 * matches)
    if not raw_scores:
        raw_scores[KnowledgeDomain.GENERAL_CHEMICAL_ENGINEERING] = 0.4
    if resolved_type == "momentum_diffusion":
        raw_scores.pop(KnowledgeDomain.MASS_TRANSFER, None)
        raw_scores[KnowledgeDomain.FLUID_MECHANICS] = 1.0

    detected = tuple(
        sorted(raw_scores.items(), key=lambda item: (-item[1], item[0].value))
    )
    profile_scores = _profile_scores(normalized, resolved_type)
    boosts: dict[str, float] = {}

    for document in catalog.documents:
        overlap = sum(raw_scores.get(domain, 0.0) for domain in document.domains)
        profile = profile_scores.get(document.document_id, 0.0)
        if overlap <= 0 and profile <= 0:
            continue
        priority_factor = 0.75 + 0.25 * document.priority / 100
        primary_bonus = sum(
            0.003 * raw_scores.get(domain, 0.0)
            for domain, primary_document in _DOMAIN_PRIMARY_DOCUMENTS.items()
            if primary_document == document.document_id
        )
        boosts[document.document_id] = round(
            min(
                maximum_source_boost,
                overlap * 0.015 * priority_factor
                + profile * 0.04
                + primary_bonus,
            ),
            6,
        )

    preferred = tuple(
        document_id
        for document_id, _boost in sorted(
            boosts.items(), key=lambda item: (-item[1], item[0])
        )
    )
    temporal_scope = (
        "live_or_current"
        if _contains_any(
            normalized,
            (
                "today",
                "now",
                "current",
                "actuel",
                "aujourd hui",
                "maintenant",
            ),
        )
        or re.search(r"\b20\d{2}\b.*\b\d{1,2}\s*h", normalized)
        else "static"
    )
    return DomainRoutingDecision(
        detected_domains=detected,
        confidence=max(score for _domain, score in detected),
        preferred_documents=preferred,
        soft_boosts=boosts,
        explanation=(
            "Intent/entity source profile plus deterministic domain overlap; "
            "all active documents remain searchable in automatic mode."
        ),
        hard_filter=None,
        source_mode="auto",
        question_type=resolved_type,
        explicit_source=None,
        temporal_scope=temporal_scope,
        section_affinity_terms=_section_affinity_terms(
            normalized,
            resolved_type,
        ),
    )
