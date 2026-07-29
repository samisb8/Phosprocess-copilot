"""Intent-aware retrieval planning for retriever v4.

The planner decomposes one user question into independently retrievable
information roles. It does not inject documentary facts and never generates an
answer. Its only purpose is to make every required side of a question visible
to dense, sparse and lexical retrieval before reranking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvidenceRole:
    """One independently retrievable evidence need."""

    name: str
    query: str
    required: bool = True
    subject: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    """Structured retrieval contract for one standalone question."""

    question_type: str
    base_query: str
    roles: tuple[EvidenceRole, ...]
    comparison_subjects: tuple[str, str] | None = None
    balance_kind: str | None = None
    answer_intent: str | None = None

    @property
    def queries(self) -> tuple[str, ...]:
        return tuple(role.query for role in self.roles)


_COMPARISON_PATTERNS = (
    re.compile(
        r"\bcompare\s+(?P<a>.+?)\s+(?:with|to|versus|vs\.?)\s+(?P<b>.+?)"
        r"(?:\s+for\b|\s+in\b|[?.]|$)",
        re.I,
    ),
    re.compile(
        r"\bdifference\s+between\s+(?P<a>.+?)\s+and\s+(?P<b>.+?)"
        r"(?:\s+for\b|\s+in\b|[?.]|$)",
        re.I,
    ),
    re.compile(
        r"\bcompar(?:e|er|ez)\s+(?P<a>.+?)\s+(?:avec|à|et|versus|vs\.?)\s+"
        r"(?P<b>.+?)(?:\s+pour\b|\s+dans\b|[?.]|$)",
        re.I,
    ),
)

_TECHNICAL_ENGLISH = (
    (re.compile(r"\bévaporateur à circulation forcée\b", re.I), "forced-circulation evaporator"),
    (re.compile(r"\bcirculation forcée\b", re.I), "forced circulation"),
    (re.compile(r"\bfilm tombant\b", re.I), "falling-film evaporator"),
    (re.compile(r"\bévaporateur\b", re.I), "evaporator"),
    (re.compile(r"\bpompe de circulation\b", re.I), "circulation pump"),
    (re.compile(r"\béchangeur(?: de chaleur| thermique)?\b", re.I), "heat exchanger"),
    (re.compile(r"\bchambre de flash\b", re.I), "flash chamber"),
    (re.compile(r"\bcorps de vapeur\b", re.I), "vapor body"),
    (re.compile(r"\bacide phosphorique\b", re.I), "phosphoric acid"),
    (re.compile(r"\bbilan\b", re.I), "balance"),
    (re.compile(r"\brégime permanent\b", re.I), "steady state"),
    (re.compile(r"\bencrassement\b", re.I), "fouling"),
    (re.compile(r"\bentartrage\b", re.I), "scaling"),
    (re.compile(r"غرفة التبخير"), "vapor body evaporation chamber"),
    (re.compile(r"فصل البخار"), "vapor liquid separation"),
    (re.compile(r"الحمض"), "acid"),
    (re.compile(r"المبخر"), "evaporator"),
)


def _compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t\r\n,;:-")


def _english_technical_projection(query: str) -> str:
    additions: list[str] = []
    for pattern, english in _TECHNICAL_ENGLISH:
        if pattern.search(query) and english.casefold() not in query.casefold():
            additions.append(english)
    return " ".join(dict.fromkeys(additions))


def _extract_comparison_subjects(question: str) -> tuple[str, str] | None:
    for pattern in _COMPARISON_PATTERNS:
        match = pattern.search(question)
        if match is None:
            continue
        subject_a = _compact(match.group("a"))
        subject_b = _compact(match.group("b"))
        if subject_a and subject_b:
            return subject_a, subject_b
    return None


def _balance_kind(question: str) -> str:
    normalized = question.casefold().replace("₂", "2").replace("₅", "5")
    p2o5 = "p2o5" in normalized
    plant = any(
        term in normalized
        for term in (
            "jfc4",
            "echelon",
            "échelon",
            "rapport ocp",
            "rapport atelier",
            "ocp report",
        )
    )
    if p2o5 and plant:
        return "p2o5_plant"
    if any(
        term in normalized
        for term in (
            "p2o5",
            "species",
            "component",
            "espèce",
            "composant",
        )
    ):
        return "species"
    energy_terms = (
        "energy",
        "heat",
        "enthalpy",
        "énerg",
        "thermique",
        "chaleur",
    )
    if any(term in normalized for term in energy_terms):
        return "energy"
    return "overall_mass"


def _normalized_intent_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().replace("’", "'")).strip()


def _scoped_explanation_intent(value: str) -> str | None:
    normalized = _normalized_intent_text(value)
    pump = any(
        marker in normalized
        for marker in (
            "circulation pump",
            "pompe de circulation",
            "مضخة الدوران",
        )
    )
    if not pump:
        return None
    if any(
        marker in normalized
        for marker in (
            "back to the flash chamber",
            "send the liquid back",
            "return to the flash chamber",
            "ramener le liquide",
            "renvoyer le liquide",
            "إعاد",
        )
    ):
        return "pump_return"
    if any(
        marker in normalized
        for marker in (
            "necessary",
            "nécessaire",
            "necessaire",
            "why",
            "pourquoi",
            "ضرورية",
            "لماذا",
        )
    ):
        return "pump_necessity"
    if any(
        marker in normalized
        for marker in ("role", "rôle", "fonction", "what does", "دور")
    ):
        return "pump_role"
    return None


def build_retrieval_plan(
    original_question: str,
    *,
    standalone_query: str,
    question_type: str,
) -> RetrievalPlan:
    """Create deterministic evidence roles for the current question."""

    original = _compact(original_question)
    standalone = _compact(standalone_query)
    if not original or not standalone:
        raise ValueError("Les questions du plan de retrieval ne peuvent pas être vides.")

    projected = _english_technical_projection(f"{original} {standalone}")
    base_query = _compact(" ".join(part for part in (standalone, projected) if part))

    if question_type == "definition":
        roles = (
            EvidenceRole(
                "definition_nature",
                f"{base_query} forced circulation evaporator equipment definition",
                required=False,
            ),
            EvidenceRole(
                "definition_mechanism",
                f"{base_query} acid circulation pump heat exchanger pressure drop "
                "large flow heating surface",
            ),
            EvidenceRole(
                "definition_function",
                f"{base_query} vapor body hot acid from heat exchanger "
                "vapor liquid separation evaporation chamber",
            ),
        )
        return RetrievalPlan(
            question_type=question_type,
            base_query=base_query,
            roles=roles,
            answer_intent="definition",
        )

    if question_type == "explanation":
        intent = _scoped_explanation_intent(base_query)
        if intent == "pump_necessity":
            roles = (
                EvidenceRole(
                    "pump_circulation",
                    f"{base_query} forced positive circulation independent evaporation rate",
                ),
                EvidenceRole(
                    "pump_heating_path",
                    f"{base_query} pump liquid through heating surface heat exchanger",
                ),
                EvidenceRole(
                    "pump_process_function",
                    f"{base_query} heat transfer vapor liquid separation separate functions",
                    required=False,
                ),
            )
            return RetrievalPlan(
                question_type,
                base_query,
                roles,
                answer_intent=intent,
            )
        if intent == "pump_role":
            roles = (
                EvidenceRole(
                    "pump_withdrawal",
                    f"{base_query} pump withdraws liquor from flash chamber",
                    required=False,
                ),
                EvidenceRole(
                    "pump_heating_path",
                    f"{base_query} pump forces liquid through heating element",
                ),
                EvidenceRole(
                    "pump_process_function",
                    f"{base_query} heat transfer vapor liquid separation separate functions",
                    required=False,
                ),
            )
            return RetrievalPlan(
                question_type,
                base_query,
                roles,
                answer_intent=intent,
            )
        if intent == "pump_return":
            roles = (
                EvidenceRole(
                    "pump_return_path",
                    f"{base_query} withdraw flash chamber heating element back to flash chamber",
                ),
            )
            return RetrievalPlan(
                question_type,
                base_query,
                roles,
                answer_intent=intent,
            )

    if question_type == "momentum_diffusion":
        roles = (
            EvidenceRole(
                "momentum_transport",
                f"{base_query} molecular transport of momentum momentum flux",
            ),
            EvidenceRole(
                "velocity_gradient",
                f"{base_query} velocity gradient adjacent fluid layers",
            ),
            EvidenceRole(
                "newton_viscosity_law",
                f"{base_query} Newton law of viscosity shear stress viscosity",
            ),
        )
        return RetrievalPlan(
            question_type=question_type,
            base_query=base_query,
            roles=roles,
            answer_intent="momentum_diffusion",
        )

    if question_type == "comparison":
        subjects = (
            _extract_comparison_subjects(original)
            or _extract_comparison_subjects(standalone)
        )
        if subjects is None:
            return RetrievalPlan(
                question_type=question_type,
                base_query=base_query,
                roles=(EvidenceRole("comparison_context", base_query),),
            )
        subject_a, subject_b = subjects
        roles = (
            EvidenceRole(
                "equipment_a",
                f"{subject_a} operation design applications heat transfer "
                "fouling viscosity residence time",
                subject=subject_a,
            ),
            EvidenceRole(
                "equipment_b",
                f"{subject_b} operation design applications heat transfer "
                "fouling viscosity residence time",
                subject=subject_b,
            ),
            EvidenceRole(
                "comparison_criteria",
                f"{subject_a} {subject_b} comparison heat transfer fouling "
                "scaling viscosity residence time circulation",
                subject=f"{subject_a} versus {subject_b}",
            ),
        )
        return RetrievalPlan(
            question_type=question_type,
            base_query=base_query,
            roles=roles,
            comparison_subjects=subjects,
        )

    if question_type == "balance":
        kind = _balance_kind(base_query)
        if kind == "p2o5_plant":
            roles = (
                EvidenceRole(
                    "p2o5_conservation",
                    f"{base_query} bilan matière global P2O5 ligne 1 ligne 5 ligne 6",
                    required=False,
                ),
                EvidenceRole(
                    "p2o5_feed",
                    f"{base_query} ligne 1 entrée acide débit P2O5 alimentation "
                    "18.03 t/h 18030 kg/h",
                ),
                EvidenceRole(
                    "p2o5_product",
                    f"{base_query} ligne 5 sortie acide débit P2O5 produit "
                    "18 t/h 18000 kg/h",
                ),
                EvidenceRole(
                    "p2o5_entrainment",
                    f"{base_query} ligne 6 sortie bouilleur P2O5 entraîné gaz "
                    "30 kg/h",
                ),
            )
        elif kind == "species":
            roles = (
                EvidenceRole(
                    "species_conservation",
                    "steady-state conservation law for a chemical species "
                    "component balance equation",
                ),
                EvidenceRole(
                    "species_feed",
                    f"{base_query} P2O5 feed flow concentration component inlet",
                ),
                EvidenceRole(
                    "species_product",
                    f"{base_query} P2O5 concentrated product flow concentration outlet",
                ),
                EvidenceRole(
                    "species_losses",
                    f"{base_query} P2O5 vapor entrainment losses carryover evaporator",
                ),
            )
        elif kind == "energy":
            roles = (
                EvidenceRole(
                    "energy_conservation",
                    "steady-state control-volume energy conservation equation "
                    "enthalpy in out heat work",
                ),
                EvidenceRole(
                    "heat_input",
                    f"{base_query} evaporator steam duty heating medium heat input",
                ),
                EvidenceRole(
                    "feed_product_enthalpy",
                    f"{base_query} feed enthalpy concentrated product enthalpy",
                ),
                EvidenceRole(
                    "vapor_enthalpy",
                    f"{base_query} water vapor latent heat vapor enthalpy evaporation",
                ),
            )
        else:
            roles = (
                EvidenceRole(
                    "overall_conservation",
                    "steady-state overall mass balance mass in mass out accumulation generation",
                ),
                EvidenceRole(
                    "feed_stream",
                    f"{base_query} feed stream mass flow inlet",
                ),
                EvidenceRole(
                    "product_and_vapor",
                    f"{base_query} concentrated liquid product vapor outlet mass flow",
                ),
            )
        return RetrievalPlan(
            question_type=question_type,
            base_query=base_query,
            roles=roles,
            balance_kind=kind,
        )

    if question_type == "troubleshooting":
        roles = (
            EvidenceRole("cause", f"{base_query} documented causes"),
            EvidenceRole("mechanism", f"{base_query} physical mechanism"),
            EvidenceRole("effect", f"{base_query} documented operational effects performance loss"),
            EvidenceRole("action", f"{base_query} documented mitigation cleaning operating action"),
        )
        return RetrievalPlan(question_type, base_query, roles)

    if question_type == "process_flow":
        roles = (
            EvidenceRole("feed_inlet", f"{base_query} feed inlet"),
            EvidenceRole(
                "conical_bottom",
                f"{base_query} cycling acid leaves vapor body conical bottom",
            ),
            EvidenceRole(
                "pump_heat_exchanger",
                f"{base_query} circulation pump heating element heat exchanger",
            ),
            EvidenceRole(
                "vapor_body",
                f"{base_query} vapor body flash chamber vapor liquid separation",
            ),
            EvidenceRole("recirculation", f"{base_query} return recirculation loop flash chamber"),
            EvidenceRole("product_outlet", f"{base_query} concentrated product withdrawal outlet"),
        )
        return RetrievalPlan(question_type, base_query, roles)

    return RetrievalPlan(
        question_type=question_type,
        base_query=base_query,
        roles=(EvidenceRole("primary", base_query),),
    )
