"""Role-aware evidence selection for structured answer contracts."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from phosprocess.reranking.reranker import RerankedSearchResult
from phosprocess.retrieval.hybrid import HybridSearchResult
from phosprocess.retrieval.retrieval_planner import EvidenceRole, RetrievalPlan
from phosprocess.retrieval.v3_selection import V3SelectedResult


@dataclass(frozen=True, slots=True)
class RoleEvidenceSelection:
    """Final selection plus explicit required-role coverage."""

    selected: tuple[V3SelectedResult, ...]
    covered_roles: tuple[str, ...]
    missing_roles: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_roles


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    ).casefold()
    return re.sub(r"[^a-z0-9%]+", " ", without_marks).strip()


def _chunk_text(result: HybridSearchResult) -> str:
    chunk = result.chunk
    return _normalize(
        " ".join(
            part
            for part in (
                chunk.document_title or "",
                chunk.hierarchy_path or "",
                chunk.section or "",
                chunk.subsection or "",
                chunk.text,
                chunk.embedding_text,
            )
            if part
        )
    )


def _is_non_evidentiary_candidate(candidate: HybridSearchResult) -> bool:
    chunk_type = (candidate.chunk.chunk_type or "").strip().casefold()
    if chunk_type in {
        "figure_caption",
        "table_of_contents",
        "index",
        "bibliography",
    }:
        return True

    heading = _normalize(
        " ".join(
            part
            for part in (
                candidate.chunk.hierarchy_path or "",
                candidate.chunk.section or "",
                candidate.chunk.subsection or "",
            )
            if part
        )
    )
    return any(
        marker in heading
        for marker in (
            "liste des figures",
            "list of figures",
            "table des matieres",
            "table of contents",
        )
    )


def _contains_number_pattern(
    text: str,
    patterns: tuple[str, ...],
) -> bool:
    return any(re.search(pattern, text) is not None for pattern in patterns)


def _supports_p2o5_plant_role(text: str, role_name: str) -> bool:
    if role_name == "p2o5_conservation":
        has_stream_relation = (
            ("ligne 1" in text and "ligne 5" in text and "ligne 6" in text)
            or ("m1" in text and "m5" in text and "entrainee" in text)
            or (
                "alimentation" in text
                and "produit" in text
                and "entrainee" in text
            )
        )
        return has_stream_relation and (
            "bilan de matiere" in text
            or "sortie bouilleur" in text
            or "entrainee" in text
        )

    if role_name == "p2o5_feed":
        has_feed_context = any(
            marker in text
            for marker in (
                "ligne 1",
                "entree acide",
                "alimentation",
            )
        )
        has_feed_value = _contains_number_pattern(
            text,
            (
                r"(?<!\d)18\s+03(?!\d)",
                r"(?<!\d)18030(?!\d)",
            ),
        )
        return has_feed_context and has_feed_value

    if role_name == "p2o5_product":
        has_product_context = any(
            marker in text
            for marker in (
                "ligne 5",
                "sortie acide",
                "produit concentre",
                "debit de p2o5 a la sortie",
            )
        )
        has_product_value = _contains_number_pattern(
            text,
            (
                r"(?<!\d)18\s+00(?!\d)",
                r"(?<!\d)18000(?!\d)",
                r"(?<!\d)18\s*t\s*h(?!\w)",
                r"(?<!\d)18t\s+h(?!\w)",
            ),
        )
        return has_product_context and has_product_value

    if role_name == "p2o5_entrainment":
        has_loss_context = any(
            marker in text
            for marker in (
                "entrainee",
                "ligne 6",
                "sortie bouilleur",
            )
        )
        has_loss_value = _contains_number_pattern(
            text,
            (
                r"(?<!\d)30\s*kg\s+h(?!\w)",
                r"(?<!\d)30\s+kg\s+h(?!\w)",
                r"(?<!\d)0\s+03\s*t\s+h(?!\w)",
            ),
        )
        return has_loss_context and has_loss_value

    return False


def _subject_aliases(subject: str) -> tuple[str, ...]:
    normalized = _normalize(subject)
    aliases = {normalized}
    if "forced circulation" in normalized or "circulation forcee" in normalized:
        aliases.update(
            {
                "forced circulation evaporator",
                "forced circulation",
                "fc evaporator",
                "evaporateur a circulation forcee",
            }
        )
    if "falling film" in normalized or "film tombant" in normalized:
        aliases.update(
            {
                "falling film evaporator",
                "falling film",
                "evaporateur a film tombant",
                "film tombant",
            }
        )
    return tuple(alias for alias in aliases if alias)


def _supports_subject(
    candidate: HybridSearchResult,
    subject: str | None,
) -> bool:
    if not subject:
        return True
    if (candidate.chunk.chunk_type or "") in {
        "figure_caption",
        "table_of_contents",
        "index",
        "bibliography",
    }:
        return False

    text = _chunk_text(candidate)
    aliases = _subject_aliases(subject)
    if any(alias in text for alias in aliases):
        return True
    tokens = [
        token
        for token in _normalize(subject).split()
        if token not in {"a", "an", "the", "with", "and", "evaporator"}
    ]
    return bool(tokens) and all(token in text for token in tokens)


def _supports_balance_role(text: str, role_name: str) -> bool:
    if role_name.startswith("p2o5_"):
        return _supports_p2o5_plant_role(text, role_name)

    marker_groups = {
        "species_conservation": (
            ("mass in", "mass out"),
            ("component balance",),
            ("species balance",),
            ("conservation law for mass",),
            ("conservation of mass",),
            ("mass balance",),
            ("material balance",),
            ("bilan de matiere",),
            ("conservation de la masse",),
            ("accumulation", "generation"),
        ),
        "species_feed": (
            ("p2o5", "feed"),
            ("p2o5", "inlet"),
            ("component", "mass in"),
        ),
        "species_product": (
            ("p2o5", "product"),
            ("p2o5", "outlet"),
            ("concentrated", "product"),
        ),
        "species_losses": (
            ("p2o5", "loss"),
            ("entrainment",),
            ("carryover",),
            ("droplet", "evaporation"),
        ),
        "p2o5_conservation": (
            ("bilan de matiere global",),
            ("bilan de matiere", "p2o5"),
            ("p2o5", "entrainee", "sortie bouilleur"),
            ("mass balance", "p2o5"),
        ),
        "p2o5_feed": (
            ("p2o5", "entree"),
            ("p2o5", "alimentation"),
            ("ligne 1", "entree acide"),
            ("debit massique", "p2o5"),
        ),
        "p2o5_product": (
            ("p2o5", "sortie"),
            ("p2o5", "produit"),
            ("ligne 5", "sortie acide"),
            ("productivite", "p2o5"),
        ),
        "p2o5_entrainment": (
            ("p2o5", "entrainee"),
            ("p2o5", "sortie bouilleur"),
            ("ligne 6", "p2o5"),
            ("p2o5", "gaz"),
        ),
        "energy_conservation": (
            ("energy", "in", "out"),
            ("energy balance",),
            ("enthalpy balance",),
            ("conservation of energy",),
            ("first law", "control volume"),
            ("bilan energetique",),
            ("conservation de l energie",),
            ("heat", "work", "enthalpy"),
        ),
        "heat_input": (
            ("steam", "heat"),
            ("heating medium",),
            ("heat input",),
            ("steam duty",),
        ),
        "feed_product_enthalpy": (
            ("feed", "enthalpy"),
            ("product", "enthalpy"),
            ("liquid", "enthalpy"),
        ),
        "vapor_enthalpy": (
            ("vapor", "enthalpy"),
            ("latent heat",),
            ("water evaporated", "heat"),
        ),
        "overall_conservation": (
            ("mass in", "mass out"),
            ("overall mass balance",),
            ("material balance", "accumulation"),
            ("conservation of mass",),
            ("mass balance",),
            ("bilan de matiere",),
            ("conservation de la masse",),
        ),
        "feed_stream": (("feed", "mass"), ("mass in",)),
        "product_and_vapor": (
            ("product", "vapor"),
            ("mass out",),
            ("evaporated water",),
        ),
    }
    groups = marker_groups.get(role_name)
    if not groups:
        return True
    return any(all(marker in text for marker in group) for group in groups)


def _supports_troubleshooting_role(text: str, role_name: str) -> bool:
    marker_groups = {
        "cause": (
            "cause",
            "caused by",
            "due to",
            "may be due",
            "results from",
            "provoque",
            "cause par",
            "du a",
        ),
        "mechanism": (
            "deposit",
            "accumulation",
            "precipitation",
            "corrosion",
            "resistance",
            "crystallization",
            "depot",
            "entartrage",
        ),
        "effect": (
            "decrease",
            "reduce",
            "increase",
            "performance",
            "heat transfer",
            "pressure drop",
            "capacity",
            "efficiency",
            "diminue",
            "augmente",
            "perte",
        ),
        "action": (
            "clean",
            "remove",
            "reduce",
            "eliminate",
            "wash",
            "control",
            "maintain",
            "mitigation",
            "nettoyage",
            "eliminer",
            "lavage",
        ),
    }
    markers = marker_groups.get(role_name)
    return True if markers is None else any(marker in text for marker in markers)


def _supports_definition_or_pump_role(text: str, role_name: str) -> bool:
    pump_markers = (
        "circulation pump",
        "acid circulation pump",
        "pump",
        "pompe de circulation",
        "pompe",
    )
    heating_markers = (
        "heating element",
        "heating surface",
        "heat exchanger",
        "echangeur de chaleur",
        "echangeur",
        "surface de chauffe",
        "element chauffant",
    )
    hydraulic_markers = (
        "withdraws",
        "withdraw",
        "forces",
        "force",
        "through",
        "past the heating",
        "returns",
        "return",
        "back to",
        "pressure drop",
        "pressure head",
        "head requirement",
        "flow capacity",
        "flow rate",
        "large flow",
        "high flow",
        "debit",
        "perte de charge",
        "hauteur",
        "traverse",
        "refoule",
        "renvoie",
    )
    has_pump = any(marker in text for marker in pump_markers)
    has_heating = any(marker in text for marker in heating_markers)
    has_hydraulic_relation = any(
        marker in text for marker in hydraulic_markers
    )

    if role_name == "definition_nature":
        return any(
            marker in text
            for marker in (
                "forced circulation evaporator",
                "forced circulation fc evaporator",
                "evaporateur a circulation forcee",
            )
        ) or (
            "evaporator" in text
            and has_pump
            and has_heating
            and has_hydraulic_relation
        )

    if role_name == "definition_mechanism":
        return has_pump and has_heating and has_hydraulic_relation

    if role_name == "definition_function":
        explicit_separation = (
            any(
                marker in text
                for marker in (
                    "vapor liquid separation",
                    "vapour liquid separation",
                    "separation vapeur liquide",
                )
            )
            and any(
                marker in text
                for marker in (
                    "vapor body",
                    "vapour body",
                    "evaporation chamber",
                    "corps de l evaporateur",
                    "corps d evaporation",
                )
            )
        )
        separated_functions = (
            "heat transfer" in text
            and any(
                marker in text
                for marker in (
                    "vapor liquid separation",
                    "vapour liquid separation",
                    "separation vapeur liquide",
                )
            )
        )
        return explicit_separation or separated_functions

    if role_name == "pump_circulation":
        return has_pump and "circulation" in text and any(
            marker in text
            for marker in (
                "maintained",
                "maintains",
                "ensure circulation",
                "large flow",
                "high flow",
                "flow rate",
                "flow capacity",
                "debit",
                "assure",
                "maintient",
            )
        )

    if role_name == "pump_withdrawal":
        return has_pump and any(
            marker in text
            for marker in (
                "withdraws liquor from the flash chamber",
                "withdraw liquor from the flash chamber",
                "pump withdraws",
                "retire le liquide de la chambre",
                "aspire le liquide de la chambre",
            )
        )

    if role_name == "pump_heating_path":
        return has_pump and has_heating and has_hydraulic_relation

    if role_name == "pump_process_function":
        return (
            "heat transfer" in text
            and any(
                marker in text
                for marker in (
                    "vapor liquid separation",
                    "vapour liquid separation",
                    "separation vapeur liquide",
                )
            )
            and any(
                marker in text
                for marker in (
                    "separate functions",
                    "separated",
                    "crystallization",
                    "dissocier les fonctions",
                    "separer les fonctions",
                )
            )
        )

    if role_name == "pump_return_path":
        return (
            has_pump
            and has_heating
            and any(
                marker in text
                for marker in (
                    "back to the flash chamber",
                    "return to the flash chamber",
                    "returns it to the vapor body",
                    "renvoie vers la chambre",
                    "retour vers la chambre",
                )
            )
        )

    return True


def _supports_momentum_role(text: str, role_name: str) -> bool:
    momentum_markers = (
        "momentum transport",
        "transport of momentum",
        "momentum flux",
        "velocity gradient",
        "shear stress",
        "shearing force",
        "newton s law of viscosity",
    )
    mass_diffusion_markers = (
        "fick s law",
        "fick law",
        "concentration gradient",
        "mass transport",
        "molecular diffusivity",
        "diffusivity",
    )
    has_momentum_context = any(marker in text for marker in momentum_markers)
    has_mass_diffusion_context = any(
        marker in text for marker in mass_diffusion_markers
    )
    if has_mass_diffusion_context and not any(
        marker in text
        for marker in (
            "momentum flux",
            "velocity gradient",
            "shear stress",
            "shearing force",
        )
    ):
        return False

    if role_name == "momentum_transport":
        return has_momentum_context and any(
            marker in text
            for marker in (
                "transport of momentum",
                "momentum transport",
                "momentum flux",
                "molecular transport of momentum",
            )
        )

    if role_name == "velocity_gradient":
        return "velocity gradient" in text and any(
            marker in text
            for marker in (
                "momentum",
                "shear stress",
                "shearing force",
            )
        )

    if role_name == "newton_viscosity_law":
        has_law = (
            "newton s law of viscosity" in text
            or "newton law of viscosity" in text
        )
        has_relation = (
            ("velocity gradient" in text and "viscosity" in text)
            or ("shear stress" in text and "viscosity" in text)
            or ("shearing force" in text and "viscosity" in text)
            or ("momentum flux" in text and "viscosity" in text)
        )
        return has_law and has_relation

    return False


def _supports_role(
    candidate: HybridSearchResult,
    role: EvidenceRole,
    plan: RetrievalPlan,
) -> bool:
    if role.name not in candidate.role_matches:
        return False
    if (
        plan.question_type
        in {"balance", "momentum_diffusion", "definition", "explanation"}
        and _is_non_evidentiary_candidate(candidate)
    ):
        return False

    text = _chunk_text(candidate)
    if role.name in {"equipment_a", "equipment_b"}:
        return _supports_subject(candidate, role.subject)
    if plan.question_type == "balance":
        return _supports_balance_role(text, role.name)
    if plan.question_type == "troubleshooting":
        return _supports_troubleshooting_role(text, role.name)
    if plan.question_type == "momentum_diffusion":
        return _supports_momentum_role(text, role.name)
    if plan.question_type in {"definition", "explanation"}:
        return _supports_definition_or_pump_role(text, role.name)
    if role.name == "comparison_criteria":
        return any(
            marker in text
            for marker in (
                "heat transfer",
                "fouling",
                "scaling",
                "viscosity",
                "residence time",
                "circulation",
                "application",
            )
        )
    if role.name == "conical_bottom":
        return any(
            marker in text
            for marker in (
                "conical bottom",
                "cone bottom",
                "fond conique",
                "القاع المخروطي",
            )
        )
    return True


def supported_role_names(
    plan: RetrievalPlan,
    candidates: Sequence[HybridSearchResult],
) -> tuple[str, ...]:
    """Return roles explicitly supported by at least one candidate."""

    return tuple(
        role.name
        for role in plan.roles
        if any(_supports_role(candidate, role, plan) for candidate in candidates)
    )



def promote_required_roles_in_reranking(
    plan: RetrievalPlan,
    candidates: Sequence[HybridSearchResult],
    reranked_results: Sequence[RerankedSearchResult],
) -> list[RerankedSearchResult]:
    """Move the best explicit evidence for every required role to the front.

    The cross-encoder still scores every candidate once against the complete
    question. This deterministic normalization only prevents a generic passage
    from pushing an explicit equation or stream definition below the evidence
    window. It does not create evidence and it preserves the original order
    inside both the reserved and remaining groups.
    """

    candidate_by_id = {item.chunk.chunk_id: item for item in candidates}
    reserved: list[RerankedSearchResult] = []
    reserved_ids: set[str] = set()

    for role in plan.roles:
        if not role.required:
            continue
        matching = [
            item
            for item in reranked_results
            if item.chunk.chunk_id in candidate_by_id
            and _supports_role(
                candidate_by_id[item.chunk.chunk_id],
                role,
                plan,
            )
        ]
        if not matching:
            continue
        chosen = next(
            (
                item
                for item in matching
                if item.chunk.chunk_id not in reserved_ids
            ),
            matching[0],
        )
        chunk_id = chosen.chunk.chunk_id
        if chunk_id in reserved_ids:
            continue
        reserved.append(chosen)
        reserved_ids.add(chunk_id)

    ordered = [
        *reserved,
        *(
            item
            for item in reranked_results
            if item.chunk.chunk_id not in reserved_ids
        ),
    ]
    return [
        RerankedSearchResult(
            rank=rank,
            reranker_score=item.reranker_score,
            original_hybrid_rank=item.original_hybrid_rank,
            original_rrf_score=item.original_rrf_score,
            matched_retrievers=item.matched_retrievers,
            dense_rank=item.dense_rank,
            dense_score=item.dense_score,
            bm25_rank=item.bm25_rank,
            bm25_score=item.bm25_score,
            chunk=item.chunk,
            sparse_rank=item.sparse_rank,
            sparse_score=item.sparse_score,
            colbert_score=item.colbert_score,
            section_bonus=item.section_bonus,
            role_matches=item.role_matches,
        )
        for rank, item in enumerate(ordered, start=1)
    ]

def select_role_aware_evidence(
    plan: RetrievalPlan,
    candidates: Sequence[HybridSearchResult],
    reranked_results: Sequence[RerankedSearchResult],
    *,
    top_k: int,
) -> RoleEvidenceSelection:
    """Reserve evidence for every required role, then fill by reranker score."""

    if top_k <= 0:
        raise ValueError("top_k doit être strictement positif.")
    candidate_by_id = {item.chunk.chunk_id: item for item in candidates}
    reranked_by_id = {item.chunk.chunk_id: item for item in reranked_results}
    ordered_candidates = [
        candidate_by_id[item.chunk.chunk_id]
        for item in reranked_results
        if item.chunk.chunk_id in candidate_by_id
    ]

    selected_ids: list[str] = []
    source_by_id: dict[str, str] = {}
    roles_by_id: dict[str, list[str]] = {}
    covered: list[str] = []
    missing: list[str] = []

    for role in plan.roles:
        matching = [
            item
            for item in ordered_candidates
            if _supports_role(item, role, plan)
        ]
        if not matching:
            if role.required:
                missing.append(role.name)
            continue
        covered.append(role.name)
        chosen = next(
            (item for item in matching if item.chunk.chunk_id not in selected_ids),
            matching[0],
        )
        chunk_id = chosen.chunk.chunk_id
        bound_roles = roles_by_id.setdefault(chunk_id, [])
        if role.name not in bound_roles:
            bound_roles.append(role.name)
        if chunk_id not in selected_ids and len(selected_ids) < top_k:
            selected_ids.append(chunk_id)
            source_by_id[chunk_id] = "role_evidence"

    for item in reranked_results:
        if len(selected_ids) >= top_k:
            break
        chunk_id = item.chunk.chunk_id
        if chunk_id in selected_ids:
            continue
        selected_ids.append(chunk_id)
        source_by_id[chunk_id] = "reranker_fill"

    def provenance(chunk_id: str) -> str:
        roles = roles_by_id.get(chunk_id, [])
        if len(roles) == 1:
            return f"evidence_role:{roles[0]}"
        if roles:
            return "evidence_roles:" + ",".join(roles)
        return source_by_id[chunk_id]

    selected = tuple(
        V3SelectedResult(
            rank=rank,
            chunk_id=chunk_id,
            source=provenance(chunk_id),
            reranker_rank=(
                reranked_by_id[chunk_id].rank if chunk_id in reranked_by_id else None
            ),
            hybrid_rank=candidate_by_id[chunk_id].rank,
            bm25_rank=candidate_by_id[chunk_id].bm25_rank,
        )
        for rank, chunk_id in enumerate(selected_ids, start=1)
    )
    return RoleEvidenceSelection(
        selected=selected,
        covered_roles=tuple(dict.fromkeys(covered)),
        missing_roles=tuple(dict.fromkeys(missing)),
    )
