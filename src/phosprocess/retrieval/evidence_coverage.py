"""Strict hierarchy filtering and mandatory evidence coverage."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from phosprocess.retrieval.hierarchical import SectionSearchResponse
from phosprocess.retrieval.v3_selection import V3SelectedResult

_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "process_flow": (
        "feed_inlet",
        "conical_bottom",
        "pump_heat_exchanger",
        "vapor_body",
        "recirculation",
        "product_outlet",
    ),
}


def required_evidence_keys(question_type: str) -> tuple[str, ...]:
    """Return the ordered mandatory evidence keys for one question type."""

    return _REQUIREMENTS.get(question_type, ())


def provenance_with_evidence_roles(
    source: str,
    roles: Sequence[str],
) -> str:
    """Attach ordered evidence roles without discarding selection provenance."""

    role_set = {role.strip() for role in roles if role.strip()}
    if not role_set:
        return source

    process_order = _REQUIREMENTS["process_flow"]
    ordered_roles = [role for role in process_order if role in role_set]
    ordered_roles.extend(sorted(role_set - set(ordered_roles)))

    provenance_parts = [
        part.strip()
        for part in source.split(";")
        if part.strip()
        and not part.strip().startswith(("evidence_role:", "evidence_roles:"))
    ]
    provenance_parts.append("evidence_roles:" + ",".join(ordered_roles))
    return ";".join(provenance_parts)


_LOW_VALUE = re.compile(
    r"\b(?:table of contents|contents|list of figures|liste des figures|"
    r"list of abbreviations|liste des abr[eé]viations|bibliography|"
    r"references|subject index|author index)\b",
    re.I,
)

_EXCLUDED_BY_TYPE: dict[str, re.Pattern[str]] = {
    "process_flow": re.compile(
        r"\b(?:agitation flow in a reaction tank|wet grinding|"
        r"hemihydrate processes?|crystallization equipment|"
        r"filter piping|scaling of filter piping)\b",
        re.I,
    ),
    "troubleshooting": re.compile(
        r"\b(?:wet grinding|hemihydrate processes?|filter piping|"
        r"scaling of filter piping)\b",
        re.I,
    ),
}


class EvidenceCoverageError(ValueError):
    """Raised before generation when mandatory evidence is incomplete."""


@dataclass(frozen=True, slots=True)
class EvidenceCoverage:
    """Coverage trace for one retrieval turn."""

    question_type: str
    required: tuple[str, ...]
    covered: tuple[str, ...]
    missing: tuple[str, ...]
    chunk_ids_by_requirement: dict[str, tuple[str, ...]]

    @property
    def complete(self) -> bool:
        return not self.missing


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _has_any(text: str, *terms: str) -> bool:
    return any(term in text for term in terms)


def _has_p2o5(text: str) -> bool:
    compact = re.sub(r"[^a-z0-9^]+", "", text)

    return _has_any(
        compact,
        "p2o5",
        "p205",
        "p9o5",
        "pgo5",
        "pgo^",
    ) or "phosphoric acid" in text


def _has_feed_inlet(text: str) -> bool:
    """Detect an explicit feed-entry statement without using query text."""

    if _has_any(
        text,
        "feed inlet",
        "acid inlet",
        "feed connection",
        "feed nozzle",
        "feed line",
        "feed pipe",
        "acid feed",
        "feed acid",
        "weak acid feed",
        "dilute acid feed",
        "feed liquor",
        "liquor feed",
        "feed is admitted",
        "feed enters",
        "feed is introduced",
        "acid is admitted",
        "acid enters",
        "acid is fed",
        "acid is introduced",
        "incoming acid",
        "alimentation en acide",
        "entrée de l'acide",
        "entree de l'acide",
        "arrivée de l'acide",
        "arrivee de l'acide",
        "ligne d'alimentation",
        "conduite d'alimentation",
        "acide est introduit",
        "acide est alimenté",
        "acide est alimente",
        "acide entre",
    ):
        return True

    has_feed_stream = _has_any(
        text,
        "feed",
        "weak acid",
        "dilute acid",
        "weak phosphoric acid",
        "dilute phosphoric acid",
        "fresh acid",
        "acide faible",
        "acide dilué",
        "acide dilue",
        "acide phosphorique dilué",
        "acide phosphorique dilue",
    )
    has_entry_action = _has_any(
        text,
        "inlet",
        "enters",
        "is fed",
        "fed to",
        "introduced into",
        "admitted to",
        "supplied to",
        "added to",
        "joins the circulation",
        "entry",
        "entrée",
        "entree",
        "arrivée",
        "arrivee",
        "introduit dans",
        "alimenté vers",
        "alimente vers",
        "admis dans",
        "entre dans",
    )
    has_equipment_target = _has_any(
        text,
        "evaporator",
        "circulation loop",
        "circulation system",
        "vapor body",
        "vapour body",
        "flash chamber",
        "evaporation chamber",
        "évaporateur",
        "evaporateur",
        "circuit de circulation",
        "chambre de flash",
    )

    return has_feed_stream and has_entry_action and has_equipment_target


def _has_product_outlet(text: str) -> bool:
    """Detect an explicit concentrated-product exit statement."""

    if _has_any(
        text,
        "product outlet",
        "product acid outlet",
        "concentrated acid outlet",
        "strong acid outlet",
        "product draw-off",
        "product draw off",
        "product take-off",
        "product take off",
        "product withdrawal",
        "withdrawal of product",
        "product is withdrawn",
        "product acid is withdrawn",
        "concentrated acid is withdrawn",
        "concentrated phosphoric acid is withdrawn",
        "strong acid is withdrawn",
        "product is drawn off",
        "product acid is drawn off",
        "concentrated acid is drawn off",
        "sortie produit",
        "sortie de l'acide concentré",
        "sortie de l'acide concentre",
        "soutirage du produit",
        "soutirage de l'acide concentré",
        "soutirage de l'acide concentre",
        "acide concentré est soutiré",
        "acide concentre est soutire",
    ):
        return True

    has_product_stream = _has_any(
        text,
        "product acid",
        "acid product",
        "concentrated acid",
        "concentrated phosphoric acid",
        "strong acid",
        "evaporator concentrate",
        "final product",
        "acide produit",
        "acide concentré",
        "acide concentre",
        "produit concentré",
        "produit concentre",
    )
    has_exit_action = _has_any(
        text,
        "outlet",
        "withdrawn",
        "drawn off",
        "draw-off",
        "draw off",
        "taken off",
        "leaves",
        "exits",
        "discharged",
        "pumped out",
        "sent to storage",
        "pumped to storage",
        "goes to storage",
        "sortie",
        "soutiré",
        "soutire",
        "évacué",
        "evacue",
        "sort de",
        "envoyé au stockage",
        "envoye au stockage",
        "pompé au stockage",
        "pompe au stockage",
    )
    has_process_context = _has_any(
        text,
        "evaporator",
        "circulation loop",
        "recirculation loop",
        "vapor body",
        "vapour body",
        "flash chamber",
        "separator",
        "product line",
        "storage tank",
        "storage",
        "évaporateur",
        "evaporateur",
        "boucle de circulation",
        "chambre de flash",
        "séparateur",
        "separateur",
        "ligne produit",
        "stockage",
        "bac de stockage",
    )

    return has_product_stream and has_exit_action and has_process_context


def coverage_keys_for_text(
    text: str,
    question_type: str,
) -> frozenset[str]:
    """Return mandatory evidence aspects explicitly present in text."""

    value = _normalize(text)
    keys: set[str] = set()

    if question_type == "process_flow":
        if _has_feed_inlet(value):
            keys.add("feed_inlet")

        if _has_any(
            value,
            "conical bottom",
            "cone bottom",
            "fond conique",
            "القاع المخروطي",
        ):
            keys.add("conical_bottom")

        has_pump = _has_any(
            value,
            "circulation pump",
            "circulating pump",
            "pump withdraws",
            "acid circulation pump",
            "pompe de circulation",
            "pompe",
        )

        has_heater = _has_any(
            value,
            "heat exchanger",
            "heating element",
            "heater",
            "tube-and-shell",
            "shell-and-tube",
            "échangeur",
            "echangeur",
        )

        if has_pump and has_heater:
            keys.add("pump_heat_exchanger")

        if _has_any(
            value,
            "vapor body",
            "vapour body",
            "flash chamber",
            "boiler chamber",
            "vapor separator",
            "vapour separator",
            "chambre de flash",
            "séparateur",
            "separateur",
            "bouilleur",
        ):
            keys.add("vapor_body")

        if _has_any(
            value,
            "recirculation line",
            "returned to the body",
            "back to the flash chamber",
            "recirculation",
            "recycle line",
            "acid cycle",
            "retour",
            "recyclage",
        ):
            keys.add("recirculation")

        if _has_product_outlet(value):
            keys.add("product_outlet")

    elif question_type == "balance":
        has_mass_balance = _has_any(
            value,
            "mass balance",
            "material balance",
            "conservation of mass",
            "bilan de matière",
            "bilan matiere",
            "bilan massique",
        )

        has_flow_terms = (
            _has_any(value, "mass in", "input", "feed")
            and _has_any(value, "mass out", "output", "product")
        ) or "accumulation" in value

        if has_mass_balance and has_flow_terms:
            keys.add("overall_mass_balance")

        if _has_p2o5(value) and _has_any(
            value,
            "p2o5 balance",
            "p₂o₅ balance",
            "mass balance",
            "material balance",
            "component balance",
            "bilan p2o5",
            "bilan de matière",
            "bilan matiere",
            "bilan massique",
        ):
            keys.add("p2o5_balance")

        has_explicit_energy_balance = _has_any(
            value,
            "energy balance",
            "heat balance",
            "enthalpy balance",
            "bilan énergétique",
            "bilan energetique",
            "bilan thermique",
        )

        has_energy_equation = _has_any(
            value,
            "steam consumption",
            "enthalpy",
            "heat",
        ) and _has_any(
            value,
            "equation",
            "balance",
            "bilan",
        )

        if has_explicit_energy_balance or has_energy_equation:
            keys.add("energy_balance")

    return frozenset(keys)


def _section_text(result: Any) -> str:
    section = result.section

    return "\n".join(
        (
            section.hierarchy_path,
            section.display_text,
            section.bm25_text,
        )
    )


def _section_is_allowed(
    result: Any,
    *,
    question_type: str,
    query: str,
) -> bool:
    path = result.section.hierarchy_path

    if _LOW_VALUE.search(path):
        return False

    excluded = _EXCLUDED_BY_TYPE.get(question_type)

    if excluded is not None and excluded.search(path):
        return False

    if (
        "evaporator" in query.casefold()
        and re.search(r"\bcompressors?\b", path, re.I)
    ):
        return False

    return True


def select_strict_sections(
    response: SectionSearchResponse,
    *,
    question_type: str,
    top_k: int = 3,
) -> SectionSearchResponse:
    """Keep at most three sections while favoring uncovered aspects."""

    eligible = [
        item
        for item in response.results
        if _section_is_allowed(
            item,
            question_type=question_type,
            query=response.query,
        )
    ]

    if not eligible:
        raise EvidenceCoverageError(
            "Aucune section pertinente après le filtre "
            "hiérarchique strict."
        )

    selected: list[Any] = []
    selected_ids: set[str] = set()
    covered: set[str] = set()

    while len(selected) < top_k:
        remaining = [
            item
            for item in eligible
            if item.section.section_id not in selected_ids
        ]

        if not remaining:
            break

        best = max(
            remaining,
            key=lambda item: (
                len(
                    coverage_keys_for_text(
                        _section_text(item),
                        question_type,
                    )
                    - covered
                ),
                float(item.final_score),
                -int(item.rank),
            ),
        )

        selected.append(best)
        selected_ids.add(best.section.section_id)

        covered.update(
            coverage_keys_for_text(
                _section_text(best),
                question_type,
            )
        )

    ranked = tuple(
        replace(item, rank=rank)
        for rank, item in enumerate(selected, start=1)
    )

    allowed = frozenset(
        chunk_id
        for result in ranked
        for chunk_id in result.section.child_chunk_ids
    )

    if not allowed:
        raise EvidenceCoverageError(
            "Les trois sections retenues ne contiennent aucun chunk."
        )

    return replace(
        response,
        results=ranked,
        allowed_chunk_ids=allowed,
    )


def evaluate_evidence_coverage_texts(
    texts: Sequence[tuple[str, str]],
    *,
    question_type: str,
) -> EvidenceCoverage:
    """Evaluate mandatory aspects against arbitrary evidence text windows."""

    required = required_evidence_keys(question_type)
    matches: dict[str, list[str]] = {key: [] for key in required}

    for chunk_id, text in texts:
        for key in coverage_keys_for_text(text, question_type):
            if key in matches:
                matches[key].append(chunk_id)

    covered = tuple(key for key in required if matches[key])
    missing = tuple(key for key in required if not matches[key])

    return EvidenceCoverage(
        question_type=question_type,
        required=tuple(required),
        covered=covered,
        missing=missing,
        chunk_ids_by_requirement={
            key: tuple(chunk_ids) for key, chunk_ids in matches.items()
        },
    )


def evaluate_evidence_coverage(
    chunks: Sequence[Any],
    *,
    question_type: str,
) -> EvidenceCoverage:
    """Evaluate mandatory aspects against selected child chunks."""

    return evaluate_evidence_coverage_texts(
        [
            (
                chunk.chunk_id,
                f"{chunk.hierarchy_path}\n{chunk.display_text}",
            )
            for chunk in chunks
        ],
        question_type=question_type,
    )


def select_coverage_aware_evidence(
    initial_selected: Sequence[V3SelectedResult],
    *,
    candidates: Sequence[Any],
    reranked_results: Sequence[Any],
    child_by_id: dict[str, Any],
    question_type: str,
    top_k: int,
    coverage_text_by_id: dict[str, str] | None = None,
) -> tuple[list[V3SelectedResult], EvidenceCoverage]:
    """Choose chunks and stop before generation if coverage is incomplete."""

    candidate_by_id = {
        item.chunk.chunk_id: item
        for item in candidates
    }

    reranked_by_id = {
        item.chunk.chunk_id: item
        for item in reranked_results
    }

    initial_by_id = {
        item.chunk_id: item
        for item in initial_selected
    }

    required = set(
        _REQUIREMENTS.get(question_type, ())
    )

    selected: list[V3SelectedResult] = []
    selected_ids: set[str] = set()
    covered: set[str] = set()

    def coverage_text(chunk_id: str) -> str:
        if coverage_text_by_id is not None and chunk_id in coverage_text_by_id:
            return coverage_text_by_id[chunk_id]

        child = child_by_id[chunk_id]
        return f"{child.hierarchy_path}\n{child.display_text}"

    def add(
        chunk_id: str,
        source: str,
    ) -> None:
        if chunk_id in selected_ids:
            return

        if len(selected) >= top_k:
            return

        reranked = reranked_by_id[chunk_id]
        candidate = candidate_by_id[chunk_id]
        original = initial_by_id.get(chunk_id)
        evidence_roles = coverage_keys_for_text(
            coverage_text(chunk_id),
            question_type,
        )
        base_source = original.source if original is not None else source
        annotated_source = provenance_with_evidence_roles(
            base_source,
            evidence_roles,
        )

        if original is not None:
            item = replace(
                original,
                rank=len(selected) + 1,
                source=annotated_source,
            )
        else:
            item = V3SelectedResult(
                rank=len(selected) + 1,
                chunk_id=chunk_id,
                source=annotated_source,
                reranker_rank=reranked.rank,
                hybrid_rank=candidate.rank,
                bm25_rank=candidate.bm25_rank,
            )

        selected.append(item)
        selected_ids.add(chunk_id)
        covered.update(evidence_roles)

    while required - covered and len(selected) < top_k:
        best: Any | None = None
        best_gain = 0

        for reranked in reranked_results:
            chunk_id = reranked.chunk.chunk_id

            if chunk_id in selected_ids:
                continue

            keys = coverage_keys_for_text(
                coverage_text(chunk_id),
                question_type,
            )

            gain = len(
                keys & (required - covered)
            )

            if gain > best_gain:
                best = reranked
                best_gain = gain

        if best is None or best_gain == 0:
            break

        add(
            best.chunk.chunk_id,
            "coverage_guard",
        )

    for item in initial_selected:
        add(
            item.chunk_id,
            item.source,
        )

    for reranked in reranked_results:
        add(
            reranked.chunk.chunk_id,
            "strict_section_fill",
        )

    ranked = [
        replace(item, rank=rank)
        for rank, item in enumerate(
            selected,
            start=1,
        )
    ]

    coverage = evaluate_evidence_coverage_texts(
        [
            (item.chunk_id, coverage_text(item.chunk_id))
            for item in ranked
        ],
        question_type=question_type,
    )

    if not coverage.complete:
        raise EvidenceCoverageError(
            "Couverture documentaire incomplète "
            "avant génération : "
            + ", ".join(coverage.missing)
        )

    return ranked, coverage
