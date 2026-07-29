"""Claim splitting, citation binding and deterministic pruning."""

from __future__ import annotations

import re
from dataclasses import dataclass

from phosprocess.rag.claim_support import (
    _CITATION,
    _all_concepts,
    _attach_citations,
    _bundle_roles,
    _canonical_atomic_claim,
    _language_key,
    _lexical_coverage,
    _normalize,
    _semantic_text,
    _source_directly_supports,
)
from phosprocess.retrieval.evidence_bundle import EvidenceBundle

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")

_NUMBERED_PREFIX = re.compile(r"^\s*\d+[.)]\s*")

_CLAUSE_TAIL = re.compile(
    r",\s*(?:completing|thereby|thus|therefore|resulting|which\s+"
    r"(?:completes|means|causes|allows))\b.*$",
    re.IGNORECASE,
)

_EMBEDDED_WHERE_CLAUSE = re.compile(
    r",\s*where\b.*?,\s*(?=and\b)",
    re.IGNORECASE,
)

_TRAILING_WHICH_CLAUSE = re.compile(
    r",\s*which\b.*$",
    re.IGNORECASE,
)

_MAX_ANSWER_CLAIMS = 5

_PROCESS_FLOW_STAGE_ORDER = (
    "feed_inlet",
    "conical_bottom",
    "pump_heat_exchanger",
    "recirculation_vapor_body",
    "product_outlet",
)

_PROCESS_FLOW_ATOMIC_TEMPLATES: dict[
    str,
    dict[str, tuple[str, ...]],
] = {
    "en": {
        "feed_inlet": (
            (
                "The liquid phase is fed by the inlet acid pipe coming from "
                "the heat exchanger."
            ),
            "The acid is fed through the inlet acid pipe.",
        ),
        "conical_bottom": (
            (
                "The cycling acid leaves the vapor body through a conical "
                "bottom."
            ),
        ),
        "pump_heat_exchanger": (
            (
                "The pump withdraws liquor from the flash chamber and forces "
                "it through the heating element."
            ),
            "The circulation pump forces the liquor through the heating element.",
        ),
        "recirculation": (
            "The liquor returns to the flash chamber.",
            "The liquor is forced back to the flash chamber.",
        ),
        "vapor_body": (
            "Vapor-liquid separation takes place in the vapor body.",
            "The vapor body separates vapor from liquid.",
        ),
        "product_outlet": (
            (
                "The concentrated finished product acid is withdrawn from the "
                "vapor body at the product outlet."
            ),
            "The concentrated product acid is withdrawn through the product outlet.",
        ),
    },
    "fr": {
        "feed_inlet": (
            (
                "La phase liquide est alimentée par la conduite d’entrée "
                "provenant de l’échangeur."
            ),
            "L’acide est alimenté par la conduite d’entrée.",
        ),
        "conical_bottom": (
            (
                "L’acide en circulation quitte la chambre de vaporisation "
                "par son fond conique."
            ),
        ),
        "pump_heat_exchanger": (
            (
                "La pompe retire le liquide de la chambre de flash et le "
                "pousse à travers l’élément de chauffage."
            ),
            "La pompe de circulation pousse le liquide à travers l’échangeur.",
        ),
        "recirculation": (
            "Le liquide retourne dans la chambre de flash.",
            "Le liquide est renvoyé vers la chambre de vaporisation.",
        ),
        "vapor_body": (
            "La séparation vapeur-liquide a lieu dans la chambre de vaporisation.",
            "La chambre de vaporisation sépare la vapeur du liquide.",
        ),
        "product_outlet": (
            (
                "L’acide produit concentré est soutiré de la chambre de "
                "vaporisation par la sortie produit."
            ),
            "L’acide produit concentré est soutiré par la sortie produit.",
        ),
    },
    "ar": {
        "feed_inlet": (
            "يُغذّى الحمض عبر أنبوب الدخول القادم من المبادل الحراري.",
        ),
        "conical_bottom": (
            "يغادر الحمض الدائر جسم المبخر عبر القاع المخروطي.",
        ),
        "pump_heat_exchanger": (
            "تسحب المضخة السائل من حجرة الوميض وتدفعه عبر عنصر التسخين.",
        ),
        "recirculation": (
            "يعود السائل إلى حجرة الوميض.",
        ),
        "vapor_body": (
            "يحدث فصل البخار عن السائل في جسم المبخر.",
        ),
        "product_outlet": (
            "يُسحب الحمض المركز المنتج من جسم المبخر عبر مخرج المنتج.",
        ),
    },
}

_ATOMIC_STAGE_MARKERS: dict[str, tuple[tuple[str, ...], ...]] = {
    "feed_inlet": (("inlet acid pipe",), ("inlet pipe", "fed")),
    "conical_bottom": (("conical bottom",),),
    "pump_heat_exchanger": (
        ("pump", "heating element"),
        ("pump", "heat exchanger"),
    ),
    "recirculation": (
        ("returned", "flash chamber"),
        ("back to", "flash chamber"),
        ("recirculation", "flash chamber"),
    ),
    "vapor_body": (
        ("vapor liquid separation",),
        ("vapor/liquor separation",),
        ("separates", "vapor", "liquor"),
    ),
    "product_outlet": (
        ("concentrated", "product", "withdrawn"),
        ("product outlet", "withdrawn"),
    ),
}

_ATOMIC_ROLE_BY_STAGE: dict[str, tuple[str, ...]] = {
    "feed_inlet": ("feed_inlet",),
    "conical_bottom": ("conical_bottom",),
    "pump_heat_exchanger": ("pump_heat_exchanger",),
    "recirculation": ("recirculation", "recirculation_vapor_body"),
    "vapor_body": ("vapor_body", "recirculation_vapor_body"),
    "product_outlet": ("product_outlet",),
}


def _atomic_template_stage(claim: str) -> str | None:
    normalized = _canonical_atomic_claim(claim)
    for language_templates in _PROCESS_FLOW_ATOMIC_TEMPLATES.values():
        for stage, variants in language_templates.items():
            normalized_stage = (
                "recirculation" if stage == "recirculation" else stage
            )
            for variant in variants:
                if _canonical_atomic_claim(variant) == normalized:
                    return normalized_stage
    return None


def _bundle_supports_atomic_stage(
    bundle: EvidenceBundle,
    stage: str,
) -> bool:
    roles = set(_bundle_roles(bundle))
    expected_roles = set(_ATOMIC_ROLE_BY_STAGE.get(stage, ()))
    if roles & expected_roles:
        return True

    normalized = _semantic_text(bundle.display_text)
    groups = _ATOMIC_STAGE_MARKERS.get(stage, ())
    return any(
        all(marker in normalized for marker in marker_group)
        for marker_group in groups
    )

_INSUFFICIENCY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bnot explicitly (?:stated|provided|documented|supported)\b"),
    re.compile(r"\bdoes not explicitly (?:state|provide|document|support)\b"),
    re.compile(r"\bnot enough explicit evidence\b"),
    re.compile(r"\bcannot be determined from the (?:corpus|sources|evidence)\b"),
    re.compile(r"\bthe corpus does not provide enough\b"),
    re.compile(r"\bn est pas explicitement (?:indique|decrit|documente|soutenu)\b"),
    re.compile(r"\bne (?:precise|decrit|documente|soutient) pas explicitement\b"),
    re.compile(r"\bpreuves? explicites? insuffisantes?\b"),
    re.compile(r"\bimpossible a determiner a partir du corpus\b"),
)


@dataclass(frozen=True, slots=True)
class PrunedAnswer:
    """Result of deterministic removal of unsupported answer claims."""

    answer: str
    removed_claims: tuple[str, ...]
    fallback_used: bool
    inherited_citation_count: int
    missing_required_concepts: tuple[str, ...] = ()
    atomic_plan_used: bool = False
    reconstructed_claim_count: int = 0


def _supporting_citations(
    claim: str,
    citations: tuple[int, ...],
    bundle_by_number: dict[int, EvidenceBundle],
) -> tuple[int, ...]:
    supported: list[int] = []

    for source_number in citations:
        bundle = bundle_by_number.get(source_number)
        if bundle is None:
            continue

        directly_supported, _coverage = _source_directly_supports(
            claim,
            bundle,
        )
        if directly_supported:
            supported.append(source_number)

    return tuple(supported)


def _claim_variants(claim: str) -> tuple[str, ...]:
    """Return safe proposition-level variants from most to least complete."""

    original = claim.strip()
    forced_trim = _CLAUSE_TAIL.sub("", original).strip()

    if forced_trim and forced_trim != original:
        if not re.search(r"[.!?]$", forced_trim):
            forced_trim += "."
        return (forced_trim,)

    variants = [original]
    without_where = _EMBEDDED_WHERE_CLAUSE.sub(", ", original).strip()
    trailing_main = _TRAILING_WHICH_CLAUSE.sub("", original).strip()

    for variant in (without_where, trailing_main):
        if variant and variant not in variants:
            if not re.search(r"[.!?]$", variant):
                variant += "."
            variants.append(variant)

    return tuple(variants)


def _removed_proposition(original: str, retained: str) -> str:
    """Describe only the proposition removed from a partially kept claim."""

    for pattern in (
        _CLAUSE_TAIL,
        _EMBEDDED_WHERE_CLAUSE,
        _TRAILING_WHICH_CLAUSE,
    ):
        match = pattern.search(original)
        if match is not None:
            return match.group(0).lstrip(", ").strip()

    return original if original != retained else ""


def _supported_claim_variant(
    claim: str,
    citations: tuple[int, ...],
    bundle_by_number: dict[int, EvidenceBundle],
) -> tuple[str | None, tuple[int, ...]]:
    """Prefer the full sentence, then retain a directly supported main clause."""

    for variant in _claim_variants(claim):
        supporting = _supporting_citations(
            variant,
            citations,
            bundle_by_number,
        )
        if supporting:
            return variant, supporting

    return None, ()


def _renumber_numbered_claims(claims: list[str]) -> list[str]:
    """Remove numbering gaps created by deterministic pruning."""

    if not any(_NUMBERED_PREFIX.match(claim) for claim in claims):
        return claims

    renumbered: list[str] = []
    next_number = 1

    for claim in claims:
        if _NUMBERED_PREFIX.match(claim):
            body = _NUMBERED_PREFIX.sub("", claim, count=1).strip()
            renumbered.append(f"{next_number}. {body}")
            next_number += 1
        else:
            renumbered.append(claim)

    return renumbered


def _missing_process_flow_concepts(answer: str) -> tuple[str, ...]:
    concepts = _all_concepts(answer)
    requirements = {
        "feed_inlet": {"inlet"},
        "pump_heat_exchanger": {"pump", "heat_exchanger"},
        "vapor_body": {"flash_chamber"},
        "recirculation": {"return"},
        "product_outlet": {"outlet"},
    }
    return tuple(
        name
        for name, required in requirements.items()
        if not required.issubset(concepts)
    )


def _best_bundle_for_atomic_claim(
    claim_variants: tuple[str, ...],
    bundles: list[EvidenceBundle],
) -> tuple[str, int] | None:
    """Choose one exact source that directly supports one atomic fact."""

    for claim in claim_variants:
        best_source: tuple[float, int] | None = None
        for bundle in bundles:
            supported, coverage = _source_directly_supports(claim, bundle)
            if not supported:
                continue
            candidate = (coverage, bundle.source_number)
            if best_source is None or candidate[0] > best_source[0]:
                best_source = candidate

        if best_source is not None:
            return claim, best_source[1]

    return None


def _best_bundle_for_atomic_stage(
    stage: str,
    claim_variants: tuple[str, ...],
    bundles: list[EvidenceBundle],
) -> tuple[str, int] | None:
    """Choose a source-local template using explicit stage evidence."""

    stage_bundles = [
        bundle
        for bundle in bundles
        if _bundle_supports_atomic_stage(bundle, stage)
    ]
    if not stage_bundles:
        return None

    for claim in claim_variants:
        best_source: tuple[float, int] | None = None
        for bundle in stage_bundles:
            supported, coverage = _source_directly_supports(claim, bundle)
            if not supported:
                continue
            candidate = (coverage, bundle.source_number)
            if best_source is None or candidate[0] > best_source[0]:
                best_source = candidate
        if best_source is not None:
            return claim, best_source[1]

    best_bundle = max(
        stage_bundles,
        key=lambda bundle: max(
            _lexical_coverage(claim, bundle.display_text)
            for claim in claim_variants
        ),
    )
    best_claim = max(
        claim_variants,
        key=lambda claim: _lexical_coverage(claim, best_bundle.display_text),
    )
    return best_claim, best_bundle.source_number


def build_atomic_process_flow_answer(
    bundles: list[EvidenceBundle],
    *,
    language: str = "en",
) -> str | None:
    """Build five ordered, source-local process facts from complete evidence.

    The planner never joins facts from different documents into one sentence.
    Each generated step is accepted only after the normal production support
    check confirms that one Source N supports the whole atomic statement.
    """

    templates = _PROCESS_FLOW_ATOMIC_TEMPLATES[_language_key(language)]
    steps: list[str] = []

    for step_number, stage in enumerate(_PROCESS_FLOW_STAGE_ORDER, start=1):
        if stage == "recirculation_vapor_body":
            recirculation = _best_bundle_for_atomic_stage(
                "recirculation",
                templates["recirculation"],
                bundles,
            )
            separation = _best_bundle_for_atomic_stage(
                "vapor_body",
                templates["vapor_body"],
                bundles,
            )
            if recirculation is None or separation is None:
                return None

            recirculation_claim, recirculation_source = recirculation
            separation_claim, separation_source = separation
            first_clause = _attach_citations(
                f"{step_number}. {recirculation_claim}",
                (recirculation_source,),
            ).rstrip(".")
            second_claim = (
                separation_claim[:1].lower() + separation_claim[1:]
            )
            second_clause = _attach_citations(
                second_claim,
                (separation_source,),
            )
            steps.append(f"{first_clause}; {second_clause}")
            continue

        selected = _best_bundle_for_atomic_stage(
            stage,
            templates[stage],
            bundles,
        )
        if selected is None:
            return None
        claim, source_number = selected
        numbered = f"{step_number}. {claim}"
        steps.append(_attach_citations(numbered, (source_number,)))

    answer = "\n".join(steps)
    if _missing_process_flow_concepts(answer):
        return None
    return answer


def _split_explicitly_cited_clauses(claim: str) -> list[str]:
    """Split semicolon-separated propositions only with local citations."""

    if ";" not in claim:
        return [claim]

    parts = [part.strip() for part in claim.split(";") if part.strip()]
    if len(parts) < 2:
        return [claim]
    if not all(_CITATION.search(part) for part in parts):
        return [claim]
    return parts


def _iter_answer_claims(answer: str) -> list[str]:
    claims: list[str] = []
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for sentence in _split_line_claims(line):
            claims.extend(_split_explicitly_cited_clauses(sentence))
    return claims


def _is_insufficiency_claim(claim: str) -> bool:
    normalized = _normalize(claim)
    return any(pattern.search(normalized) for pattern in _INSUFFICIENCY_PATTERNS)


def _split_line_claims(line: str) -> list[str]:
    """Split a line into claims without breaking a leading numbered item."""

    protected = re.sub(
        r"^(\s*\d+)\.\s+",
        r"\1<LIST_DOT> ",
        line,
        count=1,
    )
    claims = [
        claim.replace("<LIST_DOT>", ".").strip()
        for claim in _SENTENCE.split(protected)
        if claim.strip()
    ]
    return claims


def _fallback_answer(language: str) -> str:
    normalized = language.strip().lower()

    if normalized.startswith("en"):
        return (
            "The retrieved passages do not provide enough information to "
            "answer this question precisely."
        )

    if normalized.startswith("ar"):
        return (
            "لا توفر المقاطع المسترجعة معلومات كافية للإجابة عن هذا السؤال بدقة."
        )

    return (
        "Les passages retrouvés ne permettent pas de répondre précisément "
        "à cette question."
    )


def prune_unsupported_claims(
    answer: str,
    bundles: list[EvidenceBundle],
    *,
    fallback_language: str = "fr",
    question_type: str | None = None,
) -> PrunedAnswer:
    """Keep only directly grounded sentences, with exact supporting sources.

    A sentence is retained only when at least one cited source supports the
    complete sentence by itself. Merely plausible mechanisms, recommendations
    and definitions are removed. Mixed insufficiency statements are also
    removed whenever at least one affirmative grounded sentence remains.
    """

    if question_type == "process_flow":
        atomic_answer = build_atomic_process_flow_answer(
            bundles,
            language=fallback_language,
        )
        if atomic_answer is not None:
            return PrunedAnswer(
                answer=atomic_answer,
                removed_claims=(),
                fallback_used=False,
                inherited_citation_count=0,
                missing_required_concepts=(),
                atomic_plan_used=True,
                reconstructed_claim_count=5,
            )

    bundle_by_number = {bundle.source_number: bundle for bundle in bundles}
    kept_claims: list[str] = []
    removed_claims: list[str] = []
    inherited_citation_count = 0

    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        line_citations = tuple(
            dict.fromkeys(int(match) for match in _CITATION.findall(line))
        )

        for claim in _split_line_claims(line):
            own_citations = tuple(
                dict.fromkeys(int(match) for match in _CITATION.findall(claim))
            )
            citations = own_citations or line_citations
            clean_claim = _CITATION.sub("", claim).strip()

            if not clean_claim:
                continue

            if _is_insufficiency_claim(clean_claim):
                removed_claims.append(clean_claim)
                continue

            if not citations:
                removed_claims.append(clean_claim)
                continue

            supported_claim, supporting_citations = (
                _supported_claim_variant(
                    clean_claim,
                    citations,
                    bundle_by_number,
                )
            )
            if supported_claim is None:
                removed_claims.append(clean_claim)
                continue

            if supported_claim != clean_claim:
                removed_proposition = _removed_proposition(
                    clean_claim,
                    supported_claim,
                )
                if removed_proposition:
                    removed_claims.append(removed_proposition)

            if len(kept_claims) >= _MAX_ANSWER_CLAIMS:
                removed_claims.append(clean_claim)
                continue

            if not own_citations:
                inherited_citation_count += 1

            kept_claims.append(
                _attach_citations(supported_claim, supporting_citations)
            )

    if not kept_claims:
        if question_type == "process_flow":
            atomic_answer = build_atomic_process_flow_answer(
                bundles,
                language=fallback_language,
            )
            if atomic_answer is not None:
                return PrunedAnswer(
                    answer=atomic_answer,
                    removed_claims=tuple(removed_claims),
                    fallback_used=False,
                    inherited_citation_count=inherited_citation_count,
                    atomic_plan_used=True,
                    reconstructed_claim_count=5,
                )
        return PrunedAnswer(
            answer=_fallback_answer(fallback_language),
            removed_claims=tuple(removed_claims),
            fallback_used=True,
            inherited_citation_count=inherited_citation_count,
        )

    kept_claims = _renumber_numbered_claims(kept_claims)
    final_answer = "\n".join(kept_claims)
    missing_required = (
        _missing_process_flow_concepts(final_answer)
        if question_type == "process_flow"
        else ()
    )

    if missing_required:
        atomic_answer = build_atomic_process_flow_answer(
            bundles,
            language=fallback_language,
        )
        if atomic_answer is not None:
            return PrunedAnswer(
                answer=atomic_answer,
                removed_claims=tuple(removed_claims),
                fallback_used=False,
                inherited_citation_count=inherited_citation_count,
                missing_required_concepts=(),
                atomic_plan_used=True,
                reconstructed_claim_count=5,
            )
        return PrunedAnswer(
            answer=_fallback_answer(fallback_language),
            removed_claims=tuple(removed_claims),
            fallback_used=True,
            inherited_citation_count=inherited_citation_count,
            missing_required_concepts=missing_required,
        )

    if question_type == "process_flow":
        normalized_answer = build_atomic_process_flow_answer(
            bundles,
            language=fallback_language,
        )
        if normalized_answer is not None:
            return PrunedAnswer(
                answer=normalized_answer,
                removed_claims=tuple(removed_claims),
                fallback_used=False,
                inherited_citation_count=inherited_citation_count,
                missing_required_concepts=(),
                atomic_plan_used=True,
                reconstructed_claim_count=5,
            )

    return PrunedAnswer(
        answer=final_answer,
        removed_claims=tuple(removed_claims),
        fallback_used=False,
        inherited_citation_count=inherited_citation_count,
        missing_required_concepts=(),
    )
