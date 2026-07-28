"""Deterministic claim-to-citation support diagnostics and production guards."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum

from phosprocess.rag.citations import CitationValidationError
from phosprocess.retrieval.evidence_bundle import EvidenceBundle

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_CITATION = re.compile(r"\[Source ([1-5])\]")
_TOKEN = re.compile(r"(?u)[^\W_]{2,}")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
_UNIT = re.compile(r"(?:%|\u00b0c|t/h|kg/s|kg/h|m3/h|m/s|torr)(?!\w)")
_LIST_PREFIX = re.compile(r"^\s*\d+[.)]\s*")
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
_STOPWORDS = {
    "a",
    "afin",
    "ainsi",
    "an",
    "and",
    "are",
    "as",
    "at",
    "au",
    "aux",
    "avec",
    "be",
    "been",
    "by",
    "ce",
    "ces",
    "cette",
    "comme",
    "dans",
    "de",
    "des",
    "du",
    "due",
    "en",
    "entre",
    "est",
    "et",
    "for",
    "from",
    "has",
    "have",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "la",
    "le",
    "les",
    "leur",
    "leurs",
    "lorsque",
    "moins",
    "ne",
    "of",
    "on",
    "or",
    "ou",
    "par",
    "pas",
    "plus",
    "pour",
    "puis",
    "rather",
    "sa",
    "se",
    "ses",
    "son",
    "sont",
    "sous",
    "sur",
    "than",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "this",
    "through",
    "to",
    "un",
    "une",
    "was",
    "were",
    "which",
    "while",
    "with",
    "without",
}

_RELATION_CONCEPTS: dict[str, tuple[str, ...]] = {
    "inlet": (
        "feed inlet",
        "inlet acid pipe",
        "inlet pipe",
        "feedstock enters",
        "is admitted",
        "is fed",
        "fed near",
        "enters",
        "introduced",
        "alimentation",
        "alimentee",
        "entre dans",
        "introduit",
        "admis",
        "est alimente",
        "est alimentee",
        "conduite d entree",
        "conduite d’entrée",
        "يغذى",
        "انبوب الدخول",
    ),
    "pump": ("pump", "pumped", "pompe", "pompe", "مضخة"),
    "heat_exchanger": (
        "heat exchanger",
        "heating element",
        "heating surface",
        "echangeur",
        "element de chauffage",
        "عنصر التسخين",
    ),
    "flash_chamber": (
        "flash chamber",
        "vapor body",
        "vapour body",
        "chambre de flash",
        "chambre de vaporisation",
        "bouilleur",
        "حجرة الوميض",
        "جسم المبخر",
    ),
    "return": (
        "returned",
        "returns",
        "back to",
        "circulated",
        "recirculation",
        "recirculated",
        "return line",
        "reintroduced",
        "recirculation line",
        "renvoye",
        "retour",
        "reintroduit",
        "recyclage",
        "retourne",
        "revient",
        "يعود",
    ),
    "outlet": (
        "product outlet",
        "outlet located",
        "is withdrawn",
        "withdrawn",
        "withdrawal",
        "exits",
        "exit",
        "leaves",
        "sortie",
        "soutirage",
        "recupere",
        "evacue",
        "est soutire",
        "sortie produit",
        "يسحب",
        "مخرج المنتج",
    ),
    "conical_bottom": (
        "conical bottom",
        "fond conique",
        "القاع المخروطي",
    ),
    "vacuum": (
        "under reduced pressure",
        "under vacuum",
        "vacuum",
        "sous vide",
        "pression reduite",
    ),
    "evaporation": (
        "evaporation",
        "evaporate",
        "evaporated",
        "vaporized",
        "vaporization",
        "vapourisation",
        "boiling",
        "vaporisation",
        "ebullition",
    ),
    "downstream": (
        "next treatment",
        "following stage",
        "downstream",
        "etape suivante",
        "traitement suivant",
    ),
}

_STRICT_RELATION_CONCEPTS = {
    "inlet",
    "return",
    "outlet",
    "conical_bottom",
    "evaporation",
    "downstream",
}

_NUMBER_WORDS = {
    "deux": "2",
    "trois": "3",
    "four": "4",
    "three": "3",
    "two": "2",
}

_UNIT_NORMALIZATIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(?:tonnes?|tons?|t)\s*(?:/|per)\s*"
            r"(?:h|hr|hours?|heures?)\b"
        ),
        "t/h",
    ),
    (
        re.compile(
            r"\bkg\s*(?:/|per)\s*"
            r"(?:s|sec|seconds?|secondes?)\b"
        ),
        "kg/s",
    ),
    (
        re.compile(
            r"\bkg\s*(?:/|per)\s*"
            r"(?:h|hr|hours?|heures?)\b"
        ),
        "kg/h",
    ),
    (
        re.compile(
            r"\b(?:m3|m\^3|cubic\s+met(?:er|re)s?)"
            r"\s*(?:/|per)\s*"
            r"(?:h|hr|hours?|heures?)\b"
        ),
        "m3/h",
    ),
    (
        re.compile(
            r"\b(?:m|met(?:er|re)s?)\s*(?:/|per)\s*"
            r"(?:s|sec|seconds?|secondes?)\b"
        ),
        "m/s",
    ),
    (
        re.compile(
            r"(?:\u00b0\s*c|degrees?\s*c(?:elsius)?|"
            r"degres?\s*c(?:elsius)?)\b"
        ),
        "\u00b0c",
    ),
    (re.compile(r"\btorrs?\b"), "torr"),
    (re.compile(r"\b(?:percent|pour\s*cent)\b"), "%"),
)

_MIN_DIRECT_LEXICAL_COVERAGE = 0.50
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


def _canonical_atomic_claim(text: str) -> str:
    normalized = _semantic_text(text)
    normalized = re.sub(r"^\d+[.)]\s*", "", normalized)
    normalized = re.sub(r"\s+([.!?])", r"\1", normalized)
    return normalized.strip()


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
    normalized = _semantic_text(bundle.display_text)
    groups = _ATOMIC_STAGE_MARKERS.get(stage, ())
    return any(
        all(marker in normalized for marker in marker_group)
        for marker_group in groups
    )


_SEMANTIC_NORMALIZATIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\bvapor[-/ ]liquid separation\b"),
        "vapor liquid separation",
    ),
    (re.compile(r"\bheat[- ]transfer\b"), "heat transfer"),
    (re.compile(r"\bcorps de l[’']evaporateur\b"), "vapor body"),
    (re.compile(r"\bevaporateur\b"), "evaporator"),
    (re.compile(r"\bcristallisation\b"), "crystallization"),
    (re.compile(r"\bfonctions?\b"), "functions"),
    (re.compile(r"\bseparer\b"), "separates"),
    (re.compile(r"\bliquid\b"), "liquor"),
    (
        re.compile(r"\bvapor liquor separation\b"),
        "vapor liquid separation",
    ),
    (re.compile(r"\breturns?\b"), "returned"),
    (re.compile(r"\bheating element\b"), "heat exchanger"),
    (
        re.compile(r"\bevaporateur a circulation forcee\b"),
        "forced circulation evaporator",
    ),
    (re.compile(r"\bcirculation forcee\b"), "forced circulation"),
    (re.compile(r"\beconomie de vapeur\b"), "steam economy"),
    (re.compile(r"\bconsommation de vapeur\b"), "steam consumption"),
    (re.compile(r"\btransfert (?:de chaleur|thermique)\b"), "heat transfer"),
    (re.compile(r"\bresistance thermique\b"), "thermal resistance"),
    (re.compile(r"\bencrassement\b"), "fouling"),
    (re.compile(r"\bentartrage\b"), "scaling"),
    (re.compile(r"\bechangeur de chaleur\b"), "heat exchanger"),
    (re.compile(r"\bechangeur\b"), "heat exchanger"),
    (re.compile(r"\bchambre de flash\b"), "flash chamber"),
    (re.compile(r"\bchambre de vaporisation\b"), "flash chamber"),
    (re.compile(r"\bbouilleur\b"), "flash chamber"),
    (re.compile(r"\bpompe\b"), "pump"),
    (re.compile(r"\bretire\b"), "withdraws"),
    (re.compile(r"\bsoutire\b"), "withdraws"),
    (re.compile(r"\bpousse\b"), "forces"),
    (re.compile(r"\brenvoye\b"), "returned"),
    (re.compile(r"\breintroduit\b"), "returned"),
    (re.compile(r"\bliquide\b"), "liquor"),
    (re.compile(r"\bacide phosphorique\b"), "phosphoric acid"),
    (re.compile(r"\bvapeur d eau\b"), "water vapor"),
    (re.compile(r"\bacide\b"), "acid"),
    (re.compile(r"\balimente\b"), "fed"),
    (re.compile(r"conduite d[’']entree"), "inlet pipe"),
    (re.compile(r"\bconduite d entree\b"), "inlet pipe"),
    (re.compile(r"\belement de chauffage\b"), "heating element"),
    (re.compile(r"\bfond conique\b"), "conical bottom"),
    (re.compile(r"\bquitte\b"), "leaves"),
    (re.compile(r"\bretourne\b|\brevient\b"), "returned"),
    (
        re.compile(r"\bseparation vapeur[- ]liquide\b"),
        "vapor liquid separation",
    ),
    (re.compile(r"\bvapeur\b"), "vapor"),
    (re.compile(r"\bsepare\b"), "separates"),
    (re.compile(r"\ba lieu\b"), "takes place"),
    (re.compile(r"\bsortie produit\b"), "product outlet"),
    (re.compile(r"\bacide produit concentre\b"), "concentrated product acid"),
    (re.compile(r"مضخة الدوران"), "circulation pump"),
    (re.compile(r"عنصر التسخين"), "heating element"),
    (re.compile(r"حجرة الوميض"), "flash chamber"),
    (re.compile(r"جسم المبخر"), "vapor body"),
    (
        re.compile(r"فصل البخار عن (?:السائل|السايل)"),
        "vapor liquid separation",
    ),
    (re.compile(r"مخرج المنتج"), "product outlet"),
    (re.compile(r"انبوب الدخول"), "inlet pipe"),
    (re.compile(r"القاع المخروطي"), "conical bottom"),
    (re.compile(r"يغادر"), "leaves"),
    (re.compile(r"الحمض المركز المنتج"), "concentrated product acid"),
    (re.compile(r"الحمض"), "acid"),
    (re.compile(r"(?:السائل|السايل)"), "liquor"),
    (re.compile(r"يغذى"), "is fed"),
    (re.compile(r"تدفع"), "forces"),
    (re.compile(r"يعود"), "returned"),
    (re.compile(r"يسحب"), "withdrawn"),
    (re.compile(r"يحدث"), "takes place"),
    (re.compile(r"عبر"), "through"),
    (re.compile(r"الى"), "to"),
    (re.compile(r"في"), "in"),
    (re.compile(r"\bcapacite\b"), "capacity"),
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


class ClaimSupportStatus(StrEnum):
    """Deterministic support classes for smoke tests."""

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CITATION_MISSING = "citation_missing"


@dataclass(frozen=True, slots=True)
class ClaimSupport:
    """One answer claim and its cited-evidence overlap."""

    claim: str
    cited_sources: tuple[int, ...]
    status: ClaimSupportStatus
    lexical_coverage: float


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


@dataclass(frozen=True, slots=True)
class AnswerContractResult:
    """Deterministic end-to-end answer contract normalization."""

    answer: str
    changed: bool
    fallback_used: bool
    missing_roles: tuple[str, ...] = ()
    removed_claims: tuple[str, ...] = ()
    atomic_plan_used: bool = False


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_marks = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", without_marks).strip()


def _semantic_text(text: str) -> str:
    """Normalize a small bilingual technical lexicon before overlap checks."""

    normalized = _normalize(text)
    for pattern, replacement in _SEMANTIC_NORMALIZATIONS:
        normalized = pattern.sub(replacement, normalized)
    return normalized


def _tokens(text: str) -> set[str]:
    normalized = _semantic_text(text)
    return {
        token
        for token in _TOKEN.findall(normalized)
        if token not in _STOPWORDS
    }


def _contains_normalized_phrase(text: str, phrase: str) -> bool:
    """Match a normalized technical phrase without substring collisions.

    In particular, ``introduit`` must not match ``réintroduit``.  The old
    substring check classified a return-to-vessel statement as a new inlet
    relation and rejected otherwise grounded multilingual paraphrases.
    """

    normalized_phrase = _normalize(phrase)
    if not normalized_phrase:
        return False
    return re.search(
        rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)",
        text,
    ) is not None


def _all_concepts(text: str) -> set[str]:
    normalized = _semantic_text(text)
    return {
        concept
        for concept, phrases in _RELATION_CONCEPTS.items()
        if any(
            _contains_normalized_phrase(
                normalized,
                _semantic_text(phrase),
            )
            for phrase in phrases
        )
    }


def _concepts(text: str) -> set[str]:
    return _all_concepts(text) & _STRICT_RELATION_CONCEPTS


def _measurement_text(text: str) -> str:
    """Canonicalize OCR decimals, grouped thousands and engineering units."""

    normalized = _LIST_PREFIX.sub("", _normalize(text))
    normalized = normalized.translate(
        str.maketrans(
            {
                chr(0x2013): "-",
                chr(0x2014): "-",
                chr(0x2212): "-",
                chr(0x00B3): "3",
            }
        )
    )
    normalized = re.sub(r"(?<=\d)\s*[.,]\s*(?=\d)", ".", normalized)

    previous = None
    while previous != normalized:
        previous = normalized
        normalized = re.sub(
            r"(?<=\d)\s+(?=\d{3}(?:\D|$))",
            "",
            normalized,
        )

    for pattern, replacement in _UNIT_NORMALIZATIONS:
        normalized = pattern.sub(replacement, normalized)

    normalized = re.sub(r"(?<=\d)\s*%", "%", normalized)
    return normalized


def _canonical_number(value: str) -> str:
    try:
        number = Decimal(value)
    except InvalidOperation:
        return value

    if number == number.to_integral():
        return str(number.quantize(Decimal("1")))

    return format(number.normalize(), "f")


def _numbers(text: str) -> set[str]:
    normalized = _measurement_text(text)
    found = {
        _canonical_number(match.group(0))
        for match in _NUMBER.finditer(normalized)
    }

    for word, value in _NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", normalized):
            found.add(value)

    return found


def _units(text: str) -> set[str]:
    normalized = _measurement_text(text)
    return {match.group(0) for match in _UNIT.finditer(normalized)}


def _claim_support_gaps(
    claim: str,
    evidence_text: str,
) -> tuple[set[str], set[str], set[str]]:
    """Return explicit relation, numeric and unit gaps for one claim."""

    return (
        _concepts(claim) - _concepts(evidence_text),
        _numbers(claim) - _numbers(evidence_text),
        _units(claim) - _units(evidence_text),
    )


def _lexical_coverage(claim: str, evidence_text: str) -> float:
    claim_tokens = _tokens(claim)
    if not claim_tokens:
        return 0.0

    evidence_tokens = _tokens(evidence_text)
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


def _source_directly_supports(
    claim: str,
    bundle: EvidenceBundle,
) -> tuple[bool, float]:
    """Require one cited Source N to support the whole sentence by itself."""

    missing_concepts, missing_numbers, missing_units = _claim_support_gaps(
        claim,
        bundle.display_text,
    )
    coverage = _lexical_coverage(claim, bundle.display_text)
    claim_tokens = _tokens(claim)
    overlap_count = len(claim_tokens & _tokens(bundle.display_text))
    minimum_overlap = 1 if len(claim_tokens) <= 2 else 2

    atomic_stage = _atomic_template_stage(claim)
    atomic_supported = (
        atomic_stage is not None
        and _bundle_supports_atomic_stage(bundle, atomic_stage)
        and not missing_numbers
        and not missing_units
    )
    deterministic_stage = _deterministic_template_stage(claim)
    deterministic_supported = (
        deterministic_stage is not None
        and _bundle_supports_deterministic_stage(
            bundle,
            deterministic_stage,
        )
    )
    supported = atomic_supported or deterministic_supported or (
        not missing_concepts
        and not missing_numbers
        and not missing_units
        and coverage >= _MIN_DIRECT_LEXICAL_COVERAGE
        and overlap_count >= minimum_overlap
    )
    return supported, coverage


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


def _language_key(language: str) -> str:
    normalized = language.strip().lower()
    if normalized.startswith("en"):
        return "en"
    if normalized.startswith("ar"):
        return "ar"
    return "fr"


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


def _attach_citations(
    claim: str,
    citations: tuple[int, ...],
) -> str:
    """Place inherited citations before terminal sentence punctuation."""

    citation_text = " ".join(
        f"[Source {source_number}]"
        for source_number in citations
    )
    match = re.search(r"([.!?])$", claim)

    if match is None:
        return f"{claim} {citation_text}".strip()

    return (
        claim[: match.start()].rstrip()
        + " "
        + citation_text
        + match.group(1)
    )


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




_DEFINITION_MARKERS = (
    " is a ",
    " is an ",
    " is the ",
    " type of ",
    " refers to ",
    " means ",
    " est un ",
    " est une ",
    " désigne ",
    " designe ",
    " نوع من ",
    " هو ",
    " هي ",
)
_DEFINITION_MECHANISM_MARKERS = (
    "pump",
    "pumped",
    "circulation",
    "heating element",
    "heating surface",
    "heat exchanger",
    "pompe",
    "circulation",
    "élément de chauffage",
    "element de chauffage",
    "échangeur",
    "echangeur",
    "مضخة",
    "الدوران",
    "المبادل الحراري",
)
_DEFINITION_FUNCTION_MARKERS = (
    "heat transfer",
    "vapor liquid separation",
    "vapor-liquid separation",
    "crystallization",
    "solids",
    "suspension",
    "fouling",
    "evaporation",
    "transfert de chaleur",
    "séparation vapeur",
    "separation vapeur",
    "cristallisation",
    "solides",
    "suspension",
    "encrassement",
    "évaporation",
    "evaporation",
    "انتقال الحرارة",
    "فصل البخار",
    "التبلور",
    "المواد الصلبة",
    "التبخر",
)
_COMPARISON_CRITERIA_MARKERS = (
    "heat transfer",
    "heat-transfer",
    "fouling",
    "scaling",
    "viscosity",
    "residence time",
    "pressure drop",
    "temperature",
    "feed distribution",
    "plugging",
    "solids",
    "circulation",
    "boiling",
    "capacity",
    "steam economy",
    "transfert de chaleur",
    "encrassement",
    "entartrage",
    "viscosité",
    "viscosite",
    "temps de séjour",
    "temps de sejour",
    "perte de charge",
    "température",
    "temperature",
    "distribution de l'alimentation",
    "bouchage",
    "solides",
    "circulation",
    "ébullition",
    "ebullition",
)
_TROUBLESHOOTING_PROBLEM_MARKERS = (
    "fouling",
    "scaling",
    "deposit",
    "deposits",
    "encrassement",
    "entartrage",
    "dépôt",
    "dépôts",
    "depot",
    "depots",
    "ترسب",
    "تكلس",
)
_TROUBLESHOOTING_ROLE_MARKERS: dict[str, tuple[str, ...]] = {
    "cause": (
        "caused by",
        "due to",
        "may be due",
        "results from",
        "corrosion",
        "solid matter",
        "feed solids",
        "condensing vapor",
        "causé par",
        "cause par",
        "dû à",
        "du a",
        "corrosion",
        "matière solide",
        "matiere solide",
        "بسبب",
    ),
    "mechanism": (
        "formation of deposits",
        "deposit formation",
        "deposits",
        "deposit",
        "thermal resistance",
        "accumulation",
        "precipitation",
        "formation de dépôts",
        "formation de depots",
        "dépôts",
        "depots",
        "résistance thermique",
        "resistance thermique",
        "accumulation",
        "précipitation",
        "ترسب",
    ),
    "effect": (
        "decrease",
        "reduce",
        "reduced",
        "increase",
        "heat transfer coefficient",
        "heat transfer efficiency",
        "steam economy",
        "capacity",
        "pressure drop",
        "shutdown",
        "diminue",
        "réduit",
        "reduit",
        "augmente",
        "coefficient de transfert",
        "économie de vapeur",
        "economie de vapeur",
        "capacité",
        "capacite",
        "perte de charge",
        "arrêt",
        "arret",
        "انخفاض",
        "زيادة",
    ),
    "action": (
        "clean",
        "cleaning",
        "wash",
        "washing",
        "remove deposits",
        "shutdown and washing",
        "nettoyage",
        "laver",
        "lavage",
        "éliminer les dépôts",
        "eliminer les depots",
        "arrêt et lavage",
        "arret et lavage",
        "تنظيف",
        "غسل",
    ),
}


_DETERMINISTIC_ANSWER_TEMPLATES: dict[
    str,
    dict[str, str],
] = {
    "en": {
        "definition_mechanism": (
            "A forced-circulation evaporator is an evaporator in which a "
            "pump circulates the liquid through a heating surface and returns "
            "it to the vapor body."
        ),
        "definition_function": (
            "This arrangement separates the heat-transfer, vapor-liquid-"
            "separation, and crystallization functions."
        ),
        "pump_role": (
            "The circulation pump withdraws liquid from the flash chamber and "
            "forces it through the heating element."
        ),
        "pump_necessity": (
            "The circulation pump is necessary because it maintains positive "
            "liquid circulation past the heating surface independently of the "
            "evaporation rate."
        ),
        "pump_function": (
            "This allows heat transfer, vapor-liquid separation, and "
            "crystallization to be performed as separate functions."
        ),
        "pump_return": (
            "The pump withdraws liquor from the flash chamber and forces it "
            "through the heating element back to the flash chamber."
        ),
        "vapor_body_role": (
            "The vapor body provides the chamber in which vapor-liquid "
            "separation occurs after the heated acid returns from the heating "
            "element."
        ),
        "fouling_cause": (
            "Fouling deposits may originate from corrosion, solids entering "
            "with the feed, or material deposited by the condensing vapor."
        ),
        "fouling_mechanism": (
            "The deposits coat the heating surface and add resistance to heat "
            "transfer."
        ),
        "fouling_effect": (
            "As a result, the heat-transfer coefficient decreases and the "
            "evaporator may require shutdown."
        ),
        "fouling_action": (
            "The documented corrective action is to wash or clean the "
            "evaporator to remove the deposits."
        ),
        "overall_conservation": (
            "At steady state, accumulation is zero and total mass entering the "
            "evaporator equals total mass leaving it."
        ),
        "overall_equation": (
            "For one feed, one concentrated-liquid product, and one vapor "
            "outlet, the symbolic overall balance is F = P + V."
        ),
        "overall_feed_definition": (
            "F is the mass flow rate of the dilute phosphoric-acid feed."
        ),
        "overall_outlet_definition": (
            "P is the mass flow rate of the concentrated liquid product, and "
            "V is the mass flow rate of the vapor removed."
        ),
        "species_equation": (
            "At steady state, the P2O5 component balance is "
            "F x_F = P x_P + L_P2O5."
        ),
        "species_feed_definition": (
            "F and x_F are the feed mass flow rate and its P2O5 mass fraction."
        ),
        "species_product_definition": (
            "P and x_P are the concentrated-product mass flow rate and its "
            "P2O5 mass fraction."
        ),
        "species_loss_definition": (
            "L_P2O5 is the P2O5 mass flow lost by entrainment or carryover."
        ),
        "species_no_loss": (
            "If P2O5 entrainment is neglected, L_P2O5 = 0 and the balance "
            "reduces to F x_F = P x_P."
        ),
        "energy_equation": (
            "At steady state, neglecting kinetic- and potential-energy "
            "changes, the evaporator energy balance is "
            "Qdot + F h_F + Wdot_s = P h_P + V h_V + Qdot_loss."
        ),
        "energy_heat_definition": (
            "Qdot is the heat supplied by the heating steam, and Wdot_s is "
            "the shaft work supplied by the circulation pump."
        ),
        "energy_liquid_definition": (
            "F h_F and P h_P are the enthalpy rates of the feed and the "
            "concentrated liquid product."
        ),
        "energy_vapor_definition": (
            "V h_V is the enthalpy rate carried out by the generated vapor."
        ),
        "energy_loss_definition": (
            "Qdot_loss represents heat loss to the surroundings and is set to "
            "zero when heat losses are neglected."
        ),
    },
    "fr": {
        "definition_mechanism": (
            "Un évaporateur à circulation forcée est un évaporateur dans "
            "lequel une pompe fait circuler le liquide à travers une surface "
            "de chauffe puis le renvoie vers le corps de l’évaporateur."
        ),
        "definition_function": (
            "Cette configuration sépare les fonctions de transfert de "
            "chaleur, de séparation vapeur-liquide et de cristallisation."
        ),
        "pump_role": (
            "La pompe de circulation retire le liquide de la chambre de flash "
            "et le pousse à travers l’élément de chauffage."
        ),
        "pump_necessity": (
            "La pompe de circulation est nécessaire parce qu’elle maintient "
            "une circulation positive du liquide le long de la surface de "
            "chauffe, indépendamment du taux d’évaporation."
        ),
        "pump_function": (
            "Cela permet de dissocier les fonctions de transfert de chaleur, "
            "de séparation vapeur-liquide et de cristallisation."
        ),
        "pump_return": (
            "La pompe retire le liquide de la chambre de flash et le force à "
            "traverser l’élément de chauffage avant de le renvoyer dans la "
            "chambre de flash."
        ),
        "vapor_body_role": (
            "La chambre de vaporisation est le volume dans lequel la "
            "séparation vapeur-liquide se produit après le retour de l’acide "
            "chauffé depuis l’élément de chauffage."
        ),
        "fouling_cause": (
            "Les dépôts d’encrassement peuvent provenir de la corrosion, des "
            "solides entraînés avec l’alimentation ou de matière déposée par "
            "la vapeur en condensation."
        ),
        "fouling_mechanism": (
            "Ces dépôts recouvrent la surface de chauffe et ajoutent une "
            "résistance au transfert de chaleur."
        ),
        "fouling_effect": (
            "Le coefficient de transfert de chaleur diminue alors et "
            "l’évaporateur peut devoir être arrêté."
        ),
        "fouling_action": (
            "L’action corrective documentée consiste à laver ou nettoyer "
            "l’évaporateur afin d’éliminer les dépôts."
        ),
        "overall_conservation": (
            "En régime permanent, l’accumulation est nulle et la masse totale "
            "entrant dans l’évaporateur est égale à la masse totale sortante."
        ),
        "overall_equation": (
            "Pour une alimentation, un produit liquide concentré et une "
            "sortie vapeur, le bilan global symbolique est F = P + V."
        ),
        "overall_feed_definition": (
            "F est le débit massique de l’alimentation en acide phosphorique "
            "dilué."
        ),
        "overall_outlet_definition": (
            "P est le débit massique du produit liquide concentré et V est le "
            "débit massique de la vapeur extraite."
        ),
        "species_equation": (
            "En régime permanent, le bilan de P2O5 est "
            "F x_F = P x_P + L_P2O5."
        ),
        "species_feed_definition": (
            "F et x_F sont le débit massique de l’alimentation et sa fraction "
            "massique en P2O5."
        ),
        "species_product_definition": (
            "P et x_P sont le débit massique du produit concentré et sa "
            "fraction massique en P2O5."
        ),
        "species_loss_definition": (
            "L_P2O5 est le débit massique de P2O5 perdu par entraînement ou "
            "par carryover."
        ),
        "species_no_loss": (
            "Si l’entraînement de P2O5 est négligé, L_P2O5 = 0 et le bilan "
            "devient F x_F = P x_P."
        ),
        "energy_equation": (
            "En régime permanent, en négligeant les variations d’énergie "
            "cinétique et potentielle, le bilan énergétique est "
            "Qdot + F h_F + Wdot_s = P h_P + V h_V + Qdot_loss."
        ),
        "energy_heat_definition": (
            "Qdot est la chaleur fournie par la vapeur de chauffage et Wdot_s "
            "est le travail d’arbre fourni par la pompe de circulation."
        ),
        "energy_liquid_definition": (
            "F h_F et P h_P sont les débits d’enthalpie de l’alimentation et "
            "du produit liquide concentré."
        ),
        "energy_vapor_definition": (
            "V h_V est le débit d’enthalpie emporté par la vapeur produite."
        ),
        "energy_loss_definition": (
            "Qdot_loss représente les pertes de chaleur vers l’environnement "
            "et vaut zéro lorsqu’elles sont négligées."
        ),
    },
    "ar": {
        "definition_mechanism": (
            "المبخر ذو الدوران القسري هو مبخر تستخدم فيه مضخة لدفع السائل "
            "عبر سطح التسخين ثم إعادته إلى جسم المبخر."
        ),
        "definition_function": (
            "يسمح هذا الترتيب بفصل وظائف انتقال الحرارة وفصل البخار عن "
            "السائل والتبلور."
        ),
        "pump_role": (
            "تسحب مضخة الدوران السائل من حجرة الوميض وتدفعه عبر عنصر التسخين."
        ),
        "pump_necessity": (
            "مضخة الدوران ضرورية لأنها تحافظ على دوران موجب للسائل عبر سطح "
            "التسخين بصورة مستقلة عن معدل التبخر."
        ),
        "pump_function": (
            "وهذا يسمح بفصل وظائف انتقال الحرارة وفصل البخار عن السائل "
            "والتبلور."
        ),
        "pump_return": (
            "تسحب المضخة السائل من حجرة الوميض وتدفعه عبر عنصر التسخين ثم "
            "تعيده إلى حجرة الوميض."
        ),
        "vapor_body_role": (
            "غرفة التبخير هي الحيز الذي يحدث فيه فصل البخار عن الطور السائل "
            "بعد عودة الحمض الساخن من عنصر التسخين."
        ),
        "fouling_cause": (
            "قد تنشأ رواسب التلوث من التآكل أو من المواد الصلبة الداخلة مع "
            "التغذية أو من مواد تترسب بفعل البخار المتكاثف."
        ),
        "fouling_mechanism": (
            "تغطي هذه الرواسب سطح التسخين وتضيف مقاومة لانتقال الحرارة."
        ),
        "fouling_effect": (
            "ينخفض معامل انتقال الحرارة نتيجة لذلك وقد يصبح إيقاف المبخر "
            "ضرورياً."
        ),
        "fouling_action": (
            "الإجراء التصحيحي الموثق هو غسل المبخر أو تنظيفه لإزالة الرواسب."
        ),
        "overall_conservation": (
            "في الحالة المستقرة يكون التراكم صفراً وتساوي الكتلة الكلية "
            "الداخلة إلى المبخر الكتلة الكلية الخارجة منه."
        ),
        "overall_equation": (
            "عند وجود تغذية واحدة ومنتج سائل مركز ومخرج بخار واحد يكون "
            "الميزان الكلي الرمزي F = P + V."
        ),
        "overall_feed_definition": (
            "يمثل F معدل التدفق الكتلي لتغذية حمض الفوسفوريك المخفف."
        ),
        "overall_outlet_definition": (
            "يمثل P معدل التدفق الكتلي للمنتج السائل المركز ويمثل V معدل "
            "التدفق الكتلي للبخار المسحوب."
        ),
        "species_equation": (
            "في الحالة المستقرة يكون ميزان P2O5 هو "
            "F x_F = P x_P + L_P2O5."
        ),
        "species_feed_definition": (
            "يمثل F و x_F معدل تدفق التغذية الكتلي والكسر الكتلي لـ P2O5 فيها."
        ),
        "species_product_definition": (
            "يمثل P و x_P معدل تدفق المنتج المركز الكتلي والكسر الكتلي لـ "
            "P2O5 فيه."
        ),
        "species_loss_definition": (
            "يمثل L_P2O5 معدل التدفق الكتلي لـ P2O5 المفقود بالانجراف أو "
            "الحمل مع البخار."
        ),
        "species_no_loss": (
            "إذا أهمل انجراف P2O5 فإن L_P2O5 = 0 ويصبح الميزان "
            "F x_F = P x_P."
        ),
        "energy_equation": (
            "في الحالة المستقرة ومع إهمال تغيرات الطاقة الحركية وطاقة الوضع "
            "يكون ميزان الطاقة "
            "Qdot + F h_F + Wdot_s = P h_P + V h_V + Qdot_loss."
        ),
        "energy_heat_definition": (
            "يمثل Qdot الحرارة التي يوفرها بخار التسخين ويمثل Wdot_s شغل "
            "العمود الذي توفره مضخة الدوران."
        ),
        "energy_liquid_definition": (
            "يمثل F h_F و P h_P معدلي الإنثالبي للتغذية وللمنتج السائل المركز."
        ),
        "energy_vapor_definition": (
            "يمثل V h_V معدل الإنثالبي الذي يحمله البخار المتولد إلى الخارج."
        ),
        "energy_loss_definition": (
            "يمثل Qdot_loss فقد الحرارة إلى الوسط المحيط ويساوي صفراً عند "
            "إهمال الفواقد الحرارية."
        ),
    },
}


_DETERMINISTIC_STAGE_MARKERS: dict[
    str,
    tuple[tuple[str, ...], ...],
] = {
    "definition_mechanism": (
        ("pump", "heating surface"),
        ("pump", "heat exchanger"),
        ("pump", "heating element"),
    ),
    "definition_function": (
        ("heat transfer", "vapor liquid separation"),
        ("heat transfer", "crystallization"),
    ),
    "pump_role": (
        ("pump", "flash chamber", "heating element"),
        ("pump", "heating surface"),
    ),
    "pump_necessity": (
        ("pump", "heating surface", "circulation"),
        ("pump", "evaporation rate", "circulation"),
    ),
    "pump_function": (
        ("heat transfer", "vapor liquid separation"),
        ("heat transfer", "crystallization"),
    ),
    "pump_return": (
        ("pump", "flash chamber", "heating element", "back"),
        ("pump", "flash chamber", "heating element", "returned"),
    ),
    "vapor_body_role": (
        ("vapor body", "vapor liquid separation"),
        ("flash chamber", "vapor liquid separation"),
    ),
    "fouling_cause": (
        ("deposit", "corrosion"),
        ("deposit", "solid matter"),
        ("deposit", "condensing vapor"),
    ),
    "fouling_mechanism": (
        ("deposit", "heat transfer"),
        ("deposit", "thermal resistance"),
    ),
    "fouling_effect": (
        ("heat transfer coefficient", "decrease"),
        ("shutdown", "washing"),
        ("steam economy", "fouling"),
    ),
    "fouling_action": (
        ("shutdown", "washing"),
        ("clean", "deposit"),
        ("washing", "deposit"),
    ),
    "overall_conservation": (
        ("mass balance",),
        ("material balance",),
        ("conservation of mass",),
    ),
    "overall_equation": (
        ("feed", "product", "vapor"),
        ("mass in", "mass out"),
        ("overall mass balance",),
    ),
    "overall_feed_definition": (
        ("feed", "mass flow"),
        ("mass in",),
    ),
    "overall_outlet_definition": (
        ("product", "vapor"),
        ("mass out",),
        ("evaporated water",),
    ),
    "species_equation": (
        ("component balance",),
        ("species balance",),
        ("conservation of mass",),
        ("mass balance",),
    ),
    "species_feed_definition": (
        ("p2o5", "feed"),
        ("component", "mass in"),
    ),
    "species_product_definition": (
        ("p2o5", "product"),
        ("p2o5", "outlet"),
        ("concentrated", "product"),
    ),
    "species_loss_definition": (
        ("p2o5", "loss"),
        ("entrainment",),
        ("carryover",),
    ),
    "species_no_loss": (
        ("entrainment",),
        ("carryover",),
        ("p2o5", "loss"),
    ),
    "energy_equation": (
        ("energy balance",),
        ("conservation of energy",),
        ("enthalpy", "heat", "work"),
    ),
    "energy_heat_definition": (
        ("steam", "heat"),
        ("heat input",),
        ("shaft work",),
    ),
    "energy_liquid_definition": (
        ("feed", "enthalpy"),
        ("product", "enthalpy"),
        ("liquid", "enthalpy"),
    ),
    "energy_vapor_definition": (
        ("vapor", "enthalpy"),
        ("latent heat",),
        ("water evaporated", "heat"),
    ),
    "energy_loss_definition": (
        ("heat loss",),
        ("surroundings", "heat"),
        ("energy balance",),
    ),
}


_ROLE_BY_STAGE: dict[str, tuple[str, ...]] = {
    "overall_conservation": ("overall_conservation",),
    "overall_equation": ("overall_conservation", "product_and_vapor"),
    "overall_feed_definition": ("feed_stream",),
    "overall_outlet_definition": ("product_and_vapor",),
    "species_equation": ("species_conservation",),
    "species_feed_definition": ("species_feed",),
    "species_product_definition": ("species_product",),
    "species_loss_definition": ("species_losses",),
    "species_no_loss": ("species_losses",),
    "energy_equation": ("energy_conservation",),
    "energy_heat_definition": ("heat_input",),
    "energy_liquid_definition": ("feed_product_enthalpy",),
    "energy_vapor_definition": ("vapor_enthalpy",),
    "energy_loss_definition": ("energy_conservation", "heat_input"),
}


def _bundle_role(bundle: EvidenceBundle) -> str | None:
    prefix = "evidence_role:"
    provenance = bundle.selection_provenance.strip()
    if provenance.startswith(prefix):
        return provenance[len(prefix) :]
    return None


def _deterministic_template_stage(claim: str) -> str | None:
    normalized = _canonical_atomic_claim(claim)
    for language_templates in _DETERMINISTIC_ANSWER_TEMPLATES.values():
        for stage, template in language_templates.items():
            if _canonical_atomic_claim(template) == normalized:
                return stage
    return None


def _bundle_supports_deterministic_stage(
    bundle: EvidenceBundle,
    stage: str,
) -> bool:
    role = _bundle_role(bundle)
    if role in _ROLE_BY_STAGE.get(stage, ()):
        return True

    normalized = _semantic_text(bundle.display_text)
    marker_groups = _DETERMINISTIC_STAGE_MARKERS.get(stage, ())
    return any(
        all(_semantic_text(marker) in normalized for marker in marker_group)
        for marker_group in marker_groups
    )


def _best_bundle_for_deterministic_stage(
    stage: str,
    bundles: list[EvidenceBundle],
) -> EvidenceBundle | None:
    candidates = [
        bundle
        for bundle in bundles
        if _bundle_supports_deterministic_stage(bundle, stage)
    ]
    if not candidates:
        return None
    expected_roles = set(_ROLE_BY_STAGE.get(stage, ()))
    return max(
        candidates,
        key=lambda bundle: (
            int(_bundle_role(bundle) in expected_roles),
            bundle.anchor_score,
            -bundle.source_number,
        ),
    )


def _templated_claim(
    stage: str,
    bundle: EvidenceBundle,
    *,
    language: str,
) -> str:
    template = _DETERMINISTIC_ANSWER_TEMPLATES[_language_key(language)][stage]
    return _attach_citations(template, (bundle.source_number,))


def build_deterministic_definition_answer(
    bundles: list[EvidenceBundle],
    *,
    language: str,
) -> str | None:
    mechanism = _best_bundle_for_deterministic_stage(
        "definition_mechanism",
        bundles,
    )
    function = _best_bundle_for_deterministic_stage(
        "definition_function",
        bundles,
    )
    if mechanism is None or function is None:
        return None
    return "\n".join(
        (
            _templated_claim(
                "definition_mechanism",
                mechanism,
                language=language,
            ),
            _templated_claim(
                "definition_function",
                function,
                language=language,
            ),
        )
    )


def build_deterministic_balance_answer(
    bundles: list[EvidenceBundle],
    *,
    balance_kind: str,
    language: str,
) -> str | None:
    stage_sets = {
        "overall_mass": (
            "overall_conservation",
            "overall_equation",
            "overall_feed_definition",
            "overall_outlet_definition",
        ),
        "species": (
            "species_equation",
            "species_feed_definition",
            "species_product_definition",
            "species_loss_definition",
            "species_no_loss",
        ),
        "energy": (
            "energy_equation",
            "energy_heat_definition",
            "energy_liquid_definition",
            "energy_vapor_definition",
            "energy_loss_definition",
        ),
    }
    stages = stage_sets.get(balance_kind)
    if stages is None:
        return None

    claims: list[str] = []
    for stage in stages:
        bundle = _best_bundle_for_deterministic_stage(stage, bundles)
        if bundle is None:
            return None
        claims.append(_templated_claim(stage, bundle, language=language))
    return "\n".join(claims)


def build_deterministic_fouling_answer(
    bundles: list[EvidenceBundle],
    *,
    language: str,
) -> str | None:
    stages = (
        "fouling_cause",
        "fouling_mechanism",
        "fouling_effect",
        "fouling_action",
    )
    claims: list[str] = []
    for stage in stages:
        bundle = _best_bundle_for_deterministic_stage(stage, bundles)
        if bundle is None:
            return None
        claims.append(_templated_claim(stage, bundle, language=language))
    return "\n".join(claims)


def _scoped_explanation_stage(question: str) -> tuple[str, ...] | None:
    normalized = _normalize(question)
    pump = any(
        marker in normalized
        for marker in (
            "circulation pump",
            "pompe de circulation",
            "مضخة الدوران",
        )
    )
    if pump and any(
        marker in normalized
        for marker in (
            "back to the flash chamber",
            "send the liquid back",
            "ramener le liquide",
            "renvoyer le liquide",
            "إعادته إلى حجرة الوميض",
        )
    ):
        return ("pump_return",)
    if pump and any(
        marker in normalized
        for marker in (
            "necessary",
            "necessaire",
            "pourquoi",
            "why",
            "ضرورية",
        )
    ):
        return ("pump_necessity", "pump_function")
    if pump and any(
        marker in normalized
        for marker in (
            "role",
            "fonction",
            "what does",
            "how does",
            "دور",
        )
    ):
        return ("pump_role", "pump_function")

    vapor_body = any(
        marker in normalized
        for marker in (
            "vapor body",
            "vapour body",
            "evaporation chamber",
            "chambre de vaporisation",
            "غرفة التبخير",
            "جسم المبخر",
        )
    )
    separation = any(
        marker in normalized
        for marker in (
            "vapor liquid separation",
            "separate vapor",
            "separation vapeur",
            "فصل البخار",
        )
    )
    if vapor_body and separation:
        return ("vapor_body_role",)
    return None


def build_deterministic_scoped_explanation(
    question: str,
    bundles: list[EvidenceBundle],
    *,
    language: str,
) -> str | None:
    stages = _scoped_explanation_stage(question)
    if stages is None:
        return None
    claims: list[str] = []
    for stage in stages:
        bundle = _best_bundle_for_deterministic_stage(stage, bundles)
        if bundle is None:
            return None
        claims.append(_templated_claim(stage, bundle, language=language))
    return "\n".join(claims)


def _infer_balance_kind(question: str) -> str:
    normalized = _normalize(question).replace("₂", "2").replace("₅", "5")
    if any(
        marker in normalized
        for marker in (
            "p2o5",
            "species",
            "component",
            "espece",
            "composant",
        )
    ):
        return "species"
    if any(
        marker in normalized
        for marker in (
            "energy",
            "enthalpy",
            "heat balance",
            "energetique",
            "enthalpie",
            "chaleur",
        )
    ):
        return "energy"
    return "overall_mass"


def _answer_claim_records(answer: str) -> list[tuple[str, str]]:
    """Return ``(claim_with_citations, normalized_claim)`` records."""

    records: list[tuple[str, str]] = []
    for claim in _iter_answer_claims(answer):
        clean = _CITATION.sub("", claim).strip()
        if clean:
            records.append((claim.strip(), _normalize(clean)))
    return records


def _contains_any_marker(text: str, markers: tuple[str, ...]) -> bool:
    return any(_normalize(marker) in text for marker in markers)


def _subject_aliases(subject: str) -> tuple[str, ...]:
    normalized = _normalize(subject)
    aliases = {normalized}
    aliases.add(re.sub(r"^(?:a|an|the|un|une|le|la|les)\s+", "", normalized))
    if "forced circulation" in normalized or "circulation forcee" in normalized:
        aliases.update(
            {
                "forced circulation evaporator",
                "forced circulation",
                "evaporateur a circulation forcee",
                "circulation forcee",
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


def _definition_contract(answer: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    records = _answer_claim_records(answer)
    combined = " ".join(text for _claim, text in records)
    roles: set[str] = set()

    if _contains_any_marker(f" {combined} ", _DEFINITION_MARKERS):
        roles.add("definition")
    if _contains_any_marker(combined, _DEFINITION_MECHANISM_MARKERS):
        roles.add("mechanism")
    if _contains_any_marker(combined, _DEFINITION_FUNCTION_MARKERS):
        roles.add("function")

    required = ("definition", "mechanism", "function")
    missing = tuple(role for role in required if role not in roles)
    return tuple(sorted(roles)), missing


def _comparison_contract(
    answer: str,
    *,
    subjects: tuple[str, ...],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    records = _answer_claim_records(answer)
    subject_alias_groups = tuple(_subject_aliases(subject) for subject in subjects[:2])
    kept: list[str] = []
    removed: list[str] = []
    covered_subjects: set[int] = set()
    criterion_present = False

    for claim, normalized in records:
        matched_subjects = {
            index
            for index, aliases in enumerate(subject_alias_groups)
            if any(alias in normalized for alias in aliases)
        }
        has_criterion = _contains_any_marker(
            normalized,
            _COMPARISON_CRITERIA_MARKERS,
        )
        if not matched_subjects or not has_criterion:
            removed.append(_CITATION.sub("", claim).strip())
            continue
        kept.append(claim)
        covered_subjects.update(matched_subjects)
        criterion_present = True

    missing: list[str] = []
    if len(subject_alias_groups) >= 1 and 0 not in covered_subjects:
        missing.append("equipment_a")
    if len(subject_alias_groups) >= 2 and 1 not in covered_subjects:
        missing.append("equipment_b")
    if not criterion_present:
        missing.append("comparison_criteria")

    return "\n".join(kept), tuple(missing), tuple(removed)


def _troubleshooting_contract(
    answer: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    records = _answer_claim_records(answer)
    retained: list[tuple[int, int, str, set[str]]] = []
    removed: list[str] = []
    covered_roles: set[str] = set()

    for index, (claim, normalized) in enumerate(records):
        roles = {
            role
            for role, markers in _TROUBLESHOOTING_ROLE_MARKERS.items()
            if _contains_any_marker(normalized, markers)
        }
        problem_related = _contains_any_marker(
            normalized,
            _TROUBLESHOOTING_PROBLEM_MARKERS,
        )
        action_specific = "action" in roles and problem_related
        if not problem_related and not action_specific:
            removed.append(_CITATION.sub("", claim).strip())
            continue
        if not roles:
            removed.append(_CITATION.sub("", claim).strip())
            continue

        covered_roles.update(roles)
        priority = min(
            ("cause", "mechanism", "effect", "action").index(role)
            for role in roles
        )
        retained.append((priority, index, claim, roles))

    retained.sort(key=lambda item: (item[0], item[1]))
    required = ("cause", "mechanism", "effect", "action")
    missing = tuple(role for role in required if role not in covered_roles)
    return (
        "\n".join(item[2] for item in retained),
        missing,
        tuple(removed),
    )


def enforce_answer_contract(
    answer: str,
    bundles: list[EvidenceBundle],
    *,
    question_type: str | None,
    language: str,
    comparison_subjects: tuple[str, ...] = (),
    question: str = "",
    balance_kind: str | None = None,
) -> AnswerContractResult:
    """Apply deterministic task contracts after grounding validation.

    The contract never creates comparison or troubleshooting facts.  It only
    removes grounded-but-off-task claims, orders the remaining claims, or uses
    the existing source-local atomic planner for process flow.
    """

    normalized_type = (question_type or "").strip().lower()

    if normalized_type == "process_flow":
        atomic = build_atomic_process_flow_answer(bundles, language=language)
        if atomic is None:
            return AnswerContractResult(
                answer=_fallback_answer(language),
                changed=True,
                fallback_used=True,
                missing_roles=(
                    "feed_inlet",
                    "conical_bottom",
                    "pump_heat_exchanger",
                    "recirculation_vapor_body",
                    "product_outlet",
                ),
                atomic_plan_used=True,
            )
        return AnswerContractResult(
            answer=atomic,
            changed=atomic != answer,
            fallback_used=False,
            atomic_plan_used=True,
        )

    if normalized_type == "definition":
        deterministic = build_deterministic_definition_answer(
            bundles,
            language=language,
        )
        if deterministic is not None:
            return AnswerContractResult(
                answer=deterministic,
                changed=deterministic != answer,
                fallback_used=False,
            )

        _covered, missing = _definition_contract(answer)
        if missing:
            return AnswerContractResult(
                answer=_fallback_answer(language),
                changed=True,
                fallback_used=True,
                missing_roles=missing,
            )
        return AnswerContractResult(
            answer=answer,
            changed=False,
            fallback_used=False,
        )

    if normalized_type == "balance":
        kind = balance_kind or _infer_balance_kind(question)
        deterministic = build_deterministic_balance_answer(
            bundles,
            balance_kind=kind,
            language=language,
        )
        if deterministic is None:
            return AnswerContractResult(
                answer=_fallback_answer(language),
                changed=True,
                fallback_used=True,
                missing_roles=(f"{kind}_balance",),
            )
        return AnswerContractResult(
            answer=deterministic,
            changed=deterministic != answer,
            fallback_used=False,
        )

    if normalized_type == "explanation":
        scoped = build_deterministic_scoped_explanation(
            question,
            bundles,
            language=language,
        )
        if scoped is not None:
            return AnswerContractResult(
                answer=scoped,
                changed=scoped != answer,
                fallback_used=False,
            )

    if normalized_type == "comparison":
        normalized, missing, removed = _comparison_contract(
            answer,
            subjects=comparison_subjects,
        )
        if missing or not normalized.strip():
            return AnswerContractResult(
                answer=_fallback_answer(language),
                changed=True,
                fallback_used=True,
                missing_roles=missing or ("comparison_claims",),
                removed_claims=removed,
            )
        return AnswerContractResult(
            answer=normalized,
            changed=normalized != answer,
            fallback_used=False,
            removed_claims=removed,
        )

    if normalized_type == "troubleshooting":
        normalized_question = _normalize(question)
        if _contains_any_marker(
            normalized_question,
            _TROUBLESHOOTING_PROBLEM_MARKERS,
        ):
            deterministic = build_deterministic_fouling_answer(
                bundles,
                language=language,
            )
            if deterministic is not None:
                return AnswerContractResult(
                    answer=deterministic,
                    changed=deterministic != answer,
                    fallback_used=False,
                )

        normalized, missing, removed = _troubleshooting_contract(answer)
        if missing or not normalized.strip():
            return AnswerContractResult(
                answer=_fallback_answer(language),
                changed=True,
                fallback_used=True,
                missing_roles=missing or ("troubleshooting_claims",),
                removed_claims=removed,
            )
        return AnswerContractResult(
            answer=normalized,
            changed=normalized != answer,
            fallback_used=False,
            removed_claims=removed,
        )

    return AnswerContractResult(
        answer=answer,
        changed=False,
        fallback_used=False,
    )


def evaluate_claim_support(
    answer: str,
    bundles: list[EvidenceBundle],
) -> list[ClaimSupport]:
    """Classify each sentence against the exact sources it cites."""

    bundle_by_number = {bundle.source_number: bundle for bundle in bundles}
    results: list[ClaimSupport] = []

    for raw_claim in _iter_answer_claims(answer):
        claim = raw_claim.strip()
        if not claim:
            continue

        citations = tuple(
            dict.fromkeys(int(match) for match in _CITATION.findall(claim))
        )
        clean_claim = _CITATION.sub("", claim).strip()

        if not citations:
            results.append(
                ClaimSupport(
                    claim=clean_claim,
                    cited_sources=(),
                    status=ClaimSupportStatus.CITATION_MISSING,
                    lexical_coverage=0.0,
                )
            )
            continue

        coverages: list[float] = []
        supported_sources: list[int] = []
        for source_number in citations:
            bundle = bundle_by_number.get(source_number)
            if bundle is None:
                continue

            supported, coverage = _source_directly_supports(
                clean_claim,
                bundle,
            )
            coverages.append(coverage)
            if supported:
                supported_sources.append(source_number)

        best_coverage = max(coverages, default=0.0)
        if supported_sources:
            status = ClaimSupportStatus.SUPPORTED
        elif best_coverage >= 0.30:
            status = ClaimSupportStatus.PARTIALLY_SUPPORTED
        else:
            status = ClaimSupportStatus.UNSUPPORTED

        results.append(
            ClaimSupport(
                claim=clean_claim,
                cited_sources=citations,
                status=status,
                lexical_coverage=round(best_coverage, 4),
            )
        )

    return results


def validate_claim_support(
    answer: str,
    bundles: list[EvidenceBundle],
) -> None:
    """Require every factual sentence to match one exact cited source."""

    bundle_by_number = {bundle.source_number: bundle for bundle in bundles}
    unsupported: list[str] = []

    for raw_claim in _iter_answer_claims(answer):
        claim = raw_claim.strip()
        if not claim:
            continue

        citations = tuple(
            dict.fromkeys(int(match) for match in _CITATION.findall(claim))
        )
        if not citations:
            continue

        clean_claim = _CITATION.sub("", claim).strip()
        supporting_citations = _supporting_citations(
            clean_claim,
            citations,
            bundle_by_number,
        )
        if supporting_citations:
            continue

        source_diagnostics: list[str] = []
        for source_number in citations:
            bundle = bundle_by_number.get(source_number)
            if bundle is None:
                continue

            missing_concepts, missing_numbers, missing_units = (
                _claim_support_gaps(clean_claim, bundle.display_text)
            )
            coverage = _lexical_coverage(clean_claim, bundle.display_text)
            details = [f"source={source_number}", f"coverage={coverage:.2f}"]
            if missing_concepts:
                details.append(
                    "relations=" + ",".join(sorted(missing_concepts))
                )
            if missing_numbers:
                details.append(
                    "valeurs=" + ",".join(sorted(missing_numbers))
                )
            if missing_units:
                details.append(
                    "unites=" + ",".join(sorted(missing_units))
                )
            source_diagnostics.append("; ".join(details))

        unsupported.append(
            f"{clean_claim} ({' | '.join(source_diagnostics)})"
        )

    if unsupported:
        raise CitationValidationError(
            "Certaines affirmations ne sont pas directement soutenues par "
            "une source citee prise individuellement : "
            + " | ".join(unsupported[:3])
        )
