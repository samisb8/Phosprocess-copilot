"""Low-level semantic and numeric evidence support checks."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum

from phosprocess.rag.citations import CitationValidationError
from phosprocess.retrieval.evidence_bundle import EvidenceBundle

_CITATION = re.compile(r"\[Source ([1-5])\]")

_TOKEN = re.compile(r"(?u)[^\W_]{2,}")

_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")

_UNIT = re.compile(r"(?:%|\u00b0c|t/h|kg/s|kg/h|m3/h|m/s|torr)(?!\w)")

_LIST_PREFIX = re.compile(r"^\s*\d+[.)]\s*")

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
            r"\b(?:tonnes?|tons?|t)\s*"
            r"(?:\(\s*p2o5\s*\))?\s*(?:/|per)\s*"
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
            r"\bkg\s*(?:\(\s*p2o5\s*\))?\s*"
            r"(?:/|per)\s*(?:h|hr|hours?|heures?)\b"
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


def _canonical_atomic_claim(text: str) -> str:
    normalized = _semantic_text(text)
    normalized = re.sub(r"^\d+[.)]\s*", "", normalized)
    normalized = re.sub(r"\s+([.!?])", r"\1", normalized)
    return normalized.strip()

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


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_marks = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    ).casefold()
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

    from phosprocess.rag.citation_binding import (
        _atomic_template_stage,
        _bundle_supports_atomic_stage,
    )
    from phosprocess.rag.deterministic_builders import (
        _bundle_supports_deterministic_stage,
        _deterministic_template_stage,
    )

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
        and not missing_numbers
        and not missing_units
    )
    supported = atomic_supported or deterministic_supported or (
        not missing_concepts
        and not missing_numbers
        and not missing_units
        and coverage >= _MIN_DIRECT_LEXICAL_COVERAGE
        and overlap_count >= minimum_overlap
    )
    return supported, coverage


def _language_key(language: str) -> str:
    normalized = language.strip().lower()
    if normalized.startswith("en"):
        return "en"
    if normalized.startswith("ar"):
        return "ar"
    return "fr"


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


def _bundle_roles(bundle: EvidenceBundle) -> tuple[str, ...]:
    """Read legacy and multi-role tags from semicolon provenance."""

    roles: list[str] = []
    for part in bundle.selection_provenance.split(";"):
        segment = part.strip()
        if segment.startswith("evidence_role:"):
            values = segment.removeprefix("evidence_role:")
        elif segment.startswith("evidence_roles:"):
            values = segment.removeprefix("evidence_roles:")
        else:
            continue

        for value in values.split(","):
            role = value.strip()
            if role and role not in roles:
                roles.append(role)
    return tuple(roles)


def _bundle_role(bundle: EvidenceBundle) -> str | None:
    """Return the first explicit role for backward-compatible callers."""

    roles = _bundle_roles(bundle)
    return roles[0] if roles else None


def evaluate_claim_support(
    answer: str,
    bundles: list[EvidenceBundle],
) -> list[ClaimSupport]:
    """Classify each sentence against the exact sources it cites."""

    from phosprocess.rag.citation_binding import _iter_answer_claims

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

    from phosprocess.rag.citation_binding import (
        _iter_answer_claims,
        _supporting_citations,
    )

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
