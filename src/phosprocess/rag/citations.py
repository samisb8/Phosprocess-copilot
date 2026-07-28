"""Citation validation for grounded RAG answers."""

from __future__ import annotations

import re

INSUFFICIENT_CONTEXT_ANSWER = (
    "Les passages retrouvés ne permettent pas de répondre précisément "
    "à cette question."
)
INSUFFICIENT_CONTEXT_ANSWERS = {
    INSUFFICIENT_CONTEXT_ANSWER,
    (
        "The retrieved passages do not provide enough information to answer "
        "this question precisely."
    ),
    "لا توفر المقاطع المسترجعة معلومات كافية للإجابة عن هذا السؤال بدقة.",
    (
        "Les documents fournis ne permettent pas de répondre précisément "
        "à cette question."
    ),
}
_EXACT_CITATION_PATTERN = re.compile(r"\[Source ([1-9]\d*)\]")
_BRACKETED_SOURCE_PATTERN = re.compile(
    r"\[[^\]\r\n]*source[^\]\r\n]*\]",
    flags=re.IGNORECASE,
)
_UNBRACKETED_SOURCE_NUMBER_PATTERN = re.compile(
    r"(?<!\[)\bsource\s*\d+\b(?!\])",
    flags=re.IGNORECASE,
)
_ANY_SOURCE_NUMBER_PATTERN = re.compile(
    r"source\s*(\d+)",
    flags=re.IGNORECASE,
)
_NON_DOCUMENTARY_REFERENCE_PATTERN = re.compile(
    r"\[(?:mémoire|memory|historique|conversation)\]",
    flags=re.IGNORECASE,
)
_WHITESPACE = re.compile(r"\s+")


class CitationValidationError(ValueError):
    """Raised when an LLM answer cites invalid or inconsistent sources."""

    def __init__(
        self,
        message: str,
        *,
        detected_citations: list[int] | None = None,
    ) -> None:
        super().__init__(message)
        self.detected_citations = detected_citations or []


def _numbers_from_reference(reference: str) -> list[int]:
    """Extract diagnostic numbers from one malformed reference."""

    return [
        int(match.group(1))
        for match in _ANY_SOURCE_NUMBER_PATTERN.finditer(reference)
    ]


def extract_citations(
    answer: str,
    *,
    available_source_count: int,
) -> list[int]:
    """Extract exact citations, preserving first appearance and rejecting drift."""

    if available_source_count < 0:
        raise ValueError("available_source_count ne peut pas être négatif.")

    normalized_answer = _WHITESPACE.sub(" ", answer).strip()

    for insufficient_answer in INSUFFICIENT_CONTEXT_ANSWERS:
        normalized_insufficient = _WHITESPACE.sub(
            " ",
            insufficient_answer,
        ).strip()

        if (
            normalized_insufficient in normalized_answer
            and normalized_answer != normalized_insufficient
        ):
            raise CitationValidationError(
                "La formulation d'insuffisance doit constituer toute la "
                "réponse et ne peut pas être mélangée à une réponse "
                "affirmative."
            )

    exact_matches = list(_EXACT_CITATION_PATTERN.finditer(answer))
    detected = [int(match.group(1)) for match in exact_matches]
    non_documentary = _NON_DOCUMENTARY_REFERENCE_PATTERN.search(answer)

    if non_documentary is not None:
        raise CitationValidationError(
            f"Référence non documentaire interdite : "
            f"{non_documentary.group(0)!r}. "
            "La mémoire conversationnelle ne constitue pas une source.",
            detected_citations=detected,
        )

    for reference in _BRACKETED_SOURCE_PATTERN.findall(answer):
        if _EXACT_CITATION_PATTERN.fullmatch(reference) is None:
            malformed_numbers = _numbers_from_reference(reference)
            raise CitationValidationError(
                f"Format de citation non autorisé : {reference!r}. "
                "Utilisez exactement [Source N].",
                detected_citations=detected + malformed_numbers,
            )

    malformed_unbracketed = _UNBRACKETED_SOURCE_NUMBER_PATTERN.search(answer)

    if malformed_unbracketed is not None:
        raise CitationValidationError(
            f"Format de citation non autorisé : "
            f"{malformed_unbracketed.group(0)!r}. "
            "Utilisez exactement [Source N].",
            detected_citations=(
                detected
                + _numbers_from_reference(malformed_unbracketed.group(0))
            ),
        )

    citations: list[int] = []

    for source_number in detected:
        if not 1 <= source_number <= available_source_count:
            raise CitationValidationError(
                f"La citation [Source {source_number}] ne correspond à "
                "aucune source documentaire fournie.",
                detected_citations=detected,
            )

        if source_number not in citations:
            citations.append(source_number)

    return citations


def is_controlled_insufficient_answer(answer: str) -> bool:
    """Return whether the answer is the only accepted citation-free fallback."""

    normalized = _WHITESPACE.sub(" ", answer).strip()
    return normalized in INSUFFICIENT_CONTEXT_ANSWERS


def validate_grounded_answer(
    answer: str,
    *,
    available_source_count: int,
) -> tuple[list[int], bool]:
    """Validate citations and the controlled insufficient-context response."""

    citations = extract_citations(
        answer,
        available_source_count=available_source_count,
    )
    insufficient_context = is_controlled_insufficient_answer(answer)

    if not citations and not insufficient_context:
        raise CitationValidationError(
            "Une réponse métier affirmative doit contenir au moins une "
            "citation au format exact [Source N].",
            detected_citations=[],
        )

    return citations, insufficient_context
