"""Deterministic adaptive routing before any domain-context injection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class RequestPath(StrEnum):
    """Execution paths available to the conversational pipeline."""

    DOMAIN_RAG = "domain_rag"
    DIRECT_LLM = "direct_no_retrieval"


class DirectIntent(StrEnum):
    """Self-contained intents that do not need documentary retrieval."""

    TRANSLATION = "translation"
    REWRITE = "rewrite"
    SUMMARIZATION = "summarization"
    GENERAL = "general"
    CONVERSATION = "conversation"


@dataclass(frozen=True, slots=True)
class AdaptiveRouteDecision:
    """One deterministic routing decision made on the raw current request."""

    path: RequestPath
    reason: str
    direct_intent: DirectIntent | None = None
    requested_output_language: str | None = None

    @property
    def retrieval_required(self) -> bool:
        """Return whether the technical corpus must be queried."""

        return self.path is RequestPath.DOMAIN_RAG


_POLITE_PREFIX = (
    r"(?:(?:peux[- ]tu|pouvez[- ]vous|can you|could you|please)\s+)?"
)
_TRANSLATION = re.compile(
    r"^\s*" + _POLITE_PREFIX
    + r"(?:me\s+)?(?:traduis|traduire|traduisez|translate|translation|ترجم)\b",
    re.IGNORECASE,
)
_REWRITE = re.compile(
    r"^\s*(?:réécris|reecris|reformule|corrige|améliore|ameliore|"
    r"rewrite|rephrase|proofread)\b",
    re.IGNORECASE,
)
_SUMMARIZE = re.compile(
    r"^\s*(?:résume|resume|synthétise|synthetise|summarize|summarise)\b",
    re.IGNORECASE,
)
_GREETING = re.compile(
    r"^\s*(?:bonjour|bonsoir|salut|hello|hi|hey|merci|thank you|thanks)"
    r"[\s!.?]*$",
    re.IGNORECASE,
)
_GENERAL_WITHOUT_RETRIEVAL = re.compile(
    r"^\s*(?:question\s+g[eé]n[eé]rale\s*:\s*|general\s+question\s*:\s*|"
    r"qui\s+est\b|who\s+is\b|quelle\s+est\s+la\s+capitale\b|"
    r"what\s+is\s+the\s+capital\b|écris\s+(?:un|une)\b|"
    r"write\s+(?:a|an)\b)",
    re.IGNORECASE,
)
_DOMAIN_SIGNAL = re.compile(
    r"\b(?:acide|acid|phosphorique|phosphoric|évaporateur|evaporator|"
    r"pompe|pump|échangeur|exchanger|vapeur|steam|vapor|gypse|gypsum|"
    r"filtration|cristallisation|crystallization|réacteur|reactor|"
    r"thermodynamique|thermodynamic|enthalpie|enthalpy|pression|pressure|"
    r"température|temperature|corrosion|fouling|scaling|boiler|bouilleur|"
    r"bilan|balance|procédé|process|pid|mpc|contrôle|control)\b",
    re.IGNORECASE,
)
_EXPLICIT_SOURCE_MODE = {
    "becker",
    "report",
    "thermodynamics",
    "heat_transfer",
    "perry",
    "crystallization",
    "control",
    "transport",
}
_TARGET_LANGUAGES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(?:en|into|to)\s+(?:anglais|english)\b",
            re.IGNORECASE,
        ),
        "en",
    ),
    (
        re.compile(
            r"\b(?:en|into|to)\s+(?:français|francais|french)\b",
            re.IGNORECASE,
        ),
        "fr",
    ),
    (
        re.compile(
            r"\b(?:en|into|to)\s+(?:arabe|arabic)\b",
            re.IGNORECASE,
        ),
        "ar",
    ),
)


def _requested_output_language(question: str) -> str | None:
    for pattern, language in _TARGET_LANGUAGES:
        if pattern.search(question):
            return language

    return None


def _contains_transform_payload(question: str) -> bool:
    """Avoid bypassing retrieval for an empty anaphoric command."""

    if re.search(r"[\"'«»].+?[\"'«»]", question):
        return True

    if ":" in question and question.split(":", 1)[1].strip():
        return True

    words = re.findall(r"(?u)\b\w+\b", question)
    return len(words) >= 4


def decide_request_path(
    question: str,
    *,
    source_mode: str = "auto",
) -> AdaptiveRouteDecision:
    """Route before follow-up resolution can inject phosphoric context.

    The direct path is deliberately limited to self-contained language tasks
    and greetings. Open factual questions continue through the grounded RAG
    path unless a future explicit general-knowledge mode is introduced.
    """

    normalized_mode = source_mode.strip().casefold()

    if normalized_mode == "automatic":
        normalized_mode = "auto"

    if normalized_mode in _EXPLICIT_SOURCE_MODE:
        return AdaptiveRouteDecision(
            path=RequestPath.DOMAIN_RAG,
            reason="explicit_document_scope",
        )

    if _TRANSLATION.search(question) and _contains_transform_payload(question):
        return AdaptiveRouteDecision(
            path=RequestPath.DIRECT_LLM,
            reason="self_contained_translation",
            direct_intent=DirectIntent.TRANSLATION,
            requested_output_language=_requested_output_language(question),
        )

    if _REWRITE.search(question) and _contains_transform_payload(question):
        return AdaptiveRouteDecision(
            path=RequestPath.DIRECT_LLM,
            reason="self_contained_rewrite",
            direct_intent=DirectIntent.REWRITE,
        )

    if _SUMMARIZE.search(question) and _contains_transform_payload(question):
        return AdaptiveRouteDecision(
            path=RequestPath.DIRECT_LLM,
            reason="self_contained_summarization",
            direct_intent=DirectIntent.SUMMARIZATION,
        )

    if _GREETING.fullmatch(question):
        return AdaptiveRouteDecision(
            path=RequestPath.DIRECT_LLM,
            reason="simple_conversation",
            direct_intent=DirectIntent.CONVERSATION,
        )

    if (
        _GENERAL_WITHOUT_RETRIEVAL.search(question)
        and not _DOMAIN_SIGNAL.search(question)
    ):
        return AdaptiveRouteDecision(
            path=RequestPath.DIRECT_LLM,
            reason="obvious_general_request",
            direct_intent=DirectIntent.GENERAL,
        )

    return AdaptiveRouteDecision(
        path=RequestPath.DOMAIN_RAG,
        reason="documentary_or_ambiguous_request",
    )
