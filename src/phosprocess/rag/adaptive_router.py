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


_POLITE_PREFIX = r"(?:(?:peux[- ]tu|pouvez[- ]vous|can you|could you|please)\s+)?"
_TRANSLATION = re.compile(
    r"^\s*"
    + _POLITE_PREFIX
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
    """Choose retrieval without embedding domain knowledge in the router.

    Only self-contained language transformations and greetings bypass the
    documentary RAG. Every other factual or ambiguous request goes through
    retrieval, where corpus evidence decides what is relevant.
    """

    normalized_mode = source_mode.strip().casefold()
    if normalized_mode == "automatic":
        normalized_mode = "auto"

    if normalized_mode != "auto":
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

    return AdaptiveRouteDecision(
        path=RequestPath.DOMAIN_RAG,
        reason="documentary_or_ambiguous_request",
    )
