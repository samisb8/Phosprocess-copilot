"""Current-question language detection for French, English and Arabic."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

_ARABIC = re.compile(r"[\u0600-\u06ff]")
_WORDS = re.compile(r"(?u)[^\W\d_]+")
_FRENCH_WORDS = {
    "à",
    "comment",
    "dans",
    "de",
    "des",
    "est",
    "et",
    "la",
    "le",
    "les",
    "pourquoi",
    "quel",
    "quelle",
    "quels",
    "rôle",
    "une",
}
_ENGLISH_WORDS = {
    "and",
    "does",
    "how",
    "in",
    "is",
    "of",
    "role",
    "the",
    "what",
    "why",
    "with",
}


class ResponseLanguage(StrEnum):
    """Supported response languages."""

    FRENCH = "fr"
    ENGLISH = "en"
    ARABIC = "ar"

    @property
    def prompt_name(self) -> str:
        return {
            ResponseLanguage.FRENCH: "French",
            ResponseLanguage.ENGLISH: "English",
            ResponseLanguage.ARABIC: "Arabic",
        }[self]


@dataclass(frozen=True, slots=True)
class LanguageDecision:
    """Detected or user-forced response language."""

    language: ResponseLanguage
    confidence: float
    method: str


def normalize_language_mode(mode: str) -> str:
    normalized = mode.strip().casefold()

    if normalized not in {"auto", "fr", "en", "ar"}:
        raise ValueError("Mode langue invalide. Utilisez auto, fr, en ou ar.")

    return normalized


def detect_response_language(
    question: str,
    *,
    last_explicit_language: str | None = None,
    mode: str = "auto",
) -> LanguageDecision:
    """Prioritize the current question and use state only when very short."""

    normalized_mode = normalize_language_mode(mode)

    if normalized_mode != "auto":
        return LanguageDecision(
            language=ResponseLanguage(normalized_mode),
            confidence=1.0,
            method="forced",
        )

    if _ARABIC.search(question):
        return LanguageDecision(
            language=ResponseLanguage.ARABIC,
            confidence=1.0,
            method="unicode_arabic",
        )

    words = [word.casefold() for word in _WORDS.findall(question)]
    french_score = sum(word in _FRENCH_WORDS for word in words)
    english_score = sum(word in _ENGLISH_WORDS for word in words)

    if french_score > english_score:
        return LanguageDecision(
            ResponseLanguage.FRENCH,
            min(1.0, 0.55 + french_score * 0.08),
            "lexical",
        )

    if english_score > french_score:
        return LanguageDecision(
            ResponseLanguage.ENGLISH,
            min(1.0, 0.55 + english_score * 0.08),
            "lexical",
        )

    if len(words) <= 4 and last_explicit_language in {"fr", "en", "ar"}:
        return LanguageDecision(
            ResponseLanguage(last_explicit_language),
            0.5,
            "last_explicit_short_question",
        )

    return LanguageDecision(
        ResponseLanguage.FRENCH,
        0.35,
        "default_french",
    )
