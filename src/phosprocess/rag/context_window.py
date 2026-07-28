"""Question-focused context windows for the five frozen-v3 sources."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from phosprocess.observability.latency import estimate_tokens

_WORD = re.compile(r"\b[\w%+-]{3,}\b", flags=re.UNICODE)
_SENTENCE = re.compile(r".+?(?:[.!?](?=\s|$)|\n+|$)", flags=re.DOTALL)
_WHITESPACE = re.compile(r"[ \t]+")
_STOPWORDS = {
    "alors",
    "avec",
    "cette",
    "dans",
    "des",
    "elle",
    "elles",
    "est",
    "ils",
    "les",
    "leur",
    "mais",
    "pour",
    "pourquoi",
    "quel",
    "quelle",
    "quels",
    "quelles",
    "rôle",
    "sont",
    "sur",
    "une",
    "what",
    "when",
    "where",
    "which",
    "with",
}


@dataclass(frozen=True, slots=True)
class PreparedDocumentContext:
    """Five bounded source passages and their aggregate size."""

    texts: tuple[str, ...]
    tokens_per_source: tuple[int, ...]
    total_tokens: int


def query_terms(question: str) -> set[str]:
    """Extract generic content terms without benchmark-specific knowledge."""

    return {
        token.casefold()
        for token in _WORD.findall(question)
        if token.casefold() not in _STOPWORDS
    }


def _remove_repeated_lines(text: str) -> str:
    """Drop exact repeated non-empty lines while preserving first occurrence."""

    seen: set[str] = set()
    lines: list[str] = []

    for line in text.splitlines():
        normalized = _WHITESPACE.sub(" ", line).strip()

        if not normalized:
            continue

        key = normalized.casefold()

        if key in seen:
            continue

        seen.add(key)
        lines.append(normalized)

    return "\n".join(lines)


def select_relevant_window(
    text: str,
    question: str,
    *,
    maximum_tokens: int,
) -> str:
    """Select a sentence window centered on query-related content."""

    if maximum_tokens <= 0:
        raise ValueError("maximum_tokens doit être positif.")

    cleaned = _remove_repeated_lines(text)
    maximum_characters = maximum_tokens * 4

    if len(cleaned) <= maximum_characters:
        return cleaned

    terms = query_terms(question)
    sentences = [
        match.group(0).strip()
        for match in _SENTENCE.finditer(cleaned)
        if match.group(0).strip()
    ]

    if not sentences:
        midpoint = len(cleaned) // 2
        start = max(0, midpoint - maximum_characters // 2)
        return cleaned[start : start + maximum_characters].strip()

    def sentence_score(sentence: str) -> tuple[int, int]:
        lowered = sentence.casefold()
        term_score = sum(
            1
            for term in terms
            if term in lowered
        )
        technical_score = len(
            re.findall(r"\d|p2o5|so4|caso4|%", lowered)
        )
        return term_score, technical_score

    anchor = max(
        range(len(sentences)),
        key=lambda index: (
            sentence_score(sentences[index]),
            -abs(index - len(sentences) // 2),
        ),
    )
    selected = [sentences[anchor]]
    left = anchor - 1
    right = anchor + 1

    while left >= 0 or right < len(sentences):
        candidates: list[tuple[int, str, str]] = []

        if left >= 0:
            candidates.append(
                (
                    sentence_score(sentences[left])[0],
                    "left",
                    sentences[left],
                )
            )

        if right < len(sentences):
            candidates.append(
                (
                    sentence_score(sentences[right])[0],
                    "right",
                    sentences[right],
                )
            )

        _, side, sentence = max(candidates, key=lambda item: item[0])
        candidate_text = (
            " ".join([sentence, *selected])
            if side == "left"
            else " ".join([*selected, sentence])
        )

        if len(candidate_text) > maximum_characters:
            break

        if side == "left":
            selected.insert(0, sentence)
            left -= 1
        else:
            selected.append(sentence)
            right += 1

    result = " ".join(selected).strip()

    if len(result) > maximum_characters:
        result = result[:maximum_characters].rsplit(" ", 1)[0].rstrip()

    return result


def prepare_document_context(
    source_texts: Sequence[str],
    question: str,
    *,
    maximum_tokens_per_source: int,
    maximum_total_tokens: int,
) -> PreparedDocumentContext:
    """Bound five source passages without changing their identifiers or order."""

    if not source_texts:
        raise ValueError("Le contexte documentaire ne peut pas être vide.")

    if maximum_total_tokens < len(source_texts):
        raise ValueError(
            "Le budget documentaire total est insuffisant pour les sources."
        )

    prepared: list[str] = []
    token_counts: list[int] = []
    remaining_total = maximum_total_tokens

    for index, text in enumerate(source_texts):
        remaining_sources = len(source_texts) - index
        fair_share = max(1, remaining_total // remaining_sources)
        source_budget = min(maximum_tokens_per_source, fair_share)
        excerpt = select_relevant_window(
            text,
            question,
            maximum_tokens=source_budget,
        )
        tokens = estimate_tokens(excerpt)
        prepared.append(excerpt)
        token_counts.append(tokens)
        remaining_total -= tokens

    return PreparedDocumentContext(
        texts=tuple(prepared),
        tokens_per_source=tuple(token_counts),
        total_tokens=sum(token_counts),
    )
