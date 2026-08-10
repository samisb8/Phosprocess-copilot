"""Domain-neutral parsing of atomic claims and inline source citations."""

from __future__ import annotations

import re

_CITATION = re.compile(r"\[Source ([1-9]\d*)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_explicitly_cited_clauses(claim: str) -> list[str]:
    """Split semicolon clauses only when each clause carries its own citation."""

    if ";" not in claim:
        return [claim]

    parts = [part.strip() for part in claim.split(";") if part.strip()]
    if len(parts) < 2 or not all(_CITATION.search(part) for part in parts):
        return [claim]

    return parts


def _split_line_claims(line: str) -> list[str]:
    """Split prose without mistaking a numbered-list prefix for a sentence."""

    protected = re.sub(r"^(\s*\d+)\.\s+", r"\1<LIST_DOT> ", line, count=1)
    return [
        claim.replace("<LIST_DOT>", ".").strip()
        for claim in _SENTENCE.split(protected)
        if claim.strip()
    ]


def iter_answer_claims(answer: str) -> tuple[str, ...]:
    """Return non-empty answer claims without evaluating their meaning."""

    claims: list[str] = []
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for sentence in _split_line_claims(line):
            claims.extend(_split_explicitly_cited_clauses(sentence))
    return tuple(claims)


# Transitional private alias for callers that have not migrated yet.
_iter_answer_claims = iter_answer_claims
