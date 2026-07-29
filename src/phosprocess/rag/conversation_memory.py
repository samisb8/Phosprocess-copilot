"""Deterministic summary + recent-turn buffer for terminal conversations."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from phosprocess.observability.latency import estimate_tokens
from phosprocess.rag.conversation_state import ConversationState
from phosprocess.rag.schemas import ChatMessage

_CITATION = re.compile(r"\[Source [1-9]\d*\]")
_TECHNICAL_LINE = re.compile(
    r"(?im)^\s*(?:sources?|chunks?|scores?|rrf|reranker|bm25|dense)\s*:.*$"
)
_TECHNICAL_SUFFIX = re.compile(
    r"(?i)\b(?:sources?|chunks?|scores?|rrf|reranker|bm25|dense)\s*:.*$"
)
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """One clean user/assistant exchange."""

    user: str
    assistant: str


@dataclass(frozen=True, slots=True)
class ConversationHistoryContext:
    """Bounded context sent to the model as non-documentary memory."""

    summary: str
    recent_turns: tuple[ConversationTurn, ...]
    summary_token_count: int
    recent_history_token_count: int
    total_token_count: int
    business_state: ConversationState | None = None

    def messages(self) -> list[ChatMessage]:
        """Expose recent turns as ordered conversational messages."""

        result: list[ChatMessage] = []

        for turn in self.recent_turns:
            result.extend(
                [
                    ChatMessage(role="user", content=turn.user),
                    ChatMessage(
                        role="assistant",
                        content=turn.assistant,
                    ),
                ]
            )

        return result


def clean_conversation_text(text: str) -> str:
    """Remove citations and technical provenance from remembered text."""

    cleaned = _CITATION.sub("", text)
    cleaned = _TECHNICAL_LINE.sub("", cleaned)
    cleaned = _TECHNICAL_SUFFIX.sub("", cleaned)
    return _WHITESPACE.sub(" ", cleaned).strip()


def _truncate_to_token_budget(text: str, maximum_tokens: int) -> str:
    """Apply a conservative character-based token budget."""

    if maximum_tokens <= 0:
        return ""

    maximum_characters = maximum_tokens * 4

    if len(text) <= maximum_characters:
        return text

    truncated = text[:maximum_characters].rsplit(" ", 1)[0].rstrip()
    return truncated + "…"


class ConversationMemory:
    """In-memory deterministic summary with a two-turn recent window."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        recent_turns: int = 2,
        summary_max_tokens: int = 300,
        recent_history_max_tokens: int = 500,
        total_history_max_tokens: int = 800,
    ) -> None:
        if recent_turns <= 0:
            raise ValueError("recent_turns doit être positif.")

        if min(
            summary_max_tokens,
            recent_history_max_tokens,
            total_history_max_tokens,
        ) <= 0:
            raise ValueError("Les budgets mémoire doivent être positifs.")

        if (
            summary_max_tokens + recent_history_max_tokens
            > total_history_max_tokens
        ):
            raise ValueError(
                "Les budgets résumé + fenêtre dépassent le budget total."
            )

        self.enabled = enabled
        self.recent_turn_limit = recent_turns
        self.summary_max_tokens = summary_max_tokens
        self.recent_history_max_tokens = recent_history_max_tokens
        self.total_history_max_tokens = total_history_max_tokens
        self._recent_turns: list[ConversationTurn] = []
        self._summary_fragments: list[str] = []
        self._pending_user_message: str | None = None
        self.state = ConversationState()

    def add_user_message(self, content: str) -> None:
        """Stage one user message until its assistant response arrives."""

        if not self.enabled:
            return

        cleaned = clean_conversation_text(content)

        if not cleaned:
            raise ValueError("Le message utilisateur mémorisé est vide.")

        self._pending_user_message = cleaned

    def add_assistant_message(self, content: str) -> None:
        """Complete a staged turn."""

        if not self.enabled:
            return

        if self._pending_user_message is None:
            raise ValueError("Aucun message utilisateur en attente.")

        self.add_turn(self._pending_user_message, content)
        self._pending_user_message = None

    def add_turn(self, user: str, assistant: str) -> None:
        """Add one clean turn and roll evicted turns into the summary."""

        if not self.enabled:
            return

        clean_user = clean_conversation_text(user)
        clean_assistant = clean_conversation_text(assistant)

        if not clean_user or not clean_assistant:
            raise ValueError("Un tour mémorisé ne peut pas être vide.")

        self._recent_turns.append(
            ConversationTurn(
                user=clean_user,
                assistant=clean_assistant,
            )
        )
        self.state.observe_question(clean_user)
        self.update_summary_if_needed()

    def synchronize_business_state(self, state: ConversationState) -> None:
        """Replace the session state with the state resolved during one turn.

        ``build_history_context`` intentionally returns a copy so retrieval cannot
        mutate session memory before a response is validated.  Once a validated
        response is complete, the caller uses this method to persist resolved
        entities and an explicit source lock for the next follow-up.
        """

        self.state = ConversationState(**asdict(state))

    def update_summary_if_needed(self) -> bool:
        """Summarize only turns leaving the recent buffer, without an LLM."""

        updated = False

        while len(self._recent_turns) > self.recent_turn_limit:
            evicted = self._recent_turns.pop(0)
            fragment = (
                f"Sujet : {evicted.user} "
                f"Éléments discutés : {evicted.assistant}"
            )
            self._summary_fragments.append(fragment)
            updated = True

        while (
            estimate_tokens(" ".join(self._summary_fragments))
            > self.summary_max_tokens
            and len(self._summary_fragments) > 1
        ):
            self._summary_fragments.pop(0)

        self.state.rolling_summary = _truncate_to_token_budget(
            " ".join(self._summary_fragments),
            self.summary_max_tokens,
        )

        return updated

    def build_history_context(self) -> ConversationHistoryContext:
        """Build a bounded summary + recent-turn view."""

        if not self.enabled:
            return ConversationHistoryContext(
                summary="",
                recent_turns=(),
                summary_token_count=0,
                recent_history_token_count=0,
                total_token_count=0,
                business_state=ConversationState(),
            )

        summary = _truncate_to_token_budget(
            " ".join(self._summary_fragments),
            self.summary_max_tokens,
        )
        turns: list[ConversationTurn] = []
        remaining_tokens = self.recent_history_max_tokens

        for turn in reversed(self._recent_turns):
            turn_text = f"{turn.user}\n{turn.assistant}"
            turn_tokens = estimate_tokens(turn_text)

            if turn_tokens <= remaining_tokens:
                turns.append(turn)
                remaining_tokens -= turn_tokens
                continue

            if remaining_tokens <= 0:
                break

            assistant_budget = max(
                1,
                remaining_tokens - estimate_tokens(turn.user),
            )
            turns.append(
                ConversationTurn(
                    user=_truncate_to_token_budget(
                        turn.user,
                        remaining_tokens,
                    ),
                    assistant=_truncate_to_token_budget(
                        turn.assistant,
                        assistant_budget,
                    ),
                )
            )
            break

        turns.reverse()
        recent_tokens = sum(
            estimate_tokens(f"{turn.user}\n{turn.assistant}")
            for turn in turns
        )
        summary_tokens = estimate_tokens(summary)
        total_tokens = summary_tokens + recent_tokens

        if total_tokens > self.total_history_max_tokens:
            summary_budget = max(
                0,
                self.total_history_max_tokens - recent_tokens,
            )
            summary = _truncate_to_token_budget(
                summary,
                summary_budget,
            )
            summary_tokens = estimate_tokens(summary)
            total_tokens = summary_tokens + recent_tokens

        return ConversationHistoryContext(
            summary=summary,
            recent_turns=tuple(turns),
            summary_token_count=summary_tokens,
            recent_history_token_count=recent_tokens,
            total_token_count=total_tokens,
            business_state=ConversationState(
                **asdict(self.state)
            ),
        )

    def clear(self) -> None:
        """Reset summary, recent turns and pending references."""

        self._recent_turns.clear()
        self._summary_fragments.clear()
        self._pending_user_message = None
        self.state.clear()

    def get_recent_turns(self) -> list[ConversationTurn]:
        """Return a copy of recent clean turns."""

        return list(self._recent_turns)

    def get_summary(self) -> str:
        """Return the current bounded rolling summary."""

        return self.build_history_context().summary

    def token_usage(self) -> dict[str, int]:
        """Return current memory token estimates."""

        context = self.build_history_context()
        return {
            "summary_tokens": context.summary_token_count,
            "recent_tokens": context.recent_history_token_count,
            "total_tokens": context.total_token_count,
        }

    def export_debug_view(self) -> dict[str, object]:
        """Return only authorized memory data for /history diagnostics."""

        context = self.build_history_context()
        return {
            "strategy": "summary_buffer",
            "summary": context.summary,
            "recent_turns": [
                asdict(turn)
                for turn in context.recent_turns
            ],
            "token_usage": self.token_usage(),
            "business_state": self.state.debug_view(),
        }
