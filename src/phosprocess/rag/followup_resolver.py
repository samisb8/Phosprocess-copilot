"""Conservative standalone-query resolution without adding domain facts."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from phosprocess.rag.conversation_state import ConversationState

_FOLLOWUP = re.compile(
    r"^(?:et\b|and\b|then\b|ensuite\b|why\b|pourquoi\b|comment\b|how\b)|"
    r"\b(?:cela|ça|ceci|celui-ci|celle-ci|cette dernière|this|that|it|its|"
    r"elle|il|lui|son|sa|ses|ce système|cet équipement|the system|"
    r"the equipment|هي|هو|هذا|هذه)\b",
    flags=re.IGNORECASE,
)
_AUTONOMOUS_TECHNICAL = re.compile(
    r"\b(?:évaporateur|evaporator|échangeur|exchanger|réacteur|reactor|"
    r"pompe|pump|gypse|gypsum|thermodynam|enthalp|sursaturation|"
    r"supersaturation|pid|mpc|pression|pressure|température|temperature|"
    r"bouilleur|flash chamber|vapor body|condenseur|condenser)\b",
    flags=re.IGNORECASE,
)
_FRENCH = re.compile(
    r"[àâçéèêëîïôùûüÿœ]|\b(?:et|pourquoi|comment|est|elle|nécessaire)\b",
    re.I,
)
_ARABIC = re.compile(r"[\u0600-\u06ff]")

_FRENCH_SUBJECTS = {
    "pompe de circulation": ("la pompe de circulation", "elle"),
    "évaporateur à circulation forcée": (
        "l’évaporateur à circulation forcée",
        "il",
    ),
    "évaporateur": ("l’évaporateur", "il"),
    "échangeur thermique": ("l’échangeur thermique", "il"),
    "chambre de vaporisation": ("la chambre de vaporisation", "elle"),
    "pompe à vide": ("la pompe à vide", "elle"),
    "condenseur": ("le condenseur", "il"),
    "séparateur": ("le séparateur", "il"),
    "réacteur": ("le réacteur", "il"),
    "filtre": ("le filtre", "il"),
}

_ENTITY_TRANSLATIONS = {
    "pompe de circulation": "circulation pump",
    "évaporateur à circulation forcée": "forced-circulation evaporator",
    "évaporateur": "evaporator",
    "échangeur thermique": "heat exchanger",
    "chambre de vaporisation": "vapor body",
    "pompe à vide": "vacuum pump",
    "condenseur": "condenser",
    "séparateur": "separator",
    "réacteur": "reactor",
    "filtre": "filter",
}

_ARABIC_ENTITIES = {
    "pompe de circulation": "مضخة الدوران",
    "évaporateur à circulation forcée": "المبخر ذو الدوران القسري",
    "évaporateur": "المبخر",
    "échangeur thermique": "المبادل الحراري",
    "chambre de vaporisation": "غرفة التبخير",
    "pompe à vide": "مضخة التفريغ",
    "condenseur": "المكثف",
    "séparateur": "الفاصل",
}


@dataclass(frozen=True, slots=True)
class StandaloneQueryResolution:
    """Traceable resolution outcome for observability."""

    original_question: str
    standalone_query: str
    followup_detected: bool
    resolver_type: str
    entities_used: tuple[str, ...]
    focus_entity: str | None = None
    intent_hint: str | None = None
    inherited_source_mode: str | None = None
    requires_llm_resolution: bool = False
    inherits_source: bool = False
    explicit_source: str | None = None
    resolver_latency_ms: float = 0.0
    resolver_llm_call_count: int = 0


class _QueryResolverPayload(BaseModel):
    """Strict answer-free output for one ambiguous conversational query."""

    model_config = ConfigDict(extra="forbid")

    standalone_query: str = Field(min_length=1, max_length=2000)
    is_followup: bool
    inherits_source: bool
    explicit_source: str | None = None


class _OllamaQueryResolverPayload(BaseModel):
    """Grammar-compatible transport; strict validation follows JSON parsing."""

    standalone_query: str
    is_followup: bool
    inherits_source: bool
    explicit_source: str | None = None


_QUERY_RESOLVER_SYSTEM_PROMPT = "\n".join(
    (
        "You resolve ambiguous conversational questions for document retrieval.",
        "Return a standalone search question, never an answer.",
        "The standalone question must remain fully retrievable without access to history.",
        "Carry forward every subject or topic needed to identify what is being asked.",
        "Conversation history is for reference resolution only and is never evidence.",
        "Do not copy factual claims from prior assistant answers and do not add domain facts.",
        "Preserve the language of the current question.",
        "Resolve pronouns, omitted subjects, short continuations, and topic continuity.",
        "A clearly new topic is not a follow-up.",
        "An explicit source in the current question always overrides an inherited source.",
        "Set inherits_source only for a genuine follow-up with no new explicit source.",
        "explicit_source must be null unless the current question explicitly names one.",
        "Return exactly one JSON object matching the supplied schema and no reasoning.",
    )
)


def _entity_for_question(entity: str, question: str) -> str:
    if _ARABIC.search(question):
        return _ARABIC_ENTITIES.get(entity, entity)
    if _FRENCH.search(question):
        return entity
    return _ENTITY_TRANSLATIONS.get(entity, entity)


def _infer_intent_hint(question: str, focus_entity: str | None) -> str | None:
    """Keep compatibility without encoding domain-specific answer intents."""
    del question, focus_entity
    return None


def _rewrite_pronoun_followup(question: str, focus_entity: str) -> str | None:
    """Rewrite common French, English and Arabic references around the focus."""

    entity = _entity_for_question(focus_entity, question)
    normalized = question.strip()

    if re.fullmatch(r"(?:et\s+)?pourquoi\s*[?.!]*", normalized, flags=re.I):
        subject, pronoun = _FRENCH_SUBJECTS.get(focus_entity, (entity, "il"))
        return f"Pourquoi {subject} est-{pronoun} nécessaire ?"

    match = re.fullmatch(
        r"(?:et\s+)?pourquoi\s+est-(?:elle|il)\s+(.+?)[?.!]*",
        normalized,
        flags=re.I,
    )
    if match is not None:
        subject, pronoun = _FRENCH_SUBJECTS.get(focus_entity, (entity, "il"))
        return f"Pourquoi {subject} est-{pronoun} {match.group(1).strip()} ?"

    if re.fullmatch(r"(?:and\s+)?why\s*[?.!]*", normalized, flags=re.I):
        return f"Why is the {entity} necessary?"

    match = re.fullmatch(
        r"(?:and\s+)?why\s+is\s+(?:it|this|that)\s+(.+?)[?.!]*",
        normalized,
        flags=re.I,
    )
    if match is not None:
        return f"Why is the {entity} {match.group(1).strip()}?"

    match = re.fullmatch(
        r"how\s+does\s+(?:it|this|that)\s+(.+?)[?.!]*",
        normalized,
        flags=re.I,
    )
    if match is not None:
        return f"How does the {entity} {match.group(1).strip()}?"

    if _ARABIC.search(normalized):
        if re.fullmatch(r"(?:و)?لماذا(?: هي| هو)? ضروري(?:ة)?[؟?.!]*", normalized):
            return f"لماذا تعد {entity} ضرورية؟"
        replaced = re.sub(r"\b(?:هي|هو|هذا|هذه)\b", entity, normalized, count=1)
        if replaced != normalized:
            return replaced

    replaced = re.sub(
        r"\b(?:it|this|that)\b",
        f"the {entity}",
        normalized,
        count=1,
        flags=re.I,
    )
    if replaced != normalized:
        return replaced

    replaced = re.sub(
        r"\b(?:elle|il|cela|ça|ceci|celle-ci|celui-ci)\b",
        entity,
        normalized,
        count=1,
        flags=re.I,
    )
    if replaced != normalized:
        return replaced
    return None


def resolve_standalone_query(
    question: str,
    *,
    state: ConversationState,
) -> StandaloneQueryResolution:
    """Resolve vague references from explicit state, without adding facts."""

    original = question.strip()
    if not original:
        raise ValueError("La question ne peut pas être vide.")

    is_short = len(original.split()) <= 7
    has_followup_marker = _FOLLOWUP.search(original) is not None
    autonomous = _AUTONOMOUS_TECHNICAL.search(original) is not None and not has_followup_marker
    if autonomous or (not has_followup_marker and not is_short):
        state.observe_question(original)
        state.record_resolution(original)
        return StandaloneQueryResolution(
            original_question=original,
            standalone_query=original,
            followup_detected=False,
            resolver_type="none",
            entities_used=(),
            focus_entity=state.focus_entity,
            intent_hint=_infer_intent_hint(original, state.focus_entity),
        )

    focus_entity = state.focus_entity or state.current_equipment
    if focus_entity:
        rewritten = _rewrite_pronoun_followup(original, focus_entity)
        if rewritten:
            intent_hint = _infer_intent_hint(rewritten, focus_entity)
            state.observe_question(original)
            state.record_resolution(rewritten)
            return StandaloneQueryResolution(
                original_question=original,
                standalone_query=rewritten,
                followup_detected=True,
                resolver_type="focus_entity",
                entities_used=(focus_entity,),
                focus_entity=focus_entity,
                intent_hint=intent_hint,
                inherited_source_mode=(
                    state.current_source_mode if state.source_scope_explicit else None
                ),
                requires_llm_resolution=True,
            )

    entity_values = [
        focus_entity,
        state.current_process,
        state.current_equipment,
        state.current_fluid,
        state.current_operation,
        state.current_variable,
        state.current_problem,
    ]
    entities = tuple(dict.fromkeys(value for value in entity_values if value))

    if not entities:
        state.observe_question(original)
        state.record_resolution(original)
        return StandaloneQueryResolution(
            original_question=original,
            standalone_query=original,
            followup_detected=has_followup_marker or is_short,
            resolver_type="unresolved",
            entities_used=(),
            intent_hint=_infer_intent_hint(original, None),
            requires_llm_resolution=True,
        )

    context = ", ".join(entities)
    standalone = f"{original} Contexte technique : {context}."
    state.observe_question(original)
    state.record_resolution(standalone)
    return StandaloneQueryResolution(
        original_question=original,
        standalone_query=standalone,
        followup_detected=True,
        resolver_type="conversation_state",
        entities_used=entities,
        focus_entity=focus_entity,
        intent_hint=_infer_intent_hint(standalone, focus_entity),
        inherited_source_mode=(state.current_source_mode if state.source_scope_explicit else None),
        # Broad state was appended, but the actual referent was not bound.
        # A prior user turn can resolve this through the structured fallback.
        requires_llm_resolution=True,
    )


def resolve_ambiguous_query_with_llm(
    question: str,
    *,
    state: ConversationState,
    llm: Any,
) -> StandaloneQueryResolution:
    """Use one answer-free LLM call only when deterministic resolution is ambiguous."""

    previous_user_questions = tuple(state.recent_turns)
    previous_standalone_query = state.last_standalone_query
    active_explicit_source = (
        state.current_source_mode if state.source_scope_explicit else None
    )
    deterministic = resolve_standalone_query(question, state=state)
    if not deterministic.requires_llm_resolution or not previous_user_questions:
        return deterministic

    state_view = {
        "active_explicit_source": active_explicit_source,
        "last_standalone_query": previous_standalone_query,
    }
    history = "\n".join(
        f"User turn {index}: {value}"
        for index, value in enumerate(previous_user_questions[-4:], start=1)
    )
    user_prompt = "\n\n".join(
        (
            f"PREVIOUS USER QUESTIONS\n{history}",
            "CONVERSATION STATE\n"
            + json.dumps(state_view, ensure_ascii=False, sort_keys=True),
            f"CURRENT QUESTION\n{question.strip()}",
            (
                "Return standalone_query, is_followup, inherits_source, and "
                "explicit_source. Do not answer the question."
            ),
        )
    )
    started = time.perf_counter()
    try:
        _transport, raw = llm.chat_json_with_raw(
            user_prompt=user_prompt,
            system_prompt=_QUERY_RESOLVER_SYSTEM_PROMPT,
            response_model=_OllamaQueryResolverPayload,
        )
        payload = _QueryResolverPayload.model_validate(json.loads(raw))
    except Exception:
        return deterministic
    latency_ms = (time.perf_counter() - started) * 1000.0
    standalone = payload.standalone_query.strip()
    if not standalone or "[Source" in standalone or "EVIDENCE" in standalone.upper():
        return deterministic

    state.record_resolution(standalone)
    return StandaloneQueryResolution(
        original_question=question.strip(),
        standalone_query=standalone,
        followup_detected=payload.is_followup,
        resolver_type="llm_ambiguous_followup",
        entities_used=(),
        focus_entity=state.focus_entity,
        intent_hint=None,
        inherited_source_mode=(
            state.current_source_mode
            if payload.inherits_source and state.source_scope_explicit
            else None
        ),
        requires_llm_resolution=False,
        inherits_source=payload.inherits_source,
        explicit_source=(
            payload.explicit_source.strip()
            if payload.explicit_source and payload.explicit_source.strip()
            else None
        ),
        resolver_latency_ms=latency_ms,
        resolver_llm_call_count=1,
    )
