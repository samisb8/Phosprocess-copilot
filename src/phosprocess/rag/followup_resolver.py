"""Deterministic business-aware standalone-query resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass

from phosprocess.rag.conversation_state import ConversationState

_FOLLOWUP = re.compile(
    r"^(?:et\b|and\b|why\b|pourquoi\b)|"
    r"\b(?:cela|ça|ceci|this|that|it|its|elle|il|lui|son|sa|ses|"
    r"ce système|cet équipement|the system|the equipment)\b",
    flags=re.IGNORECASE,
)
_AUTONOMOUS_TECHNICAL = re.compile(
    r"\b(?:évaporateur|evaporator|échangeur|exchanger|réacteur|reactor|"
    r"pompe|pump|gypse|gypsum|thermodynam|enthalp|sursaturation|"
    r"supersaturation|pid|mpc|pression|pressure|température|temperature)\b",
    flags=re.IGNORECASE,
)
_STATE_DEPENDENT_INTENT = re.compile(
    r"\b(?:contr[oô]l\w*|r[eé]gul\w*|manipulat\w*|"
    r"maint(?:ain|enir)\w*|stabilis\w*|consigne|setpoint|sortie|outlet|"
    r"n[eé]cessaire|necessary|send|ramen\w*|return)\b",
    flags=re.IGNORECASE,
)
_FRENCH = re.compile(r"[àâçéèêëîïôùûüÿœ]|\b(?:et|pourquoi|comment|est|elle)\b", re.I)

_FRENCH_SUBJECTS = {
    "pompe de circulation": ("la pompe de circulation", "elle"),
    "évaporateur à circulation forcée": ("l’évaporateur à circulation forcée", "il"),
    "évaporateur": ("l’évaporateur", "il"),
    "échangeur thermique": ("l’échangeur thermique", "il"),
    "réacteur": ("le réacteur", "il"),
    "filtre": ("le filtre", "il"),
}

_ENTITY_TRANSLATIONS = {
    "pompe de circulation": "circulation pump",
    "évaporateur à circulation forcée": "forced-circulation evaporator",
    "évaporateur": "evaporator",
    "échangeur thermique": "heat exchanger",
    "réacteur": "reactor",
    "filtre": "filter",
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
    requires_llm_resolution: bool = False


def _entity_for_question(entity: str, question: str) -> str:
    if _FRENCH.search(question):
        return entity
    return _ENTITY_TRANSLATIONS.get(entity, entity)


def _rewrite_pronoun_followup(question: str, focus_entity: str) -> str | None:
    """Rewrite common French/English pronoun forms around the focus entity."""

    entity = _entity_for_question(focus_entity, question)
    normalized = question.strip()

    match = re.fullmatch(
        r"(?:et\s+)?pourquoi\s+est-(?:elle|il)\s+(.+?)[?.!]*",
        normalized,
        flags=re.I,
    )
    if match is not None:
        subject, pronoun = _FRENCH_SUBJECTS.get(
            focus_entity,
            (entity, "il"),
        )
        return f"Pourquoi {subject} est-{pronoun} {match.group(1).strip()} ?"

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
        r"\b(?:elle|il)\b",
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
    autonomous = (
        _AUTONOMOUS_TECHNICAL.search(original) is not None
        and not has_followup_marker
    )
    state_entities_available = any(
        (
            state.focus_entity,
            state.current_process,
            state.current_equipment,
            state.current_fluid,
            state.current_operation,
            state.current_variable,
            state.current_problem,
        )
    )
    needs_state_context = (
        state_entities_available
        and _STATE_DEPENDENT_INTENT.search(original) is not None
        and _AUTONOMOUS_TECHNICAL.search(original) is None
    )

    if not needs_state_context and (
        autonomous or (not has_followup_marker and not is_short)
    ):
        state.observe_question(original)
        state.record_resolution(original)
        return StandaloneQueryResolution(
            original_question=original,
            standalone_query=original,
            followup_detected=False,
            resolver_type="none",
            entities_used=(),
            focus_entity=state.focus_entity,
        )

    focus_entity = state.focus_entity or state.current_equipment
    if focus_entity:
        rewritten = _rewrite_pronoun_followup(original, focus_entity)
        if rewritten:
            state.observe_question(original)
            state.record_resolution(rewritten)
            return StandaloneQueryResolution(
                original_question=original,
                standalone_query=rewritten,
                followup_detected=True,
                resolver_type="focus_entity",
                entities_used=(focus_entity,),
                focus_entity=focus_entity,
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
    )
