"""Deterministic technical question classification and answer policy."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum


class QuestionType(StrEnum):
    """Supported response and evidence-expansion intents."""

    DEFINITION = "definition"
    EXPLANATION = "explanation"
    PROCESS_FLOW = "process_flow"
    PROCEDURE = "procedure"
    COMPARISON = "comparison"
    THERMODYNAMIC_RELATION = "thermodynamic_relation"
    BALANCE = "balance"
    EQUATION_EXPLANATION = "equation_explanation"
    TABLE_QUESTION = "table_question"
    CALCULATION = "calculation"
    MOMENTUM_DIFFUSION = "momentum_diffusion"
    TROUBLESHOOTING = "troubleshooting"
    CONTROL_STRATEGY = "control_strategy"
    SAFETY = "safety"
    PLANT_SPECIFIC = "plant_specific"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class AnswerPolicy:
    """Formatting and size limits selected from the question type."""

    numbered_steps: bool = False
    define_variables: bool = False
    preserve_units: bool = False
    show_assumptions: bool = False
    organize_as_causes_effects_actions: bool = False


ANSWER_POLICIES: dict[QuestionType, AnswerPolicy] = {
    QuestionType.DEFINITION: AnswerPolicy(),
    QuestionType.EXPLANATION: AnswerPolicy(),
    QuestionType.PROCESS_FLOW: AnswerPolicy(numbered_steps=True),
    QuestionType.PROCEDURE: AnswerPolicy(numbered_steps=True),
    QuestionType.COMPARISON: AnswerPolicy(),
    QuestionType.BALANCE: AnswerPolicy(
        define_variables=True,
        preserve_units=True,
    ),
    QuestionType.THERMODYNAMIC_RELATION: AnswerPolicy(
        define_variables=True,
        preserve_units=True,
    ),
    QuestionType.EQUATION_EXPLANATION: AnswerPolicy(
        define_variables=True,
        preserve_units=True,
    ),
    QuestionType.TABLE_QUESTION: AnswerPolicy(),
    QuestionType.CALCULATION: AnswerPolicy(
        show_assumptions=True,
        preserve_units=True,
    ),
    QuestionType.MOMENTUM_DIFFUSION: AnswerPolicy(
        define_variables=True,
        preserve_units=True,
    ),
    QuestionType.TROUBLESHOOTING: AnswerPolicy(
        organize_as_causes_effects_actions=True,
    ),
    QuestionType.CONTROL_STRATEGY: AnswerPolicy(),
    QuestionType.SAFETY: AnswerPolicy(),
    QuestionType.PLANT_SPECIFIC: AnswerPolicy(),
    QuestionType.AMBIGUOUS: AnswerPolicy(),
}


@dataclass(frozen=True, slots=True)
class QuestionClassification:
    """Question type, confidence and deterministic cues."""

    question_type: QuestionType
    confidence: float
    cues: tuple[str, ...]
    answer_policy: AnswerPolicy


def _normalize_question(question: str) -> str:
    """Normalize accents, apostrophes and informal spacing for rules."""

    value = question.strip().casefold().replace("’", "'").replace("‘", "'")
    decomposed = unicodedata.normalize("NFKD", value)
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    without_apostrophes = re.sub(r"['’‘-]+", " ", without_marks)
    return re.sub(r"\s+", " ", without_apostrophes).strip()


def classify_question(question: str) -> QuestionClassification:
    """Classify in priority order without an LLM call."""

    normalized = _normalize_question(question)

    if not normalized:
        raise ValueError("La question ne peut pas être vide.")

    rules: tuple[tuple[QuestionType, re.Pattern[str], str], ...] = (
        (
            QuestionType.AMBIGUOUS,
            re.compile(
                r"^(?:what is|qu est ce que|c est quoi|cest quoi)\s+"
                r"(?:the |le |la )?boiler\??$"
            ),
            "ambiguous_boiler",
        ),
        (
            QuestionType.CALCULATION,
            re.compile(r"\b(?:calculate|compute|calcule|determine numeriquement)\b"),
            "calculation_verb",
        ),
        (
            QuestionType.TROUBLESHOOTING,
            re.compile(
                r"\b(?:fouling|encrassement|fault|panne|symptom|cause|"
                r"remedy|remedier|bouchage|lavage|shutdown|arret)\b"
            ),
            "problem_or_remedy",
        ),
        (
            QuestionType.SAFETY,
            re.compile(r"\b(?:safety|securite|hazard|danger|risk|risque)\b"),
            "safety_term",
        ),
        (
            QuestionType.CONTROL_STRATEGY,
            re.compile(
                r"\b(?:pid|mpc|controller|control(?:led|ler|ling)?|"
                r"control\w*|regul\w*|control strategy|setpoint|"
                r"consigne|manipulated variable|variable manipulee)\b"
            ),
            "control_term",
        ),
        (
            QuestionType.MOMENTUM_DIFFUSION,
            re.compile(
                r"\b(?:momentum diffusion|momentum transport|"
                r"transport of momentum|diffusion de quantite de mouvement|"
                r"transport de quantite de mouvement|انتقال الزخم)\b"
            ),
            "momentum_transport_term",
        ),
        (
            QuestionType.BALANCE,
            re.compile(
                r"\b(?:mass|material|component|species|energy|heat|enthalpy|p2o5)"
                r"(?:\s+(?:or|and)\s+\w+)?\s+balance\b|"
                r"\bbalance\s+(?:of\s+)?(?:mass|material|component|species|"
                r"energy|heat|enthalpy|p2o5)\b|"
                r"\bbilan\s+(?:(?:global|matiere|massique|"
                r"energetique|thermique)\b|(?:de|du|des|d)\s*p2o5\b|p2o5\b)"
            ),
            "balance_term",
        ),
        (
            QuestionType.EQUATION_EXPLANATION,
            re.compile(r"\b(?:equation|formula|formule|define variables|variables)\b"),
            "equation_term",
        ),
        (
            QuestionType.TABLE_QUESTION,
            re.compile(r"\b(?:table|tableau|tabulated|valeurs tabulees)\b"),
            "table_term",
        ),
        (
            QuestionType.THERMODYNAMIC_RELATION,
            re.compile(
                r"\b(?:relation).*"
                r"(?:pression|pressure|temperature|enthalp|entropy)\b"
            ),
            "thermodynamic_relation",
        ),
        (
            QuestionType.COMPARISON,
            re.compile(r"\b(?:difference|compare|versus|vs\.?)\b"),
            "comparison_term",
        ),
        (
            QuestionType.PROCESS_FLOW,
            re.compile(r"\b(?:trajet|path|flow through|etape par etape|step by step)\b"),
            "flow_sequence",
        ),
        (
            QuestionType.PROCEDURE,
            re.compile(r"\b(?:procedure|how to|comment demarrer|sequence)\b"),
            "procedure_term",
        ),
        (
            QuestionType.EXPLANATION,
            re.compile(r"\b(?:why|pourquoi|role|rôle|how does|دور|لماذا|كيف)\b"),
            "explanation_focus",
        ),
        (
            QuestionType.DEFINITION,
            re.compile(
                r"(?:^|\b)(?:what is|what are|define|definition of|"
                r"qu est ce que|qu est ce qu un|qu est ce qu une|"
                r"c est quoi|cest quoi|c quoi|definis?|explique moi ce qu est|"
                r"ما هو|ما هي|عرّف|عرف)\b"
            ),
            "definition_form",
        ),
        (
            QuestionType.PLANT_SPECIFIC,
            re.compile(
                r"\b(?:atelier|installation|ocp|jfc4|echelon|sur site|"
                r"installed plant|design reel|historique)\b"
            ),
            "plant_term",
        ),
        (
            QuestionType.EXPLANATION,
            re.compile(
                r"\b(?:why|pourquoi|explain|explique|role|how does|"
                r"comment|دور|لماذا|كيف)\b"
            ),
            "explanation_form",
        ),
    )

    for question_type, pattern, cue in rules:
        if pattern.search(normalized):
            return QuestionClassification(
                question_type=question_type,
                confidence=0.9,
                cues=(cue,),
                answer_policy=ANSWER_POLICIES[question_type],
            )

    return QuestionClassification(
        question_type=QuestionType.EXPLANATION,
        confidence=0.4,
        cues=("default_explanation",),
        answer_policy=ANSWER_POLICIES[QuestionType.EXPLANATION],
    )
