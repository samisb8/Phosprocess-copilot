"""Deterministic technical question classification and answer policy."""

from __future__ import annotations

import re
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
    TROUBLESHOOTING = "troubleshooting"
    CONTROL_STRATEGY = "control_strategy"
    SAFETY = "safety"
    PLANT_SPECIFIC = "plant_specific"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class AnswerPolicy:
    """Formatting and size limits selected from the question type."""

    max_words: int
    numbered_steps: bool = False
    define_variables: bool = False
    preserve_units: bool = False
    show_assumptions: bool = False
    organize_as_causes_effects_actions: bool = False


ANSWER_POLICIES: dict[QuestionType, AnswerPolicy] = {
    QuestionType.DEFINITION: AnswerPolicy(140),
    QuestionType.EXPLANATION: AnswerPolicy(230),
    QuestionType.PROCESS_FLOW: AnswerPolicy(320, numbered_steps=True),
    QuestionType.PROCEDURE: AnswerPolicy(320, numbered_steps=True),
    QuestionType.COMPARISON: AnswerPolicy(260),
    QuestionType.BALANCE: AnswerPolicy(
        300,
        define_variables=True,
        preserve_units=True,
    ),
    QuestionType.THERMODYNAMIC_RELATION: AnswerPolicy(
        300,
        define_variables=True,
        preserve_units=True,
    ),
    QuestionType.EQUATION_EXPLANATION: AnswerPolicy(
        300,
        define_variables=True,
        preserve_units=True,
    ),
    QuestionType.TABLE_QUESTION: AnswerPolicy(280),
    QuestionType.CALCULATION: AnswerPolicy(
        350,
        show_assumptions=True,
        preserve_units=True,
    ),
    QuestionType.TROUBLESHOOTING: AnswerPolicy(
        300,
        organize_as_causes_effects_actions=True,
    ),
    QuestionType.CONTROL_STRATEGY: AnswerPolicy(300),
    QuestionType.SAFETY: AnswerPolicy(260),
    QuestionType.PLANT_SPECIFIC: AnswerPolicy(300),
    QuestionType.AMBIGUOUS: AnswerPolicy(180),
}


@dataclass(frozen=True, slots=True)
class QuestionClassification:
    """Question type, confidence and deterministic cues."""

    question_type: QuestionType
    confidence: float
    cues: tuple[str, ...]
    answer_policy: AnswerPolicy


def classify_question(question: str) -> QuestionClassification:
    """Classify in priority order without an LLM call."""

    normalized = (
        question.strip()
        .casefold()
        .replace("’", "'")
        .replace("‘", "'")
    )

    if not normalized:
        raise ValueError("La question ne peut pas être vide.")

    rules: tuple[tuple[QuestionType, re.Pattern[str], str], ...] = (
        (
            QuestionType.AMBIGUOUS,
            re.compile(r"^(?:what is|qu'est-ce que|c'est quoi)\s+(?:the |le |la )?boiler\??$"),
            "ambiguous_boiler",
        ),
        (
            QuestionType.CALCULATION,
            re.compile(r"\b(?:calculate|compute|calcule|détermine numériquement)\b"),
            "calculation_verb",
        ),
        (
            QuestionType.TROUBLESHOOTING,
            re.compile(r"\b(?:fouling|encrassement|fault|panne|symptom|cause|remedy|remédier)\b"),
            "problem_or_remedy",
        ),
        (
            QuestionType.SAFETY,
            re.compile(r"\b(?:safety|sécurité|hazard|danger|risk|risque)\b"),
            "safety_term",
        ),
        (
            QuestionType.CONTROL_STRATEGY,
            re.compile(
                r"\b(?:pid|mpc|controller|control(?:led|ler|ling)?|"
                r"contr[oô]l\w*|r[eé]gul\w*|control strategy|setpoint|"
                r"consigne|manipulated variable|variable manipul[eé]e)\b"
            ),
            "control_term",
        ),
        (
            QuestionType.BALANCE,
            re.compile(
                r"\b(?:mass|material|component|species|energy|heat|enthalpy|p2o5)"
                r"(?:\s+(?:or|and)\s+\w+)?\s+balance\b|"
                r"\bbalance\s+(?:of\s+)?(?:mass|material|component|species|"
                r"energy|heat|enthalpy|p2o5)\b|"
                r"\bbilan\s+(?:(?:global|mati[eè]re|massique|"
                r"[eé]nerg[eé]tique|thermique)\b|(?:de|du|des|d['’])\s*p2o5\b|p2o5\b)"
            ),
            "balance_term",
        ),
        (
            QuestionType.EQUATION_EXPLANATION,
            re.compile(r"\b(?:equation|équation|formula|formule|define variables|variables)\b"),
            "equation_term",
        ),
        (
            QuestionType.TABLE_QUESTION,
            re.compile(r"\b(?:table|tableau|tabulated|valeurs tabulées)\b"),
            "table_term",
        ),
        (
            QuestionType.THERMODYNAMIC_RELATION,
            re.compile(
                r"\b(?:relation).*"
                r"(?:pression|pressure|température|temperature|enthalp|entropy)\b"
            ),
            "thermodynamic_relation",
        ),
        (
            QuestionType.COMPARISON,
            re.compile(r"\b(?:difference|différence|compare|versus|vs\.?)\b"),
            "comparison_term",
        ),
        (
            QuestionType.PROCESS_FLOW,
            re.compile(r"\b(?:trajet|path|flow through|étape par étape|step by step)\b"),
            "flow_sequence",
        ),
        (
            QuestionType.PROCEDURE,
            re.compile(r"\b(?:procedure|procédure|how to|comment démarrer|séquence)\b"),
            "procedure_term",
        ),
        (
            QuestionType.PLANT_SPECIFIC,
            re.compile(r"\b(?:atelier|installation|ocp|sur site|installed plant)\b"),
            "plant_term",
        ),
        (
            QuestionType.DEFINITION,
            re.compile(
                r"^(?:what is|what are|qu'est-ce que|qu'est-ce qu['’](?:un|une)|"
                r"c'est quoi|définis?)\b"
            ),
            "definition_form",
        ),
        (
            QuestionType.EXPLANATION,
            re.compile(r"\b(?:why|pourquoi|explain|explique|role|rôle|how does|comment)\b"),
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
