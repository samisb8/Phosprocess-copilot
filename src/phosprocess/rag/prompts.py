"""Compact prompts and deterministic follow-up resolution."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from phosprocess.observability.latency import estimate_tokens
from phosprocess.rag.adaptive_router import AdaptiveRouteDecision, DirectIntent
from phosprocess.rag.citations import INSUFFICIENT_CONTEXT_ANSWER
from phosprocess.rag.conversation_memory import (
    ConversationHistoryContext,
)
from phosprocess.rag.language import ResponseLanguage
from phosprocess.rag.question_classifier import QuestionClassification
from phosprocess.rag.schemas import ChatMessage, RAGSource
from phosprocess.retrieval.evidence_bundle import EvidenceBundle

SYSTEM_PROMPT = f"""\
Réponds uniquement avec les cinq nouvelles sources fournies. N'invente rien.
Cite chaque affirmation technique par [Source N]. Si elles sont insuffisantes,
réponds exactement : "{INSUFFICIENT_CONTEXT_ANSWER}" Réponse industrielle,
précise et concise. Aucun raisonnement interne. Retourne uniquement
{{"answer":"..."}} sans autre champ."""

STREAMING_SYSTEM_PROMPT = f"""\
Réponds uniquement avec les cinq nouvelles sources fournies ; la mémoire sert
seulement à comprendre le suivi et n'est jamais une preuve. N'invente rien.
Cite chaque affirmation technique par [Source N]. Si les sources sont
insuffisantes, réponds exactement : "{INSUFFICIENT_CONTEXT_ANSWER}" Réponse
industrielle de 100 mots maximum, sans raisonnement interne, JSON ni Markdown.
N'écris jamais [Mémoire] et termine par une phrase complète."""


DIRECT_SYSTEM_PROMPT = """\
Fulfill the user's self-contained request directly, without documentary retrieval.
Do not mention any documentary corpus, sources, retrieval, or citations.
Do not add technical context that the user did not request.
For translation, output only the translated text unless the user requests an explanation.
For rewriting or summarization, preserve the requested meaning and format.
Do not expose hidden reasoning."""

REPAIR_SYSTEM_PROMPT = """\
Corrige les citations et supprime toute affirmation que les passages cités ne
soutiennent pas explicitement. N'ajoute aucun fait. Si aucune réponse fiable
ne reste, utilise exactement la formulation d'insuffisance demandée. Utilise
seulement [Source 1] à [Source 5] et retourne uniquement la sortie corrigée."""

_QUALITY_INSUFFICIENCY = {
    ResponseLanguage.FRENCH: (
        "Les passages retrouvés ne permettent pas de répondre précisément "
        "à cette question."
    ),
    ResponseLanguage.ENGLISH: (
        "The retrieved passages do not provide enough information to answer "
        "this question precisely."
    ),
    ResponseLanguage.ARABIC: (
        "لا توفر المقاطع المسترجعة معلومات كافية للإجابة عن هذا السؤال بدقة."
    ),
}

_FOLLOWUP_CONNECTOR = re.compile(
    r"^(?:et\b|donc\b|dans ce cas\b|alors\b|qu'en est-il\b)",
    flags=re.IGNORECASE,
)
_AMBIGUOUS_PRONOUN = re.compile(
    r"\b(?:cela|ça|ceci|celle-ci|celui-ci|elle|elles|il|ils|les|sa|son|leur)\b",
    flags=re.IGNORECASE,
)
_BUSINESS_TERMS = {
    "acide",
    "attaque",
    "bouillie",
    "cristal",
    "cristaux",
    "filtration",
    "gypse",
    "phosphate",
    "p2o5",
    "réacteur",
    "recirculation",
    "sulfate",
    "supersaturation",
}


@dataclass(frozen=True, slots=True)
class PromptSizeBreakdown:
    """Character and estimated-token contributions of one final prompt."""

    system_characters: int
    memory_characters: int
    question_characters: int
    document_characters: int
    total_characters: int
    system_tokens: int
    memory_tokens: int
    question_tokens: int
    document_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class PromptPackage:
    """One compact user prompt and its safe size telemetry."""

    user_prompt: str
    size: PromptSizeBreakdown


def build_direct_prompt_package(
    question: str,
    decision: AdaptiveRouteDecision,
    *,
    json_output: bool,
) -> tuple[str, PromptPackage]:
    """Build a compact prompt for a request that must bypass retrieval."""

    intent = decision.direct_intent or DirectIntent.CONVERSATION
    instructions = [
        f"Direct intent: {intent.value}",
        "Execute the current user request exactly.",
    ]

    if decision.requested_output_language is not None:
        instructions.append(
            "Requested output language: "
            + decision.requested_output_language
        )

    if json_output:
        instructions.append('Return only {"answer":"..."}.')
    else:
        instructions.append("Return only the final answer.")

    user_prompt = "\n".join(instructions) + f"\n\nUSER REQUEST\n{question.strip()}"
    size = PromptSizeBreakdown(
        system_characters=len(DIRECT_SYSTEM_PROMPT),
        memory_characters=0,
        question_characters=len(question),
        document_characters=0,
        total_characters=len(DIRECT_SYSTEM_PROMPT) + len(user_prompt),
        system_tokens=estimate_tokens(DIRECT_SYSTEM_PROMPT),
        memory_tokens=0,
        question_tokens=estimate_tokens(question),
        document_tokens=0,
        total_tokens=(
            estimate_tokens(DIRECT_SYSTEM_PROMPT)
            + estimate_tokens(user_prompt)
        ),
    )
    return DIRECT_SYSTEM_PROMPT, PromptPackage(
        user_prompt=user_prompt,
        size=size,
    )


def _quality_bundle_blocks(
    bundles: Sequence[EvidenceBundle],
) -> str:
    blocks: list[str] = []

    for bundle in bundles:
        blocks.append(
            "\n".join(
                [
                    f"[Source {bundle.source_number}]",
                    f"Document: {bundle.document_title}",
                    f"Filename: {bundle.filename}",
                    f"Chapter: {bundle.chapter or 'Not specified'}",
                    f"Section: {bundle.section or 'Not specified'}",
                    f"Pages: {bundle.page_start}-{bundle.page_end}",
                    bundle.display_text,
                ]
            )
        )

    return "\n\n".join(blocks)


def build_quality_prompt_package(
    question: str,
    bundles: Sequence[EvidenceBundle],
    *,
    response_language: ResponseLanguage,
    classification: QuestionClassification,
    memory: ConversationHistoryContext | None = None,
    json_output: bool,
) -> tuple[str, PromptPackage]:
    """Build the strict multilingual evidence-bundle generation prompt."""

    if not 1 <= len(bundles) <= 5:
        raise ValueError("Le prompt qualité exige entre une et cinq sources.")

    policy = classification.answer_policy
    system_prompt = "\n".join(
        [
            f"Answer exclusively in {response_language.prompt_name}.",
            "Use only the supplied evidence bundles.",
            "Never use conversation memory as factual evidence.",
            "Do not infer or complete missing technical facts.",
            "Cite every factual sentence as [Source N].",
            "Write 3 to 5 complete sentences for a substantive answer.",
            (
                "Each sentence must be directly supported by at least one "
                "single cited source; do not assemble a new claim from "
                "fragments found in different sources."
            ),
            (
                "Do not add plausible mechanisms, definitions, consequences "
                "or recommendations unless the cited source states them."
            ),
            (
                "A citation supports only facts explicitly stated in that "
                "same evidence bundle."
            ),
            "Distinguish general information from plant-specific information.",
            (
                "When sources describe different designs, operating periods or "
                "values, attribute each context explicitly and never merge them "
                "into one fact."
            ),
            "Preserve equations, units, symbols, formulas and document titles.",
            "Do not expose hidden reasoning.",
            (
                "If evidence is insufficient, answer exactly: "
                f'"{_QUALITY_INSUFFICIENCY[response_language]}"'
            ),
            (
                "Never append the insufficiency sentence to an otherwise "
                "substantive answer."
            ),
            (
                'Return only {"answer":"..."} with no other field.'
                if json_output
                else "Return only the final cited answer."
            ),
        ]
    )
    formatting: list[str] = [
        f"Question type: {classification.question_type.value}",
        f"Maximum answer length: {policy.max_words} words",
    ]

    if policy.numbered_steps:
        formatting.extend(
            [
                (
                    "Use numbered steps only when the evidence explicitly "
                    "gives the sequence."
                ),
                (
                    "Each step must explicitly state its own action and destination "
                    "in the cited bundle; do not invent connectors such as then, "
                    "after, returns, enters or is sent."
                ),
                (
                    "Do not combine a design target, a simulation result and an "
                    "operating value as if they described the same state."
                ),
            ]
        )

    if classification.question_type.value == "process_flow":
        formatting.extend(
            [
                (
                    "Output exactly five numbered steps in this order: feed inlet; "
                    "conical-bottom withdrawal; circulation pump and heating "
                    "element; return to the flash chamber plus vapor-liquid "
                    "separation; concentrated-product outlet."
                ),
                (
                    "Write one source-local factual action per step. Step 4 may use "
                    "exactly two independently cited clauses separated by a "
                    "semicolon: return to the flash chamber; vapor-liquid "
                    "separation."
                ),
                (
                    "Do not repeat the return-to-flash action or the product outlet. "
                    "Do not join any other separate actions with and, where, which, "
                    "thereby, resulting, or a semicolon."
                ),
                (
                    "Reuse the source wording as closely as possible; never add an "
                    "unstated mechanism such as heating causes evaporation."
                ),
                (
                    "Treat 'is fed by the inlet acid pipe' as the inlet stage and "
                    "'is withdrawn from the vapor body at an outlet' as the "
                    "product-outlet stage."
                ),
                (
                    "Do not add concluding filler such as 'completing its path'."
                ),
                (
                    "If one required stage is not explicitly supported by the "
                    "bundles, return the exact insufficiency answer."
                ),
            ]
        )

    if classification.question_type.value == "definition":
        formatting.extend(
            [
                (
                    "The answer must contain all three supported parts: "
                    "what the equipment is; how its defining mechanism works; "
                    "and its documented function or use."
                ),
                (
                    "Do not replace the definition with only an advantage, "
                    "application or design consequence."
                ),
            ]
        )

    if classification.question_type.value == "momentum_diffusion":
        formatting.extend(
            [
                (
                    "Explain momentum diffusion only through momentum flux, "
                    "velocity gradient, shear stress and dynamic viscosity."
                ),
                (
                    "For a Newtonian fluid, state Newton's law of viscosity "
                    "when it is supported by the evidence and define every symbol."
                ),
                (
                    "Do not use Fick's law, concentration gradients or species "
                    "diffusivity unless the question explicitly asks for a contrast."
                ),
            ]
        )

    normalized_question = question.casefold().replace("₂", "2").replace("₅", "5")
    plant_p2o5_balance = (
        classification.question_type.value == "balance"
        and "p2o5" in normalized_question
        and any(
            marker in normalized_question
            for marker in ("jfc4", "échelon", "echelon", "rapport ocp")
        )
    )
    if plant_p2o5_balance:
        formatting.extend(
            [
                (
                    "Use the report's three P2O5 terms: feed at line 1, "
                    "concentrated product at line 5, and entrainment at line 6."
                ),
                (
                    "Do not set the entrainment loss to zero when the report "
                    "provides a non-zero value."
                ),
                (
                    "Keep report values and units distinct from generic textbook "
                    "equations or design assumptions."
                ),
            ]
        )

    if classification.question_type.value == "comparison":
        formatting.extend(
            [
                (
                    "Every factual sentence must explicitly name equipment A, "
                    "equipment B, or both; never use vague phrases such as "
                    "the two systems."
                ),
                (
                    "Discuss only documented comparison criteria requested by "
                    "the question, such as operation, heat transfer, fouling, "
                    "viscosity, residence time, pressure drop or solids handling."
                ),
                (
                    "Exclude unrelated process facts, concentration stages, "
                    "sludge treatment and impurity-removal details unless the "
                    "question explicitly asks for them."
                ),
            ]
        )

    if classification.question_type.value == "troubleshooting":
        formatting.extend(
            [
                (
                    "Organize only problem-specific evidence in this logical "
                    "order: cause; physical mechanism; operational effect; "
                    "documented action."
                ),
                (
                    "Every action must explicitly mitigate, remove, clean or "
                    "control the named problem. Exclude generic operating advice "
                    "for carryover, filtration or production rate."
                ),
            ]
        )

    if policy.define_variables:
        formatting.append("Define variables found in the supplied equations.")

    if policy.show_assumptions:
        formatting.append("State only assumptions explicitly supported by evidence.")

    if (
        policy.organize_as_causes_effects_actions
        and classification.question_type.value != "troubleshooting"
    ):
        formatting.append("Organize supported facts as causes, effects and actions.")

    # Conversation memory is used upstream only to resolve references.
    # Raw summaries, previous answers, values and citations are deliberately
    # excluded from the generation prompt.
    memory_text = ""
    evidence_text = _quality_bundle_blocks(bundles)
    user_parts = [
        part
        for part in (
            memory_text,
            "\n".join(formatting),
            f"QUESTION\n{question.strip()}",
            f"EVIDENCE BUNDLES\n{evidence_text}",
        )
        if part
    ]
    user_prompt = "\n\n".join(user_parts)
    size = PromptSizeBreakdown(
        system_characters=len(system_prompt),
        memory_characters=len(memory_text),
        question_characters=len(question),
        document_characters=len(evidence_text),
        total_characters=len(system_prompt) + len(user_prompt),
        system_tokens=estimate_tokens(system_prompt),
        memory_tokens=estimate_tokens(memory_text),
        question_tokens=estimate_tokens(question),
        document_tokens=estimate_tokens(evidence_text),
        total_tokens=estimate_tokens(system_prompt) + estimate_tokens(user_prompt),
    )
    return system_prompt, PromptPackage(user_prompt=user_prompt, size=size)


@dataclass(frozen=True, slots=True)
class FollowUpResolution:
    """Result of deterministic conversational-reference analysis."""

    retrieval_query: str
    is_follow_up: bool
    reformulated: bool
    method: str


def format_pages(pages: Sequence[int]) -> str:
    """Format source pages for the LLM context."""

    return ",".join(str(page) for page in pages)


def _document_blocks(
    sources: Sequence[RAGSource],
    source_texts: Sequence[str],
) -> list[str]:
    """Build compact source blocks without scores or duplicated metadata."""

    if len(sources) != len(source_texts):
        raise ValueError("Chaque source doit posséder un passage.")

    blocks: list[str] = []

    for source, text in zip(sources, source_texts, strict=True):
        section = source.section or "Section non renseignée"
        blocks.append(
            "\n".join(
                [
                    (
                        f"[Source {source.source_number}] "
                        f"{source.document_name} | "
                        f"p.{format_pages(source.pages)} | {section}"
                    ),
                    text.strip(),
                ]
            )
        )

    return blocks


def _memory_block(
    memory: ConversationHistoryContext | None,
) -> str:
    """Format clean memory explicitly as non-documentary context."""

    if memory is None or memory.total_token_count == 0:
        return ""

    lines = ["MÉMOIRE CONVERSATIONNELLE — NON PROBANTE"]

    if memory.summary:
        lines.append(f"Résumé : {memory.summary}")

    for turn in memory.recent_turns:
        lines.append(f"Utilisateur : {turn.user}")
        lines.append(f"Assistant : {turn.assistant}")

    return "\n".join(lines)


def build_prompt_package(
    question: str,
    sources: Sequence[RAGSource],
    source_texts: Sequence[str],
    *,
    memory: ConversationHistoryContext | None = None,
    json_output: bool,
    maximum_answer_words: int = 100,
) -> PromptPackage:
    """Build the compact final prompt and contribution metrics."""

    # Keep conversational state for follow-up resolution, but never expose
    # previous factual content to the answer-generation model.
    memory_text = ""
    document_text = "\n\n".join(
        _document_blocks(sources, source_texts)
    )
    output_instruction = (
        'Sortie : {"answer":"réponse citée"}'
        if json_output
        else (
            "Sortie : texte final cité uniquement, "
            f"{maximum_answer_words} mots maximum."
        )
    )
    parts = [
        part
        for part in (
            memory_text,
            f"QUESTION\n{question.strip()}",
            f"NOUVELLES SOURCES\n{document_text}",
            output_instruction,
        )
        if part
    ]
    user_prompt = "\n\n".join(parts)
    system_prompt = (
        SYSTEM_PROMPT
        if json_output
        else STREAMING_SYSTEM_PROMPT
    )
    total_characters = len(system_prompt) + len(user_prompt)
    size = PromptSizeBreakdown(
        system_characters=len(system_prompt),
        memory_characters=len(memory_text),
        question_characters=len(question),
        document_characters=len(document_text),
        total_characters=total_characters,
        system_tokens=estimate_tokens(system_prompt),
        memory_tokens=estimate_tokens(memory_text),
        question_tokens=estimate_tokens(question),
        document_tokens=estimate_tokens(document_text),
        total_tokens=estimate_tokens(system_prompt)
        + estimate_tokens(user_prompt),
    )
    return PromptPackage(user_prompt=user_prompt, size=size)


def build_user_prompt(
    question: str,
    sources: Sequence[RAGSource],
    source_texts: Sequence[str],
) -> str:
    """Build the blocking answer-only JSON prompt."""

    return build_prompt_package(
        question,
        sources,
        source_texts,
        json_output=True,
    ).user_prompt


def build_streaming_user_prompt(
    question: str,
    sources: Sequence[RAGSource],
    source_texts: Sequence[str],
    *,
    memory: ConversationHistoryContext | None = None,
) -> str:
    """Build a compact document-grounded plain-text prompt."""

    return build_prompt_package(
        question,
        sources,
        source_texts,
        memory=memory,
        json_output=False,
    ).user_prompt


def build_repair_prompt(
    *,
    original_prompt: str,
    invalid_output: str,
    rejection_reason: str,
    json_output: bool,
) -> str:
    """Ask for one format-only repair using the same new sources."""

    expected = (
        'Retourne uniquement {"answer":"..."} sans autre champ.'
        if json_output
        else "Retourne uniquement le texte corrigé."
    )
    return "\n\n".join(
        [
            original_prompt,
            f"SORTIE INVALIDE\n{invalid_output}",
            f"REJET\n{rejection_reason}",
            (
                "Corrige les références et supprime les affirmations non "
                "soutenues ; aucun fait nouveau."
            ),
            expected,
        ]
    )


def limit_history(
    history: Sequence[ChatMessage],
    *,
    maximum_messages: int,
    maximum_characters: int,
) -> list[ChatMessage]:
    """Compatibility helper retaining newest messages within hard limits."""

    if maximum_messages <= 0 or maximum_characters <= 0:
        return []

    selected: list[ChatMessage] = []
    used_characters = 0

    for message in reversed(history[-maximum_messages:]):
        remaining = maximum_characters - used_characters

        if remaining <= 0:
            break

        content = message.content[-remaining:]
        selected.append(
            ChatMessage(role=message.role, content=content)
        )
        used_characters += len(content)

    selected.reverse()
    return selected


def _last_user_question(
    history: Sequence[ChatMessage],
) -> str:
    """Return the most recent user question."""

    return next(
        (
            message.content
            for message in reversed(history)
            if message.role == "user"
        ),
        "",
    )


def _explicit_business_term_count(question: str) -> int:
    """Count generic process entities explicitly present in a question."""

    lowered = question.casefold()
    return sum(term in lowered for term in _BUSINESS_TERMS)


def detect_follow_up(
    question: str,
    history: Sequence[ChatMessage],
) -> bool:
    """Detect dependency on a prior turn without an LLM call."""

    if not history:
        return False

    normalized = question.strip()
    short = len(normalized.split()) <= 12
    connector = _FOLLOWUP_CONNECTOR.search(normalized) is not None
    pronoun = _AMBIGUOUS_PRONOUN.search(normalized) is not None
    explicit_terms = _explicit_business_term_count(normalized)
    return (connector or pronoun) and short and explicit_terms <= 1


def _extract_previous_topic(previous_question: str) -> tuple[str, str]:
    """Extract a generic subject and optional process context."""

    topic_match = re.search(
        r"(?:rôle|effet|impact|importance)\s+de\s+"
        r"(.+?)(?:\s+dans\s+|\s+sur\s+|\?|$)",
        previous_question,
        flags=re.IGNORECASE,
    )
    context_match = re.search(
        r"\bdans\s+(.+?)(?:\?|$)",
        previous_question,
        flags=re.IGNORECASE,
    )
    topic = topic_match.group(1).strip() if topic_match else ""
    context = context_match.group(1).strip() if context_match else ""
    return topic, context


def resolve_follow_up(
    question: str,
    history: Sequence[ChatMessage],
    *,
    summary: str = "",
) -> FollowUpResolution:
    """Prefer a short deterministic standalone retrieval query."""

    normalized = question.strip()
    summary_request = re.search(
        r"\b(?:résume|résumer|synthèse|récapitule)\b",
        normalized,
        flags=re.IGNORECASE,
    )

    if not history:
        return FollowUpResolution(
            retrieval_query=normalized,
            is_follow_up=False,
            reformulated=False,
            method="none",
        )

    if (
        summary_request is None
        and not detect_follow_up(normalized, history)
    ):
        return FollowUpResolution(
            retrieval_query=normalized,
            is_follow_up=False,
            reformulated=False,
            method="none",
        )

    previous = _last_user_question(history)

    if summary_request is not None:
        summary_subjects = re.findall(
            r"Sujet\s*:\s*(.+?)\s+Éléments discutés\s*:",
            summary,
            flags=re.IGNORECASE,
        )
        recent_subjects = [
            message.content
            for message in history
            if message.role == "user"
        ]
        subjects = list(
            dict.fromkeys(
                subject.strip()
                for subject in [*summary_subjects, *recent_subjects]
                if subject.strip()
            )
        )
        query = "Synthèse documentaire : " + " ; ".join(subjects[-6:])
        return FollowUpResolution(
            retrieval_query=query[:1000],
            is_follow_up=True,
            reformulated=True,
            method="deterministic_summary_topics",
        )

    topic, context = _extract_previous_topic(previous)
    cleaned = re.sub(
        r"^(?:et|donc|alors)\s+",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    pronoun_pattern = re.match(
        r"pourquoi\s+([\wÀ-ÿ]+)-t-elle\s+(.+)",
        cleaned,
        flags=re.IGNORECASE,
    )

    if topic and pronoun_pattern:
        verb, remainder = pronoun_pattern.groups()

        if context:
            context_without_article = re.sub(
                r"^(?:le|la|les|un|une)\s+",
                "",
                context,
                flags=re.IGNORECASE,
            )
            remainder = re.sub(
                r"\bdu procédé\b",
                f"du {context_without_article}",
                remainder,
                flags=re.IGNORECASE,
            )

        query = f"Pourquoi {topic} {verb}-t-elle {remainder}"
        query = query[0].upper() + query[1:]
        return FollowUpResolution(
            retrieval_query=query,
            is_follow_up=True,
            reformulated=True,
            method="deterministic_antecedent",
        )

    return FollowUpResolution(
        retrieval_query=(
            f"Contexte du sujet : {previous[:300]}. "
            f"Question : {cleaned}"
        ),
        is_follow_up=True,
        reformulated=True,
        method="deterministic_context",
    )


def build_standalone_retrieval_query(
    question: str,
    history: Sequence[ChatMessage],
) -> str:
    """Compatibility wrapper returning only the retrieval query."""

    return resolve_follow_up(
        question,
        history,
    ).retrieval_query
