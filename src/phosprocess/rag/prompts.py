"""Compact prompts and deterministic follow-up resolution."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

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

SYSTEM_PROMPT = (
    "Réponds uniquement à partir des sources documentaires fournies. "
    "N'invente aucun fait. "
    "Chaque affirmation factuelle ou technique doit être reliée à une "
    "citation [Source N] correspondant réellement à une source fournie. "
    "Lorsque les sources ne permettent pas de répondre de manière fiable, "
    f'réponds exactement : "{INSUFFICIENT_CONTEXT_ANSWER}" '
    "Réponds directement et complètement à la question, sans répéter un fait ni "
    "ajouter de contexte hors sujet, puis arrête-toi quand la réponse est complète. "
    "N'impose aucune limite arbitraire de mots, phrases, étapes ou éléments. "
    "Ne révèle pas de raisonnement interne. "
    'Retourne uniquement un objet JSON de forme {"answer":"..."} '
    "sans autre champ."
)


STREAMING_SYSTEM_PROMPT = (
    "Réponds uniquement à partir des sources documentaires fournies. "
    "La mémoire conversationnelle sert uniquement à comprendre le suivi "
    "et ne constitue jamais une preuve documentaire. "
    "N'invente aucun fait. "
    "Chaque affirmation factuelle ou technique doit être reliée à une "
    "citation [Source N] correspondant réellement à une source fournie. "
    "Lorsque les sources sont insuffisantes, "
    f'réponds exactement : "{INSUFFICIENT_CONTEXT_ANSWER}" '
    "Réponds directement et complètement à la question, sans répéter un fait ni "
    "ajouter de contexte hors sujet, puis arrête-toi quand la réponse est complète. "
    "Ne révèle pas de raisonnement interne et n'écris jamais [Mémoire]."
)

DIRECT_SYSTEM_PROMPT = """\
Fulfill the user's self-contained request directly, without documentary retrieval.
Do not mention any documentary corpus, sources, retrieval, or citations.
Do not add technical context that the user did not request.
For translation, output only the translated text unless the user requests an explanation.
For rewriting or summarization, preserve the requested meaning and format.
Do not expose hidden reasoning."""


_QUALITY_INSUFFICIENCY = {
    ResponseLanguage.FRENCH: (
        "Les passages retrouvés ne permettent pas de répondre précisément à cette question."
    ),
    ResponseLanguage.ENGLISH: (
        "The retrieved passages do not provide enough information to answer "
        "this question precisely."
    ),
    ResponseLanguage.ARABIC: (
        "لا توفر المقاطع المسترجعة معلومات كافية للإجابة عن هذا السؤال بدقة."
    ),
}

GenerationPromptVariant = Literal["baseline", "grounded_evidence_utilization_v1"]


def _quality_system_instructions(
    *,
    response_language: ResponseLanguage,
    json_output: bool,
    prompt_variant: GenerationPromptVariant,
) -> list[str]:
    """Return the frozen baseline or the single Phase-11 research variant."""

    output_instruction = (
        'Return only {"answer":"..."} with no other field.'
        if json_output
        else "Return only the final cited answer."
    )
    insufficiency_instruction = (
        "If the evidence is insufficient, answer exactly: "
        f'"{_QUALITY_INSUFFICIENCY[response_language]}"'
    )
    shared = [f"Answer exclusively in {response_language.prompt_name}."]

    if prompt_variant == "baseline":
        return [
            *shared,
            "Use only the supplied evidence bundles as factual authority.",
            "Conversation memory may resolve references but is never evidence.",
            "Do not infer, complete or import missing technical facts.",
            "Cite every factual sentence with one or more [Source N] citations.",
            "A cited source must actually support the statement that cites it.",
            "Preserve numbers, units, equations, symbols and source-specific context.",
            "If sources disagree or describe different contexts, attribute them separately.",
            (
                "Answer the exact question directly and completely. Include each supported "
                "fact only once, omit unrelated context, and stop when the question is "
                "answered; use no fixed word, sentence, step or item count."
            ),
            "Do not expose hidden reasoning.",
            insufficiency_instruction,
            output_instruction,
        ]

    if prompt_variant != "grounded_evidence_utilization_v1":
        raise ValueError(f"Unknown generation prompt variant: {prompt_variant}")

    return [
        *shared,
        "Answer the user's exact question directly; do not substitute a related question.",
        (
            "Use only the supplied documentary evidence as factual authority; "
            "conversation memory may resolve references but is never evidence."
        ),
        (
            "Do not add facts from model memory, infer unstated causal relations, or turn "
            "keywords and headings into conclusions unsupported by documentary text."
        ),
        (
            "First identify the information needed for the actual question, then cover every "
            "distinct supported fact necessary to answer it completely."
        ),
        "Do not summarize or dump evidence that is unrelated to the question.",
        (
            "Keep different examples, equipment, operating conditions, numerical states, "
            "and documentary cases separate; when more than one is useful, attribute each "
            "explicitly and never fabricate a relationship between them."
        ),
        (
            "Place the relevant [Source N] citation immediately after each substantive "
            "factual claim; split claims or cite every source needed by the sentence."
        ),
        (
            "For a sequence, path, procedure, or ordered mechanism, follow the order "
            "established by the evidence without backward jumps; never concatenate distinct "
            "documentary sequences into one."
        ),
        (
            "Copy documentary numbers and units faithfully. Do not alter, normalize, merge, "
            "or calculate values unless the user requests a calculation and all operands are "
            "provided. Attribute different numerical contexts separately."
        ),
        (
            "Answer as fully as necessary for the question and available evidence, with no "
            "fixed length or item count, and stop when the question is completely answered."
        ),
        insufficiency_instruction,
        "Do not expose hidden reasoning, an internal plan, or evidence-selection notes.",
        output_instruction,
    ]

_FOLLOWUP_CONNECTOR = re.compile(
    r"^(?:et\b|donc\b|dans ce cas\b|alors\b|qu'en est-il\b)",
    flags=re.IGNORECASE,
)
_AMBIGUOUS_PRONOUN = re.compile(
    r"\b(?:cela|ça|ceci|celle-ci|celui-ci|elle|elles|il|ils|les|sa|son|leur)\b",
    flags=re.IGNORECASE,
)


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
        instructions.append("Requested output language: " + decision.requested_output_language)

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
        total_tokens=(estimate_tokens(DIRECT_SYSTEM_PROMPT) + estimate_tokens(user_prompt)),
    )
    return DIRECT_SYSTEM_PROMPT, PromptPackage(
        user_prompt=user_prompt,
        size=size,
    )


def _quality_bundle_blocks(
    bundles: Sequence[EvidenceBundle],
) -> str:
    return "\n\n".join(bundle.render_prompt_block() for bundle in bundles)


def build_quality_prompt_package(
    question: str,
    bundles: Sequence[EvidenceBundle],
    *,
    response_language: ResponseLanguage,
    classification: QuestionClassification,
    memory: ConversationHistoryContext | None = None,
    json_output: bool,
    prompt_variant: GenerationPromptVariant = "baseline",
) -> tuple[str, PromptPackage]:
    """Build a domain-neutral, evidence-grounded generation prompt."""

    if not bundles:
        raise ValueError("Le prompt qualité exige au moins une source.")

    policy = classification.answer_policy
    system_prompt = "\n".join(
        _quality_system_instructions(
            response_language=response_language,
            json_output=json_output,
            prompt_variant=prompt_variant,
        )
    )

    formatting: list[str] = [
        f"Question type: {classification.question_type.value}",
    ]

    if policy.numbered_steps:
        formatting.append(
            "When the question asks for a sequence and the evidence "
            "establishes an order, use numbered steps; otherwise answer "
            "naturally."
        )
    if policy.define_variables:
        formatting.append(
            "Define variables that are necessary to understand equations used in the answer."
        )
    if policy.preserve_units:
        formatting.append("Preserve units exactly when reporting documentary values.")
    if policy.show_assumptions:
        formatting.append(
            "State assumptions only when they are supported by the evidence "
            "or explicitly supplied by the user."
        )
    if policy.organize_as_causes_effects_actions:
        formatting.append(
            "When useful, organize supported troubleshooting evidence into "
            "causes, effects and actions without inventing missing links."
        )

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
) -> PromptPackage:
    """Build the legacy grounded prompt without an answer-length cap."""

    del memory
    memory_text = ""
    document_text = "\n\n".join(_document_blocks(sources, source_texts))
    output_instruction = (
        'Sortie : {"answer":"réponse citée"}'
        if json_output
        else "Sortie : texte final cité uniquement."
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
    system_prompt = SYSTEM_PROMPT if json_output else STREAMING_SYSTEM_PROMPT
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
        total_tokens=estimate_tokens(system_prompt) + estimate_tokens(user_prompt),
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
        selected.append(ChatMessage(role=message.role, content=content))
        used_characters += len(content)

    selected.reverse()
    return selected


def _last_user_question(
    history: Sequence[ChatMessage],
) -> str:
    """Return the most recent user question."""

    return next(
        (message.content for message in reversed(history) if message.role == "user"),
        "",
    )


def detect_follow_up(
    question: str,
    history: Sequence[ChatMessage],
) -> bool:
    """Detect a short anaphoric follow-up without domain-specific vocabulary."""

    if not history:
        return False

    normalized = question.strip()
    short = len(normalized.split()) <= 12
    connector = _FOLLOWUP_CONNECTOR.search(normalized) is not None
    pronoun = _AMBIGUOUS_PRONOUN.search(normalized) is not None
    return short and (connector or pronoun)


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

    if summary_request is None and not detect_follow_up(normalized, history):
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
        recent_subjects = [message.content for message in history if message.role == "user"]
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
        retrieval_query=(f"Contexte du sujet : {previous[:300]}. Question : {cleaned}"),
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
