"""Interactive terminal session for the streaming production RAG service."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TextIO

from phosprocess.rag.conversation_memory import ConversationMemory
from phosprocess.rag.language import normalize_language_mode
from phosprocess.rag.pipeline import PhosProcessRAG
from phosprocess.rag.schemas import RAGSource
from phosprocess.retrieval.domain_router import SUPPORTED_SOURCE_MODES

HELP_TEXT = """\
Commandes disponibles :
  /help     Afficher cette aide.
  /exit     Quitter proprement.
  /clear    Effacer l'historique en mémoire.
  /sources  Réafficher les sources de la dernière réponse.
  /history  Afficher les messages conservés dans la session.
  /source auto|becker|report|thermodynamics|heat_transfer|perry|
          crystallization|control|transport
            Choisir la politique documentaire de la session.
  /lang auto|fr|en|ar
            Choisir la langue de réponse.
  /debug on|off
            Afficher ou masquer les diagnostics de retrieval.
"""


def normalize_quality_source_mode(mode: str) -> str:
    """Validate one global or explicitly filtered quality source mode."""

    normalized = mode.strip().casefold()

    if normalized == "automatic":
        normalized = "auto"

    if normalized not in SUPPORTED_SOURCE_MODES:
        raise ValueError(
            "Mode source invalide. Utilisez auto, becker, report, "
            "thermodynamics, heat_transfer, perry, crystallization, "
            "control ou transport."
        )

    return normalized


@dataclass(slots=True)
class ChatSessionState:
    """Mutable in-memory state for one terminal session."""

    memory: ConversationMemory = field(default_factory=ConversationMemory)
    last_sources: list[RAGSource] = field(default_factory=list)
    source_mode: str = "auto"
    language_mode: str = "auto"
    debug_enabled: bool = False

    def __post_init__(self) -> None:
        self.source_mode = normalize_quality_source_mode(self.source_mode)
        self.language_mode = normalize_language_mode(self.language_mode)

    def remember(self, role: str, content: str) -> None:
        """Store a clean message through the summary-buffer memory."""

        if role == "user":
            self.memory.add_user_message(content)
        elif role == "assistant":
            self.memory.add_assistant_message(content)
        else:
            raise ValueError(f"Rôle conversationnel invalide : {role}")

    @property
    def history_enabled(self) -> bool:
        """Expose whether session memory is active."""

        return self.memory.enabled

    @property
    def history(self) -> list[object]:
        """Compatibility view containing only the bounded recent messages."""

        return list(self.memory.build_history_context().messages())

    def clear(self) -> None:
        """Clear all conversational state without touching disk."""

        self.memory.clear()
        self.last_sources.clear()


def format_source(source: RAGSource, *, detailed: bool = False) -> str:
    """Format one cited or retrieved source for terminal display."""

    pages = ", ".join(str(page) for page in source.pages)
    base = (
        f"[Source {source.source_number}] {source.document_name}, "
        f"page(s) {pages} — {source.chunk_id}"
    )

    if not detailed:
        return base

    details = [
        base,
        f"  document={source.document_title or source.document_name}",
        f"  domain={source.domain or 'non renseigné'}",
        f"  chapter={source.chapter or 'non renseigné'}",
        f"  section={source.section or 'non renseignée'}",
        f"  chunk_type={source.chunk_type or 'legacy'}",
        f"  child={source.anchor_chunk_id or source.chunk_id}",
        f"  parent={source.parent_id or 'aucun'}",
        (
            f"  dense={source.dense_score} bm25={source.bm25_score} "
            f"fusion={source.rrf_score:.6f} "
            f"reranker={source.reranker_score} "
            f"boost={source.source_boost or 0.0}"
        ),
        (
            f"  selection={source.selection_source} "
            f"context_tokens={source.context_added_tokens or 0}"
        ),
    ]

    if source.anchor_text:
        details.append(f"  texte_child={source.anchor_text}")

    if source.display_text and source.display_text != source.anchor_text:
        details.append(f"  contexte_ajouté={source.display_text}")

    return "\n".join(details)


def print_sources(
    sources: list[RAGSource],
    *,
    output: TextIO,
    detailed: bool = False,
) -> None:
    """Print source metadata without exposing full document passages."""

    if not sources:
        print("Sources : aucune source citée.", file=output)
        return

    print("Sources :", file=output)

    for source in sources:
        print(
            "  "
            + format_source(source, detailed=detailed).replace(
                "\n",
                "\n  ",
            ),
            file=output,
        )


def print_latency_table(
    metrics: dict[str, object],
    *,
    output: TextIO,
) -> None:
    """Print compact, content-safe per-turn diagnostics."""

    if not metrics:
        print("Latence détaillée : indisponible.", file=output)
        return

    print(
        f"Politique documentaire : {metrics.get('source_policy_route', 'indisponible')}",
        file=output,
    )
    print(
        f"Source prioritaire : {metrics.get('source_policy_primary', 'indisponible')}",
        file=output,
    )
    print(
        f"Fallback utilisé : {'oui' if metrics.get('source_policy_fallback_used') else 'non'}",
        file=output,
    )

    rows = (
        ("Validation question", "question_validation_ms"),
        ("Suivi / reformulation", "followup_detection_ms"),
        ("Embedding", "embedding_ms"),
        ("Recherche dense", "dense_search_ms"),
        ("Recherche BM25", "bm25_search_ms"),
        ("Fusion hybride", "hybrid_fusion_ms"),
        ("Reranking", "reranking_ms"),
        ("Safeguard lexical", "lexical_selection_ms"),
        ("Préparation extraits", "excerpt_preparation_ms"),
        ("Mémoire", "memory_build_ms"),
        ("Prompt", "prompt_build_ms"),
        ("Connexion Ollama", "ollama_connection_ms"),
        ("Premier événement", "ollama_time_to_first_event_ms"),
        ("Premier token Ollama", "ollama_time_to_first_token_ms"),
        ("Génération Ollama", "ollama_generation_ms"),
        ("Validation citations", "citation_validation_ms"),
        ("Réparation", "repair_ms"),
        ("Total", "total_ms"),
    )
    print("Latence détaillée :", file=output)

    for label, key in rows:
        value = metrics.get(key, 0.0)

        if isinstance(value, (int, float)):
            print(f"  {label:<24} {value:>10.1f} ms", file=output)

    print(
        "  "
        f"prompt={metrics.get('estimated_prompt_tokens', 0)} tokens | "
        f"contexte={metrics.get('document_context_token_count', 0)} | "
        f"mémoire={metrics.get('recent_history_token_count', 0)}+"
        f"{metrics.get('summary_token_count', 0)} | "
        f"appels Ollama={metrics.get('ollama_call_count', 0)}",
        file=output,
    )


def handle_command(
    command: str,
    *,
    state: ChatSessionState,
    output: TextIO,
) -> bool:
    """Handle one slash command and return whether the session should continue."""

    normalized = command.strip().casefold()

    if normalized == "/exit":
        print("Au revoir.", file=output)
        return False

    if normalized == "/help":
        print(HELP_TEXT, file=output)
        return True

    if normalized == "/clear":
        state.clear()
        print("Historique effacé.", file=output)
        return True

    if normalized == "/sources":
        print_sources(state.last_sources, output=output)
        return True

    if normalized == "/history":
        if not state.history_enabled:
            print("Historique désactivé (--no-history).", file=output)
        elif not state.memory.get_recent_turns() and not state.memory.get_summary():
            print("Historique vide.", file=output)
        else:
            debug = state.memory.export_debug_view()
            print("Mémoire conversationnelle :", file=output)
            print(
                f"  Résumé : {debug['summary'] or '(vide)'}",
                file=output,
            )
            print("  Fenêtre récente :", file=output)

            for turn in debug["recent_turns"]:
                print(f"    Vous > {turn['user']}", file=output)
                print(
                    f"    Assistant > {turn['assistant']}",
                    file=output,
                )

            usage = debug["token_usage"]
            print(
                "  Tokens estimés : "
                f"résumé={usage['summary_tokens']} "
                f"récents={usage['recent_tokens']} "
                f"total={usage['total_tokens']}",
                file=output,
            )

        return True

    if normalized.startswith("/source"):
        parts = normalized.split()

        if len(parts) != 2:
            print(
                "Usage : /source auto|becker|report|thermodynamics|"
                "heat_transfer|perry|crystallization|control|transport",
                file=output,
            )
            return True

        try:
            state.source_mode = normalize_quality_source_mode(parts[1])
        except ValueError as error:
            print(str(error), file=output)
            return True

        state.last_sources.clear()
        if state.source_mode == "auto":
            state.memory.state.release_source_scope()
        else:
            state.memory.state.record_source_scope(
                state.source_mode,
                explicit=True,
                origin="terminal_command",
            )
        print(
            f"Politique documentaire active : {state.source_mode}.",
            file=output,
        )
        return True

    if normalized.startswith("/lang"):
        parts = normalized.split()

        if len(parts) != 2:
            print("Usage : /lang auto|fr|en|ar", file=output)
            return True

        try:
            state.language_mode = normalize_language_mode(parts[1])
        except ValueError as error:
            print(str(error), file=output)
            return True

        print(f"Langue de réponse : {state.language_mode}.", file=output)
        return True

    if normalized.startswith("/debug"):
        parts = normalized.split()

        if len(parts) != 2 or parts[1] not in {"on", "off"}:
            print("Usage : /debug on|off", file=output)
            return True

        state.debug_enabled = parts[1] == "on"
        print(
            f"Debug retrieval : {'activé' if state.debug_enabled else 'désactivé'}.",
            file=output,
        )
        return True

    print(
        "Commande inconnue. Utilisez /help.",
        file=output,
    )
    return True


class TerminalChat:
    """Drive an interactive streaming conversation without persistence."""

    def __init__(
        self,
        service: PhosProcessRAG,
        *,
        show_retrieval: bool = False,
        show_query: bool = False,
        show_latency: bool = False,
        history_enabled: bool = True,
        source_mode: str = "auto",
        input_function: Callable[[str], str] = input,
        output: TextIO = sys.stdout,
    ) -> None:
        self.service = service
        self.show_retrieval = show_retrieval
        self.show_query = show_query
        self.show_latency = show_latency
        self.input_function = input_function
        self.output = output
        self.state = ChatSessionState(
            memory=service.create_conversation_memory(
                enabled=history_enabled,
            ),
            source_mode=source_mode,
        )

    def run(self) -> int:
        """Run until /exit, EOF or a confirmed Ctrl+C at the input prompt."""

        print("PhosProcess Copilot", file=self.output)
        print("Tapez /help pour afficher les commandes.\n", file=self.output)

        while True:
            try:
                question = self.input_function("Vous > ").strip()
            except EOFError:
                print("\nAu revoir.", file=self.output)
                return 0
            except KeyboardInterrupt:
                if self._confirm_exit():
                    print("Au revoir.", file=self.output)
                    return 0

                continue

            if not question:
                continue

            if question.startswith("/"):
                if not handle_command(
                    question,
                    state=self.state,
                    output=self.output,
                ):
                    return 0

                continue

            self._answer(question)

    def _answer(self, question: str) -> None:
        """Display one real token stream and retain only validated responses."""

        history_context = self.state.memory.build_history_context()
        events = self.service.stream_answer(
            question,
            history_context,
            source_mode=self.state.source_mode,
            language_mode=self.state.language_mode,
        )
        active_attempt: str | None = None
        completed_answer: str | None = None
        response_started = False

        try:
            for event in events:
                if event.event_type == "retrieval_started" and (
                    self.show_query or self.state.debug_enabled
                ):
                    print("\nRequête :", file=self.output)
                    print(
                        f"  originale={question}",
                        file=self.output,
                    )
                    print(
                        f"  autonome={event.metadata.get('standalone_query', question)}",
                        file=self.output,
                    )
                    print(
                        f"  langue={event.metadata.get('language', 'inconnue')} "
                        f"type={event.metadata.get('question_type', 'inconnu')}",
                        file=self.output,
                    )
                    print(
                        f"  domaines={event.metadata.get('source_policy_route', '')}",
                        file=self.output,
                    )

                elif event.event_type == "retrieval_completed":
                    retrieval_skipped = bool(event.metadata.get("retrieval_skipped"))
                    if self.show_query or self.state.debug_enabled:
                        added_terms = event.metadata.get(
                            "query_expansion",
                            [],
                        )
                        print(
                            "  expansion="
                            + (
                                ", ".join(str(term) for term in added_terms)
                                if added_terms
                                else (
                                    "non applicable (mode direct)"
                                    if retrieval_skipped
                                    else "aucune"
                                )
                            ),
                            file=self.output,
                        )

                    if retrieval_skipped:
                        if self.show_retrieval or self.state.debug_enabled:
                            print(
                                "\nRetrieval documentaire ignoré : "
                                "requête autonome sans besoin de sources.",
                                file=self.output,
                            )
                        continue

                    if self.show_retrieval or self.state.debug_enabled:
                        sections = event.metadata.get(
                            "hierarchical_sections",
                            [],
                        )
                        if sections:
                            print("\nSections sélectionnées :", file=self.output)
                            for index, section in enumerate(sections, start=1):
                                pages = section.get("pages", ["?", "?"])
                                print(
                                    f"  [{index}] {section.get('hierarchy_path', '')} "
                                    f"| pages={pages[0]}-{pages[1]} "
                                    f"| score={section.get('score', 0)}",
                                    file=self.output,
                                )
                        print(
                            "\nRetrieval hiérarchique + v3 :",
                            file=self.output,
                        )
                        print_sources(
                            event.sources,
                            output=self.output,
                            detailed=True,
                        )

                elif event.event_type == "token":
                    attempt = str(event.metadata.get("attempt", "initial"))

                    if attempt != active_attempt:
                        if active_attempt is not None:
                            print(
                                "\n\nRéparation du format des citations...",
                                file=self.output,
                            )

                        print(
                            "\nAssistant > ",
                            end="",
                            file=self.output,
                            flush=True,
                        )
                        active_attempt = attempt
                        response_started = True

                    print(
                        event.content or "",
                        end="",
                        file=self.output,
                        flush=True,
                    )

                elif event.event_type == "sources":
                    self.state.last_sources = list(event.sources)

                elif event.event_type == "completed":
                    if response_started:
                        print(file=self.output)

                    response = event.response

                    if response is None:
                        continue

                    completed_answer = response.answer
                    if response.standalone_query is not None:
                        self.state.memory.state.record_resolution(response.standalone_query)

                    if response.response_language is not None:
                        self.state.memory.state.record_response(
                            cited_documents=tuple(
                                source.document_name for source in response.sources
                            ),
                            language=response.response_language,
                        )
                    if response.source_policy_route != "direct_no_retrieval":
                        print_sources(
                            response.sources,
                            output=self.output,
                        )
                    elif self.show_query or self.state.debug_enabled:
                        print(
                            "Mode direct : aucune recherche documentaire.",
                            file=self.output,
                        )
                    first_token_ms = response.timings.first_token_ms
                    first_token_text = (
                        f"{first_token_ms / 1000.0:.2f} s" if first_token_ms is not None else "n/a"
                    )
                    print(
                        "Latence premier token : "
                        f"{first_token_text} | Durée totale : "
                        f"{response.timings.total_ms / 1000.0:.2f} s",
                        file=self.output,
                    )

                    if self.show_latency:
                        print_latency_table(
                            response.latency,
                            output=self.output,
                        )

                elif event.event_type == "error":
                    if response_started:
                        print(file=self.output)

                    print(
                        f"Erreur : {event.content}",
                        file=self.output,
                    )
        except KeyboardInterrupt:
            events.close()
            print(
                "\nGénération interrompue. Aucune réponse partielle "
                "n'a été ajoutée à l'historique.",
                file=self.output,
            )
            return

        if history_context.business_state is not None:
            self.state.memory.synchronize_business_state(history_context.business_state)

        if completed_answer is not None:
            self.state.memory.add_turn(question, completed_answer)

    def _confirm_exit(self) -> bool:
        """Ask for confirmation after Ctrl+C at the empty input prompt."""

        print(file=self.output)

        try:
            answer = self.input_function("Quitter PhosProcess Copilot ? [o/N] ")
        except (EOFError, KeyboardInterrupt):
            print(file=self.output)
            return True

        return answer.strip().casefold() in {"o", "oui", "y", "yes"}
