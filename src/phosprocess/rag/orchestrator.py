"""Public RAG orchestrator and runtime lifecycle."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from phosprocess.embeddings.embedder import resolve_cached_model_source
from phosprocess.knowledge_base.runtime import (
    ActiveKnowledgeBase,
    load_active_knowledge_base,
)
from phosprocess.llm.ollama_client import (
    OllamaConfig,
    OllamaError,
    OllamaLLM,
    load_ollama_config,
)
from phosprocess.observability.latency import (
    OllamaCallMetrics,
    RAGLatencyMetrics,
    WarmupMetrics,
    estimate_tokens,
)
from phosprocess.rag.adaptive_router import RequestPath, decide_request_path
from phosprocess.rag.answer_validation_service import AnswerValidationService
from phosprocess.rag.citations import CitationValidationError
from phosprocess.rag.conversation_memory import (
    ConversationHistoryContext,
    ConversationMemory,
)
from phosprocess.rag.conversation_state import ConversationState
from phosprocess.rag.followup_resolver import resolve_ambiguous_query_with_llm
from phosprocess.rag.generation_service import (
    GenerationService,
)
from phosprocess.rag.language import detect_response_language
from phosprocess.rag.prompts import (
    STREAMING_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    PromptPackage,
    build_prompt_package,
    build_quality_prompt_package,
    resolve_follow_up,
)
from phosprocess.rag.quality_retrieval import QualityRetrievalEngine
from phosprocess.rag.question_classifier import classify_question
from phosprocess.rag.retrieval_service import (
    RAGConfigurationError,
    RAGError,
    RetrievalService,
    _RetrievedContext,
)
from phosprocess.rag.schemas import ChatMessage, RAGResponse, RAGStreamEvent
from phosprocess.rag.source_policy import SourcePolicyConfig
from phosprocess.reranking.reranker import BGEReranker, load_reranking_config
from phosprocess.retrieval.domain_router import (
    detect_explicit_source_mode,
    requests_automatic_source_scope,
    route_query,
)
from phosprocess.retrieval.hybrid import HybridRetriever

LOGGER = logging.getLogger("phosprocess.rag.pipeline")

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_SNAPSHOT_DIRECTORY = (
    PROJECT_ROOT / "data" / "evaluation" / "retrieval" / "v0.1" / "frozen" / "dev_best_v3"
)

DEFAULT_RUNTIME_CONFIG_PATH = PROJECT_ROOT / "configs" / "rag_production.yaml"

DEFAULT_RAG_V1_CONFIG_PATH = PROJECT_ROOT / "configs" / "rag_v1.yaml"

DEFAULT_EMBEDDING_CONFIG_PATH = PROJECT_ROOT / "configs" / "embeddings.yaml"

EXPECTED_SELECTED_VARIANT = "lexical_safeguard_001"

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class FrozenV3Config:
    """Exact retrieval parameters loaded from dev_best_v3."""

    snapshot_directory: Path
    snapshot_sha256: str
    selected_variant: str
    candidate_k: int
    dense_candidates: int
    bm25_candidates: int
    query_expansion: bool
    top_k: int
    lexical_slots: int
    reranker_leading_slots: int
    lexical_source: str
    duplicate_policy: str
    fallback: str
    retrieval_config_path: Path
    reranking_config_path: Path

    def __post_init__(self) -> None:
        """Reject any runtime drift from the frozen winner."""

        if self.selected_variant != EXPECTED_SELECTED_VARIANT:
            raise RAGConfigurationError("dev_best_v3 ne sélectionne pas lexical_safeguard_001.")

        exact_parameters = {
            "candidate_k": (self.candidate_k, 20),
            "dense_candidates": (self.dense_candidates, 20),
            "bm25_candidates": (self.bm25_candidates, 20),
            "top_k": (self.top_k, 5),
            "lexical_slots": (self.lexical_slots, 1),
            "reranker_leading_slots": (
                self.reranker_leading_slots,
                4,
            ),
        }

        for name, (actual, expected) in exact_parameters.items():
            if actual != expected:
                raise RAGConfigurationError(f"Paramètre figé inattendu {name}: {actual}.")

        if self.query_expansion is not True:
            raise RAGConfigurationError("L'expansion de requête figée doit être activée.")

        if self.lexical_source != "bm25":
            raise RAGConfigurationError("La source lexicale figée doit être BM25.")

        if self.duplicate_policy != "skip":
            raise RAGConfigurationError("La politique figée doit ignorer les doublons.")

        if self.fallback != "next_reranker_result":
            raise RAGConfigurationError("Le fallback figé doit utiliser le prochain résultat.")


@dataclass(frozen=True, slots=True)
class ConversationRuntimeConfig:
    """Summary-buffer memory configuration."""

    enabled: bool = True
    strategy: str = "summary_buffer"
    recent_turns: int = 2
    summary_max_tokens: int = 300
    recent_history_max_tokens: int = 500
    total_history_max_tokens: int = 800

    def __post_init__(self) -> None:
        if self.strategy != "summary_buffer":
            raise ValueError("Seule la stratégie summary_buffer est supportée.")

        if self.recent_turns <= 0:
            raise ValueError("recent_turns doit être positif.")

        if self.summary_max_tokens + self.recent_history_max_tokens > self.total_history_max_tokens:
            raise ValueError("Budgets de mémoire conversationnelle invalides.")


@dataclass(frozen=True, slots=True)
class GenerationRuntimeConfig:
    """Document-context budgets without changing retrieval selection."""

    max_context_tokens_per_source: int = 350
    max_total_document_context_tokens: int = 1750

    def __post_init__(self) -> None:
        if self.max_context_tokens_per_source <= 0:
            raise ValueError("max_context_tokens_per_source doit être positif.")

        if self.max_total_document_context_tokens < 5:
            raise ValueError("max_total_document_context_tokens est insuffisant.")


@dataclass(frozen=True, slots=True)
class WarmupRuntimeConfig:
    """One-time startup warm-up switches."""

    enabled: bool = True
    embedding: bool = True
    reranker: bool = True
    ollama: bool = True


def _default_source_policy_config() -> SourcePolicyConfig:
    """Return a neutral source policy: automatic retrieval is global."""

    return SourcePolicyConfig(
        enabled=False,
        default_priority=(),
        domain_routes={},
        minimum_preferred_chunks=1,
        allow_fallback=False,
    )


@dataclass(frozen=True, slots=True)
class RAGRuntimeConfig:
    """Generation and response settings outside frozen retrieval."""

    ollama: OllamaConfig
    maximum_question_characters: int = 2000
    source_excerpt_characters: int = 700
    conversation: ConversationRuntimeConfig = ConversationRuntimeConfig()
    generation: GenerationRuntimeConfig = GenerationRuntimeConfig()
    warmup: WarmupRuntimeConfig = WarmupRuntimeConfig()
    source_policy: SourcePolicyConfig = field(
        default_factory=_default_source_policy_config,
    )
    logging_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.maximum_question_characters <= 0:
            raise ValueError("maximum_question_characters doit être positif.")

        if self.source_excerpt_characters <= 0:
            raise ValueError("source_excerpt_characters doit être positif.")


class _TimedTokenizerProxy:
    """Measure reranker tokenization without changing frozen code or outputs."""

    def __init__(
        self,
        tokenizer: Any,
        metrics_provider: Callable[[], RAGLatencyMetrics | None],
    ) -> None:
        self._tokenizer = tokenizer
        self._metrics_provider = metrics_provider

    def _timed_call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = getattr(self._tokenizer, method_name)(*args, **kwargs)
        elapsed = (time.perf_counter() - started) * 1000.0
        metrics = self._metrics_provider()

        if metrics is not None:
            metrics.reranker_tokenization_ms += elapsed

        return result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._timed_call("__call__", *args, **kwargs)

    def prepare_for_model(self, *args: Any, **kwargs: Any) -> Any:
        return self._timed_call("prepare_for_model", *args, **kwargs)

    def pad(self, *args: Any, **kwargs: Any) -> Any:
        return self._timed_call("pad", *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)


def sha256_file(path: Path) -> str:
    """Return an uppercase SHA-256 digest."""

    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest().upper()


def _required_mapping(
    raw: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    """Load one required YAML mapping."""

    value = raw.get(name)

    if not isinstance(value, dict):
        raise RAGConfigurationError(f"Section {name} absente ou invalide.")

    return value


def load_runtime_config(
    path: Path = DEFAULT_RUNTIME_CONFIG_PATH,
) -> RAGRuntimeConfig:
    """Load generation, memory, context and warm-up settings."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise RAGConfigurationError("Configuration RAG de production invalide.")

    response = _required_mapping(raw, "response")
    conversation = _required_mapping(raw, "conversation_memory")
    generation = _required_mapping(raw, "generation")
    warmup = _required_mapping(raw, "warmup")
    source_policy = _required_mapping(raw, "source_policy")
    logging_config = _required_mapping(raw, "logging")
    excluded_memory_fields = (
        "exclude_sources",
        "exclude_retrieval_metadata",
        "exclude_system_prompts",
        "exclude_logs",
    )

    if not all(conversation.get(field_name) is True for field_name in excluded_memory_fields):
        raise RAGConfigurationError("La mémoire doit exclure sources, scores, prompts et logs.")

    default_priority = source_policy.get("default_priority")
    domain_routes = source_policy.get("domain_routes")

    if (
        not isinstance(default_priority, list)
        or not all(isinstance(source, str) and source.strip() for source in default_priority)
        or not isinstance(domain_routes, dict)
    ):
        raise RAGConfigurationError("Configuration source_policy invalide.")

    parsed_routes: dict[str, tuple[str, ...]] = {}

    for route_name, route in domain_routes.items():
        if not isinstance(route_name, str) or not isinstance(route, dict):
            raise RAGConfigurationError("Route documentaire invalide dans source_policy.")
        preferred_sources = route.get("preferred_sources")
        if not isinstance(preferred_sources, list) or not all(
            isinstance(source, str) and source.strip() for source in preferred_sources
        ):
            raise RAGConfigurationError(f"Sources préférées invalides : {route_name}.")
        parsed_routes[route_name] = tuple(preferred_sources)

    return RAGRuntimeConfig(
        ollama=load_ollama_config(path),
        maximum_question_characters=int(response["maximum_question_characters"]),
        source_excerpt_characters=int(response["source_excerpt_characters"]),
        conversation=ConversationRuntimeConfig(
            enabled=bool(conversation["enabled"]),
            strategy=str(conversation["strategy"]),
            recent_turns=int(conversation["recent_turns"]),
            summary_max_tokens=int(conversation["summary_max_tokens"]),
            recent_history_max_tokens=int(conversation["recent_history_max_tokens"]),
            total_history_max_tokens=int(conversation["total_history_max_tokens"]),
        ),
        generation=GenerationRuntimeConfig(
            max_context_tokens_per_source=int(generation["max_context_tokens_per_source"]),
            max_total_document_context_tokens=int(generation["max_total_document_context_tokens"]),
        ),
        warmup=WarmupRuntimeConfig(
            enabled=bool(warmup["enabled"]),
            embedding=bool(warmup["embedding"]),
            reranker=bool(warmup["reranker"]),
            ollama=bool(warmup["ollama"]),
        ),
        source_policy=SourcePolicyConfig(
            enabled=bool(source_policy["enabled"]),
            default_priority=tuple(default_priority),
            domain_routes=parsed_routes,
            minimum_preferred_chunks=int(source_policy["minimum_preferred_chunks"]),
            allow_fallback=bool(source_policy["allow_fallback"]),
        ),
        logging_level=str(logging_config.get("level", "INFO")),
    )


def load_frozen_v3_config(
    snapshot_directory: Path = DEFAULT_SNAPSHOT_DIRECTORY,
    *,
    verify_integrity: bool = True,
    verify_runtime_sources: bool = False,
) -> FrozenV3Config:
    """Load frozen v3 settings without coupling them to the v4 runtime.

    ``verify_integrity`` validates the immutable files stored inside the
    snapshot. ``verify_runtime_sources`` is an explicit legacy-only check that
    additionally requires the active Python sources to be byte-identical to
    the historical v3 runtime.
    """

    snapshot_directory = snapshot_directory.resolve()
    manifest_path = snapshot_directory / "freeze_manifest.json"
    safeguard_path = snapshot_directory / "lexical_safeguard_v3.yaml"
    retrieval_path = snapshot_directory / "retrieval_v2.yaml"
    reranking_path = snapshot_directory / "reranking.yaml"

    for required_path in (
        manifest_path,
        safeguard_path,
        retrieval_path,
        reranking_path,
    ):
        if not required_path.is_file():
            raise RAGConfigurationError(f"Composant dev_best_v3 introuvable: {required_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    safeguard = yaml.safe_load(safeguard_path.read_text(encoding="utf-8"))

    if not isinstance(manifest, dict) or not isinstance(safeguard, dict):
        raise RAGConfigurationError("Le snapshot dev_best_v3 est invalide.")

    if (
        manifest.get("status") != "frozen_dev_best_v3"
        or manifest.get("scope") != "dev_only"
        or manifest.get("single_selected_variant") is not True
    ):
        raise RAGConfigurationError("Le manifeste dev_best_v3 ne respecte pas le gel DEV.")

    selection = safeguard.get("selection")
    retrieval = safeguard.get("retrieval")

    if not isinstance(selection, dict) or not isinstance(retrieval, dict):
        raise RAGConfigurationError("Configuration safeguard figée invalide.")

    if verify_integrity or verify_runtime_sources:
        _verify_snapshot_components(
            snapshot_directory=snapshot_directory,
            manifest=manifest,
            verify_runtime_sources=verify_runtime_sources,
        )

    return FrozenV3Config(
        snapshot_directory=snapshot_directory,
        snapshot_sha256=str(manifest["snapshot_sha256"]),
        selected_variant=str(manifest["selected_variant"]),
        candidate_k=int(retrieval["candidate_k"]),
        dense_candidates=int(retrieval["dense_candidates"]),
        bm25_candidates=int(retrieval["bm25_candidates"]),
        query_expansion=bool(retrieval["query_expansion"]),
        top_k=int(selection["top_k"]),
        lexical_slots=int(selection["lexical_slots"]),
        reranker_leading_slots=int(selection["reranker_leading_slots"]),
        lexical_source=str(selection["lexical_source"]),
        duplicate_policy=str(selection["duplicate_policy"]),
        fallback=str(selection["fallback"]),
        retrieval_config_path=retrieval_path,
        reranking_config_path=reranking_path,
    )


def load_rag_v1_retrieval_config(
    config_path: Path = DEFAULT_RAG_V1_CONFIG_PATH,
    *,
    verify_integrity: bool = True,
) -> FrozenV3Config:
    """Load the production retrieval freeze without reading evaluation paths."""

    config_path = config_path.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or str(payload.get("rag_version")) != "1.0.0":
        raise RAGConfigurationError("La configuration RAG v1 est invalide.")

    retrieval = payload.get("retrieval")
    reranker = payload.get("reranker")
    if not isinstance(retrieval, dict) or not isinstance(reranker, dict):
        raise RAGConfigurationError("Le gel retrieval/reranker RAG v1 est incomplet.")

    def project_file(raw_value: Any, field_name: str) -> Path:
        candidate = (PROJECT_ROOT / str(raw_value)).resolve()
        if not candidate.is_relative_to(PROJECT_ROOT) or not candidate.is_file():
            raise RAGConfigurationError(
                f"Chemin RAG v1 invalide pour {field_name}: {raw_value}"
            )
        return candidate

    retrieval_path = project_file(retrieval.get("config"), "retrieval.config")
    safeguard_path = project_file(
        retrieval.get("safeguard_config"),
        "retrieval.safeguard_config",
    )
    reranking_path = project_file(reranker.get("config"), "reranker.config")

    if verify_integrity:
        expected_hashes = (
            (retrieval_path, retrieval.get("config_sha256")),
            (safeguard_path, retrieval.get("safeguard_sha256")),
            (reranking_path, reranker.get("config_sha256")),
        )
        for path, expected in expected_hashes:
            if not isinstance(expected, str) or sha256_file(path).lower() != expected.lower():
                raise RAGConfigurationError(
                    f"Empreinte de configuration RAG v1 incorrecte: {path.name}."
                )

    return FrozenV3Config(
        snapshot_directory=config_path.parent,
        snapshot_sha256=str(retrieval["historical_snapshot_sha256"]),
        selected_variant=str(retrieval["selected_variant"]),
        candidate_k=int(retrieval["candidate_k"]),
        dense_candidates=int(retrieval["dense_candidates"]),
        bm25_candidates=int(retrieval["bm25_candidates"]),
        query_expansion=bool(retrieval["query_expansion"]),
        top_k=int(retrieval["top_k"]),
        lexical_slots=int(retrieval["lexical_slots"]),
        reranker_leading_slots=int(retrieval["reranker_leading_slots"]),
        lexical_source=str(retrieval["lexical_source"]),
        duplicate_policy=str(retrieval["duplicate_policy"]),
        fallback=str(retrieval["fallback"]),
        retrieval_config_path=retrieval_path,
        reranking_config_path=reranking_path,
    )


def load_rag_v1_runtime_config(
    config_path: Path = DEFAULT_RAG_V1_CONFIG_PATH,
) -> RAGRuntimeConfig:
    """Load and cross-check the generation runtime named by the RAG v1 freeze."""

    config_path = config_path.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or str(payload.get("rag_version")) != "1.0.0":
        raise RAGConfigurationError("La configuration RAG v1 est invalide.")

    runtime_section = payload.get("runtime")
    generation = payload.get("generation")
    if not isinstance(runtime_section, dict) or not isinstance(generation, dict):
        raise RAGConfigurationError("Le gel de génération RAG v1 est incomplet.")

    runtime_path = (PROJECT_ROOT / str(runtime_section.get("config"))).resolve()
    if not runtime_path.is_relative_to(PROJECT_ROOT) or not runtime_path.is_file():
        raise RAGConfigurationError("Le chemin runtime RAG v1 est invalide.")
    runtime = load_runtime_config(runtime_path)
    expected = {
        "model": runtime.ollama.model,
        "temperature": runtime.ollama.temperature,
        "seed": runtime.ollama.seed,
        "context_size": runtime.ollama.context_size,
        "max_output_tokens": runtime.ollama.max_output_tokens,
        "keep_alive": runtime.ollama.keep_alive,
    }
    drift = {
        key: {"freeze": generation.get(key), "runtime": value}
        for key, value in expected.items()
        if generation.get(key) != value
    }
    if drift:
        raise RAGConfigurationError(
            f"Dérive entre rag_v1.yaml et le runtime de génération: {drift}"
        )
    return runtime


def _verify_snapshot_components(
    *,
    snapshot_directory: Path,
    manifest: dict[str, Any],
    verify_runtime_sources: bool,
) -> None:
    """Verify snapshot files and, only on request, the legacy v3 runtime."""

    components = manifest.get("components")

    if not isinstance(components, list):
        raise RAGConfigurationError("Liste des composants absente du manifeste.")

    records = {
        str(record["file"]): record
        for record in components
        if isinstance(record, dict) and "file" in record
    }
    required_frozen_files = (
        "retrieval_v2.yaml",
        "reranking.yaml",
        "v3_selection.py",
        "hybrid.py",
        "reranker.py",
        "lexical_safeguard_v3.yaml",
    )

    for file_name in required_frozen_files:
        record = records.get(file_name)
        frozen_path = snapshot_directory / file_name

        if record is None or not frozen_path.is_file():
            raise RAGConfigurationError(f"Composant figé manquant: {file_name}.")

        if sha256_file(frozen_path) != record["frozen_sha256"]:
            raise RAGConfigurationError(f"Empreinte figée incorrecte: {file_name}.")

    if not verify_runtime_sources:
        return

    active_sources = {
        "hybrid.py": (PROJECT_ROOT / "src" / "phosprocess" / "retrieval" / "hybrid.py"),
        "reranker.py": (PROJECT_ROOT / "src" / "phosprocess" / "reranking" / "reranker.py"),
        "v3_selection.py": (PROJECT_ROOT / "src" / "phosprocess" / "retrieval" / "v3_selection.py"),
    }

    for file_name, active_path in active_sources.items():
        if sha256_file(active_path) != records[file_name]["source_sha256"]:
            raise RAGConfigurationError(f"Le code runtime diffère de dev_best_v3: {file_name}.")


def validate_question(
    question: str,
    *,
    maximum_characters: int,
) -> str:
    """Normalize and validate one user question."""

    if not isinstance(question, str):
        raise TypeError("La question doit être une chaîne de caractères.")

    if "\x00" in question:
        raise ValueError("La question contient un caractère nul interdit.")

    normalized = _WHITESPACE.sub(" ", question).strip()

    if not normalized:
        raise ValueError("La question ne peut pas être vide.")

    if len(normalized) > maximum_characters:
        raise ValueError("La question dépasse la longueur maximale autorisée.")

    if not any(character.isalnum() for character in normalized):
        raise ValueError("La question doit contenir un caractère alphanumérique.")

    return normalized


class PhosProcessRAG(
    RetrievalService,
    GenerationService,
    AnswerValidationService,
):
    def __init__(
        self,
        *,
        frozen_config: FrozenV3Config | None = None,
        runtime_config: RAGRuntimeConfig | None = None,
        retriever: Any | None = None,
        reranker: Any | None = None,
        llm: Any | None = None,
        verify_snapshot: bool = True,
    ) -> None:
        loading_started = time.perf_counter()
        self.frozen_config = frozen_config or load_rag_v1_retrieval_config(
            verify_integrity=verify_snapshot,
        )
        self.runtime_config = runtime_config or load_rag_v1_runtime_config()
        self._lifecycle_counts = {
            "pipeline": 1,
            "retriever": 0,
            "embedding_model": 0,
            "bm25_index": 0,
            "reranker": 0,
            "ollama_client": 0,
        }
        self.active_knowledge_base: ActiveKnowledgeBase | None = None

        if retriever is None:
            retriever = self._build_retriever()

        self._lifecycle_counts["retriever"] = 1
        self._lifecycle_counts["embedding_model"] = 1
        self._lifecycle_counts["bm25_index"] = 1

        if reranker is None:
            reranker_config = load_reranking_config(self.frozen_config.reranking_config_path)
            reranker = BGEReranker(
                replace(
                    reranker_config,
                    model_name=resolve_cached_model_source(
                        reranker_config.model_name,
                    ),
                )
            )

        self._lifecycle_counts["reranker"] = 1
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm or OllamaLLM(self.runtime_config.ollama)
        self.quality_engine: QualityRetrievalEngine | None = None

        if self.active_knowledge_base is not None and QualityRetrievalEngine.is_quality_index(
            self.active_knowledge_base.version_directory
        ):
            self.quality_engine = QualityRetrievalEngine(
                version_directory=self.active_knowledge_base.version_directory,
                retriever=self.retriever,
                reranker=self.reranker,
                require_sparse_index=True,
            )

        self._lifecycle_counts["ollama_client"] = 1
        self.initial_loading_ms = (time.perf_counter() - loading_started) * 1000.0
        self._turn_counter = 0
        self._active_metrics: RAGLatencyMetrics | None = None
        self._embedding_hook_installed = False
        self._reranker_tokenizer_hook_installed = False
        self._warmup_metrics: WarmupMetrics | None = None
        self._install_embedding_timer()
        self._install_reranker_tokenizer_timer()

    def _build_retriever(self) -> HybridRetriever:
        """Load the atomically activated corpus with frozen v3 settings."""

        self.active_knowledge_base = load_active_knowledge_base()

        return HybridRetriever(
            dense_index_directory=(self.active_knowledge_base.dense_index_directory),
            bm25_index_directory=(self.active_knowledge_base.bm25_index_directory),
            embedding_config_path=DEFAULT_EMBEDDING_CONFIG_PATH,
            retrieval_config_path=self.frozen_config.retrieval_config_path,
        )

    def knowledge_base_status(self) -> dict[str, Any] | None:
        """Expose active version counts without document contents."""

        active = self.active_knowledge_base

        if active is None:
            return None

        return {
            "version": active.version,
            "document_count": active.document_count,
            "chunk_count": active.chunk_count,
            "documents": [dict(document) for document in active.documents],
        }

    def _install_embedding_timer(self) -> None:
        """Instrument the existing embedder without reconstructing it."""

        dense = getattr(self.retriever, "dense_retriever", None)
        embedder = getattr(dense, "embedder", None)
        original = getattr(embedder, "embed_query", None)

        if not callable(original):
            return

        def timed_embed_query(query: str) -> Any:
            started = time.perf_counter()
            result = original(query)
            elapsed = (time.perf_counter() - started) * 1000.0

            if self._active_metrics is not None:
                self._active_metrics.embedding_ms += elapsed

            return result

        embedder.embed_query = timed_embed_query
        self._embedding_hook_installed = True

    def lifecycle_debug(self) -> dict[str, Any]:
        """Return identities and construction counters, without document data."""

        dense = getattr(self.retriever, "dense_retriever", None)
        bm25 = getattr(self.retriever, "bm25_retriever", None)
        embedder = getattr(dense, "embedder", None)
        return {
            "counts": dict(self._lifecycle_counts),
            "ids": {
                "pipeline": id(self),
                "retriever": id(self.retriever),
                "dense_retriever": id(dense),
                "embedder": id(embedder),
                "bm25_retriever": id(bm25),
                "reranker": id(self.reranker),
                "ollama_client": id(self.llm),
                "ollama_http_client": id(getattr(self.llm, "stream_http_client", None)),
            },
            "embedding_hook_installed": self._embedding_hook_installed,
            "reranker_tokenizer_hook_installed": (self._reranker_tokenizer_hook_installed),
            "initial_loading_ms": self.initial_loading_ms,
        }

    def _install_reranker_tokenizer_timer(self) -> None:
        """Wrap tokenizer calls while preserving frozen reranker behavior."""

        model = getattr(self.reranker, "_model", None)
        tokenizer = getattr(model, "tokenizer", None)

        if tokenizer is None or isinstance(tokenizer, _TimedTokenizerProxy):
            return

        model.tokenizer = _TimedTokenizerProxy(
            tokenizer,
            lambda: self._active_metrics,
        )
        self._reranker_tokenizer_hook_installed = True

    def warmup(self, *, enabled: bool | None = None) -> WarmupMetrics:
        """Warm each long-lived model once without corpus or benchmark queries."""

        config = self.runtime_config.warmup
        effective_enabled = config.enabled if enabled is None else enabled
        metrics = WarmupMetrics(enabled=effective_enabled)

        if not effective_enabled:
            return metrics

        if self._warmup_metrics is not None:
            return self._warmup_metrics

        total_started = time.perf_counter()

        if config.embedding:
            embedder = getattr(
                getattr(self.retriever, "dense_retriever", None),
                "embedder",
                None,
            )

            if embedder is not None:
                started = time.perf_counter()
                embedder.embed_query("procédé phosphorique")
                metrics.embedding_ms = (time.perf_counter() - started) * 1000.0

        if config.reranker and hasattr(self.reranker, "_compute_scores"):
            started = time.perf_counter()
            self.reranker._compute_scores(pairs=[["procédé", "passage industriel"]])
            metrics.reranker_ms = (time.perf_counter() - started) * 1000.0

        if config.ollama and hasattr(self.llm, "stream_chat"):
            telemetry = OllamaCallMetrics(
                call_type="warmup",
                model=self.runtime_config.ollama.model,
                streaming=True,
            )
            started = time.perf_counter()
            list(
                self.llm.stream_chat(
                    [
                        {
                            "role": "user",
                            "content": "Réponds uniquement : OK",
                        }
                    ],
                    max_output_tokens=4,
                    call_type="warmup",
                    telemetry=telemetry,
                )
            )
            metrics.ollama_ms = (time.perf_counter() - started) * 1000.0
            metrics.ollama_call_count = 1

        metrics.total_ms = (time.perf_counter() - total_started) * 1000.0
        self._warmup_metrics = metrics
        return metrics

    def answer(
        self,
        question: str,
        *,
        source_mode: str = "automatic",
        language_mode: str = "auto",
    ) -> RAGResponse:
        """Blocking answer path, retaining strict answer-only JSON."""

        started = time.perf_counter()
        normalized = validate_question(
            question,
            maximum_characters=(self.runtime_config.maximum_question_characters),
        )
        adaptive_decision = decide_request_path(
            normalized,
            source_mode=source_mode,
        )
        if adaptive_decision.path is RequestPath.DIRECT_LLM:
            return self._answer_direct(
                normalized,
                adaptive_decision,
                language_mode=language_mode,
                started=started,
            )

        effective_source_mode = source_mode
        if self.quality_engine is not None:
            effective_source_mode = detect_explicit_source_mode(
                normalized
            ) or self._quality_source_mode(source_mode)

        retrieved = self._retrieve_with_source_policy(
            normalized,
            policy_question=normalized,
            source_mode=effective_source_mode,
        )
        system_prompt = SYSTEM_PROMPT

        if retrieved.quality_result is not None:
            classification = classify_question(normalized)
            language = detect_response_language(
                normalized,
                mode=language_mode,
            )
            retrieved.response_language = language.language.value
            retrieved.question_type = classification.question_type.value
            system_prompt, package = build_quality_prompt_package(
                normalized,
                retrieved.quality_result.bundles,
                response_language=language.language,
                classification=classification,
                json_output=True,
            )
        else:
            prepared = self._prepare_context(
                retrieved.source_texts,
                normalized,
            )
            package = build_prompt_package(
                normalized,
                retrieved.sources,
                prepared.texts,
                json_output=True,
            )

        generation_started = time.perf_counter()
        payload, citations, insufficient = self._generate_json_answer(
            user_prompt=package.user_prompt,
            system_prompt=system_prompt,
            available_source_count=len(retrieved.sources),
        )
        generation_ms = (time.perf_counter() - generation_started) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0
        return self._build_response(
            question=normalized,
            answer=payload.answer,
            cited_sources=self._cited_sources(
                retrieved.sources,
                citations,
            ),
            cited_source_numbers=citations,
            insufficient_context=insufficient,
            retrieved=retrieved,
            generation_ms=generation_ms,
            total_ms=total_ms,
        )

    def stream_answer(
        self,
        question: str,
        history: (list[ChatMessage] | ConversationHistoryContext | None) = None,
        *,
        source_mode: str = "automatic",
        language_mode: str = "auto",
    ) -> Iterator[RAGStreamEvent]:
        """Yield real Ollama tokens plus phase-level latency metrics."""

        turn_started = time.perf_counter()
        self._turn_counter += 1
        metrics = RAGLatencyMetrics(question_id=f"session-{self._turn_counter:04d}")

        try:
            phase_started = time.perf_counter()
            normalized = validate_question(
                question,
                maximum_characters=(self.runtime_config.maximum_question_characters),
            )
            metrics.question_validation_ms = (time.perf_counter() - phase_started) * 1000.0

            phase_started = time.perf_counter()
            memory_context = self._coerce_memory_context(history)
            metrics.memory_build_ms = (time.perf_counter() - phase_started) * 1000.0
            metrics.history_turn_count = len(memory_context.recent_turns)
            metrics.summary_token_count = memory_context.summary_token_count
            metrics.recent_history_token_count = memory_context.recent_history_token_count
            history_messages = memory_context.messages()

            phase_started = time.perf_counter()
            business_state = memory_context.business_state or ConversationState()
            adaptive_decision = decide_request_path(
                normalized,
                source_mode=source_mode,
            )

            if adaptive_decision.path is RequestPath.DIRECT_LLM:
                metrics.followup_detection_ms = (time.perf_counter() - phase_started) * 1000.0
                metrics.reformulation_method = "adaptive_router"
                yield from self._stream_direct_request(
                    normalized,
                    adaptive_decision,
                    language_mode=language_mode,
                    last_explicit_language=business_state.last_language,
                    metrics=metrics,
                    turn_started=turn_started,
                )
                return

            if self.quality_engine is not None:
                quality_resolution = resolve_ambiguous_query_with_llm(
                    normalized,
                    state=business_state,
                    llm=self.llm,
                )
                retrieval_query = quality_resolution.standalone_query
                follow_up = quality_resolution.followup_detected
                reformulated = quality_resolution.standalone_query != normalized
                reformulation_method = quality_resolution.resolver_type
                metrics.resolver_llm_ms = quality_resolution.resolver_latency_ms
                metrics.resolver_llm_call_count = (
                    quality_resolution.resolver_llm_call_count
                )
                metrics.ollama_call_count += quality_resolution.resolver_llm_call_count
            else:
                resolution = resolve_follow_up(
                    normalized,
                    history_messages,
                    summary=memory_context.summary,
                )
                retrieval_query = resolution.retrieval_query
                follow_up = resolution.is_follow_up
                reformulated = resolution.reformulated
                reformulation_method = resolution.method

            detection_elapsed = (time.perf_counter() - phase_started) * 1000.0
            metrics.followup_detection_ms = detection_elapsed
            metrics.reformulation_attempted = reformulated
            metrics.reformulation_method = reformulation_method
            metrics.retrieval_query = retrieval_query

            if reformulated:
                metrics.reformulation_ms = detection_elapsed

            effective_source_mode = self._resolve_turn_source_mode(
                source_mode,
                question=normalized,
                follow_up=follow_up,
                state=business_state,
            )
            LOGGER.info(
                "CONVERSATION_CONTEXT follow_up=%s focus_entity=%s "
                "active_source=%s source_lock=%s source_origin=%s",
                follow_up,
                business_state.focus_entity or "none",
                business_state.current_source_mode,
                business_state.source_scope_explicit,
                business_state.source_scope_origin or "none",
            )

            language = detect_response_language(
                normalized,
                last_explicit_language=business_state.last_language,
                mode=language_mode,
            )
            classification = classify_question(retrieval_query)
            business_state.record_question_type(classification.question_type.value)

            if self.quality_engine is not None:
                routing_preview = route_query(
                    retrieval_query,
                    catalog=self.quality_engine.catalog,
                    source_mode=self._quality_source_mode(effective_source_mode),
                    question_type=classification.question_type.value,
                    focus_entity=business_state.focus_entity,
                )
                route_label = ",".join(
                    domain.value for domain, _confidence in (routing_preview.detected_domains)
                )
                primary_label = (
                    next(
                        entry.display_title
                        for entry in self.quality_engine.catalog.documents
                        if entry.document_id == routing_preview.preferred_documents[0]
                    )
                    if routing_preview.preferred_documents
                    else "Aucune"
                )
            else:
                policy_decision = self._decide_source_policy(
                    f"{normalized} {retrieval_query}",
                    mode=effective_source_mode,
                )
                route_label = policy_decision.route
                primary_label = policy_decision.primary_label

            yield RAGStreamEvent(
                event_type="retrieval_started",
                metadata={
                    "question_id": metrics.question_id,
                    "follow_up": follow_up,
                    "reformulated": reformulated,
                    "reformulation_method": reformulation_method,
                    "standalone_query": retrieval_query,
                    "language": language.language.value,
                    "question_type": classification.question_type.value,
                    "source_policy_route": route_label,
                    "source_policy_primary": primary_label,
                },
            )
            retrieved = self._retrieve_with_source_policy(
                retrieval_query,
                policy_question=normalized,
                source_mode=effective_source_mode,
                metrics=metrics,
            )
            retrieved.response_language = language.language.value
            retrieved.question_type = classification.question_type.value
            yield RAGStreamEvent(
                event_type="retrieval_completed",
                sources=retrieved.sources,
                metadata={
                    "candidate_count": len(retrieved.candidates),
                    "selected_count": len(retrieved.selected),
                    "hybrid_ms": float(retrieved.hybrid_response.total_duration_ms),
                    "reranking_ms": metrics.reranking_ms,
                    "source_policy_route": (metrics.source_policy_route),
                    "source_policy_primary": (metrics.source_policy_primary),
                    "source_policy_fallback_used": (metrics.source_policy_fallback_used),
                    "query_expansion": (
                        list(retrieved.quality_result.query.added_terms)
                        if retrieved.quality_result is not None
                        else []
                    ),
                    "hierarchical_sections": (
                        [
                            {
                                "document": item.section.source_file,
                                "hierarchy_path": item.section.hierarchy_path,
                                "pages": [
                                    item.section.page_start,
                                    item.section.page_end,
                                ],
                                "score": round(item.final_score, 6),
                            }
                            for item in retrieved.quality_result.section_search.results
                        ]
                        if (
                            retrieved.quality_result is not None
                            and retrieved.quality_result.section_search is not None
                        )
                        else []
                    ),
                    "section_retrieval_ms": (
                        retrieved.quality_result.section_search.duration_ms
                        if (
                            retrieved.quality_result is not None
                            and retrieved.quality_result.section_search is not None
                        )
                        else 0.0
                    ),
                },
            )

            phase_started = time.perf_counter()
            system_prompt_for_turn = STREAMING_SYSTEM_PROMPT

            if retrieved.quality_result is not None:
                prepared = None
            else:
                prepared = self._prepare_context(
                    retrieved.source_texts,
                    retrieval_query,
                )

            metrics.excerpt_preparation_ms = (time.perf_counter() - phase_started) * 1000.0

            if prepared is not None:
                metrics.document_context_token_count = prepared.total_tokens

            phase_started = time.perf_counter()

            if retrieved.quality_result is not None:
                (
                    system_prompt_for_turn,
                    package,
                ) = build_quality_prompt_package(
                    normalized,
                    retrieved.quality_result.bundles,
                    response_language=language.language,
                    classification=classification,
                    memory=memory_context,
                    json_output=False,
                )
            else:
                assert prepared is not None
                package = build_prompt_package(
                    normalized,
                    retrieved.sources,
                    prepared.texts,
                    memory=memory_context,
                    json_output=False,
                )

            metrics.prompt_build_ms = (time.perf_counter() - phase_started) * 1000.0
            self._record_prompt_metrics(
                metrics,
                package,
                retrieved,
                memory_context,
                normalized,
            )
            messages = [
                {
                    "role": "system",
                    "content": system_prompt_for_turn,
                },
                {
                    "role": "user",
                    "content": package.user_prompt,
                },
            ]
            generation_started = time.perf_counter()
            first_turn_token_ms: float | None = None
            fragments: list[str] = []
            call = OllamaCallMetrics(
                call_type="generation_main",
                model=self.runtime_config.ollama.model,
                streaming=True,
            )

            try:
                for fragment in self.llm.stream_chat(
                    messages,
                    call_type=call.call_type,
                    telemetry=call,
                ):
                    if first_turn_token_ms is None and fragment.strip():
                        first_turn_token_ms = (time.perf_counter() - turn_started) * 1000.0
                        metrics.turn_time_to_first_token_ms = first_turn_token_ms
                    fragments.append(fragment)
            finally:
                metrics.absorb_ollama_call(call)

            answer = "".join(fragments).strip()
            yield RAGStreamEvent(
                event_type="validation_started",
                metadata={"attempt": "initial"},
            )

            try:
                answer, metrics.truncation_salvaged = self._finalize_likely_truncation(
                    answer,
                    generated_token_count=call.generated_token_count,
                    response_language=language.language.value,
                )
                citations, insufficient = self._validate_answer_with_metrics(
                    answer=answer,
                    available_source_count=len(retrieved.sources),
                    attempt="initial",
                    metrics=metrics,
                )
            except CitationValidationError as error:
                self._log_validation_rejection(
                    attempt="initial",
                    error=error,
                    available_source_count=len(retrieved.sources),
                    raw_output=answer,
                    final=True,
                )
                metrics.total_ms = (time.perf_counter() - turn_started) * 1000.0
                yield RAGStreamEvent(
                    event_type="error",
                    content=f"Réponse invalide : {error}",
                    metadata={"latency": metrics.to_dict()},
                )
                return

            generation_ms = (time.perf_counter() - generation_started) * 1000.0
            metrics.total_ms = (time.perf_counter() - turn_started) * 1000.0
            metrics.citations = citations
            cited_sources = self._cited_sources(retrieved.sources, citations)
            metrics.displayed_source_count = len(cited_sources)
            response = self._build_response(
                question=normalized,
                answer=answer,
                cited_sources=cited_sources,
                cited_source_numbers=citations,
                insufficient_context=insufficient,
                retrieved=retrieved,
                generation_ms=generation_ms,
                total_ms=metrics.total_ms,
                first_token_ms=first_turn_token_ms,
                latency=metrics,
            )
            LOGGER.info("RAG_LATENCY %s", metrics.concise_log_fields())
            yield RAGStreamEvent(
                event_type="token",
                content=answer,
                metadata={"attempt": "initial", "response_validated": True},
            )
            yield RAGStreamEvent(
                event_type="sources",
                sources=cited_sources,
                metadata={"citations": citations},
            )
            yield RAGStreamEvent(
                event_type="completed",
                response=response,
                metadata={"latency": metrics.to_dict()},
            )
            return
        except KeyboardInterrupt:
            raise
        except (RAGError, OllamaError, ValueError, TypeError) as error:
            metrics.total_ms = (time.perf_counter() - turn_started) * 1000.0
            LOGGER.error(
                "Streaming RAG failed error_type=%s reason=%s",
                type(error).__name__,
                error,
            )
            yield RAGStreamEvent(
                event_type="error",
                content=str(error),
                metadata={"latency": metrics.to_dict()},
            )
        except Exception as error:
            metrics.total_ms = (time.perf_counter() - turn_started) * 1000.0
            LOGGER.exception("Unexpected streaming RAG failure")
            yield RAGStreamEvent(
                event_type="error",
                content="Une erreur inattendue a interrompu la réponse.",
                metadata={
                    "error_type": type(error).__name__,
                    "latency": metrics.to_dict(),
                },
            )
        finally:
            self._active_metrics = None

    def _coerce_memory_context(
        self,
        history: (list[ChatMessage] | ConversationHistoryContext | None),
    ) -> ConversationHistoryContext:
        """Accept the new memory object while preserving list compatibility."""

        if isinstance(history, ConversationHistoryContext):
            return history

        memory = self.create_conversation_memory()
        pending_user: str | None = None

        for message in history or []:
            if message.role == "user":
                pending_user = message.content
            elif pending_user is not None:
                memory.add_turn(pending_user, message.content)
                pending_user = None

        return memory.build_history_context()

    def create_conversation_memory(
        self,
        *,
        enabled: bool | None = None,
    ) -> ConversationMemory:
        """Create session memory from production configuration."""

        config = self.runtime_config.conversation
        return ConversationMemory(
            enabled=config.enabled if enabled is None else enabled,
            recent_turns=config.recent_turns,
            summary_max_tokens=config.summary_max_tokens,
            recent_history_max_tokens=(config.recent_history_max_tokens),
            total_history_max_tokens=config.total_history_max_tokens,
        )

    def _resolve_turn_source_mode(
        self,
        requested_mode: str,
        *,
        question: str,
        follow_up: bool,
        state: ConversationState,
    ) -> str:
        """Resolve explicit, inherited and automatic source scope."""

        if self.quality_engine is None:
            return requested_mode

        configured = self._quality_source_mode(requested_mode)
        explicit = detect_explicit_source_mode(question)

        if configured != "auto":
            state.record_source_scope(
                configured,
                explicit=True,
                origin="runtime_configuration",
            )
            return configured
        if requests_automatic_source_scope(question):
            state.release_source_scope()
            return "auto"
        if explicit is not None:
            state.record_source_scope(
                explicit,
                explicit=True,
                origin="user_question",
            )
            return explicit
        if follow_up and state.source_scope_explicit:
            return state.current_source_mode

        if not follow_up:
            state.release_source_scope()
        return "auto"

    @staticmethod
    def _quality_source_mode(source_mode: str) -> str:
        """Normalize legacy terminal names to the quality router contract."""

        normalized = source_mode.strip().casefold()
        aliases = {
            "automatic": "auto",
            "atelier": "report",
        }
        return aliases.get(normalized, normalized)

    @staticmethod
    def _record_prompt_metrics(
        metrics: RAGLatencyMetrics,
        package: PromptPackage,
        retrieved: _RetrievedContext,
        memory: ConversationHistoryContext,
        question: str,
    ) -> None:
        """Record optimized and old-equivalent prompt sizes."""

        size = package.size
        metrics.prompt_character_count = size.total_characters
        metrics.estimated_prompt_tokens = size.total_tokens
        metrics.system_prompt_token_count = size.system_tokens
        metrics.question_token_count = size.question_tokens
        baseline_characters = (
            len(STREAMING_SYSTEM_PROMPT)
            + len(question)
            + sum(len(text) for text in retrieved.source_texts)
            + sum(len(message.content) for message in memory.messages())
            + 500
        )
        metrics.baseline_equivalent_prompt_characters = baseline_characters
        metrics.baseline_equivalent_prompt_tokens = estimate_tokens("x" * baseline_characters)

    def close(self) -> None:
        """Release the persistent Ollama HTTP client."""

        close = getattr(self.llm, "close", None)

        if callable(close):
            close()
