"""End-to-end production RAG backed by immutable dev_best_v3."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable, Iterator, Sequence
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
    OllamaResponseValidationError,
    load_ollama_config,
)
from phosprocess.observability.latency import (
    OllamaCallMetrics,
    RAGLatencyMetrics,
    WarmupMetrics,
    estimate_tokens,
)
from phosprocess.rag.adaptive_router import (
    AdaptiveRouteDecision,
    RequestPath,
    decide_request_path,
)
from phosprocess.rag.citations import (
    CitationValidationError,
    extract_citations,
    is_controlled_insufficient_answer,
)
from phosprocess.rag.context_window import (
    PreparedDocumentContext,
    prepare_document_context,
    query_terms,
)
from phosprocess.rag.conversation_memory import (
    ConversationHistoryContext,
    ConversationMemory,
)
from phosprocess.rag.conversation_state import ConversationState
from phosprocess.rag.fidelity import (
    enforce_answer_contract,
    prune_unsupported_claims,
    validate_claim_support,
)
from phosprocess.rag.followup_resolver import resolve_standalone_query
from phosprocess.rag.language import detect_response_language
from phosprocess.rag.prompts import (
    REPAIR_SYSTEM_PROMPT,
    STREAMING_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    PromptPackage,
    build_direct_prompt_package,
    build_prompt_package,
    build_quality_prompt_package,
    build_repair_prompt,
    resolve_follow_up,
)
from phosprocess.rag.quality_retrieval import (
    QualityRetrievalEngine,
    QualityRetrievalResult,
)
from phosprocess.rag.question_classifier import classify_question
from phosprocess.rag.schemas import (
    ChatMessage,
    GroundedAnswerPayload,
    RAGResponse,
    RAGSource,
    RAGStreamEvent,
    RAGTimings,
)
from phosprocess.rag.source_policy import (
    AppliedSourcePolicy,
    SourcePolicyConfig,
    SourcePolicyDecision,
    decide_source_policy,
    detect_explicit_active_source,
    document_id_from_source,
)
from phosprocess.reranking.reranker import (
    BGEReranker,
    clean_passage_text,
    load_reranking_config,
)
from phosprocess.retrieval.domain_router import (
    detect_explicit_source_mode,
    requests_automatic_source_scope,
    route_query,
)
from phosprocess.retrieval.evidence_bundle import EvidenceBundle
from phosprocess.retrieval.hybrid import (
    HybridRetriever,
    expand_lexical_query,
)
from phosprocess.retrieval.v3_selection import (
    select_with_lexical_safeguard,
)

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval"
    / "v0.1"
    / "frozen"
    / "dev_best_v3"
)
DEFAULT_RUNTIME_CONFIG_PATH = PROJECT_ROOT / "configs" / "rag_production.yaml"
DEFAULT_EMBEDDING_CONFIG_PATH = PROJECT_ROOT / "configs" / "embeddings.yaml"
EXPECTED_SELECTED_VARIANT = "lexical_safeguard_001"
_WHITESPACE = re.compile(r"\s+")


def _controlled_fallback_for_language(language: str) -> str:
    normalized = language.strip().lower()
    if normalized.startswith("en"):
        return (
            "The retrieved passages do not provide enough information to "
            "answer this question precisely."
        )
    if normalized.startswith("ar"):
        return (
            "لا توفر المقاطع المسترجعة معلومات كافية للإجابة عن هذا السؤال بدقة."
        )
    return (
        "Les passages retrouvés ne permettent pas de répondre précisément "
        "à cette question."
    )


class RAGError(RuntimeError):
    """Base class for production RAG failures."""


class RAGConfigurationError(RAGError):
    """Raised when the frozen runtime configuration is invalid."""


class RAGRetrievalError(RAGError):
    """Raised when retrieval cannot produce five grounded sources."""


class RAGGenerationError(RAGError):
    """Raised when local Qwen generation fails."""


class RAGResponseValidationError(RAGError):
    """Raised when citations or structured output are invalid."""


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
            raise RAGConfigurationError(
                "dev_best_v3 ne sélectionne pas lexical_safeguard_001."
            )

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
                raise RAGConfigurationError(
                    f"Paramètre figé inattendu {name}: {actual}."
                )

        if self.query_expansion is not True:
            raise RAGConfigurationError(
                "L'expansion de requête figée doit être activée."
            )

        if self.lexical_source != "bm25":
            raise RAGConfigurationError(
                "La source lexicale figée doit être BM25."
            )

        if self.duplicate_policy != "skip":
            raise RAGConfigurationError(
                "La politique figée doit ignorer les doublons."
            )

        if self.fallback != "next_reranker_result":
            raise RAGConfigurationError(
                "Le fallback figé doit utiliser le prochain résultat."
            )


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

        if (
            self.summary_max_tokens + self.recent_history_max_tokens
            > self.total_history_max_tokens
        ):
            raise ValueError("Budgets de mémoire conversationnelle invalides.")


@dataclass(frozen=True, slots=True)
class GenerationRuntimeConfig:
    """Document-context budgets without changing retrieval selection."""

    max_context_tokens_per_source: int = 350
    max_total_document_context_tokens: int = 1750
    maximum_answer_words: int = 100

    def __post_init__(self) -> None:
        if self.max_context_tokens_per_source <= 0:
            raise ValueError(
                "max_context_tokens_per_source doit être positif."
            )

        if self.max_total_document_context_tokens < 5:
            raise ValueError(
                "max_total_document_context_tokens est insuffisant."
            )

        if self.maximum_answer_words <= 0:
            raise ValueError("maximum_answer_words doit être positif.")


@dataclass(frozen=True, slots=True)
class WarmupRuntimeConfig:
    """One-time startup warm-up switches."""

    enabled: bool = True
    embedding: bool = True
    reranker: bool = True
    ollama: bool = True


def _default_source_policy_config() -> SourcePolicyConfig:
    """Return eight-document defaults for legacy/rollback indexes."""

    sources = (
        "01_becker_phosphates_and_phosphoric_acid.pdf",
        "02_chemical_engineering_thermodynamics_9e.pdf",
        "03_fundamentals_heat_mass_transfer.pdf",
        "04_rapport_atelier_acide_phosphorique.pdf",
        "05_perrys_chemical_engineers_handbook_9e.pdf",
        "06_mullin_crystallization_4e.pdf",
        "07_process_dynamics_control_seborg_4e.pdf",
        "08_transport_phenomena_bird_2e.pdf",
    )
    return SourcePolicyConfig(
        enabled=False,
        default_priority=sources,
        domain_routes={
            "general": (sources[4], sources[0]),
            "phosphoric_acid": (sources[0], sources[3], sources[4]),
            "plant_specific": (sources[3], sources[0], sources[4]),
            "thermodynamics": (sources[1], sources[4], sources[3]),
            "heat_transfer": (sources[2], sources[4], sources[3]),
            "equipment": (sources[4], sources[0], sources[3]),
            "crystallization": (sources[5], sources[0]),
            "control": (sources[6], sources[3]),
            "transport": (sources[7], sources[4]),
        },
        minimum_preferred_chunks=2,
        allow_fallback=True,
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
            raise ValueError(
                "maximum_question_characters doit être positif."
            )

        if self.source_excerpt_characters <= 0:
            raise ValueError(
                "source_excerpt_characters doit être positif."
            )


@dataclass(slots=True)
class _RetrievedContext:
    """Internal retrieval result shared by blocking and streaming calls."""

    candidates: list[Any]
    selected: list[Any]
    sources: list[RAGSource]
    source_texts: list[str]
    hybrid_response: Any
    reranked_response: Any
    source_policy: AppliedSourcePolicy | None = None
    quality_result: QualityRetrievalResult | None = None
    response_language: str | None = None
    question_type: str | None = None


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
        raise RAGConfigurationError(
            f"Section {name} absente ou invalide."
        )

    return value


def load_runtime_config(
    path: Path = DEFAULT_RUNTIME_CONFIG_PATH,
) -> RAGRuntimeConfig:
    """Load generation, memory, context and warm-up settings."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise RAGConfigurationError(
            "Configuration RAG de production invalide."
        )

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

    if not all(
        conversation.get(field_name) is True
        for field_name in excluded_memory_fields
    ):
        raise RAGConfigurationError(
            "La mémoire doit exclure sources, scores, prompts et logs."
        )

    default_priority = source_policy.get("default_priority")
    domain_routes = source_policy.get("domain_routes")

    if (
        not isinstance(default_priority, list)
        or not all(
            isinstance(source, str) and source.strip()
            for source in default_priority
        )
        or not isinstance(domain_routes, dict)
    ):
        raise RAGConfigurationError(
            "Configuration source_policy invalide."
        )

    parsed_routes: dict[str, tuple[str, ...]] = {}

    for route_name, route in domain_routes.items():
        if not isinstance(route_name, str) or not isinstance(route, dict):
            raise RAGConfigurationError(
                "Route documentaire invalide dans source_policy."
            )
        preferred_sources = route.get("preferred_sources")
        if (
            not isinstance(preferred_sources, list)
            or not all(
                isinstance(source, str) and source.strip()
                for source in preferred_sources
            )
        ):
            raise RAGConfigurationError(
                f"Sources préférées invalides : {route_name}."
            )
        parsed_routes[route_name] = tuple(preferred_sources)

    if "general" not in parsed_routes:
        raise RAGConfigurationError(
            "Route documentaire absente : general."
        )

    return RAGRuntimeConfig(
        ollama=load_ollama_config(path),
        maximum_question_characters=int(
            response["maximum_question_characters"]
        ),
        source_excerpt_characters=int(
            response["source_excerpt_characters"]
        ),
        conversation=ConversationRuntimeConfig(
            enabled=bool(conversation["enabled"]),
            strategy=str(conversation["strategy"]),
            recent_turns=int(conversation["recent_turns"]),
            summary_max_tokens=int(
                conversation["summary_max_tokens"]
            ),
            recent_history_max_tokens=int(
                conversation["recent_history_max_tokens"]
            ),
            total_history_max_tokens=int(
                conversation["total_history_max_tokens"]
            ),
        ),
        generation=GenerationRuntimeConfig(
            max_context_tokens_per_source=int(
                generation["max_context_tokens_per_source"]
            ),
            max_total_document_context_tokens=int(
                generation["max_total_document_context_tokens"]
            ),
            maximum_answer_words=int(
                generation.get("maximum_answer_words", 100)
            ),
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
            minimum_preferred_chunks=int(
                source_policy["minimum_preferred_chunks"]
            ),
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
            raise RAGConfigurationError(
                f"Composant dev_best_v3 introuvable: {required_path}"
            )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    safeguard = yaml.safe_load(
        safeguard_path.read_text(encoding="utf-8")
    )

    if not isinstance(manifest, dict) or not isinstance(safeguard, dict):
        raise RAGConfigurationError(
            "Le snapshot dev_best_v3 est invalide."
        )

    if (
        manifest.get("status") != "frozen_dev_best_v3"
        or manifest.get("scope") != "dev_only"
        or manifest.get("single_selected_variant") is not True
    ):
        raise RAGConfigurationError(
            "Le manifeste dev_best_v3 ne respecte pas le gel DEV."
        )

    selection = safeguard.get("selection")
    retrieval = safeguard.get("retrieval")

    if not isinstance(selection, dict) or not isinstance(retrieval, dict):
        raise RAGConfigurationError(
            "Configuration safeguard figée invalide."
        )

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
        reranker_leading_slots=int(
            selection["reranker_leading_slots"]
        ),
        lexical_source=str(selection["lexical_source"]),
        duplicate_policy=str(selection["duplicate_policy"]),
        fallback=str(selection["fallback"]),
        retrieval_config_path=retrieval_path,
        reranking_config_path=reranking_path,
    )


def _verify_snapshot_components(
    *,
    snapshot_directory: Path,
    manifest: dict[str, Any],
    verify_runtime_sources: bool,
) -> None:
    """Verify snapshot files and, only on request, the legacy v3 runtime."""

    components = manifest.get("components")

    if not isinstance(components, list):
        raise RAGConfigurationError(
            "Liste des composants absente du manifeste."
        )

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
            raise RAGConfigurationError(
                f"Composant figé manquant: {file_name}."
            )

        if sha256_file(frozen_path) != record["frozen_sha256"]:
            raise RAGConfigurationError(
                f"Empreinte figée incorrecte: {file_name}."
            )

    if not verify_runtime_sources:
        return

    active_sources = {
        "hybrid.py": (
            PROJECT_ROOT
            / "src"
            / "phosprocess"
            / "retrieval"
            / "hybrid.py"
        ),
        "reranker.py": (
            PROJECT_ROOT
            / "src"
            / "phosprocess"
            / "reranking"
            / "reranker.py"
        ),
        "v3_selection.py": (
            PROJECT_ROOT
            / "src"
            / "phosprocess"
            / "retrieval"
            / "v3_selection.py"
        ),
    }

    for file_name, active_path in active_sources.items():
        if sha256_file(active_path) != records[file_name]["source_sha256"]:
            raise RAGConfigurationError(
                f"Le code runtime diffère de dev_best_v3: {file_name}."
            )


def validate_question(
    question: str,
    *,
    maximum_characters: int,
) -> str:
    """Normalize and validate one user question."""

    if not isinstance(question, str):
        raise TypeError("La question doit être une chaîne de caractères.")

    if "\x00" in question:
        raise ValueError(
            "La question contient un caractère nul interdit."
        )

    normalized = _WHITESPACE.sub(" ", question).strip()

    if not normalized:
        raise ValueError("La question ne peut pas être vide.")

    if len(normalized) > maximum_characters:
        raise ValueError(
            "La question dépasse la longueur maximale autorisée."
        )

    if not any(character.isalnum() for character in normalized):
        raise ValueError(
            "La question doit contenir un caractère alphanumérique."
        )

    return normalized


class PhosProcessRAG:
    """Long-lived RAG using frozen v3 settings with the active runtime."""

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
        self.frozen_config = frozen_config or load_frozen_v3_config(
            verify_integrity=verify_snapshot,
        )
        self.runtime_config = runtime_config or load_runtime_config()
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
            reranker_config = load_reranking_config(
                self.frozen_config.reranking_config_path
            )
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

        if (
            self.active_knowledge_base is not None
            and QualityRetrievalEngine.is_quality_index(
                self.active_knowledge_base.version_directory
            )
        ):
            self.quality_engine = QualityRetrievalEngine(
                version_directory=self.active_knowledge_base.version_directory,
                retriever=self.retriever,
                reranker=self.reranker,
                require_sparse_index=True,
            )

        self._lifecycle_counts["ollama_client"] = 1
        self.initial_loading_ms = (
            time.perf_counter() - loading_started
        ) * 1000.0
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
            dense_index_directory=(
                self.active_knowledge_base.dense_index_directory
            ),
            bm25_index_directory=(
                self.active_knowledge_base.bm25_index_directory
            ),
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
            "documents": [
                dict(document)
                for document in active.documents
            ],
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
                "ollama_http_client": id(
                    getattr(self.llm, "stream_http_client", None)
                ),
            },
            "embedding_hook_installed": self._embedding_hook_installed,
            "reranker_tokenizer_hook_installed": (
                self._reranker_tokenizer_hook_installed
            ),
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
                metrics.embedding_ms = (
                    time.perf_counter() - started
                ) * 1000.0

        if config.reranker and hasattr(self.reranker, "_compute_scores"):
            started = time.perf_counter()
            self.reranker._compute_scores(
                pairs=[["procédé", "passage industriel"]]
            )
            metrics.reranker_ms = (
                time.perf_counter() - started
            ) * 1000.0

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
            metrics.ollama_ms = (
                time.perf_counter() - started
            ) * 1000.0
            metrics.ollama_call_count = 1

        metrics.total_ms = (
            time.perf_counter() - total_started
        ) * 1000.0
        self._warmup_metrics = metrics
        return metrics

    @staticmethod
    def _direct_language_code(
        decision: AdaptiveRouteDecision,
        question: str,
        *,
        language_mode: str,
        last_explicit_language: str | None = None,
    ) -> str:
        """Resolve direct-output language without document-language leakage."""

        if decision.requested_output_language is not None:
            return decision.requested_output_language

        detected = detect_response_language(
            question,
            last_explicit_language=last_explicit_language,
            mode=language_mode,
        )
        return detected.language.value

    def _build_direct_response(
        self,
        *,
        question: str,
        answer: str,
        decision: AdaptiveRouteDecision,
        response_language: str,
        generation_ms: float,
        total_ms: float,
        first_token_ms: float | None = None,
        latency: RAGLatencyMetrics | None = None,
    ) -> RAGResponse:
        """Build a source-free response for an intentional retrieval bypass."""

        model_name = getattr(
            self.llm,
            "model_name",
            self.runtime_config.ollama.model,
        )
        return RAGResponse(
            question=question,
            answer=answer,
            sources=[],
            cited_source_numbers=[],
            insufficient_context=False,
            model_name=str(model_name),
            selected_variant=self.frozen_config.selected_variant,
            snapshot_sha256=self.frozen_config.snapshot_sha256,
            candidate_count=0,
            selected_count=0,
            source_policy_route=RequestPath.DIRECT_LLM.value,
            source_policy_mode="bypass",
            source_policy_primary=None,
            source_policy_fallback_used=False,
            source_policy_forced=False,
            response_language=response_language,
            standalone_query=question,
            question_type=(
                decision.direct_intent.value
                if decision.direct_intent is not None
                else "direct"
            ),
            detected_domains=[],
            timings=RAGTimings(
                hybrid_ms=0.0,
                reranking_ms=0.0,
                generation_ms=generation_ms,
                total_ms=total_ms,
                first_token_ms=first_token_ms,
            ),
            latency=latency.to_dict() if latency is not None else {},
        )

    def _answer_direct(
        self,
        question: str,
        decision: AdaptiveRouteDecision,
        *,
        language_mode: str,
        started: float,
    ) -> RAGResponse:
        """Execute one self-contained request without touching retrieval."""

        response_language = self._direct_language_code(
            decision,
            question,
            language_mode=language_mode,
        )
        system_prompt, package = build_direct_prompt_package(
            question,
            decision,
            json_output=True,
        )
        generation_started = time.perf_counter()

        try:
            payload, _raw = self.llm.chat_json_with_raw(
                user_prompt=package.user_prompt,
                system_prompt=system_prompt,
                response_model=GroundedAnswerPayload,
            )
        except OllamaError as error:
            raise RAGGenerationError(str(error)) from error
        except Exception as error:
            raise RAGGenerationError(
                "La génération directe locale Qwen a échoué."
            ) from error

        generation_ms = (
            time.perf_counter() - generation_started
        ) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0
        return self._build_direct_response(
            question=question,
            answer=payload.answer,
            decision=decision,
            response_language=response_language,
            generation_ms=generation_ms,
            total_ms=total_ms,
        )

    def _stream_direct_request(
        self,
        question: str,
        decision: AdaptiveRouteDecision,
        *,
        language_mode: str,
        last_explicit_language: str | None,
        metrics: RAGLatencyMetrics,
        turn_started: float,
    ) -> Iterator[RAGStreamEvent]:
        """Stream a direct answer while explicitly reporting retrieval bypass."""

        response_language = self._direct_language_code(
            decision,
            question,
            language_mode=language_mode,
            last_explicit_language=last_explicit_language,
        )
        metrics.retrieval_query = question
        metrics.source_policy_route = RequestPath.DIRECT_LLM.value
        metrics.source_policy_mode = "bypass"
        metrics.source_policy_primary = "Aucune"
        metrics.source_policy_attempt_count = 0
        direct_type = (
            decision.direct_intent.value
            if decision.direct_intent is not None
            else "direct"
        )

        yield RAGStreamEvent(
            event_type="retrieval_started",
            metadata={
                "question_id": metrics.question_id,
                "follow_up": False,
                "reformulated": False,
                "reformulation_method": "adaptive_router",
                "standalone_query": question,
                "language": response_language,
                "question_type": direct_type,
                "source_policy_route": RequestPath.DIRECT_LLM.value,
                "source_policy_primary": "Aucune",
                "retrieval_skipped": True,
                "routing_reason": decision.reason,
            },
        )
        yield RAGStreamEvent(
            event_type="retrieval_completed",
            sources=[],
            metadata={
                "candidate_count": 0,
                "selected_count": 0,
                "hybrid_ms": 0.0,
                "reranking_ms": 0.0,
                "source_policy_route": RequestPath.DIRECT_LLM.value,
                "source_policy_primary": "Aucune",
                "source_policy_fallback_used": False,
                "query_expansion": [],
                "hierarchical_sections": [],
                "section_retrieval_ms": 0.0,
                "retrieval_skipped": True,
                "routing_reason": decision.reason,
            },
        )

        prompt_started = time.perf_counter()
        system_prompt, package = build_direct_prompt_package(
            question,
            decision,
            json_output=False,
        )
        metrics.prompt_build_ms = (
            time.perf_counter() - prompt_started
        ) * 1000.0
        metrics.prompt_character_count = package.size.total_characters
        metrics.estimated_prompt_tokens = package.size.total_tokens
        metrics.system_prompt_token_count = package.size.system_tokens
        metrics.question_token_count = package.size.question_tokens
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": package.user_prompt},
        ]
        call = OllamaCallMetrics(
            call_type="generation_direct",
            model=self.runtime_config.ollama.model,
            streaming=True,
        )
        fragments: list[str] = []
        generation_started = time.perf_counter()
        first_turn_token_ms: float | None = None

        try:
            for fragment in self.llm.stream_chat(
                messages,
                call_type=call.call_type,
                telemetry=call,
            ):
                if first_turn_token_ms is None and fragment.strip():
                    first_turn_token_ms = (
                        time.perf_counter() - turn_started
                    ) * 1000.0
                    metrics.turn_time_to_first_token_ms = (
                        first_turn_token_ms
                    )
                fragments.append(fragment)
                yield RAGStreamEvent(
                    event_type="token",
                    content=fragment,
                    metadata={
                        "attempt": "direct",
                        "retrieval_skipped": True,
                    },
                )
        finally:
            metrics.absorb_ollama_call(call)

        answer = "".join(fragments).strip()
        if not answer:
            raise RAGGenerationError(
                "Qwen a retourné une réponse directe vide."
            )

        yield RAGStreamEvent(
            event_type="validation_started",
            metadata={
                "attempt": "direct",
                "citation_validation_skipped": True,
            },
        )
        generation_ms = (
            time.perf_counter() - generation_started
        ) * 1000.0
        metrics.total_ms = (
            time.perf_counter() - turn_started
        ) * 1000.0
        response = self._build_direct_response(
            question=question,
            answer=answer,
            decision=decision,
            response_language=response_language,
            generation_ms=generation_ms,
            total_ms=metrics.total_ms,
            first_token_ms=first_turn_token_ms,
            latency=metrics,
        )
        LOGGER.info(
            "ADAPTIVE_RAG bypass intent=%s reason=%s %s",
            direct_type,
            decision.reason,
            metrics.concise_log_fields(),
        )
        yield RAGStreamEvent(
            event_type="sources",
            sources=[],
            metadata={"citations": [], "retrieval_skipped": True},
        )
        yield RAGStreamEvent(
            event_type="completed",
            response=response,
            metadata={"latency": metrics.to_dict()},
        )

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
            maximum_characters=(
                self.runtime_config.maximum_question_characters
            ),
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
            effective_source_mode = (
                detect_explicit_source_mode(normalized)
                or self._quality_source_mode(source_mode)
            )

        retrieved = self._retrieve_with_source_policy(
            normalized,
            policy_question=normalized,
            source_mode=effective_source_mode,
        )
        system_prompt = SYSTEM_PROMPT
        repair_system_prompt = REPAIR_SYSTEM_PROMPT

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
            repair_system_prompt = (
                REPAIR_SYSTEM_PROMPT
                + "\nPreserve the existing answer language: "
                + language.language.prompt_name
                + "."
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
                maximum_answer_words=(
                    self.runtime_config.generation.maximum_answer_words
                ),
            )

        generation_started = time.perf_counter()
        payload, citations, insufficient = self._generate_json_answer(
            user_prompt=package.user_prompt,
            system_prompt=system_prompt,
            repair_system_prompt=repair_system_prompt,
            available_source_count=len(retrieved.sources),
            evidence_bundles=(
                retrieved.quality_result.bundles
                if retrieved.quality_result is not None
                else None
            ),
            question_type=retrieved.question_type,
            response_language=retrieved.response_language,
            comparison_subjects=self._comparison_subjects(
                retrieved.quality_result
            ),
            contract_question=normalized,
            balance_kind=(
                retrieved.quality_result.retrieval_plan.balance_kind
                if (
                    retrieved.quality_result is not None
                    and retrieved.quality_result.retrieval_plan is not None
                )
                else None
            ),
        )
        generation_ms = (
            time.perf_counter() - generation_started
        ) * 1000.0
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
        history: (
            list[ChatMessage]
            | ConversationHistoryContext
            | None
        ) = None,
        *,
        source_mode: str = "automatic",
        language_mode: str = "auto",
    ) -> Iterator[RAGStreamEvent]:
        """Yield real Ollama tokens plus phase-level latency metrics."""

        turn_started = time.perf_counter()
        self._turn_counter += 1
        metrics = RAGLatencyMetrics(
            question_id=f"session-{self._turn_counter:04d}"
        )

        try:
            phase_started = time.perf_counter()
            normalized = validate_question(
                question,
                maximum_characters=(
                    self.runtime_config.maximum_question_characters
                ),
            )
            metrics.question_validation_ms = (
                time.perf_counter() - phase_started
            ) * 1000.0

            phase_started = time.perf_counter()
            memory_context = self._coerce_memory_context(history)
            metrics.memory_build_ms = (
                time.perf_counter() - phase_started
            ) * 1000.0
            metrics.history_turn_count = len(
                memory_context.recent_turns
            )
            metrics.summary_token_count = (
                memory_context.summary_token_count
            )
            metrics.recent_history_token_count = (
                memory_context.recent_history_token_count
            )
            history_messages = memory_context.messages()

            phase_started = time.perf_counter()
            business_state = (
                memory_context.business_state or ConversationState()
            )
            adaptive_decision = decide_request_path(
                normalized,
                source_mode=source_mode,
            )

            if adaptive_decision.path is RequestPath.DIRECT_LLM:
                metrics.followup_detection_ms = (
                    time.perf_counter() - phase_started
                ) * 1000.0
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
                quality_resolution = resolve_standalone_query(
                    normalized,
                    state=business_state,
                )
                retrieval_query = quality_resolution.standalone_query
                follow_up = quality_resolution.followup_detected
                reformulated = (
                    quality_resolution.standalone_query != normalized
                )
                reformulation_method = quality_resolution.resolver_type
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

            detection_elapsed = (
                time.perf_counter() - phase_started
            ) * 1000.0
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
            business_state.record_question_type(
                classification.question_type.value
            )

            if self.quality_engine is not None:
                routing_preview = route_query(
                    retrieval_query,
                    catalog=self.quality_engine.catalog,
                    source_mode=self._quality_source_mode(
                        effective_source_mode
                    ),
                    question_type=classification.question_type.value,
                    focus_entity=business_state.focus_entity,
                )
                route_label = ",".join(
                    domain.value
                    for domain, _confidence in (
                        routing_preview.detected_domains
                    )
                )
                primary_label = (
                    next(
                        entry.display_title
                        for entry in self.quality_engine.catalog.documents
                        if entry.document_id
                        == routing_preview.preferred_documents[0]
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
                    "hybrid_ms": float(
                        retrieved.hybrid_response.total_duration_ms
                    ),
                    "reranking_ms": metrics.reranking_ms,
                    "source_policy_route": (
                        metrics.source_policy_route
                    ),
                    "source_policy_primary": (
                        metrics.source_policy_primary
                    ),
                    "source_policy_fallback_used": (
                        metrics.source_policy_fallback_used
                    ),
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

            metrics.excerpt_preparation_ms = (
                time.perf_counter() - phase_started
            ) * 1000.0

            if prepared is not None:
                metrics.document_context_token_count = (
                    prepared.total_tokens
                )

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
                    maximum_answer_words=(
                        self.runtime_config.generation.maximum_answer_words
                    ),
                )

            metrics.prompt_build_ms = (
                time.perf_counter() - phase_started
            ) * 1000.0
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
            repair_started: float | None = None
            quality_bundles = (
                retrieved.quality_result.bundles
                if retrieved.quality_result is not None
                else None
            )
            comparison_subjects = self._comparison_subjects(
                retrieved.quality_result
            )
            balance_kind = (
                retrieved.quality_result.retrieval_plan.balance_kind
                if (
                    retrieved.quality_result is not None
                    and retrieved.quality_result.retrieval_plan is not None
                )
                else None
            )
            buffer_until_validated = quality_bundles is not None
            deterministic_pruning_applied = False

            for attempt_index, attempt in enumerate(
                ("initial", "repair")
            ):
                fragments: list[str] = []
                call = OllamaCallMetrics(
                    call_type=(
                        "generation_main"
                        if attempt == "initial"
                        else "citation_repair"
                    ),
                    model=self.runtime_config.ollama.model,
                    streaming=True,
                )

                try:
                    for fragment in self.llm.stream_chat(
                        messages,
                        call_type=call.call_type,
                        telemetry=call,
                    ):
                        if (
                            first_turn_token_ms is None
                            and fragment.strip()
                        ):
                            first_turn_token_ms = (
                                time.perf_counter() - turn_started
                            ) * 1000.0
                            metrics.turn_time_to_first_token_ms = (
                                first_turn_token_ms
                            )

                        fragments.append(fragment)

                        if not buffer_until_validated:
                            yield RAGStreamEvent(
                                event_type="token",
                                content=fragment,
                                metadata={"attempt": attempt},
                            )
                finally:
                    metrics.absorb_ollama_call(call)

                answer = "".join(fragments).strip()
                yield RAGStreamEvent(
                    event_type="validation_started",
                    metadata={"attempt": attempt},
                )

                try:
                    self._reject_likely_truncation(
                        answer,
                        generated_token_count=(
                            call.generated_token_count
                        ),
                    )
                    citations, insufficient = (
                        self._validate_answer_with_metrics(
                            answer=answer,
                            available_source_count=len(
                                retrieved.sources
                            ),
                            attempt=attempt,
                            metrics=metrics,
                            evidence_bundles=quality_bundles,
                        )
                    )
                except CitationValidationError as error:
                    self._log_validation_rejection(
                        attempt=attempt,
                        error=error,
                        available_source_count=len(
                            retrieved.sources
                        ),
                        raw_output=answer,
                        final=attempt_index == 1,
                    )

                    if quality_bundles is not None and attempt == "initial":
                        pruned = prune_unsupported_claims(
                            answer,
                            list(quality_bundles),
                            fallback_language=language.language.value,
                            question_type=classification.question_type.value,
                        )
                        answer = pruned.answer
                        deterministic_pruning_applied = True

                        LOGGER.info(
                            "RAG deterministic pruning removed_claims=%d "
                            "inherited_citations=%d fallback=%s "
                            "missing_required=%s atomic_plan=%s "
                            "reconstructed_claims=%d",
                            len(pruned.removed_claims),
                            pruned.inherited_citation_count,
                            pruned.fallback_used,
                            pruned.missing_required_concepts,
                            pruned.atomic_plan_used,
                            pruned.reconstructed_claim_count,
                        )

                        if pruned.fallback_used:
                            citations = []
                            insufficient = True
                        else:
                            try:
                                citations, insufficient = (
                                    self._validate_answer_with_metrics(
                                        answer=answer,
                                        available_source_count=len(
                                            retrieved.sources
                                        ),
                                        attempt="deterministic_pruning",
                                        metrics=metrics,
                                        evidence_bundles=quality_bundles,
                                    )
                                )
                            except CitationValidationError as pruning_error:
                                LOGGER.warning(
                                    "RAG deterministic pruning fallback "
                                    "reason=%s",
                                    pruning_error,
                                )
                                answer = (
                                    "The corpus does not provide enough "
                                    "explicit evidence to answer reliably."
                                    if language.language.value.startswith("en")
                                    else (
                                        "Impossible a determiner a partir "
                                        "du corpus documentaire : les "
                                        "passages recuperes ne soutiennent "
                                        "pas une reponse fiable."
                                    )
                                )
                                citations = []
                                insufficient = True
                    else:
                        if attempt_index == 1:
                            if repair_started is not None:
                                metrics.repair_ms = (
                                    time.perf_counter() - repair_started
                                ) * 1000.0

                            metrics.total_ms = (
                                time.perf_counter() - turn_started
                            ) * 1000.0
                            yield RAGStreamEvent(
                                event_type="error",
                                content=(
                                    "Réponse invalide après une réparation : "
                                    f"{error}"
                                ),
                                metadata={"latency": metrics.to_dict()},
                            )
                            return

                        metrics.repair_attempted = True
                        metrics.repair_reason = str(error)
                        repair_started = time.perf_counter()
                        repair_prompt = build_repair_prompt(
                            original_prompt=package.user_prompt,
                            invalid_output=answer,
                            rejection_reason=str(error),
                            json_output=False,
                        )
                        messages = [
                            {
                                "role": "system",
                                "content": (
                                    REPAIR_SYSTEM_PROMPT
                                    + "\nPreserve the existing answer language: "
                                    + language.language.prompt_name
                                    + "."
                                ),
                            },
                            {
                                "role": "user",
                                "content": repair_prompt,
                            },
                        ]
                        continue

                if quality_bundles is not None:
                    contract = enforce_answer_contract(
                        answer,
                        list(quality_bundles),
                        question_type=classification.question_type.value,
                        language=language.language.value,
                        comparison_subjects=comparison_subjects,
                        question=retrieval_query,
                        balance_kind=balance_kind,
                    )
                    answer = contract.answer
                    LOGGER.info(
                        "RAG answer contract type=%s changed=%s fallback=%s "
                        "missing_roles=%s removed_claims=%d atomic_plan=%s",
                        classification.question_type.value,
                        contract.changed,
                        contract.fallback_used,
                        contract.missing_roles,
                        len(contract.removed_claims),
                        contract.atomic_plan_used,
                    )
                    if contract.fallback_used:
                        citations = []
                        insufficient = True
                    elif contract.changed or insufficient:
                        try:
                            citations, insufficient = (
                                self._validate_answer_with_metrics(
                                    answer=answer,
                                    available_source_count=len(
                                        retrieved.sources
                                    ),
                                    attempt="answer_contract",
                                    metrics=metrics,
                                    evidence_bundles=quality_bundles,
                                )
                            )
                        except CitationValidationError as contract_error:
                            LOGGER.warning(
                                "RAG deterministic answer builder fallback "
                                "reason=%s",
                                contract_error,
                            )
                            answer = _controlled_fallback_for_language(
                                language.language.value
                            )
                            citations = []
                            insufficient = True

                if repair_started is not None:
                    metrics.repair_ms = (
                        time.perf_counter() - repair_started
                    ) * 1000.0

                generation_ms = (
                    time.perf_counter() - generation_started
                ) * 1000.0
                metrics.total_ms = (
                    time.perf_counter() - turn_started
                ) * 1000.0
                metrics.citations = citations
                cited_sources = self._cited_sources(
                    retrieved.sources,
                    citations,
                )
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
                LOGGER.info(
                    "RAG_LATENCY %s",
                    metrics.concise_log_fields(),
                )
                if buffer_until_validated:
                    yield RAGStreamEvent(
                        event_type="token",
                        content=answer,
                        metadata={
                            "attempt": attempt,
                            "deterministic_pruning": (
                                deterministic_pruning_applied
                            ),
                            "response_validated": True,
                        },
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
            metrics.total_ms = (
                time.perf_counter() - turn_started
            ) * 1000.0
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
            metrics.total_ms = (
                time.perf_counter() - turn_started
            ) * 1000.0
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

    def _reject_likely_truncation(
        self,
        answer: str,
        *,
        generated_token_count: int | None,
    ) -> None:
        """Reject an output that exhausted its budget mid-sentence."""

        if (
            generated_token_count is None
            or generated_token_count
            < self.runtime_config.ollama.max_output_tokens
        ):
            return

        if re.search(
            r"(?:[.!?…]|(?:\[Source [1-5]\]))\s*$",
            answer,
        ):
            return

        raise CitationValidationError(
            "La sortie a atteint la limite de tokens au milieu "
            "d'une phrase."
        )

    def _coerce_memory_context(
        self,
        history: (
            list[ChatMessage]
            | ConversationHistoryContext
            | None
        ),
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
            recent_history_max_tokens=(
                config.recent_history_max_tokens
            ),
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

    def _retrieve_quality(
        self,
        query: str,
        *,
        original_question: str,
        question_type: str,
        source_mode: str,
        metrics: RAGLatencyMetrics | None,
    ) -> _RetrievedContext:
        """Run global quality retrieval with soft routing and evidence bundles."""

        engine = self.quality_engine

        if engine is None:
            raise RAGRetrievalError("L'index qualité n'est pas actif.")

        try:
            result = engine.retrieve(
                original_question,
                standalone_query=query,
                question_type=question_type,
                source_mode=self._quality_source_mode(source_mode),
                candidate_k=self.frozen_config.candidate_k,
                dense_candidate_k=self.frozen_config.dense_candidates,
                bm25_candidate_k=self.frozen_config.bm25_candidates,
                top_k=self.frozen_config.top_k,
                lexical_slots=self.frozen_config.lexical_slots,
            )
        except Exception as error:
            detail = str(error).strip() or repr(error)
            LOGGER.error(
                "Structured quality retrieval failed "
                "error_type=%s reason=%s question_type=%s "
                "source_mode=%s original_question=%r standalone_query=%r",
                type(error).__name__,
                detail,
                question_type,
                source_mode,
                original_question,
                query,
            )
            raise RAGRetrievalError(
                "La recherche structurée qualité a échoué : "
                f"{type(error).__name__}: {detail}"
            ) from error

        LOGGER.info(
            "ROUTING_DECISION intent=%s source_mode=%s explicit_source=%s "
            "temporal_scope=%s domains=%s primary=%s hard_filter=%s "
            "section_hints=%s",
            result.routing.question_type,
            result.routing.source_mode,
            result.routing.explicit_source or "none",
            result.routing.temporal_scope,
            ",".join(
                domain.value
                for domain, _confidence in result.routing.detected_domains
            )
            or "none",
            (
                result.routing.preferred_documents[0]
                if result.routing.preferred_documents
                else "none"
            ),
            ",".join(sorted(result.routing.hard_filter or ())) or "none",
            "|".join(result.routing.section_affinity_terms) or "none",
        )

        LOGGER.debug(
            "RAG_QUALITY_QUERY original=%r standalone=%r dense=%r bm25=%r "
            "added_terms=%s domains=%s preferred_documents=%s hard_filter=%s",
            original_question,
            result.query.standalone_query,
            result.query.dense_query,
            result.query.bm25_expanded_query,
            result.query.added_terms,
            tuple(
                domain.value
                for domain, _confidence in result.routing.detected_domains
            ),
            result.routing.preferred_documents,
            result.routing.hard_filter,
        )

        candidate_by_id = {
            candidate.chunk.chunk_id: candidate
            for candidate in result.hybrid.results
        }
        reranked_by_id = {
            reranked.chunk.chunk_id: reranked
            for reranked in result.reranking.results
        }
        selection_by_id = {
            selection.chunk_id: selection
            for selection in result.selected
        }
        sources: list[RAGSource] = []

        for bundle in result.bundles:
            candidate = candidate_by_id[bundle.anchor_chunk_id]
            reranked = reranked_by_id.get(bundle.anchor_chunk_id)
            selection = selection_by_id[bundle.anchor_chunk_id]
            child = engine.child_by_id[bundle.anchor_chunk_id]
            sources.append(
                RAGSource(
                    source_number=bundle.source_number,
                    chunk_id=bundle.anchor_chunk_id,
                    document_name=bundle.filename,
                    pages=list(range(bundle.page_start, bundle.page_end + 1)),
                    section=bundle.section,
                    excerpt=self._excerpt(bundle.display_text),
                    document_title=bundle.document_title,
                    filename=bundle.filename,
                    chapter=bundle.chapter,
                    page_start=bundle.page_start,
                    page_end=bundle.page_end,
                    anchor_chunk_id=bundle.anchor_chunk_id,
                    expanded_chunk_ids=list(bundle.expanded_chunk_ids),
                    display_text=bundle.display_text,
                    anchor_text=child.display_text,
                    domain=", ".join(child.domains),
                    chunk_type=child.chunk_type.value,
                    parent_id=child.parent_id,
                    source_boost=result.source_boosts.get(
                        bundle.anchor_chunk_id,
                        0.0,
                    ),
                    context_added_tokens=bundle.context_token_count,
                    context_truncated=bundle.context_truncated,
                    selection_source=selection.source,
                    hybrid_rank=candidate.rank,
                    rrf_score=float(candidate.rrf_score),
                    dense_rank=candidate.dense_rank,
                    dense_score=candidate.dense_score,
                    bm25_rank=candidate.bm25_rank,
                    bm25_score=candidate.bm25_score,
                    reranker_rank=(
                        reranked.rank if reranked is not None else None
                    ),
                    reranker_score=(
                        reranked.reranker_score
                        if reranked is not None
                        else None
                    ),
                )
            )

        detected_domains = tuple(
            domain.value
            for domain, _confidence in result.routing.detected_domains
        )
        catalog_by_id = {
            entry.document_id: entry
            for entry in engine.catalog.documents
        }
        preferred_files = tuple(
            catalog_by_id[document_id].canonical_filename
            for document_id in result.routing.preferred_documents
        )
        primary = preferred_files[0] if preferred_files else None
        application = AppliedSourcePolicy(
            route=",".join(detected_domains) or "general_chemical_engineering",
            mode=result.routing.source_mode,
            primary_source=primary,
            preferred_sources=preferred_files,
            selected_scope=tuple(
                dict.fromkeys(bundle.filename for bundle in result.bundles)
            ),
            fallback_used=False,
            forced=result.routing.hard_filter is not None,
            attempt_count=1,
            sufficient_preferred_chunks=sum(
                bundle.document_id in result.routing.preferred_documents
                for bundle in result.bundles
            ),
        )

        if metrics is not None:
            metrics.retrieval_query = result.query.standalone_query
            metrics.query_expansion_ms = 0.0
            metrics.dense_search_ms += result.hybrid.dense_duration_ms
            metrics.bm25_search_ms += result.hybrid.bm25_duration_ms
            metrics.hybrid_fusion_ms += max(
                0.0,
                result.hybrid.total_duration_ms
                - result.hybrid.dense_duration_ms
                - result.hybrid.bm25_duration_ms,
            )
            metrics.reranking_ms += result.reranking.reranking_duration_ms
            metrics.document_context_token_count = sum(
                bundle.token_count for bundle in result.bundles
            )

        retrieved = _RetrievedContext(
            candidates=list(result.hybrid.results),
            selected=list(result.selected),
            sources=sources,
            source_texts=[
                bundle.display_text for bundle in result.bundles
            ],
            hybrid_response=result.hybrid,
            reranked_response=result.reranking,
            quality_result=result,
        )
        return self._attach_source_policy(
            retrieved,
            application,
            metrics=metrics,
        )

    def _retrieve_with_source_policy(
        self,
        query: str,
        *,
        policy_question: str,
        source_mode: str,
        metrics: RAGLatencyMetrics | None = None,
    ) -> _RetrievedContext:
        """Apply production document routing around the frozen retrieval."""

        if self.quality_engine is not None:
            classification = classify_question(query)
            return self._retrieve_quality(
                query,
                original_question=policy_question,
                question_type=classification.question_type.value,
                source_mode=source_mode,
                metrics=metrics,
            )

        config = self.runtime_config.source_policy
        decision = self._decide_source_policy(
            policy_question,
            mode=source_mode,
        )

        if metrics is not None:
            metrics.source_policy_route = decision.route
            metrics.source_policy_mode = decision.mode
            metrics.source_policy_primary = decision.primary_label
            metrics.source_policy_forced = decision.forced

        if decision.route == "disabled":
            retrieved = self._retrieve(query, metrics=metrics)
            application = AppliedSourcePolicy(
                route=decision.route,
                mode=decision.mode,
                primary_source=None,
                preferred_sources=decision.preferred_sources,
                selected_scope=config.default_priority,
                fallback_used=False,
                forced=False,
                attempt_count=1,
                sufficient_preferred_chunks=0,
            )
            return self._attach_source_policy(
                retrieved,
                application,
                metrics=metrics,
            )

        primary_scope = (
            (decision.primary_source,)
            if decision.primary_source is not None
            else decision.preferred_sources
        )
        attempts = 1
        primary_retrieved: _RetrievedContext | None = None
        primary_error: RAGRetrievalError | None = None

        try:
            primary_retrieved = self._retrieve(
                query,
                metrics=metrics,
                document_ids=self._document_ids(primary_scope),
            )
        except RAGRetrievalError as error:
            primary_error = error

        sufficient_count = (
            self._count_sufficient_preferred_chunks(
                primary_retrieved,
                query=query,
                preferred_sources=primary_scope,
            )
            if primary_retrieved is not None
            else 0
        )

        if decision.forced:
            if primary_retrieved is None:
                raise RAGRetrievalError(
                    "La source forcée ne fournit pas vingt candidats "
                    "permettant de conserver cinq chunks."
                ) from primary_error

            application = AppliedSourcePolicy(
                route=decision.route,
                mode=decision.mode,
                primary_source=decision.primary_source,
                preferred_sources=decision.preferred_sources,
                selected_scope=primary_scope,
                fallback_used=False,
                forced=True,
                attempt_count=attempts,
                sufficient_preferred_chunks=sufficient_count,
            )
            return self._attach_source_policy(
                primary_retrieved,
                application,
                metrics=metrics,
            )

        if (
            primary_retrieved is not None
            and sufficient_count >= config.minimum_preferred_chunks
        ):
            application = AppliedSourcePolicy(
                route=decision.route,
                mode=decision.mode,
                primary_source=decision.primary_source,
                preferred_sources=decision.preferred_sources,
                selected_scope=primary_scope,
                fallback_used=False,
                forced=False,
                attempt_count=attempts,
                sufficient_preferred_chunks=sufficient_count,
            )
            return self._attach_source_policy(
                primary_retrieved,
                application,
                metrics=metrics,
            )

        if not decision.allow_fallback:
            if primary_retrieved is not None:
                return self._attach_source_policy(
                    primary_retrieved,
                    AppliedSourcePolicy(
                        route=decision.route,
                        mode=decision.mode,
                        primary_source=decision.primary_source,
                        preferred_sources=decision.preferred_sources,
                        selected_scope=primary_scope,
                        fallback_used=False,
                        forced=False,
                        attempt_count=attempts,
                        sufficient_preferred_chunks=sufficient_count,
                    ),
                    metrics=metrics,
                )

            raise RAGRetrievalError(
                "La source prioritaire est insuffisante et le fallback "
                "documentaire est désactivé."
            ) from primary_error

        fallback_sources = self._available_fallback_sources(
            config.default_priority
        )
        fallback_scopes = (
            [decision.preferred_sources, fallback_sources]
            if len(decision.preferred_sources) > 1
            else [fallback_sources]
        )
        unique_scopes: list[tuple[str, ...]] = []

        for scope in fallback_scopes:
            if scope not in unique_scopes and scope != primary_scope:
                unique_scopes.append(scope)

        last_error = primary_error

        for scope in unique_scopes:
            attempts += 1

            try:
                fallback_retrieved = self._retrieve(
                    query,
                    metrics=metrics,
                    document_ids=self._document_ids(scope),
                )
            except RAGRetrievalError as error:
                last_error = error
                continue

            application = AppliedSourcePolicy(
                route=decision.route,
                mode=decision.mode,
                primary_source=decision.primary_source,
                preferred_sources=decision.preferred_sources,
                selected_scope=scope,
                fallback_used=True,
                forced=False,
                attempt_count=attempts,
                sufficient_preferred_chunks=sufficient_count,
            )
            return self._attach_source_policy(
                fallback_retrieved,
                application,
                metrics=metrics,
            )

        raise RAGRetrievalError(
            "La politique documentaire n'a pas trouvé cinq chunks "
            "exploitables, y compris avec fallback."
        ) from last_error

    def _decide_source_policy(
        self,
        question: str,
        *,
        mode: str,
    ) -> SourcePolicyDecision:
        """Apply static routes plus deterministic active-document mentions."""

        config = self.runtime_config.source_policy

        if mode.strip().casefold() == "automatic" and config.enabled:
            explicit_source = detect_explicit_active_source(
                question,
                self._active_source_filenames(),
            )

            if explicit_source is not None:
                return SourcePolicyDecision(
                    route="explicit_document",
                    mode="automatic",
                    preferred_sources=(explicit_source,),
                    primary_source=explicit_source,
                    forced=True,
                    allow_fallback=False,
                )

        return decide_source_policy(
            question,
            config=config,
            mode=mode,
        )

    def _active_source_filenames(self) -> tuple[str, ...]:
        active = self.active_knowledge_base

        if active is None:
            return ()

        return tuple(
            str(document["filename"])
            for document in active.documents
            if isinstance(document.get("filename"), str)
        )

    @staticmethod
    def _document_ids(sources: Sequence[str]) -> set[str]:
        """Build exact indexed document IDs from configured filenames."""

        return {
            document_id_from_source(source)
            for source in sources
        }

    def _available_fallback_sources(
        self,
        configured_priority: Sequence[str],
    ) -> tuple[str, ...]:
        """Append active user-managed PDFs after configured priorities."""

        available = list(self._active_source_filenames())
        return tuple(
            dict.fromkeys(
                [
                    *configured_priority,
                    *available,
                ]
            )
        )

    @staticmethod
    def _count_sufficient_preferred_chunks(
        retrieved: _RetrievedContext,
        *,
        query: str,
        preferred_sources: Sequence[str],
    ) -> int:
        """Count strong preferred passages using only runtime retrieval data."""

        allowed_documents = {
            document_id_from_source(source)
            for source in preferred_sources
        }
        candidate_by_id = {
            candidate.chunk.chunk_id: candidate
            for candidate in retrieved.candidates
        }
        terms = query_terms(query)
        count = 0

        for selection in retrieved.selected:
            candidate = candidate_by_id[selection.chunk_id]
            chunk = candidate.chunk

            if chunk.document_id not in allowed_documents:
                continue

            matched_by_both = (
                candidate.dense_rank is not None
                and candidate.bm25_rank is not None
            )
            passage_terms = query_terms(
                " ".join(
                    [
                        " ".join(chunk.heading_path),
                        chunk.embedding_text,
                    ]
                )
            )
            lexical_support = bool(terms & passage_terms)

            if matched_by_both or lexical_support:
                count += 1

        return count

    @staticmethod
    def _attach_source_policy(
        retrieved: _RetrievedContext,
        application: AppliedSourcePolicy,
        *,
        metrics: RAGLatencyMetrics | None,
    ) -> _RetrievedContext:
        """Attach, log and expose one content-safe policy outcome."""

        retrieved.source_policy = application

        if metrics is not None:
            metrics.source_policy_route = application.route
            metrics.source_policy_mode = application.mode
            metrics.source_policy_primary = application.primary_label
            metrics.source_policy_fallback_used = (
                application.fallback_used
            )
            metrics.source_policy_attempt_count = (
                application.attempt_count
            )
            metrics.source_policy_sufficient_preferred_chunks = (
                application.sufficient_preferred_chunks
            )

        LOGGER.info(
            "RAG_SOURCE_POLICY route=%s mode=%s primary=%s "
            "fallback=%s attempts=%d selected_scope=%s",
            application.route,
            application.mode,
            application.primary_label,
            application.fallback_used,
            application.attempt_count,
            ",".join(application.selected_scope),
        )
        return retrieved

    def _retrieve(
        self,
        query: str,
        *,
        metrics: RAGLatencyMetrics | None = None,
        document_ids: set[str] | None = None,
    ) -> _RetrievedContext:
        """Run exact frozen hybrid → reranker → safeguard sequence."""

        self._active_metrics = metrics
        embedding_before = (
            metrics.embedding_ms
            if metrics is not None
            else 0.0
        )

        if metrics is not None:
            expansion_started = time.perf_counter()
            hybrid_config = getattr(self.retriever, "config", None)
            expansion_version = getattr(
                hybrid_config,
                "query_expansion_version",
                "phosphoric_v2",
            )
            expand_lexical_query(
                query,
                version=expansion_version,
            )
            metrics.query_expansion_ms += (
                time.perf_counter() - expansion_started
            ) * 1000.0

        try:
            hybrid_response = self.retriever.search(
                query,
                top_k=self.frozen_config.candidate_k,
                dense_candidate_k=self.frozen_config.dense_candidates,
                bm25_candidate_k=self.frozen_config.bm25_candidates,
                document_ids=document_ids,
                use_query_expansion=self.frozen_config.query_expansion,
            )
        except Exception as error:
            raise RAGRetrievalError(
                "La recherche hybride a échoué."
            ) from error

        if metrics is not None:
            dense_duration_ms = float(
                getattr(hybrid_response, "dense_duration_ms", 0.0)
            )
            bm25_duration_ms = float(
                getattr(hybrid_response, "bm25_duration_ms", 0.0)
            )
            hybrid_total_ms = float(
                getattr(hybrid_response, "total_duration_ms", 0.0)
            )
            attempt_embedding_ms = (
                metrics.embedding_ms - embedding_before
            )
            metrics.dense_search_ms += max(
                0.0,
                dense_duration_ms - attempt_embedding_ms,
            )
            metrics.bm25_search_ms += bm25_duration_ms
            metrics.hybrid_fusion_ms += max(
                0.0,
                hybrid_total_ms
                - dense_duration_ms
                - bm25_duration_ms,
            )

        phase_started = time.perf_counter()
        candidates = list(hybrid_response.results)
        candidate_ids = [
            result.chunk.chunk_id
            for result in candidates
        ]

        if len(candidates) != self.frozen_config.candidate_k:
            raise RAGRetrievalError(
                "Le retriever doit conserver exactement 20 candidats."
            )

        if len(candidate_ids) != len(set(candidate_ids)):
            raise RAGRetrievalError(
                "Le retriever a retourné des chunks dupliqués."
            )

        if metrics is not None:
            metrics.candidate_preparation_ms += (
                time.perf_counter() - phase_started
            ) * 1000.0

        phase_started = time.perf_counter()
        tokenization_before = (
            metrics.reranker_tokenization_ms
            if metrics is not None
            else 0.0
        )

        try:
            reranked_response = self.reranker.rerank(
                query,
                candidates,
                top_k=self.frozen_config.candidate_k,
            )
        except Exception as error:
            raise RAGRetrievalError(
                "Le reranking BGE a échoué."
            ) from error

        reranking_total_ms = (
            time.perf_counter() - phase_started
        ) * 1000.0

        if metrics is not None:
            metrics.reranking_ms += reranking_total_ms
            reranker_internal_ms = float(
                getattr(
                    reranked_response,
                    "reranking_duration_ms",
                    reranking_total_ms,
                )
            )
            attempt_tokenization_ms = (
                metrics.reranker_tokenization_ms
                - tokenization_before
            )
            metrics.reranker_scoring_ms += max(
                0.0,
                reranker_internal_ms
                - attempt_tokenization_ms,
            )

        reranked_results = list(reranked_response.results)

        if len(reranked_results) != len(candidates):
            raise RAGRetrievalError(
                "Le reranker doit conserver les 20 candidats."
            )

        phase_started = time.perf_counter()

        try:
            selected = select_with_lexical_safeguard(
                candidates,
                reranked_results,
                top_k=self.frozen_config.top_k,
                lexical_slots=self.frozen_config.lexical_slots,
            )
        except ValueError as error:
            raise RAGRetrievalError(
                "La sélection lexical_safeguard_001 a échoué."
            ) from error

        if metrics is not None:
            metrics.lexical_selection_ms += (
                time.perf_counter() - phase_started
            ) * 1000.0

        if (
            len(selected) != self.frozen_config.top_k
            or len({result.chunk_id for result in selected})
            != self.frozen_config.top_k
        ):
            raise RAGRetrievalError(
                "La sélection finale doit contenir cinq chunks uniques."
            )

        phase_started = time.perf_counter()
        sources, source_texts = self._build_sources(
            candidates=candidates,
            reranked_results=reranked_results,
            selected=selected,
        )

        if metrics is not None:
            metrics.source_loading_ms += (
                time.perf_counter() - phase_started
            ) * 1000.0

        return _RetrievedContext(
            candidates=candidates,
            selected=selected,
            sources=sources,
            source_texts=source_texts,
            hybrid_response=hybrid_response,
            reranked_response=reranked_response,
        )

    def _prepare_context(
        self,
        source_texts: Sequence[str],
        question: str,
    ) -> PreparedDocumentContext:
        """Select bounded relevant excerpts for all five sources."""

        config = self.runtime_config.generation
        return prepare_document_context(
            source_texts,
            question,
            maximum_tokens_per_source=(
                config.max_context_tokens_per_source
            ),
            maximum_total_tokens=(
                config.max_total_document_context_tokens
            ),
        )

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
            + sum(
                len(message.content)
                for message in memory.messages()
            )
            + 500
        )
        metrics.baseline_equivalent_prompt_characters = (
            baseline_characters
        )
        metrics.baseline_equivalent_prompt_tokens = estimate_tokens(
            "x" * baseline_characters
        )

    def _generate_json_answer(
        self,
        *,
        user_prompt: str,
        available_source_count: int,
        system_prompt: str = SYSTEM_PROMPT,
        repair_system_prompt: str = REPAIR_SYSTEM_PROMPT,
        evidence_bundles: Sequence[EvidenceBundle] | None = None,
        question_type: str | None = None,
        response_language: str | None = None,
        comparison_subjects: tuple[str, ...] = (),
        contract_question: str = "",
        balance_kind: str | None = None,
    ) -> tuple[GroundedAnswerPayload, list[int], bool]:
        """Generate answer-only JSON, with at most one repair."""

        prompt = user_prompt
        active_system_prompt = system_prompt

        def apply_contract(
            payload: GroundedAnswerPayload,
            citations: list[int],
            insufficient: bool,
        ) -> tuple[GroundedAnswerPayload, list[int], bool]:
            if evidence_bundles is None:
                return payload, citations, insufficient

            contract_language = response_language or (
                "en"
                if re.search(
                    r"\b(?:the|is|are|does|cannot|corpus)\b",
                    payload.answer.casefold(),
                )
                else "fr"
            )
            contract = enforce_answer_contract(
                payload.answer,
                list(evidence_bundles),
                question_type=question_type,
                language=contract_language,
                comparison_subjects=comparison_subjects,
                question=contract_question,
                balance_kind=balance_kind,
            )
            LOGGER.info(
                "RAG answer contract type=%s changed=%s fallback=%s "
                "missing_roles=%s removed_claims=%d atomic_plan=%s",
                question_type,
                contract.changed,
                contract.fallback_used,
                contract.missing_roles,
                len(contract.removed_claims),
                contract.atomic_plan_used,
            )
            normalized_payload = payload.model_copy(
                update={"answer": contract.answer}
            )
            if contract.fallback_used:
                return normalized_payload, [], True
            if contract.changed or insufficient:
                try:
                    normalized_citations, normalized_insufficient = (
                        self._validate_answer(
                            answer=contract.answer,
                            available_source_count=available_source_count,
                            attempt="answer_contract",
                            evidence_bundles=evidence_bundles,
                        )
                    )
                except CitationValidationError as contract_error:
                    LOGGER.warning(
                        "RAG deterministic answer builder fallback reason=%s",
                        contract_error,
                    )
                    fallback = _controlled_fallback_for_language(
                        contract_language
                    )
                    return (
                        normalized_payload.model_copy(
                            update={"answer": fallback}
                        ),
                        [],
                        True,
                    )
                return (
                    normalized_payload,
                    normalized_citations,
                    normalized_insufficient,
                )
            return normalized_payload, citations, insufficient

        for attempt_index, attempt in enumerate(("initial", "repair")):
            raw_output = ""

            try:
                payload, raw_output = self.llm.chat_json_with_raw(
                    user_prompt=prompt,
                    system_prompt=active_system_prompt,
                    response_model=GroundedAnswerPayload,
                )
            except OllamaResponseValidationError as error:
                raw_output = error.raw_response or ""
                rejection: Exception = error
            except OllamaError as error:
                raise RAGGenerationError(str(error)) from error
            except Exception as error:
                raise RAGGenerationError(
                    "La génération locale Qwen a échoué."
                ) from error
            else:
                try:
                    citations, insufficient = self._validate_answer(
                        answer=payload.answer,
                        available_source_count=available_source_count,
                        attempt=attempt,
                        evidence_bundles=evidence_bundles,
                    )
                except CitationValidationError as error:
                    if evidence_bundles is None:
                        rejection = error
                    else:
                        fallback_language = (
                            "en"
                            if re.search(
                                r"\b(?:the|is|are|does|cannot|corpus)\b",
                                payload.answer.casefold(),
                            )
                            else "fr"
                        )
                        pruned = prune_unsupported_claims(
                            payload.answer,
                            list(evidence_bundles),
                            fallback_language=fallback_language,
                            question_type=question_type,
                        )
                        payload = payload.model_copy(
                            update={"answer": pruned.answer}
                        )

                        if pruned.fallback_used:
                            return apply_contract(payload, [], True)

                        try:
                            citations, insufficient = self._validate_answer(
                                answer=payload.answer,
                                available_source_count=available_source_count,
                                attempt="deterministic_pruning",
                                evidence_bundles=evidence_bundles,
                            )
                        except CitationValidationError as pruning_error:
                            LOGGER.warning(
                                "RAG deterministic pruning fallback "
                                "reason=%s",
                                pruning_error,
                            )
                            fallback = prune_unsupported_claims(
                                "",
                                list(evidence_bundles),
                                fallback_language=fallback_language,
                            )
                            payload = payload.model_copy(
                                update={"answer": fallback.answer}
                            )
                            return apply_contract(payload, [], True)

                        LOGGER.info(
                            "RAG deterministic pruning removed_claims=%d "
                            "inherited_citations=%d fallback=%s "
                            "missing_required=%s atomic_plan=%s "
                            "reconstructed_claims=%d",
                            len(pruned.removed_claims),
                            pruned.inherited_citation_count,
                            pruned.fallback_used,
                            pruned.missing_required_concepts,
                            pruned.atomic_plan_used,
                            pruned.reconstructed_claim_count,
                        )
                        return apply_contract(
                            payload,
                            citations,
                            insufficient,
                        )
                else:
                    return apply_contract(
                        payload,
                        citations,
                        insufficient,
                    )

            LOGGER.warning(
                "RAG output rejected attempt=%s reason=%s "
                "available_sources=%d",
                attempt,
                rejection,
                available_source_count,
            )

            if attempt_index == 1:
                LOGGER.error(
                    "RAG output invalid after repair reason=%s raw_output=%r",
                    rejection,
                    raw_output,
                )
                raise RAGResponseValidationError(
                    "Réponse Qwen invalide après une réparation : "
                    f"{rejection}"
                ) from rejection

            prompt = build_repair_prompt(
                original_prompt=user_prompt,
                invalid_output=raw_output,
                rejection_reason=str(rejection),
                json_output=True,
            )
            active_system_prompt = repair_system_prompt

        raise AssertionError("La boucle de réparation aurait dû se terminer.")

    def _validate_answer(
        self,
        *,
        answer: str,
        available_source_count: int,
        attempt: str,
        evidence_bundles: Sequence[EvidenceBundle] | None = None,
    ) -> tuple[list[int], bool]:
        """Validate citations with answer as the sole source of truth."""

        citations = extract_citations(
            answer,
            available_source_count=available_source_count,
        )
        insufficient = is_controlled_insufficient_answer(answer)

        if not citations and not insufficient:
            raise CitationValidationError(
                "Une réponse affirmative exige une citation [Source N]."
            )

        if evidence_bundles is not None and not insufficient:
            validate_claim_support(answer, list(evidence_bundles))

        LOGGER.info(
            "RAG citations validated attempt=%s citations=%s "
            "available_sources=%d",
            attempt,
            citations,
            available_source_count,
        )
        return citations, insufficient

    def _validate_answer_with_metrics(
        self,
        *,
        answer: str,
        available_source_count: int,
        attempt: str,
        metrics: RAGLatencyMetrics,
        evidence_bundles: Sequence[EvidenceBundle] | None = None,
    ) -> tuple[list[int], bool]:
        """Measure citation extraction separately from policy validation."""

        phase_started = time.perf_counter()
        citations = extract_citations(
            answer,
            available_source_count=available_source_count,
        )
        metrics.citation_extraction_ms += (
            time.perf_counter() - phase_started
        ) * 1000.0
        phase_started = time.perf_counter()
        insufficient = is_controlled_insufficient_answer(answer)

        if not citations and not insufficient:
            metrics.citation_validation_ms += (
                time.perf_counter() - phase_started
            ) * 1000.0
            raise CitationValidationError(
                "Une réponse affirmative exige une citation [Source N]."
            )

        if evidence_bundles is not None and not insufficient:
            validate_claim_support(answer, list(evidence_bundles))

        metrics.citation_validation_ms += (
            time.perf_counter() - phase_started
        ) * 1000.0
        LOGGER.info(
            "RAG citations validated attempt=%s citations=%s "
            "available_sources=%d",
            attempt,
            citations,
            available_source_count,
        )
        return citations, insufficient

    @staticmethod
    def _log_validation_rejection(
        *,
        attempt: str,
        error: CitationValidationError,
        available_source_count: int,
        raw_output: str,
        final: bool,
    ) -> None:
        """Log safe diagnostics; raw output appears only after final failure."""

        LOGGER.warning(
            "RAG citations rejected attempt=%s detected_citations=%s "
            "available_sources=%d reason=%s",
            attempt,
            error.detected_citations,
            available_source_count,
            error,
        )

        if final:
            LOGGER.error(
                "RAG streaming invalid after repair raw_output=%r",
                raw_output,
            )

    @staticmethod
    def _comparison_subjects(
        quality_result: QualityRetrievalResult | None,
    ) -> tuple[str, ...]:
        """Return explicit A/B subjects from the retrieval plan."""

        if quality_result is None or quality_result.retrieval_plan is None:
            return ()
        return tuple(
            role.subject
            for role in quality_result.retrieval_plan.roles
            if role.name in {"equipment_a", "equipment_b"}
            and role.subject
        )

    @staticmethod
    def _cited_sources(
        sources: Sequence[RAGSource],
        cited_source_numbers: Sequence[int],
    ) -> list[RAGSource]:
        """Return only cited source objects in first-appearance order."""

        source_by_number = {
            source.source_number: source
            for source in sources
        }
        return [
            source_by_number[source_number]
            for source_number in cited_source_numbers
        ]

    def _build_response(
        self,
        *,
        question: str,
        answer: str,
        cited_sources: list[RAGSource],
        cited_source_numbers: list[int],
        insufficient_context: bool,
        retrieved: _RetrievedContext,
        generation_ms: float,
        total_ms: float,
        first_token_ms: float | None = None,
        latency: RAGLatencyMetrics | None = None,
    ) -> RAGResponse:
        """Build the public response for blocking or streaming calls."""

        model_name = getattr(
            self.llm,
            "model_name",
            self.runtime_config.ollama.model,
        )
        source_policy = retrieved.source_policy
        return RAGResponse(
            question=question,
            answer=answer,
            sources=cited_sources,
            cited_source_numbers=cited_source_numbers,
            insufficient_context=insufficient_context,
            model_name=str(model_name),
            selected_variant=self.frozen_config.selected_variant,
            snapshot_sha256=self.frozen_config.snapshot_sha256,
            candidate_count=len(retrieved.candidates),
            selected_count=len(retrieved.selected),
            source_policy_route=(
                source_policy.route
                if source_policy is not None
                else "disabled"
            ),
            source_policy_mode=(
                source_policy.mode
                if source_policy is not None
                else "automatic"
            ),
            source_policy_primary=(
                source_policy.primary_label
                if source_policy is not None
                else None
            ),
            source_policy_fallback_used=(
                source_policy.fallback_used
                if source_policy is not None
                else False
            ),
            source_policy_forced=(
                source_policy.forced
                if source_policy is not None
                else False
            ),
            response_language=retrieved.response_language,
            standalone_query=(
                retrieved.quality_result.query.standalone_query
                if retrieved.quality_result is not None
                else None
            ),
            question_type=retrieved.question_type,
            detected_domains=(
                [
                    domain.value
                    for domain, _confidence in (
                        retrieved.quality_result.routing.detected_domains
                    )
                ]
                if retrieved.quality_result is not None
                else []
            ),
            timings=RAGTimings(
                hybrid_ms=float(
                    retrieved.hybrid_response.total_duration_ms
                ),
                reranking_ms=float(
                    retrieved.reranked_response.reranking_duration_ms
                ),
                generation_ms=generation_ms,
                total_ms=total_ms,
                first_token_ms=first_token_ms,
            ),
            latency=(
                latency.to_dict()
                if latency is not None
                else {}
            ),
        )

    def _build_sources(
        self,
        *,
        candidates: list[Any],
        reranked_results: list[Any],
        selected: list[Any],
    ) -> tuple[list[RAGSource], list[str]]:
        """Rehydrate full text and provenance for five selections."""

        candidate_by_id = {
            result.chunk.chunk_id: result
            for result in candidates
        }
        reranked_by_id = {
            result.chunk.chunk_id: result
            for result in reranked_results
        }
        sources: list[RAGSource] = []
        full_texts: list[str] = []

        for source_number, selection in enumerate(selected, start=1):
            candidate = candidate_by_id[selection.chunk_id]
            reranked = reranked_by_id.get(selection.chunk_id)
            chunk = candidate.chunk
            full_text = clean_passage_text(chunk.text)

            if not full_text:
                raise RAGRetrievalError(
                    f"Le chunk {chunk.chunk_id} possède un texte vide."
                )

            section = (
                " > ".join(
                    heading
                    for heading in chunk.heading_path
                    if heading
                )
                or None
            )
            sources.append(
                RAGSource(
                    source_number=source_number,
                    chunk_id=chunk.chunk_id,
                    document_name=Path(chunk.source_file).name,
                    pages=list(chunk.source_pages),
                    section=section,
                    excerpt=self._excerpt(full_text),
                    selection_source=selection.source,
                    hybrid_rank=candidate.rank,
                    rrf_score=float(candidate.rrf_score),
                    dense_rank=candidate.dense_rank,
                    dense_score=candidate.dense_score,
                    bm25_rank=candidate.bm25_rank,
                    bm25_score=candidate.bm25_score,
                    reranker_rank=(
                        reranked.rank
                        if reranked is not None
                        else None
                    ),
                    reranker_score=(
                        float(reranked.reranker_score)
                        if reranked is not None
                        else None
                    ),
                )
            )
            full_texts.append(full_text)

        return sources, full_texts

    def _excerpt(self, text: str) -> str:
        """Create a bounded source display excerpt."""

        limit = self.runtime_config.source_excerpt_characters

        if len(text) <= limit:
            return text

        return text[: limit - 1].rstrip() + "…"

    def close(self) -> None:
        """Release the persistent Ollama HTTP client."""

        close = getattr(self.llm, "close", None)

        if callable(close):
            close()
