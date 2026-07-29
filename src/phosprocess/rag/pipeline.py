"""Backward-compatible façade for the refactored RAG runtime."""

from phosprocess.rag.orchestrator import (
    DEFAULT_EMBEDDING_CONFIG_PATH,
    DEFAULT_RUNTIME_CONFIG_PATH,
    DEFAULT_SNAPSHOT_DIRECTORY,
    EXPECTED_SELECTED_VARIANT,
    ConversationRuntimeConfig,
    FrozenV3Config,
    GenerationRuntimeConfig,
    PhosProcessRAG,
    RAGRuntimeConfig,
    WarmupRuntimeConfig,
    load_frozen_v3_config,
    load_runtime_config,
    sha256_file,
    validate_question,
)
from phosprocess.rag.retrieval_service import (
    RAGConfigurationError,
    RAGError,
    RAGGenerationError,
    RAGResponseValidationError,
    RAGRetrievalError,
)

__all__ = [
    "DEFAULT_EMBEDDING_CONFIG_PATH",
    "DEFAULT_RUNTIME_CONFIG_PATH",
    "DEFAULT_SNAPSHOT_DIRECTORY",
    "EXPECTED_SELECTED_VARIANT",
    "ConversationRuntimeConfig",
    "FrozenV3Config",
    "GenerationRuntimeConfig",
    "PhosProcessRAG",
    "RAGConfigurationError",
    "RAGError",
    "RAGGenerationError",
    "RAGResponseValidationError",
    "RAGRetrievalError",
    "RAGRuntimeConfig",
    "WarmupRuntimeConfig",
    "load_frozen_v3_config",
    "load_runtime_config",
    "sha256_file",
    "validate_question",
]
