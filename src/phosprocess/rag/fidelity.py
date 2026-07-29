"""Backward-compatible façade for RAG fidelity services.

Implementation lives in four focused modules:
- claim_support
- citation_binding
- deterministic_builders
- answer_contracts
"""

from phosprocess.rag.answer_contracts import (
    AnswerContractResult,
    enforce_answer_contract,
)
from phosprocess.rag.citation_binding import (
    PrunedAnswer,
    build_atomic_process_flow_answer,
    prune_unsupported_claims,
)
from phosprocess.rag.claim_support import (
    ClaimSupport,
    ClaimSupportStatus,
    evaluate_claim_support,
    validate_claim_support,
)
from phosprocess.rag.deterministic_builders import (
    build_deterministic_balance_answer,
    build_deterministic_definition_answer,
    build_deterministic_fouling_answer,
    build_deterministic_momentum_diffusion_answer,
    build_deterministic_scoped_explanation,
)

__all__ = [
    "AnswerContractResult",
    "ClaimSupport",
    "ClaimSupportStatus",
    "PrunedAnswer",
    "build_atomic_process_flow_answer",
    "build_deterministic_balance_answer",
    "build_deterministic_definition_answer",
    "build_deterministic_fouling_answer",
    "build_deterministic_momentum_diffusion_answer",
    "build_deterministic_scoped_explanation",
    "enforce_answer_contract",
    "evaluate_claim_support",
    "prune_unsupported_claims",
    "validate_claim_support",
]
