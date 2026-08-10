"""Small configurable source and chunk-type boosts after global retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

from phosprocess.ingestion.chunk_serialization import TechnicalChunkType
from phosprocess.retrieval.domain_router import DomainRoutingDecision


class BoostableResult(Protocol):
    """Minimum fields required for deterministic score adjustment."""

    document_id: str
    chunk_type: TechnicalChunkType


T = TypeVar("T", bound=BoostableResult)


@dataclass(frozen=True, slots=True)
class BoostedResult:
    """Original result plus transparent application-layer scoring."""

    result: BoostableResult
    base_score: float
    source_boost: float
    chunk_type_boost: float
    final_score: float


DEFAULT_CHUNK_TYPE_BOOSTS: dict[TechnicalChunkType, float] = {
    TechnicalChunkType.DEFINITION: 0.01,
    TechnicalChunkType.PROCESS_DESCRIPTION: 0.012,
    TechnicalChunkType.EQUIPMENT_DESCRIPTION: 0.012,
    TechnicalChunkType.PROCEDURE: 0.012,
    TechnicalChunkType.EQUATION: 0.01,
    TechnicalChunkType.EQUATION_EXPLANATION: 0.01,
    TechnicalChunkType.TABLE: 0.006,
    TechnicalChunkType.CONTROL_STRATEGY: 0.01,
    TechnicalChunkType.BALANCE: 0.012,
    TechnicalChunkType.OPERATING_PROBLEM: 0.012,
    TechnicalChunkType.SIMULATION_RESULTS: -0.02,
    TechnicalChunkType.ABBREVIATIONS: -0.08,
    TechnicalChunkType.EXERCISE: -0.015,
}


def apply_soft_boosts(
    results: list[T],
    scores: list[float],
    *,
    routing: DomainRoutingDecision,
    chunk_type_boosts: dict[TechnicalChunkType, float] | None = None,
) -> list[BoostedResult]:
    """Boost but never exclude an automatic non-preferred source."""

    if len(results) != len(scores):
        raise ValueError("Un score est requis pour chaque résultat.")

    boosts = chunk_type_boosts or DEFAULT_CHUNK_TYPE_BOOSTS
    adjusted: list[BoostedResult] = []

    for result, base_score in zip(results, scores, strict=True):
        if routing.hard_filter is not None and result.document_id not in routing.hard_filter:
            continue

        source_boost = routing.soft_boosts.get(result.document_id, 0.0)
        type_boost = boosts.get(result.chunk_type, 0.0)
        adjusted.append(
            BoostedResult(
                result=result,
                base_score=float(base_score),
                source_boost=source_boost,
                chunk_type_boost=type_boost,
                final_score=float(base_score) + source_boost + type_boost,
            )
        )

    return sorted(
        adjusted,
        key=lambda item: (
            -item.final_score,
            -item.base_score,
            item.result.document_id,
        ),
    )
