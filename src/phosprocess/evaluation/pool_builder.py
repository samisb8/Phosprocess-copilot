"""Construction reproductible du pool d'annotation retrieval."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from phosprocess.evaluation.schemas import (
    EvaluationConfig,
    EvaluationQuery,
)
from phosprocess.preprocessing.chunk_schemas import DocumentChunk
from phosprocess.reranking.reranker import (
    BGEReranker,
    RerankingConfig,
)
from phosprocess.retrieval.hybrid import (
    HybridRetriever,
    expand_lexical_query,
)

SYSTEM_ORDER = (
    "dense",
    "bm25",
    "hybrid",
    "reranker",
)

CHECKPOINT_FILENAME = "annotation_pool.checkpoint.jsonl"
POOL_FILENAME = "annotation_pool.jsonl"
SUMMARY_FILENAME = "annotation_pool_summary.json"


class SystemEvidence(BaseModel):
    """Présence d'un chunk dans un système de recherche."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(gt=0)
    score: float


class AnnotationPoolItem(BaseModel):
    """Paire question-chunk à juger manuellement."""

    model_config = ConfigDict(extra="forbid")

    pool_item_id: str

    query_id: str
    question: str
    language: str
    category: str
    difficulty: str
    split: str
    question_family_id: str

    answerable: bool
    expected_answer: str | None
    query_notes: str
    reference_documents: list[str]

    chunk_id: str
    document_id: str
    source_file: str
    chunk_index: int

    source_pages: list[int]
    page_start: int
    page_end: int

    heading_path: list[str]
    content_types: list[str]
    text: str

    systems: dict[str, SystemEvidence]
    retrieved_by: list[str]
    best_rank: int

    # Ordre aveuglé et déterministe pour l'annotation.
    display_order: int

    # Champs remplis à l'étape d'annotation.
    relevance: int | None = Field(
        default=None,
        ge=0,
        le=3,
    )
    rationale: str | None = None
    assessor_id: str | None = None
    annotation_status: str = "unjudged"


class QueryPoolCheckpoint(BaseModel):
    """Résultat complet et reprenable pour une question."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    query_sha256: str
    build_signature: str
    completed_at_utc: datetime

    lexical_query: str
    timings_ms: dict[str, float]

    pool_size: int
    items: list[AnnotationPoolItem]


def load_evaluation_queries(
    path: Path,
) -> list[EvaluationQuery]:
    """Charger les questions validées du benchmark."""

    if not path.exists():
        raise FileNotFoundError(f"Questions introuvables : {path}")

    queries: list[EvaluationQuery] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as source:
        for line_number, line in enumerate(
            source,
            start=1,
        ):
            if not line.strip():
                raise ValueError(f"{path.name}, ligne {line_number} vide.")

            try:
                query = EvaluationQuery.model_validate_json(line)
            except Exception as error:
                raise ValueError(f"{path.name}, ligne {line_number} invalide.") from error

            queries.append(query)

    query_ids = [query.query_id for query in queries]

    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Des query_id sont dupliqués.")

    return queries


def sha256_file(path: Path) -> str:
    """Calculer l'empreinte SHA-256 d'un fichier."""

    if not path.exists():
        raise FileNotFoundError(f"Artefact de signature introuvable : {path}")

    digest = hashlib.sha256()

    with path.open("rb") as source:
        for block in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def query_sha256(
    query: EvaluationQuery,
) -> str:
    """Calculer l'empreinte d'une question."""

    payload = json.dumps(
        query.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    return hashlib.sha256(payload).hexdigest()


def build_artifact_signature(
    paths: list[Path],
) -> str:
    """Signer les questions, configs et index utilisés."""

    digest = hashlib.sha256()

    for path in sorted(
        (path.resolve() for path in paths),
        key=str,
    ):
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\0")

    return digest.hexdigest()


def atomic_write_json(
    data: dict[str, Any],
    path: Path,
) -> None:
    """Écrire un objet JSON atomiquement."""

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    temporary_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(path)


def atomic_write_jsonl(
    records: list[dict[str, Any]],
    path: Path,
) -> None:
    """Écrire un fichier JSONL atomiquement."""

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as output:
        for record in records:
            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    temporary_path.replace(path)


def percentile(
    values: list[float],
    fraction: float,
) -> float:
    """Calculer un percentile avec interpolation linéaire."""

    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = fraction * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(
        lower_index + 1,
        len(ordered) - 1,
    )

    weight = position - lower_index

    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


class EvaluationPoolBuilder:
    """Exécuter les moteurs et construire le pool dédupliqué."""

    def __init__(
        self,
        *,
        hybrid_retriever: HybridRetriever,
        reranker: BGEReranker,
        evaluation_config: EvaluationConfig,
        reranking_config: RerankingConfig,
        output_directory: Path,
        build_signature: str,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker

        self.evaluation_config = evaluation_config
        self.reranking_config = reranking_config

        self.output_directory = output_directory.resolve()
        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.build_signature = build_signature

        self.checkpoint_path = self.output_directory / CHECKPOINT_FILENAME

        self.pool_path = self.output_directory / POOL_FILENAME

        self.summary_path = self.output_directory / SUMMARY_FILENAME

    def build(
        self,
        *,
        all_queries: list[EvaluationQuery],
        selected_queries: list[EvaluationQuery],
        force: bool = False,
    ) -> dict[str, Any]:
        """Construire ou reprendre le pool."""

        if force:
            self._remove_generated_artifacts()

        query_map = {query.query_id: query for query in all_queries}

        completed = self._load_checkpoints(query_map)

        selected_ids = {query.query_id for query in selected_queries}

        pending_queries = [query for query in selected_queries if query.query_id not in completed]

        print("\n=== Construction du pool d'annotation ===")
        print(f"Questions du benchmark : {len(all_queries)}")
        print(f"Questions sélectionnées: {len(selected_queries)}")
        print(f"Déjà terminées         : {len(completed)}")
        print(f"À traiter maintenant   : {len(pending_queries)}")

        for position, query in enumerate(
            pending_queries,
            start=1,
        ):
            print(f"\n[{position}/{len(pending_queries)}] {query.query_id} — {query.question}")

            checkpoint = self._process_query(query)

            self._append_checkpoint(checkpoint)

            completed[query.query_id] = checkpoint

            print(
                f"[OK] {query.query_id} : "
                f"{checkpoint.pool_size} chunks uniques | "
                f"{checkpoint.timings_ms['total']:.1f} ms"
            )

        unexpected_completed = set(completed) - set(query_map)

        if unexpected_completed:
            raise ValueError(
                "Le checkpoint contient des questions "
                "absentes du benchmark : "
                f"{sorted(unexpected_completed)}"
            )

        # selected_ids est utilisé pour détecter une sélection vide.
        if not selected_ids:
            raise ValueError("Aucune question sélectionnée.")

        return self._publish(
            all_queries=all_queries,
            checkpoints=completed,
        )

    def _process_query(
        self,
        query: EvaluationQuery,
    ) -> QueryPoolCheckpoint:
        """Exécuter les quatre systèmes pour une question."""

        pooling = self.evaluation_config.pooling

        total_start = time.perf_counter()

        dense_response = self.hybrid_retriever.dense_retriever.search(
            query.question,
            top_k=pooling.dense_depth,
        )

        expansion_enabled = self.hybrid_retriever.config.lexical_query_expansion

        lexical_query = (
            expand_lexical_query(
                query.question,
                version=(self.hybrid_retriever.config.query_expansion_version),
            )
            if expansion_enabled
            else query.question
        )

        bm25_response = self.hybrid_retriever.bm25_retriever.search(
            lexical_query,
            top_k=pooling.bm25_depth,
        )

        hybrid_candidate_k = max(
            pooling.hybrid_depth,
            pooling.reranker_depth,
            self.reranking_config.hybrid_candidate_k,
        )

        hybrid_response = self.hybrid_retriever.search(
            query.question,
            top_k=hybrid_candidate_k,
        )

        reranking_response = self.reranker.rerank(
            query.question,
            hybrid_response.results,
            top_k=pooling.reranker_depth,
        )

        candidates: dict[
            str,
            dict[str, Any],
        ] = {}

        for result in dense_response.results:
            self._register_candidate(
                candidates,
                system="dense",
                rank=result.rank,
                score=result.score,
                chunk=result.chunk,
            )

        for result in bm25_response.results:
            self._register_candidate(
                candidates,
                system="bm25",
                rank=result.rank,
                score=result.score,
                chunk=result.chunk,
            )

        for result in hybrid_response.results[: pooling.hybrid_depth]:
            self._register_candidate(
                candidates,
                system="hybrid",
                rank=result.rank,
                score=result.rrf_score,
                chunk=result.chunk,
            )

        for result in reranking_response.results[: pooling.reranker_depth]:
            self._register_candidate(
                candidates,
                system="reranker",
                rank=result.rank,
                score=result.reranker_score,
                chunk=result.chunk,
            )

        items = self._build_pool_items(
            query=query,
            candidates=candidates,
        )

        total_duration_ms = (time.perf_counter() - total_start) * 1000

        return QueryPoolCheckpoint(
            query_id=query.query_id,
            query_sha256=query_sha256(query),
            build_signature=self.build_signature,
            completed_at_utc=datetime.now(UTC),
            lexical_query=lexical_query,
            timings_ms={
                "dense": (dense_response.search_duration_ms),
                "bm25": (bm25_response.search_duration_ms),
                "hybrid": (hybrid_response.total_duration_ms),
                "reranker": (reranking_response.reranking_duration_ms),
                "total": round(
                    total_duration_ms,
                    3,
                ),
            },
            pool_size=len(items),
            items=items,
        )

    @staticmethod
    def _register_candidate(
        candidates: dict[str, dict[str, Any]],
        *,
        system: str,
        rank: int,
        score: float,
        chunk: DocumentChunk,
    ) -> None:
        """Ajouter un résultat en le dédupliquant par chunk_id."""

        if system not in SYSTEM_ORDER:
            raise ValueError(f"Système inconnu : {system}")

        existing = candidates.get(chunk.chunk_id)

        if existing is None:
            existing = {
                "chunk": chunk,
                "systems": {},
            }

            candidates[chunk.chunk_id] = existing

        existing_chunk: DocumentChunk = existing["chunk"]

        if (
            existing_chunk.text != chunk.text
            or existing_chunk.source_file != chunk.source_file
            or existing_chunk.source_pages != chunk.source_pages
        ):
            raise ValueError(f"Métadonnées incohérentes pour {chunk.chunk_id}.")

        systems: dict[str, SystemEvidence] = existing["systems"]

        systems[system] = SystemEvidence(
            rank=rank,
            score=float(score),
        )

    def _build_pool_items(
        self,
        *,
        query: EvaluationQuery,
        candidates: dict[str, dict[str, Any]],
    ) -> list[AnnotationPoolItem]:
        """Transformer l'union en paires à annoter."""

        blinded_entries = sorted(
            candidates.items(),
            key=lambda entry: hashlib.sha256(
                (f"{self.evaluation_config.dataset.version}|{query.query_id}|{entry[0]}").encode()
            ).hexdigest(),
        )

        items: list[AnnotationPoolItem] = []

        for display_order, (
            chunk_id,
            payload,
        ) in enumerate(
            blinded_entries,
            start=1,
        ):
            chunk: DocumentChunk = payload["chunk"]

            systems: dict[str, SystemEvidence] = payload["systems"]

            retrieved_by = [system for system in SYSTEM_ORDER if system in systems]

            best_rank = min(evidence.rank for evidence in systems.values())

            content_types = [
                (value.value if hasattr(value, "value") else str(value))
                for value in chunk.content_types
            ]

            items.append(
                AnnotationPoolItem(
                    pool_item_id=(f"{query.query_id}::{chunk_id}"),
                    query_id=query.query_id,
                    question=query.question,
                    language=query.language.value,
                    category=query.category.value,
                    difficulty=query.difficulty.value,
                    split=query.split.value,
                    question_family_id=(query.question_family_id),
                    answerable=query.answerable,
                    expected_answer=(query.expected_answer),
                    query_notes=query.notes,
                    reference_documents=list(query.reference_documents),
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source_file=chunk.source_file,
                    chunk_index=chunk.chunk_index,
                    source_pages=list(chunk.source_pages),
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    heading_path=list(chunk.heading_path),
                    content_types=content_types,
                    text=chunk.text,
                    systems=systems,
                    retrieved_by=retrieved_by,
                    best_rank=best_rank,
                    display_order=display_order,
                )
            )

        return items

    def _append_checkpoint(
        self,
        checkpoint: QueryPoolCheckpoint,
    ) -> None:
        """Sauvegarder immédiatement une question terminée."""

        with self.checkpoint_path.open(
            "a",
            encoding="utf-8",
            newline="\n",
        ) as output:
            output.write(checkpoint.model_dump_json() + "\n")

            output.flush()
            os.fsync(output.fileno())

    def _load_checkpoints(
        self,
        query_map: dict[str, EvaluationQuery],
    ) -> dict[str, QueryPoolCheckpoint]:
        """Recharger les questions déjà terminées."""

        if not self.checkpoint_path.exists():
            return {}

        checkpoints: dict[
            str,
            QueryPoolCheckpoint,
        ] = {}

        with self.checkpoint_path.open(
            "r",
            encoding="utf-8",
        ) as source:
            for line_number, line in enumerate(
                source,
                start=1,
            ):
                if not line.strip():
                    raise ValueError(f"Checkpoint vide à la ligne {line_number}.")

                try:
                    checkpoint = QueryPoolCheckpoint.model_validate_json(line)
                except Exception as error:
                    raise ValueError(f"Checkpoint invalide à la ligne {line_number}.") from error

                if checkpoint.build_signature != self.build_signature:
                    raise ValueError(
                        "Le checkpoint a été construit avec "
                        "d'autres questions, configs ou index. "
                        "Relance avec --force."
                    )

                query = query_map.get(checkpoint.query_id)

                if query is None:
                    raise ValueError(f"Question de checkpoint inconnue : {checkpoint.query_id}.")

                if checkpoint.query_sha256 != query_sha256(query):
                    raise ValueError(
                        f"La question {checkpoint.query_id} a changé. Relance avec --force."
                    )

                if checkpoint.query_id in checkpoints:
                    raise ValueError(f"Checkpoint dupliqué pour {checkpoint.query_id}.")

                checkpoints[checkpoint.query_id] = checkpoint

        return checkpoints

    def _publish(
        self,
        *,
        all_queries: list[EvaluationQuery],
        checkpoints: dict[str, QueryPoolCheckpoint],
    ) -> dict[str, Any]:
        """Publier le pool aplati et son résumé."""

        all_items: list[AnnotationPoolItem] = []

        for query_id in sorted(checkpoints):
            checkpoint = checkpoints[query_id]

            all_items.extend(
                sorted(
                    checkpoint.items,
                    key=lambda item: item.display_order,
                )
            )

        atomic_write_jsonl(
            [item.model_dump(mode="json") for item in all_items],
            self.pool_path,
        )

        pool_sizes = [checkpoint.pool_size for checkpoint in checkpoints.values()]

        system_counts = {
            system: sum(system in item.systems for item in all_items) for system in SYSTEM_ORDER
        }

        overlap_counts: dict[str, int] = {}

        for item in all_items:
            key = str(len(item.retrieved_by))

            overlap_counts[key] = overlap_counts.get(key, 0) + 1

        timing_names = (
            "dense",
            "bm25",
            "hybrid",
            "reranker",
            "total",
        )

        timing_summary: dict[
            str,
            dict[str, float],
        ] = {}

        for timing_name in timing_names:
            values = [checkpoint.timings_ms[timing_name] for checkpoint in checkpoints.values()]

            timing_summary[timing_name] = {
                "mean": round(
                    statistics.fmean(values),
                    3,
                )
                if values
                else 0.0,
                "p50": round(
                    percentile(values, 0.50),
                    3,
                ),
                "p95": round(
                    percentile(values, 0.95),
                    3,
                ),
                "max": round(
                    max(values),
                    3,
                )
                if values
                else 0.0,
            }

        completed_ids = set(checkpoints)

        all_query_ids = {query.query_id for query in all_queries}

        missing_query_ids = sorted(all_query_ids - completed_ids)

        status = "complete" if not missing_query_ids else "partial"

        summary: dict[str, Any] = {
            "created_at_utc": (datetime.now(UTC).isoformat()),
            "status": status,
            "build_signature": (self.build_signature),
            "dataset": {
                "name": (self.evaluation_config.dataset.name),
                "version": (self.evaluation_config.dataset.version),
            },
            "queries": {
                "expected": len(all_queries),
                "completed": len(checkpoints),
                "missing": missing_query_ids,
            },
            "pool": {
                "items": len(all_items),
                "minimum_per_query": (min(pool_sizes) if pool_sizes else 0),
                "maximum_per_query": (max(pool_sizes) if pool_sizes else 0),
                "mean_per_query": round(
                    statistics.fmean(pool_sizes),
                    3,
                )
                if pool_sizes
                else 0.0,
                "median_per_query": round(
                    statistics.median(pool_sizes),
                    3,
                )
                if pool_sizes
                else 0.0,
            },
            "retrieved_item_counts": (system_counts),
            "systems_per_item": overlap_counts,
            "timings_ms": timing_summary,
            "files": {
                "checkpoint": (self.checkpoint_path.name),
                "pool": self.pool_path.name,
                "summary": (self.summary_path.name),
            },
        }

        atomic_write_json(
            summary,
            self.summary_path,
        )

        print("\n=== Pool publié ===")
        print(f"Statut           : {status}")
        print(f"Questions        : {len(checkpoints)}/{len(all_queries)}")
        print(f"Paires à juger  : {len(all_items)}")
        print(f"Pool            : {self.pool_path}")
        print(f"Résumé          : {self.summary_path}")
        print(f"Checkpoint      : {self.checkpoint_path}")

        if missing_query_ids:
            print(f"Questions restantes: {missing_query_ids}")

        return summary

    def _remove_generated_artifacts(
        self,
    ) -> None:
        """Supprimer uniquement les artefacts de pooling."""

        for path in (
            self.checkpoint_path,
            self.pool_path,
            self.summary_path,
        ):
            path.unlink(missing_ok=True)
