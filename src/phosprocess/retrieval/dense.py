"""Recherche sémantique dense avec BGE-M3 et FAISS."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from pydantic import ValidationError

from phosprocess.embeddings.embedder import (
    BGEEmbedder,
    load_embedding_config,
)
from phosprocess.preprocessing.chunk_schemas import DocumentChunk


@dataclass(frozen=True, slots=True)
class DenseSearchResult:
    """Résultat individuel d'une recherche dense."""

    rank: int
    vector_id: int
    score: float
    chunk: DocumentChunk


@dataclass(frozen=True, slots=True)
class DenseSearchResponse:
    """Réponse complète d'une recherche dense."""

    query: str
    top_k_requested: int
    search_duration_ms: float
    results: list[DenseSearchResult]


class DenseRetriever:
    """Charger BGE-M3 et un index FAISS pour rechercher des chunks."""

    def __init__(
        self,
        *,
        index_directory: Path,
        embedding_config_path: Path,
    ) -> None:
        self.index_directory = index_directory.resolve()
        self.embedding_config_path = (
            embedding_config_path.resolve()
        )

        self.index_path = (
            self.index_directory / "index.faiss"
        )
        self.metadata_path = (
            self.index_directory / "metadata.jsonl"
        )
        self.manifest_path = (
            self.index_directory / "manifest.json"
        )

        self._check_required_files()

        self.index: Any = faiss.read_index(
            str(self.index_path)
        )

        self.metadata = self._load_metadata(
            self.metadata_path
        )

        self._validate_index_and_metadata()

        embedding_config = load_embedding_config(
            self.embedding_config_path
        )

        if (
            embedding_config.embedding_dimension
            != int(self.index.d)
        ):
            raise ValueError(
                "La dimension du modèle d'embeddings ne "
                "correspond pas à celle de l'index FAISS : "
                f"{embedding_config.embedding_dimension} != "
                f"{self.index.d}."
            )

        self.embedder = BGEEmbedder(embedding_config)

    @property
    def total_vectors(self) -> int:
        """Nombre de vecteurs disponibles."""

        return int(self.index.ntotal)

    @property
    def dimension(self) -> int:
        """Dimension des vecteurs FAISS."""

        return int(self.index.d)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        minimum_score: float | None = None,
        document_ids: set[str] | None = None,
        chunk_ids: set[str] | None = None,
    ) -> DenseSearchResponse:
        """Rechercher les chunks les plus proches d'une question."""

        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError(
                "La requête de recherche ne peut pas être vide."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k doit être strictement positif."
            )

        if minimum_score is not None and not -1 <= minimum_score <= 1:
            raise ValueError(
                "minimum_score doit être compris entre -1 et 1."
            )

        normalized_chunk_ids = (
            {chunk_id.strip() for chunk_id in chunk_ids if chunk_id.strip()}
            if chunk_ids
            else None
        )
        normalized_document_ids = (
            {
                document_id.strip()
                for document_id in document_ids
                if document_id.strip()
            }
            if document_ids
            else None
        )

        # Sans filtre, il suffit de demander directement top_k.
        # Avec filtre documentaire, on récupère tous les vecteurs
        # puis on conserve uniquement les documents autorisés.
        faiss_k = (
            self.total_vectors
            if normalized_document_ids or normalized_chunk_ids
            else min(top_k, self.total_vectors)
        )

        search_start = time.perf_counter()

        query_vector = self.embedder.embed_query(
            cleaned_query
        )

        query_matrix = np.ascontiguousarray(
            query_vector.reshape(1, -1),
            dtype=np.float32,
        )

        if query_matrix.shape != (1, self.dimension):
            raise ValueError(
                "Dimension inattendue pour l'embedding de "
                f"requête : {query_matrix.shape}."
            )

        scores, vector_ids = self.index.search(
            query_matrix,
            faiss_k,
        )

        results: list[DenseSearchResult] = []

        for score, vector_id in zip(
            scores[0],
            vector_ids[0],
            strict=True,
        ):
            vector_id = int(vector_id)
            score = float(score)

            if vector_id < 0:
                continue

            chunk = self.metadata[vector_id]

            if (
                normalized_document_ids
                and chunk.document_id not in normalized_document_ids
            ):
                continue

            if normalized_chunk_ids and chunk.chunk_id not in normalized_chunk_ids:
                continue

            if (
                minimum_score is not None
                and score < minimum_score
            ):
                continue

            results.append(
                DenseSearchResult(
                    rank=len(results) + 1,
                    vector_id=vector_id,
                    score=score,
                    chunk=chunk,
                )
            )

            if len(results) >= top_k:
                break

        duration_ms = (
            time.perf_counter() - search_start
        ) * 1000

        return DenseSearchResponse(
            query=cleaned_query,
            top_k_requested=top_k,
            search_duration_ms=round(duration_ms, 3),
            results=results,
        )

    def _check_required_files(self) -> None:
        """Vérifier la présence des artefacts nécessaires."""

        required_files = [
            self.index_path,
            self.metadata_path,
            self.manifest_path,
            self.embedding_config_path,
        ]

        missing_files = [
            path
            for path in required_files
            if not path.exists()
        ]

        if missing_files:
            formatted_paths = "\n".join(
                f"- {path}"
                for path in missing_files
            )

            raise FileNotFoundError(
                "Fichiers nécessaires introuvables :\n"
                f"{formatted_paths}"
            )

    @staticmethod
    def _load_metadata(
        metadata_path: Path,
    ) -> list[DocumentChunk]:
        """Charger les chunks associés aux vecteurs FAISS."""

        chunks: list[DocumentChunk] = []
        vector_ids: list[int] = []

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as source:
            for line_number, line in enumerate(
                source,
                start=1,
            ):
                if not line.strip():
                    raise ValueError(
                        f"{metadata_path.name}, ligne "
                        f"{line_number} vide."
                    )

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{metadata_path.name}, ligne "
                        f"{line_number} : JSON invalide."
                    ) from error

                if not isinstance(record, dict):
                    raise ValueError(
                        f"{metadata_path.name}, ligne "
                        f"{line_number} : objet attendu."
                    )

                vector_id = record.pop(
                    "vector_id",
                    None,
                )

                if (
                    not isinstance(vector_id, int)
                    or isinstance(vector_id, bool)
                ):
                    raise ValueError(
                        f"{metadata_path.name}, ligne "
                        f"{line_number} : vector_id invalide."
                    )

                try:
                    chunk = DocumentChunk.model_validate(
                        record
                    )
                except ValidationError as error:
                    raise ValueError(
                        f"{metadata_path.name}, ligne "
                        f"{line_number} : chunk invalide."
                    ) from error

                vector_ids.append(vector_id)
                chunks.append(chunk)

        expected_vector_ids = list(range(len(chunks)))

        if vector_ids != expected_vector_ids:
            raise ValueError(
                "Les vector_id des métadonnées ne sont pas "
                "continus et correctement ordonnés."
            )

        return chunks

    def _validate_index_and_metadata(self) -> None:
        """Contrôler la cohérence minimale au chargement."""

        if type(self.index).__name__ != "IndexFlatIP":
            raise ValueError(
                "Type d'index inattendu : "
                f"{type(self.index).__name__}."
            )

        if (
            int(self.index.metric_type)
            != faiss.METRIC_INNER_PRODUCT
        ):
            raise ValueError(
                "L'index FAISS n'utilise pas le produit "
                "scalaire."
            )

        if int(self.index.ntotal) != len(self.metadata):
            raise ValueError(
                "Le nombre de vecteurs et le nombre de "
                "métadonnées sont différents : "
                f"{self.index.ntotal} != "
                f"{len(self.metadata)}."
            )

        if int(self.index.ntotal) == 0:
            raise ValueError(
                "L'index FAISS est vide."
            )

        chunk_ids = [
            chunk.chunk_id
            for chunk in self.metadata
        ]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                "Des chunk_id sont dupliqués dans les "
                "métadonnées."
            )
