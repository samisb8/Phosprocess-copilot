"""Reranking cross-encoder avec BGE-Reranker-v2-M3."""

from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from FlagEmbedding import FlagReranker

from phosprocess.preprocessing.chunk_schemas import DocumentChunk
from phosprocess.retrieval.hybrid import HybridSearchResult

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"[ \t]+")
_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")


@dataclass(frozen=True, slots=True)
class RerankingConfig:
    """Configuration du modèle de reranking."""

    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "auto"
    use_fp16: bool = True
    normalize_scores: bool = True

    batch_size: int = 2
    max_length: int = 1024

    hybrid_candidate_k: int = 15
    final_top_k: int = 5

    include_source_file: bool = True
    include_heading_path: bool = True

    pipeline_version: str = "0.1.0"

    def __post_init__(self) -> None:
        """Valider les paramètres."""

        if not self.model_name.strip():
            raise ValueError(
                "Le nom du modèle de reranking ne peut pas être vide."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size doit être strictement positif."
            )

        if self.max_length <= 0:
            raise ValueError(
                "max_length doit être strictement positif."
            )

        if self.hybrid_candidate_k <= 0:
            raise ValueError(
                "hybrid_candidate_k doit être strictement positif."
            )

        if self.final_top_k <= 0:
            raise ValueError(
                "final_top_k doit être strictement positif."
            )

        if self.final_top_k > self.hybrid_candidate_k:
            raise ValueError(
                "final_top_k ne peut pas dépasser "
                "hybrid_candidate_k."
            )


@dataclass(frozen=True, slots=True)
class RerankedSearchResult:
    """Passage après classement par le cross-encoder."""

    rank: int
    reranker_score: float

    original_hybrid_rank: int
    original_rrf_score: float

    matched_retrievers: tuple[str, ...]

    dense_rank: int | None
    dense_score: float | None

    bm25_rank: int | None
    bm25_score: float | None

    chunk: DocumentChunk

    sparse_rank: int | None = None
    sparse_score: float | None = None
    colbert_score: float | None = None
    section_bonus: float = 0.0
    role_matches: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RerankingResponse:
    """Réponse complète du reranker."""

    query: str
    model_name: str
    device: str

    candidates_received: int
    top_k_requested: int
    reranking_duration_ms: float

    results: list[RerankedSearchResult]


def load_reranking_config(
    config_path: Path,
) -> RerankingConfig:
    """Lire la configuration YAML du reranker."""

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration introuvable : {config_path}"
        )

    raw_config = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )

    if not isinstance(raw_config, dict):
        raise ValueError(
            "Le fichier reranking.yaml est invalide."
        )

    model_config = raw_config.get("model")
    inference_config = raw_config.get("inference")
    retrieval_config = raw_config.get("retrieval")
    passage_config = raw_config.get("passage")

    if not isinstance(model_config, dict):
        raise ValueError(
            "La section 'model' est absente ou invalide."
        )

    if not isinstance(inference_config, dict):
        raise ValueError(
            "La section 'inference' est absente ou invalide."
        )

    if not isinstance(retrieval_config, dict):
        raise ValueError(
            "La section 'retrieval' est absente ou invalide."
        )

    if not isinstance(passage_config, dict):
        raise ValueError(
            "La section 'passage' est absente ou invalide."
        )

    return RerankingConfig(
        model_name=str(model_config["name"]),
        device=str(model_config.get("device", "auto")),
        use_fp16=bool(
            model_config.get("use_fp16", True)
        ),
        normalize_scores=bool(
            model_config.get("normalize_scores", True)
        ),
        batch_size=int(inference_config["batch_size"]),
        max_length=int(inference_config["max_length"]),
        hybrid_candidate_k=int(
            retrieval_config["hybrid_candidate_k"]
        ),
        final_top_k=int(
            retrieval_config["final_top_k"]
        ),
        include_source_file=bool(
            passage_config.get(
                "include_source_file",
                True,
            )
        ),
        include_heading_path=bool(
            passage_config.get(
                "include_heading_path",
                True,
            )
        ),
        pipeline_version=str(
            raw_config.get(
                "pipeline_version",
                "unknown",
            )
        ),
    )


def clean_passage_text(text: str) -> str:
    """Retirer les balises HTML tout en conservant le contenu."""

    cleaned = html.unescape(text)
    cleaned = _HTML_TAG.sub(" ", cleaned)

    lines: list[str] = []

    for line in cleaned.splitlines():
        normalized_line = _WHITESPACE.sub(
            " ",
            line,
        ).strip()

        if normalized_line:
            lines.append(normalized_line)

    cleaned = "\n".join(lines)
    cleaned = _MULTIPLE_NEWLINES.sub(
        "\n\n",
        cleaned,
    )

    return cleaned.strip()


def build_reranking_passage(
    chunk: DocumentChunk,
    config: RerankingConfig,
) -> str:
    """Construire le passage présenté au cross-encoder."""

    context_lines: list[str] = []

    if config.include_source_file:
        document_name = (
            Path(chunk.source_file)
            .stem
            .replace("_", " ")
        )

        context_lines.append(
            f"Document: {document_name}"
        )

    if (
        config.include_heading_path
        and chunk.heading_path
    ):
        context_lines.append(
            "Section: "
            + " > ".join(chunk.heading_path)
        )

    passage_text = clean_passage_text(
        chunk.text
    )

    if not passage_text:
        raise ValueError(
            f"Le chunk {chunk.chunk_id} possède un texte vide."
        )

    if context_lines:
        return (
            "\n".join(context_lines)
            + "\n\nPassage:\n"
            + passage_text
        )

    return passage_text


class BGEReranker:
    """Reranker multilingue basé sur un cross-encoder BGE."""

    def __init__(
        self,
        config: RerankingConfig,
    ) -> None:
        self.config = config
        self._device = self._resolve_device(
            config.device
        )

        effective_fp16 = (
            config.use_fp16
            and self._device.startswith("cuda")
        )

        devices: str | list[str] = (
            [self._device]
            if self._device.startswith("cuda")
            else "cpu"
        )

        print(
            f"Chargement de {config.model_name} "
            f"sur {self._device}..."
        )

        self._model: Any = FlagReranker(
            config.model_name,
            devices=devices,
            use_fp16=effective_fp16,
        )

    @property
    def model_name(self) -> str:
        """Nom du modèle de reranking."""

        return self.config.model_name

    @property
    def device(self) -> str:
        """Appareil utilisé par le modèle."""

        return self._device

    def rerank(
        self,
        query: str,
        candidates: list[HybridSearchResult],
        *,
        top_k: int | None = None,
    ) -> RerankingResponse:
        """Reclasser les candidats hybrides."""

        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError(
                "La question de reranking ne peut pas être vide."
            )

        effective_top_k = (
            self.config.final_top_k
            if top_k is None
            else top_k
        )

        if effective_top_k <= 0:
            raise ValueError(
                "top_k doit être strictement positif."
            )

        if not candidates:
            return RerankingResponse(
                query=cleaned_query,
                model_name=self.model_name,
                device=self.device,
                candidates_received=0,
                top_k_requested=effective_top_k,
                reranking_duration_ms=0.0,
                results=[],
            )

        chunk_ids = [
            candidate.chunk.chunk_id
            for candidate in candidates
        ]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                "Les candidats contiennent des chunk_id dupliqués."
            )

        pairs = [
            [
                cleaned_query,
                build_reranking_passage(
                    candidate.chunk,
                    self.config,
                ),
            ]
            for candidate in candidates
        ]

        self._synchronize_cuda()
        start_time = time.perf_counter()

        scores = self._compute_scores(
            pairs=pairs,
        )

        self._synchronize_cuda()

        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        scored_candidates = list(
            zip(
                candidates,
                scores,
                strict=True,
            )
        )

        scored_candidates.sort(
            key=lambda item: (
                -float(item[1]),
                item[0].rank,
                item[0].chunk.chunk_id,
            )
        )

        selected_candidates = scored_candidates[
            : min(
                effective_top_k,
                len(scored_candidates),
            )
        ]

        results = [
            RerankedSearchResult(
                rank=final_rank,
                reranker_score=float(score),
                original_hybrid_rank=candidate.rank,
                original_rrf_score=candidate.rrf_score,
                matched_retrievers=(
                    candidate.matched_retrievers
                ),
                dense_rank=candidate.dense_rank,
                dense_score=candidate.dense_score,
                bm25_rank=candidate.bm25_rank,
                bm25_score=candidate.bm25_score,
                chunk=candidate.chunk,
                sparse_rank=candidate.sparse_rank,
                sparse_score=candidate.sparse_score,
                colbert_score=candidate.colbert_score,
                section_bonus=candidate.section_bonus,
                role_matches=candidate.role_matches,
            )
            for final_rank, (candidate, score) in enumerate(
                selected_candidates,
                start=1,
            )
        ]

        return RerankingResponse(
            query=cleaned_query,
            model_name=self.model_name,
            device=self.device,
            candidates_received=len(candidates),
            top_k_requested=effective_top_k,
            reranking_duration_ms=round(
                duration_ms,
                3,
            ),
            results=results,
        )

    def _compute_scores(
        self,
        *,
        pairs: list[list[str]],
    ) -> np.ndarray:
        """Calculer les scores avec réduction automatique du batch."""

        batch_size = min(
            self.config.batch_size,
            len(pairs),
        )

        while True:
            try:
                raw_scores = self._model.compute_score(
                    pairs,
                    batch_size=batch_size,
                    max_length=self.config.max_length,
                    normalize=self.config.normalize_scores,
                )

                scores = np.asarray(
                    raw_scores,
                    dtype=np.float32,
                ).reshape(-1)

                self._validate_scores(
                    scores,
                    expected_count=len(pairs),
                )

                return scores

            except (
                torch.OutOfMemoryError,
                RuntimeError,
            ) as error:
                if not self._is_cuda_oom(error):
                    raise

                if batch_size <= 1:
                    raise RuntimeError(
                        "Mémoire GPU insuffisante pour le "
                        "reranker même avec batch_size=1."
                    ) from error

                previous_batch_size = batch_size
                batch_size = max(
                    batch_size // 2,
                    1,
                )

                torch.cuda.empty_cache()

                print(
                    "Mémoire GPU insuffisante pour le reranker : "
                    f"batch_size {previous_batch_size} "
                    f"→ {batch_size}"
                )

    def _validate_scores(
        self,
        scores: np.ndarray,
        *,
        expected_count: int,
    ) -> None:
        """Contrôler la sortie du cross-encoder."""

        if scores.shape != (expected_count,):
            raise ValueError(
                "Nombre de scores inattendu : "
                f"{scores.shape}, attendu ({expected_count},)."
            )

        if not np.isfinite(scores).all():
            raise ValueError(
                "Le reranker a produit NaN ou une valeur infinie."
            )

        if self.config.normalize_scores:
            outside_range = (
                (scores < 0.0)
                | (scores > 1.0)
            )

            if bool(outside_range.any()):
                raise ValueError(
                    "Les scores normalisés doivent être compris "
                    "entre 0 et 1."
                )

    def _is_cuda_oom(
        self,
        error: BaseException,
    ) -> bool:
        """Déterminer si une erreur vient de la mémoire CUDA."""

        if not self._device.startswith("cuda"):
            return False

        if isinstance(
            error,
            torch.OutOfMemoryError,
        ):
            return True

        return (
            "out of memory"
            in str(error).casefold()
        )

    def _synchronize_cuda(self) -> None:
        """Synchroniser CUDA pour mesurer correctement la durée."""

        if self._device.startswith("cuda"):
            torch.cuda.synchronize()

    @staticmethod
    def _resolve_device(
        requested_device: str,
    ) -> str:
        """Choisir automatiquement GPU ou CPU."""

        normalized = (
            requested_device.strip().casefold()
        )

        if normalized == "auto":
            return (
                "cuda:0"
                if torch.cuda.is_available()
                else "cpu"
            )

        if normalized.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA est demandé, mais PyTorch ne détecte "
                    "aucun GPU compatible."
                )

            return requested_device

        if normalized == "cpu":
            return "cpu"

        raise ValueError(
            "device doit être 'auto', 'cpu' ou une valeur "
            "comme 'cuda:0'."
        )