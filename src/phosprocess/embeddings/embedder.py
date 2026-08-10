"""Génération d'embeddings denses avec BGE-M3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from FlagEmbedding import BGEM3FlagModel
from huggingface_hub.constants import HF_HUB_CACHE


def resolve_cached_model_source(
    model_name: str,
    *,
    cache_dir: str | None = None,
) -> str:
    """Prefer an installed Hub snapshot to avoid startup network probes."""

    direct_path = Path(model_name)

    if direct_path.is_dir():
        return str(direct_path.resolve())

    cache_root = Path(cache_dir) if cache_dir is not None else Path(HF_HUB_CACHE)
    model_cache = cache_root / f"models--{model_name.replace('/', '--')}"
    main_reference = model_cache / "refs" / "main"

    if main_reference.is_file():
        revision = main_reference.read_text(encoding="utf-8").strip()
        snapshot = model_cache / "snapshots" / revision

        if snapshot.is_dir():
            return str(snapshot.resolve())

    snapshots = model_cache / "snapshots"

    if snapshots.is_dir():
        available = sorted(path for path in snapshots.iterdir() if path.is_dir())

        if len(available) == 1:
            return str(available[0].resolve())

    return model_name


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Configuration du modèle d'embeddings."""

    model_name: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024

    device: str = "auto"
    use_fp16: bool = True
    normalize_embeddings: bool = True

    trust_remote_code: bool = False
    cache_dir: str | None = None

    batch_size: int = 4
    passage_max_length: int = 768
    query_max_length: int = 256

    pipeline_version: str = "0.1.0"

    def __post_init__(self) -> None:
        """Valider les paramètres essentiels."""

        if not self.model_name.strip():
            raise ValueError("model_name ne peut pas être vide.")

        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension doit être strictement positif.")

        if self.batch_size <= 0:
            raise ValueError("batch_size doit être strictement positif.")

        if self.passage_max_length <= 0:
            raise ValueError("passage_max_length doit être strictement positif.")

        if self.query_max_length <= 0:
            raise ValueError("query_max_length doit être strictement positif.")


def load_embedding_config(config_path: Path) -> EmbeddingConfig:
    """Lire la configuration YAML des embeddings."""

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration introuvable : {config_path}")

    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if not isinstance(raw_config, dict):
        raise ValueError(f"Configuration YAML invalide : {config_path}")

    model_config = raw_config.get("model")
    inference_config = raw_config.get("inference")

    if not isinstance(model_config, dict):
        raise ValueError("La section 'model' est absente ou invalide.")

    if not isinstance(inference_config, dict):
        raise ValueError("La section 'inference' est absente ou invalide.")

    cache_dir_value = model_config.get("cache_dir")

    cache_dir = str(cache_dir_value) if cache_dir_value is not None else None

    return EmbeddingConfig(
        model_name=str(model_config["name"]),
        embedding_dimension=int(model_config["dimension"]),
        device=str(model_config.get("device", "auto")),
        use_fp16=bool(model_config.get("use_fp16", True)),
        normalize_embeddings=bool(model_config.get("normalize_embeddings", True)),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        cache_dir=cache_dir,
        batch_size=int(inference_config["batch_size"]),
        passage_max_length=int(inference_config["passage_max_length"]),
        query_max_length=int(inference_config["query_max_length"]),
        pipeline_version=str(raw_config.get("pipeline_version", "0.1.0")),
    )


class BGEEmbedder:
    """Encodeur dense local basé sur BGE-M3."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._device = self._resolve_device(config.device)

        effective_fp16 = config.use_fp16 and self._device.startswith("cuda")

        print(f"Chargement de {config.model_name} sur {self._device}...")

        model_source = resolve_cached_model_source(
            config.model_name,
            cache_dir=config.cache_dir,
        )
        self._model: Any = BGEM3FlagModel(
            model_source,
            normalize_embeddings=(config.normalize_embeddings),
            use_fp16=effective_fp16,
            devices=self._device,
            trust_remote_code=config.trust_remote_code,
            cache_dir=config.cache_dir,
        )

    @property
    def model_name(self) -> str:
        """Nom du modèle utilisé."""

        return self.config.model_name

    @property
    def device(self) -> str:
        """Appareil réellement utilisé."""

        return self._device

    @property
    def dimension(self) -> int:
        """Dimension attendue d'un embedding."""

        return self.config.embedding_dimension

    def count_tokens(self, text: str) -> int:
        """Count text with the tokenizer already loaded by BGE-M3."""

        tokenizer = getattr(self._model, "tokenizer", None)
        if tokenizer is None or not callable(getattr(tokenizer, "encode", None)):
            raise RuntimeError("Le tokenizer BGE-M3 n'est pas disponible.")
        return len(tokenizer.encode(text, add_special_tokens=False))

    def embed_query(self, query: str) -> np.ndarray:
        """Encoder une requête et retourner un vecteur 1D."""

        vectors = self._encode_texts(
            texts=[query],
            max_length=self.config.query_max_length,
        )

        return vectors[0]

    def embed_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Encoder plusieurs passages en une matrice 2D."""

        return self._encode_texts(
            texts=texts,
            max_length=self.config.passage_max_length,
        )

    def embed_sparse_query(self, query: str) -> dict[int, float]:
        """Encode one query as BGE-M3 lexical weights."""

        return self._encode_sparse_texts(
            [query],
            max_length=self.config.query_max_length,
            query_mode=True,
        )[0]

    def embed_sparse_documents(
        self,
        texts: list[str],
    ) -> list[dict[int, float]]:
        """Encode passages as BGE-M3 sparse lexical weights."""

        return self._encode_sparse_texts(
            texts,
            max_length=self.config.passage_max_length,
            query_mode=False,
        )

    def embed_colbert_query(self, query: str) -> np.ndarray:
        """Encode one query as BGE-M3 token-level ColBERT vectors."""

        return self._encode_colbert_texts(
            [query],
            max_length=self.config.query_max_length,
            query_mode=True,
        )[0]

    def embed_colbert_documents(self, texts: list[str]) -> list[np.ndarray]:
        """Encode candidate passages as BGE-M3 ColBERT vectors."""

        return self._encode_colbert_texts(
            texts,
            max_length=self.config.passage_max_length,
            query_mode=False,
        )

    def colbert_score(
        self,
        query_vectors: np.ndarray,
        passage_vectors: np.ndarray,
    ) -> float:
        """Compute the official BGE-M3 late-interaction score."""

        scorer = getattr(self._model, "colbert_score", None)
        if not callable(scorer):
            raise RuntimeError("Cette version de FlagEmbedding n'expose pas colbert_score.")
        return float(scorer(query_vectors, passage_vectors))

    def _encode_sparse_texts(
        self,
        texts: list[str],
        *,
        max_length: int,
        query_mode: bool,
    ) -> list[dict[int, float]]:
        output = self._encode_features(
            texts,
            max_length=max_length,
            query_mode=query_mode,
            return_sparse=True,
            return_colbert=False,
        )
        raw_weights = output.get("lexical_weights")
        if not isinstance(raw_weights, list) or len(raw_weights) != len(texts):
            raise ValueError("BGE-M3 a retourné des poids lexicaux invalides.")
        normalized: list[dict[int, float]] = []
        for weights in raw_weights:
            if not isinstance(weights, dict):
                raise TypeError("Chaque embedding sparse doit être un dictionnaire.")
            normalized.append(
                {
                    int(token_id): float(weight)
                    for token_id, weight in weights.items()
                    if float(weight) > 0.0
                }
            )
        return normalized

    def _encode_colbert_texts(
        self,
        texts: list[str],
        *,
        max_length: int,
        query_mode: bool,
    ) -> list[np.ndarray]:
        output = self._encode_features(
            texts,
            max_length=max_length,
            query_mode=query_mode,
            return_sparse=False,
            return_colbert=True,
        )
        raw_vectors = output.get("colbert_vecs")
        if not isinstance(raw_vectors, (list, tuple, np.ndarray)) or (
            len(raw_vectors) != len(texts)
        ):
            raise ValueError("BGE-M3 a retourné des vecteurs ColBERT invalides.")
        vectors = [np.asarray(item, dtype=np.float32) for item in raw_vectors]
        if any(item.ndim != 2 or not np.isfinite(item).all() for item in vectors):
            raise ValueError("Un embedding ColBERT possède une forme ou des valeurs invalides.")
        return vectors

    def _encode_features(
        self,
        texts: list[str],
        *,
        max_length: int,
        query_mode: bool,
        return_sparse: bool,
        return_colbert: bool,
    ) -> dict[str, Any]:
        """Encode sparse or ColBERT features with automatic OOM batch reduction."""

        cleaned = [text.strip() for text in texts]
        if not cleaned or any(not text for text in cleaned):
            raise ValueError("Les textes BGE-M3 sparse/ColBERT ne peuvent pas être vides.")
        batch_size = min(self.config.batch_size, len(cleaned))
        method_name = "encode_queries" if query_mode else "encode_corpus"
        encoder = getattr(self._model, method_name, None)
        if not callable(encoder):
            encoder = self._model.encode

        while True:
            try:
                output = encoder(
                    cleaned,
                    batch_size=batch_size,
                    max_length=max_length,
                    return_dense=False,
                    return_sparse=return_sparse,
                    return_colbert_vecs=return_colbert,
                )
                if not isinstance(output, dict):
                    raise TypeError("La sortie BGE-M3 doit être un dictionnaire.")
                return output
            except (torch.OutOfMemoryError, RuntimeError) as error:
                if not self._is_cuda_oom(error) or batch_size <= 1:
                    raise
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()

    def _encode_texts(
        self,
        *,
        texts: list[str],
        max_length: int,
    ) -> np.ndarray:
        """Encoder les textes avec réduction automatique du batch."""

        if not texts:
            return np.empty(
                (0, self.config.embedding_dimension),
                dtype=np.float32,
            )

        cleaned_texts = [text.strip() for text in texts]

        empty_positions = [index for index, text in enumerate(cleaned_texts) if not text]

        if empty_positions:
            raise ValueError(f"Textes vides détectés aux positions : {empty_positions}")

        batch_size = min(
            self.config.batch_size,
            len(cleaned_texts),
        )

        while True:
            try:
                vectors = self._run_encoding(
                    texts=cleaned_texts,
                    batch_size=batch_size,
                    max_length=max_length,
                )

                return self._validate_vectors(
                    vectors,
                    expected_rows=len(cleaned_texts),
                )

            except (torch.OutOfMemoryError, RuntimeError) as error:
                if not self._is_cuda_oom(error):
                    raise

                if batch_size <= 1:
                    raise RuntimeError(
                        "Mémoire GPU insuffisante même avec batch_size=1."
                    ) from error

                previous_batch_size = batch_size
                batch_size = max(batch_size // 2, 1)

                torch.cuda.empty_cache()

                print(f"Mémoire GPU insuffisante : batch_size {previous_batch_size} → {batch_size}")

    def _run_encoding(
        self,
        *,
        texts: list[str],
        batch_size: int,
        max_length: int,
    ) -> np.ndarray:
        """Appeler l'API officielle de BGE-M3."""

        output = self._model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        if not isinstance(output, dict):
            raise TypeError("La sortie de BGE-M3 doit être un dictionnaire.")

        if "dense_vecs" not in output:
            raise KeyError("La sortie BGE-M3 ne contient pas 'dense_vecs'.")

        return np.asarray(
            output["dense_vecs"],
            dtype=np.float32,
        )

    def _validate_vectors(
        self,
        vectors: np.ndarray,
        *,
        expected_rows: int,
    ) -> np.ndarray:
        """Vérifier la forme et les valeurs des embeddings."""

        if vectors.ndim != 2:
            raise ValueError(
                f"Les embeddings doivent former une matrice 2D, forme reçue : {vectors.shape}."
            )

        expected_shape = (
            expected_rows,
            self.config.embedding_dimension,
        )

        if vectors.shape != expected_shape:
            raise ValueError(f"Dimension inattendue : {vectors.shape}, attendu {expected_shape}.")

        if not np.isfinite(vectors).all():
            raise ValueError("Les embeddings contiennent NaN ou une valeur infinie.")

        if self.config.normalize_embeddings:
            vectors = self._normalize_l2(vectors)

            norms = np.linalg.norm(
                vectors,
                axis=1,
            )

            if not np.allclose(
                norms,
                1.0,
                atol=1e-4,
            ):
                raise ValueError("Les embeddings ne sont pas correctement normalisés.")

        return np.ascontiguousarray(
            vectors,
            dtype=np.float32,
        )

    @staticmethod
    def _normalize_l2(
        vectors: np.ndarray,
    ) -> np.ndarray:
        """Normaliser chaque vecteur avec la norme L2."""

        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True,
        )

        if np.any(norms <= 0):
            raise ValueError("Un embedding possède une norme nulle.")

        return vectors / norms

    def _is_cuda_oom(
        self,
        error: BaseException,
    ) -> bool:
        """Déterminer si l'erreur vient de la mémoire GPU."""

        if not self._device.startswith("cuda"):
            return False

        if isinstance(error, torch.OutOfMemoryError):
            return True

        return "out of memory" in str(error).casefold()

    @staticmethod
    def _resolve_device(requested_device: str) -> str:
        """Choisir automatiquement CUDA ou CPU."""

        normalized = requested_device.strip().casefold()

        if normalized == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"

        if normalized.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA a été demandé, mais aucun GPU CUDA n'est accessible par PyTorch."
                )

            return requested_device

        if normalized == "cpu":
            return "cpu"

        raise ValueError("device doit être 'auto', 'cpu' ou une valeur comme 'cuda:0'.")
