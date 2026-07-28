"""Recherche lexicale BM25 adaptée au corpus phosphorique."""

from __future__ import annotations

import html
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bm25s
import yaml
from pydantic import ValidationError

from phosprocess.preprocessing.chunk_schemas import DocumentChunk

TOKENIZER_VERSION = "technical_v1"

_SUB_SUP_TAG = re.compile(
    r"</?(?:sup|sub)\b[^>]*>",
    flags=re.IGNORECASE,
)

_HTML_TAG = re.compile(r"<[^>]+>")

_TOKEN_PATTERN = re.compile(
    r"""
    \d+(?:[.,]\d+)?\s*:\s*\d+(?:[.,]\d+)?
    |
    \d+(?:[.,]\d+)?%
    |
    [^\W_][\w]*(?:[./·_-][\w]+)*
    """,
    flags=re.UNICODE | re.VERBOSE,
)

_TRANSLATION_TABLE = str.maketrans(
    {
        "‐": "-",
        "-": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "−": "-",
        "⁄": "/",
        "×": "x",
        "\u00a0": " ",
    }
)


@dataclass(frozen=True, slots=True)
class BM25Config:
    """Configuration du moteur BM25."""

    method: str
    k1: float
    b: float

    backend: str
    csc_backend: str

    minimum_score: float

    tokenizer_version: str
    use_stemming: bool
    remove_stopwords: bool

    chunks_directory: str
    output_directory: str
    metadata_filename: str
    manifest_filename: str

    pipeline_version: str

    def __post_init__(self) -> None:
        """Contrôler les paramètres essentiels."""

        allowed_methods = {
            "lucene",
            "robertson",
            "atire",
            "bm25l",
            "bm25+",
        }

        if self.method not in allowed_methods:
            raise ValueError(
                f"Méthode BM25 non prise en charge : {self.method}"
            )

        if self.k1 < 0:
            raise ValueError("k1 doit être positif ou nul.")

        if not 0 <= self.b <= 1:
            raise ValueError("b doit être compris entre 0 et 1.")

        if self.minimum_score < 0:
            raise ValueError(
                "minimum_score doit être positif ou nul."
            )

        if self.tokenizer_version != TOKENIZER_VERSION:
            raise ValueError(
                "Version de tokenizer inconnue : "
                f"{self.tokenizer_version}"
            )

        if self.use_stemming:
            raise ValueError(
                "Le stemming n'est pas encore activé dans "
                "technical_v1."
            )

        if self.remove_stopwords:
            raise ValueError(
                "La suppression des stopwords n'est pas encore "
                "activée dans technical_v1."
            )


@dataclass(frozen=True, slots=True)
class BM25SearchResult:
    """Résultat individuel d'une recherche BM25."""

    rank: int
    lexical_id: int
    score: float
    chunk: DocumentChunk


@dataclass(frozen=True, slots=True)
class BM25SearchResponse:
    """Réponse complète du moteur lexical."""

    query: str
    query_tokens: list[str]
    top_k_requested: int
    search_duration_ms: float
    results: list[BM25SearchResult]


def load_bm25_config(config_path: Path) -> BM25Config:
    """Lire la configuration BM25 depuis retrieval.yaml."""

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration introuvable : {config_path}"
        )

    raw_config = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )

    if not isinstance(raw_config, dict):
        raise ValueError("Configuration retrieval.yaml invalide.")

    bm25_config = raw_config.get("bm25")

    if not isinstance(bm25_config, dict):
        raise ValueError(
            "La section 'bm25' est absente ou invalide."
        )

    tokenizer_config = bm25_config.get("tokenizer")
    data_config = bm25_config.get("data")
    index_config = bm25_config.get("index")

    if not isinstance(tokenizer_config, dict):
        raise ValueError(
            "La section bm25.tokenizer est invalide."
        )

    if not isinstance(data_config, dict):
        raise ValueError("La section bm25.data est invalide.")

    if not isinstance(index_config, dict):
        raise ValueError("La section bm25.index est invalide.")

    return BM25Config(
        method=str(bm25_config["method"]),
        k1=float(bm25_config["k1"]),
        b=float(bm25_config["b"]),
        backend=str(bm25_config.get("backend", "numpy")),
        csc_backend=str(
            bm25_config.get("csc_backend", "numpy")
        ),
        minimum_score=float(
            bm25_config.get("minimum_score", 0.0)
        ),
        tokenizer_version=str(tokenizer_config["version"]),
        use_stemming=bool(
            tokenizer_config.get("use_stemming", False)
        ),
        remove_stopwords=bool(
            tokenizer_config.get("remove_stopwords", False)
        ),
        chunks_directory=str(
            data_config["chunks_directory"]
        ),
        output_directory=str(
            index_config["output_directory"]
        ),
        metadata_filename=str(
            index_config["metadata_filename"]
        ),
        manifest_filename=str(
            index_config["manifest_filename"]
        ),
        pipeline_version=str(
            bm25_config.get("pipeline_version", "unknown")
        ),
    )


def normalize_lexical_text(text: str) -> str:
    """Normaliser le texte sans détruire les symboles techniques."""

    normalized = html.unescape(text)

    # Conserver les exposants dans la formule :
    # m<sup>3</sup> devient m3.
    normalized = _SUB_SUP_TAG.sub("", normalized)

    # Retirer les autres balises HTML.
    normalized = _HTML_TAG.sub(" ", normalized)

    normalized = unicodedata.normalize(
        "NFKC",
        normalized,
    )

    normalized = normalized.translate(
        _TRANSLATION_TABLE
    )

    normalized = normalized.casefold()

    return re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()


def technical_tokenize(text: str) -> list[str]:
    """Tokeniser le texte en préservant les expressions techniques."""

    normalized = normalize_lexical_text(text)

    tokens: list[str] = []

    for match in _TOKEN_PATTERN.finditer(normalized):
        token = re.sub(
            r"\s+",
            "",
            match.group(0),
        )

        if token:
            tokens.append(token)

    return tokens


def build_lexical_text(chunk: DocumentChunk) -> str:
    """Construire le champ textuel réellement indexé par BM25."""

    source_name = (
        Path(chunk.source_file)
        .stem
        .replace("_", " ")
    )

    parts = [source_name]

    parts.extend(
        heading
        for heading in chunk.heading_path
        if heading.strip()
    )

    parts.append(chunk.text)

    return "\n".join(parts)


class BM25Retriever:
    """Charger et interroger un index lexical BM25S."""

    def __init__(
        self,
        *,
        index_directory: Path,
        config_path: Path,
    ) -> None:
        self.index_directory = index_directory.resolve()
        self.config_path = config_path.resolve()

        self.config = load_bm25_config(
            self.config_path
        )

        self.metadata_path = (
            self.index_directory
            / self.config.metadata_filename
        )

        self.manifest_path = (
            self.index_directory
            / self.config.manifest_filename
        )

        self._check_required_files()

        self.model: Any = bm25s.BM25.load(
            str(self.index_directory),
            load_corpus=False,
            mmap=False,
            show_progress=False,
        )

        self.metadata = self._load_metadata(
            self.metadata_path
        )

        self.manifest = self._load_manifest(
            self.manifest_path
        )

        self._validate_loaded_index()

    @property
    def total_documents(self) -> int:
        """Nombre de chunks dans l'index lexical."""

        return len(self.metadata)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        minimum_score: float | None = None,
        document_ids: set[str] | None = None,
        chunk_ids: set[str] | None = None,
    ) -> BM25SearchResponse:
        """Rechercher les chunks correspondant aux termes de la requête."""

        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError(
                "La requête BM25 ne peut pas être vide."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k doit être strictement positif."
            )

        threshold = (
            self.config.minimum_score
            if minimum_score is None
            else minimum_score
        )

        if threshold < 0:
            raise ValueError(
                "minimum_score doit être positif ou nul."
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

        query_tokens = technical_tokenize(
            cleaned_query
        )

        if not query_tokens:
            raise ValueError(
                "Aucun token exploitable dans la requête."
            )

        retrieval_k = (
            self.total_documents
            if normalized_document_ids or normalized_chunk_ids
            else min(top_k, self.total_documents)
        )

        start_time = time.perf_counter()

        raw_results = self.model.retrieve(
            [query_tokens],
            k=retrieval_k,
            sorted=True,
            return_as="tuple",
            show_progress=False,
        )

        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        lexical_ids = raw_results.documents[0]
        scores = raw_results.scores[0]

        results: list[BM25SearchResult] = []

        for lexical_id_value, score_value in zip(
            lexical_ids,
            scores,
            strict=True,
        ):
            lexical_id = int(lexical_id_value)
            score = float(score_value)

            if not 0 <= lexical_id < self.total_documents:
                continue

            # Les documents sans terme commun peuvent être
            # retournés avec un score nul.
            if score <= threshold:
                continue

            chunk = self.metadata[lexical_id]

            if (
                normalized_document_ids
                and chunk.document_id not in normalized_document_ids
            ):
                continue

            if normalized_chunk_ids and chunk.chunk_id not in normalized_chunk_ids:
                continue

            results.append(
                BM25SearchResult(
                    rank=len(results) + 1,
                    lexical_id=lexical_id,
                    score=score,
                    chunk=chunk,
                )
            )

            if len(results) >= top_k:
                break

        return BM25SearchResponse(
            query=cleaned_query,
            query_tokens=query_tokens,
            top_k_requested=top_k,
            search_duration_ms=round(
                duration_ms,
                3,
            ),
            results=results,
        )

    def _check_required_files(self) -> None:
        """Contrôler la présence des artefacts indispensables."""

        required_files = [
            self.metadata_path,
            self.manifest_path,
            self.config_path,
            self.index_directory / "params.index.json",
            self.index_directory / "vocab.index.json",
            self.index_directory / "data.csc.index.npy",
            self.index_directory / "indices.csc.index.npy",
            self.index_directory / "indptr.csc.index.npy",
        ]

        missing_files = [
            path
            for path in required_files
            if not path.exists()
        ]

        if missing_files:
            formatted = "\n".join(
                f"- {path}"
                for path in missing_files
            )

            raise FileNotFoundError(
                "Artefacts BM25 manquants :\n"
                f"{formatted}"
            )

    @staticmethod
    def _load_metadata(
        path: Path,
    ) -> list[DocumentChunk]:
        """Charger la correspondance lexical_id → chunk."""

        lexical_ids: list[int] = []
        chunks: list[DocumentChunk] = []

        with path.open(
            "r",
            encoding="utf-8",
        ) as source:
            for line_number, line in enumerate(
                source,
                start=1,
            ):
                if not line.strip():
                    raise ValueError(
                        f"{path.name}, ligne "
                        f"{line_number} vide."
                    )

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{path.name}, ligne "
                        f"{line_number} : JSON invalide."
                    ) from error

                if not isinstance(record, dict):
                    raise ValueError(
                        f"{path.name}, ligne "
                        f"{line_number} : objet attendu."
                    )

                lexical_id = record.pop(
                    "lexical_id",
                    None,
                )

                if (
                    not isinstance(lexical_id, int)
                    or isinstance(lexical_id, bool)
                ):
                    raise ValueError(
                        f"{path.name}, ligne "
                        f"{line_number} : lexical_id invalide."
                    )

                try:
                    chunk = DocumentChunk.model_validate(
                        record
                    )
                except ValidationError as error:
                    raise ValueError(
                        f"{path.name}, ligne "
                        f"{line_number} : chunk invalide."
                    ) from error

                lexical_ids.append(lexical_id)
                chunks.append(chunk)

        expected_ids = list(range(len(chunks)))

        if lexical_ids != expected_ids:
            raise ValueError(
                "Les lexical_id ne sont pas continus "
                "et correctement ordonnés."
            )

        return chunks

    @staticmethod
    def _load_manifest(path: Path) -> dict[str, Any]:
        """Charger le manifest de construction."""

        try:
            manifest = json.loads(
                path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Manifest BM25 invalide : {path}"
            ) from error

        if not isinstance(manifest, dict):
            raise ValueError(
                "Le manifest BM25 doit être un objet JSON."
            )

        return manifest

    def _validate_loaded_index(self) -> None:
        """Comparer index, configuration et métadonnées."""

        indexed_documents = int(
            self.model.scores["num_docs"]
        )

        if indexed_documents != self.total_documents:
            raise ValueError(
                "Nombre de documents incohérent : "
                f"index={indexed_documents}, "
                f"métadonnées={self.total_documents}."
            )

        if self.model.method != self.config.method:
            raise ValueError(
                "La méthode BM25 de l'index ne correspond "
                "pas à retrieval.yaml."
            )

        if not abs(
            float(self.model.k1) - self.config.k1
        ) < 1e-12:
            raise ValueError(
                "Le paramètre k1 de l'index a changé."
            )

        if not abs(
            float(self.model.b) - self.config.b
        ) < 1e-12:
            raise ValueError(
                "Le paramètre b de l'index a changé."
            )

        tokenizer = self.manifest.get("tokenizer")

        if not isinstance(tokenizer, dict):
            raise ValueError(
                "Informations tokenizer absentes du manifest."
            )

        if (
            tokenizer.get("version")
            != TOKENIZER_VERSION
        ):
            raise ValueError(
                "Le tokenizer du manifest est incompatible."
            )

        chunk_ids = [
            chunk.chunk_id
            for chunk in self.metadata
        ]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                "Des chunk_id sont dupliqués dans BM25."
            )
