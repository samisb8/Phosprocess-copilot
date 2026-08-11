"""Construire l'index dense exact BGE-M3 + FAISS."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import yaml
from pydantic import ValidationError

from phosprocess.embeddings.embedder import (
    BGEEmbedder,
    load_embedding_config,
)
from phosprocess.preprocessing.chunk_schemas import DocumentChunk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "embeddings.yaml"


def load_raw_config(config_path: Path) -> dict[str, Any]:
    """Lire toute la configuration YAML."""

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration introuvable : {config_path}"
        )

    raw_config = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )

    if not isinstance(raw_config, dict):
        raise ValueError("La configuration YAML est invalide.")

    return raw_config


def resolve_project_path(path_value: str) -> Path:
    """Résoudre un chemin relativement à la racine du projet."""

    path = Path(path_value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def sha256_file(path: Path) -> str:
    """Calculer l'empreinte SHA-256 d'un fichier."""

    digest = hashlib.sha256()

    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def load_chunks(
    chunks_directory: Path,
) -> tuple[list[DocumentChunk], list[dict[str, Any]]]:
    """Charger tous les chunks finaux dans un ordre déterministe."""

    chunk_files = sorted(
        chunks_directory.glob("*_chunks.jsonl")
    )

    if not chunk_files:
        raise FileNotFoundError(
            f"Aucun fichier de chunks trouvé dans "
            f"{chunks_directory}"
        )

    all_chunks: list[DocumentChunk] = []
    source_files: list[dict[str, Any]] = []

    for chunks_path in chunk_files:
        expected_document_id = (
            chunks_path.stem.removesuffix("_chunks")
        )

        document_chunks: list[DocumentChunk] = []

        with chunks_path.open(
            "r",
            encoding="utf-8",
        ) as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise ValueError(
                        f"{chunks_path.name}, ligne "
                        f"{line_number} vide."
                    )

                try:
                    chunk = DocumentChunk.model_validate_json(
                        line
                    )
                except ValidationError as error:
                    raise ValueError(
                        f"{chunks_path.name}, ligne "
                        f"{line_number} invalide."
                    ) from error

                if chunk.document_id != expected_document_id:
                    raise ValueError(
                        f"{chunks_path.name}, ligne "
                        f"{line_number} : document_id "
                        f"incohérent."
                    )

                document_chunks.append(chunk)

        if not document_chunks:
            raise ValueError(
                f"Aucun chunk dans {chunks_path.name}."
            )

        expected_indices = list(
            range(len(document_chunks))
        )
        actual_indices = [
            chunk.chunk_index
            for chunk in document_chunks
        ]

        if actual_indices != expected_indices:
            raise ValueError(
                f"{expected_document_id} : chunk_index "
                "non continu."
            )

        all_chunks.extend(document_chunks)

        source_files.append(
            {
                "document_id": expected_document_id,
                "path": str(chunks_path),
                "sha256": sha256_file(chunks_path),
                "chunk_count": len(document_chunks),
            }
        )

        print(
            f"[LOAD] {expected_document_id} : "
            f"{len(document_chunks)} chunks"
        )

    validate_global_chunk_ids(all_chunks)

    return all_chunks, source_files


def validate_global_chunk_ids(
    chunks: list[DocumentChunk],
) -> None:
    """Vérifier l'unicité globale des identifiants."""

    counts = Counter(chunk.chunk_id for chunk in chunks)

    duplicates = sorted(
        chunk_id
        for chunk_id, count in counts.items()
        if count > 1
    )

    if duplicates:
        raise ValueError(
            f"chunk_id dupliqués : {duplicates}"
        )


def validate_embedding_matrix(
    vectors: np.ndarray,
    *,
    expected_rows: int,
    expected_dimension: int,
    normalized: bool,
) -> np.ndarray:
    """Valider la matrice avant son insertion dans FAISS."""

    expected_shape = (
        expected_rows,
        expected_dimension,
    )

    if vectors.shape != expected_shape:
        raise ValueError(
            f"Forme des embeddings : {vectors.shape}, "
            f"attendu : {expected_shape}."
        )

    if vectors.dtype != np.float32:
        vectors = vectors.astype(
            np.float32,
            copy=False,
        )

    if not np.isfinite(vectors).all():
        raise ValueError(
            "La matrice contient NaN ou une valeur infinie."
        )

    norms = np.linalg.norm(vectors, axis=1)

    if np.any(norms <= 0):
        raise ValueError(
            "Au moins un embedding possède une norme nulle."
        )

    if normalized and not np.allclose(
        norms,
        1.0,
        atol=1e-4,
    ):
        raise ValueError(
            "Les embeddings ne sont pas normalisés."
        )

    return np.ascontiguousarray(
        vectors,
        dtype=np.float32,
    )


def build_faiss_index(
    vectors: np.ndarray,
    dimension: int,
) -> faiss.Index:
    """Créer un index exact par produit scalaire."""

    index = faiss.IndexFlatIP(dimension)

    if not index.is_trained:
        raise RuntimeError(
            "IndexFlatIP devrait être immédiatement entraîné."
        )

    index.add(vectors)

    if index.ntotal != vectors.shape[0]:
        raise RuntimeError(
            f"FAISS contient {index.ntotal} vecteurs, "
            f"mais {vectors.shape[0]} étaient attendus."
        )

    return index


def write_metadata(
    chunks: list[DocumentChunk],
    path: Path,
) -> None:
    """Écrire la correspondance vecteur → chunk."""

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as output:
        for vector_id, chunk in enumerate(chunks):
            record = {
                "vector_id": vector_id,
                **chunk.model_dump(mode="json"),
            }

            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def atomic_write_json(
    data: dict[str, Any],
    path: Path,
) -> None:
    """Écrire un JSON atomiquement."""

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(path)


def remove_stale_temporary_files(
    paths: list[Path],
) -> None:
    """Retirer les fichiers temporaires d'une ancienne exécution."""

    for path in paths:
        if path.exists():
            path.unlink()


def main() -> None:
    """Encoder les chunks et construire l'index dense."""

    raw_config = load_raw_config(CONFIG_PATH)
    embedding_config = load_embedding_config(CONFIG_PATH)

    data_config = raw_config.get("data")
    index_config = raw_config.get("index")

    if not isinstance(data_config, dict):
        raise ValueError(
            "La section 'data' est absente ou invalide."
        )

    if not isinstance(index_config, dict):
        raise ValueError(
            "La section 'index' est absente ou invalide."
        )

    index_type = str(index_config["type"])
    metric = str(index_config["metric"])

    if index_type != "IndexFlatIP":
        raise ValueError(
            "Ce pipeline prend uniquement en charge "
            "IndexFlatIP."
        )

    if metric != "inner_product":
        raise ValueError(
            "IndexFlatIP exige metric=inner_product."
        )

    chunks_directory = resolve_project_path(
        str(data_config["chunks_directory"])
    )

    output_directory = resolve_project_path(
        str(index_config["output_directory"])
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_path = (
        output_directory
        / str(index_config["index_filename"])
    )
    metadata_path = (
        output_directory
        / str(index_config["metadata_filename"])
    )
    manifest_path = (
        output_directory
        / str(index_config["manifest_filename"])
    )

    temporary_index_path = index_path.with_suffix(
        index_path.suffix + ".tmp"
    )
    temporary_metadata_path = metadata_path.with_suffix(
        metadata_path.suffix + ".tmp"
    )

    remove_stale_temporary_files(
        [
            temporary_index_path,
            temporary_metadata_path,
        ]
    )

    print("\n=== Chargement des chunks ===")

    chunks, source_files = load_chunks(
        chunks_directory
    )

    texts = [
        chunk.embedding_text
        for chunk in chunks
    ]

    print("\n=== Génération des embeddings ===")
    print(f"Chunks       : {len(chunks)}")
    print(f"Modèle       : {embedding_config.model_name}")
    print(
        f"Dimension    : "
        f"{embedding_config.embedding_dimension}"
    )

    embedding_start = time.perf_counter()

    embedder = BGEEmbedder(embedding_config)

    vectors = embedder.embed_documents(texts)

    embedding_duration = (
        time.perf_counter() - embedding_start
    )

    vectors = validate_embedding_matrix(
        vectors,
        expected_rows=len(chunks),
        expected_dimension=(
            embedding_config.embedding_dimension
        ),
        normalized=(
            embedding_config.normalize_embeddings
        ),
    )

    norms = np.linalg.norm(vectors, axis=1)

    print(
        f"Embeddings   : {vectors.shape}"
    )
    print(
        f"Device       : {embedder.device}"
    )
    print(
        f"Durée        : {embedding_duration:.2f} s"
    )
    print(
        f"Normes       : "
        f"{norms.min():.6f} → {norms.max():.6f}"
    )

    print("\n=== Construction de l'index FAISS ===")

    index_start = time.perf_counter()

    index = build_faiss_index(
        vectors=vectors,
        dimension=embedding_config.embedding_dimension,
    )

    index_duration = time.perf_counter() - index_start

    faiss.write_index(
        index,
        str(temporary_index_path),
    )

    write_metadata(
        chunks,
        temporary_metadata_path,
    )

    index_sha256 = sha256_file(
        temporary_index_path
    )
    metadata_sha256 = sha256_file(
        temporary_metadata_path
    )

    # Les artefacts principaux sont publiés avant le manifest.
    # Le manifest constitue ainsi le marqueur d'une exécution terminée.
    temporary_index_path.replace(index_path)
    temporary_metadata_path.replace(metadata_path)

    document_counts = Counter(
        chunk.document_id
        for chunk in chunks
    )

    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "pipeline_version": str(
            raw_config.get(
                "pipeline_version",
                "unknown",
            )
        ),
        "model": {
            "name": embedding_config.model_name,
            "dimension": (
                embedding_config.embedding_dimension
            ),
            "device": embedder.device,
            "use_fp16": (
                embedding_config.use_fp16
                and embedder.device.startswith("cuda")
            ),
            "normalize_embeddings": (
                embedding_config.normalize_embeddings
            ),
            "passage_max_length": (
                embedding_config.passage_max_length
            ),
            "batch_size": embedding_config.batch_size,
        },
        "index": {
            "library": "faiss",
            "type": index_type,
            "metric": metric,
            "dimension": index.d,
            "total_vectors": index.ntotal,
            "is_trained": bool(index.is_trained),
        },
        "corpus": {
            "total_documents": len(document_counts),
            "total_chunks": len(chunks),
            "chunks_per_document": dict(document_counts),
            "source_files": source_files,
        },
        "quality": {
            "finite_values": bool(
                np.isfinite(vectors).all()
            ),
            "minimum_l2_norm": float(norms.min()),
            "maximum_l2_norm": float(norms.max()),
            "average_l2_norm": float(norms.mean()),
        },
        "timings_seconds": {
            "embedding": round(
                embedding_duration,
                4,
            ),
            "index_construction": round(
                index_duration,
                4,
            ),
            "total": round(
                embedding_duration + index_duration,
                4,
            ),
        },
        "artifacts": {
            "index": {
                "path": str(index_path),
                "sha256": index_sha256,
                "size_bytes": index_path.stat().st_size,
            },
            "metadata": {
                "path": str(metadata_path),
                "sha256": metadata_sha256,
                "size_bytes": metadata_path.stat().st_size,
            },
        },
        "configuration": {
            "path": str(CONFIG_PATH),
            "sha256": sha256_file(CONFIG_PATH),
        },
    }

    atomic_write_json(
        manifest,
        manifest_path,
    )

    print(f"Type         : {index_type}")
    print(f"Dimension    : {index.d}")
    print(f"Vecteurs     : {index.ntotal}")
    print(
        f"Construction : {index_duration:.4f} s"
    )

    print("\n=== Index dense construit ===")
    print(f"Index        : {index_path}")
    print(f"Métadonnées  : {metadata_path}")
    print(f"Manifest     : {manifest_path}")


if __name__ == "__main__":
    main()