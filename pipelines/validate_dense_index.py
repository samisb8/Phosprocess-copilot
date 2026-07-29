"""Valider l'intégrité de l'index dense FAISS."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from pydantic import ValidationError

from phosprocess.preprocessing.chunk_schemas import DocumentChunk

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INDEX_DIRECTORY = (
    PROJECT_ROOT / "data" / "indexes" / "dense" / "bge_m3"
)

DEFAULT_REPORT_PATH = (
    DEFAULT_INDEX_DIRECTORY / "validation_report.json"
)

INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.jsonl"
MANIFEST_FILENAME = "manifest.json"


def sha256_file(path: Path) -> str:
    """Calculer l'empreinte SHA-256 d'un fichier."""

    digest = hashlib.sha256()

    with path.open("rb") as source:
        for block in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    """Charger un fichier JSON devant contenir un objet."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"JSON invalide dans {path.name}, "
            f"ligne {error.lineno}, colonne {error.colno}."
        ) from error

    if not isinstance(data, dict):
        raise ValueError(
            f"{path.name} doit contenir un objet JSON."
        )

    return data


def load_metadata(
    metadata_path: Path,
) -> tuple[list[int], list[DocumentChunk]]:
    """Lire et valider toutes les métadonnées vectorielles."""

    vector_ids: list[int] = []
    chunks: list[DocumentChunk] = []

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as source:
        for line_number, line in enumerate(source, start=1):
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
                    f"{line_number} : objet JSON attendu."
                )

            vector_id = record.get("vector_id")

            if (
                not isinstance(vector_id, int)
                or isinstance(vector_id, bool)
                or vector_id < 0
            ):
                raise ValueError(
                    f"{metadata_path.name}, ligne "
                    f"{line_number} : vector_id invalide."
                )

            chunk_record = {
                key: value
                for key, value in record.items()
                if key != "vector_id"
            }

            try:
                chunk = DocumentChunk.model_validate(
                    chunk_record
                )
            except ValidationError as error:
                raise ValueError(
                    f"{metadata_path.name}, ligne "
                    f"{line_number} : chunk invalide."
                ) from error

            vector_ids.append(vector_id)
            chunks.append(chunk)

    return vector_ids, chunks


def reconstruct_vectors(index: Any) -> np.ndarray:
    """Reconstruire les vecteurs stockés dans IndexFlatIP."""

    total_vectors = int(index.ntotal)
    dimension = int(index.d)

    vectors = np.empty(
        (total_vectors, dimension),
        dtype=np.float32,
    )

    for vector_id in range(total_vectors):
        vector = np.asarray(
            index.reconstruct(vector_id),
            dtype=np.float32,
        )

        if vector.shape != (dimension,):
            raise ValueError(
                f"Vecteur {vector_id} : forme {vector.shape}, "
                f"attendu ({dimension},)."
            )

        vectors[vector_id] = vector

    return np.ascontiguousarray(
        vectors,
        dtype=np.float32,
    )


def validate_metadata_identity(
    vector_ids: list[int],
    chunks: list[DocumentChunk],
    errors: list[str],
) -> None:
    """Vérifier IDs, ordre et indices des métadonnées."""

    expected_vector_ids = list(range(len(chunks)))

    if vector_ids != expected_vector_ids:
        errors.append(
            "Les vector_id ne forment pas une séquence "
            "continue commençant à zéro."
        )

    chunk_id_counts = Counter(
        chunk.chunk_id for chunk in chunks
    )

    duplicate_chunk_ids = sorted(
        chunk_id
        for chunk_id, count in chunk_id_counts.items()
        if count > 1
    )

    if duplicate_chunk_ids:
        errors.append(
            "chunk_id dupliqués dans les métadonnées : "
            f"{duplicate_chunk_ids}"
        )

    indices_per_document: defaultdict[str, list[int]] = (
        defaultdict(list)
    )

    for chunk in chunks:
        indices_per_document[chunk.document_id].append(
            chunk.chunk_index
        )

    for document_id, indices in indices_per_document.items():
        expected_indices = list(range(len(indices)))

        if indices != expected_indices:
            errors.append(
                f"{document_id} : chunk_index non continu "
                "ou mal ordonné."
            )


def validate_manifest_date(
    manifest: dict[str, Any],
    warnings: list[str],
) -> None:
    """Vérifier que la date du manifest est lisible."""

    created_at = manifest.get("created_at_utc")

    if not isinstance(created_at, str):
        warnings.append(
            "created_at_utc est absent ou invalide."
        )
        return

    try:
        datetime.fromisoformat(created_at)
    except ValueError:
        warnings.append(
            f"created_at_utc n'est pas une date ISO valide : "
            f"{created_at}"
        )


def validate_file_artifact(
    *,
    artifact_name: str,
    actual_path: Path,
    manifest: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    """Comparer un artefact au hash et à la taille du manifest."""

    result: dict[str, Any] = {
        "path": str(actual_path),
        "exists": actual_path.exists(),
    }

    if not actual_path.exists():
        errors.append(
            f"Artefact introuvable : {actual_path}"
        )
        return result

    actual_hash = sha256_file(actual_path)
    actual_size = actual_path.stat().st_size

    result["sha256"] = actual_hash
    result["size_bytes"] = actual_size

    artifacts = manifest.get("artifacts")

    if not isinstance(artifacts, dict):
        errors.append(
            "La section artifacts est absente du manifest."
        )
        return result

    artifact_manifest = artifacts.get(artifact_name)

    if not isinstance(artifact_manifest, dict):
        errors.append(
            f"Artefact {artifact_name} absent du manifest."
        )
        return result

    expected_hash = artifact_manifest.get("sha256")
    expected_size = artifact_manifest.get("size_bytes")

    result["expected_sha256"] = expected_hash
    result["expected_size_bytes"] = expected_size

    if expected_hash != actual_hash:
        errors.append(
            f"SHA-256 incorrect pour {artifact_name}."
        )

    if expected_size != actual_size:
        errors.append(
            f"Taille incorrecte pour {artifact_name} : "
            f"{actual_size}, attendu {expected_size}."
        )

    return result


def resolve_source_chunks_path(
    source_record: dict[str, Any],
) -> Path | None:
    """Retrouver un fichier source même si le projet a été déplacé."""

    stored_path = source_record.get("path")

    if isinstance(stored_path, str):
        candidate = Path(stored_path)

        if candidate.exists():
            return candidate

    document_id = source_record.get("document_id")

    if not isinstance(document_id, str):
        return None

    inferred_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "final_chunks"
        / f"{document_id}_chunks.jsonl"
    )

    return inferred_path if inferred_path.exists() else None


def validate_source_chunk_files(
    manifest: dict[str, Any],
    metadata_counts: Counter[str],
    errors: list[str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Vérifier les fichiers ayant servi à construire l'index."""

    corpus = manifest.get("corpus")

    if not isinstance(corpus, dict):
        errors.append(
            "La section corpus est absente du manifest."
        )
        return []

    source_files = corpus.get("source_files")

    if not isinstance(source_files, list):
        errors.append(
            "corpus.source_files est absent ou invalide."
        )
        return []

    results: list[dict[str, Any]] = []

    for source_record in source_files:
        if not isinstance(source_record, dict):
            errors.append(
                "Une entrée de corpus.source_files est invalide."
            )
            continue

        document_id = source_record.get("document_id")
        expected_hash = source_record.get("sha256")
        expected_count = source_record.get("chunk_count")

        result: dict[str, Any] = {
            "document_id": document_id,
            "expected_chunk_count": expected_count,
        }

        source_path = resolve_source_chunks_path(source_record)

        if source_path is None:
            errors.append(
                f"Fichier de chunks source introuvable : "
                f"{document_id}"
            )
            results.append(result)
            continue

        actual_hash = sha256_file(source_path)
        actual_count = metadata_counts.get(
            str(document_id),
            0,
        )

        result.update(
            {
                "path": str(source_path),
                "sha256": actual_hash,
                "metadata_chunk_count": actual_count,
            }
        )

        if actual_hash != expected_hash:
            errors.append(
                f"{document_id} : le fichier de chunks a "
                "changé depuis la construction de l'index."
            )

        if actual_count != expected_count:
            errors.append(
                f"{document_id} : {actual_count} métadonnées, "
                f"mais {expected_count} attendues."
            )

        results.append(result)

    manifest_counts = corpus.get("chunks_per_document")

    if isinstance(manifest_counts, dict):
        normalized_manifest_counts = {
            str(key): int(value)
            for key, value in manifest_counts.items()
        }

        if dict(metadata_counts) != normalized_manifest_counts:
            errors.append(
                "La répartition des chunks par document ne "
                "correspond pas au manifest."
            )
    else:
        warnings.append(
            "chunks_per_document absent du manifest."
        )

    return results


def validate_configuration(
    manifest: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    """Vérifier que la configuration n'a pas changé."""

    configuration = manifest.get("configuration")

    if not isinstance(configuration, dict):
        warnings.append(
            "La configuration est absente du manifest."
        )
        return {}

    expected_hash = configuration.get("sha256")
    stored_path = configuration.get("path")

    candidates: list[Path] = []

    if isinstance(stored_path, str):
        candidates.append(Path(stored_path))

    candidates.append(
        PROJECT_ROOT / "configs" / "embeddings.yaml"
    )

    config_path = next(
        (
            candidate
            for candidate in candidates
            if candidate.exists()
        ),
        None,
    )

    if config_path is None:
        warnings.append(
            "Le fichier configs/embeddings.yaml "
            "est introuvable."
        )
        return {
            "exists": False,
        }

    actual_hash = sha256_file(config_path)

    if actual_hash != expected_hash:
        errors.append(
            "configs/embeddings.yaml a changé depuis la "
            "construction de l'index."
        )

    return {
        "exists": True,
        "path": str(config_path),
        "sha256": actual_hash,
        "expected_sha256": expected_hash,
    }


def validate_index_structure(
    *,
    index: Any,
    manifest: dict[str, Any],
    metadata_count: int,
    errors: list[str],
) -> dict[str, Any]:
    """Vérifier la structure et les dimensions FAISS."""

    index_type = type(index).__name__
    total_vectors = int(index.ntotal)
    dimension = int(index.d)
    is_trained = bool(index.is_trained)
    metric_type = int(index.metric_type)

    if index_type != "IndexFlatIP":
        errors.append(
            f"Type FAISS inattendu : {index_type}."
        )

    if metric_type != faiss.METRIC_INNER_PRODUCT:
        errors.append(
            "La métrique FAISS n'est pas inner product."
        )

    if not is_trained:
        errors.append(
            "L'index FAISS n'est pas entraîné."
        )

    if total_vectors != metadata_count:
        errors.append(
            f"FAISS contient {total_vectors} vecteurs, mais "
            f"{metadata_count} métadonnées ont été trouvées."
        )

    manifest_index = manifest.get("index")

    if not isinstance(manifest_index, dict):
        errors.append(
            "La section index est absente du manifest."
        )
    else:
        comparisons = {
            "type": index_type,
            "dimension": dimension,
            "total_vectors": total_vectors,
            "is_trained": is_trained,
        }

        for field, actual_value in comparisons.items():
            expected_value = manifest_index.get(field)

            if expected_value != actual_value:
                errors.append(
                    f"Manifest index.{field}={expected_value}, "
                    f"mais la valeur réelle est {actual_value}."
                )

        if manifest_index.get("metric") != "inner_product":
            errors.append(
                "Le manifest ne déclare pas "
                "metric=inner_product."
            )

    return {
        "type": index_type,
        "dimension": dimension,
        "total_vectors": total_vectors,
        "is_trained": is_trained,
        "metric_type": metric_type,
    }


def validate_vectors(
    *,
    index: Any,
    manifest: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    """Vérifier valeurs, normes et recherche exacte."""

    vectors = reconstruct_vectors(index)

    if vectors.ndim != 2:
        errors.append(
            f"Matrice reconstruite invalide : {vectors.shape}."
        )
        return {}

    finite_values = bool(np.isfinite(vectors).all())

    if not finite_values:
        errors.append(
            "L'index contient NaN ou une valeur infinie."
        )

    norms = np.linalg.norm(vectors, axis=1)

    minimum_norm = float(norms.min())
    maximum_norm = float(norms.max())
    average_norm = float(norms.mean())

    if np.any(norms <= 0):
        errors.append(
            "Au moins un vecteur possède une norme nulle."
        )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-4,
    ):
        errors.append(
            "Les vecteurs FAISS ne sont pas correctement "
            "normalisés."
        )

    scores, neighbors = index.search(vectors, 1)

    if scores.shape != (index.ntotal, 1):
        errors.append(
            f"Forme des scores inattendue : {scores.shape}."
        )

    if neighbors.shape != (index.ntotal, 1):
        errors.append(
            "Forme des identifiants retournés inattendue : "
            f"{neighbors.shape}."
        )

    valid_neighbor_ids = (
        (neighbors >= 0)
        & (neighbors < index.ntotal)
    )

    if not bool(valid_neighbor_ids.all()):
        errors.append(
            "FAISS a retourné un identifiant hors limites."
        )

    self_scores = scores[:, 0]

    if not np.allclose(
        self_scores,
        1.0,
        atol=1e-3,
    ):
        errors.append(
            "La recherche d'un vecteur par lui-même ne "
            "retourne pas un score proche de 1."
        )

    manifest_quality = manifest.get("quality")

    if isinstance(manifest_quality, dict):
        expected_values = {
            "minimum_l2_norm": minimum_norm,
            "maximum_l2_norm": maximum_norm,
            "average_l2_norm": average_norm,
        }

        for field, actual_value in expected_values.items():
            expected_value = manifest_quality.get(field)

            if not isinstance(
                expected_value,
                (int, float),
            ):
                errors.append(
                    f"quality.{field} absent ou invalide."
                )
                continue

            if not np.isclose(
                float(expected_value),
                actual_value,
                atol=1e-5,
            ):
                errors.append(
                    f"quality.{field} ne correspond pas "
                    "aux vecteurs reconstruits."
                )
    else:
        errors.append(
            "La section quality est absente du manifest."
        )

    return {
        "shape": list(vectors.shape),
        "dtype": str(vectors.dtype),
        "finite_values": finite_values,
        "minimum_l2_norm": minimum_norm,
        "maximum_l2_norm": maximum_norm,
        "average_l2_norm": average_norm,
        "minimum_self_search_score": float(
            self_scores.min()
        ),
        "maximum_self_search_score": float(
            self_scores.max()
        ),
    }


def atomic_write_json(
    data: dict[str, Any],
    path: Path,
) -> None:
    """Écrire le rapport de manière atomique."""

    path.parent.mkdir(parents=True, exist_ok=True)

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


def parse_arguments() -> argparse.Namespace:
    """Lire les arguments de la commande."""

    parser = argparse.ArgumentParser(
        description="Valider un index dense FAISS."
    )

    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIRECTORY,
    )

    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )

    return parser.parse_args()


def main() -> None:
    """Exécuter toute la validation."""

    arguments = parse_arguments()

    index_directory = arguments.index_dir.resolve()
    report_path = arguments.report_path.resolve()

    index_path = index_directory / INDEX_FILENAME
    metadata_path = index_directory / METADATA_FILENAME
    manifest_path = index_directory / MANIFEST_FILENAME

    errors: list[str] = []
    warnings: list[str] = []

    required_paths = [
        index_path,
        metadata_path,
        manifest_path,
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        report = {
            "status": "invalid",
            "errors": [
                f"Fichier introuvable : {path}"
                for path in missing_paths
            ],
            "warnings": [],
        }

        atomic_write_json(report, report_path)

        for error in report["errors"]:
            print(f"[ERROR] {error}")

        raise SystemExit(1)

    print("\n=== Chargement des artefacts ===")

    try:
        manifest = load_json_object(manifest_path)
        vector_ids, chunks = load_metadata(metadata_path)
        index = faiss.read_index(str(index_path))
    except Exception as error:
        report = {
            "status": "invalid",
            "errors": [str(error)],
            "warnings": [],
        }

        atomic_write_json(report, report_path)
        print(f"[ERROR] {error}")
        raise SystemExit(1) from error

    print(f"Index        : {index_path}")
    print(f"Métadonnées  : {metadata_path}")
    print(f"Manifest     : {manifest_path}")

    validate_manifest_date(manifest, warnings)

    index_artifact = validate_file_artifact(
        artifact_name="index",
        actual_path=index_path,
        manifest=manifest,
        errors=errors,
    )

    metadata_artifact = validate_file_artifact(
        artifact_name="metadata",
        actual_path=metadata_path,
        manifest=manifest,
        errors=errors,
    )

    validate_metadata_identity(
        vector_ids,
        chunks,
        errors,
    )

    index_statistics = validate_index_structure(
        index=index,
        manifest=manifest,
        metadata_count=len(chunks),
        errors=errors,
    )

    vector_statistics = validate_vectors(
        index=index,
        manifest=manifest,
        errors=errors,
    )

    metadata_counts = Counter(
        chunk.document_id
        for chunk in chunks
    )

    source_file_results = validate_source_chunk_files(
        manifest=manifest,
        metadata_counts=metadata_counts,
        errors=errors,
        warnings=warnings,
    )

    configuration_result = validate_configuration(
        manifest,
        errors,
        warnings,
    )

    corpus = manifest.get("corpus")

    if isinstance(corpus, dict):
        expected_total_chunks = corpus.get("total_chunks")
        expected_total_documents = corpus.get(
            "total_documents"
        )

        if expected_total_chunks != len(chunks):
            errors.append(
                f"Le manifest annonce "
                f"{expected_total_chunks} chunks, "
                f"mais {len(chunks)} ont été chargés."
            )

        if expected_total_documents != len(metadata_counts):
            errors.append(
                f"Le manifest annonce "
                f"{expected_total_documents} documents, "
                f"mais {len(metadata_counts)} ont été trouvés."
            )
    else:
        errors.append(
            "La section corpus est absente du manifest."
        )

    status = (
        "invalid"
        if errors
        else "valid_with_warnings"
        if warnings
        else "valid"
    )

    report: dict[str, Any] = {
        "status": status,
        "index_directory": str(index_directory),
        "documents": len(metadata_counts),
        "metadata_records": len(chunks),
        "chunks_per_document": dict(metadata_counts),
        "index": index_statistics,
        "vectors": vector_statistics,
        "artifacts": {
            "index": index_artifact,
            "metadata": metadata_artifact,
        },
        "source_chunk_files": source_file_results,
        "configuration": configuration_result,
        "errors": errors,
        "warnings": warnings,
    }

    atomic_write_json(report, report_path)

    print("\n=== Validation de l'index dense ===")
    print(f"Statut       : {status}")
    print(f"Documents    : {len(metadata_counts)}")
    print(f"Métadonnées  : {len(chunks)}")
    print(f"Vecteurs     : {index.ntotal}")
    print(f"Dimension    : {index.d}")
    print(f"Erreurs      : {len(errors)}")
    print(f"Warnings     : {len(warnings)}")
    print(f"Rapport      : {report_path}")

    for error in errors:
        print(f"[ERROR] {error}")

    for warning in warnings:
        print(f"[WARNING] {warning}")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()