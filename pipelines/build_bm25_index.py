"""Construire l'index lexical BM25 des chunks finaux."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import bm25s
from pydantic import ValidationError

from phosprocess.preprocessing.chunk_schemas import DocumentChunk
from phosprocess.retrieval.bm25 import (
    TOKENIZER_VERSION,
    build_lexical_text,
    load_bm25_config,
    technical_tokenize,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "retrieval.yaml"


def resolve_project_path(path_value: str) -> Path:
    """Résoudre un chemin depuis la racine du projet."""

    path = Path(path_value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


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


def load_chunks(
    chunks_directory: Path,
) -> tuple[list[DocumentChunk], list[dict[str, Any]]]:
    """Charger tous les chunks dans un ordre déterministe."""

    chunk_files = sorted(
        chunks_directory.glob("*_chunks.jsonl")
    )

    if not chunk_files:
        raise FileNotFoundError(
            f"Aucun chunk trouvé dans {chunks_directory}"
        )

    all_chunks: list[DocumentChunk] = []
    source_records: list[dict[str, Any]] = []

    for chunks_path in chunk_files:
        expected_document_id = (
            chunks_path.stem.removesuffix("_chunks")
        )

        document_chunks: list[DocumentChunk] = []

        with chunks_path.open(
            "r",
            encoding="utf-8",
        ) as source:
            for line_number, line in enumerate(
                source,
                start=1,
            ):
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

                if (
                    chunk.document_id
                    != expected_document_id
                ):
                    raise ValueError(
                        f"{chunks_path.name}, ligne "
                        f"{line_number} : document_id "
                        "incohérent."
                    )

                document_chunks.append(chunk)

        actual_indices = [
            chunk.chunk_index
            for chunk in document_chunks
        ]

        expected_indices = list(
            range(len(document_chunks))
        )

        if actual_indices != expected_indices:
            raise ValueError(
                f"{expected_document_id} : chunk_index "
                "non continu."
            )

        all_chunks.extend(document_chunks)

        source_records.append(
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

    chunk_ids = [
        chunk.chunk_id
        for chunk in all_chunks
    ]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            "Des chunk_id sont dupliqués globalement."
        )

    return all_chunks, source_records


def write_metadata(
    chunks: list[DocumentChunk],
    path: Path,
) -> None:
    """Sauvegarder lexical_id → chunk."""

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as output:
        for lexical_id, chunk in enumerate(chunks):
            record = {
                "lexical_id": lexical_id,
                **chunk.model_dump(mode="json"),
            }

            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def collect_artifacts(
    directory: Path,
    *,
    excluded_filename: str,
) -> list[dict[str, Any]]:
    """Lister les artefacts produits par BM25S."""

    artifacts: list[dict[str, Any]] = []

    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue

        if path.name == excluded_filename:
            continue

        artifacts.append(
            {
                "path": str(path.relative_to(directory)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )

    return artifacts


def atomic_write_json(
    data: dict[str, Any],
    path: Path,
) -> None:
    """Écrire un fichier JSON atomiquement."""

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


def publish_directory(
    temporary_directory: Path,
    final_directory: Path,
) -> None:
    """Publier le nouvel index en conservant l'ancien en secours."""

    backup_directory = final_directory.with_name(
        final_directory.name + ".backup"
    )

    if backup_directory.exists():
        shutil.rmtree(backup_directory)

    try:
        if final_directory.exists():
            final_directory.replace(
                backup_directory
            )

        temporary_directory.replace(
            final_directory
        )

    except Exception:
        if final_directory.exists():
            shutil.rmtree(final_directory)

        if backup_directory.exists():
            backup_directory.replace(
                final_directory
            )

        raise

    else:
        if backup_directory.exists():
            shutil.rmtree(backup_directory)


def main() -> None:
    """Construire et publier l'index BM25."""

    config = load_bm25_config(CONFIG_PATH)

    chunks_directory = resolve_project_path(
        config.chunks_directory
    )

    output_directory = resolve_project_path(
        config.output_directory
    )

    temporary_directory = output_directory.with_name(
        output_directory.name + ".tmp"
    )

    if temporary_directory.exists():
        shutil.rmtree(temporary_directory)

    temporary_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n=== Chargement des chunks ===")

    chunks, source_records = load_chunks(
        chunks_directory
    )

    lexical_texts = [
        build_lexical_text(chunk)
        for chunk in chunks
    ]

    print("\n=== Tokenisation technique ===")

    tokenized_corpus = [
        technical_tokenize(text)
        for text in lexical_texts
    ]

    empty_documents = [
        index
        for index, tokens in enumerate(tokenized_corpus)
        if not tokens
    ]

    if empty_documents:
        raise ValueError(
            "Chunks sans token lexical : "
            f"{empty_documents}"
        )

    token_counts = [
        len(tokens)
        for tokens in tokenized_corpus
    ]

    total_tokens = sum(token_counts)

    print(f"Documents     : {len(chunks)}")
    print(f"Tokens        : {total_tokens}")
    print(
        f"Moyenne       : "
        f"{total_tokens / len(chunks):.2f}"
    )
    print(f"Minimum       : {min(token_counts)}")
    print(f"Maximum       : {max(token_counts)}")

    print("\n=== Construction BM25 ===")

    start_time = time.perf_counter()

    model = bm25s.BM25(
        method=config.method,
        k1=config.k1,
        b=config.b,
        backend=config.backend,
        csc_backend=config.csc_backend,
    )

    model.index(
        tokenized_corpus,
        show_progress=True,
        leave_progress=False,
    )

    construction_duration = (
        time.perf_counter() - start_time
    )

    if int(model.scores["num_docs"]) != len(chunks):
        raise RuntimeError(
            "Le nombre de documents BM25 est incorrect."
        )

    model.save(
        str(temporary_directory),
        show_progress=False,
    )

    metadata_path = (
        temporary_directory
        / config.metadata_filename
    )

    write_metadata(
        chunks,
        metadata_path,
    )

    document_counts = Counter(
        chunk.document_id
        for chunk in chunks
    )

    manifest_path = (
        temporary_directory
        / config.manifest_filename
    )

    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "pipeline_version": config.pipeline_version,
        "library": {
            "name": "bm25s",
            "version": version("bm25s"),
        },
        "bm25": {
            "method": config.method,
            "k1": config.k1,
            "b": config.b,
            "backend": config.backend,
            "csc_backend": config.csc_backend,
        },
        "tokenizer": {
            "version": TOKENIZER_VERSION,
            "normalization": "NFKC + casefold + HTML cleanup",
            "stemming": False,
            "stopwords_removed": False,
            "preserves_examples": [
                "p2o5",
                "caso4·2h2o",
                "40:1",
                "1.8%",
                "t/m3/d",
            ],
        },
        "corpus": {
            "total_documents": len(document_counts),
            "total_chunks": len(chunks),
            "chunks_per_document": dict(
                document_counts
            ),
            "source_files": source_records,
        },
        "token_statistics": {
            "vocabulary_size": len(model.vocab_dict),
            "total_tokens": total_tokens,
            "minimum_tokens": min(token_counts),
            "maximum_tokens": max(token_counts),
            "average_tokens": round(
                total_tokens / len(chunks),
                4,
            ),
        },
        "index_statistics": {
            "documents": int(
                model.scores["num_docs"]
            ),
            "nonzero_scores": int(
                model.scores["data"].size
            ),
        },
        "timings_seconds": {
            "construction": round(
                construction_duration,
                4,
            ),
        },
        "configuration": {
            "path": str(CONFIG_PATH),
            "sha256": sha256_file(CONFIG_PATH),
        },
    }

    manifest["artifacts"] = collect_artifacts(
        temporary_directory,
        excluded_filename=config.manifest_filename,
    )

    atomic_write_json(
        manifest,
        manifest_path,
    )

    output_directory.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    publish_directory(
        temporary_directory,
        output_directory,
    )

    print("\n=== Index BM25 construit ===")
    print(f"Méthode      : {config.method}")
    print(f"k1 / b       : {config.k1} / {config.b}")
    print(f"Documents    : {len(chunks)}")
    print(f"Vocabulaire  : {len(model.vocab_dict)}")
    print(
        f"Durée        : "
        f"{construction_duration:.4f} s"
    )
    print(f"Répertoire   : {output_directory}")
    print(
        f"Métadonnées  : "
        f"{output_directory / config.metadata_filename}"
    )
    print(
        f"Manifest     : "
        f"{output_directory / config.manifest_filename}"
    )


if __name__ == "__main__":
    main()