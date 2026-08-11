"""Validation automatique des chunks documentaires."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from phosprocess.ingestion.schemas import ParsedPage
from phosprocess.preprocessing.chunk_schemas import DocumentChunk
from phosprocess.preprocessing.chunker import (
    ChunkingConfig,
    StructureAwareChunker,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHUNKS_DIRECTORY = (
    PROJECT_ROOT / "data" / "processed" / "chunks"
)
DEFAULT_PAGES_DIRECTORY = (
    PROJECT_ROOT / "data" / "interim" / "pages"
)
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "chunking.yaml"
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunk_validation_report.json"
)


def load_chunking_config(config_path: Path) -> ChunkingConfig:
    """Lire et valider la configuration du chunking."""

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration introuvable : {config_path}"
        )

    raw_config = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )

    if not isinstance(raw_config, dict):
        raise ValueError("Configuration de chunking invalide.")

    config = ChunkingConfig(
        tokenizer_name=str(raw_config["tokenizer_name"]),
        target_tokens=int(raw_config["target_tokens"]),
        max_tokens=int(raw_config["max_tokens"]),
        overlap_tokens=int(raw_config["overlap_tokens"]),
        min_chunk_tokens=int(raw_config["min_chunk_tokens"]),
        include_document_context=bool(
            raw_config["include_document_context"]
        ),
    )

    if config.target_tokens > config.max_tokens:
        raise ValueError(
            "target_tokens ne peut pas dépasser max_tokens."
        )

    return config


def load_chunks(
    chunks_path: Path,
) -> tuple[list[DocumentChunk], list[str]]:
    """Lire et valider chaque ligne d'un fichier de chunks."""

    chunks: list[DocumentChunk] = []
    errors: list[str] = []

    with chunks_path.open("r", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                errors.append(
                    f"Ligne {line_number} vide dans le JSONL."
                )
                continue

            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(
                    f"Ligne {line_number} : JSON invalide, "
                    f"colonne {error.colno}."
                )
                continue

            try:
                chunk = DocumentChunk.model_validate(raw_record)
            except ValidationError as error:
                errors.append(
                    f"Ligne {line_number} : schéma invalide, "
                    f"{error.error_count()} erreur(s)."
                )
                continue

            chunks.append(chunk)

    return chunks, errors


def load_non_empty_source_pages(
    pages_path: Path,
) -> tuple[set[int], list[str]]:
    """Charger les numéros des pages contenant du texte."""

    page_numbers: set[int] = set()
    errors: list[str] = []

    if not pages_path.exists():
        return page_numbers, [
            f"Fichier source des pages introuvable : {pages_path.name}"
        ]

    with pages_path.open("r", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                continue

            try:
                raw_record = json.loads(line)
                page = ParsedPage.model_validate(raw_record)
            except (json.JSONDecodeError, ValidationError):
                errors.append(
                    f"Page source invalide à la ligne {line_number}."
                )
                continue

            if not page.quality.is_empty:
                page_numbers.add(page.provenance.page_number)

    return page_numbers, errors


def validate_identity(
    chunks: list[DocumentChunk],
    expected_document_id: str,
) -> tuple[list[str], list[str]]:
    """Vérifier les IDs, documents et indices des chunks."""

    errors: list[str] = []
    warnings: list[str] = []

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    duplicate_ids = sorted(
        chunk_id
        for chunk_id, count in Counter(chunk_ids).items()
        if count > 1
    )

    if duplicate_ids:
        errors.append(f"chunk_id dupliqués : {duplicate_ids}")

    document_ids = {chunk.document_id for chunk in chunks}

    if document_ids != {expected_document_id}:
        errors.append(
            f"document_id incohérents : {sorted(document_ids)}"
        )

    source_files = {chunk.source_file for chunk in chunks}

    if len(source_files) > 1:
        errors.append(
            f"Plusieurs fichiers sources détectés : {sorted(source_files)}"
        )

    indices = [chunk.chunk_index for chunk in chunks]
    expected_indices = list(range(len(chunks)))

    if indices != expected_indices:
        errors.append(
            "Les chunk_index ne forment pas une séquence continue "
            "commençant à zéro."
        )

    if len(set(chunk.text for chunk in chunks)) != len(chunks):
        warnings.append(
            "Des chunks possèdent exactement le même texte."
        )

    return errors, warnings


def validate_pages(
    chunks: list[DocumentChunk],
    source_pages: set[int],
) -> tuple[list[str], list[str], set[int]]:
    """Vérifier la cohérence et la couverture des pages sources."""

    errors: list[str] = []
    warnings: list[str] = []
    referenced_pages: set[int] = set()

    for chunk in chunks:
        pages = chunk.source_pages

        if pages != sorted(set(pages)):
            errors.append(
                f"{chunk.chunk_id} : source_pages doit être trié "
                "et sans doublons."
            )

        if chunk.page_start != min(pages):
            errors.append(
                f"{chunk.chunk_id} : page_start incohérent."
            )

        if chunk.page_end != max(pages):
            errors.append(
                f"{chunk.chunk_id} : page_end incohérent."
            )

        if chunk.page_start > chunk.page_end:
            errors.append(
                f"{chunk.chunk_id} : page_start > page_end."
            )

        referenced_pages.update(pages)

    uncovered_pages = sorted(source_pages - referenced_pages)

    if uncovered_pages:
        warnings.append(
            "Pages textuelles non représentées dans les chunks : "
            f"{uncovered_pages}"
        )

    unknown_pages = sorted(referenced_pages - source_pages)

    if unknown_pages:
        warnings.append(
            "Pages référencées sans texte source correspondant : "
            f"{unknown_pages}"
        )

    return errors, warnings, referenced_pages


def validate_tokens(
    chunks: list[DocumentChunk],
    chunker: StructureAwareChunker,
    config: ChunkingConfig,
) -> tuple[list[str], list[str]]:
    """Recalculer et vérifier les nombres de tokens."""

    errors: list[str] = []
    warnings: list[str] = []

    for chunk in chunks:
        actual_body_tokens = chunker.count_tokens(chunk.text)
        actual_total_tokens = chunker.count_tokens(
            chunk.embedding_text
        )

        if chunk.body_token_count != actual_body_tokens:
            errors.append(
                f"{chunk.chunk_id} : body_token_count="
                f"{chunk.body_token_count}, attendu={actual_body_tokens}."
            )

        if chunk.token_count != actual_total_tokens:
            errors.append(
                f"{chunk.chunk_id} : token_count="
                f"{chunk.token_count}, attendu={actual_total_tokens}."
            )

        if actual_total_tokens > config.max_tokens:
            errors.append(
                f"{chunk.chunk_id} dépasse max_tokens : "
                f"{actual_total_tokens} > {config.max_tokens}."
            )

        if actual_total_tokens < config.min_chunk_tokens:
            warnings.append(
                f"{chunk.chunk_id} est court : "
                f"{actual_total_tokens} tokens."
            )

        if chunk.text not in chunk.embedding_text:
            errors.append(
                f"{chunk.chunk_id} : le texte du chunk est absent "
                "de embedding_text."
            )

    return errors, warnings


def validate_document(
    chunks_path: Path,
    pages_directory: Path,
    chunker: StructureAwareChunker,
    config: ChunkingConfig,
) -> dict[str, Any]:
    """Valider tous les chunks d'un document."""

    document_id = chunks_path.stem.removesuffix("_chunks")
    pages_path = pages_directory / f"{document_id}_pages.jsonl"

    chunks, errors = load_chunks(chunks_path)
    warnings: list[str] = []

    if not chunks:
        errors.append("Aucun chunk valide trouvé.")

        return {
            "document_id": document_id,
            "status": "invalid",
            "total_chunks": 0,
            "errors": errors,
            "warnings": warnings,
        }

    identity_errors, identity_warnings = validate_identity(
        chunks,
        document_id,
    )
    errors.extend(identity_errors)
    warnings.extend(identity_warnings)

    source_pages, source_page_errors = load_non_empty_source_pages(
        pages_path
    )
    errors.extend(source_page_errors)

    page_errors, page_warnings, referenced_pages = validate_pages(
        chunks,
        source_pages,
    )
    errors.extend(page_errors)
    warnings.extend(page_warnings)

    token_errors, token_warnings = validate_tokens(
        chunks,
        chunker,
        config,
    )
    errors.extend(token_errors)
    warnings.extend(token_warnings)

    token_counts = [chunk.token_count for chunk in chunks]

    if errors:
        status = "invalid"
    elif warnings:
        status = "valid_with_warnings"
    else:
        status = "valid"

    return {
        "document_id": document_id,
        "source_file": chunks[0].source_file,
        "status": status,
        "total_chunks": len(chunks),
        "minimum_tokens": min(token_counts),
        "maximum_tokens": max(token_counts),
        "average_tokens": round(
            sum(token_counts) / len(token_counts),
            2,
        ),
        "source_text_pages": len(source_pages),
        "referenced_pages": len(referenced_pages),
        "errors": errors,
        "warnings": warnings,
    }


def write_report(
    report: dict[str, Any],
    report_path: Path,
) -> None:
    """Écrire le rapport sans risquer un fichier incomplet."""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = report_path.with_suffix(".json.tmp")

    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    temporary_path.replace(report_path)


def parse_arguments() -> argparse.Namespace:
    """Lire les options de la commande."""

    parser = argparse.ArgumentParser(
        description="Valider tous les chunks documentaires."
    )

    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=DEFAULT_CHUNKS_DIRECTORY,
    )

    parser.add_argument(
        "--pages-dir",
        type=Path,
        default=DEFAULT_PAGES_DIRECTORY,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )

    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Traiter les warnings comme des erreurs.",
    )

    return parser.parse_args()


def main() -> None:
    """Valider les chunks de tous les documents."""

    arguments = parse_arguments()

    chunks_directory = arguments.chunks_dir.resolve()
    pages_directory = arguments.pages_dir.resolve()
    config_path = arguments.config.resolve()
    report_path = arguments.report_path.resolve()

    config = load_chunking_config(config_path)

    print(
        f"Chargement du tokenizer : {config.tokenizer_name}"
    )
    chunker = StructureAwareChunker(config)

    chunk_files = sorted(
        chunks_directory.glob("*_chunks.jsonl")
    )

    if not chunk_files:
        raise FileNotFoundError(
            f"Aucun fichier de chunks trouvé dans {chunks_directory}"
        )

    results = [
        validate_document(
            chunks_path=chunks_path,
            pages_directory=pages_directory,
            chunker=chunker,
            config=config,
        )
        for chunks_path in chunk_files
    ]

    status_counts = Counter(
        result["status"] for result in results
    )

    all_chunk_ids: list[str] = []

    for chunks_path in chunk_files:
        chunks, _ = load_chunks(chunks_path)
        all_chunk_ids.extend(chunk.chunk_id for chunk in chunks)

    global_duplicate_ids = sorted(
        chunk_id
        for chunk_id, count in Counter(all_chunk_ids).items()
        if count > 1
    )

    report = {
        "documents_validated": len(results),
        "total_chunks": len(all_chunk_ids),
        "status_counts": dict(status_counts),
        "global_duplicate_chunk_ids": global_duplicate_ids,
        "documents": results,
    }

    write_report(report, report_path)

    print("\n=== Validation des chunks ===")

    for result in results:
        print(
            f"{result['document_id']} | "
            f"{result['status']} | "
            f"chunks={result['total_chunks']} | "
            f"errors={len(result['errors'])} | "
            f"warnings={len(result['warnings'])}"
        )

    print("\n=== Résumé ===")
    print(f"Documents : {len(results)}")
    print(f"Chunks    : {len(all_chunk_ids)}")
    print(f"Valides   : {status_counts['valid']}")
    print(
        "Warnings  : "
        f"{status_counts['valid_with_warnings']}"
    )
    print(f"Invalides : {status_counts['invalid']}")
    print(f"Rapport   : {report_path}")

    has_errors = (
        status_counts["invalid"] > 0
        or bool(global_duplicate_ids)
    )

    has_strict_warnings = (
        arguments.strict
        and status_counts["valid_with_warnings"] > 0
    )

    if has_errors or has_strict_warnings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()