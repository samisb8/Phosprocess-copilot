"""Construire automatiquement les chunks de tous les documents."""

import json
import statistics
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from phosprocess.ingestion.schemas import ParsedPage
from phosprocess.preprocessing.chunker import (
    ChunkingConfig,
    StructureAwareChunker,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAGES_DIRECTORY = PROJECT_ROOT / "data" / "interim" / "pages"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "chunks"
REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunking_report.json"
)
CONFIG_PATH = PROJECT_ROOT / "configs" / "chunking.yaml"


def load_config() -> tuple[ChunkingConfig, dict[str, Any]]:
    """Lire et valider la configuration YAML."""

    raw_config = yaml.safe_load(
        CONFIG_PATH.read_text(encoding="utf-8")
    )

    if not isinstance(raw_config, dict):
        raise ValueError("La configuration de chunking est invalide.")

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

    if config.overlap_tokens >= config.target_tokens:
        raise ValueError(
            "overlap_tokens doit être inférieur à target_tokens."
        )

    return config, raw_config


def load_pages(pages_path: Path) -> list[ParsedPage]:
    """Lire et valider toutes les pages d'un JSONL."""

    pages: list[ParsedPage] = []

    with pages_path.open("r", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                pages.append(ParsedPage.model_validate(record))
            except (json.JSONDecodeError, ValidationError) as error:
                raise ValueError(
                    f"{pages_path.name}, ligne {line_number} invalide."
                ) from error

    return pages


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Écrire atomiquement les chunks en JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".jsonl.tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as output_file:
        for record in records:
            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    temporary_path.replace(path)


def main() -> None:
    """Construire les chunks de tous les documents validés."""

    config, raw_config = load_config()
    chunker = StructureAwareChunker(config)

    page_files = sorted(
        PAGES_DIRECTORY.glob("*_pages.jsonl")
    )

    if not page_files:
        raise FileNotFoundError(
            f"Aucun JSONL trouvé dans {PAGES_DIRECTORY}"
        )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    document_results: list[dict[str, Any]] = []

    for pages_path in page_files:
        print(f"\n[START] {pages_path.name}")

        pages = load_pages(pages_path)
        chunks = chunker.chunk_document(pages)

        if not chunks:
            print("[WARNING] Aucun chunk créé.")
            continue

        document_id = chunks[0].document_id
        output_path = OUTPUT_DIRECTORY / f"{document_id}_chunks.jsonl"

        save_jsonl(
            [chunk.model_dump(mode="json") for chunk in chunks],
            output_path,
        )

        token_counts = [chunk.token_count for chunk in chunks]

        over_limit = [
            chunk.chunk_id
            for chunk in chunks
            if chunk.token_count > config.max_tokens
        ]

        result = {
            "document_id": document_id,
            "source_file": chunks[0].source_file,
            "total_pages": len(pages),
            "total_chunks": len(chunks),
            "minimum_tokens": min(token_counts),
            "maximum_tokens": max(token_counts),
            "average_tokens": round(
                statistics.mean(token_counts),
                2,
            ),
            "median_tokens": round(
                statistics.median(token_counts),
                2,
            ),
            "chunks_over_limit": over_limit,
            "output_file": str(output_path),
        }

        document_results.append(result)

        print(
            f"[OK] chunks={len(chunks)} | "
            f"moyenne={result['average_tokens']} tokens | "
            f"maximum={result['maximum_tokens']}"
        )

    report = {
        "configuration": raw_config,
        "documents_processed": len(document_results),
        "documents": document_results,
    }

    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== Résumé du chunking ===")
    print(f"Documents : {len(document_results)}")
    print(
        "Chunks    : "
        f"{sum(item['total_chunks'] for item in document_results)}"
    )
    print(f"Rapport   : {REPORT_PATH}")


if __name__ == "__main__":
    main()