"""Post-traiter automatiquement tous les fichiers de chunks."""

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from phosprocess.ingestion.schemas import ParsedPage
from phosprocess.preprocessing.chunk_postprocessor import (
    ChunkPostprocessingConfig,
    ChunkPostprocessor,
)
from phosprocess.preprocessing.chunk_schemas import DocumentChunk
from phosprocess.preprocessing.chunker import (
    ChunkingConfig,
    StructureAwareChunker,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_CHUNKS_DIRECTORY = (
    PROJECT_ROOT / "data" / "processed" / "chunks"
)

PAGES_DIRECTORY = (
    PROJECT_ROOT / "data" / "interim" / "pages"
)

FINAL_CHUNKS_DIRECTORY = (
    PROJECT_ROOT / "data" / "processed" / "final_chunks"
)

CHUNKING_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "chunking.yaml"
)

POSTPROCESSING_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "chunk_postprocessing.yaml"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunk_postprocessing_report.json"
)


def load_yaml(path: Path) -> dict[str, Any]:
    """Lire un fichier YAML."""

    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"Configuration YAML invalide : {path}")

    return data


def load_chunks(path: Path) -> list[DocumentChunk]:
    """Lire et valider les chunks bruts."""

    chunks: list[DocumentChunk] = []

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            try:
                chunks.append(
                    DocumentChunk.model_validate_json(line)
                )
            except ValidationError as error:
                raise ValueError(
                    f"{path.name}, ligne {line_number} invalide."
                ) from error

    return chunks


def load_pages(path: Path) -> list[ParsedPage]:
    """Lire et valider les pages sources."""

    pages: list[ParsedPage] = []

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            try:
                pages.append(
                    ParsedPage.model_validate_json(line)
                )
            except ValidationError as error:
                raise ValueError(
                    f"{path.name}, ligne {line_number} invalide."
                ) from error

    return pages


def save_jsonl(
    chunks: list[DocumentChunk],
    path: Path,
) -> None:
    """Sauvegarder atomiquement les chunks finaux."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".jsonl.tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as output:
        for chunk in chunks:
            output.write(
                chunk.model_dump_json() + "\n"
            )

    temporary_path.replace(path)


def main() -> None:
    """Post-traiter tous les documents."""

    chunking_raw = load_yaml(CHUNKING_CONFIG_PATH)
    postprocessing_raw = load_yaml(
        POSTPROCESSING_CONFIG_PATH
    )

    chunking_config = ChunkingConfig(
        tokenizer_name=str(chunking_raw["tokenizer_name"]),
        target_tokens=int(chunking_raw["target_tokens"]),
        max_tokens=int(chunking_raw["max_tokens"]),
        overlap_tokens=int(chunking_raw["overlap_tokens"]),
        min_chunk_tokens=int(
            chunking_raw["min_chunk_tokens"]
        ),
        include_document_context=bool(
            chunking_raw["include_document_context"]
        ),
    )

    postprocessing_config = ChunkPostprocessingConfig(
        remove_boilerplate=bool(
            postprocessing_raw["remove_boilerplate"]
        ),
        deduplicate_exact=bool(
            postprocessing_raw["deduplicate_exact"]
        ),
        merge_small_chunks=bool(
            postprocessing_raw["merge_small_chunks"]
        ),
        restore_uncovered_pages=bool(
            postprocessing_raw["restore_uncovered_pages"]
        ),
        min_chunk_tokens=int(
            postprocessing_raw["min_chunk_tokens"]
        ),
        max_tokens=int(postprocessing_raw["max_tokens"]),
        boilerplate_min_occurrences=int(
            postprocessing_raw["boilerplate_min_occurrences"]
        ),
        boilerplate_min_fraction=float(
            postprocessing_raw["boilerplate_min_fraction"]
        ),
        boilerplate_max_line_characters=int(
            postprocessing_raw[
                "boilerplate_max_line_characters"
            ]
        ),
        fallback_overlap_tokens=int(
            postprocessing_raw["fallback_overlap_tokens"]
        ),
    )

    print(
        f"Chargement du tokenizer : "
        f"{chunking_config.tokenizer_name}"
    )

    token_counter = StructureAwareChunker(chunking_config)

    postprocessor = ChunkPostprocessor(
        config=postprocessing_config,
        token_counter=token_counter,
    )

    chunk_files = sorted(
        RAW_CHUNKS_DIRECTORY.glob("*_chunks.jsonl")
    )

    if not chunk_files:
        raise FileNotFoundError(
            f"Aucun chunk trouvé dans {RAW_CHUNKS_DIRECTORY}"
        )

    document_reports: list[dict[str, object]] = []

    for chunks_path in chunk_files:
        document_id = chunks_path.stem.removesuffix("_chunks")
        pages_path = (
            PAGES_DIRECTORY / f"{document_id}_pages.jsonl"
        )

        print(f"\n[START] {document_id}")

        chunks = load_chunks(chunks_path)
        pages = load_pages(pages_path)

        result = postprocessor.process(
            chunks=chunks,
            pages=pages,
        )

        output_path = (
            FINAL_CHUNKS_DIRECTORY
            / f"{document_id}_chunks.jsonl"
        )

        save_jsonl(result.chunks, output_path)

        document_report = {
            "document_id": document_id,
            **result.statistics,
            "output_file": str(output_path),
        }

        document_reports.append(document_report)

        print(
            f"[OK] {result.statistics['input_chunks']} → "
            f"{result.statistics['output_chunks']} chunks | "
            f"doublons={result.statistics['duplicates_removed']} | "
            f"fusionnés="
            f"{result.statistics['small_chunks_merged']} | "
            f"pages restaurées="
            f"{result.statistics['restored_pages']}"
        )

    report = {
        "configuration": postprocessing_raw,
        "documents_processed": len(document_reports),
        "documents": document_reports,
    }

    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== Résumé ===")
    print(f"Documents : {len(document_reports)}")
    print(
        "Chunks finaux : "
        f"{sum(int(item['output_chunks']) for item in document_reports)}"
    )
    print(f"Rapport : {REPORT_PATH}")


if __name__ == "__main__":
    main()