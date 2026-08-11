"""Pipeline automatique d'ingestion de tous les documents PDF."""

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from phosprocess.ingestion.parser_router import parse_pdf_automatically
from phosprocess.ingestion.schemas import ParsedPage

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "public"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "interim"

PIPELINE_VERSION = "0.2.0"


def calculate_sha256(file_path: Path) -> str:
    """Calculer l'empreinte SHA-256 d'un fichier."""

    digest = hashlib.sha256()

    with file_path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def create_document_id(pdf_path: Path, input_directory: Path) -> str:
    """Créer un identifiant stable à partir du chemin relatif."""

    relative_path = pdf_path.relative_to(input_directory).with_suffix("")
    raw_identifier = "__".join(relative_path.parts).lower()

    document_id = re.sub(r"[^a-z0-9]+", "_", raw_identifier)
    return document_id.strip("_")


def discover_pdf_files(input_directory: Path) -> list[Path]:
    """Détecter automatiquement tous les PDF du corpus."""

    if not input_directory.exists():
        raise FileNotFoundError(
            f"Dossier des documents introuvable : {input_directory}"
        )

    return sorted(
        path
        for path in input_directory.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


def load_existing_manifest(manifest_path: Path) -> dict[str, Any] | None:
    """Lire un manifeste existant lorsqu'il est valide."""

    if not manifest_path.exists():
        return None

    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def should_skip_document(
    *,
    manifest_path: Path,
    pages_path: Path,
    current_sha256: str,
    force: bool,
) -> bool:
    """Éviter de retraiter un document identique."""

    if force:
        return False

    manifest = load_existing_manifest(manifest_path)

    if manifest is None:
        return False

    return (
        manifest.get("status") == "success"
        and manifest.get("sha256") == current_sha256
        and manifest.get("pipeline_version") == PIPELINE_VERSION
        and pages_path.exists()
    )


def normalize_document_id(
    pages: list[ParsedPage],
    document_id: str,
) -> list[ParsedPage]:
    """Appliquer l'identifiant stable à toutes les pages."""

    normalized_pages: list[ParsedPage] = []

    for page in pages:
        provenance = page.provenance.model_copy(
            update={"document_id": document_id}
        )

        normalized_pages.append(
            page.model_copy(update={"provenance": provenance})
        )

    return normalized_pages


def save_pages_jsonl(
    pages: list[ParsedPage],
    output_path: Path,
) -> None:
    """Sauvegarder une page validée par ligne JSONL."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(".jsonl.tmp")

    with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for page in pages:
            record = page.model_dump(mode="json")

            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    temporary_path.replace(output_path)


def build_document_manifest(
    *,
    pdf_path: Path,
    input_directory: Path,
    document_id: str,
    sha256: str,
    pages: list[ParsedPage],
    elapsed_seconds: float,
    pages_path: Path,
) -> dict[str, Any]:
    """Construire le rapport technique d'un document."""

    parser_counts = Counter(
        page.provenance.parser
        for page in pages
    )

    warning_counts = Counter(
        warning
        for page in pages
        for warning in page.quality.warnings
    )

    empty_pages = [
        page.provenance.page_number
        for page in pages
        if page.quality.is_empty
    ]

    review_pages = [
        page.provenance.page_number
        for page in pages
        if page.quality.needs_review
    ]

    ocr_pages = [
        page.provenance.page_number
        for page in pages
        if page.provenance.ocr_used
    ]

    return {
        "status": "success",
        "pipeline_version": PIPELINE_VERSION,
        "document_id": document_id,
        "source_file": pdf_path.name,
        "source_relative_path": str(
            pdf_path.relative_to(input_directory)
        ),
        "sha256": sha256,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "total_pages": len(pages),
        "pages_with_text": sum(
            not page.quality.is_empty
            for page in pages
        ),
        "empty_pages": empty_pages,
        "review_pages": review_pages,
        "ocr_pages": ocr_pages,
        "parser_counts": dict(parser_counts),
        "warning_counts": dict(warning_counts),
        "output_pages_file": str(pages_path),
    }


def write_json(data: dict[str, Any], output_path: Path) -> None:
    """Écrire un fichier JSON lisible."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def process_document(
    *,
    pdf_path: Path,
    input_directory: Path,
    pages_directory: Path,
    manifests_directory: Path,
    force: bool,
) -> dict[str, Any]:
    """Traiter complètement un document PDF."""

    document_id = create_document_id(pdf_path, input_directory)

    pages_path = pages_directory / f"{document_id}_pages.jsonl"
    manifest_path = (
        manifests_directory / f"{document_id}_manifest.json"
    )

    sha256 = calculate_sha256(pdf_path)

    if should_skip_document(
        manifest_path=manifest_path,
        pages_path=pages_path,
        current_sha256=sha256,
        force=force,
    ):
        print(f"[SKIP] {pdf_path.name} — document inchangé")

        return {
            "document_id": document_id,
            "source_file": pdf_path.name,
            "status": "skipped",
        }

    print(f"\n[START] {pdf_path.name}")
    start_time = time.perf_counter()

    try:
        pages = parse_pdf_automatically(pdf_path)
        pages = normalize_document_id(pages, document_id)

        save_pages_jsonl(pages, pages_path)

        elapsed_seconds = time.perf_counter() - start_time

        manifest = build_document_manifest(
            pdf_path=pdf_path,
            input_directory=input_directory,
            document_id=document_id,
            sha256=sha256,
            pages=pages,
            elapsed_seconds=elapsed_seconds,
            pages_path=pages_path,
        )

        write_json(manifest, manifest_path)

        print(
            f"[OK] {pdf_path.name} — "
            f"{len(pages)} pages en {elapsed_seconds:.1f} s"
        )

        return {
            "document_id": document_id,
            "source_file": pdf_path.name,
            "status": "success",
            "total_pages": len(pages),
            "elapsed_seconds": round(elapsed_seconds, 3),
        }

    except Exception as error:
        elapsed_seconds = time.perf_counter() - start_time

        failure_manifest = {
            "status": "failed",
            "pipeline_version": PIPELINE_VERSION,
            "document_id": document_id,
            "source_file": pdf_path.name,
            "sha256": sha256,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

        write_json(failure_manifest, manifest_path)

        print(
            f"[ERROR] {pdf_path.name} — "
            f"{type(error).__name__}: {error}"
        )

        return {
            "document_id": document_id,
            "source_file": pdf_path.name,
            "status": "failed",
            "error": str(error),
        }


def parse_arguments() -> argparse.Namespace:
    """Lire les options de la ligne de commande."""

    parser = argparse.ArgumentParser(
        description="Ingérer automatiquement tous les PDF du corpus."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIRECTORY,
        help="Dossier contenant les PDF sources.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Dossier des résultats intermédiaires.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Retraiter les documents même s'ils sont inchangés.",
    )

    return parser.parse_args()


def main() -> None:
    """Exécuter l'ingestion batch complète."""

    arguments = parse_arguments()

    input_directory = arguments.input_dir.resolve()
    output_directory = arguments.output_dir.resolve()

    pages_directory = output_directory / "pages"
    manifests_directory = output_directory / "manifests"
    reports_directory = output_directory / "reports"

    pdf_files = discover_pdf_files(input_directory)

    if not pdf_files:
        print(f"Aucun PDF trouvé dans : {input_directory}")
        return

    print(f"{len(pdf_files)} document(s) PDF détecté(s).")

    run_started_at = datetime.now(UTC)
    results: list[dict[str, Any]] = []

    for pdf_path in pdf_files:
        result = process_document(
            pdf_path=pdf_path,
            input_directory=input_directory,
            pages_directory=pages_directory,
            manifests_directory=manifests_directory,
            force=arguments.force,
        )

        results.append(result)

    status_counts = Counter(
        result["status"]
        for result in results
    )

    run_report = {
        "pipeline_version": PIPELINE_VERSION,
        "started_at_utc": run_started_at.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "input_directory": str(input_directory),
        "documents_detected": len(pdf_files),
        "status_counts": dict(status_counts),
        "documents": results,
    }

    latest_report_path = reports_directory / "latest_ingestion_run.json"
    write_json(run_report, latest_report_path)

    print("\n=== Résumé ===")
    print(f"Documents détectés : {len(pdf_files)}")
    print(f"Réussis            : {status_counts['success']}")
    print(f"Ignorés            : {status_counts['skipped']}")
    print(f"Échoués            : {status_counts['failed']}")
    print(f"Rapport             : {latest_report_path}")


if __name__ == "__main__":
    main()
