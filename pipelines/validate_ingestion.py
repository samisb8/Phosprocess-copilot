"""Validation automatique des résultats d'ingestion documentaire."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from phosprocess.ingestion.schemas import ParsedPage

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PAGES_DIRECTORY = PROJECT_ROOT / "data" / "interim" / "pages"
DEFAULT_MANIFESTS_DIRECTORY = PROJECT_ROOT / "data" / "interim" / "manifests"
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "reports"
    / "ingestion_validation.json"
)

CHEMICAL_PATTERNS = {
    "P2O5": re.compile(r"\bP2O5\b", flags=re.IGNORECASE),
    "SO4": re.compile(r"\bSO4\b", flags=re.IGNORECASE),
    "CaO": re.compile(r"\bCaO\b", flags=re.IGNORECASE),
    "CaSO4": re.compile(r"\bCaSO4\b", flags=re.IGNORECASE),
}


def read_json_file(file_path: Path) -> dict[str, Any]:
    """Lire un fichier JSON et retourner un dictionnaire."""

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"JSON invalide dans {file_path.name}, "
            f"ligne {error.lineno}, colonne {error.colno}"
        ) from error

    if not isinstance(data, dict):
        raise ValueError(
            f"Le fichier {file_path.name} doit contenir un objet JSON."
        )

    return data


def count_chemical_tokens(pages: list[ParsedPage]) -> dict[str, int]:
    """Compter quelques termes chimiques importants dans le corpus."""

    combined_text = "\n".join(
        page.content.plain_text
        for page in pages
    )

    return {
        token: len(pattern.findall(combined_text))
        for token, pattern in CHEMICAL_PATTERNS.items()
    }


def validate_jsonl_lines(
    pages_path: Path,
    expected_document_id: str,
) -> tuple[list[ParsedPage], list[str]]:
    """Lire et valider chaque ligne d'un fichier JSONL."""

    pages: list[ParsedPage] = []
    errors: list[str] = []

    with pages_path.open("r", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                errors.append(f"Ligne {line_number} vide dans le JSONL.")
                continue

            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(
                    f"Ligne {line_number} : JSON invalide "
                    f"à la colonne {error.colno}."
                )
                continue

            try:
                page = ParsedPage.model_validate(raw_record)
            except ValidationError as error:
                errors.append(
                    f"Ligne {line_number} : schéma ParsedPage invalide : "
                    f"{error.error_count()} erreur(s)."
                )
                continue

            if page.provenance.document_id != expected_document_id:
                errors.append(
                    f"Ligne {line_number} : document_id incorrect "
                    f"({page.provenance.document_id})."
                )

            pages.append(page)

    return pages, errors


def validate_page_metrics(
    pages: list[ParsedPage],
) -> tuple[list[str], list[str]]:
    """Vérifier la cohérence interne des métriques des pages."""

    errors: list[str] = []
    warnings: list[str] = []

    for page in pages:
        page_number = page.provenance.page_number
        text = page.content.plain_text

        actual_character_count = len(text)
        actual_word_count = len(text.split())
        actual_is_empty = not text.strip()

        if page.quality.character_count != actual_character_count:
            errors.append(
                f"Page {page_number} : character_count incohérent "
                f"({page.quality.character_count} au lieu de "
                f"{actual_character_count})."
            )

        if page.quality.word_count != actual_word_count:
            errors.append(
                f"Page {page_number} : word_count incohérent "
                f"({page.quality.word_count} au lieu de {actual_word_count})."
            )

        if page.quality.is_empty != actual_is_empty:
            errors.append(
                f"Page {page_number} : indicateur is_empty incohérent."
            )

        if "\x00" in text:
            errors.append(
                f"Page {page_number} : caractère NULL détecté."
            )

        if "\ufffd" in text:
            warnings.append(
                f"Page {page_number} : caractère de remplacement Unicode détecté."
            )

        if not actual_is_empty and actual_word_count < 5:
            warnings.append(
                f"Page {page_number} : très peu de texte "
                f"({actual_word_count} mots)."
            )

    return errors, warnings


def validate_page_sequence(
    pages: list[ParsedPage],
    expected_total_pages: int | None,
) -> tuple[list[str], list[str]]:
    """Vérifier les numéros, doublons, ordre et pages manquantes."""

    errors: list[str] = []
    warnings: list[str] = []

    page_numbers = [
        page.provenance.page_number
        for page in pages
    ]

    counts = Counter(page_numbers)

    duplicate_pages = sorted(
        page_number
        for page_number, count in counts.items()
        if count > 1
    )

    if duplicate_pages:
        errors.append(
            f"Pages dupliquées : {duplicate_pages}."
        )

    if page_numbers != sorted(page_numbers):
        warnings.append(
            "Les pages ne sont pas enregistrées dans l'ordre croissant."
        )

    if expected_total_pages is not None:
        expected_pages = set(range(1, expected_total_pages + 1))
    elif page_numbers:
        expected_pages = set(range(1, max(page_numbers) + 1))
    else:
        expected_pages = set()

    missing_pages = sorted(expected_pages - set(page_numbers))

    if missing_pages:
        errors.append(
            f"Pages manquantes : {missing_pages}."
        )

    if expected_total_pages is not None and len(pages) != expected_total_pages:
        errors.append(
            f"Nombre de pages incorrect : {len(pages)} au lieu de "
            f"{expected_total_pages}."
        )

    return errors, warnings


def validate_manifest_consistency(
    pages: list[ParsedPage],
    manifest: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Comparer le JSONL avec les statistiques du manifeste."""

    errors: list[str] = []
    warnings: list[str] = []

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

    parser_counts = dict(
        Counter(
            page.provenance.parser
            for page in pages
        )
    )

    pages_with_text = sum(
        not page.quality.is_empty
        for page in pages
    )

    if manifest.get("pages_with_text") != pages_with_text:
        errors.append(
            "Le nombre pages_with_text ne correspond pas au manifeste."
        )

    if manifest.get("empty_pages") != empty_pages:
        warnings.append(
            "La liste empty_pages diffère du manifeste."
        )

    if manifest.get("review_pages") != review_pages:
        warnings.append(
            "La liste review_pages diffère du manifeste."
        )

    if manifest.get("parser_counts") != parser_counts:
        warnings.append(
            "Les statistiques parser_counts diffèrent du manifeste."
        )

    return errors, warnings


def validate_document(
    pages_path: Path,
    manifests_directory: Path,
) -> dict[str, Any]:
    """Valider un document et construire son rapport qualité."""

    document_id = pages_path.stem.removesuffix("_pages")
    manifest_path = (
        manifests_directory / f"{document_id}_manifest.json"
    )

    errors: list[str] = []
    warnings: list[str] = []

    if not manifest_path.exists():
        return {
            "document_id": document_id,
            "status": "invalid",
            "errors": [
                f"Manifeste introuvable : {manifest_path.name}"
            ],
            "warnings": [],
        }

    try:
        manifest = read_json_file(manifest_path)
    except ValueError as error:
        return {
            "document_id": document_id,
            "status": "invalid",
            "errors": [str(error)],
            "warnings": [],
        }

    pages, line_errors = validate_jsonl_lines(
        pages_path=pages_path,
        expected_document_id=document_id,
    )

    errors.extend(line_errors)

    metric_errors, metric_warnings = validate_page_metrics(pages)
    errors.extend(metric_errors)
    warnings.extend(metric_warnings)

    total_pages_value = manifest.get("total_pages")
    expected_total_pages = (
        int(total_pages_value)
        if isinstance(total_pages_value, int)
        else None
    )

    sequence_errors, sequence_warnings = validate_page_sequence(
        pages=pages,
        expected_total_pages=expected_total_pages,
    )

    errors.extend(sequence_errors)
    warnings.extend(sequence_warnings)

    manifest_errors, manifest_warnings = validate_manifest_consistency(
        pages=pages,
        manifest=manifest,
    )

    errors.extend(manifest_errors)
    warnings.extend(manifest_warnings)

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

    parser_counts = dict(
        Counter(
            page.provenance.parser
            for page in pages
        )
    )

    if errors:
        status = "invalid"
    elif warnings:
        status = "valid_with_warnings"
    else:
        status = "valid"

    return {
        "document_id": document_id,
        "source_file": manifest.get("source_file"),
        "status": status,
        "total_pages": len(pages),
        "pages_with_text": sum(
            not page.quality.is_empty
            for page in pages
        ),
        "empty_pages": empty_pages,
        "review_pages": review_pages,
        "ocr_pages": ocr_pages,
        "parser_counts": parser_counts,
        "chemical_token_counts": count_chemical_tokens(pages),
        "total_characters": sum(
            page.quality.character_count
            for page in pages
        ),
        "total_words": sum(
            page.quality.word_count
            for page in pages
        ),
        "errors": errors,
        "warnings": warnings,
    }


def write_json_report(
    report: dict[str, Any],
    report_path: Path,
) -> None:
    """Écrire le rapport avec remplacement atomique."""

    report_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = report_path.with_suffix(".json.tmp")

    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    temporary_path.replace(report_path)


def parse_arguments() -> argparse.Namespace:
    """Lire les options de la ligne de commande."""

    parser = argparse.ArgumentParser(
        description="Valider les résultats de l'ingestion documentaire."
    )

    parser.add_argument(
        "--pages-dir",
        type=Path,
        default=DEFAULT_PAGES_DIRECTORY,
    )

    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=DEFAULT_MANIFESTS_DIRECTORY,
    )

    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Traiter les warnings comme des échecs.",
    )

    return parser.parse_args()


def main() -> None:
    """Valider tous les documents ingérés."""

    arguments = parse_arguments()

    pages_directory = arguments.pages_dir.resolve()
    manifests_directory = arguments.manifests_dir.resolve()
    report_path = arguments.report_path.resolve()

    page_files = sorted(
        pages_directory.glob("*_pages.jsonl")
    )

    if not page_files:
        raise FileNotFoundError(
            f"Aucun fichier JSONL trouvé dans {pages_directory}"
        )

    results = [
        validate_document(
            pages_path=pages_path,
            manifests_directory=manifests_directory,
        )
        for pages_path in page_files
    ]

    status_counts = Counter(
        result["status"]
        for result in results
    )

    report = {
        "documents_validated": len(results),
        "status_counts": dict(status_counts),
        "documents": results,
    }

    write_json_report(report, report_path)

    print("\n=== Validation de l'ingestion ===")

    for result in results:
        print(
            f"{result['document_id']} | "
            f"{result['status']} | "
            f"pages={result.get('total_pages', 0)} | "
            f"errors={len(result['errors'])} | "
            f"warnings={len(result['warnings'])}"
        )

    print("\n=== Résumé ===")
    print(f"Documents validés       : {len(results)}")
    print(f"Valides                  : {status_counts['valid']}")
    print(
        "Valides avec warnings    : "
        f"{status_counts['valid_with_warnings']}"
    )
    print(f"Invalides                : {status_counts['invalid']}")
    print(f"Rapport                  : {report_path}")

    has_errors = status_counts["invalid"] > 0
    has_strict_warnings = (
        arguments.strict
        and status_counts["valid_with_warnings"] > 0
    )

    if has_errors or has_strict_warnings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()