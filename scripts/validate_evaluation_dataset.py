"""Valider le dataset d'évaluation PhosProcess."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from phosprocess.evaluation.schemas import (
    EvaluationQuery,
    RelevanceJudgment,
    load_evaluation_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "evaluation.yaml"
)

ModelT = TypeVar(
    "ModelT",
    bound=BaseModel,
)


def resolve_project_path(
    path_value: str,
) -> Path:
    """Résoudre un chemin depuis la racine."""

    path = Path(path_value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def sha256_file(path: Path) -> str:
    """Calculer une empreinte SHA-256."""

    digest = hashlib.sha256()

    with path.open("rb") as source:
        for block in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def normalize_question(
    question: str,
) -> str:
    """Normaliser une question pour détecter les doublons."""

    return re.sub(
        r"\s+",
        " ",
        question,
    ).strip().casefold()


def load_jsonl(
    path: Path,
    model_type: type[ModelT],
) -> list[ModelT]:
    """Charger et valider un fichier JSONL."""

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )

    records: list[ModelT] = []

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
                raw_record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path.name}, ligne "
                    f"{line_number} : JSON invalide."
                ) from error

            try:
                record = model_type.model_validate(
                    raw_record
                )
            except ValidationError as error:
                raise ValueError(
                    f"{path.name}, ligne "
                    f"{line_number} : schéma invalide.\n"
                    f"{error}"
                ) from error

            records.append(record)

    return records


def write_report(
    report: dict[str, object],
    path: Path,
) -> None:
    """Écrire le rapport atomiquement."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(path)


def main() -> None:
    """Valider tous les artefacts."""

    config = load_evaluation_config(
        CONFIG_PATH
    )

    output_directory = resolve_project_path(
        config.dataset.output_directory
    )

    queries_path = (
        output_directory
        / config.dataset.queries_filename
    )

    judgments_path = (
        output_directory
        / config.dataset.judgments_filename
    )

    manifest_path = (
        output_directory
        / config.dataset.manifest_filename
    )

    report_path = (
        output_directory
        / config.dataset.validation_report_filename
    )

    errors: list[str] = []
    warnings: list[str] = []

    try:
        queries = load_jsonl(
            queries_path,
            EvaluationQuery,
        )
    except Exception as error:
        queries = []
        errors.append(str(error))

    try:
        judgments = load_jsonl(
            judgments_path,
            RelevanceJudgment,
        )
    except Exception as error:
        judgments = []

        if (
            judgments_path.exists()
            and judgments_path.stat().st_size == 0
        ):
            warnings.append(
                "judgments.jsonl est vide : les qrels "
                "seront construits à l'étape suivante."
            )
        else:
            errors.append(str(error))

    query_ids = [
        query.query_id
        for query in queries
    ]

    if len(query_ids) != len(set(query_ids)):
        errors.append(
            "Des query_id sont dupliqués."
        )

    expected_query_ids = [
        f"Q{index:03d}"
        for index in range(
            1,
            len(queries) + 1,
        )
    ]

    if query_ids != expected_query_ids:
        errors.append(
            "Les query_id ne sont pas continus et "
            "correctement ordonnés."
        )

    normalized_questions = [
        normalize_question(query.question)
        for query in queries
    ]

    if (
        len(normalized_questions)
        != len(set(normalized_questions))
    ):
        errors.append(
            "Des questions textuellement identiques "
            "ont été détectées."
        )

    family_splits: dict[str, set[str]] = defaultdict(set)

    for query in queries:
        family_splits[
            query.question_family_id
        ].add(query.split.value)

    leaking_families = {
        family: sorted(splits)
        for family, splits in family_splits.items()
        if len(splits) > 1
    }

    if leaking_families:
        errors.append(
            "Des familles de questions traversent DEV et TEST : "
            f"{leaking_families}"
        )

    total_queries = len(queries)

    if total_queries != config.expected.total_queries:
        errors.append(
            "Nombre total incorrect : "
            f"{total_queries} != "
            f"{config.expected.total_queries}."
        )

    answerable_count = sum(
        query.answerable
        for query in queries
    )

    if (
        answerable_count
        != config.expected.answerable_queries
    ):
        errors.append(
            "Nombre répondable incorrect : "
            f"{answerable_count} != "
            f"{config.expected.answerable_queries}."
        )

    unanswerable_count = (
        total_queries - answerable_count
    )

    if (
        unanswerable_count
        != config.expected.unanswerable_queries
    ):
        errors.append(
            "Nombre non répondable incorrect : "
            f"{unanswerable_count} != "
            f"{config.expected.unanswerable_queries}."
        )

    split_counts = Counter(
        query.split.value
        for query in queries
    )

    if dict(split_counts) != config.expected.splits:
        errors.append(
            "Répartition des splits incorrecte : "
            f"{dict(split_counts)} != "
            f"{config.expected.splits}."
        )

    category_counts = Counter(
        query.category.value
        for query in queries
    )

    if (
        dict(category_counts)
        != config.expected.categories
    ):
        errors.append(
            "Répartition des catégories incorrecte : "
            f"{dict(category_counts)} != "
            f"{config.expected.categories}."
        )

    language_counts = Counter(
        query.language.value
        for query in queries
    )

    difficulty_counts = Counter(
        query.difficulty.value
        for query in queries
    )

    query_map = {
        query.query_id: query
        for query in queries
    }

    judgment_keys: set[tuple[str, str]] = set()

    for judgment in judgments:
        key = (
            judgment.query_id,
            judgment.chunk_id,
        )

        if key in judgment_keys:
            errors.append(
                "Jugement dupliqué pour "
                f"{judgment.query_id} / "
                f"{judgment.chunk_id}."
            )

        judgment_keys.add(key)

        query = query_map.get(
            judgment.query_id
        )

        if query is None:
            errors.append(
                "Jugement associé à une question inconnue : "
                f"{judgment.query_id}."
            )
            continue

        if (
            not query.answerable
            and judgment.relevance > 0
        ):
            errors.append(
                "Une question non répondable possède un "
                "jugement positif : "
                f"{judgment.query_id} / "
                f"{judgment.chunk_id}."
            )

        if (
            config.annotation.require_rationale
            and not judgment.rationale.strip()
        ):
            errors.append(
                "Rationale manquante pour "
                f"{judgment.query_id} / "
                f"{judgment.chunk_id}."
            )

    judged_query_ids = {
        judgment.query_id
        for judgment in judgments
    }

    if judgments:
        missing_judgment_queries = sorted(
            set(query_ids) - judged_query_ids
        )

        if missing_judgment_queries:
            warnings.append(
                "Questions sans aucun jugement : "
                f"{missing_judgment_queries}"
            )

    manifest: dict[str, object] | None = None

    if not manifest_path.exists():
        errors.append(
            f"Manifest introuvable : {manifest_path}"
        )
    else:
        try:
            raw_manifest = json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )

            if not isinstance(raw_manifest, dict):
                raise ValueError(
                    "Le manifest doit être un objet JSON."
                )

            manifest = raw_manifest

        except Exception as error:
            errors.append(
                f"Manifest invalide : {error}"
            )

    if manifest is not None:
        files = manifest.get("files")

        if not isinstance(files, dict):
            errors.append(
                "La section files du manifest est absente."
            )
        else:
            for key, path in (
                ("queries", queries_path),
                ("judgments", judgments_path),
            ):
                file_record = files.get(key)

                if not isinstance(file_record, dict):
                    errors.append(
                        f"Artefact {key} absent du manifest."
                    )
                    continue

                recorded_hash = file_record.get(
                    "sha256"
                )

                if (
                    path.exists()
                    and recorded_hash
                    != sha256_file(path)
                ):
                    warnings.append(
                        f"Le hash de {path.name} diffère "
                        "du manifest. C'est normal après une "
                        "annotation, mais le manifest devra "
                        "être régénéré avant de geler v0.1."
                    )

    status = (
        "valid"
        if not errors
        else "invalid"
    )

    report: dict[str, object] = {
        "validated_at_utc": (
            datetime.now(UTC).isoformat()
        ),
        "status": status,
        "dataset": {
            "name": config.dataset.name,
            "version": config.dataset.version,
            "directory": str(output_directory),
        },
        "counts": {
            "queries": total_queries,
            "answerable": answerable_count,
            "unanswerable": unanswerable_count,
            "judgments": len(judgments),
            "judged_queries": len(
                judged_query_ids
            ),
            "splits": dict(split_counts),
            "categories": dict(category_counts),
            "languages": dict(language_counts),
            "difficulties": dict(
                difficulty_counts
            ),
        },
        "errors": errors,
        "warnings": warnings,
    }

    write_report(
        report,
        report_path,
    )

    print("\n=== Validation du benchmark ===")
    print(f"Statut         : {status}")
    print(f"Questions      : {total_queries}")
    print(f"Répondables    : {answerable_count}")
    print(f"Non répondables: {unanswerable_count}")
    print(f"Jugements      : {len(judgments)}")
    print(f"Splits         : {dict(split_counts)}")
    print(
        f"Catégories     : {dict(category_counts)}"
    )
    print(f"Rapport        : {report_path}")

    if warnings:
        print("\nAvertissements :")

        for warning in warnings:
            print(f"- {warning}")

    if errors:
        print("\nErreurs :")

        for error in errors:
            print(f"- {error}")

        raise SystemExit(1)


if __name__ == "__main__":
    main()