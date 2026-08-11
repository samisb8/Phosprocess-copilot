"""Freeze the selected DEV v3 configuration from existing robustness artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import stat
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval"
    / "v0.1"
)
ROBUSTNESS_DIRECTORY = EVALUATION_ROOT / "v3" / "robustness"
ROBUSTNESS_SUMMARY_PATH = (
    ROBUSTNESS_DIRECTORY / "robustness_summary.json"
)
SENSITIVITY_PATH = (
    ROBUSTNESS_DIRECTORY / "parameter_sensitivity.csv"
)
PER_QUERY_PATH = (
    ROBUSTNESS_DIRECTORY / "robustness_per_query.csv"
)
ROBUSTNESS_REPORT_PATH = (
    ROBUSTNESS_DIRECTORY / "robustness_report.md"
)
FROZEN_ROOT = EVALUATION_ROOT / "frozen"
FINAL_DIRECTORY = FROZEN_ROOT / "dev_best_v3"
STAGING_DIRECTORY = FROZEN_ROOT / ".dev_best_v3_staging"
SAFEGUARD_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "lexical_safeguard_v3.yaml"
)

EXPECTED_SELECTION_RULE = [
    "no_candidate_recall_at_20_regression",
    "maximize_evidence_recall_at_5",
    "maximize_hit_at_5",
    "maximize_mrr_at_5",
    "maximize_hit_at_1",
    "minimize_complexity",
    "minimize_policy_latency",
]
EXPECTED_VARIANT_IDS = {
    "strict_lexical_slots_0",
    "lexical_safeguard_001",
    "permissive_lexical_slots_2",
}
SELECTED_VARIANT_ID = "lexical_safeguard_001"
METRIC_FIELDS = (
    "candidate_recall_at_20",
    "evidence_recall_at_5",
    "hit_at_5",
    "mrr_at_5",
    "hit_at_1",
)
FORBIDDEN_PATH_TOKENS = (
    "test_pool",
    "test_gold",
    "test_hybrid",
)

COPY_SOURCES = {
    "retrieval_v2.yaml": (
        EVALUATION_ROOT
        / "frozen"
        / "dev_best_v2"
        / "retrieval_v2.yaml"
    ),
    "reranking.yaml": (
        EVALUATION_ROOT
        / "frozen"
        / "dev_best_v2"
        / "reranking.yaml"
    ),
    "hybrid.py": (
        PROJECT_ROOT
        / "src"
        / "phosprocess"
        / "retrieval"
        / "hybrid.py"
    ),
    "reranker.py": (
        PROJECT_ROOT
        / "src"
        / "phosprocess"
        / "reranking"
        / "reranker.py"
    ),
    "v3_selection.py": (
        PROJECT_ROOT
        / "src"
        / "phosprocess"
        / "retrieval"
        / "v3_selection.py"
    ),
    "lexical_safeguard_v3.yaml": SAFEGUARD_CONFIG_PATH,
    "robustness_summary.json": ROBUSTNESS_SUMMARY_PATH,
    "parameter_sensitivity.csv": SENSITIVITY_PATH,
}


class FreezeValidationError(ValueError):
    """Raised before any snapshot write when the DEV decision is invalid."""


def sha256_file(path: Path) -> str:
    """Return an uppercase SHA-256 digest."""

    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest().upper()


def sha256_bytes(value: bytes) -> str:
    """Hash in-memory canonical snapshot identity material."""

    return hashlib.sha256(value).hexdigest().upper()


def parse_bool(value: Any) -> bool:
    """Parse JSON or CSV booleans strictly."""

    if isinstance(value, bool):
        return value

    normalized = str(value).strip().casefold()

    if normalized in {"true", "1"}:
        return True

    if normalized in {"false", "0"}:
        return False

    raise FreezeValidationError(
        f"Valeur booléenne invalide: {value!r}."
    )


def ensure_dev_only_path(path: Path) -> Path:
    """Reject any path whose name indicates a non-DEV evaluation artifact."""

    resolved = path.resolve()
    normalized = str(resolved).replace("\\", "/").casefold()

    for token in FORBIDDEN_PATH_TOKENS:
        if token in normalized:
            raise FreezeValidationError(
                f"Chemin interdit pour le gel DEV: {resolved}"
            )

    return resolved


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read one validated DEV CSV artifact."""

    ensure_dev_only_path(path)

    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def parse_sensitivity_rows(
    raw_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Normalize the three sensitivity variants."""

    if len(raw_rows) != 3:
        raise FreezeValidationError(
            "parameter_sensitivity.csv doit contenir trois variantes."
        )

    rows: list[dict[str, Any]] = []

    for raw in raw_rows:
        rows.append(
            {
                "variant_id": raw["variant_id"],
                "label": raw["label"],
                "lexical_slots": int(raw["lexical_slots"]),
                "complexity": int(raw["complexity"]),
                **{
                    field: float(raw[field])
                    for field in METRIC_FIELDS
                },
                "median_policy_latency_us": float(
                    raw["median_policy_latency_us"]
                ),
                "metrics_identical_across_runs": parse_bool(
                    raw["metrics_identical_across_runs"]
                ),
                "eligible_candidate_recall": parse_bool(
                    raw["eligible_candidate_recall"]
                ),
                "selection_rank": int(raw["selection_rank"]),
                "selected": parse_bool(raw["selected"]),
            }
        )

    variant_ids = {
        row["variant_id"]
        for row in rows
    }

    if variant_ids != EXPECTED_VARIANT_IDS:
        raise FreezeValidationError(
            f"Variantes inattendues: {sorted(variant_ids)}."
        )

    return rows


def select_variant(
    rows: list[dict[str, Any]],
    *,
    baseline_candidate_recall: float,
) -> list[dict[str, Any]]:
    """Apply the predefined selection rule without mutating the inputs."""

    eligible = [
        row
        for row in rows
        if (
            row["candidate_recall_at_20"]
            >= baseline_candidate_recall - 1e-12
        )
    ]

    if not eligible:
        raise FreezeValidationError(
            "Toutes les variantes régressent en Candidate Recall@20."
        )

    return sorted(
        eligible,
        key=lambda row: (
            -row["evidence_recall_at_5"],
            -row["hit_at_5"],
            -row["mrr_at_5"],
            -row["hit_at_1"],
            row["complexity"],
            row["median_policy_latency_us"],
            row["variant_id"],
        ),
    )


def count_regressions(
    per_query_rows: list[dict[str, str]],
) -> dict[str, int]:
    """Count per-query metric regressions for every variant."""

    if len(per_query_rows) != 48:
        raise FreezeValidationError(
            "robustness_per_query.csv doit contenir 48 lignes."
        )

    grouped: dict[str, list[dict[str, str]]] = {
        variant_id: []
        for variant_id in EXPECTED_VARIANT_IDS
    }

    for row in per_query_rows:
        variant_id = row["variant_id"]

        if variant_id not in grouped:
            raise FreezeValidationError(
                f"Variante par requête inattendue: {variant_id}."
            )

        grouped[variant_id].append(row)

    regressions: dict[str, int] = {}

    for variant_id, rows in grouped.items():
        query_ids = [
            row["query_id"]
            for row in rows
        ]

        if len(rows) != 16 or len(query_ids) != len(set(query_ids)):
            raise FreezeValidationError(
                f"{variant_id}: 16 query_id uniques sont requis."
            )

        if not all(
            parse_bool(row["same_selection_across_runs"])
            for row in rows
        ):
            raise FreezeValidationError(
                f"{variant_id}: sélection non déterministe."
            )

        regressions[variant_id] = sum(
            row["outcome_vs_baseline"] == "regressed"
            for row in rows
        )

    return regressions


def validate_summary_consistency(
    summary: dict[str, Any],
    sensitivity_rows: list[dict[str, Any]],
    per_query_rows: list[dict[str, str]],
    report_text: str,
) -> dict[str, Any]:
    """Validate the winning DEV decision before creating any file."""

    if summary.get("split") != "dev":
        raise FreezeValidationError(
            "Le résumé de robustesse n'est pas DEV."
        )

    required_false_flags = (
        "test_artifacts_read",
        "test_evaluation_run",
        "reference_answers_used_for_inference",
        "gold_used_for_inference",
        "v3_frozen",
    )

    for field in required_false_flags:
        if summary.get(field) is not False:
            raise FreezeValidationError(
                f"Le champ {field} doit être false avant gel."
            )

    if summary.get("robustness_passed") is not True:
        raise FreezeValidationError(
            "La validation de robustesse n'est pas réussie."
        )

    if summary.get("selection_rule") != EXPECTED_SELECTION_RULE:
        raise FreezeValidationError(
            "La règle de sélection enregistrée est inattendue."
        )

    determinism = summary.get("determinism", {})
    required_determinism = (
        "all_variant_selections_stable",
        "baseline_same_top5",
        "current_same_metrics",
        "current_same_top5",
    )

    if determinism.get("repetitions") != 3:
        raise FreezeValidationError(
            "Trois répétitions sont requises."
        )

    if not all(
        determinism.get(field) is True
        for field in required_determinism
    ):
        raise FreezeValidationError(
            "Le candidat gagnant n'est pas déterministe."
        )

    if (
        determinism.get("stable_current_queries") != 16
        or determinism.get("total_current_queries") != 16
    ):
        raise FreezeValidationError(
            "Les 16 sélections DEV doivent être stables."
        )

    selected_rows = [
        row
        for row in sensitivity_rows
        if row["selected"]
    ]

    if len(selected_rows) != 1:
        raise FreezeValidationError(
            "Une seule variante doit être marquée gagnante."
        )

    baseline_metrics = summary.get("baseline_v2_metrics", {})
    baseline_candidate_recall = float(
        baseline_metrics["candidate_recall_at_20"]
    )
    ranking = select_variant(
        sensitivity_rows,
        baseline_candidate_recall=baseline_candidate_recall,
    )
    winner = ranking[0]

    if winner["variant_id"] != SELECTED_VARIANT_ID:
        raise FreezeValidationError(
            "La règle prédéfinie ne sélectionne pas lexical_safeguard_001."
        )

    if (
        selected_rows[0]["variant_id"] != winner["variant_id"]
        or summary.get("selected_variant") != winner["variant_id"]
    ):
        raise FreezeValidationError(
            "Le gagnant enregistré diffère du gagnant recomputé."
        )

    if not winner["metrics_identical_across_runs"]:
        raise FreezeValidationError(
            "Les métriques du gagnant ne sont pas stables."
        )

    if not winner["eligible_candidate_recall"]:
        raise FreezeValidationError(
            "Le gagnant régresse en Candidate Recall@20."
        )

    regressions = count_regressions(per_query_rows)

    if regressions[winner["variant_id"]] != 0:
        raise FreezeValidationError(
            "Le gagnant présente une régression DEV."
        )

    if (
        summary.get("recommendation")
        != "freeze lexical_safeguard_001"
    ):
        raise FreezeValidationError(
            "La recommandation de robustesse n'autorise pas le gel."
        )

    if "freeze lexical_safeguard_001" not in report_text:
        raise FreezeValidationError(
            "Le rapport de robustesse ne recommande pas le gel."
        )

    summary_variants = {
        row["variant_id"]: row
        for row in summary.get("variants", [])
    }

    if set(summary_variants) != EXPECTED_VARIANT_IDS:
        raise FreezeValidationError(
            "Les variantes du résumé sont incomplètes."
        )

    for row in sensitivity_rows:
        summary_row = summary_variants[row["variant_id"]]

        for field in METRIC_FIELDS:
            if abs(float(summary_row[field]) - row[field]) > 1e-12:
                raise FreezeValidationError(
                    f"Métrique incohérente pour {row['variant_id']}: {field}."
                )

    return {
        "winner": winner,
        "ranking": [
            row["variant_id"]
            for row in ranking
        ],
        "regressions": regressions,
        "baseline_metrics": {
            field: float(baseline_metrics[field])
            for field in METRIC_FIELDS
        },
    }


def validate_safeguard_config(
    path: Path,
    winner: dict[str, Any],
) -> dict[str, Any]:
    """Require the persisted safeguard configuration to match the winner."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise FreezeValidationError(
            "La configuration du safeguard est invalide."
        )

    selection = raw.get("selection")
    retrieval = raw.get("retrieval")

    if not isinstance(selection, dict) or not isinstance(retrieval, dict):
        raise FreezeValidationError(
            "Sections retrieval/selection manquantes."
        )

    expected = {
        "experiment_id": SELECTED_VARIANT_ID,
        "development_split": "dev",
        "candidate_k": 20,
        "dense_candidates": 20,
        "bm25_candidates": 20,
        "query_expansion": True,
        "method": "lexical_safeguard",
        "top_k": 5,
        "lexical_slots": winner["lexical_slots"],
        "reranker_leading_slots": 4,
        "lexical_source": "bm25",
        "duplicate_policy": "skip",
        "fallback": "next_reranker_result",
    }
    actual = {
        "experiment_id": raw.get("experiment_id"),
        "development_split": raw.get("development_split"),
        "candidate_k": retrieval.get("candidate_k"),
        "dense_candidates": retrieval.get("dense_candidates"),
        "bm25_candidates": retrieval.get("bm25_candidates"),
        "query_expansion": retrieval.get("query_expansion"),
        "method": selection.get("method"),
        "top_k": selection.get("top_k"),
        "lexical_slots": selection.get("lexical_slots"),
        "reranker_leading_slots": selection.get(
            "reranker_leading_slots"
        ),
        "lexical_source": selection.get("lexical_source"),
        "duplicate_policy": selection.get("duplicate_policy"),
        "fallback": selection.get("fallback"),
    }

    if actual != expected:
        raise FreezeValidationError(
            f"Configuration safeguard inattendue: {actual}."
        )

    return raw


def snapshot_identity(
    component_hashes: dict[str, str],
) -> str:
    """Create one stable SHA-256 identity for the frozen component set."""

    canonical = "".join(
        f"{file_name},{component_hashes[file_name]}\n"
        for file_name in sorted(component_hashes)
    )
    return sha256_bytes(canonical.encode("utf-8"))


def relative_project_path(path: Path) -> str:
    """Return a portable project-relative path."""

    return str(path.resolve().relative_to(PROJECT_ROOT)).replace(
        "\\",
        "/",
    )


def build_report(
    *,
    manifest: dict[str, Any],
    variants: list[dict[str, Any]],
) -> str:
    """Build the frozen DEV v3 decision report."""

    metrics = manifest["dev_metrics"]
    parameters = manifest["parameters"]
    lines = [
        "# dev_best_v3 — configuration DEV figée",
        "",
        "## Décision",
        "",
        f"- Variante retenue : `{manifest['selected_variant']}`.",
        "- Décision fondée exclusivement sur les artefacts de robustesse DEV.",
        "- Aucun TEST utilisé, lu ou exécuté.",
        "- Une seule variante est marquée gagnante.",
        "",
        "## Métriques DEV finales",
        "",
        f"- Candidate Recall@20 : {metrics['candidate_recall_at_20']:.4f}",
        f"- Evidence Recall@5 : {metrics['evidence_recall_at_5']:.4f}",
        f"- Hit@5 : {metrics['hit_at_5']:.4f}",
        f"- MRR@5 : {metrics['mrr_at_5']:.4f}",
        f"- Hit@1 : {metrics['hit_at_1']:.4f}",
        "",
        "## Paramètres",
        "",
        f"- candidate_k : {parameters['candidate_k']}",
        f"- dense_candidates : {parameters['dense_candidates']}",
        f"- bm25_candidates : {parameters['bm25_candidates']}",
        f"- query_expansion : {str(parameters['query_expansion']).lower()}",
        f"- top_k : {parameters['top_k']}",
        f"- lexical_slots : {parameters['lexical_slots']}",
        (
            "- reranker_leading_slots : "
            f"{parameters['reranker_leading_slots']}"
        ),
        "- lexical_source : bm25",
        "- fallback : next_reranker_result",
        "",
        "## Comparaison des variantes",
        "",
        (
            "| Variante | Candidate Recall@20 | Evidence Recall@5 | "
            "Hit@5 | MRR@5 | Hit@1 | Régressions |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in sorted(
        variants,
        key=lambda item: item["selection_rank"],
    ):
        lines.append(
            f"| {row['variant_id']} | "
            f"{row['candidate_recall_at_20']:.4f} | "
            f"{row['evidence_recall_at_5']:.4f} | "
            f"{row['hit_at_5']:.4f} | "
            f"{row['mrr_at_5']:.4f} | "
            f"{row['hit_at_1']:.4f} | "
            f"{manifest['variant_regressions'][row['variant_id']]} |"
        )

    lines.extend(
        [
            "",
            "## Intégrité",
            "",
            (
                "- Identité SHA-256 du snapshot : "
                f"`{manifest['snapshot_sha256']}`."
            ),
            "- Les copies ont été comparées octet par octet aux sources.",
            "- Les composants sources v2 n'ont pas été modifiés.",
            "",
        ]
    )

    return "\n".join(lines)


def set_read_only(directory: Path) -> None:
    """Set every frozen file to read-only on Windows-compatible filesystems."""

    for path in directory.iterdir():
        if path.is_file():
            path.chmod(stat.S_IREAD)


def is_read_only(path: Path) -> bool:
    """Check the Windows read-only attribute with a portable fallback."""

    file_stat = path.stat()
    file_attributes = getattr(
        file_stat,
        "st_file_attributes",
        None,
    )
    read_only_attribute = getattr(
        stat,
        "FILE_ATTRIBUTE_READONLY",
        None,
    )

    if (
        file_attributes is not None
        and read_only_attribute is not None
    ):
        return bool(
            file_attributes
            & read_only_attribute
        )

    return not bool(
        file_stat.st_mode
        & stat.S_IWUSR
    )


def main() -> None:
    """Validate, freeze, verify, and protect dev_best_v3."""

    if FINAL_DIRECTORY.exists():
        raise FileExistsError(
            f"Le snapshot existe déjà: {FINAL_DIRECTORY}"
        )

    if STAGING_DIRECTORY.exists():
        raise FileExistsError(
            f"Le staging existe déjà: {STAGING_DIRECTORY}"
        )

    for path in (
        ROBUSTNESS_SUMMARY_PATH,
        SENSITIVITY_PATH,
        PER_QUERY_PATH,
        ROBUSTNESS_REPORT_PATH,
        SAFEGUARD_CONFIG_PATH,
        *COPY_SOURCES.values(),
    ):
        ensure_dev_only_path(path)

        if not path.is_file():
            raise FileNotFoundError(path)

    summary = json.loads(
        ROBUSTNESS_SUMMARY_PATH.read_text(encoding="utf-8")
    )
    sensitivity_rows = parse_sensitivity_rows(
        read_csv_rows(SENSITIVITY_PATH)
    )
    per_query_rows = read_csv_rows(PER_QUERY_PATH)
    report_text = ROBUSTNESS_REPORT_PATH.read_text(
        encoding="utf-8"
    )
    decision = validate_summary_consistency(
        summary,
        sensitivity_rows,
        per_query_rows,
        report_text,
    )
    safeguard_config = validate_safeguard_config(
        SAFEGUARD_CONFIG_PATH,
        decision["winner"],
    )

    expected_source_hashes = summary["source_sha256"]
    source_hashes = {
        file_name: sha256_file(source_path)
        for file_name, source_path in COPY_SOURCES.items()
    }
    expected_hash_names = {
        "retrieval_v2.yaml",
        "reranking.yaml",
        "hybrid.py",
        "reranker.py",
        "v3_selection.py",
    }

    for file_name in expected_hash_names:
        if (
            source_hashes[file_name]
            != expected_source_hashes[file_name]
        ):
            raise FreezeValidationError(
                f"La source utilisée a changé: {file_name}."
            )

    pre_copy_v2_hashes = {
        file_name: source_hashes[file_name]
        for file_name in (
            "retrieval_v2.yaml",
            "reranking.yaml",
            "hybrid.py",
            "reranker.py",
        )
    }
    FROZEN_ROOT.mkdir(parents=True, exist_ok=True)
    STAGING_DIRECTORY.mkdir()

    for file_name, source_path in COPY_SOURCES.items():
        shutil.copy2(
            source_path,
            STAGING_DIRECTORY / file_name,
        )

    copied_component_hashes: dict[str, str] = {}
    component_records: list[dict[str, Any]] = []

    for file_name, source_path in COPY_SOURCES.items():
        copied_path = STAGING_DIRECTORY / file_name
        source_bytes = source_path.read_bytes()
        copied_bytes = copied_path.read_bytes()

        if copied_bytes != source_bytes:
            raise FreezeValidationError(
                f"Copie non identique pour {file_name}."
            )

        copied_hash = sha256_file(copied_path)

        if copied_hash != source_hashes[file_name]:
            raise FreezeValidationError(
                f"Empreinte de copie incorrecte pour {file_name}."
            )

        copied_component_hashes[file_name] = copied_hash
        component_records.append(
            {
                "file": file_name,
                "source": relative_project_path(source_path),
                "source_sha256": source_hashes[file_name],
                "frozen_sha256": copied_hash,
                "byte_identical": True,
            }
        )

    frozen_at = datetime.now().astimezone().isoformat()
    winner = decision["winner"]
    snapshot_sha256 = snapshot_identity(
        copied_component_hashes
    )
    snapshot_files = [
        *sorted(COPY_SOURCES),
        "freeze_manifest.json",
        "sha256.csv",
        "dev_best_v3_report.md",
    ]
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "frozen_dev_best_v3",
        "frozen_at": frozen_at,
        "scope": "dev_only",
        "selected_variant": winner["variant_id"],
        "single_selected_variant": True,
        "dev_metrics": {
            field: winner[field]
            for field in METRIC_FIELDS
        },
        "parameters": {
            "candidate_k": safeguard_config["retrieval"][
                "candidate_k"
            ],
            "dense_candidates": safeguard_config["retrieval"][
                "dense_candidates"
            ],
            "bm25_candidates": safeguard_config["retrieval"][
                "bm25_candidates"
            ],
            "query_expansion": safeguard_config["retrieval"][
                "query_expansion"
            ],
            "top_k": safeguard_config["selection"]["top_k"],
            "lexical_slots": safeguard_config["selection"][
                "lexical_slots"
            ],
            "reranker_leading_slots": safeguard_config[
                "selection"
            ]["reranker_leading_slots"],
            "lexical_source": safeguard_config["selection"][
                "lexical_source"
            ],
            "lexical_order": safeguard_config["selection"][
                "lexical_order"
            ],
            "duplicate_policy": safeguard_config["selection"][
                "duplicate_policy"
            ],
            "fallback": safeguard_config["selection"]["fallback"],
        },
        "selection_rule": EXPECTED_SELECTION_RULE,
        "selection_ranking": decision["ranking"],
        "variant_regressions": decision["regressions"],
        "determinism": summary["determinism"],
        "robustness_validation_id": summary["validation_id"],
        "components": component_records,
        "files": snapshot_files,
        "snapshot_sha256": snapshot_sha256,
        "integrity": {
            "all_source_copies_byte_identical": True,
            "all_component_hashes_verified": True,
            "v2_sources_unchanged_after_copy": True,
            "one_selected_variant": True,
            "read_only_windows": True,
        },
        "test_usage": {
            "test_artifacts_read": False,
            "test_evaluation_run": False,
            "explicit_statement": "Aucun TEST utilisé.",
        },
    }
    report = build_report(
        manifest=manifest,
        variants=sensitivity_rows,
    )
    manifest_path = STAGING_DIRECTORY / "freeze_manifest.json"
    report_path = STAGING_DIRECTORY / "dev_best_v3_report.md"
    registry_path = STAGING_DIRECTORY / "sha256.csv"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path.write_text(report, encoding="utf-8")
    registry_rows = [
        {
            "file": record["file"],
            "sha256": record["frozen_sha256"],
            "source": record["source"],
            "source_sha256": record["source_sha256"],
        }
        for record in component_records
    ]
    registry_rows.extend(
        [
            {
                "file": "freeze_manifest.json",
                "sha256": sha256_file(manifest_path),
                "source": "",
                "source_sha256": "",
            },
            {
                "file": "dev_best_v3_report.md",
                "sha256": sha256_file(report_path),
                "source": "",
                "source_sha256": "",
            },
        ]
    )

    with registry_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "file",
                "sha256",
                "source",
                "source_sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(registry_rows)

    for row in read_csv_rows(registry_path):
        frozen_path = STAGING_DIRECTORY / row["file"]

        if sha256_file(frozen_path) != row["sha256"]:
            raise FreezeValidationError(
                f"Registre SHA-256 incorrect pour {row['file']}."
            )

    post_copy_v2_hashes = {
        file_name: sha256_file(COPY_SOURCES[file_name])
        for file_name in pre_copy_v2_hashes
    }

    if post_copy_v2_hashes != pre_copy_v2_hashes:
        raise FreezeValidationError(
            "Un composant source v2 a été modifié pendant le gel."
        )

    os.replace(STAGING_DIRECTORY, FINAL_DIRECTORY)
    set_read_only(FINAL_DIRECTORY)

    for file_name, expected_hash in copied_component_hashes.items():
        frozen_path = FINAL_DIRECTORY / file_name

        if sha256_file(frozen_path) != expected_hash:
            raise FreezeValidationError(
                f"Vérification finale échouée pour {file_name}."
            )

    for frozen_path in FINAL_DIRECTORY.iterdir():
        if frozen_path.is_file() and not is_read_only(frozen_path):
            raise FreezeValidationError(
                "Le fichier n'est pas en lecture seule: "
                f"{frozen_path.name}."
            )

    print("dev_best_v3 frozen successfully.")
    print(f"Selected variant: {winner['variant_id']}")
    print(f"Snapshot SHA-256: {snapshot_sha256}")
    print(f"Manifest SHA-256: {sha256_file(FINAL_DIRECTORY / 'freeze_manifest.json')}")
    print(f"Directory: {FINAL_DIRECTORY}")
    print("TEST used: False")


if __name__ == "__main__":
    main()
