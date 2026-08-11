from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
import zipfile
from collections import Counter
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path.cwd().resolve()
ACTIVE_KB_NAME = "kb_quality_20260727_221312"

if not (PROJECT_ROOT / "src" / "phosprocess").exists():
    raise SystemExit(
        "Erreur : lance ce script depuis la racine de phosprocess-copilot."
    )

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
AUDIT_ROOT = PROJECT_ROOT / ".audit" / f"retriever_v4_{STAMP}"
REPORTS_DIR = AUDIT_ROOT / "reports"
SNAPSHOT_DIR = AUDIT_ROOT / "snapshot"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def save_text(name: str, content: str) -> None:
    path = REPORTS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", errors="replace")


def save_json(name: str, payload: Any) -> None:
    path = REPORTS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def run_command(
    name: str,
    command: list[str],
    timeout: int = 180,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "command": command,
        "returncode": None,
        "stdout": "",
        "stderr": "",
    }

    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
        result["returncode"] = completed.returncode
        result["stdout"] = completed.stdout
        result["stderr"] = completed.stderr

    except FileNotFoundError as exc:
        result["stderr"] = f"Commande introuvable : {exc}"

    except subprocess.TimeoutExpired as exc:
        result["stderr"] = f"Timeout après {timeout} secondes : {exc}"

    except Exception:
        result["stderr"] = traceback.format_exc()

    rendered = [
        f"COMMAND: {' '.join(command)}",
        f"RETURN CODE: {result['returncode']}",
        "",
        "STDOUT:",
        result["stdout"],
        "",
        "STDERR:",
        result["stderr"],
    ]
    save_text(f"commands/{name}.txt", "\n".join(rendered))
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def copy_source_snapshot() -> None:
    allowed_extensions = {
        ".py",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".md",
        ".txt",
        ".ini",
        ".cfg",
        ".lock",
    }

    excluded_names = {
        ".venv",
        ".git",
        ".audit",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        "data",
        "artifacts",
        "indexes",
        "models",
    }

    roots = [
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "tests",
        PROJECT_ROOT / "config",
        PROJECT_ROOT / "scripts",
    ]

    for root in roots:
        if not root.exists():
            continue

        for source in root.rglob("*"):
            if not source.is_file():
                continue

            if any(part in excluded_names for part in source.parts):
                continue

            if source.suffix.lower() not in allowed_extensions:
                continue

            # Évite d'inclure par erreur un gros artefact.
            if source.stat().st_size > 5 * 1024 * 1024:
                continue

            relative = source.relative_to(PROJECT_ROOT)
            destination = SNAPSHOT_DIR / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    for root_file in [
        "pyproject.toml",
        "README.md",
        ".gitignore",
        "requirements.txt",
        "requirements-dev.txt",
    ]:
        source = PROJECT_ROOT / root_file
        if source.exists() and source.is_file():
            destination = SNAPSHOT_DIR / root_file
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def build_source_inventory() -> None:
    roots = [
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "tests",
        PROJECT_ROOT / "config",
        PROJECT_ROOT / "scripts",
    ]

    inventory: list[dict[str, Any]] = []

    for root in roots:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if any(
                part in {
                    ".venv",
                    ".git",
                    ".audit",
                    "__pycache__",
                    ".pytest_cache",
                }
                for part in path.parts
            ):
                continue

            record: dict[str, Any] = {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "size_bytes": path.stat().st_size,
            }

            if path.stat().st_size <= 10 * 1024 * 1024:
                try:
                    record["sha256"] = sha256_file(path)
                except OSError:
                    record["sha256"] = "unavailable"

            inventory.append(record)

    save_json("source_inventory.json", inventory)


def build_python_ast_inventory() -> None:
    src_root = PROJECT_ROOT / "src" / "phosprocess"
    inventory: list[dict[str, Any]] = []

    for path in sorted(src_root.rglob("*.py")):
        relative = str(path.relative_to(PROJECT_ROOT))

        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except Exception as exc:
            inventory.append(
                {
                    "path": relative,
                    "parse_error": str(exc),
                }
            )
            continue

        functions: list[str] = []
        classes: dict[str, list[str]] = {}

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)

            elif isinstance(node, ast.ClassDef):
                methods: list[str] = []

                for child in node.body:
                    if isinstance(
                        child,
                        (ast.FunctionDef, ast.AsyncFunctionDef),
                    ):
                        methods.append(child.name)

                classes[node.name] = methods

        inventory.append(
            {
                "path": relative,
                "functions": functions,
                "classes": classes,
            }
        )

    save_json("python_ast_inventory.json", inventory)


def scan_retrieval_architecture() -> None:
    keywords = [
        "allowed_chunk_ids",
        "select_strict_sections",
        "candidate_k",
        "section_k",
        "top_k",
        "dense",
        "sparse",
        "colbert",
        "bm25",
        "rrf",
        "fusion",
        "rerank",
        "hierarch",
        "section_score",
        "parent_id",
        "previous_chunk",
        "next_chunk",
        "contextual",
        "embedding_text",
        "index_text",
        "query_expansion",
        "standalone_query",
        "followup",
        "coreference",
        "comparison",
        "balance",
        "evidence_role",
        "coverage_recovery",
    ]

    scan_roots = [
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "tests",
        PROJECT_ROOT / "config",
        PROJECT_ROOT / "scripts",
    ]

    accepted_extensions = {
        ".py",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".md",
    }

    matches: list[str] = []

    for root in scan_roots:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in accepted_extensions:
                continue

            if any(
                part in {
                    ".venv",
                    ".git",
                    ".audit",
                    "__pycache__",
                }
                for part in path.parts
            ):
                continue

            try:
                lines = path.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).splitlines()
            except OSError:
                continue

            for number, line in enumerate(lines, start=1):
                lowered = line.lower()

                if any(keyword.lower() in lowered for keyword in keywords):
                    relative = path.relative_to(PROJECT_ROOT)
                    matches.append(
                        f"{relative}:{number}: {line.rstrip()}"
                    )

    save_text(
        "retrieval_architecture_matches.txt",
        "\n".join(matches),
    )


def inspect_python_environment() -> None:
    distributions = [
        "FlagEmbedding",
        "torch",
        "transformers",
        "sentence-transformers",
        "faiss-cpu",
        "faiss-gpu",
        "bm25s",
        "numpy",
        "scipy",
        "ollama",
        "pydantic",
        "pytest",
        "ruff",
    ]

    package_versions: dict[str, str] = {}

    for distribution in distributions:
        try:
            package_versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            package_versions[distribution] = "not installed"

    environment = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": package_versions,
    }

    bge_api: dict[str, Any] = {}

    try:
        import FlagEmbedding  # type: ignore

        bge_api["module_file"] = getattr(
            FlagEmbedding,
            "__file__",
            "unknown",
        )

        from FlagEmbedding import BGEM3FlagModel  # type: ignore

        bge_api["class"] = str(BGEM3FlagModel)

        try:
            bge_api["constructor_signature"] = str(
                inspect.signature(BGEM3FlagModel)
            )
        except Exception as exc:
            bge_api["constructor_signature_error"] = str(exc)

        for method_name in [
            "__init__",
            "encode",
            "encode_queries",
            "encode_corpus",
            "compute_score",
            "compute_colbert_score",
        ]:
            method = getattr(BGEM3FlagModel, method_name, None)

            if method is None:
                continue

            try:
                bge_api[method_name] = str(inspect.signature(method))
            except Exception as exc:
                bge_api[f"{method_name}_error"] = str(exc)

        bge_api["interesting_attributes"] = sorted(
            attribute
            for attribute in dir(BGEM3FlagModel)
            if any(
                token in attribute.lower()
                for token in [
                    "encode",
                    "sparse",
                    "colbert",
                    "score",
                    "dense",
                ]
            )
        )

    except Exception:
        bge_api["import_error"] = traceback.format_exc()

    environment["bge_m3_api"] = bge_api
    save_json("python_environment.json", environment)


def find_active_kb() -> Path | None:
    excluded = {
        ".venv",
        ".git",
        ".audit",
        "__pycache__",
        "node_modules",
    }

    for current_root, directories, _files in os.walk(PROJECT_ROOT):
        directories[:] = [
            directory
            for directory in directories
            if directory not in excluded
        ]

        current = Path(current_root)

        if current.name == ACTIVE_KB_NAME:
            return current

    return None


def flatten_keys(
    value: Any,
    prefix: str = "",
    depth: int = 0,
) -> list[str]:
    if depth > 2 or not isinstance(value, dict):
        return []

    keys: list[str] = []

    for key, child in value.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        keys.append(full_key)

        if isinstance(child, dict):
            keys.extend(
                flatten_keys(
                    child,
                    prefix=full_key,
                    depth=depth + 1,
                )
            )

    return keys


def compact_chunk_sample(record: dict[str, Any]) -> dict[str, Any]:
    selected: dict[str, Any] = {}

    important_fields = [
        "id",
        "chunk_id",
        "parent_id",
        "previous_chunk_id",
        "next_chunk_id",
        "document",
        "document_title",
        "source",
        "source_file",
        "filename",
        "page",
        "pages",
        "chapter",
        "section",
        "subsection",
        "chunk_type",
        "domain",
        "domains",
        "text",
        "content",
        "chunk_text",
        "embedding_text",
        "index_text",
        "contextual_text",
    ]

    for field in important_fields:
        if field not in record:
            continue

        value = record[field]

        if isinstance(value, str) and len(value) > 700:
            value = value[:700] + " ...[truncated]"

        selected[field] = value

    if "metadata" in record and isinstance(record["metadata"], dict):
        selected["metadata"] = record["metadata"]

    return selected


def audit_active_kb() -> None:
    kb_path = find_active_kb()

    if kb_path is None:
        save_json(
            "kb_audit.json",
            {
                "active_kb_name": ACTIVE_KB_NAME,
                "found": False,
            },
        )
        return

    files: list[dict[str, Any]] = []

    for path in sorted(kb_path.rglob("*")):
        if not path.is_file():
            continue

        files.append(
            {
                "path": str(path.relative_to(kb_path)),
                "size_bytes": path.stat().st_size,
            }
        )

    jsonl_candidates = sorted(
        [
            path
            for path in kb_path.rglob("*.jsonl")
            if "chunk" in path.name.lower()
        ],
        key=lambda item: item.stat().st_size,
        reverse=True,
    )

    chunks_report: dict[str, Any] = {
        "found": False,
    }

    if jsonl_candidates:
        chunks_file = jsonl_candidates[0]
        total_records = 0
        invalid_json = 0
        field_counts: Counter[str] = Counter()
        nested_field_counts: Counter[str] = Counter()
        text_field_counts: Counter[str] = Counter()
        samples: list[dict[str, Any]] = []
        document_samples: dict[str, dict[str, Any]] = {}
        seen_chunk_ids: set[str] = set()
        duplicate_chunk_ids = 0

        with chunks_file.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as handle:
            for line in handle:
                if not line.strip():
                    continue

                total_records += 1

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    invalid_json += 1
                    continue

                if not isinstance(record, dict):
                    continue

                field_counts.update(record.keys())
                nested_field_counts.update(flatten_keys(record))

                for key, value in record.items():
                    key_lower = key.lower()

                    if (
                        isinstance(value, str)
                        and any(
                            token in key_lower
                            for token in [
                                "text",
                                "content",
                                "context",
                            ]
                        )
                    ):
                        text_field_counts[key] += 1

                chunk_id = str(
                    record.get("chunk_id")
                    or record.get("id")
                    or ""
                )

                if chunk_id:
                    if chunk_id in seen_chunk_ids:
                        duplicate_chunk_ids += 1
                    else:
                        seen_chunk_ids.add(chunk_id)

                if len(samples) < 8:
                    samples.append(compact_chunk_sample(record))

                document_name = str(
                    record.get("document")
                    or record.get("document_title")
                    or record.get("source_file")
                    or record.get("filename")
                    or ""
                )

                if document_name and document_name not in document_samples:
                    document_samples[document_name] = compact_chunk_sample(
                        record
                    )

        chunks_report = {
            "found": True,
            "file": str(chunks_file.relative_to(kb_path)),
            "total_records": total_records,
            "invalid_json_lines": invalid_json,
            "duplicate_chunk_ids": duplicate_chunk_ids,
            "top_level_field_counts": dict(
                field_counts.most_common()
            ),
            "nested_field_counts": dict(
                nested_field_counts.most_common()
            ),
            "text_field_counts": dict(
                text_field_counts.most_common()
            ),
            "samples": samples,
            "one_sample_per_document": document_samples,
        }

    faiss_report: list[dict[str, Any]] = []
    possible_indexes = [
        path
        for path in kb_path.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() in {".faiss", ".index"}
            or "faiss" in path.name.lower()
        )
    ]

    try:
        import faiss  # type: ignore

        for index_file in possible_indexes[:5]:
            record: dict[str, Any] = {
                "file": str(index_file.relative_to(kb_path)),
                "size_bytes": index_file.stat().st_size,
            }

            try:
                index = faiss.read_index(str(index_file))
                record["dimension"] = int(index.d)
                record["ntotal"] = int(index.ntotal)
                record["index_type"] = type(index).__name__
            except Exception as exc:
                record["read_error"] = str(exc)

            faiss_report.append(record)

    except Exception:
        faiss_report.append(
            {
                "faiss_import_error": traceback.format_exc(),
                "candidate_files": [
                    str(path.relative_to(kb_path))
                    for path in possible_indexes
                ],
            }
        )

    save_json(
        "kb_audit.json",
        {
            "active_kb_name": ACTIVE_KB_NAME,
            "found": True,
            "path": str(kb_path),
            "files": files,
            "chunks": chunks_report,
            "faiss_indexes": faiss_report,
        },
    )


def write_smoke_query_manifest() -> None:
    queries = [
        {
            "id": "definition_fc",
            "type": "definition",
            "question": "Qu’est-ce qu’un évaporateur à circulation forcée ?",
            "required_roles": [
                "definition",
                "pump_circulation",
                "heating_surface",
            ],
        },
        {
            "id": "explanation_fc",
            "type": "explanation",
            "question": (
                "Why is forced circulation used in an industrial evaporator?"
            ),
            "required_roles": [
                "mechanism",
                "benefit",
                "limitation",
            ],
        },
        {
            "id": "process_flow",
            "type": "process_flow",
            "question": (
                "Describe step by step the path of phosphoric acid through "
                "a forced-circulation evaporator, from the feed inlet to "
                "the concentrated product outlet."
            ),
            "required_roles": [
                "feed_inlet",
                "conical_bottom",
                "pump_heat_exchanger",
                "return_flash_chamber",
                "product_outlet",
            ],
        },
        {
            "id": "comparison_fc_falling_film",
            "type": "comparison",
            "question": (
                "Compare a forced-circulation evaporator with a "
                "falling-film evaporator for phosphoric acid concentration."
            ),
            "required_roles": [
                "equipment_a",
                "equipment_b",
                "comparison_criterion",
            ],
        },
        {
            "id": "p2o5_balance",
            "type": "balance",
            "question": (
                "Établis le bilan de P2O5 autour d’un évaporateur "
                "d’acide phosphorique en régime permanent."
            ),
            "required_roles": [
                "species_conservation",
                "feed_term",
                "product_term",
                "loss_or_entrainment_term",
            ],
        },
        {
            "id": "energy_balance",
            "type": "balance",
            "question": (
                "Establish the steady-state energy balance of a "
                "forced-circulation evaporator and define every term used."
            ),
            "required_roles": [
                "energy_conservation",
                "steam_or_heat_input",
                "feed_enthalpy",
                "product_enthalpy",
                "vapor_enthalpy",
            ],
        },
        {
            "id": "followup_pump",
            "type": "followup",
            "question": "Et pourquoi est-elle nécessaire ?",
            "required_entity": "pompe de circulation",
        },
        {
            "id": "followup_pump_en",
            "type": "followup",
            "question": (
                "How does it send the liquid back to the flash chamber?"
            ),
            "required_entity": "circulation pump",
        },
        {
            "id": "arabic_vapor_body",
            "type": "explanation",
            "question": (
                "ما هو دور غرفة التبخير في فصل البخار عن الحمض؟"
            ),
            "required_roles": [
                "vapor_body",
                "vapor_liquid_separation",
            ],
        },
    ]

    save_json("retrieval_smoke_queries.json", queries)


def write_summary() -> None:
    summary = f"""
# Retriever v4 audit

Generated: {datetime.now().isoformat()}
Project: {PROJECT_ROOT}
Python: {sys.executable}
Active KB requested: {ACTIVE_KB_NAME}

## Collected

- Current Git status and diff
- Source and test snapshot
- Source hashes
- Python AST inventory
- Retrieval architecture keyword map
- Python and package versions
- BGE-M3 API signatures available locally
- Active knowledge-base schema
- Chunk metadata and text-field samples
- FAISS dimensions and vector counts when readable
- Test collection
- Ruff and compilation diagnostics
- Retrieval-only smoke-query manifest

## Intentionally excluded

- PDF documents
- Raw datasets
- Vector binaries
- model weights
- `.venv`
- `.env`
- secrets
- Ollama model files

This archive is an audit bundle only. It does not modify the retrieval code,
the chunks, or the active indexes.
""".strip()

    save_text("AUDIT_SUMMARY.md", summary)


def main() -> None:
    print(f"Audit directory: {AUDIT_ROOT}")

    copy_source_snapshot()
    build_source_inventory()
    build_python_ast_inventory()
    scan_retrieval_architecture()
    inspect_python_environment()
    audit_active_kb()
    write_smoke_query_manifest()

    commands = {
        "git_status": ["git", "status", "--short"],
        "git_last_commit": [
            "git",
            "log",
            "-1",
            "--decorate",
            "--oneline",
        ],
        "git_diff_stat": ["git", "diff", "--stat"],
        "git_diff_check": ["git", "diff", "--check"],
        "git_diff_retrieval": [
            "git",
            "diff",
            "--",
            "src/phosprocess/rag",
            "src/phosprocess/retrieval",
            "tests",
            "config",
        ],
        "python_version": [sys.executable, "--version"],
        "pip_freeze": [
            sys.executable,
            "-m",
            "pip",
            "freeze",
        ],
        "compileall": [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "src",
            "tests",
        ],
        "pytest_collect": [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
        ],
    }

    ruff_path = PROJECT_ROOT / ".venv" / "Scripts" / "ruff.exe"
    if ruff_path.exists():
        commands["ruff_check"] = [
            str(ruff_path),
            "check",
            "src",
            "tests",
        ]

    for name, command in commands.items():
        print(f"Running: {name}")
        run_command(name, command)

    for optional_name, optional_command in [
        ("nvidia_smi", ["nvidia-smi"]),
        ("ollama_list", ["ollama", "list"]),
    ]:
        print(f"Running optional command: {optional_name}")
        run_command(
            optional_name,
            optional_command,
            timeout=60,
        )

    write_summary()

    zip_path = (
        PROJECT_ROOT.parent
        / f"phosprocess_retriever_v4_audit_{STAMP}.zip"
    )

    with zipfile.ZipFile(
        zip_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in AUDIT_ROOT.rglob("*"):
            if path.is_file():
                archive.write(
                    path,
                    arcname=path.relative_to(AUDIT_ROOT),
                )

    print()
    print("AUDIT COMPLETED")
    print(f"Audit folder : {AUDIT_ROOT}")
    print(f"Archive      : {zip_path}")


if __name__ == "__main__":
    main()
