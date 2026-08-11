# ruff: noqa: E501
"""Phase-12 structured EvidencePlanner same-evidence experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from phosprocess.evaluation.context_engine_v01 import read_jsonl, write_jsonl
from phosprocess.evaluation.evidence_planner import (
    PLANNER_SYSTEM_PROMPT,
    EvidencePlan,
    EvidencePlanner,
    evidence_plan_transport_schema,
)
from phosprocess.llm.ollama_client import OllamaLLM
from phosprocess.observability.latency import OllamaCallMetrics
from phosprocess.rag.citations import is_controlled_insufficient_answer
from phosprocess.rag.language import ResponseLanguage
from phosprocess.rag.orchestrator import load_runtime_config
from phosprocess.rag.prompts import build_quality_prompt_package
from phosprocess.rag.question_classifier import classify_question
from phosprocess.retrieval.evidence_bundle import EvidenceBundle

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PHASE10_OUTPUT = PROJECT_ROOT / "data/evaluation/generation_baseline/v0.1"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/evaluation/structured_evidence_planning/v0.1"
PLANNER_MODULE = PROJECT_ROOT / "src/phosprocess/evaluation/evidence_planner.py"
RUNNER_MODULE = Path(__file__).resolve()
VALID_CITATION_RE = re.compile(r"\[Source\s+([1-9]\d*)\]")
OLD_PLAN_MARKER = "\n\nPRECOMPUTED EVIDENCE COVERAGE PLAN\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _language(value: str) -> ResponseLanguage:
    return {
        "fr": ResponseLanguage.FRENCH,
        "en": ResponseLanguage.ENGLISH,
        "ar": ResponseLanguage.ARABIC,
    }[value]


def _initial_prompt_call(row: dict[str, Any]) -> dict[str, Any]:
    calls = [call for call in row["prompt_calls"] if call["call_type"] == "generation_main"]
    if len(calls) != 1 or len(calls[0]["messages"]) != 2:
        raise RuntimeError(f"unexpected Phase-10 prompt calls for {row['id']}")
    return calls[0]


def _without_embedded_phase10_plan(user_prompt: str) -> str:
    """Remove the embedded legacy requirements plan, preserving question/evidence."""

    if OLD_PLAN_MARKER not in user_prompt:
        raise RuntimeError("frozen Phase-10 prompt has no embedded requirements plan")
    return user_prompt.split(OLD_PLAN_MARKER, 1)[0].rstrip()


def _structured_plan_block(plan: EvidencePlan) -> str:
    return "\n\n".join(
        (
            "STRUCTURED EVIDENCE PLAN (JSON)",
            json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, indent=2),
            (
                "Use this plan as the answer-organization contract. The QUESTION determines "
                "relevance and the EVIDENCE BUNDLES remain the only factual authority. Cover "
                "each plan item once, cite its listed [Source N] locally, preserve sequence "
                "indexes, comparison sides, and case boundaries, and do not add unplanned "
                "documentary material. If insufficient_evidence is true, use the controlled "
                "insufficiency answer from the system instruction. Return only the answer."
            ),
        )
    )


def _source_from_bundle(bundle: EvidenceBundle) -> dict[str, Any]:
    return {
        "source_number": bundle.source_number,
        "chunk_id": bundle.anchor_chunk_ids[0],
        "document_name": bundle.filename,
        "pages": list(range(bundle.page_start, bundle.page_end + 1)),
        "section": bundle.section,
        "excerpt": bundle.display_text[:1200],
        "document_title": bundle.document_title,
        "filename": bundle.filename,
        "chapter": bundle.chapter,
        "page_start": bundle.page_start,
        "page_end": bundle.page_end,
        "anchor_chunk_id": bundle.anchor_chunk_ids[0],
        "anchor_chunk_ids": list(bundle.anchor_chunk_ids),
        "expanded_chunk_ids": list(bundle.supporting_chunk_ids),
        "supporting_chunk_ids": list(bundle.supporting_chunk_ids),
        "display_text": bundle.display_text,
        "parent_id": bundle.parent_id,
        "context_scope": bundle.context_scope.value,
        "best_anchor_score": bundle.best_anchor_score,
        "context_truncated": bundle.context_truncated,
        "selection_source": bundle.selection_provenance,
    }


class _ExperimentRuntime:
    def __init__(self) -> None:
        self.runtime_config = load_runtime_config()
        self.llm = OllamaLLM(self.runtime_config.ollama)
        self.planner = EvidencePlanner(self.llm)

    def close(self) -> None:
        self.llm.close()


def _run_one(runtime: _ExperimentRuntime, baseline: dict[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(baseline)
    bundles = [EvidenceBundle.model_validate(item) for item in row["retrieval"]["evidence_bundles"]]
    initial = _initial_prompt_call(row)
    baseline_system = initial["messages"][0]["content"]
    evidence_user_prompt = _without_embedded_phase10_plan(
        initial["messages"][1]["content"]
    )
    started = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    planner_record: dict[str, Any] | None = None
    prompt_calls: list[dict[str, Any]] = []
    response: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    status = "completed"

    try:
        execution = runtime.planner.plan(
            question=row["question"],
            evidence_bundles=bundles,
        )
        planner_record = {
            "plan": execution.plan.model_dump(mode="json"),
            "raw_output": execution.raw_output,
            "latency_ms": execution.latency_ms,
            "system_prompt_sha256": execution.system_prompt_sha256,
            "user_prompt_sha256": execution.user_prompt_sha256,
            "call_count": 1,
            "repair_attempted": False,
            "verifier_attempted": False,
        }
    except Exception as planner_error:
        status = "planner_error"
        error = {
            "stage": "planner",
            "type": type(planner_error).__name__,
            "message": str(planner_error),
        }
    else:
        generation_user_prompt = (
            evidence_user_prompt + "\n\n" + _structured_plan_block(execution.plan)
        )
        messages = [
            {"role": "system", "content": baseline_system},
            {"role": "user", "content": generation_user_prompt},
        ]
        call = OllamaCallMetrics(
            call_type="structured_plan_generation",
            model=runtime.runtime_config.ollama.model,
            streaming=True,
        )
        generation_started = time.perf_counter()
        fragments: list[str] = []
        try:
            fragments.extend(
                runtime.llm.stream_chat(
                    messages,
                    call_type=call.call_type,
                    telemetry=call,
                )
            )
        except Exception as generation_error:
            status = "generation_error"
            error = {
                "stage": "generation",
                "type": type(generation_error).__name__,
                "message": str(generation_error),
            }
        finally:
            prompt_calls.append(
                {
                    "messages": messages,
                    "call_type": call.call_type,
                    "model": runtime.runtime_config.ollama.model,
                    "prompt_sha256": _sha256_bytes(_json_dump(messages).encode("utf-8")),
                    "telemetry": call.to_dict(),
                }
            )
        if status == "completed":
            answer = "".join(fragments).strip()
            citation_numbers = sorted(
                {
                    int(value)
                    for value in VALID_CITATION_RE.findall(answer)
                    if int(value) <= len(bundles)
                }
            )
            bundle_by_number = {bundle.source_number: bundle for bundle in bundles}
            generation_ms = (time.perf_counter() - generation_started) * 1000.0
            total_ms = (time.perf_counter() - started) * 1000.0
            response = copy.deepcopy(baseline["response"])
            response.update(
                {
                    "answer": answer,
                    "sources": [
                        _source_from_bundle(bundle_by_number[number])
                        for number in citation_numbers
                        if number in bundle_by_number
                    ],
                    "cited_source_numbers": citation_numbers,
                    "insufficient_context": is_controlled_insufficient_answer(answer),
                }
            )
            response["timings"] = {
                "hybrid_ms": 0.0,
                "reranking_ms": 0.0,
                "generation_ms": generation_ms,
                "total_ms": total_ms,
                "first_token_ms": call.time_to_first_token_ms,
            }
            response["latency"] = {
                "total_ms": total_ms,
                "planner_ms": execution.latency_ms,
                "generation_ms": generation_ms,
                "turn_time_to_first_token_ms": call.time_to_first_token_ms,
                "ollama_call_count": 2,
                "planner_call_count": 1,
                "generation_call_count": 1,
                "verifier_call_count": 0,
                "repair_call_count": 0,
                "prompt_character_count": sum(
                    len(message["content"]) for message in messages
                ),
                "estimated_prompt_tokens": call.estimated_prompt_tokens,
                "generated_character_count": len(answer),
                "generated_token_count": call.generated_token_count,
                "repair_attempted": False,
                "citations": citation_numbers,
                "displayed_source_count": len(citation_numbers),
                "ollama_calls": [call.to_dict()],
            }

    row.update(
        {
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "prompt_calls": prompt_calls,
            "status": status,
            "error": error,
            "response": response,
            "hidden_chain_of_thought_stored": False,
            "phase12": {
                "experiment": "structured_evidence_planning_v0.1",
                "same_question": row["question"] == baseline["question"],
                "same_evidence": (
                    row["retrieval"]["serialized_evidence_context"]
                    == baseline["retrieval"]["serialized_evidence_context"]
                ),
                "phase10_embedded_plan_removed": True,
                "planner": planner_record,
                "retrieval_calls": 0,
                "verifier_calls": 0,
                "repair_calls": 0,
            },
        }
    )
    return row


def _phase10_rows(filename: str) -> list[dict[str, Any]]:
    return read_jsonl(PHASE10_OUTPUT / filename)


def _baseline_system_prompts(rows: list[dict[str, Any]]) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for language in ("fr", "en", "ar"):
        row = next(item for item in rows if item["language"] == language)
        prompts[language] = _initial_prompt_call(row)["messages"][0]["content"]
    return prompts


def freeze_experiment(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Freeze Phase-10 A and the Phase-12 planner before DEV execution."""

    path = output_directory / "experiment_freeze.json"
    if path.exists():
        raise RuntimeError("Phase-12 experiment is already frozen")
    output_directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        PHASE10_OUTPUT / "questions_snapshot.jsonl",
        output_directory / "questions_snapshot.jsonl",
    )
    baseline_manifest = json.loads(
        (PHASE10_OUTPUT / "baseline_manifest.json").read_text(encoding="utf-8")
    )
    dev_rows = _phase10_rows("dev_generation_results.jsonl")
    baseline_systems = _baseline_system_prompts(dev_rows)
    for row in dev_rows:
        if _initial_prompt_call(row)["messages"][0]["content"] != baseline_systems[row["language"]]:
            raise RuntimeError(f"Phase-10 system prompt drift for {row['id']}")
        bundles = [EvidenceBundle.model_validate(item) for item in row["retrieval"]["evidence_bundles"]]
        current_system, package = build_quality_prompt_package(
            row["question"],
            bundles,
            response_language=_language(row["language"]),
            classification=classify_question(row["question"]),
            json_output=False,
        )
        if current_system != baseline_systems[row["language"]]:
            raise RuntimeError("current production default no longer matches Phase-10 baseline")
        if package.user_prompt != _without_embedded_phase10_plan(
            _initial_prompt_call(row)["messages"][1]["content"]
        ):
            raise RuntimeError(f"Phase-10 question/evidence prompt drift for {row['id']}")

    protocol = {
        "phase": 12,
        "mission": "structured EvidencePlanner before generation",
        "comparison": "Phase-10 baseline A versus Phase-12 B on identical DEV evidence",
        "planner_input": ["question", "EvidenceBundles"],
        "planner_output": "strict JSON EvidencePlan",
        "generation_input": ["question", "evidence", "evidence plan"],
        "planner_modes": ["sequence", "comparison", "multiple_cases", "simple"],
        "holdout_policy": "open once only after a strictly better DEV decision and planner freeze",
        "prohibited": {
            "evidence_judge": False,
            "verifier": False,
            "repair_loop": False,
            "new_retrieval": False,
            "rechunking": False,
            "fine_tuning": False,
        },
    }
    (output_directory / "evaluation_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "phase": 12,
        "frozen_at": datetime.now(UTC).isoformat(),
        "baseline_sha256": baseline_manifest["baseline_sha256"],
        "phase10_dev_results_sha256": _sha256_file(
            PHASE10_OUTPUT / "dev_generation_results.jsonl"
        ),
        "dataset_sha256": baseline_manifest["dataset"]["questions_sha256"],
        "qwen": baseline_manifest["qwen"],
        "active_kb": baseline_manifest["active_kb"],
        "baseline_system_prompts": baseline_systems,
        "baseline_system_prompt_sha256": _sha256_bytes(
            _json_dump(baseline_systems).encode("utf-8")
        ),
        "planner_system_prompt": PLANNER_SYSTEM_PROMPT,
        "planner_system_prompt_sha256": _sha256_bytes(
            PLANNER_SYSTEM_PROMPT.encode("utf-8")
        ),
        "planner_schema": EvidencePlan.model_json_schema(),
        "planner_schema_sha256": _sha256_bytes(
            _json_dump(EvidencePlan.model_json_schema()).encode("utf-8")
        ),
        "planner_transport_schema": evidence_plan_transport_schema(),
        "planner_transport_schema_sha256": _sha256_bytes(
            _json_dump(evidence_plan_transport_schema()).encode("utf-8")
        ),
        "planner_module_sha256": _sha256_file(PLANNER_MODULE),
        "runner_module_sha256": _sha256_file(RUNNER_MODULE),
        "same_evidence_replay": True,
        "production_default_matches_phase10": True,
        "verifier_enabled": False,
        "repair_enabled": False,
        "retrieval_enabled": False,
    }
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _verify_freeze(output_directory: Path) -> dict[str, Any]:
    manifest = json.loads(
        (output_directory / "experiment_freeze.json").read_text(encoding="utf-8")
    )
    checks = {
        "planner module": (_sha256_file(PLANNER_MODULE), manifest["planner_module_sha256"]),
        "runner module": (_sha256_file(RUNNER_MODULE), manifest["runner_module_sha256"]),
        "Phase-10 DEV": (
            _sha256_file(PHASE10_OUTPUT / "dev_generation_results.jsonl"),
            manifest["phase10_dev_results_sha256"],
        ),
    }
    drift = [name for name, (actual, expected) in checks.items() if actual != expected]
    if drift:
        raise RuntimeError(f"Phase-12 frozen experiment drift: {drift}")
    return manifest


def _refresh_combined(output_directory: Path) -> None:
    rows: list[dict[str, Any]] = []
    for filename in ("dev_generation_results.jsonl", "holdout_generation_results.jsonl"):
        path = output_directory / filename
        if path.exists():
            rows.extend(read_jsonl(path))
    write_jsonl(output_directory / "generation_results.jsonl", rows)


def run_split(
    split: str,
    output_directory: Path = DEFAULT_OUTPUT,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    manifest = _verify_freeze(output_directory)
    if split == "dev":
        source_name = "dev_generation_results.jsonl"
        target_name = source_name
    elif split == "final_holdout":
        planner_freeze = output_directory / "planner_freeze_manifest.json"
        if not planner_freeze.exists():
            raise RuntimeError("freeze a DEV-winning planner before opening holdout")
        marker = output_directory / "holdout_open_manifest.json"
        if marker.exists():
            raise RuntimeError("Phase-12 holdout was already opened")
        marker.write_text(
            json.dumps(
                {
                    "opened_at": datetime.now(UTC).isoformat(),
                    "opening_count": 1,
                    "planner_freeze_sha256": _sha256_file(planner_freeze),
                    "status": "in_progress",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        source_name = "holdout_generation_results.jsonl"
        target_name = source_name
    else:
        raise ValueError(split)

    source_rows = _phase10_rows(source_name)
    target = output_directory / target_name
    completed = {row["id"] for row in read_jsonl(target)} if target.exists() else set()
    pending = [row for row in source_rows if row["id"] not in completed]
    if limit is not None:
        pending = pending[:limit]
    runtime = _ExperimentRuntime()
    written = 0
    try:
        for index, baseline in enumerate(pending, 1):
            result = _run_one(runtime, baseline)
            with target.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1
            print(
                json.dumps(
                    {
                        "progress": f"{index}/{len(pending)}",
                        "id": baseline["id"],
                        "status": result["status"],
                        "plan_mode": (
                            ((result.get("phase12") or {}).get("planner") or {}).get("plan")
                            or {}
                        ).get("mode"),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
    finally:
        runtime.close()
    rows = read_jsonl(target) if target.exists() else []
    complete = len({row["id"] for row in rows}) == len(source_rows)
    if split == "final_holdout" and complete:
        marker = output_directory / "holdout_open_manifest.json"
        value = json.loads(marker.read_text(encoding="utf-8"))
        value.update(
            {
                "status": "complete",
                "completed_at": datetime.now(UTC).isoformat(),
                "question_count": len(source_rows),
            }
        )
        marker.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    _refresh_combined(output_directory)
    return {
        "split": split,
        "written": written,
        "record_count": len(rows),
        "expected": len(source_rows),
        "complete": complete,
        "retrieval_calls": 0,
        "verifier_calls": 0,
        "repair_calls": 0,
        "baseline_sha256": manifest["baseline_sha256"],
    }


def freeze_planner(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    manifest = _verify_freeze(output_directory)
    decision = json.loads((output_directory / "dev_decision.json").read_text(encoding="utf-8"))
    if not decision.get("accepted_for_holdout"):
        raise RuntimeError("Phase-12 DEV did not accept the planner")
    path = output_directory / "planner_freeze_manifest.json"
    if path.exists():
        raise RuntimeError("Phase-12 planner is already frozen")
    value = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "planner_prompt_sha256": manifest["planner_system_prompt_sha256"],
        "planner_schema_sha256": manifest["planner_schema_sha256"],
        "planner_transport_schema_sha256": manifest[
            "planner_transport_schema_sha256"
        ],
        "planner_module_sha256": manifest["planner_module_sha256"],
        "runner_module_sha256": manifest["runner_module_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "qwen": manifest["qwen"],
        "dev_results_sha256": _sha256_file(
            output_directory / "dev_generation_results.jsonl"
        ),
        "dev_decision_sha256": _sha256_file(output_directory / "dev_decision.json"),
        "holdout_seen": False,
    }
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run-dev", action="store_true")
    parser.add_argument("--freeze-planner", action="store_true")
    parser.add_argument("--open-holdout", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.freeze:
        result = freeze_experiment(args.output)
    elif args.run_dev:
        result = run_split("dev", args.output, limit=args.limit)
    elif args.freeze_planner:
        result = freeze_planner(args.output)
    elif args.open_holdout:
        result = run_split("final_holdout", args.output, limit=args.limit)
    else:
        parser.error("choose a Phase-12 action")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
