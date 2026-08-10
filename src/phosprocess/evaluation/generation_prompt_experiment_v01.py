# ruff: noqa: E501
"""Phase-11 same-evidence generation-prompt experiment."""

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
from phosprocess.evaluation.legacy_answer_validation_service import (
    AnswerValidationService,
    _EvidenceRequirement,
    _EvidenceRequirementPlan,
)
from phosprocess.evaluation.legacy_generation_prompts import (
    REPAIR_SYSTEM_PROMPT,
    build_repair_prompt,
)
from phosprocess.llm.ollama_client import OllamaLLM
from phosprocess.observability.latency import OllamaCallMetrics
from phosprocess.rag.citations import CitationValidationError
from phosprocess.rag.generation_service import GenerationService
from phosprocess.rag.language import ResponseLanguage
from phosprocess.rag.orchestrator import load_runtime_config
from phosprocess.rag.prompts import build_quality_prompt_package
from phosprocess.rag.question_classifier import classify_question
from phosprocess.retrieval.evidence_bundle import EvidenceBundle

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PHASE10_OUTPUT = PROJECT_ROOT / "data/evaluation/generation_baseline/v0.1"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/evaluation/generation_prompt_experiment/v0.1"
VARIANT = "grounded_evidence_utilization_v1"
PROMPT_FILE = PROJECT_ROOT / "src/phosprocess/rag/prompts.py"
EVALUATOR_FILES = (
    "src/phosprocess/evaluation/generation_prompt_experiment_v01.py",
    "src/phosprocess/evaluation/generation_baseline_analysis_v01.py",
    "src/phosprocess/evaluation/generation_manual_review_v01.py",
)
_PLAN_RE = re.compile(
    r"^R(?P<index>\d+) \| sources=(?P<sources>[\d,]+)"
    r"(?: \| order=(?P<order>-?\d+))? \| (?P<description>.+)$"
)


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
        raise RuntimeError(f"unexpected frozen prompt calls for {row['id']}")
    return calls[0]


def _new_system_prompt(row: dict[str, Any], bundles: list[EvidenceBundle]) -> str:
    system, _package = build_quality_prompt_package(
        row["question"],
        bundles,
        response_language=_language(row["language"]),
        classification=classify_question(row["question"]),
        json_output=False,
        prompt_variant=VARIANT,
    )
    return system


def _parse_requirement_plan(user_prompt: str) -> _EvidenceRequirementPlan | None:
    marker = "PRECOMPUTED EVIDENCE COVERAGE PLAN\n"
    if marker not in user_prompt:
        return None
    block = user_prompt.split(marker, 1)[1].split("\n\nUse this plan only", 1)[0]
    lines = block.splitlines()
    if not lines or not lines[0].startswith("Focus: "):
        raise RuntimeError("invalid frozen evidence requirement plan")
    requirements: list[_EvidenceRequirement] = []
    for line in lines[1:]:
        match = _PLAN_RE.fullmatch(line)
        if match is None:
            raise RuntimeError(f"invalid frozen requirement line: {line}")
        requirements.append(
            _EvidenceRequirement(
                description=match.group("description"),
                source_numbers=[int(value) for value in match.group("sources").split(",")],
                sequence_index=(
                    int(match.group("order")) if match.group("order") is not None else None
                ),
            )
        )
    return _EvidenceRequirementPlan(focus=lines[0][7:], requirements=requirements)


class _ReplayGenerationService(GenerationService, AnswerValidationService):
    """Use the unchanged production generation validation without retrieval."""

    def __init__(self) -> None:
        self.runtime_config = load_runtime_config()
        self.llm = OllamaLLM(self.runtime_config.ollama)

    def close(self) -> None:
        self.llm.close()


def _source_from_bundle(bundle: EvidenceBundle) -> dict[str, Any]:
    pages = list(range(bundle.page_start, bundle.page_end + 1))
    return {
        "source_number": bundle.source_number,
        "chunk_id": bundle.anchor_chunk_ids[0],
        "document_name": bundle.filename,
        "pages": pages,
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


def _run_one(service: _ReplayGenerationService, baseline: dict[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(baseline)
    bundles = [EvidenceBundle.model_validate(item) for item in row["retrieval"]["evidence_bundles"]]
    initial = _initial_prompt_call(row)
    user_prompt = initial["messages"][1]["content"]
    baseline_user_hash = _sha256_bytes(user_prompt.encode("utf-8"))
    if baseline_user_hash != _sha256_bytes(initial["messages"][1]["content"].encode("utf-8")):
        raise RuntimeError("frozen user prompt changed")
    requirement_plan = _parse_requirement_plan(user_prompt)
    system_prompt = _new_system_prompt(row, bundles)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_calls: list[dict[str, Any]] = []
    started = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    first_token_ms: float | None = None
    repair_attempted = False
    repair_reason: str | None = None
    answer = ""
    citations: list[int] = []
    insufficient = False
    failure: str | None = None

    for attempt_index, attempt in enumerate(("initial", "repair")):
        call = OllamaCallMetrics(
            call_type="generation_main" if attempt == "initial" else "citation_repair",
            model=service.runtime_config.ollama.model,
            streaming=True,
        )
        fragments: list[str] = []
        active_messages = [dict(message) for message in messages]
        try:
            fragments.extend(
                service.llm.stream_chat(
                    active_messages,
                    call_type=call.call_type,
                    telemetry=call,
                )
            )
        finally:
            prompt_calls.append(
                {
                    "messages": active_messages,
                    "call_type": call.call_type,
                    "model": service.runtime_config.ollama.model,
                    "prompt_sha256": _sha256_bytes(
                        _json_dump(active_messages).encode("utf-8")
                    ),
                    "telemetry": call.to_dict(),
                }
            )
        if first_token_ms is None and call.time_to_first_token_ms is not None:
            first_token_ms = call.time_to_first_token_ms
        answer = "".join(fragments).strip()
        try:
            service._reject_likely_truncation(
                answer,
                generated_token_count=call.generated_token_count,
            )
            service._validate_answer_semantics(
                question=row["question"],
                answer=answer,
                evidence_bundles=bundles,
                lexical_rejection="",
                requirement_plan=requirement_plan,
            )
            citations, insufficient = service._validate_answer(
                answer=answer,
                available_source_count=len(bundles),
                attempt=attempt,
            )
        except CitationValidationError as error:
            if attempt_index == 1:
                failure = str(error)
                break
            repair_attempted = True
            repair_reason = str(error)
            messages = [
                {
                    "role": "system",
                    "content": (
                        REPAIR_SYSTEM_PROMPT
                        + "\nPreserve the existing answer language: "
                        + _language(row["language"]).prompt_name
                        + "."
                    ),
                },
                {
                    "role": "user",
                    "content": build_repair_prompt(
                        original_prompt=user_prompt,
                        invalid_output=answer,
                        rejection_reason=str(error),
                        json_output=False,
                    ),
                },
            ]
            continue
        break

    duration_ms = (time.perf_counter() - started) * 1000.0
    row["phase11"] = {
        "variant": VARIANT,
        "same_question": row["question"] == baseline["question"],
        "same_evidence": row["retrieval"]["serialized_evidence_context"]
        == baseline["retrieval"]["serialized_evidence_context"],
        "same_user_prompt": True,
        "baseline_user_prompt_sha256": baseline_user_hash,
        "new_system_prompt_sha256": _sha256_bytes(system_prompt.encode("utf-8")),
    }
    row["started_at"] = started_at
    row["completed_at"] = datetime.now(UTC).isoformat()
    row["prompt_calls"] = prompt_calls
    row["hidden_chain_of_thought_stored"] = False
    row["status"] = "completed" if failure is None else "error"
    row["error"] = None if failure is None else {"message": failure, "metadata": {}}
    response = copy.deepcopy(baseline["response"])
    response["answer"] = answer
    response["cited_source_numbers"] = citations
    response["insufficient_context"] = insufficient
    bundle_by_number = {bundle.source_number: bundle for bundle in bundles}
    response["sources"] = [
        _source_from_bundle(bundle_by_number[number])
        for number in citations
        if number in bundle_by_number
    ]
    response["timings"]["hybrid_ms"] = 0.0
    response["timings"]["reranking_ms"] = 0.0
    response["timings"]["generation_ms"] = duration_ms
    response["timings"]["total_ms"] = duration_ms
    response["timings"]["first_token_ms"] = first_token_ms
    latency = response.get("latency") or {}
    telemetry = [call["telemetry"] for call in prompt_calls]
    latency.update(
        {
            "hybrid_search_ms": 0.0,
            "reranking_ms": 0.0,
            "total_ms": duration_ms,
            "turn_time_to_first_token_ms": first_token_ms,
            "ollama_call_count": len(telemetry),
            "prompt_character_count": sum(
                len(message["content"]) for message in prompt_calls[0]["messages"]
            ),
            "estimated_prompt_tokens": telemetry[0].get("estimated_prompt_tokens"),
            "generated_character_count": len(answer),
            "generated_token_count": sum(
                int(item.get("generated_token_count") or 0) for item in telemetry
            ),
            "repair_attempted": repair_attempted,
            "repair_reason": repair_reason,
            "citations": citations,
            "displayed_source_count": len(citations),
            "ollama_calls": telemetry,
        }
    )
    response["latency"] = latency
    row["response"] = response if failure is None else None
    return row


def _phase10_rows(name: str) -> list[dict[str, Any]]:
    return read_jsonl(PHASE10_OUTPUT / name)


def _prompt_texts(rows: list[dict[str, Any]], *, variant: str) -> dict[str, str]:
    texts: dict[str, str] = {}
    for language in ("fr", "en", "ar"):
        row = next(item for item in rows if item["language"] == language)
        if variant == "baseline":
            texts[language] = _initial_prompt_call(row)["messages"][0]["content"]
        else:
            bundles = [
                EvidenceBundle.model_validate(item)
                for item in row["retrieval"]["evidence_bundles"]
            ]
            texts[language] = _new_system_prompt(row, bundles)
    return texts


def freeze_experiment(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Freeze references after the Phase-10 baseline was verified pre-change."""

    path = output_directory / "experiment_freeze.json"
    if path.exists():
        raise RuntimeError("Phase-11 experiment is already frozen")
    output_directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        PHASE10_OUTPUT / "questions_snapshot.jsonl",
        output_directory / "questions_snapshot.jsonl",
    )
    shutil.copyfile(
        PHASE10_OUTPUT / "evaluation_protocol.json",
        output_directory / "evaluation_protocol.json",
    )
    baseline_manifest = json.loads(
        (PHASE10_OUTPUT / "baseline_manifest.json").read_text(encoding="utf-8")
    )
    dev_rows = _phase10_rows("dev_generation_results.jsonl")
    baseline_prompts = _prompt_texts(dev_rows, variant="baseline")
    new_prompts = _prompt_texts(dev_rows, variant=VARIANT)
    for row in dev_rows:
        initial = _initial_prompt_call(row)
        if initial["messages"][0]["content"] != baseline_prompts[row["language"]]:
            raise RuntimeError(f"Phase-10 baseline prompt drift within language for {row['id']}")
    manifest = {
        "phase": 11,
        "frozen_at": datetime.now(UTC).isoformat(),
        "baseline_sha256": baseline_manifest["baseline_sha256"],
        "phase10_baseline_manifest_sha256": _sha256_file(
            PHASE10_OUTPUT / "baseline_manifest.json"
        ),
        "phase10_dev_results_sha256": _sha256_file(
            PHASE10_OUTPUT / "dev_generation_results.jsonl"
        ),
        "phase10_holdout_results_sha256": _sha256_file(
            PHASE10_OUTPUT / "holdout_generation_results.jsonl"
        ),
        "dataset_sha256": baseline_manifest["dataset"]["questions_sha256"],
        "qwen": baseline_manifest["qwen"],
        "active_kb": baseline_manifest["active_kb"],
        "same_evidence_replay": True,
        "production_baseline_verified_before_prompt_change": True,
        "variant": VARIANT,
        "baseline_prompt_texts": baseline_prompts,
        "new_prompt_texts": new_prompts,
        "baseline_prompt_sha256": _sha256_bytes(
            _json_dump(baseline_prompts).encode("utf-8")
        ),
        "new_prompt_sha256": _sha256_bytes(_json_dump(new_prompts).encode("utf-8")),
        "prompt_file_sha256": _sha256_file(PROMPT_FILE),
        "evaluator_file_sha256": {
            item: _sha256_file(PROJECT_ROOT / item) for item in EVALUATOR_FILES
        },
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
    if _sha256_file(PROMPT_FILE) != manifest["prompt_file_sha256"]:
        raise RuntimeError("Phase-11 prompt file drift detected")
    if _sha256_file(PHASE10_OUTPUT / "dev_generation_results.jsonl") != manifest[
        "phase10_dev_results_sha256"
    ]:
        raise RuntimeError("Phase-10 DEV evidence/results drift detected")
    return manifest


def _refresh_combined(output_directory: Path) -> None:
    rows: list[dict[str, Any]] = []
    for name in (
        "dev_generation_results.jsonl",
        "holdout_generation_results.jsonl",
        "followup_results.jsonl",
    ):
        path = output_directory / name
        if path.exists():
            rows.extend(read_jsonl(path))
    write_jsonl(output_directory / "generation_results.jsonl", rows)


def run_replay(
    split: str,
    output_directory: Path = DEFAULT_OUTPUT,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Replay one split using frozen prompts/evidence and no retrieval."""

    manifest = _verify_freeze(output_directory)
    if split == "dev":
        source_name = "dev_generation_results.jsonl"
        target_name = source_name
    elif split == "final_holdout":
        freeze_path = output_directory / "prompt_freeze_manifest.json"
        if not freeze_path.exists():
            raise RuntimeError("freeze the successful DEV prompt before opening holdout")
        marker_path = output_directory / "holdout_open_manifest.json"
        if marker_path.exists():
            raise RuntimeError("Phase-11 FINAL HOLDOUT was already opened")
        marker_path.write_text(
            json.dumps(
                {
                    "opened_at": datetime.now(UTC).isoformat(),
                    "opening_count": 1,
                    "new_prompt_sha256": manifest["new_prompt_sha256"],
                    "prompt_freeze_sha256": _sha256_file(freeze_path),
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
    target_path = output_directory / target_name
    completed = {row["id"] for row in read_jsonl(target_path)} if target_path.exists() else set()
    pending = [row for row in source_rows if row["id"] not in completed]
    if limit is not None:
        pending = pending[:limit]
    service = _ReplayGenerationService()
    written = 0
    try:
        for index, row in enumerate(pending, 1):
            result = _run_one(service, row)
            with target_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1
            print(
                json.dumps(
                    {
                        "progress": f"{index}/{len(pending)}",
                        "id": row["id"],
                        "status": result["status"],
                        "answer_chars": len((result.get("response") or {}).get("answer", "")),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        service.close()
    all_rows = read_jsonl(target_path) if target_path.exists() else []
    complete = len({row["id"] for row in all_rows}) == len(source_rows)
    if split == "final_holdout" and complete:
        marker_path = output_directory / "holdout_open_manifest.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker.update(
            {
                "status": "complete",
                "completed_at": datetime.now(UTC).isoformat(),
                "question_count": len(source_rows),
            }
        )
        marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    _refresh_combined(output_directory)
    return {
        "split": split,
        "written": written,
        "completed_records": len(all_rows),
        "expected_records": len(source_rows),
        "complete": complete,
        "retrieval_calls": 0,
    }


def freeze_prompt(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Freeze the sole prompt variant after a successful DEV decision."""

    manifest = _verify_freeze(output_directory)
    decision = json.loads((output_directory / "dev_decision.json").read_text(encoding="utf-8"))
    if not decision.get("accepted_for_holdout"):
        raise RuntimeError("DEV decision did not accept the prompt for holdout")
    path = output_directory / "prompt_freeze_manifest.json"
    if path.exists():
        raise RuntimeError("Phase-11 prompt is already frozen")
    value = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "variant": VARIANT,
        "prompt_sha256": manifest["new_prompt_sha256"],
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


def run_followups(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Replay the seven Phase-10 follow-ups as a read-only diagnostic."""

    if not (output_directory / "prompt_freeze_manifest.json").exists():
        raise RuntimeError("select and freeze the DEV prompt before follow-up diagnostics")
    source_rows = _phase10_rows("followup_results.jsonl")
    target_path = output_directory / "followup_results.jsonl"
    if target_path.exists():
        raise RuntimeError("Phase-11 follow-up diagnostic was already run")
    service = _ReplayGenerationService()
    results: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(source_rows, 1):
            result = _run_one(service, row)
            results.append(result)
            print(json.dumps({"progress": f"{index}/{len(source_rows)}", "id": row["id"]}), flush=True)
    finally:
        service.close()
    write_jsonl(target_path, results)
    _refresh_combined(output_directory)
    return {"written": len(results), "retrieval_calls": 0, "selection_metric": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run-dev", action="store_true")
    parser.add_argument("--freeze-prompt", action="store_true")
    parser.add_argument("--open-holdout", action="store_true")
    parser.add_argument("--run-followups", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.freeze:
        result = freeze_experiment(args.output)
    elif args.run_dev:
        result = run_replay("dev", args.output, limit=args.limit)
    elif args.freeze_prompt:
        result = freeze_prompt(args.output)
    elif args.open_holdout:
        result = run_replay("final_holdout", args.output, limit=args.limit)
    elif args.run_followups:
        result = run_followups(args.output)
    else:
        parser.error("choose one Phase-11 action")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
