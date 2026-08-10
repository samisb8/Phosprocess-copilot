# ruff: noqa: E501
"""Final, single-candidate generation-model replay on frozen Phase-10 DEV.

Only the Ollama generation model changes. Retrieval, planning, semantic
verification and repair are never executed by this evaluation-only runner.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from phosprocess.evaluation.context_engine_v01 import read_jsonl, write_jsonl
from phosprocess.llm.ollama_client import OllamaLLM
from phosprocess.observability.latency import OllamaCallMetrics
from phosprocess.rag.citations import is_controlled_insufficient_answer
from phosprocess.rag.orchestrator import load_runtime_config
from phosprocess.retrieval.evidence_bundle import EvidenceBundle

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PHASE10_OUTPUT = PROJECT_ROOT / "data/evaluation/generation_baseline/v0.1"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/evaluation/final_model_capability/v0.1"
RUNNER_MODULE = Path(__file__).resolve()
BASELINE_MODEL = "qwen3:8b"
CANDIDATE_MODEL = "qwen3:14b"
VALID_CITATION_RE = re.compile(r"\[Source\s+([1-9]\d*)\]")


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


def _initial_prompt_call(row: dict[str, Any]) -> dict[str, Any]:
    calls = [call for call in row["prompt_calls"] if call["call_type"] == "generation_main"]
    if len(calls) != 1 or len(calls[0]["messages"]) != 2:
        raise RuntimeError(f"unexpected frozen Phase-10 prompt for {row['id']}")
    if calls[0]["model"] != BASELINE_MODEL:
        raise RuntimeError(f"unexpected Phase-10 model for {row['id']}")
    return calls[0]


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


class _CandidateRuntime:
    def __init__(self) -> None:
        baseline = load_runtime_config()
        candidate_ollama = replace(baseline.ollama, model=CANDIDATE_MODEL)
        self.config = replace(baseline, ollama=candidate_ollama)
        self.llm = OllamaLLM(candidate_ollama)

    def close(self) -> None:
        self.llm.close()


def _run_one(runtime: _CandidateRuntime, baseline: dict[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(baseline)
    initial = _initial_prompt_call(row)
    messages = copy.deepcopy(initial["messages"])
    bundles = [
        EvidenceBundle.model_validate(item)
        for item in row["retrieval"]["evidence_bundles"]
    ]
    call = OllamaCallMetrics(
        call_type="generation_main",
        model=CANDIDATE_MODEL,
        streaming=True,
    )
    started = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    fragments: list[str] = []
    error: dict[str, str] | None = None
    try:
        fragments.extend(
            runtime.llm.stream_chat(
                messages,
                call_type=call.call_type,
                telemetry=call,
            )
        )
    except Exception as generation_error:
        error = {
            "stage": "generation",
            "type": type(generation_error).__name__,
            "message": str(generation_error),
        }
    duration_ms = (time.perf_counter() - started) * 1000.0
    prompt_call = {
        "messages": messages,
        "call_type": call.call_type,
        "model": CANDIDATE_MODEL,
        "prompt_sha256": _sha256_bytes(_json_dump(messages).encode("utf-8")),
        "telemetry": call.to_dict(),
    }
    row.update(
        {
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "prompt_calls": [prompt_call],
            "hidden_chain_of_thought_stored": False,
            "status": "completed" if error is None else "generation_error",
            "error": error,
            "response": None,
            "final_model_experiment": {
                "baseline_model": BASELINE_MODEL,
                "candidate_model": CANDIDATE_MODEL,
                "same_question": row["question"] == baseline["question"],
                "same_evidence": (
                    row["retrieval"]["serialized_evidence_context"]
                    == baseline["retrieval"]["serialized_evidence_context"]
                ),
                "same_messages": messages == initial["messages"],
                "generation_calls": 1,
                "retrieval_calls": 0,
                "planner_calls": 0,
                "evidence_judge_calls": 0,
                "semantic_verifier_calls": 0,
                "repair_calls": 0,
            },
        }
    )
    if error is not None:
        return row

    answer = "".join(fragments).strip()
    citation_numbers = sorted(
        {
            int(value)
            for value in VALID_CITATION_RE.findall(answer)
            if 1 <= int(value) <= len(bundles)
        }
    )
    bundle_by_number = {bundle.source_number: bundle for bundle in bundles}
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
            "model_name": CANDIDATE_MODEL,
        }
    )
    response["timings"] = {
        "hybrid_ms": 0.0,
        "reranking_ms": 0.0,
        "generation_ms": duration_ms,
        "total_ms": duration_ms,
        "first_token_ms": call.time_to_first_token_ms,
    }
    response["latency"] = {
        "total_ms": duration_ms,
        "turn_time_to_first_token_ms": call.time_to_first_token_ms,
        "ollama_call_count": 1,
        "generation_call_count": 1,
        "retrieval_call_count": 0,
        "planner_call_count": 0,
        "evidence_judge_call_count": 0,
        "semantic_verifier_call_count": 0,
        "repair_call_count": 0,
        "prompt_character_count": sum(len(message["content"]) for message in messages),
        "estimated_prompt_tokens": call.estimated_prompt_tokens,
        "generated_character_count": len(answer),
        "generated_token_count": call.generated_token_count,
        "repair_attempted": False,
        "citations": citation_numbers,
        "displayed_source_count": len(citation_numbers),
        "ollama_calls": [call.to_dict()],
    }
    row["response"] = response
    return row


def freeze_experiment(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Freeze the one permitted stronger-model candidate before DEV replay."""

    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / "experiment_freeze.json"
    if path.exists():
        raise RuntimeError("final model experiment is already frozen")
    baseline_manifest = json.loads(
        (PHASE10_OUTPUT / "baseline_manifest.json").read_text(encoding="utf-8")
    )
    rows = read_jsonl(PHASE10_OUTPUT / "dev_generation_results.jsonl")
    prompt_hashes = {
        row["id"]: _sha256_bytes(
            _json_dump(_initial_prompt_call(row)["messages"]).encode("utf-8")
        )
        for row in rows
    }
    shutil.copyfile(
        PHASE10_OUTPUT / "questions_snapshot.jsonl",
        output_directory / "questions_snapshot.jsonl",
    )
    protocol = {
        "mission": "last generation-model capability test before RAG v1 freeze",
        "comparison": {"A": BASELINE_MODEL, "B": CANDIDATE_MODEL},
        "split": "DEV only",
        "only_variable": "generation_model",
        "identical": [
            "question",
            "EvidenceBundles",
            "serialized documentary context",
            "frozen Phase-10 prompt messages",
            "temperature",
            "seed",
            "context size",
            "maximum output tokens",
        ],
        "disabled": [
            "retrieval",
            "EvidencePlanner",
            "Evidence Judge",
            "semantic verifier",
            "repair",
            "region expansion",
            "query planner",
            "new chunking",
            "new indexing",
        ],
        "candidate_count": 1,
        "holdout_policy": "no holdout unless candidate clearly wins DEV",
    }
    (output_directory / "evaluation_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "baseline_sha256": baseline_manifest["baseline_sha256"],
        "phase10_dev_results_sha256": _sha256_file(
            PHASE10_OUTPUT / "dev_generation_results.jsonl"
        ),
        "dataset_sha256": baseline_manifest["dataset"]["questions_sha256"],
        "active_kb": baseline_manifest["active_kb"],
        "baseline_model": BASELINE_MODEL,
        "candidate_model": CANDIDATE_MODEL,
        "generation_config": {
            "temperature": baseline_manifest["qwen"]["temperature"],
            "seed": baseline_manifest["qwen"]["seed"],
            "context_size": baseline_manifest["qwen"]["context_size"],
            "max_output_tokens": baseline_manifest["qwen"]["max_output_tokens"],
            "thinking": baseline_manifest["qwen"]["think"],
        },
        "prompt_message_hashes": prompt_hashes,
        "runner_sha256": _sha256_file(RUNNER_MODULE),
        "record_count": len(rows),
        "holdout_seen": False,
    }
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _verify_freeze(output_directory: Path) -> dict[str, Any]:
    manifest = json.loads(
        (output_directory / "experiment_freeze.json").read_text(encoding="utf-8")
    )
    if _sha256_file(RUNNER_MODULE) != manifest["runner_sha256"]:
        raise RuntimeError("final model runner drift detected")
    if _sha256_file(PHASE10_OUTPUT / "dev_generation_results.jsonl") != manifest[
        "phase10_dev_results_sha256"
    ]:
        raise RuntimeError("Phase-10 DEV drift detected")
    return manifest


def run_dev(
    output_directory: Path = DEFAULT_OUTPUT,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Replay the frozen DEV rows with exactly one stronger model."""

    manifest = _verify_freeze(output_directory)
    source_rows = read_jsonl(PHASE10_OUTPUT / "dev_generation_results.jsonl")
    target = output_directory / "dev_generation_results.jsonl"
    existing = read_jsonl(target) if target.exists() else []
    completed_ids = {row["id"] for row in existing}
    pending = [row for row in source_rows if row["id"] not in completed_ids]
    if limit is not None:
        pending = pending[:limit]
    runtime = _CandidateRuntime()
    written = 0
    try:
        for index, baseline in enumerate(pending, 1):
            row = _run_one(runtime, baseline)
            with target.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            print(
                json.dumps(
                    {
                        "progress": f"{index}/{len(pending)}",
                        "id": row["id"],
                        "status": row["status"],
                        "generation_ms": (
                            row["response"]["timings"]["generation_ms"]
                            if row["response"] is not None
                            else None
                        ),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
    finally:
        runtime.close()
    rows = read_jsonl(target) if target.exists() else []
    write_jsonl(output_directory / "generation_results.jsonl", rows)
    forbidden_total = sum(
        sum(
            int(row["final_model_experiment"][key])
            for key in (
                "retrieval_calls",
                "planner_calls",
                "evidence_judge_calls",
                "semantic_verifier_calls",
                "repair_calls",
            )
        )
        for row in rows
    )
    return {
        "written": written,
        "record_count": len(rows),
        "expected": manifest["record_count"],
        "complete": len({row["id"] for row in rows}) == manifest["record_count"],
        "forbidden_call_total": forbidden_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run-dev", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.freeze:
        result = freeze_experiment(args.output)
    elif args.run_dev:
        result = run_dev(args.output, limit=args.limit)
    else:
        parser.error("choose --freeze or --run-dev")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
