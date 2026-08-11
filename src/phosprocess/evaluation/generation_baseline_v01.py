# ruff: noqa: E501
"""Frozen end-to-end generation baseline over the audited Phase-8 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from phosprocess.evaluation.context_engine_v01 import read_jsonl, write_jsonl
from phosprocess.observability.latency import OllamaCallMetrics
from phosprocess.rag.orchestrator import (
    DEFAULT_RUNTIME_CONFIG_PATH,
    PhosProcessRAG,
    load_rag_v1_retrieval_config,
    load_runtime_config,
)
from phosprocess.rag.prompts import (
    STREAMING_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)
from phosprocess.rag.schemas import ChatMessage, RAGResponse

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "data/evaluation/generation_baseline/v0.1"
EVALUATOR_MODULE = PROJECT_ROOT / "src/phosprocess/evaluation/generation_baseline_analysis_v01.py"
PHASE7_OUTPUT = PROJECT_ROOT / "data/evaluation/candidate_preservation/v0.1"
PHASE8_OUTPUT = PROJECT_ROOT / "data/evaluation/evidence_ground_truth_audit/v0.1"
ACTIVE_POINTER = PROJECT_ROOT / "data/knowledge_base/current_index.json"

PRODUCTION_BASELINE_FILES = (
    "configs/rag_v1.yaml",
    "configs/rag_production.yaml",
    "configs/quality_pipeline.yaml",
    "configs/knowledge_base_catalog.yaml",
    "configs/embeddings.yaml",
    "configs/lexical_safeguard_v3.yaml",
    "configs/retrieval_v2.yaml",
    "configs/reranking.yaml",
    "data/knowledge_base/current_index.json",
    "src/phosprocess/embeddings/embedder.py",
    "src/phosprocess/knowledge_base/catalog.py",
    "src/phosprocess/knowledge_base/runtime.py",
    "src/phosprocess/knowledge_base/source_resolution.py",
    "src/phosprocess/llm/ollama_client.py",
    "src/phosprocess/rag/adaptive_router.py",
    "src/phosprocess/rag/answer_validation_service.py",
    "src/phosprocess/rag/citations.py",
    "src/phosprocess/rag/context_window.py",
    "src/phosprocess/rag/conversation_memory.py",
    "src/phosprocess/rag/conversation_state.py",
    "src/phosprocess/rag/followup_resolver.py",
    "src/phosprocess/rag/generation_service.py",
    "src/phosprocess/rag/orchestrator.py",
    "src/phosprocess/rag/prompts.py",
    "src/phosprocess/rag/quality_retrieval.py",
    "src/phosprocess/rag/question_classifier.py",
    "src/phosprocess/rag/retrieval_service.py",
    "src/phosprocess/rag/schemas.py",
    "src/phosprocess/rag/source_policy.py",
    "src/phosprocess/reranking/reranker.py",
    "src/phosprocess/retrieval/bge_sparse.py",
    "src/phosprocess/retrieval/bm25.py",
    "src/phosprocess/retrieval/context_expander.py",
    "src/phosprocess/retrieval/dense.py",
    "src/phosprocess/retrieval/document_selector.py",
    "src/phosprocess/retrieval/domain_router.py",
    "src/phosprocess/retrieval/evidence_bundle.py",
    "src/phosprocess/retrieval/evidence_roles.py",
    "src/phosprocess/retrieval/hierarchical.py",
    "src/phosprocess/retrieval/hybrid.py",
    "src/phosprocess/retrieval/quality_hybrid.py",
    "src/phosprocess/retrieval/query_expansion.py",
    "src/phosprocess/retrieval/retrieval_planner.py",
    "src/phosprocess/retrieval/source_boosting.py",
    "src/phosprocess/retrieval/technical_lexicon.py",
    "src/phosprocess/retrieval/v3_selection.py",
)

SUPPLEMENTAL_QUESTIONS = (
    {
        "id": "ABS_DQ050",
        "question": "Quelle est la couleur exacte du casque de l’opérateur mentionné dans l’atelier ?",
        "language": "fr",
        "split": "dev",
        "answerability": "unanswerable",
        "dataset_scope": "absent_corpus",
    },
    {
        "id": "ABS_CE058",
        "question": "What was the evaporator’s electricity consumption yesterday at 14:05?",
        "language": "en",
        "split": "dev",
        "answerability": "unanswerable",
        "dataset_scope": "absent_corpus",
    },
    {
        "id": "ABS_CE059",
        "question": "من هو المشغل المناوب حالياً في وحدة التركيز؟",
        "language": "ar",
        "split": "dev",
        "answerability": "unanswerable",
        "dataset_scope": "absent_corpus",
    },
    {
        "id": "PROCESS_FLOW_ACCEPTANCE",
        "question": "Décris étape par étape le trajet de l’acide phosphorique dans un évaporateur à circulation forcée.",
        "language": "fr",
        "split": "dev",
        "answerability": "answerable",
        "dataset_scope": "process_flow_acceptance",
    },
)

# Conversation membership is evaluation metadata only.  The production
# pipeline still receives ordinary ChatMessage history and performs its own
# follow-up detection, reformulation, routing, and retrieval.
FOLLOWUP_CONVERSATIONS = (
    {"id": "CONV02", "turn_ids": ["DQ020", "DQ021"], "split": "dev"},
    {"id": "CONV04", "turn_ids": ["DQ043", "DQ044"], "split": "dev"},
    {"id": "CONV06", "turn_ids": ["DQ048", "DQ049"], "split": "dev"},
    {"id": "CONV01", "turn_ids": ["DQ001", "DQ002", "DQ003"], "split": "final_holdout"},
    {"id": "CONV03", "turn_ids": ["DQ041", "DQ042"], "split": "final_holdout"},
    {"id": "CONV05", "turn_ids": ["DQ046", "DQ047"], "split": "final_holdout"},
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repair_display_text(value: str) -> str:
    """Repair legacy UTF-8/CP1252 mojibake in evaluation inputs only."""

    repaired = value
    for _attempt in range(2):
        if not any(marker in repaired for marker in ("Ã", "â", "Â", "Ù", "Ø")):
            break
        converted = None
        for encoding in ("cp1252", "latin1"):
            try:
                converted = repaired.encode(encoding).decode("utf-8")
                break
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        if converted is None:
            break
        repaired = converted
    return repaired


def _command_output(*command: str) -> str:
    return subprocess.check_output(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).strip()


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_question_snapshot() -> list[dict[str, Any]]:
    source_rows = {
        row["id"]: row for row in read_jsonl(PHASE7_OUTPUT / "questions.jsonl")
    }
    evidence = json.loads(
        (PHASE8_OUTPUT / "evidence_sets.json").read_text(encoding="utf-8")
    )
    primary: list[dict[str, Any]] = []
    for question_id, annotation in evidence.items():
        source = source_rows[question_id]
        primary.append(
            {
                "id": question_id,
                "question": _repair_display_text(str(source["question"])),
                "source_question": source["question"],
                "language": source["language"],
                "question_type": source["question_type"],
                "split": source["split"],
                "locked_document": source["locked_document"],
                "answerability": "answerable",
                "dataset_scope": "phase8_primary",
                "valid_evidence_sets": annotation["valid_evidence_sets"],
                "region_chunk_ids": annotation["region_chunk_ids"],
                "documentary_justification": annotation["documentary_justification"],
                "historical_gold_chunk_ids": annotation["historical_gold_chunk_ids"],
                "expected_concepts": source.get("expected_concepts", []),
                "gold_verification": source.get("gold_verification", []),
            }
        )
    primary.sort(key=lambda item: item["id"])
    return [*primary, *(dict(item) for item in SUPPLEMENTAL_QUESTIONS)]


def _protocol() -> dict[str, Any]:
    return {
        "version": "generation_baseline_v0.1",
        "frozen_before_generation": True,
        "gold_visibility": "evaluation_after_generation_only",
        "claim_extraction": "deterministic atomic sentence/list splitter; extraction only",
        "claim_labels": ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"],
        "completeness_labels": ["COMPLETE", "MOSTLY_COMPLETE", "PARTIAL", "MISSED"],
        "evidence_use_labels": ["YES", "PARTIAL", "NO"],
        "failure_taxonomy": {
            "A": "RETRIEVAL_MISS",
            "B": "CONTEXT_PACKING_MISS",
            "C": "GENERATION_OMISSION",
            "D": "UNSUPPORTED_ADDITION",
            "E": "CITATION_ERROR",
            "F": "NUMERIC_ERROR",
            "G": "STRUCTURE_ORDER_ERROR",
            "H": "LANGUAGE_ERROR",
            "I": "CORRECT_REFUSAL",
            "J": "INCORRECT_REFUSAL",
            "K": "SHOULD_HAVE_REFUSED_BUT_ANSWERED",
        },
        "citation_validity": "syntactically valid citations resolving to an existing bundle in the locked document",
        "citation_precision": "manually supported claims among claims carrying a valid citation",
        "citation_coverage": "important factual claims carrying at least one valid citation",
        "numeric_grounding": "answer numbers found in the cited documentary wording",
        "unit_grounding": "answer units preserved in the cited documentary wording",
        "holdout_policy": "protocol and evaluator code frozen on DEV before one holdout opening",
        "production_thresholds_added": False,
        "semantic_judge_used": False,
    }


def freeze_baseline(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Freeze production configuration and audited questions before generation."""

    manifest_path = output_directory / "baseline_manifest.json"
    if manifest_path.exists():
        raise RuntimeError("Phase-10 baseline is already frozen")
    output_directory.mkdir(parents=True, exist_ok=True)
    questions = _load_question_snapshot()
    write_jsonl(output_directory / "questions_snapshot.jsonl", questions)
    (output_directory / "evaluation_protocol.json").write_text(
        json.dumps(_protocol(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    runtime = load_runtime_config()
    frozen = load_rag_v1_retrieval_config()
    active = json.loads(ACTIVE_POINTER.read_text(encoding="utf-8"))
    file_hashes = {
        relative: _sha256_file(PROJECT_ROOT / relative)
        for relative in PRODUCTION_BASELINE_FILES
    }
    manifest = {
        "phase": 10,
        "frozen_at": datetime.now(UTC).isoformat(),
        "git_commit": _command_output("git", "rev-parse", "HEAD"),
        "git_worktree_status": _command_output("git", "status", "--short").splitlines(),
        "git_diff_sha256": _sha256_bytes(
            _command_output("git", "diff", "--binary").encode("utf-8")
        ),
        "active_kb": active,
        "runtime_config_path": str(DEFAULT_RUNTIME_CONFIG_PATH.relative_to(PROJECT_ROOT)),
        "runtime_config_sha256": file_hashes["configs/rag_production.yaml"],
        "qwen": {
            "model": runtime.ollama.model,
            "temperature": runtime.ollama.temperature,
            "seed": runtime.ollama.seed,
            "context_size": runtime.ollama.context_size,
            "max_output_tokens": runtime.ollama.max_output_tokens,
            "keep_alive": runtime.ollama.keep_alive,
            "num_gpu": runtime.ollama.num_gpu,
            "ollama_list": _command_output("ollama", "list").splitlines(),
            "think": False,
        },
        "retriever": {
            "selected_variant": frozen.selected_variant,
            "snapshot_sha256": frozen.snapshot_sha256,
            "candidate_k": frozen.candidate_k,
            "dense_candidates": frozen.dense_candidates,
            "bm25_candidates": frozen.bm25_candidates,
            "query_expansion": frozen.query_expansion,
            "top_k": frozen.top_k,
            "lexical_slots": frozen.lexical_slots,
            "retrieval_config": str(frozen.retrieval_config_path.relative_to(PROJECT_ROOT)),
        },
        "reranker": {
            "config": str(frozen.reranking_config_path.relative_to(PROJECT_ROOT)),
            "config_sha256": _sha256_file(frozen.reranking_config_path),
        },
        "prompt_hashes": {
            "SYSTEM_PROMPT": _sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
            "STREAMING_SYSTEM_PROMPT": _sha256_bytes(
                STREAMING_SYSTEM_PROMPT.encode("utf-8")
            ),
            "prompts_py": file_hashes["src/phosprocess/rag/prompts.py"],
        },
        "production_file_hashes": file_hashes,
        "dataset": {
            "primary_questions": sum(
                item["dataset_scope"] == "phase8_primary" for item in questions
            ),
            "dev_primary": sum(
                item["dataset_scope"] == "phase8_primary" and item["split"] == "dev"
                for item in questions
            ),
            "holdout_primary": sum(
                item["dataset_scope"] == "phase8_primary"
                and item["split"] == "final_holdout"
                for item in questions
            ),
            "supplemental_questions": sum(
                item["dataset_scope"] != "phase8_primary" for item in questions
            ),
            "questions_sha256": _sha256_file(
                output_directory / "questions_snapshot.jsonl"
            ),
            "evidence_sets_sha256": _sha256_file(
                PHASE8_OUTPUT / "evidence_sets.json"
            ),
        },
        "holdout_opened": False,
        "production_changed": False,
    }
    manifest["baseline_sha256"] = _sha256_bytes(
        _json_dump(manifest).encode("utf-8")
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "baseline_sha256": manifest["baseline_sha256"],
        "primary_questions": manifest["dataset"]["primary_questions"],
        "supplemental_questions": manifest["dataset"]["supplemental_questions"],
        "holdout_opened": False,
    }


def _verify_baseline(output_directory: Path) -> dict[str, Any]:
    manifest = json.loads(
        (output_directory / "baseline_manifest.json").read_text(encoding="utf-8")
    )
    drift = {
        relative: {
            "expected": expected,
            "actual": _sha256_file(PROJECT_ROOT / relative),
        }
        for relative, expected in manifest["production_file_hashes"].items()
        if _sha256_file(PROJECT_ROOT / relative) != expected
    }
    if drift:
        raise RuntimeError(f"production baseline drift detected: {sorted(drift)}")
    if _sha256_file(output_directory / "questions_snapshot.jsonl") != manifest[
        "dataset"
    ]["questions_sha256"]:
        raise RuntimeError("question snapshot drift detected")
    return manifest


def _serialize_quality_result(result: Any) -> dict[str, Any]:
    return {
        "resolved_query": {
            "original_query": result.query.original_query,
            "standalone_query": result.query.standalone_query,
            "dense_query": result.query.dense_query,
            "bm25_expanded_query": result.query.bm25_expanded_query,
            "added_terms": list(result.query.added_terms),
        },
        "selected_document": (
            result.routing.preferred_documents[0]
            if result.routing.preferred_documents
            else None
        ),
        "source_lock": sorted(result.routing.hard_filter or ()),
        "routing": {
            "question_type": result.routing.question_type,
            "source_mode": result.routing.source_mode,
            "explicit_source": result.routing.explicit_source,
            "preferred_documents": list(result.routing.preferred_documents),
            "detected_domains": [
                [domain.value, confidence]
                for domain, confidence in result.routing.detected_domains
            ],
        },
        "hybrid_candidates": [
            {
                "rank": item.rank,
                "chunk_id": item.chunk.chunk_id,
                "document_id": item.chunk.document_id,
                "rrf_score": float(item.rrf_score),
                "matched_retrievers": list(item.matched_retrievers),
            }
            for item in result.hybrid.results
        ],
        "reranked_candidates": [
            {
                "rank": item.rank,
                "chunk_id": item.chunk.chunk_id,
                "document_id": item.chunk.document_id,
                "score": float(item.reranker_score),
            }
            for item in result.reranking.results
        ],
        "selected": [
            {
                "rank": rank,
                "chunk_id": item.chunk_id,
                "source": item.source,
            }
            for rank, item in enumerate(result.selected, 1)
        ],
        "evidence_bundles": [bundle.model_dump(mode="json") for bundle in result.bundles],
        "serialized_evidence_context": "\n\n".join(
            bundle.render_prompt_block() for bundle in result.bundles
        ),
        "source_lock_exact": all(
            bundle.document_id
            == (
                result.routing.preferred_documents[0]
                if result.routing.preferred_documents
                else bundle.document_id
            )
            for bundle in result.bundles
        ),
    }


class _TransparentObserver:
    """Capture actual production inputs/outputs while delegating every call."""

    def __init__(self, rag: PhosProcessRAG) -> None:
        self.rag = rag
        self.current: dict[str, Any] = {}
        if rag.quality_engine is None:
            raise RuntimeError("Phase-10 requires the production quality engine")
        self._original_retrieve = rag.quality_engine.retrieve
        self._original_stream_chat = rag.llm.stream_chat

        def observed_retrieve(*args: Any, **kwargs: Any) -> Any:
            result = self._original_retrieve(*args, **kwargs)
            self.current["retrieval"] = _serialize_quality_result(result)
            return result

        def observed_stream_chat(
            messages: Sequence[dict[str, str]],
            *args: Any,
            **kwargs: Any,
        ) -> Iterator[str]:
            call_record = {
                "messages": [dict(message) for message in messages],
                "call_type": kwargs.get("call_type", "generation"),
                "model": self.rag.runtime_config.ollama.model,
                "prompt_sha256": _sha256_bytes(
                    _json_dump(list(messages)).encode("utf-8")
                ),
            }
            self.current.setdefault("prompt_calls", []).append(call_record)
            yield from self._original_stream_chat(messages, *args, **kwargs)
            telemetry = kwargs.get("telemetry")
            if isinstance(telemetry, OllamaCallMetrics):
                call_record["telemetry"] = telemetry.to_dict()

        rag.quality_engine.retrieve = observed_retrieve
        rag.llm.stream_chat = observed_stream_chat

    def reset(self) -> None:
        self.current = {}

    def close(self) -> None:
        assert self.rag.quality_engine is not None
        self.rag.quality_engine.retrieve = self._original_retrieve
        self.rag.llm.stream_chat = self._original_stream_chat


def _existing_completed_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Return the latest successful record for each ID."""

    if not path.exists():
        return {}
    return {
        row["id"]: row
        for row in read_jsonl(path)
        if row.get("status") == "completed"
    }


def _existing_ids(path: Path) -> set[str]:
    """Return only successfully completed record IDs."""

    return set(_existing_completed_rows(path))


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_one(
    rag: PhosProcessRAG,
    observer: _TransparentObserver,
    question: dict[str, Any],
    *,
    history: Sequence[ChatMessage] | None = None,
    record_type: str = "primary",
) -> dict[str, Any]:
    observer.reset()
    started_at = datetime.now(UTC).isoformat()
    event_trace: list[dict[str, Any]] = []
    response: RAGResponse | None = None
    error: dict[str, Any] | None = None
    for event in rag.stream_answer(
        question["question"],
        history=list(history) if history is not None else None,
        source_mode="automatic",
        language_mode="auto",
    ):
        if event.event_type in {
            "retrieval_started",
            "retrieval_completed",
            "validation_started",
            "error",
        }:
            event_trace.append(
                {
                    "event_type": event.event_type,
                    "content": event.content,
                    "metadata": event.metadata,
                }
            )
        if event.event_type == "completed":
            response = event.response
        elif event.event_type == "error":
            error = {"message": event.content, "metadata": event.metadata}
    payload: dict[str, Any] = {
        "id": question["id"],
        "record_type": record_type,
        "dataset_scope": question["dataset_scope"],
        "split": question["split"],
        "language": question["language"],
        "answerability": question["answerability"],
        "question": question["question"],
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "history": [message.model_dump(mode="json") for message in history or ()],
        "event_trace": event_trace,
        "retrieval": observer.current.get("retrieval"),
        "prompt_calls": observer.current.get("prompt_calls", []),
        "hidden_chain_of_thought_stored": False,
        "status": "completed" if response is not None else "error",
        "error": error,
        "response": response.model_dump(mode="json") if response is not None else None,
    }
    return payload


def _questions_for_split(
    output_directory: Path,
    split: str,
) -> list[dict[str, Any]]:
    rows = read_jsonl(output_directory / "questions_snapshot.jsonl")
    return [row for row in rows if row["split"] == split]


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


def _primary_result_by_id(output_directory: Path) -> dict[str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ("dev_generation_results.jsonl", "holdout_generation_results.jsonl"):
        path = output_directory / name
        if path.exists():
            rows.extend(read_jsonl(path))
    return {row["id"]: row for row in rows if row.get("status") == "completed"}


def run_followups(
    split: str,
    output_directory: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Run frozen real conversations without treating history as evidence."""

    manifest = _verify_baseline(output_directory)
    if split not in {"dev", "final_holdout"}:
        raise ValueError(split)
    if split == "final_holdout":
        marker_path = output_directory / "holdout_open_manifest.json"
        if not marker_path.exists():
            raise RuntimeError("FINAL HOLDOUT must be opened by the primary run first")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("status") != "complete":
            raise RuntimeError("primary FINAL HOLDOUT generation must complete first")

    questions = {
        row["id"]: row
        for row in read_jsonl(output_directory / "questions_snapshot.jsonl")
    }
    primary = _primary_result_by_id(output_directory)
    conversations = [item for item in FOLLOWUP_CONVERSATIONS if item["split"] == split]
    missing_first_turns = [
        item["turn_ids"][0]
        for item in conversations
        if item["turn_ids"][0] not in primary
    ]
    if missing_first_turns:
        raise RuntimeError(f"missing primary first turns: {missing_first_turns}")

    result_path = output_directory / "followup_results.jsonl"
    completed_by_id = _existing_completed_rows(result_path)
    completed = set(completed_by_id)
    expected_ids = [
        f"{conversation['id']}_T{turn_number}_{source_id}"
        for conversation in conversations
        for turn_number, source_id in enumerate(conversation["turn_ids"][1:], 2)
    ]
    rag = PhosProcessRAG()
    warmup = rag.warmup().to_dict()
    observer = _TransparentObserver(rag)
    written = 0
    try:
        for conversation in conversations:
            turn_ids = conversation["turn_ids"]
            first = primary[turn_ids[0]]
            history = [
                ChatMessage(role="user", content=questions[turn_ids[0]]["question"]),
                ChatMessage(role="assistant", content=first["response"]["answer"]),
            ]
            for turn_number, source_id in enumerate(turn_ids[1:], 2):
                result_id = f"{conversation['id']}_T{turn_number}_{source_id}"
                question = dict(questions[source_id])
                question["id"] = result_id
                question["split"] = split
                if result_id in completed:
                    existing = completed_by_id[result_id]
                    history.extend(
                        [
                            ChatMessage(role="user", content=question["question"]),
                            ChatMessage(
                                role="assistant", content=existing["response"]["answer"]
                            ),
                        ]
                    )
                    continue
                record = _run_one(
                    rag,
                    observer,
                    question,
                    history=history,
                    record_type="followup",
                )
                record["conversation_id"] = conversation["id"]
                record["turn_number"] = turn_number
                record["source_question_id"] = source_id
                record["history_is_evidence"] = False
                _append_jsonl(result_path, record)
                written += 1
                print(
                    json.dumps(
                        {
                            "conversation": conversation["id"],
                            "turn": turn_number,
                            "id": source_id,
                            "status": record["status"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if record["status"] != "completed":
                    break
                completed.add(result_id)
                completed_by_id[result_id] = record
                history.extend(
                    [
                        ChatMessage(role="user", content=question["question"]),
                        ChatMessage(role="assistant", content=record["response"]["answer"]),
                    ]
                )
    finally:
        observer.close()
        rag.llm.close()

    rows = read_jsonl(result_path) if result_path.exists() else []
    split_completed = {
        row["id"]: row
        for row in rows
        if row["split"] == split and row.get("status") == "completed"
    }
    complete = set(split_completed) == set(expected_ids)
    (output_directory / f"{split}_followup_run_metadata.json").write_text(
        json.dumps(
            {
                "run_at": datetime.now(UTC).isoformat(),
                "baseline_sha256": manifest["baseline_sha256"],
                "warmup": warmup,
                "written_this_run": written,
                "completed_records": len(split_completed),
                "expected_records": len(expected_ids),
                "complete": complete,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _refresh_combined(output_directory)
    return {
        "split": split,
        "written": written,
        "completed_records": len(split_completed),
        "expected_records": len(expected_ids),
        "complete": complete,
    }



def run_split(
    split: str,
    output_directory: Path = DEFAULT_OUTPUT,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run actual production streaming generation for DEV or FINAL HOLDOUT."""

    manifest = _verify_baseline(output_directory)
    if split not in {"dev", "final_holdout"}:
        raise ValueError(split)
    result_path = output_directory / (
        "dev_generation_results.jsonl"
        if split == "dev"
        else "holdout_generation_results.jsonl"
    )
    if split == "final_holdout":
        marker_path = output_directory / "holdout_open_manifest.json"
        if marker_path.exists():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("status") == "complete":
                raise RuntimeError("Phase-10 FINAL HOLDOUT generation was already completed")
        else:
            evaluator_freeze_path = output_directory / "evaluator_freeze_manifest.json"
            if not evaluator_freeze_path.exists():
                raise RuntimeError("freeze the DEV evaluator before opening FINAL HOLDOUT")
            evaluator_freeze = json.loads(
                evaluator_freeze_path.read_text(encoding="utf-8")
            )
            if _sha256_file(EVALUATOR_MODULE) != evaluator_freeze["evaluator_sha256"]:
                raise RuntimeError("evaluator code changed after its DEV freeze")
            marker = {
                "opened_at": datetime.now(UTC).isoformat(),
                "opening_count": 1,
                "baseline_sha256": manifest["baseline_sha256"],
                "protocol_sha256": _sha256_file(
                    output_directory / "evaluation_protocol.json"
                ),
                "evaluator_sha256": evaluator_freeze["evaluator_sha256"],
                "evaluator_freeze_sha256": _sha256_file(evaluator_freeze_path),
                "policy_frozen_before_open": True,
                "status": "in_progress",
            }
            marker_path.write_text(
                json.dumps(marker, indent=2) + "\n", encoding="utf-8"
            )
    completed = _existing_ids(result_path)
    pending = [
        item
        for item in _questions_for_split(output_directory, split)
        if item["id"] not in completed
    ]
    if limit is not None:
        pending = pending[:limit]
    rag = PhosProcessRAG()
    warmup = rag.warmup().to_dict()
    observer = _TransparentObserver(rag)
    written = 0
    try:
        for index, question in enumerate(pending, 1):
            record = _run_one(rag, observer, question)
            _append_jsonl(result_path, record)
            written += 1
            print(
                json.dumps(
                    {
                        "progress": f"{index}/{len(pending)}",
                        "id": question["id"],
                        "status": record["status"],
                        "answer_chars": len(
                            (record.get("response") or {}).get("answer", "")
                        ),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        observer.close()
        rag.llm.close()
    all_rows = read_jsonl(result_path) if result_path.exists() else []
    expected_ids = {
        item["id"] for item in _questions_for_split(output_directory, split)
    }
    completed_ids = {
        row["id"] for row in all_rows if row.get("status") == "completed"
    }
    expected = len(expected_ids)
    complete = completed_ids == expected_ids
    if split == "final_holdout" and complete:
        marker_path = output_directory / "holdout_open_manifest.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["status"] = "complete"
        marker["completed_at"] = datetime.now(UTC).isoformat()
        marker["question_count"] = expected
        marker_path.write_text(
            json.dumps(marker, indent=2) + "\n", encoding="utf-8"
        )
    (output_directory / f"{split}_run_metadata.json").write_text(
        json.dumps(
            {
                "run_at": datetime.now(UTC).isoformat(),
                "baseline_sha256": manifest["baseline_sha256"],
                "warmup": warmup,
                "written_this_run": written,
                "completed_records": len(completed_ids),
                "expected_records": expected,
                "complete": complete,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _refresh_combined(output_directory)
    return {
        "split": split,
        "written": written,
        "completed_records": len(completed_ids),
        "expected_records": expected,
        "complete": complete,
        "holdout_opened": split == "final_holdout",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run-dev", action="store_true")
    parser.add_argument("--open-holdout", action="store_true")
    parser.add_argument("--run-dev-followups", action="store_true")
    parser.add_argument("--run-holdout-followups", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.freeze:
        result = freeze_baseline(args.output)
    elif args.run_dev:
        result = run_split("dev", args.output, limit=args.limit)
    elif args.open_holdout:
        result = run_split("final_holdout", args.output, limit=args.limit)
    elif args.run_dev_followups:
        result = run_followups("dev", args.output)
    elif args.run_holdout_followups:
        result = run_followups("final_holdout", args.output)
    else:
        parser.error("choose a freeze, primary-run, or follow-up-run action")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
