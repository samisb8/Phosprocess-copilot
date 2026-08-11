from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def first(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def key_of(record: dict[str, Any]) -> tuple[str, str]:
    return (
        first(record, "query_id", "qid"),
        first(record, "chunk_id", "passage_id", "document_id"),
    )


def relevance_of(record: dict[str, Any]) -> int:
    value = first(record, "relevance", "label", "score", "grade")
    return int(float(value))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    directory = root / "data" / "evaluation" / "retrieval" / "v0.1"

    parser = argparse.ArgumentParser(
        description="Compare les brouillons Ollama aux jugements humains existants."
    )
    parser.add_argument(
        "--drafts",
        type=Path,
        default=directory / "llm_judgment_drafts.jsonl",
    )
    parser.add_argument(
        "--human",
        type=Path,
        default=directory / "judgments.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.drafts.exists():
        raise FileNotFoundError(f"Fichier introuvable : {args.drafts}")
    if not args.human.exists():
        raise FileNotFoundError(f"Fichier introuvable : {args.human}")

    human = {key_of(item): relevance_of(item) for item in iter_jsonl(args.human)}

    pairs: list[tuple[tuple[str, str], int, int, float]] = []
    for draft in iter_jsonl(args.drafts):
        key = key_of(draft)
        if key not in human:
            continue
        pairs.append(
            (
                key,
                human[key],
                relevance_of(draft),
                float(draft.get("confidence", 0)),
            )
        )

    if not pairs:
        print(
            "Aucune paire commune. Lance d'abord le brouillonneur avec "
            "--include-human --query-id Q001,Q004."
        )
        return

    exact = sum(human_label == llm_label for _, human_label, llm_label, _ in pairs)
    within_one = sum(
        abs(human_label - llm_label) <= 1
        for _, human_label, llm_label, _ in pairs
    )

    matrix: dict[int, Counter[int]] = defaultdict(Counter)
    for _, human_label, llm_label, _ in pairs:
        matrix[human_label][llm_label] += 1

    print("=== Accord LLM / humain ===")
    print(f"Paires comparées : {len(pairs)}")
    print(f"Accord exact     : {exact / len(pairs):.2%}")
    print(f"Écart <= 1       : {within_one / len(pairs):.2%}")
    print()
    print("Matrice : lignes=humain, colonnes=LLM")
    print("       LLM 0  LLM 1  LLM 2  LLM 3")
    for human_label in range(4):
        row = matrix[human_label]
        print(
            f"H {human_label} : "
            f"{row[0]:5d}  {row[1]:5d}  {row[2]:5d}  {row[3]:5d}"
        )

    differences = [
        item for item in pairs if item[1] != item[2]
    ]
    differences.sort(key=lambda item: (-abs(item[1] - item[2]), item[0]))

    print()
    print(f"Désaccords : {len(differences)}")
    for (query_id, chunk_id), human_label, llm_label, confidence in differences[:20]:
        print(
            f"- {query_id}::{chunk_id} "
            f"humain={human_label}, LLM={llm_label}, "
            f"confiance={confidence:.2f}"
        )


if __name__ == "__main__":
    main()
