from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SECTION_RE = re.compile(
    r"##\s+\d+/48\s+[—-]\s+(Q\d{3})\s*"
    r".*?### Question\s*\n+(.*?)\n+"
    r"### Réponse de Sami\s*\n+(.*?)(?=\n---|\Z)",
    flags=re.DOTALL,
)


def clean_markdown(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_answers(markdown: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []

    for match in SECTION_RE.finditer(markdown):
        query_id = match.group(1)
        question = clean_markdown(match.group(2))
        answer = clean_markdown(match.group(3))

        records.append(
            {
                "query_id": query_id,
                "question": question,
                "reference_answer": answer,
            }
        )

    records.sort(key=lambda item: item["query_id"])

    ids = [item["query_id"] for item in records]
    expected = [f"Q{index:03d}" for index in range(1, 49)]

    if ids != expected:
        missing = sorted(set(expected) - set(ids))
        duplicates = sorted({qid for qid in ids if ids.count(qid) > 1})
        raise ValueError(
            "Le document ne contient pas exactement Q001 à Q048. "
            f"Manquantes={missing}, doublons={duplicates}, trouvées={len(ids)}"
        )

    if any("[ÉCRIRE LA RÉPONSE ICI]" in item["reference_answer"] for item in records):
        raise ValueError("Il reste au moins une réponse non remplie.")

    return records


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    directory = root / "data" / "evaluation" / "retrieval" / "v0.1"

    parser = argparse.ArgumentParser(
        description="Convertit le fichier Markdown rempli en JSONL exploitable."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=directory / "questions_batch_filled.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=directory / "reference_answers.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Fichier introuvable : {args.input}")

    markdown = args.input.read_text(encoding="utf-8-sig")
    records = parse_answers(markdown)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("=== Réponses de référence exportées ===")
    print(f"Questions : {len(records)}")
    print(f"Entrée    : {args.input}")
    print(f"Sortie    : {args.output}")


if __name__ == "__main__":
    main()
