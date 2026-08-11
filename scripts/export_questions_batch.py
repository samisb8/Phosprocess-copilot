import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "evaluation"
    / "retrieval"
    / "v0.1"
    / "queries.jsonl"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "evaluation"
    / "retrieval"
    / "v0.1"
    / "questions_batch.md"
)


def first_value(data: dict, *keys: str, default: str = "") -> str:
    for key in keys:
        value = data.get(key)

        if value is not None and str(value).strip():
            return str(value).strip()

    return default


def natural_key(value: str):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Fichier introuvable : {INPUT_FILE}")


questions = []

with INPUT_FILE.open("r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        line = line.strip()

        if not line:
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"JSON invalide à la ligne {line_number}: {error}"
            ) from error

        query_id = first_value(
            item,
            "query_id",
            "qid",
            "id",
            default=f"Q{line_number:03d}",
        )

        question = first_value(
            item,
            "question",
            "query",
            "text",
        )

        if not question:
            raise ValueError(
                f"Aucune question trouvée à la ligne {line_number}"
            )

        questions.append(
            {
                "query_id": query_id,
                "question": question,
                "split": first_value(item, "split", default="unknown"),
                "category": first_value(
                    item,
                    "category",
                    "question_type",
                    default="unknown",
                ),
                "language": first_value(
                    item,
                    "language",
                    "lang",
                    default="unknown",
                ),
            }
        )


split_order = {
    "dev": 0,
    "test": 1,
    "unknown": 2,
}

questions.sort(
    key=lambda item: (
        split_order.get(item["split"].lower(), 99),
        natural_key(item["query_id"]),
    )
)


lines = [
    "# PhosProcess Copilot — Réponses aux 48 questions",
    "",
    "Remplace chaque marqueur `[ÉCRIRE LA RÉPONSE ICI]` par ta réponse.",
    "",
    "Ne consulte pas les réponses de référence pendant cet exercice.",
    "",
]

current_split = None

for position, item in enumerate(questions, start=1):
    split = item["split"].upper()

    if split != current_split:
        current_split = split
        lines.extend(
            [
                "---",
                "",
                f"# Split {split}",
                "",
            ]
        )

    lines.extend(
        [
            f"## {position}/48 — {item['query_id']}",
            "",
            f"**Catégorie :** {item['category']}",
            "",
            f"**Langue :** {item['language']}",
            "",
            f"### Question",
            "",
            item["question"],
            "",
            "### Réponse de Sami",
            "",
            "[ÉCRIRE LA RÉPONSE ICI]",
            "",
            "---",
            "",
        ]
    )


OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")

print("=== Export terminé ===")
print(f"Questions exportées : {len(questions)}")
print(f"Fichier             : {OUTPUT_FILE}")
