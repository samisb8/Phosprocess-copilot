"""Nettoyer les pages extraites du document Jacobs."""

import json
from pathlib import Path

from phosprocess.preprocessing.cleaner import (
    classify_content,
    clean_pdf_text,
    needs_manual_review,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "02_jacobs_largest_phosphoric_acid_plant_pages.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "02_jacobs_largest_phosphoric_acid_plant_clean_pages.jsonl"
)


def main() -> None:
    """Lire, nettoyer et sauvegarder toutes les pages."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Fichier d'entrée introuvable : {INPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    empty_pages = 0
    review_pages: list[int] = []

    with (
        INPUT_PATH.open("r", encoding="utf-8") as source_file,
        OUTPUT_PATH.open("w", encoding="utf-8") as output_file,
    ):
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON invalide à la ligne {line_number}"
                ) from error

            raw_text = str(record.get("text", ""))
            clean_text = clean_pdf_text(raw_text)

            content_type = classify_content(raw_text, clean_text)
            requires_review = needs_manual_review(raw_text, clean_text)

            page_number = int(record["page_number"])

            processed_record = {
                key: value
                for key, value in record.items()
                if key != "text"
            }

            processed_record.update(
                {
                    "raw_text": raw_text,
                    "clean_text": clean_text,
                    "content_type": content_type,
                    "needs_manual_review": requires_review,
                }
            )

            output_file.write(
                json.dumps(processed_record, ensure_ascii=False) + "\n"
            )

            total_pages += 1

            if not clean_text:
                empty_pages += 1

            if requires_review:
                review_pages.append(page_number)

    print(f"Fichier nettoyé : {OUTPUT_PATH}")
    print(f"Pages traitées : {total_pages}")
    print(f"Pages vides : {empty_pages}")
    print(f"Pages à vérifier : {review_pages or 'aucune'}")


if __name__ == "__main__":
    main()