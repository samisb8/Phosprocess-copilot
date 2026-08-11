"""Test du routeur automatique de parseurs."""

from pathlib import Path

from phosprocess.ingestion.parser_router import parse_pdf_automatically

PDF_PATH = Path(
    "data/raw/public/02_jacobs_largest_phosphoric_acid_plant.pdf"
)


def main() -> None:
    """Afficher la décision automatique pour chaque page."""

    pages = parse_pdf_automatically(PDF_PATH)

    for page in pages:
        print(
            f"Page {page.provenance.page_number:>2} | "
            f"parseur={page.provenance.parser:<12} | "
            f"mots={page.quality.word_count:<5} | "
            f"warnings={page.quality.warnings}"
        )


if __name__ == "__main__":
    main()