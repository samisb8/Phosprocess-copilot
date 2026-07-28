"""Test manuel du schéma d'ingestion."""

from phosprocess.ingestion.schemas import (
    PageContent,
    PageProvenance,
    PageQuality,
    ParsedPage,
)


def main() -> None:
    """Créer et valider une page fictive."""

    text = (
        "The Jacobs reactor uses slurry recirculation "
        "to support gypsum crystallization."
    )

    page = ParsedPage(
        content=PageContent(
            plain_text=text,
            markdown=text,
        ),
        provenance=PageProvenance(
            source_file="02_jacobs_largest_phosphoric_acid_plant.pdf",
            document_id="02_jacobs_largest_phosphoric_acid_plant",
            page_number=3,
            parser="pymupdf4llm",
            ocr_used=False,
        ),
        quality=PageQuality(
            character_count=len(text),
            word_count=len(text.split()),
            is_empty=False,
        ),
    )

    print(page.model_dump_json(indent=2))


if __name__ == "__main__":
    main()