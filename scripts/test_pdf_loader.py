"""Test manuel de l'extraction du premier PDF."""

from pathlib import Path

from phosprocess.ingestion.pdf_loader import extract_pdf_pages

PDF_PATH = Path(
    "data/raw/public/02_jacobs_largest_phosphoric_acid_plant.pdf"
)


def main() -> None:
    """Extraire le PDF Jacobs et afficher un résumé."""

    pages = extract_pdf_pages(PDF_PATH)
    empty_pages = [page.page_number for page in pages if page.is_empty]

    print(f"Document : {PDF_PATH.name}")
    print(f"Nombre de pages : {len(pages)}")
    print(f"Pages sans texte : {empty_pages or 'aucune'}")

    if pages:
        print("\n--- Extrait de la première page ---")
        print(pages[0].text[:700])


if __name__ == "__main__":
    main()