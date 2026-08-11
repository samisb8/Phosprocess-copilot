"""Chargement et extraction brute des documents PDF."""

from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True, slots=True)
class PDFPage:
    """Contenu extrait d'une page PDF."""

    document_name: str
    page_number: int
    text: str

    @property
    def is_empty(self) -> bool:
        """Indiquer si la page ne contient aucun texte exploitable."""
        return not self.text.strip()


def extract_pdf_pages(pdf_path: Path) -> list[PDFPage]:
    """Extraire le texte d'un PDF page par page."""

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Le fichier n'est pas un PDF : {pdf_path}")

    pages: list[PDFPage] = []

    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            text = page.get_text("text", sort=True).strip()

            pages.append(
                PDFPage(
                    document_name=pdf_path.name,
                    page_number=page_index + 1,
                    text=text,
                )
            )

    return pages
