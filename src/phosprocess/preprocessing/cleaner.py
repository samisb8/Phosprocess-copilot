"""Nettoyage conservateur du texte extrait des PDF."""

import re

_LAYOUT_PATTERN = re.compile(
    r"\b(?:figure|fig\.|table|diagram)\b",
    flags=re.IGNORECASE,
)


def clean_pdf_text(text: str) -> str:
    """Nettoyer le texte sans modifier les termes techniques."""

    if not text.strip():
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\u00a0", " ")

    # Réunit les mots coupés en fin de ligne :
    # "crystalli-\nzation" devient "crystallization".
    cleaned = re.sub(r"(?<=\w)-\n(?=\w)", "", cleaned)

    cleaned_lines: list[str] = []

    for line in cleaned.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)

    # Limite les grands espaces verticaux à une seule ligne vide.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def classify_content(raw_text: str, clean_text: str) -> str:
    """Identifier le type principal de contenu de la page."""

    if not clean_text:
        return "empty"

    layout_mentions = len(_LAYOUT_PATTERN.findall(raw_text))
    large_spacing_blocks = len(re.findall(r" {4,}", raw_text))

    if layout_mentions >= 2 or large_spacing_blocks >= 3:
        return "figure_table_and_text"

    return "text"


def needs_manual_review(raw_text: str, clean_text: str) -> bool:
    """Indiquer si une page nécessite une inspection humaine."""

    content_type = classify_content(raw_text, clean_text)

    return content_type in {"empty", "figure_table_and_text"}
