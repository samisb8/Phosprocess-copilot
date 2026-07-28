"""Chunking structuré et contrôlé par le tokenizer d'embeddings."""

import hashlib
import re
from dataclasses import dataclass
from typing import Literal

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from phosprocess.ingestion.schemas import ParsedPage
from phosprocess.preprocessing.chunk_schemas import DocumentChunk

UnitKind = Literal["paragraph", "table", "caption"]

_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_MARKDOWN_IMAGE = re.compile(r"!\[([^\]]*)]\([^)]*\)")
_IMAGE_COMMENT = re.compile(r"<!--\s*image\s*-->", flags=re.IGNORECASE)
_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ0-9])"
)


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Paramètres du chunker."""

    tokenizer_name: str = "BAAI/bge-m3"
    target_tokens: int = 500
    max_tokens: int = 700
    overlap_tokens: int = 80
    min_chunk_tokens: int = 80
    include_document_context: bool = True


@dataclass(frozen=True, slots=True)
class TextUnit:
    """Unité indivisible issue d'un paragraphe ou tableau."""

    text: str
    page_number: int
    heading_path: tuple[str, ...]
    kind: UnitKind


class StructureAwareChunker:
    """Découper les documents selon leur structure et leur tokenizer."""

    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

        self.tokenizer: PreTrainedTokenizerBase = (
            AutoTokenizer.from_pretrained(
                config.tokenizer_name,
                use_fast=True,
            )
        )

        # Réserve de la place pour le titre, la section et les pages.
        self.max_body_tokens = max(config.max_tokens - 80, 100)

    def count_tokens(self, text: str) -> int:
        """Compter les tokens comme le fera le modèle d'embeddings."""

        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    def chunk_document(
        self,
        pages: list[ParsedPage],
    ) -> list[DocumentChunk]:
        """Créer tous les chunks d'un document."""

        if not pages:
            return []

        sorted_pages = sorted(
            pages,
            key=lambda page: page.provenance.page_number,
        )

        units = self._extract_units(sorted_pages)
        groups = self._group_by_heading(units)

        chunk_units: list[list[TextUnit]] = []

        for group in groups:
            chunk_units.extend(self._chunk_group(group))

        chunk_units = self._merge_final_small_chunks(chunk_units)

        document_id = sorted_pages[0].provenance.document_id
        source_file = sorted_pages[0].provenance.source_file

        return [
            self._build_chunk(
                units=units_in_chunk,
                document_id=document_id,
                source_file=source_file,
                chunk_index=index,
            )
            for index, units_in_chunk in enumerate(chunk_units)
            if units_in_chunk
        ]

    def _extract_units(
        self,
        pages: list[ParsedPage],
    ) -> list[TextUnit]:
        """Transformer le Markdown des pages en unités structurées."""

        units: list[TextUnit] = []
        heading_stack: list[str] = []

        for page in pages:
            page_number = page.provenance.page_number
            markdown = page.content.markdown.strip()

            markdown = _IMAGE_COMMENT.sub("", markdown)
            markdown = _MARKDOWN_IMAGE.sub(
                lambda match: match.group(1),
                markdown,
            )

            if not markdown:
                markdown = page.content.plain_text.strip()

            page_units = self._parse_markdown_page(
                markdown=markdown,
                page_number=page_number,
                heading_stack=heading_stack,
            )

            # Les tableaux extraits séparément sont conservés si le
            # Markdown principal ne les contient pas déjà.
            existing_text = "\n".join(
                unit.text for unit in page_units
            )

            for table in page.content.tables:
                table = table.strip()

                if table and table not in existing_text:
                    page_units.append(
                        TextUnit(
                            text=table,
                            page_number=page_number,
                            heading_path=tuple(heading_stack),
                            kind="table",
                        )
                    )

            units.extend(page_units)

        return units

    def _parse_markdown_page(
        self,
        *,
        markdown: str,
        page_number: int,
        heading_stack: list[str],
    ) -> list[TextUnit]:
        """Analyser les titres, paragraphes, listes et tableaux."""

        units: list[TextUnit] = []
        paragraph_lines: list[str] = []
        table_lines: list[str] = []

        def flush_paragraph() -> None:
            if not paragraph_lines:
                return

            text = self._join_paragraph_lines(paragraph_lines)
            paragraph_lines.clear()

            if not text:
                return

            kind: UnitKind = (
                "caption"
                if self._looks_like_caption(text)
                else "paragraph"
            )

            units.append(
                TextUnit(
                    text=text,
                    page_number=page_number,
                    heading_path=tuple(heading_stack),
                    kind=kind,
                )
            )

        def flush_table() -> None:
            if not table_lines:
                return

            table_text = "\n".join(table_lines).strip()
            table_lines.clear()

            if table_text:
                units.append(
                    TextUnit(
                        text=table_text,
                        page_number=page_number,
                        heading_path=tuple(heading_stack),
                        kind="table",
                    )
                )

        for raw_line in markdown.splitlines():
            line = raw_line.strip()

            if not line:
                flush_paragraph()
                flush_table()
                continue

            heading_match = _MARKDOWN_HEADING.match(line)

            if heading_match:
                flush_paragraph()
                flush_table()

                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                self._update_heading_stack(
                    heading_stack,
                    level,
                    title,
                )
                continue

            if self._looks_like_plain_heading(line):
                flush_paragraph()
                flush_table()

                level = 1 if line.upper().startswith("CHAPTER ") else 2

                self._update_heading_stack(
                    heading_stack,
                    level,
                    line,
                )
                continue

            if _TABLE_LINE.match(line):
                flush_paragraph()
                table_lines.append(line)
                continue

            flush_table()
            paragraph_lines.append(line)

        flush_paragraph()
        flush_table()

        return units

    @staticmethod
    def _update_heading_stack(
        heading_stack: list[str],
        level: int,
        title: str,
    ) -> None:
        """Mettre à jour le chemin chapitre → section."""

        del heading_stack[level - 1 :]

        while len(heading_stack) < level - 1:
            heading_stack.append("")

        heading_stack.append(title)

    @staticmethod
    def _looks_like_plain_heading(line: str) -> bool:
        """Détecter prudemment un titre sans marqueur Markdown."""

        if len(line) > 100:
            return False

        words = line.split()

        if not 1 <= len(words) <= 12:
            return False

        letters = [character for character in line if character.isalpha()]

        if len(letters) < 4:
            return False

        return line == line.upper() and not line.endswith(".")

    @staticmethod
    def _looks_like_caption(text: str) -> bool:
        """Identifier une légende de figure ou tableau."""

        lowered = text.casefold()

        return (
            lowered.startswith("fig.")
            or lowered.startswith("figure ")
            or lowered.startswith("table ")
            or lowered.startswith("diagram ")
        )

    @staticmethod
    def _join_paragraph_lines(lines: list[str]) -> str:
        """Réunir les lignes sans casser les listes Markdown."""

        if any(
            re.match(r"^[-*+]\s+", line)
            or re.match(r"^\d+[.)]\s+", line)
            for line in lines
        ):
            return "\n".join(lines).strip()

        return " ".join(lines).strip()

    @staticmethod
    def _group_by_heading(
        units: list[TextUnit],
    ) -> list[list[TextUnit]]:
        """Regrouper les unités consécutives d'une même section."""

        groups: list[list[TextUnit]] = []

        for unit in units:
            if (
                not groups
                or groups[-1][0].heading_path != unit.heading_path
            ):
                groups.append([unit])
            else:
                groups[-1].append(unit)

        return groups

    def _chunk_group(
        self,
        units: list[TextUnit],
    ) -> list[list[TextUnit]]:
        """Découper un groupe sans dépasser la limite de tokens."""

        expanded_units: list[TextUnit] = []

        for unit in units:
            expanded_units.extend(self._split_oversized_unit(unit))

        chunks: list[list[TextUnit]] = []
        current: list[TextUnit] = []

        for unit in expanded_units:
            candidate = [*current, unit]
            candidate_tokens = self._count_unit_tokens(candidate)
            current_tokens = self._count_unit_tokens(current)

            should_close = current and (
                candidate_tokens > self.max_body_tokens
                or current_tokens >= self.config.target_tokens
            )

            if should_close:
                chunks.append(current)
                current = self._overlap_tail(current)

            candidate = [*current, unit]

            if (
                current
                and self._count_unit_tokens(candidate)
                > self.max_body_tokens
            ):
                chunks.append(current)
                current = []

            current.append(unit)

        if current:
            chunks.append(current)

        return chunks

    def _split_oversized_unit(
        self,
        unit: TextUnit,
    ) -> list[TextUnit]:
        """Découper un paragraphe trop long sans couper brutalement."""

        if self.count_tokens(unit.text) <= self.max_body_tokens:
            return [unit]

        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_BOUNDARY.split(unit.text)
            if sentence.strip()
        ]

        if len(sentences) <= 1:
            return self._split_by_words(unit)

        pieces: list[TextUnit] = []
        current_sentences: list[str] = []

        for sentence in sentences:
            candidate = " ".join([*current_sentences, sentence])

            if (
                current_sentences
                and self.count_tokens(candidate)
                > self.max_body_tokens
            ):
                pieces.append(
                    self._copy_unit(
                        unit,
                        " ".join(current_sentences),
                    )
                )
                current_sentences = []

            if self.count_tokens(sentence) > self.max_body_tokens:
                pieces.extend(
                    self._split_by_words(
                        self._copy_unit(unit, sentence)
                    )
                )
            else:
                current_sentences.append(sentence)

        if current_sentences:
            pieces.append(
                self._copy_unit(
                    unit,
                    " ".join(current_sentences),
                )
            )

        return pieces

    def _split_by_words(
        self,
        unit: TextUnit,
    ) -> list[TextUnit]:
        """Fallback pour une phrase dépassant seule la limite."""

        pieces: list[TextUnit] = []
        current_words: list[str] = []

        for word in unit.text.split():
            candidate = " ".join([*current_words, word])

            if (
                current_words
                and self.count_tokens(candidate)
                > self.max_body_tokens
            ):
                pieces.append(
                    self._copy_unit(
                        unit,
                        " ".join(current_words),
                    )
                )
                current_words = []

            current_words.append(word)

        if current_words:
            pieces.append(
                self._copy_unit(
                    unit,
                    " ".join(current_words),
                )
            )

        return pieces

    @staticmethod
    def _copy_unit(
        unit: TextUnit,
        text: str,
    ) -> TextUnit:
        """Reproduire une unité avec un autre texte."""

        return TextUnit(
            text=text.strip(),
            page_number=unit.page_number,
            heading_path=unit.heading_path,
            kind=unit.kind,
        )

    def _overlap_tail(
        self,
        units: list[TextUnit],
    ) -> list[TextUnit]:
        """Reprendre seulement la fin utile du chunk précédent."""

        overlap: list[TextUnit] = []

        for unit in reversed(units):
            candidate = [unit, *overlap]

            if (
                self._count_unit_tokens(candidate)
                > self.config.overlap_tokens
            ):
                break

            overlap.insert(0, unit)

        return overlap

    def _merge_final_small_chunks(
        self,
        chunks: list[list[TextUnit]],
    ) -> list[list[TextUnit]]:
        """Fusionner les derniers petits chunks lorsque c'est possible."""

        if len(chunks) < 2:
            return chunks

        merged: list[list[TextUnit]] = []

        for chunk in chunks:
            chunk_tokens = self._count_unit_tokens(chunk)

            if (
                merged
                and chunk_tokens < self.config.min_chunk_tokens
            ):
                previous = merged[-1]

                unique_units = [
                    unit
                    for unit in chunk
                    if unit not in previous
                ]

                candidate = [*previous, *unique_units]

                if (
                    self._count_unit_tokens(candidate)
                    <= self.max_body_tokens
                ):
                    merged[-1] = candidate
                    continue

            merged.append(chunk)

        return merged

    def _count_unit_tokens(
        self,
        units: list[TextUnit],
    ) -> int:
        """Compter les tokens du contenu d'une liste d'unités."""

        if not units:
            return 0

        return self.count_tokens(
            "\n\n".join(unit.text for unit in units)
        )

    def _build_chunk(
        self,
        *,
        units: list[TextUnit],
        document_id: str,
        source_file: str,
        chunk_index: int,
    ) -> DocumentChunk:
        """Construire et valider le chunk final."""

        text = "\n\n".join(unit.text for unit in units).strip()

        source_pages = sorted(
            {unit.page_number for unit in units}
        )

        heading_path = list(
            max(
                (unit.heading_path for unit in units),
                key=len,
                default=(),
            )
        )

        content_types = sorted(
            {unit.kind for unit in units}
        )

        context_lines = [
            f"Document: {source_file}",
            f"Pages: {source_pages[0]}-{source_pages[-1]}",
        ]

        if heading_path:
            context_lines.append(
                f"Section: {' > '.join(heading_path)}"
            )

        embedding_text = (
            "\n".join(context_lines) + "\n\n" + text
            if self.config.include_document_context
            else text
        )

        digest_source = (
            f"{document_id}|{chunk_index}|{text}"
        ).encode()

        digest = hashlib.sha256(digest_source).hexdigest()[:12]

        return DocumentChunk(
            chunk_id=f"{document_id}_{chunk_index:06d}_{digest}",
            document_id=document_id,
            source_file=source_file,
            chunk_index=chunk_index,
            heading_path=heading_path,
            source_pages=source_pages,
            page_start=source_pages[0],
            page_end=source_pages[-1],
            content_types=content_types,
            text=text,
            embedding_text=embedding_text,
            body_token_count=self.count_tokens(text),
            token_count=self.count_tokens(embedding_text),
        )
