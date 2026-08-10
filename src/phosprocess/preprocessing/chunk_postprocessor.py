"""Post-traitement automatique des chunks documentaires."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from phosprocess.ingestion.schemas import ParsedPage
from phosprocess.preprocessing.chunk_schemas import DocumentChunk
from phosprocess.preprocessing.chunker import StructureAwareChunker

_BOILERPLATE_HINT = re.compile(
    r"copyright|all rights reserved|secretariat|technical conference|"
    r"https?://|www\.|tel\.?|fax|e-?mail|^\s*page\s+\d+\s*$|^\s*\d+\s*$",
    flags=re.IGNORECASE,
)

_PROTECTED_TECHNICAL_LINE = re.compile(
    r"P2O5|SO4|CaO|CaSO4|H2SO4|H3PO4|NH3|m3|t/m2|t/m3",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ChunkPostprocessingConfig:
    """Paramètres du post-traitement."""

    remove_boilerplate: bool = True
    deduplicate_exact: bool = True
    merge_small_chunks: bool = True
    restore_uncovered_pages: bool = True

    min_chunk_tokens: int = 80
    max_tokens: int = 700

    boilerplate_min_occurrences: int = 4
    boilerplate_min_fraction: float = 0.20
    boilerplate_max_line_characters: int = 160

    fallback_overlap_tokens: int = 40


@dataclass(slots=True)
class WorkingChunk:
    """Chunk modifiable pendant le post-traitement."""

    document_id: str
    source_file: str
    original_index: int
    heading_path: list[str]
    source_pages: set[int]
    content_types: set[str]
    text: str

    source_chunk_ids: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)

    @classmethod
    def from_document_chunk(
        cls,
        chunk: DocumentChunk,
    ) -> WorkingChunk:
        """Créer un chunk de travail depuis un chunk validé."""

        source_ids = chunk.source_chunk_ids if chunk.source_chunk_ids else [chunk.chunk_id]

        return cls(
            document_id=chunk.document_id,
            source_file=chunk.source_file,
            original_index=chunk.chunk_index,
            heading_path=list(chunk.heading_path),
            source_pages=set(chunk.source_pages),
            content_types=set(chunk.content_types),
            text=chunk.text.strip(),
            source_chunk_ids=list(source_ids),
            actions=list(chunk.postprocessing_actions),
        )


@dataclass(frozen=True, slots=True)
class PostprocessingResult:
    """Résultat final et statistiques du post-traitement."""

    chunks: list[DocumentChunk]
    statistics: dict[str, object]


class ChunkPostprocessor:
    """Nettoyer et consolider automatiquement les chunks."""

    def __init__(
        self,
        config: ChunkPostprocessingConfig,
        token_counter: StructureAwareChunker,
    ) -> None:
        self.config = config
        self.token_counter = token_counter

    def process(
        self,
        chunks: list[DocumentChunk],
        pages: list[ParsedPage],
    ) -> PostprocessingResult:
        """Exécuter toutes les opérations de post-traitement."""

        working_chunks = [WorkingChunk.from_document_chunk(chunk) for chunk in chunks]

        input_chunk_count = len(working_chunks)

        boilerplate_lines = self._detect_boilerplate_lines(working_chunks)

        removed_line_count = 0

        if self.config.remove_boilerplate:
            for chunk in working_chunks:
                removed_line_count += self._remove_boilerplate(
                    chunk,
                    boilerplate_lines,
                )

        duplicates_removed = 0

        if self.config.deduplicate_exact:
            working_chunks, duplicates_removed = self._deduplicate_exact(working_chunks)

        restored_pages: list[int] = []
        fallback_chunks_added = 0

        if self.config.restore_uncovered_pages:
            fallback_chunks, restored_pages = self._create_missing_page_chunks(
                chunks=working_chunks,
                pages=pages,
            )

            fallback_chunks_added = len(fallback_chunks)
            working_chunks.extend(fallback_chunks)

        merged_chunks = 0

        if self.config.merge_small_chunks:
            working_chunks, merged_chunks = self._merge_small_chunks(working_chunks)

        working_chunks.sort(
            key=lambda chunk: (
                min(chunk.source_pages),
                chunk.original_index,
            )
        )

        final_chunks = [
            self._build_final_chunk(
                chunk=chunk,
                chunk_index=index,
            )
            for index, chunk in enumerate(working_chunks)
            if chunk.text.strip()
        ]

        covered_pages = {
            page_number for chunk in final_chunks for page_number in chunk.source_pages
        }

        expected_pages = {
            page.provenance.page_number for page in pages if not page.quality.is_empty
        }

        uncovered_pages = sorted(expected_pages - covered_pages)

        remaining_small_chunks = [
            chunk.chunk_id
            for chunk in final_chunks
            if chunk.token_count < self.config.min_chunk_tokens
        ]

        statistics: dict[str, object] = {
            "input_chunks": input_chunk_count,
            "output_chunks": len(final_chunks),
            "boilerplate_patterns_detected": len(boilerplate_lines),
            "boilerplate_lines_removed": removed_line_count,
            "duplicates_removed": duplicates_removed,
            "small_chunks_merged": merged_chunks,
            "fallback_chunks_added": fallback_chunks_added,
            "restored_pages": restored_pages,
            "uncovered_pages_after": uncovered_pages,
            "remaining_small_chunks": remaining_small_chunks,
        }

        return PostprocessingResult(
            chunks=final_chunks,
            statistics=statistics,
        )

    def _detect_boilerplate_lines(
        self,
        chunks: list[WorkingChunk],
    ) -> set[str]:
        """Détecter les lignes répétées probablement non informatives."""

        line_document_frequency: Counter[str] = Counter()
        original_lines: dict[str, str] = {}

        for chunk in chunks:
            unique_lines: set[str] = set()

            for line in chunk.text.splitlines():
                stripped = line.strip()
                normalized = self._normalize_text(stripped)

                if not normalized:
                    continue

                if len(stripped) > (self.config.boilerplate_max_line_characters):
                    continue

                if self._is_protected_line(stripped):
                    continue

                unique_lines.add(normalized)
                original_lines.setdefault(normalized, stripped)

            line_document_frequency.update(unique_lines)

        threshold = max(
            self.config.boilerplate_min_occurrences,
            math.ceil(len(chunks) * self.config.boilerplate_min_fraction),
        )

        boilerplate: set[str] = set()

        for normalized, count in line_document_frequency.items():
            original = original_lines[normalized]
            word_count = len(original.split())

            repeated_short_line = count >= threshold and word_count <= 10

            hinted_line = count >= 2 and bool(_BOILERPLATE_HINT.search(original))

            if repeated_short_line or hinted_line:
                boilerplate.add(normalized)

        return boilerplate

    def _remove_boilerplate(
        self,
        chunk: WorkingChunk,
        boilerplate_lines: set[str],
    ) -> int:
        """Retirer les lignes répétitives sans supprimer la technique."""

        kept_lines: list[str] = []
        removed_count = 0

        for line in chunk.text.splitlines():
            normalized = self._normalize_text(line)

            if normalized in boilerplate_lines and not self._is_protected_line(line):
                removed_count += 1
                continue

            kept_lines.append(line)

        cleaned_text = "\n".join(kept_lines)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()

        # On ne supprime pas complètement un chunk si toutes ses
        # lignes ont été classées comme boilerplate.
        if cleaned_text:
            chunk.text = cleaned_text

            if removed_count:
                chunk.actions.append("boilerplate_removed")
        elif removed_count:
            chunk.actions.append("boilerplate_only_retained")

        return removed_count

    def _deduplicate_exact(
        self,
        chunks: list[WorkingChunk],
    ) -> tuple[list[WorkingChunk], int]:
        """Fusionner les chunks ayant exactement le même contenu."""

        unique_chunks: list[WorkingChunk] = []
        signatures: dict[str, WorkingChunk] = {}
        removed_count = 0

        for chunk in chunks:
            signature = self._normalize_text(chunk.text)

            if not signature:
                continue

            existing = signatures.get(signature)

            if existing is None:
                signatures[signature] = chunk
                unique_chunks.append(chunk)
                continue

            existing.source_pages.update(chunk.source_pages)
            existing.content_types.update(chunk.content_types)

            existing.source_chunk_ids = self._unique_list(
                [
                    *existing.source_chunk_ids,
                    *chunk.source_chunk_ids,
                ]
            )

            existing.actions = self._unique_list(
                [
                    *existing.actions,
                    *chunk.actions,
                    "exact_duplicate_merged",
                ]
            )

            removed_count += 1

        return unique_chunks, removed_count

    def _create_missing_page_chunks(
        self,
        *,
        chunks: list[WorkingChunk],
        pages: list[ParsedPage],
    ) -> tuple[list[WorkingChunk], list[int]]:
        """Créer des chunks de secours pour les pages oubliées."""

        covered_pages = {page_number for chunk in chunks for page_number in chunk.source_pages}

        fallback_chunks: list[WorkingChunk] = []
        restored_pages: list[int] = []

        for page in sorted(
            pages,
            key=lambda item: item.provenance.page_number,
        ):
            page_number = page.provenance.page_number

            if page.quality.is_empty or page_number in covered_pages:
                continue

            text = page.content.plain_text.strip()

            if not text:
                text = page.content.markdown.strip()

            if not text:
                continue

            pieces = self._split_fallback_text(text)

            for piece_index, piece in enumerate(pieces):
                fallback_chunks.append(
                    WorkingChunk(
                        document_id=page.provenance.document_id,
                        source_file=page.provenance.source_file,
                        original_index=1_000_000 + page_number,
                        heading_path=[],
                        source_pages={page_number},
                        content_types={"fallback_page"},
                        text=piece,
                        source_chunk_ids=[],
                        actions=[
                            "uncovered_page_restored",
                            f"fallback_piece_{piece_index}",
                        ],
                    )
                )

            restored_pages.append(page_number)

        return fallback_chunks, restored_pages

    def _split_fallback_text(self, text: str) -> list[str]:
        """Découper une page de secours trop longue par tokens."""

        context_reserve = 80
        max_body_tokens = max(
            self.config.max_tokens - context_reserve,
            100,
        )

        token_ids = self.token_counter.tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        if len(token_ids) <= max_body_tokens:
            return [text]

        step = max(
            max_body_tokens - self.config.fallback_overlap_tokens,
            1,
        )

        pieces: list[str] = []

        for start in range(0, len(token_ids), step):
            end = start + max_body_tokens
            piece_ids = token_ids[start:end]

            piece = self.token_counter.tokenizer.decode(
                piece_ids,
                skip_special_tokens=True,
            ).strip()

            if piece:
                pieces.append(piece)

            if end >= len(token_ids):
                break

        return pieces

    def _merge_small_chunks(
        self,
        chunks: list[WorkingChunk],
    ) -> tuple[list[WorkingChunk], int]:
        """Fusionner automatiquement les chunks trop courts."""

        chunks = sorted(
            chunks,
            key=lambda chunk: (
                min(chunk.source_pages),
                chunk.original_index,
            ),
        )

        merged_count = 0
        index = 0

        while index < len(chunks):
            current = chunks[index]

            if self._working_token_count(current) >= self.config.min_chunk_tokens:
                index += 1
                continue

            candidates: list[tuple[tuple[int, int], int, WorkingChunk]] = []

            for candidate_index in (index - 1, index + 1):
                if not 0 <= candidate_index < len(chunks):
                    continue

                candidate = chunks[candidate_index]
                merged = self._merge_two_chunks(
                    current,
                    candidate,
                )

                merged_tokens = self._working_token_count(merged)

                if merged_tokens > self.config.max_tokens:
                    continue

                same_heading = int(current.heading_path == candidate.heading_path)

                score = (same_heading, merged_tokens)

                candidates.append((score, candidate_index, merged))

            if not candidates:
                index += 1
                continue

            _, candidate_index, merged_chunk = max(
                candidates,
                key=lambda item: item[0],
            )

            lower_index = min(index, candidate_index)
            upper_index = max(index, candidate_index)

            chunks[lower_index] = merged_chunk
            del chunks[upper_index]

            merged_count += 1
            index = max(lower_index - 1, 0)

        return chunks, merged_count

    def _merge_two_chunks(
        self,
        first: WorkingChunk,
        second: WorkingChunk,
    ) -> WorkingChunk:
        """Fusionner deux chunks en évitant l'overlap dupliqué."""

        ordered = sorted(
            [first, second],
            key=lambda chunk: (
                min(chunk.source_pages),
                chunk.original_index,
            ),
        )

        left, right = ordered

        merged_text = self._merge_texts(
            left.text,
            right.text,
        )

        heading_path = self._common_heading_path(
            left.heading_path,
            right.heading_path,
        )

        if not heading_path:
            heading_path = left.heading_path if left.heading_path else right.heading_path

        return WorkingChunk(
            document_id=left.document_id,
            source_file=left.source_file,
            original_index=min(
                left.original_index,
                right.original_index,
            ),
            heading_path=list(heading_path),
            source_pages={
                *left.source_pages,
                *right.source_pages,
            },
            content_types={
                *left.content_types,
                *right.content_types,
            },
            text=merged_text,
            source_chunk_ids=self._unique_list(
                [
                    *left.source_chunk_ids,
                    *right.source_chunk_ids,
                ]
            ),
            actions=self._unique_list(
                [
                    *left.actions,
                    *right.actions,
                    "small_chunk_merged",
                ]
            ),
        )

    def _build_final_chunk(
        self,
        *,
        chunk: WorkingChunk,
        chunk_index: int,
    ) -> DocumentChunk:
        """Reconstruire un chunk final validé."""

        source_pages = sorted(chunk.source_pages)
        text = chunk.text.strip()

        embedding_text = self._build_embedding_text(chunk)

        digest_input = (f"{chunk.document_id}|{chunk_index}|{source_pages}|{text}").encode()

        digest = hashlib.sha256(digest_input).hexdigest()[:12]

        return DocumentChunk(
            chunk_id=(f"{chunk.document_id}_{chunk_index:06d}_{digest}"),
            document_id=chunk.document_id,
            source_file=chunk.source_file,
            chunk_index=chunk_index,
            heading_path=chunk.heading_path,
            source_pages=source_pages,
            page_start=source_pages[0],
            page_end=source_pages[-1],
            content_types=sorted(chunk.content_types),
            text=text,
            embedding_text=embedding_text,
            body_token_count=self.token_counter.count_tokens(text),
            token_count=self.token_counter.count_tokens(embedding_text),
            source_chunk_ids=self._unique_list(chunk.source_chunk_ids),
            postprocessing_actions=self._unique_list(chunk.actions),
        )

    def _build_embedding_text(
        self,
        chunk: WorkingChunk,
    ) -> str:
        """Construire le texte envoyé au modèle d'embeddings."""

        source_pages = sorted(chunk.source_pages)

        context = [
            f"Document: {chunk.source_file}",
            f"Pages: {self._format_pages(source_pages)}",
        ]

        if chunk.heading_path:
            context.append(f"Section: {' > '.join(chunk.heading_path)}")

        return "\n".join(context) + "\n\n" + chunk.text.strip()

    def _working_token_count(
        self,
        chunk: WorkingChunk,
    ) -> int:
        """Compter les tokens d'un chunk de travail."""

        return self.token_counter.count_tokens(self._build_embedding_text(chunk))

    @staticmethod
    def _merge_texts(left: str, right: str) -> str:
        """Fusionner deux textes sans répéter les mêmes paragraphes."""

        paragraphs: list[str] = []
        signatures: set[str] = set()

        for text in (left, right):
            for paragraph in re.split(r"\n\s*\n", text):
                paragraph = paragraph.strip()

                if not paragraph:
                    continue

                signature = ChunkPostprocessor._normalize_text(paragraph)

                if signature in signatures:
                    continue

                signatures.add(signature)
                paragraphs.append(paragraph)

        return "\n\n".join(paragraphs)

    @staticmethod
    def _common_heading_path(
        left: list[str],
        right: list[str],
    ) -> list[str]:
        """Retourner la partie commune de deux chemins de section."""

        common: list[str] = []

        for left_item, right_item in zip(left, right, strict=False):
            if left_item != right_item:
                break

            common.append(left_item)

        return common

    @staticmethod
    def _format_pages(pages: list[int]) -> str:
        """Compacter une liste de pages en intervalles."""

        if not pages:
            return ""

        ranges: list[str] = []
        start = pages[0]
        previous = pages[0]

        for page in pages[1:]:
            if page == previous + 1:
                previous = page
                continue

            ranges.append(str(start) if start == previous else f"{start}-{previous}")

            start = page
            previous = page

        ranges.append(str(start) if start == previous else f"{start}-{previous}")

        return ", ".join(ranges)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normaliser un texte pour les comparaisons."""

        normalized = unicodedata.normalize("NFKC", text)
        normalized = re.sub(r"\s+", " ", normalized)

        return normalized.strip().casefold()

    @staticmethod
    def _is_protected_line(line: str) -> bool:
        """Protéger les tableaux et termes techniques."""

        stripped = line.strip()

        return stripped.startswith("|") or bool(_PROTECTED_TECHNICAL_LINE.search(stripped))

    @staticmethod
    def _unique_list(items: list[str]) -> list[str]:
        """Supprimer les doublons tout en conservant l'ordre."""

        return list(dict.fromkeys(items))
