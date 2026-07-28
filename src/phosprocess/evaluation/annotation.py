"""Outils de stockage pour l'annotation du benchmark."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from phosprocess.evaluation.pool_builder import (
    AnnotationPoolItem,
)
from phosprocess.evaluation.schemas import (
    JudgmentStatus,
    RelevanceJudgment,
)

JUDGMENTS_FILENAME = "judgments.jsonl"
PROGRESS_FILENAME = "annotation_progress.json"

DEFAULT_RATIONALES = {
    0: "Le passage ne répond pas à la question.",
    1: ("Le passage fournit du contexte utile, mais ne répond pas directement à la question."),
    2: ("Le passage contient une réponse pertinente, mais partielle ou incomplète."),
    3: ("Le passage répond directement et précisément à la question."),
}


def pool_item_key(
    query_id: str,
    chunk_id: str,
) -> str:
    """Construire l'identifiant unique d'une paire."""

    return f"{query_id}::{chunk_id}"


def load_annotation_pool(
    path: Path,
) -> list[AnnotationPoolItem]:
    """Charger et valider annotation_pool.jsonl."""

    if not path.exists():
        raise FileNotFoundError(f"Pool d'annotation introuvable : {path}")

    items: list[AnnotationPoolItem] = []
    identifiers: set[str] = set()

    with path.open(
        "r",
        encoding="utf-8",
    ) as source:
        for line_number, line in enumerate(
            source,
            start=1,
        ):
            if not line.strip():
                raise ValueError(f"Ligne vide dans {path.name} : {line_number}.")

            try:
                item = AnnotationPoolItem.model_validate_json(line)
            except Exception as error:
                raise ValueError(f"Pool invalide à la ligne {line_number}.") from error

            if item.pool_item_id in identifiers:
                raise ValueError(f"pool_item_id dupliqué : {item.pool_item_id}.")

            identifiers.add(item.pool_item_id)
            items.append(item)

    return items


def load_judgments(
    path: Path,
) -> list[RelevanceJudgment]:
    """Charger les jugements existants."""

    if not path.exists():
        return []

    judgments: list[RelevanceJudgment] = []
    identifiers: set[str] = set()

    with path.open(
        "r",
        encoding="utf-8",
    ) as source:
        for line_number, line in enumerate(
            source,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                judgment = RelevanceJudgment.model_validate_json(line)
            except Exception as error:
                raise ValueError(f"Jugement invalide à la ligne {line_number}.") from error

            identifier = pool_item_key(
                judgment.query_id,
                judgment.chunk_id,
            )

            if identifier in identifiers:
                raise ValueError(f"Jugement dupliqué : {identifier}.")

            identifiers.add(identifier)
            judgments.append(judgment)

    return judgments


def atomic_write_jsonl(
    records: Iterable[RelevanceJudgment],
    path: Path,
) -> None:
    """Écrire les jugements de manière atomique."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as output:
        for record in records:
            output.write(record.model_dump_json() + "\n")

    temporary_path.replace(path)


class JudgmentStore:
    """Stocker et sauvegarder les jugements."""

    def __init__(
        self,
        *,
        path: Path,
        pool_items: list[AnnotationPoolItem],
    ) -> None:
        self.path = path

        self.pool_order = {item.pool_item_id: index for index, item in enumerate(pool_items)}

        self.allowed_identifiers = set(self.pool_order)

        loaded_judgments = load_judgments(path)

        self.judgments = {
            pool_item_key(
                judgment.query_id,
                judgment.chunk_id,
            ): judgment
            for judgment in loaded_judgments
        }

        unknown_identifiers = set(self.judgments) - self.allowed_identifiers

        if unknown_identifiers:
            raise ValueError(
                "judgments.jsonl contient des paires "
                "absentes du pool : "
                f"{sorted(unknown_identifiers)[:10]}"
            )

    def __len__(self) -> int:
        return len(self.judgments)

    def get(
        self,
        item: AnnotationPoolItem,
    ) -> RelevanceJudgment | None:
        """Lire le jugement d'une paire."""

        return self.judgments.get(item.pool_item_id)

    def upsert(
        self,
        judgment: RelevanceJudgment,
    ) -> None:
        """Ajouter ou remplacer puis sauvegarder."""

        self.upsert_many([judgment])

    def upsert_many(
        self,
        judgments: Iterable[RelevanceJudgment],
    ) -> None:
        """Ajouter plusieurs jugements en une écriture."""

        for judgment in judgments:
            identifier = pool_item_key(
                judgment.query_id,
                judgment.chunk_id,
            )

            if identifier not in self.allowed_identifiers:
                raise ValueError(f"Paire absente du pool : {identifier}.")

            self.judgments[identifier] = judgment

        self.flush()

    def flush(self) -> None:
        """Sauvegarder dans l'ordre du pool."""

        ordered_judgments = sorted(
            self.judgments.values(),
            key=lambda judgment: self.pool_order[
                pool_item_key(
                    judgment.query_id,
                    judgment.chunk_id,
                )
            ],
        )

        atomic_write_jsonl(
            ordered_judgments,
            self.path,
        )


def create_judgment(
    *,
    item: AnnotationPoolItem,
    relevance: int,
    assessor_id: str,
    rationale: str | None = None,
    status: JudgmentStatus = JudgmentStatus.VERIFIED,
) -> RelevanceJudgment:
    """Créer un jugement validé."""

    if relevance not in DEFAULT_RATIONALES:
        raise ValueError("La pertinence doit être comprise entre 0 et 3.")

    cleaned_rationale = rationale.strip() if rationale else DEFAULT_RATIONALES[relevance]

    return RelevanceJudgment(
        query_id=item.query_id,
        chunk_id=item.chunk_id,
        relevance=relevance,
        rationale=cleaned_rationale,
        assessor_id=assessor_id,
        status=status,
        judged_at_utc=datetime.now(UTC),
    )


def create_excerpt(
    text: str,
    *,
    maximum_characters: int,
) -> str:
    """Créer un extrait lisible."""

    normalized = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if len(normalized) <= maximum_characters:
        return normalized

    shortened = normalized[:maximum_characters]

    final_space = shortened.rfind(" ")

    if final_space > 0:
        shortened = shortened[:final_space]

    return shortened.rstrip() + "..."


def format_pages(
    pages: list[int],
) -> str:
    """Transformer une liste de pages en intervalles."""

    if not pages:
        return "inconnues"

    sorted_pages = sorted(set(pages))

    ranges: list[str] = []
    start = sorted_pages[0]
    previous = sorted_pages[0]

    for page in sorted_pages[1:]:
        if page == previous + 1:
            previous = page
            continue

        ranges.append(str(start) if start == previous else f"{start}-{previous}")

        start = page
        previous = page

    ranges.append(str(start) if start == previous else f"{start}-{previous}")

    return ", ".join(ranges)
