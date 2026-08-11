"""Adjudication automatique et prudente des gold evidence du TEST v2.

Ce script ne lance aucune évaluation. Il conserve les gold humains tels quels,
adjudique uniquement les questions TEST restantes, puis produit un fichier
fusionné séparé.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval"
    / "v0.1"
    / "test_pool_v2"
)
CHUNKS_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "final_chunks"
RERANKING_CONFIG_PATH = PROJECT_ROOT / "configs" / "reranking.yaml"

DEFAULT_QUERIES_PATH = TEST_DIRECTORY / "queries.jsonl"
DEFAULT_POOL_PATH = TEST_DIRECTORY / "annotation_pool.jsonl"
DEFAULT_DRAFTS_PATH = TEST_DIRECTORY / "gold_evidence_drafts.jsonl"
DEFAULT_HUMAN_GOLD_PATH = (
    TEST_DIRECTORY / "gold_evidence_test_verified.jsonl"
)
DEFAULT_AUTO_OUTPUT_PATH = (
    TEST_DIRECTORY / "gold_evidence_test_auto.jsonl"
)
DEFAULT_MANUAL_REVIEW_PATH = (
    TEST_DIRECTORY / "manual_review_queue.jsonl"
)
DEFAULT_MERGED_OUTPUT_PATH = (
    TEST_DIRECTORY / "gold_evidence_test_verified_auto.jsonl"
)

EXPECTED_TEST_IDS = {
    f"Q{number:03d}"
    for number in range(21, 49)
}
EXPECTED_UNANSWERABLE_IDS = {
    "Q045",
    "Q046",
    "Q047",
    "Q048",
}

# Preuves identifiées manuellement dans le corpus complet mais absentes du
# pool de leur question. Elles restent soumises aux contrôles de couverture.
EVIDENCE_OVERRIDES: dict[str, tuple[str, ...]] = {
    "Q029": (
        "02_jacobs_largest_phosphoric_acid_plant_000009_df11a7538dab",
    ),
    "Q034": (
        "02_jacobs_largest_phosphoric_acid_plant_000004_f6d2f1f977b0",
    ),
}

# Passages découverts lors du contrôle contradictoire des premières décisions.
# Ils sont déjà dans le pool, mais le top-k du reranker les avait masqués au
# modèle. Ils ne remplacent pas l'adjudication : Qwen doit encore les retenir
# et tous les garde-fous doivent passer.
CLAIM_COVERAGE_HINTS: dict[str, tuple[str, ...]] = {
    "Q038": (
        "01_becker_phosphates_and_phosphoric_acid_000481_5c315f6d96ad",
    ),
    "Q043": (
        "01_becker_phosphates_and_phosphoric_acid_000665_f57b9a2db1c7",
    ),
}

DECISION_KEYS = {
    "question_id",
    "selected_chunk_ids",
    "confidence",
    "justification",
    "missing_claims",
}

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z])\d+(?:[.,]\d+)?(?::\d+(?:[.,]\d+)?)?%?"
)
FORMULA_PATTERN = re.compile(
    r"\b(?:[A-Z][a-z]?\d*){2,}\b"
)
FORMULA_OCR_ALIASES: dict[str, tuple[str, ...]] = {
    "al2o3": ("a1203", "algog", "al2°3", "a!2°3"),
    "sio2": ("si02",),
}

# Garde-fous de polarité pour les formulations qui ont déjà provoqué des
# faux positifs (un passage décrivant l'effet opposé ne doit pas passer).
CRITICAL_PHRASE_GROUPS: tuple[
    tuple[tuple[str, ...], tuple[str, ...]],
    ...,
] = (
    (
        ("low sulfate", "faible taux de sulfate"),
        (
            "low sulfate",
            "lower sulfate",
            "very low sulfate",
            "faible taux de sulfate",
        ),
    ),
    (
        (
            "high solids",
            "high solid content",
            "higher solid content",
            "teneur elevee en solides",
        ),
        (
            "high solids",
            "high solid content",
            "higher solid content",
            "teneur elevee en solides",
        ),
    ),
)

SYSTEM_PROMPT = """Tu es l'adjudicateur final des preuves d'un benchmark RAG industriel.

Règles impératives :
1. Sélectionne le plus petit ensemble de 1 à 3 chunks qui permet de justifier
   TOUTES les affirmations importantes de la réponse de référence.
2. Un passage seulement proche du thème, mais qui n'établit pas la réponse,
   ne doit jamais être sélectionné.
3. N'utilise aucune connaissance externe et n'invente aucun lien causal.
4. Pour les nombres, tableaux, comparaisons et causalités, vérifie que chaque
   valeur, entité, cause, mécanisme et conséquence importante est attesté.
5. Utilise uniquement des chunk_id fournis. Ne reformule jamais un identifiant.
6. Préfère les preuves du document attendu.
7. Si la couverture est incomplète, laisse selected_chunk_ids vide ou conserve
   seulement les preuves réelles, mets confidence sous 0.85 et énumère
   précisément les éléments manquants dans missing_claims.
8. confidence mesure la probabilité que l'ensemble sélectionné couvre
   complètement et exactement la réponse. N'accorde jamais >= 0.85 à une
   preuve seulement thématique ou partielle.
9. Retourne uniquement l'objet JSON conforme au schéma, sans commentaire."""


class GoldValidationError(ValueError):
    """Erreur de structure ou de cohérence d'un artefact gold."""


def natural_key(value: str) -> list[int | str]:
    """Construire une clé de tri naturelle pour les identifiants."""

    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    ]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Lire un JSONL UTF-8 et contrôler chaque objet."""

    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    records: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise GoldValidationError(
                f"JSON invalide dans {path}, ligne {line_number}: {error}"
            ) from error

        if not isinstance(record, dict):
            raise GoldValidationError(
                f"Objet JSON attendu dans {path}, ligne {line_number}."
            )

        records.append(record)

    return records


def read_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    """Lire un JSONL optionnel, ou retourner une liste vide."""

    if not path.exists():
        return []

    return read_jsonl(path)


def write_jsonl_atomic(
    records: Iterable[dict[str, Any]],
    path: Path,
) -> None:
    """Écrire un JSONL, avec fallback Windows si os.replace est refusé."""

    resolved_path = path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    if resolved_path.exists() and resolved_path.is_dir():
        raise IsADirectoryError(
            f"Le chemin de sortie est un dossier : {resolved_path}"
        )

    payload = "".join(
        json.dumps(record, ensure_ascii=False) + "\n"
        for record in records
    )
    temporary_path = resolved_path.with_name(
        f".{resolved_path.name}.{os.getpid()}.tmp"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        try:
            os.replace(temporary_path, resolved_path)
        except PermissionError:
            # Sur certains volumes Windows, le renommage est bloqué alors que
            # l'écriture directe est autorisée.
            with resolved_path.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
    finally:
        temporary_path.unlink(missing_ok=True)


def records_by_question(
    records: Iterable[dict[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    """Indexer des objets par question_id sans tolérer de doublon."""

    indexed: dict[str, dict[str, Any]] = {}

    for record in records:
        question_id = str(
            record.get("query_id")
            or record.get("question_id")
            or ""
        ).strip()

        if not question_id:
            raise GoldValidationError(
                f"{label}: question_id/query_id absent."
            )

        if question_id in indexed:
            raise GoldValidationError(
                f"{label}: question_id dupliqué : {question_id}."
            )

        indexed[question_id] = record

    return indexed


def load_chunks(
    chunks_directory: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Charger tous les chunks complets et les grouper par document."""

    chunk_paths = sorted(chunks_directory.glob("*_chunks.jsonl"))

    if not chunk_paths:
        raise FileNotFoundError(
            f"Aucun fichier de chunks dans {chunks_directory}"
        )

    chunks_by_id: dict[str, dict[str, Any]] = {}
    chunks_by_document: dict[str, list[dict[str, Any]]] = {}

    for chunk_path in chunk_paths:
        for chunk in read_jsonl(chunk_path):
            chunk_id = str(chunk.get("chunk_id", "")).strip()
            document_id = str(chunk.get("document_id", "")).strip()
            text = str(chunk.get("text", "")).strip()

            if not chunk_id or not document_id or not text:
                raise GoldValidationError(
                    f"Chunk incomplet dans {chunk_path}: {chunk_id!r}."
                )

            if chunk_id in chunks_by_id:
                raise GoldValidationError(
                    f"chunk_id dupliqué dans le corpus : {chunk_id}."
                )

            chunks_by_id[chunk_id] = chunk
            chunks_by_document.setdefault(document_id, []).append(chunk)

    for chunks in chunks_by_document.values():
        chunks.sort(
            key=lambda chunk: (
                int(chunk.get("chunk_index", 0)),
                str(chunk["chunk_id"]),
            )
        )

    return chunks_by_id, chunks_by_document


def load_pool_by_question(
    pool_path: Path,
    chunks_by_id: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Charger le pool et vérifier qu'il pointe vers le corpus complet."""

    pool_by_question: dict[str, list[dict[str, Any]]] = {}
    seen_pairs: set[tuple[str, str]] = set()

    for pool_item in read_jsonl(pool_path):
        question_id = str(pool_item.get("query_id", "")).strip()
        chunk_id = str(pool_item.get("chunk_id", "")).strip()
        pair = (question_id, chunk_id)

        if not question_id or not chunk_id:
            raise GoldValidationError(
                "Le pool contient une paire sans query_id/chunk_id."
            )

        if pair in seen_pairs:
            raise GoldValidationError(
                f"Paire dupliquée dans le pool : {question_id} / {chunk_id}."
            )

        if chunk_id not in chunks_by_id:
            raise GoldValidationError(
                f"Chunk du pool absent du corpus : {chunk_id}."
            )

        seen_pairs.add(pair)
        item = dict(pool_item)
        item["_full_chunk"] = chunks_by_id[chunk_id]
        pool_by_question.setdefault(question_id, []).append(item)

    for items in pool_by_question.values():
        items.sort(
            key=lambda item: (
                int(item.get("display_order", 10_000)),
                str(item["chunk_id"]),
            )
        )

    return pool_by_question


def make_decision_schema(question_id: str) -> dict[str, Any]:
    """Créer le schéma JSON strict demandé à Ollama."""

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "question_id": {
                "type": "string",
                "const": question_id,
            },
            "selected_chunk_ids": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 0,
                "maxItems": 3,
                "uniqueItems": True,
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "justification": {
                "type": "string",
                "minLength": 1,
            },
            "missing_claims": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "uniqueItems": True,
            },
        },
        "required": sorted(DECISION_KEYS),
    }


def ollama_json(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    prompt: str,
    schema: dict[str, Any],
    num_ctx: int,
    retries: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Obtenir une réponse JSON structurée depuis l'API locale Ollama."""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,
        "format": schema,
        "options": {
            "temperature": 0,
            "num_ctx": num_ctx,
        },
        "keep_alive": "10m",
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout_seconds,
            ) as response:
                response_payload = json.loads(
                    response.read().decode("utf-8")
                )

            decision = json.loads(
                response_payload["message"]["content"]
            )

            if not isinstance(decision, dict):
                raise GoldValidationError(
                    "Ollama n'a pas retourné un objet JSON."
                )

            return decision
        except (
            GoldValidationError,
            TimeoutError,
            KeyError,
            ValueError,
            json.JSONDecodeError,
            urllib.error.HTTPError,
            urllib.error.URLError,
        ) as error:
            last_error = error

            if attempt < retries:
                time.sleep(min(2 * attempt, 5))

    raise RuntimeError(
        f"Ollama a échoué après {retries} tentative(s): {last_error}"
    )


def validate_llm_decision(
    raw_decision: dict[str, Any],
    *,
    question_id: str,
    allowed_chunk_ids: set[str],
) -> dict[str, Any]:
    """Refuser toute sortie non stricte ou contenant un identifiant inventé."""

    actual_keys = set(raw_decision)

    if actual_keys != DECISION_KEYS:
        missing = sorted(DECISION_KEYS - actual_keys)
        unexpected = sorted(actual_keys - DECISION_KEYS)
        raise GoldValidationError(
            "Sortie Ollama non stricte. "
            f"Champs absents={missing}, inattendus={unexpected}."
        )

    if str(raw_decision["question_id"]).strip() != question_id:
        raise GoldValidationError(
            "Ollama a retourné un question_id incorrect."
        )

    raw_selected = raw_decision["selected_chunk_ids"]

    if not isinstance(raw_selected, list):
        raise GoldValidationError(
            "selected_chunk_ids doit être une liste."
        )

    selected_chunk_ids = [
        str(chunk_id).strip()
        for chunk_id in raw_selected
    ]

    if any(not chunk_id for chunk_id in selected_chunk_ids):
        raise GoldValidationError(
            "selected_chunk_ids contient un identifiant vide."
        )

    if len(selected_chunk_ids) > 3:
        raise GoldValidationError(
            "Ollama a sélectionné plus de trois chunks."
        )

    if len(selected_chunk_ids) != len(set(selected_chunk_ids)):
        raise GoldValidationError(
            "Ollama a sélectionné deux fois le même chunk."
        )

    unknown_chunk_ids = sorted(
        set(selected_chunk_ids) - allowed_chunk_ids
    )

    if unknown_chunk_ids:
        raise GoldValidationError(
            "Identifiant(s) Ollama absent(s) des candidats autorisés : "
            + ", ".join(unknown_chunk_ids)
        )

    try:
        confidence = float(raw_decision["confidence"])
    except (TypeError, ValueError) as error:
        raise GoldValidationError(
            "confidence doit être un nombre."
        ) from error

    if not 0 <= confidence <= 1:
        raise GoldValidationError(
            "confidence doit être comprise entre 0 et 1."
        )

    justification = str(raw_decision["justification"]).strip()

    if not justification:
        raise GoldValidationError(
            "justification ne peut pas être vide."
        )

    raw_missing_claims = raw_decision["missing_claims"]

    if not isinstance(raw_missing_claims, list):
        raise GoldValidationError(
            "missing_claims doit être une liste."
        )

    missing_claims = [
        str(claim).strip()
        for claim in raw_missing_claims
    ]

    if any(not claim for claim in missing_claims):
        raise GoldValidationError(
            "missing_claims contient une affirmation vide."
        )

    if len(missing_claims) != len(set(missing_claims)):
        raise GoldValidationError(
            "missing_claims contient un doublon."
        )

    return {
        "question_id": question_id,
        "selected_chunk_ids": selected_chunk_ids,
        "confidence": confidence,
        "justification": justification,
        "missing_claims": missing_claims,
    }


def reranking_query(question: dict[str, Any]) -> str:
    """Construire la requête du reranker avec la réponse à couvrir."""

    expected_documents = ", ".join(
        str(value)
        for value in question.get("reference_documents", [])
    )

    return (
        f"Question: {question['question']}\n"
        f"Réponse de référence à prouver: {question['expected_answer']}\n"
        f"Document attendu: {expected_documents}"
    )


class ProjectReranker:
    """Adaptateur léger vers le BGEReranker déjà utilisé par le projet."""

    def __init__(self, config_path: Path) -> None:
        # Imports différés : les validations unitaires n'ont pas à charger
        # torch/FlagEmbedding ni le modèle.
        from phosprocess.reranking.reranker import (
            BGEReranker,
            load_reranking_config,
        )

        self._config = load_reranking_config(config_path)
        self._reranker = BGEReranker(self._config)

    @property
    def model_name(self) -> str:
        return str(self._config.model_name)

    def rerank(
        self,
        *,
        question: dict[str, Any],
        chunks: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Classer tous les chunks reçus et retourner les top_k."""

        from phosprocess.preprocessing.chunk_schemas import DocumentChunk
        from phosprocess.retrieval.hybrid import HybridSearchResult

        if not chunks:
            return []

        candidates = [
            HybridSearchResult(
                rank=rank,
                rrf_score=0.0,
                matched_retrievers=("adjudication",),
                dense_rank=None,
                dense_score=None,
                dense_rrf_contribution=0.0,
                bm25_rank=None,
                bm25_score=None,
                bm25_rrf_contribution=0.0,
                chunk=DocumentChunk.model_validate(chunk),
            )
            for rank, chunk in enumerate(chunks, start=1)
        ]
        response = self._reranker.rerank(
            reranking_query(question),
            candidates,
            top_k=min(top_k, len(candidates)),
        )

        ranked_chunks: list[dict[str, Any]] = []

        for result in response.results:
            chunk = result.chunk.model_dump()
            chunk["_reranker_rank"] = result.rank
            chunk["_reranker_score"] = result.reranker_score
            ranked_chunks.append(chunk)

        return ranked_chunks


def compact_text(text: str, maximum_characters: int) -> str:
    """Limiter un passage en conservant ses espaces lisibles."""

    compacted = re.sub(r"[ \t]+", " ", text).strip()

    if len(compacted) <= maximum_characters:
        return compacted

    return compacted[: maximum_characters - 1].rstrip() + "…"


def normalize_evidence_text(text: str) -> str:
    """Normaliser le texte sans perdre les relations lexicales importantes."""

    normalized = html.unescape(str(text))
    normalized = HTML_TAG_PATTERN.sub(" ", normalized)
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    normalized = normalized.casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def important_numbers(text: str) -> set[str]:
    """Extraire les valeurs, sans confondre les indices de formules/unités."""

    return {
        value.replace(",", ".")
        for value in NUMBER_PATTERN.findall(text)
        if value not in {"0", "1", "2", "3"}
    }


def chemical_formulas(text: str) -> set[str]:
    """Extraire les formules comportant au moins un indice numérique."""

    return {
        formula.casefold()
        for formula in FORMULA_PATTERN.findall(text)
        if any(character.isdigit() for character in formula)
    }


def formulas_are_covered(
    expected_formulas: set[str],
    evidence_text: str,
) -> set[str]:
    """Retourner les formules non attestées, avec tolérance OCR limitée."""

    evidence_formulas = chemical_formulas(evidence_text)
    normalized_raw_evidence = normalize_evidence_text(evidence_text)
    missing: set[str] = set()

    for expected in expected_formulas:
        if expected in evidence_formulas:
            continue

        if any(
            alias in normalized_raw_evidence
            for alias in FORMULA_OCR_ALIASES.get(expected, ())
        ):
            continue

        # Tolérer les confusions OCR usuelles (Al/A1, O/0), sans accepter
        # l'absence totale d'une formule telle que K2O.
        ocr_expected = (
            expected.replace("l", "1")
            .replace("o", "0")
        )
        normalized_evidence = {
            formula.replace("l", "1").replace("o", "0")
            for formula in evidence_formulas
        }

        if ocr_expected not in normalized_evidence:
            missing.add(expected)

    return missing


def inject_priority_chunks(
    ranked_chunks: list[dict[str, Any]],
    *,
    priority_chunk_ids: Iterable[str],
    available_chunks: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Garantir la présence des preuves imposées/drafts dans le prompt."""

    result = list(ranked_chunks)
    present_ids = {
        str(chunk["chunk_id"])
        for chunk in result
    }

    for chunk_id in priority_chunk_ids:
        if chunk_id in available_chunks and chunk_id not in present_ids:
            priority = dict(available_chunks[chunk_id])
            priority["_reranker_rank"] = None
            priority["_reranker_score"] = None
            priority["_priority_injected"] = True
            result.insert(0, priority)
            present_ids.add(chunk_id)

    return result


def build_adjudication_prompt(
    *,
    question: dict[str, Any],
    candidates: list[dict[str, Any]],
    stage: str,
    draft: dict[str, Any] | None,
    override_ids: tuple[str, ...],
    maximum_characters: int,
) -> str:
    """Présenter les passages rerankés et les contraintes à Qwen."""

    candidate_blocks: list[str] = []

    for candidate in candidates:
        headings = " > ".join(
            str(value)
            for value in candidate.get("heading_path", [])
            if str(value).strip()
        )
        pages = ", ".join(
            str(value)
            for value in candidate.get("source_pages", [])
        )
        reranker_score = candidate.get("_reranker_score")
        score_label = (
            f"{float(reranker_score):.6f}"
            if reranker_score is not None
            else "override/draft injecté"
        )
        candidate_blocks.append(
            "\n".join(
                [
                    f"chunk_id: {candidate['chunk_id']}",
                    f"document_id: {candidate['document_id']}",
                    f"pages: {pages or '?'}",
                    f"section: {headings or 'inconnue'}",
                    f"reranker_score: {score_label}",
                    "--- PASSAGE ---",
                    compact_text(
                        str(candidate["text"]),
                        maximum_characters,
                    ),
                    "--- FIN PASSAGE ---",
                ]
            )
        )

    draft_note = "Aucun draft."

    if draft is not None:
        draft_note = (
            "Draft Qwen antérieur, fourni uniquement comme piste non fiable : "
            f"ids={draft.get('gold_chunk_ids', [])}; "
            f"confiance={draft.get('confidence', 0)}; "
            f"raison={draft.get('reason', '')}"
        )

    override_note = (
        "Aucun override."
        if not override_ids
        else (
            "Preuve(s) explicitement identifiée(s) hors pool à contrôler et "
            "à inclure si elle(s) couvre(nt) réellement la réponse : "
            + ", ".join(override_ids)
        )
    )

    return f"""Étape de recherche: {stage}

question_id: {question['query_id']}
Question: {question['question']}
Réponse de référence: {question['expected_answer']}
Catégorie: {question['category']}
Document(s) attendu(s): {question.get('reference_documents', [])}

{draft_note}
{override_note}

Passages classés par BAAI/bge-reranker-v2-m3 :
{'=' * 100}
{chr(10).join(candidate_blocks)}
{'=' * 100}

Vérifie séparément chaque affirmation importante de la réponse, puis choisis
le plus petit ensemble qui les couvre toutes. Si une affirmation manque,
inscris-la dans missing_claims et baisse la confiance sous 0.85."""


def request_decision(
    *,
    question: dict[str, Any],
    candidates: list[dict[str, Any]],
    stage: str,
    draft: dict[str, Any] | None,
    override_ids: tuple[str, ...],
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    """Demander et valider une adjudication stricte."""

    question_id = str(question["query_id"])
    allowed_chunk_ids = {
        str(candidate["chunk_id"])
        for candidate in candidates
    }

    if not allowed_chunk_ids:
        raise GoldValidationError(
            f"Aucun candidat à présenter pour {question_id}."
        )

    raw_decision = ollama_json(
        base_url=arguments.ollama_url,
        model=arguments.model,
        system_prompt=SYSTEM_PROMPT,
        prompt=build_adjudication_prompt(
            question=question,
            candidates=candidates,
            stage=stage,
            draft=draft,
            override_ids=override_ids,
            maximum_characters=arguments.maximum_chunk_characters,
        ),
        schema=make_decision_schema(question_id),
        num_ctx=arguments.num_ctx,
        retries=arguments.retries,
        timeout_seconds=arguments.timeout_seconds,
    )

    return validate_llm_decision(
        raw_decision,
        question_id=question_id,
        allowed_chunk_ids=allowed_chunk_ids,
    )


def decision_safety_issues(
    decision: dict[str, Any],
    *,
    question: dict[str, Any],
    chunks_by_id: dict[str, dict[str, Any]],
    confidence_threshold: float,
    required_override_ids: tuple[str, ...] = (),
) -> list[str]:
    """Énumérer les raisons qui imposent une revue humaine."""

    issues: list[str] = []
    selected_ids = [
        str(value)
        for value in decision["selected_chunk_ids"]
    ]

    if not 1 <= len(selected_ids) <= 3:
        issues.append(
            "Une question répondable doit avoir entre 1 et 3 chunks."
        )

    if float(decision["confidence"]) < confidence_threshold:
        issues.append(
            f"Confiance inférieure à {confidence_threshold:.2f}."
        )

    if decision["missing_claims"]:
        issues.append(
            "Des affirmations importantes restent non couvertes."
        )

    combined_evidence = "\n".join(
        str(chunks_by_id[chunk_id]["text"])
        for chunk_id in selected_ids
    )
    expected_answer = str(question.get("expected_answer") or "")
    expected_numbers = important_numbers(expected_answer)
    evidence_numbers = important_numbers(combined_evidence)
    missing_numbers = sorted(expected_numbers - evidence_numbers)

    if missing_numbers:
        issues.append(
            "Valeur(s) de la réponse absente(s) des preuves : "
            + ", ".join(missing_numbers)
        )

    formula_source = (
        str(question.get("question") or "")
        + "\n"
        + expected_answer
    )
    missing_formulas = sorted(
        formulas_are_covered(
            chemical_formulas(formula_source),
            combined_evidence,
        )
    )

    if missing_formulas:
        issues.append(
            "Formule(s) chimique(s) non attestée(s) : "
            + ", ".join(missing_formulas)
        )

    normalized_claims = normalize_evidence_text(formula_source)
    normalized_evidence = normalize_evidence_text(combined_evidence)

    for triggers, evidence_phrases in CRITICAL_PHRASE_GROUPS:
        if not any(trigger in normalized_claims for trigger in triggers):
            continue

        if not any(
            phrase in normalized_evidence
            for phrase in evidence_phrases
        ):
            issues.append(
                "Polarité/condition importante non attestée : "
                + triggers[0]
            )

    expected_documents = {
        str(value)
        for value in question.get("reference_documents", [])
    }
    off_document_ids = [
        chunk_id
        for chunk_id in selected_ids
        if str(chunks_by_id[chunk_id]["document_id"])
        not in expected_documents
    ]

    if off_document_ids:
        issues.append(
            "Sélection hors document attendu : "
            + ", ".join(off_document_ids)
        )

    missing_override_ids = sorted(
        set(required_override_ids) - set(selected_ids)
    )

    if missing_override_ids:
        issues.append(
            "Override explicite non retenu : "
            + ", ".join(missing_override_ids)
        )

    return issues


def automatic_unanswerable_decision(
    question_id: str,
) -> dict[str, Any]:
    """Créer la décision déterministe des quatre questions non répondables."""

    return {
        "question_id": question_id,
        "selected_chunk_ids": [],
        "confidence": 1.0,
        "justification": (
            "Question TEST déclarée non répondable dans le benchmark."
        ),
        "missing_claims": [],
    }


def make_manual_review_record(
    *,
    question: dict[str, Any],
    decision: dict[str, Any] | None,
    issues: list[str],
    stage: str,
    error: str | None = None,
) -> dict[str, Any]:
    """Créer une entrée autonome pour la revue manuelle."""

    return {
        "question_id": str(question["query_id"]),
        "question": str(question["question"]),
        "reference_answer": question.get("expected_answer"),
        "reference_documents": list(
            question.get("reference_documents", [])
        ),
        "last_decision": decision,
        "review_reasons": issues,
        "last_stage": stage,
        "error": error,
    }


def merged_gold_record(
    decision: dict[str, Any],
    *,
    question: dict[str, Any],
    assessor_id: str,
    updated_at: str,
) -> dict[str, Any]:
    """Convertir une décision automatique vers le format gold existant."""

    return {
        "query_id": str(question["query_id"]),
        "split": str(question.get("split", "test")),
        "category": str(question["category"]),
        "answerable": bool(question["answerable"]),
        "gold_chunk_ids": list(decision["selected_chunk_ids"]),
        "status": "verified",
        "assessor_id": assessor_id,
        "updated_at": updated_at,
    }


def validate_human_gold(
    human_records: list[dict[str, Any]],
    *,
    questions_by_id: dict[str, dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
) -> None:
    """Contrôler les gold humains sans les corriger ni les réécrire."""

    indexed = records_by_question(
        human_records,
        label="gold humains",
    )
    unknown_questions = sorted(
        set(indexed) - set(questions_by_id)
    )

    if unknown_questions:
        raise GoldValidationError(
            "Gold humains hors TEST : " + ", ".join(unknown_questions)
        )

    for question_id, record in indexed.items():
        chunk_ids = record.get("gold_chunk_ids")

        if not isinstance(chunk_ids, list):
            raise GoldValidationError(
                f"{question_id}: gold_chunk_ids humain doit être une liste."
            )

        if len(chunk_ids) != len(set(chunk_ids)):
            raise GoldValidationError(
                f"{question_id}: chunk humain dupliqué."
            )

        missing_ids = sorted(set(chunk_ids) - set(chunks_by_id))

        if missing_ids:
            raise GoldValidationError(
                f"{question_id}: chunks humains absents du corpus : "
                + ", ".join(missing_ids)
            )

        answerable = bool(questions_by_id[question_id]["answerable"])

        if answerable and not 1 <= len(chunk_ids) <= 3:
            raise GoldValidationError(
                f"{question_id}: gold humain répondable invalide."
            )

        if not answerable and chunk_ids:
            raise GoldValidationError(
                f"{question_id}: gold humain non répondable non vide."
            )


def validate_final_dataset(
    merged_records: list[dict[str, Any]],
    *,
    test_questions: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    human_records: list[dict[str, Any]],
) -> None:
    """Appliquer toutes les validations finales exigées."""

    questions_by_id = records_by_question(
        test_questions,
        label="questions TEST",
    )
    merged_by_id = records_by_question(
        merged_records,
        label="gold fusionnés",
    )
    human_by_id = records_by_question(
        human_records,
        label="gold humains",
    )

    if len(questions_by_id) != 28:
        raise GoldValidationError(
            f"Le TEST doit contenir 28 questions, trouvé {len(questions_by_id)}."
        )

    if set(questions_by_id) != EXPECTED_TEST_IDS:
        raise GoldValidationError(
            "Les question_id TEST doivent couvrir exactement Q021 à Q048."
        )

    if len(merged_by_id) != 28:
        raise GoldValidationError(
            f"Le gold fusionné doit contenir 28 entrées, trouvé {len(merged_by_id)}."
        )

    if set(merged_by_id) != set(questions_by_id):
        missing = sorted(set(questions_by_id) - set(merged_by_id))
        unexpected = sorted(set(merged_by_id) - set(questions_by_id))
        raise GoldValidationError(
            "Couverture TEST incorrecte. "
            f"Absents={missing}, inattendus={unexpected}."
        )

    answerable_count = 0
    unanswerable_count = 0

    for question_id, question in questions_by_id.items():
        record = merged_by_id[question_id]
        chunk_ids = record.get("gold_chunk_ids")

        if not isinstance(chunk_ids, list):
            raise GoldValidationError(
                f"{question_id}: gold_chunk_ids doit être une liste."
            )

        if len(chunk_ids) != len(set(chunk_ids)):
            raise GoldValidationError(
                f"{question_id}: chunk_id dupliqué dans les gold."
            )

        missing_chunk_ids = sorted(set(chunk_ids) - set(chunks_by_id))

        if missing_chunk_ids:
            raise GoldValidationError(
                f"{question_id}: chunk_id absent du corpus : "
                + ", ".join(missing_chunk_ids)
            )

        if bool(question["answerable"]):
            answerable_count += 1

            if not 1 <= len(chunk_ids) <= 3:
                raise GoldValidationError(
                    f"{question_id}: une question répondable doit avoir "
                    "entre 1 et 3 gold chunks."
                )
        else:
            unanswerable_count += 1

            if chunk_ids:
                raise GoldValidationError(
                    f"{question_id}: une question non répondable doit avoir "
                    "zéro gold chunk."
                )

    if answerable_count != 24:
        raise GoldValidationError(
            f"24 questions répondables attendues, trouvé {answerable_count}."
        )

    if unanswerable_count != 4:
        raise GoldValidationError(
            f"4 questions non répondables attendues, trouvé {unanswerable_count}."
        )

    actual_unanswerable_ids = {
        question_id
        for question_id, question in questions_by_id.items()
        if not bool(question["answerable"])
    }

    if actual_unanswerable_ids != EXPECTED_UNANSWERABLE_IDS:
        raise GoldValidationError(
            "Les questions non répondables doivent être Q045 à Q048."
        )

    for question_id in EXPECTED_UNANSWERABLE_IDS:
        if merged_by_id[question_id]["gold_chunk_ids"] != []:
            raise GoldValidationError(
                f"{question_id}: selected_chunk_ids/gold_chunk_ids doit être []."
            )

    for question_id, human_record in human_by_id.items():
        if merged_by_id.get(question_id) != human_record:
            raise GoldValidationError(
                f"Le gold humain {question_id} a été modifié."
            )


def ensure_output_paths_are_safe(
    *,
    human_gold_path: Path,
    output_paths: Iterable[Path],
) -> None:
    """Empêcher qu'une sortie cible le fichier humain protégé."""

    protected = human_gold_path.resolve()
    seen: set[Path] = set()

    for output_path in output_paths:
        resolved = output_path.resolve()

        if resolved == protected:
            raise GoldValidationError(
                "Une sortie ne peut pas écraser "
                "gold_evidence_test_verified.jsonl."
            )

        if resolved in seen:
            raise GoldValidationError(
                f"Deux sorties utilisent le même chemin : {resolved}."
            )

        seen.add(resolved)


def draft_priority_ids(
    draft: dict[str, Any] | None,
) -> tuple[str, ...]:
    """Extraire les propositions d'un draft sans leur faire confiance."""

    if draft is None:
        return ()

    raw_ids = draft.get("gold_chunk_ids", [])

    if not isinstance(raw_ids, list):
        return ()

    return tuple(
        str(chunk_id).strip()
        for chunk_id in raw_ids
        if str(chunk_id).strip()
    )


def parse_arguments() -> argparse.Namespace:
    """Lire les options de l'adjudication automatique."""

    parser = argparse.ArgumentParser(
        description=(
            "Adjudication automatique prudente des gold evidence TEST v2."
        )
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
    )
    parser.add_argument(
        "--pool",
        type=Path,
        default=DEFAULT_POOL_PATH,
    )
    parser.add_argument(
        "--chunks-directory",
        type=Path,
        default=CHUNKS_DIRECTORY,
    )
    parser.add_argument(
        "--drafts",
        type=Path,
        default=DEFAULT_DRAFTS_PATH,
    )
    parser.add_argument(
        "--human-gold",
        type=Path,
        default=DEFAULT_HUMAN_GOLD_PATH,
    )
    parser.add_argument(
        "--automatic-output",
        type=Path,
        default=DEFAULT_AUTO_OUTPUT_PATH,
    )
    parser.add_argument(
        "--manual-review-queue",
        type=Path,
        default=DEFAULT_MANUAL_REVIEW_PATH,
    )
    parser.add_argument(
        "--merged-output",
        type=Path,
        default=DEFAULT_MERGED_OUTPUT_PATH,
    )
    parser.add_argument(
        "--reranking-config",
        type=Path,
        default=RERANKING_CONFIG_PATH,
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
    )
    parser.add_argument(
        "--model",
        default="qwen3:8b",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--pool-top-k",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--document-top-k",
        type=int,
        default=18,
    )
    parser.add_argument(
        "--maximum-chunk-characters",
        type=int,
        default=5_500,
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=32_768,
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=240,
    )

    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    """Refuser les options incohérentes avant de charger les modèles."""

    if not 0 <= arguments.confidence_threshold <= 1:
        raise ValueError(
            "--confidence-threshold doit être comprise entre 0 et 1."
        )

    for name in (
        "pool_top_k",
        "document_top_k",
        "maximum_chunk_characters",
        "num_ctx",
        "retries",
        "timeout_seconds",
    ):
        if int(getattr(arguments, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} doit être positif.")


def main() -> None:
    """Exécuter l'adjudication, la fusion et les validations finales."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    arguments = parse_arguments()
    validate_arguments(arguments)
    ensure_output_paths_are_safe(
        human_gold_path=arguments.human_gold,
        output_paths=(
            arguments.automatic_output,
            arguments.manual_review_queue,
            arguments.merged_output,
        ),
    )

    human_bytes_before = arguments.human_gold.read_bytes()
    all_questions = read_jsonl(arguments.queries)
    test_questions = sorted(
        (
            question
            for question in all_questions
            if str(question.get("split", "")).casefold() == "test"
        ),
        key=lambda question: natural_key(str(question["query_id"])),
    )
    questions_by_id = records_by_question(
        test_questions,
        label="questions TEST",
    )
    chunks_by_id, chunks_by_document = load_chunks(
        arguments.chunks_directory
    )
    pool_by_question = load_pool_by_question(
        arguments.pool,
        chunks_by_id,
    )
    drafts_by_id = records_by_question(
        read_jsonl(arguments.drafts),
        label="drafts Qwen",
    )
    human_records = read_jsonl(arguments.human_gold)
    human_by_id = records_by_question(
        human_records,
        label="gold humains",
    )

    if set(questions_by_id) != EXPECTED_TEST_IDS:
        raise GoldValidationError(
            "Le split TEST doit contenir exactement Q021 à Q048."
        )

    validate_human_gold(
        human_records,
        questions_by_id=questions_by_id,
        chunks_by_id=chunks_by_id,
    )

    for evidence_label, evidence_by_question in (
        ("Override", EVIDENCE_OVERRIDES),
        ("Indice de couverture", CLAIM_COVERAGE_HINTS),
    ):
        for question_id, evidence_ids in evidence_by_question.items():
            question = questions_by_id.get(question_id)

            if question is None:
                raise GoldValidationError(
                    f"{evidence_label}: question absente : {question_id}."
                )

            expected_documents = {
                str(value)
                for value in question.get("reference_documents", [])
            }

            for chunk_id in evidence_ids:
                if chunk_id not in chunks_by_id:
                    raise GoldValidationError(
                        f"{evidence_label} absent du corpus : {chunk_id}."
                    )

                if (
                    str(chunks_by_id[chunk_id]["document_id"])
                    not in expected_documents
                ):
                    raise GoldValidationError(
                        f"{evidence_label} hors document attendu pour "
                        f"{question_id}: {chunk_id}."
                    )

    candidate_remaining_questions = [
        question
        for question in test_questions
        if str(question["query_id"]) not in human_by_id
    ]
    cached_automatic_by_id = records_by_question(
        read_optional_jsonl(arguments.automatic_output),
        label="gold automatiques existants",
    )
    cached_manual_review_by_id = records_by_question(
        read_optional_jsonl(arguments.manual_review_queue),
        label="revue manuelle existante",
    )
    automatic_by_id: dict[str, dict[str, Any]] = {}
    cached_rejected_by_id: dict[str, tuple[dict[str, Any], list[str]]] = {}

    for question in candidate_remaining_questions:
        question_id = str(question["query_id"])
        cached = cached_automatic_by_id.get(question_id)

        if cached is None:
            continue

        try:
            validated_cached = validate_llm_decision(
                cached,
                question_id=question_id,
                allowed_chunk_ids=set(chunks_by_id),
            )
            cached_issues = (
                []
                if not bool(question["answerable"])
                else decision_safety_issues(
                    validated_cached,
                    question=question,
                    chunks_by_id=chunks_by_id,
                    confidence_threshold=(
                        arguments.confidence_threshold
                    ),
                    required_override_ids=(
                        EVIDENCE_OVERRIDES.get(question_id, ())
                    ),
                )
            )
        except Exception as error:
            validated_cached = cached
            cached_issues = [
                "Décision automatique existante invalide : "
                f"{type(error).__name__}: {error}"
            ]

        if not cached_issues:
            automatic_by_id[question_id] = validated_cached
        else:
            cached_rejected_by_id[question_id] = (
                validated_cached,
                cached_issues,
            )

    for question in candidate_remaining_questions:
        question_id = str(question["query_id"])

        if (
            question_id in automatic_by_id
            or question_id in cached_rejected_by_id
        ):
            continue

        cached_review = cached_manual_review_by_id.get(question_id)

        if cached_review is None:
            continue

        last_decision = cached_review.get("last_decision")

        if not isinstance(last_decision, dict):
            continue

        cached_rejected_by_id[question_id] = (
            last_decision,
            [
                str(issue)
                for issue in cached_review.get("review_reasons", [])
            ]
            or ["Décision précédente envoyée en revue manuelle."],
        )

    remaining_questions = [
        question
        for question in candidate_remaining_questions
        if str(question["query_id"]) not in automatic_by_id
    ]
    remaining_answerable = [
        question
        for question in remaining_questions
        if bool(question["answerable"])
    ]

    print("\n=== Auto-vérification Gold Evidence TEST v2 ===")
    print(f"Questions TEST          : {len(test_questions)}")
    print(f"Gold humains conservés  : {len(human_records)}")
    print(f"Gold auto réutilisés    : {len(automatic_by_id)}")
    print(f"Questions à adjuger     : {len(remaining_questions)}")
    print(f"Modèle Ollama           : {arguments.model}")
    print("Reranker                : BAAI/bge-reranker-v2-m3")

    reranker: ProjectReranker | None = None

    if remaining_answerable:
        reranker = ProjectReranker(arguments.reranking_config)

    manual_review_by_id: dict[str, dict[str, Any]] = {}
    automatic_timestamps: dict[str, str] = {
        question_id: datetime.now(UTC).isoformat()
        for question_id in automatic_by_id
    }

    for position, question in enumerate(remaining_questions, start=1):
        question_id = str(question["query_id"])
        print(
            f"\n[{position}/{len(remaining_questions)}] "
            f"{question_id} | {question['category']}"
        )

        if not bool(question["answerable"]):
            automatic_by_id[question_id] = (
                automatic_unanswerable_decision(question_id)
            )
            automatic_timestamps[question_id] = datetime.now(UTC).isoformat()
            print("  -> non répondable, gold=[]")
            continue

        if reranker is None:
            raise AssertionError("Le reranker n'a pas été initialisé.")

        draft = drafts_by_id.get(question_id)
        override_ids = EVIDENCE_OVERRIDES.get(question_id, ())
        coverage_hint_ids = CLAIM_COVERAGE_HINTS.get(question_id, ())
        pool_items = pool_by_question.get(question_id, [])
        pool_chunks = [
            item["_full_chunk"]
            for item in pool_items
        ]
        last_decision: dict[str, Any] | None = None
        last_issues: list[str] = []
        last_stage = "pool"
        last_error: str | None = None
        rejected_cached = cached_rejected_by_id.get(question_id)

        if rejected_cached is not None:
            last_decision, last_issues = rejected_cached
            print(
                "  Décision auto existante refusée : "
                + "; ".join(last_issues)
            )
        else:
            try:
                print(
                    f"  Pool : reranking de {len(pool_chunks)} candidat(s)..."
                )
                pool_ranked = reranker.rerank(
                    question=question,
                    chunks=pool_chunks,
                    top_k=arguments.pool_top_k,
                )
                pool_available = {
                    str(chunk["chunk_id"]): chunk
                    for chunk in pool_chunks
                }
                pool_candidates = inject_priority_chunks(
                    pool_ranked,
                    priority_chunk_ids=draft_priority_ids(draft),
                    available_chunks=pool_available,
                )
                last_decision = request_decision(
                    question=question,
                    candidates=pool_candidates,
                    stage="pool complet reranké",
                    draft=draft,
                    override_ids=(),
                    arguments=arguments,
                )
                last_issues = decision_safety_issues(
                    last_decision,
                    question=question,
                    chunks_by_id=chunks_by_id,
                    confidence_threshold=(
                        arguments.confidence_threshold
                    ),
                )
            except Exception as error:
                last_error = f"{type(error).__name__}: {error}"
                last_issues = [
                    "L'adjudication sur le pool a échoué."
                ]

        needs_document_search = bool(last_issues) or bool(override_ids)

        if needs_document_search:
            last_stage = "document_attendu_complet"
            expected_documents = [
                str(value)
                for value in question.get("reference_documents", [])
            ]
            document_chunks = [
                chunk
                for document_id in expected_documents
                for chunk in chunks_by_document.get(document_id, [])
            ]

            try:
                print(
                    "  Pool insuffisant : reranking de tous les chunks "
                    f"du document attendu ({len(document_chunks)})..."
                )
                document_ranked = reranker.rerank(
                    question=question,
                    chunks=document_chunks,
                    top_k=arguments.document_top_k,
                )
                document_available = {
                    str(chunk["chunk_id"]): chunk
                    for chunk in document_chunks
                }
                priority_ids = (
                    override_ids
                    + coverage_hint_ids
                    + draft_priority_ids(draft)
                )
                document_candidates = inject_priority_chunks(
                    document_ranked,
                    priority_chunk_ids=priority_ids,
                    available_chunks=document_available,
                )
                last_decision = request_decision(
                    question=question,
                    candidates=document_candidates,
                    stage=(
                        "tous les chunks du document attendu rerankés"
                    ),
                    draft=draft,
                    override_ids=(
                        override_ids
                        + coverage_hint_ids
                    ),
                    arguments=arguments,
                )
                last_issues = decision_safety_issues(
                    last_decision,
                    question=question,
                    chunks_by_id=chunks_by_id,
                    confidence_threshold=(
                        arguments.confidence_threshold
                    ),
                    required_override_ids=(
                        override_ids
                        + coverage_hint_ids
                    ),
                )
                last_error = None
            except Exception as error:
                last_error = f"{type(error).__name__}: {error}"
                last_issues = [
                    "La recherche dans le document attendu a échoué."
                ]

        if not last_issues and last_decision is not None:
            automatic_by_id[question_id] = last_decision
            automatic_timestamps[question_id] = datetime.now(UTC).isoformat()
            print(
                "  -> accepté automatiquement "
                f"(confiance={last_decision['confidence']:.2f}, "
                f"chunks={last_decision['selected_chunk_ids']})"
            )
        else:
            manual_review_by_id[question_id] = make_manual_review_record(
                question=question,
                decision=last_decision,
                issues=last_issues,
                stage=last_stage,
                error=last_error,
            )
            print(
                "  -> revue manuelle : "
                + "; ".join(last_issues)
            )

        write_jsonl_atomic(
            (
                automatic_by_id[key]
                for key in sorted(automatic_by_id, key=natural_key)
            ),
            arguments.automatic_output,
        )
        write_jsonl_atomic(
            (
                manual_review_by_id[key]
                for key in sorted(manual_review_by_id, key=natural_key)
            ),
            arguments.manual_review_queue,
        )

    # Créer aussi les fichiers vides si aucune question n'a été ajoutée.
    write_jsonl_atomic(
        (
            automatic_by_id[key]
            for key in sorted(automatic_by_id, key=natural_key)
        ),
        arguments.automatic_output,
    )
    write_jsonl_atomic(
        (
            manual_review_by_id[key]
            for key in sorted(manual_review_by_id, key=natural_key)
        ),
        arguments.manual_review_queue,
    )

    merged_by_id = dict(human_by_id)
    assessor_id = f"auto:{arguments.model}"

    for question_id, decision in automatic_by_id.items():
        merged_by_id[question_id] = merged_gold_record(
            decision,
            question=questions_by_id[question_id],
            assessor_id=assessor_id,
            updated_at=automatic_timestamps[question_id],
        )

    merged_records = [
        merged_by_id[key]
        for key in sorted(merged_by_id, key=natural_key)
    ]
    write_jsonl_atomic(
        merged_records,
        arguments.merged_output,
    )

    if arguments.human_gold.read_bytes() != human_bytes_before:
        raise GoldValidationError(
            "Le fichier de gold humains a changé pendant l'exécution."
        )

    uncertain_ids = sorted(
        manual_review_by_id,
        key=natural_key,
    )
    print("\n=== Résultat de l'auto-vérification ===")
    print(f"Gold humains conservés       : {len(human_records)}")
    print(f"Gold ajoutés automatiquement : {len(automatic_by_id)}")
    print(f"Questions en revue manuelle  : {len(uncertain_ids)}")
    print(
        "Question_id incertaines     : "
        + (", ".join(uncertain_ids) if uncertain_ids else "aucune")
    )
    print(
        "Fichier automatique         : "
        f"{arguments.automatic_output.resolve()}"
    )
    print(
        "File de revue manuelle      : "
        f"{arguments.manual_review_queue.resolve()}"
    )
    print(
        "Gold fusionnés              : "
        f"{arguments.merged_output.resolve()}"
    )

    validate_final_dataset(
        merged_records,
        test_questions=test_questions,
        chunks_by_id=chunks_by_id,
        human_records=human_records,
    )
    print("Validations finales          : OK")


if __name__ == "__main__":
    main()
