from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relevance": {
            "type": "integer",
            "minimum": 0,
            "maximum": 3,
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "reason": {"type": "string"},
        "evidence_quote": {"type": "string"},
    },
    "required": [
        "relevance",
        "confidence",
        "reason",
        "evidence_quote",
    ],
}


SYSTEM_PROMPT = """Tu es un évaluateur strict de passages pour un benchmark de recherche documentaire industrielle.

Tu dois évaluer UNIQUEMENT si le passage permet de répondre à la question en soutenant la réponse de référence.

Échelle obligatoire :
3 = réponse directe, précise et suffisamment complète.
2 = passage fortement pertinent, mais il manque une partie importante de la réponse.
1 = contexte utile seulement ; même thème, même procédé ou même variable, sans répondre réellement.
0 = non pertinent, mauvais procédé, mauvaise entité, mauvaise valeur numérique, contradiction, ou simple présence de mots-clés.

Règles :
- Pour une question numérique, une valeur concernant un autre équipement ou un autre procédé vaut 0 ou 1, jamais 2 ou 3.
- Ne donne pas 3 parce que le passage ressemble au thème : l'information demandée doit être présente.
- La réponse de référence aide à identifier les éléments nécessaires, mais tu juges le passage lui-même.
- Une citation vide est autorisée seulement pour la note 0.
- Réponds uniquement avec l'objet JSON demandé.
"""


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON invalide dans {path}, ligne {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Objet JSON attendu dans {path}, ligne {line_number}"
                )
            yield value


def recursive_values(value: Any, key_names: set[str]) -> list[Any]:
    found: list[Any] = []

    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in key_names:
                found.append(child)
            found.extend(recursive_values(child, key_names))
    elif isinstance(value, list):
        for child in value:
            found.extend(recursive_values(child, key_names))

    return found


def first_text(record: dict[str, Any], *keys: str) -> str:
    key_names = {key.lower() for key in keys}
    for value in recursive_values(record, key_names):
        if isinstance(value, (str, int, float)):
            text = str(value).strip()
            if text:
                return text
    return ""


def load_references(path: Path) -> dict[str, dict[str, str]]:
    references: dict[str, dict[str, str]] = {}
    for item in iter_jsonl(path):
        query_id = first_text(item, "query_id", "qid", "id")
        question = first_text(item, "question", "query")
        answer = first_text(item, "reference_answer", "answer", "gold_answer")

        if not query_id or not question or not answer:
            raise ValueError(f"Référence incomplète : {item}")

        references[query_id] = {
            "question": question,
            "reference_answer": answer,
        }

    if len(references) != 48:
        raise ValueError(
            f"48 réponses attendues, {len(references)} trouvées dans {path}"
        )

    return references


def pair_key(record: dict[str, Any]) -> tuple[str, str]:
    query_id = first_text(record, "query_id", "qid")
    chunk_id = first_text(record, "chunk_id", "passage_id", "document_id")
    return query_id, chunk_id


def call_ollama(
    *,
    base_url: str,
    model: str,
    prompt: str,
    num_ctx: int,
    retries: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,
        "format": SCHEMA,
        "options": {
            "temperature": 0,
            "num_ctx": num_ctx,
        },
        "keep_alive": "10m",
    }

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))

            content = result["message"]["content"]
            judgment = json.loads(content)

            relevance = int(judgment["relevance"])
            raw_confidence = judgment.get("confidence", 0.5)

            try:
                confidence = float(raw_confidence)
            except (TypeError, ValueError):
                confidence = 0.5

            if relevance not in {0, 1, 2, 3}:
                raise ValueError(f"Note invalide : {relevance}")

            # Qwen peut parfois produire une confiance hors de [0, 1].
            # La confiance est une m?tadonn?e secondaire : on la borne
            # au lieu d'interrompre toute l'annotation.
            if not 0 <= confidence <= 1:
                print(
                    f"[AVERTISSEMENT] Confiance invalide "
                    f"{confidence}, ramen?e dans [0, 1]."
                )
                confidence = max(0.0, min(1.0, confidence))

            return {
                "relevance": relevance,
                "confidence": confidence,
                "reason": str(judgment["reason"]).strip(),
                "evidence_quote": str(judgment["evidence_quote"]).strip(),
            }

        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            KeyError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)

    raise RuntimeError(
        f"Échec Ollama après {retries} tentatives : {last_error}"
    )


def parse_query_ids(raw_values: list[str] | None) -> set[str]:
    if not raw_values:
        return set()

    output: set[str] = set()
    for raw in raw_values:
        for value in raw.split(","):
            value = value.strip().upper()
            if value:
                output.add(value)
    return output


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    directory = root / "data" / "evaluation" / "retrieval" / "v0.1"

    parser = argparse.ArgumentParser(
        description=(
            "Produit des brouillons de jugements 0–3 avec Ollama. "
            "Le fichier judgments.jsonl n'est jamais modifié."
        )
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=directory / "reference_answers.jsonl",
    )
    parser.add_argument(
        "--pool",
        type=Path,
        default=directory / "annotation_pool.jsonl",
    )
    parser.add_argument(
        "--human-judgments",
        type=Path,
        default=directory / "judgments.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=directory / "llm_judgment_drafts.jsonl",
    )
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--split",
        choices=["dev", "test", "all"],
        default="dev",
    )
    parser.add_argument(
        "--query-id",
        action="append",
        help="Exemple : --query-id Q001,Q004",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = aucune limite.",
    )
    parser.add_argument(
        "--include-human",
        action="store_true",
        help=(
            "Évalue aussi les paires déjà jugées, utile pour calibrer "
            "le juge LLM sur Q001/Q004."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for required in (args.references, args.pool):
        if not required.exists():
            raise FileNotFoundError(f"Fichier introuvable : {required}")

    references = load_references(args.references)
    selected_query_ids = parse_query_ids(args.query_id)

    human_by_pair: dict[tuple[str, str], int] = {}
    if args.human_judgments.exists():
        for item in iter_jsonl(args.human_judgments):
            key = pair_key(item)
            relevance_text = first_text(
                item,
                "relevance",
                "label",
                "score",
                "grade",
            )
            if key[0] and key[1] and relevance_text:
                human_by_pair[key] = int(float(relevance_text))

    completed_drafts: set[tuple[str, str]] = set()
    if args.output.exists():
        for item in iter_jsonl(args.output):
            key = pair_key(item)
            if key[0] and key[1]:
                completed_drafts.add(key)

    candidates: list[dict[str, Any]] = []
    for item in iter_jsonl(args.pool):
        query_id, chunk_id = pair_key(item)
        split = first_text(item, "split").lower()

        if not query_id or not chunk_id:
            raise ValueError(
                "Impossible de lire query_id/chunk_id dans une ligne du pool. "
                f"Clés disponibles : {sorted(item.keys())}"
            )

        if args.split != "all" and split and split != args.split:
            continue
        if selected_query_ids and query_id not in selected_query_ids:
            continue
        if (query_id, chunk_id) in completed_drafts:
            continue
        if not args.include_human and (query_id, chunk_id) in human_by_pair:
            continue

        candidates.append(item)

    if args.limit > 0:
        candidates = candidates[: args.limit]

    print("=== Brouillons de pertinence avec Ollama ===")
    print(f"Modèle              : {args.model}")
    print(f"Split               : {args.split}")
    print(f"Paires sélectionnées: {len(candidates)}")
    print(f"Déjà en brouillon   : {len(completed_drafts)}")
    print(f"Jugements humains   : {len(human_by_pair)}")
    print(f"Sortie              : {args.output}")

    if not candidates:
        print("Aucune nouvelle paire à traiter.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("a", encoding="utf-8") as output:
        for index, item in enumerate(candidates, start=1):
            query_id, chunk_id = pair_key(item)
            reference = references[query_id]

            chunk_text = first_text(
                item,
                "chunk_text",
                "text",
                "content",
                "passage",
                "body",
            )
            if not chunk_text:
                raise ValueError(
                    f"Texte du chunk introuvable pour {query_id}::{chunk_id}. "
                    f"Clés disponibles : {sorted(item.keys())}"
                )

            document = first_text(
                item,
                "document",
                "document_name",
                "source",
                "filename",
                "file_name",
            )
            pages = first_text(item, "pages", "page", "page_range")
            section = first_text(item, "section", "heading", "title")
            category = first_text(item, "category")
            language = first_text(item, "language", "lang")

            prompt = f"""Question :
{reference["question"]}

Réponse de référence :
{reference["reference_answer"]}

Catégorie :
{category or "non renseignée"}

Langue :
{language or "non renseignée"}

Document :
{document or "non renseigné"}

Pages :
{pages or "non renseignées"}

Section :
{section or "non renseignée"}

Passage à évaluer :
--- DÉBUT DU PASSAGE ---
{chunk_text}
--- FIN DU PASSAGE ---

Attribue une note 0, 1, 2 ou 3 conformément aux règles.
Dans evidence_quote, recopie une courte preuve du passage si elle existe.
"""

            judgment = call_ollama(
                base_url=args.base_url,
                model=args.model,
                prompt=prompt,
                num_ctx=args.num_ctx,
                retries=args.retries,
            )

            record = {
                "query_id": query_id,
                "chunk_id": chunk_id,
                "relevance": judgment["relevance"],
                "confidence": judgment["confidence"],
                "reason": judgment["reason"],
                "evidence_quote": judgment["evidence_quote"],
                "model": args.model,
                "split": first_text(item, "split"),
                "human_relevance": human_by_pair.get((query_id, chunk_id)),
                "status": "draft",
            }

            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()

            comparison = ""
            if record["human_relevance"] is not None:
                comparison = (
                    f" | humain={record['human_relevance']} "
                    f"| {'OK' if record['human_relevance'] == record['relevance'] else 'DIFF'}"
                )

            print(
                f"[{index}/{len(candidates)}] "
                f"{query_id}::{chunk_id} "
                f"-> {record['relevance']} "
                f"(conf={record['confidence']:.2f})"
                f"{comparison}"
            )

    print("=== Terminé ===")
    print(f"Brouillons ajoutés : {len(candidates)}")
    print(f"Fichier            : {args.output}")


if __name__ == "__main__":
    main()
