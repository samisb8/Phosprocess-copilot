from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------- Generic helpers ----------

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON invalide: {path}, ligne {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Objet JSON attendu: {path}, ligne {line_no}")
            rows.append(row)
    return rows


def natural_key(value: str) -> list[Any]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", value)]


def write_by_query(path: Path, records: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for qid in sorted(records, key=natural_key):
            handle.write(json.dumps(records[qid], ensure_ascii=False) + "\n")


def deep_values(value: Any, names: set[str]) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in names:
                found.append(child)
            found.extend(deep_values(child, names))
    elif isinstance(value, list):
        for child in value:
            found.extend(deep_values(child, names))
    return found


def text_of(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    for value in deep_values(row, {key.lower() for key in keys}):
        if isinstance(value, (str, int, float, bool)) and str(value).strip():
            return str(value).strip()

    return ""


def pair_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        text_of(row, "query_id", "qid"),
        text_of(row, "chunk_id", "passage_id", "document_id"),
    )


def compact(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def parse_query_ids(values: list[str] | None) -> set[str]:
    result: set[str] = set()
    for raw in values or []:
        result.update(x.strip().upper() for x in raw.split(",") if x.strip())
    return result


# ---------- Project data ----------

def load_queries(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    for row in read_jsonl(path):
        qid = text_of(row, "query_id", "qid", "id")
        category = text_of(row, "category", "question_type") or "unknown"
        raw_answerable = row.get("answerable")

        if raw_answerable is None:
            answerable = category.lower() != "unanswerable"
        elif isinstance(raw_answerable, bool):
            answerable = raw_answerable
        else:
            answerable = str(raw_answerable).strip().lower() not in {
                "false", "0", "no", "non", "unanswerable"
            }

        result[qid] = {
            "query_id": qid,
            "split": (text_of(row, "split") or "unknown").lower(),
            "category": category,
            "language": text_of(row, "language", "lang") or "unknown",
            "answerable": answerable,
        }

    return result


def load_references(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}

    for row in read_jsonl(path):
        qid = text_of(row, "query_id", "qid", "id")
        question = text_of(row, "question", "query", "text")
        answer = text_of(row, "reference_answer", "gold_answer", "answer")

        if not qid or not question or not answer:
            raise ValueError(f"Référence incomplète: {row}")

        result[qid] = {"question": question, "reference_answer": answer}

    return result


def load_pool(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}

    for order, row in enumerate(read_jsonl(path), 1):
        qid, chunk_id = pair_key(row)
        if not qid or not chunk_id:
            raise ValueError(f"query_id/chunk_id introuvable dans le pool: {row.keys()}")

        item = dict(row)
        item["_query_id"] = qid
        item["_chunk_id"] = chunk_id
        item["_order"] = order
        grouped.setdefault(qid, []).append(item)

    return grouped


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {
        text_of(row, "query_id", "qid"): row
        for row in read_jsonl(path)
        if text_of(row, "query_id", "qid")
    }


def chunk_text(row: dict[str, Any]) -> str:
    return text_of(row, "chunk_text", "text", "content", "passage", "body")


def metadata(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        text_of(row, "document", "document_name", "source", "filename", "file_name")
        or "document inconnu",
        text_of(row, "pages", "page", "page_range") or "?",
        text_of(row, "section", "heading", "title") or "section inconnue",
    )


# ---------- Numeric guardrail ----------

def ratio_tokens(text: str) -> set[str]:
    pattern = re.compile(r"(?<![A-Za-z0-9])(\d+(?:[.,]\d+)?)\s*:\s*(\d+(?:[.,]\d+)?)(?![A-Za-z0-9])")
    return {
        f"{a.replace(',', '.')}:{b.replace(',', '.')}"
        for a, b in pattern.findall(text)
    }


def number_tokens(text: str) -> set[str]:
    ratios = ratio_tokens(text)
    if ratios:
        return ratios

    pattern = re.compile(r"(?<![A-Za-z0-9])\d+(?:[.,]\d+)?(?![A-Za-z0-9])")
    return {
        token.replace(",", ".")
        for token in pattern.findall(text)
        if token.replace(",", ".") not in {"0", "1", "2", "3"}
    }


def has_expected_number(candidate: dict[str, Any], answer: str) -> bool:
    expected_ratios = ratio_tokens(answer)
    if expected_ratios:
        return bool(expected_ratios & ratio_tokens(chunk_text(candidate)))

    expected = number_tokens(answer)
    return bool(expected & number_tokens(chunk_text(candidate))) if expected else False


# ---------- Ollama ----------

SHORTLIST_SCHEMA = {
    "type": "object",
    "properties": {
        "shortlist_ids": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
        },
        "reason": {"type": "string"},
    },
    "required": ["shortlist_ids", "reason"],
}

FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "gold_chunk_ids": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 3,
        },
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
        "evidence_quotes": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 3,
        },
    },
    "required": ["gold_chunk_ids", "confidence", "reason", "evidence_quotes"],
}

SHORTLIST_SYSTEM = """Tu présélectionnes les passages qui peuvent contenir la preuve exacte d'une réponse industrielle.
Ne te fie pas aux mots-clés seuls. Pour une question numérique, vérifie la bonne valeur et la bonne entité.
Pour une question causale, cherche réellement les liens cause -> mécanisme -> conséquence.
Retourne uniquement des chunk_id présents dans la liste et uniquement l'objet JSON demandé."""

FINAL_SYSTEM = """Tu sélectionnes les gold evidence d'une question industrielle.
Un gold evidence contient réellement la preuve permettant de répondre.
Choisis généralement un seul passage; deux ou trois seulement si des preuves complémentaires sont nécessaires.
Un simple contexte ne doit pas être choisi. Pour une question numérique, la valeur doit concerner la bonne entité.
N'utilise aucune connaissance externe. Retourne uniquement des chunk_id finalistes et uniquement l'objet JSON demandé."""


def ollama_json(
    *,
    base_url: str,
    model: str,
    system: str,
    prompt: str,
    schema: dict[str, Any],
    num_ctx: int,
    retries: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,
        "format": schema,
        "options": {"temperature": 0, "num_ctx": num_ctx},
        "keep_alive": "10m",
    }

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                result = json.loads(response.read().decode("utf-8"))
            parsed = json.loads(result["message"]["content"])
            if not isinstance(parsed, dict):
                raise ValueError("Objet JSON attendu.")
            return parsed
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

    raise RuntimeError(f"Ollama a échoué après {retries} tentatives: {last_error}")


def validated_ids(values: Any, allowed: set[str], maximum: int) -> list[str]:
    result: list[str] = []
    if not isinstance(values, list):
        return result

    for value in values:
        chunk_id = str(value).strip()
        if chunk_id in allowed and chunk_id not in result:
            result.append(chunk_id)
        if len(result) >= maximum:
            break

    return result


def shortlist_prompt(
    question: str,
    answer: str,
    category: str,
    candidates: list[dict[str, Any]],
    excerpt_chars: int,
) -> str:
    blocks: list[str] = []

    for row in candidates:
        document, pages, section = metadata(row)
        blocks.append(
            f"""chunk_id: {row["_chunk_id"]}
document: {document}
pages: {pages}
section: {section}
excerpt: {compact(chunk_text(row), excerpt_chars)}"""
        )

    return f"""Question:
{question}

Réponse de référence:
{answer}

Catégorie:
{category}

Candidats:
{'-' * 80}
{chr(10).join(blocks)}
{'-' * 80}

Choisis entre un et cinq chunk_id à examiner en détail."""


def final_prompt(
    question: str,
    answer: str,
    category: str,
    finalists: list[dict[str, Any]],
    chars: int,
) -> str:
    blocks: list[str] = []

    for row in finalists:
        document, pages, section = metadata(row)
        blocks.append(
            f"""chunk_id: {row["_chunk_id"]}
document: {document}
pages: {pages}
section: {section}
--- PASSAGE ---
{compact(chunk_text(row), chars)}
--- FIN ---"""
        )

    return f"""Question:
{question}

Réponse de référence:
{answer}

Catégorie:
{category}

Finalistes:
{'=' * 80}
{chr(10).join(blocks)}
{'=' * 80}

Choisis de zéro à trois gold evidence. Zéro signifie qu'une vérification humaine est nécessaire."""


# ---------- Propose ----------

def propose_one(
    query: dict[str, Any],
    reference: dict[str, str],
    candidates: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    qid = query["query_id"]
    now = datetime.now(timezone.utc).isoformat()

    if not query["answerable"]:
        return {
            "query_id": qid,
            "split": query["split"],
            "category": query["category"],
            "answerable": False,
            "gold_chunk_ids": [],
            "confidence": 1.0,
            "reason": "Question non répondable.",
            "evidence_quotes": [],
            "status": "draft",
            "model": args.model,
            "updated_at": now,
        }

    if not candidates:
        return {
            "query_id": qid,
            "split": query["split"],
            "category": query["category"],
            "answerable": True,
            "gold_chunk_ids": [],
            "confidence": 0.0,
            "reason": "Aucun candidat dans le pool.",
            "evidence_quotes": [],
            "status": "needs_review",
            "model": args.model,
            "updated_at": now,
        }

    # Exact numeric: put chunks containing expected numeric tokens first.
    if query["category"].lower() == "exact_numeric":
        numeric = [c for c in candidates if has_expected_number(c, reference["reference_answer"])]
        others = [c for c in candidates if c not in numeric]
        ordered = numeric + others
    else:
        numeric = []
        ordered = list(candidates)

    all_ids = {c["_chunk_id"] for c in candidates}

    first = ollama_json(
        base_url=args.base_url,
        model=args.model,
        system=SHORTLIST_SYSTEM,
        prompt=shortlist_prompt(
            reference["question"],
            reference["reference_answer"],
            query["category"],
            ordered,
            args.excerpt_chars,
        ),
        schema=SHORTLIST_SCHEMA,
        num_ctx=args.num_ctx,
        retries=args.retries,
    )

    shortlist = validated_ids(
        first.get("shortlist_ids"),
        all_ids,
        args.shortlist_size,
    )

    # Ensure numeric candidates remain visible in stage 2.
    for row in numeric:
        chunk_id = row["_chunk_id"]
        if chunk_id not in shortlist:
            shortlist.insert(0, chunk_id)
        shortlist = shortlist[: args.shortlist_size]

    if not shortlist:
        shortlist = [c["_chunk_id"] for c in ordered[: args.shortlist_size]]

    by_id = {c["_chunk_id"]: c for c in candidates}
    finalists = [by_id[chunk_id] for chunk_id in shortlist]
    finalist_ids = set(shortlist)

    second = ollama_json(
        base_url=args.base_url,
        model=args.model,
        system=FINAL_SYSTEM,
        prompt=final_prompt(
            reference["question"],
            reference["reference_answer"],
            query["category"],
            finalists,
            args.finalist_chars,
        ),
        schema=FINAL_SCHEMA,
        num_ctx=args.num_ctx,
        retries=args.retries,
    )

    selected = validated_ids(second.get("gold_chunk_ids"), finalist_ids, 3)

    try:
        confidence = float(second.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    # Final numeric guardrail.
    if (
        query["category"].lower() == "exact_numeric"
        and number_tokens(reference["reference_answer"])
        and selected
        and not any(has_expected_number(by_id[x], reference["reference_answer"]) for x in selected)
    ):
        selected = []
        confidence = min(confidence, 0.3)

    quotes = second.get("evidence_quotes", [])
    if not isinstance(quotes, list):
        quotes = []

    return {
        "query_id": qid,
        "split": query["split"],
        "category": query["category"],
        "answerable": True,
        "gold_chunk_ids": selected,
        "shortlist_ids": shortlist,
        "confidence": confidence,
        "reason": str(second.get("reason", "")).strip(),
        "evidence_quotes": [str(x).strip() for x in quotes[:3] if str(x).strip()],
        "status": "draft" if selected else "needs_review",
        "model": args.model,
        "updated_at": now,
    }


def run_propose(args: argparse.Namespace) -> None:
    queries = load_queries(args.queries)
    references = load_references(args.references)
    pool = load_pool(args.pool)
    verified = load_records(args.gold)
    drafts = load_records(args.drafts)
    selected_ids = parse_query_ids(args.query_id)

    todo = [
        query for query in queries.values()
        if (args.split == "all" or query["split"] == args.split)
        and (not selected_ids or query["query_id"] in selected_ids)
        and (args.redo or query["query_id"] not in verified)
        and (args.redo or query["query_id"] not in drafts)
    ]
    todo.sort(key=lambda row: natural_key(row["query_id"]))

    if args.limit:
        todo = todo[: args.limit]

    print("=== Proposition automatique de gold evidence ===")
    print(f"Modèle              : {args.model}")
    print(f"Questions à traiter : {len(todo)}")
    print(f"Gold déjà vérifiés  : {len(verified)}")
    print(f"Brouillons existants: {len(drafts)}")
    print(f"Sortie              : {args.drafts}")

    for index, query in enumerate(todo, 1):
        qid = query["query_id"]
        print(f"[{index}/{len(todo)}] {qid} | {query['category']}")

        try:
            proposal = propose_one(
                query,
                references[qid],
                pool.get(qid, []),
                args,
            )
        except Exception as exc:
            proposal = {
                "query_id": qid,
                "split": query["split"],
                "category": query["category"],
                "answerable": query["answerable"],
                "gold_chunk_ids": [],
                "confidence": 0.0,
                "reason": f"Erreur: {exc}",
                "evidence_quotes": [],
                "status": "error",
                "model": args.model,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

        drafts[qid] = proposal
        write_by_query(args.drafts, drafts)

        print(
            f"    proposition={proposal['gold_chunk_ids']} "
            f"| confiance={proposal['confidence']:.2f} "
            f"| statut={proposal['status']}"
        )

    print("=== Terminé ===")
    print(f"Total brouillons: {len(drafts)}")
    print(f"Fichier         : {args.drafts}")


# ---------- Review ----------

def show_full(row: dict[str, Any]) -> None:
    document, pages, section = metadata(row)
    print("\n" + "=" * 100)
    print(f"Chunk ID : {row['_chunk_id']}")
    print(f"Document : {document}")
    print(f"Pages    : {pages}")
    print(f"Section  : {section}")
    print("-" * 100)
    print(chunk_text(row))
    print("=" * 100 + "\n")


def show_all(candidates: list[dict[str, Any]]) -> None:
    for index, row in enumerate(candidates, 1):
        document, pages, section = metadata(row)
        print(f"[{index:02d}] {row['_chunk_id']}")
        print(f"     Document : {document}")
        print(f"     Pages    : {pages}")
        print(f"     Section  : {compact(section, 110)}")
        print(f"     Extrait  : {compact(chunk_text(row), 260)}")
        print()


def save_verified(
    gold: dict[str, dict[str, Any]],
    path: Path,
    query: dict[str, Any],
    chunk_ids: list[str],
    draft: dict[str, Any],
    assessor_id: str,
) -> None:
    gold[query["query_id"]] = {
        "query_id": query["query_id"],
        "split": query["split"],
        "category": query["category"],
        "answerable": query["answerable"],
        "gold_chunk_ids": chunk_ids,
        "status": "verified",
        "assessor_id": assessor_id,
        "proposal_model": draft.get("model"),
        "proposal_confidence": draft.get("confidence"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_by_query(path, gold)


def run_review(args: argparse.Namespace) -> None:
    queries = load_queries(args.queries)
    references = load_references(args.references)
    pool = load_pool(args.pool)
    drafts = load_records(args.drafts)
    gold = load_records(args.gold)
    selected_ids = parse_query_ids(args.query_id)

    pending = [
        drafts[qid]
        for qid in sorted(drafts, key=natural_key)
        if (args.redo or qid not in gold)
        and (args.split == "all" or drafts[qid].get("split") == args.split)
        and (not selected_ids or qid in selected_ids)
    ]

    if args.limit:
        pending = pending[: args.limit]

    print("=== Vérification rapide ===")
    print(f"Questions à vérifier: {len(pending)}")
    print(f"Gold déjà vérifiés : {len(gold)}")

    verified_count = 0

    for position, draft in enumerate(pending, 1):
        qid = draft["query_id"]
        query = queries[qid]
        reference = references[qid]
        candidates = pool.get(qid, [])
        by_id = {row["_chunk_id"]: row for row in candidates}
        proposed = [x for x in draft.get("gold_chunk_ids", []) if x in by_id]

        print("\n" + "=" * 100)
        print(f"Question {position}/{len(pending)} | {qid} | {query['category']}")
        print("=" * 100)
        print(f"Question    : {reference['question']}")
        print(f"Réponse     : {reference['reference_answer']}")
        print(f"Proposition : {proposed or 'AUCUNE'}")
        print(f"Confiance   : {float(draft.get('confidence', 0)):.2f}")
        print(f"Raison      : {draft.get('reason', '')}")

        if not query["answerable"]:
            save_verified(gold, args.gold, query, [], draft, args.assessor_id)
            verified_count += 1
            print("[AUTO-VALIDÉ] Non répondable.")
            continue

        for chunk_id in proposed:
            show_full(by_id[chunk_id])

        while True:
            raw = input(
                "a=accepter | m=tous | f <n>=complet | <indices>=corriger "
                "| s=passer | q=quitter : "
            ).strip().lower()

            if raw == "q":
                print(f"Session arrêtée. Nouveaux gold validés: {verified_count}")
                return

            if raw == "s":
                break

            if raw == "a":
                if not proposed:
                    print("Aucune proposition. Tape m pour choisir.")
                    continue
                save_verified(gold, args.gold, query, proposed, draft, args.assessor_id)
                verified_count += 1
                print(f"[VALIDÉ] {qid}")
                break

            if raw == "m":
                show_all(candidates)
                continue

            if raw.startswith("f "):
                try:
                    index = int(raw.split(maxsplit=1)[1])
                except (IndexError, ValueError):
                    print("Exemple valide: f 3")
                    continue

                if not 1 <= index <= len(candidates):
                    print("Indice hors plage.")
                    continue

                show_full(candidates[index - 1])
                continue

            try:
                indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
            except ValueError:
                print("Exemple valide: 3 ou 3,7")
                continue

            indices = list(dict.fromkeys(indices))
            if not 1 <= len(indices) <= 3:
                print("Choisis 1 à 3 indices.")
                continue
            if any(index < 1 or index > len(candidates) for index in indices):
                print("Indice hors plage.")
                continue

            chosen = [candidates[index - 1]["_chunk_id"] for index in indices]
            print("Nouvelle sélection:", chosen)

            if input("Confirmer [o/n]: ").strip().lower() not in {"o", "oui", "y", "yes"}:
                continue

            save_verified(gold, args.gold, query, chosen, draft, args.assessor_id)
            verified_count += 1
            print(f"[VALIDÉ] {qid}")
            break

    print("=== Vérification terminée ===")
    print(f"Nouveaux gold validés: {verified_count}")
    print(f"Total gold vérifiés : {len(gold)}")
    print(f"Fichier             : {args.gold}")


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    directory = root / "data" / "evaluation" / "retrieval" / "v0.1"

    parser = argparse.ArgumentParser(
        description="Propose et vérifie rapidement les gold evidence avec Ollama."
    )
    parser.add_argument("--mode", choices=["propose", "review"], default="propose")
    parser.add_argument("--queries", type=Path, default=directory / "queries.jsonl")
    parser.add_argument("--references", type=Path, default=directory / "reference_answers.jsonl")
    parser.add_argument("--pool", type=Path, default=directory / "annotation_pool.jsonl")
    parser.add_argument("--drafts", type=Path, default=directory / "gold_evidence_drafts.jsonl")
    parser.add_argument("--gold", type=Path, default=directory / "gold_evidence.jsonl")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--query-id", action="append")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--assessor-id", default="sami")

    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--shortlist-size", type=int, default=4)
    parser.add_argument("--excerpt-chars", type=int, default=650)
    parser.add_argument("--finalist-chars", type=int, default=1900)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for path in (args.queries, args.references, args.pool):
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable: {path}")

    if args.mode == "review" and not args.drafts.exists():
        raise FileNotFoundError(f"Brouillons introuvables: {args.drafts}")

    if args.mode == "propose":
        run_propose(args)
    else:
        run_review(args)


if __name__ == "__main__":
    main()
