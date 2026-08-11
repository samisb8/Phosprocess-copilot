from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


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
                raise ValueError(f"Objet JSON attendu dans {path}, ligne {line_number}")
            yield value


def first(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def recursive_values(value: Any, wanted: set[str]) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in wanted:
                found.append(child)
            found.extend(recursive_values(child, wanted))
    elif isinstance(value, list):
        for child in value:
            found.extend(recursive_values(child, wanted))
    return found


def first_any(record: dict[str, Any], *keys: str) -> str:
    direct = first(record, *keys)
    if direct:
        return direct
    for value in recursive_values(record, {key.lower() for key in keys}):
        if isinstance(value, (str, int, float, bool)) and str(value).strip():
            return str(value).strip()
    return ""


def natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", value)]


def load_queries(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for item in iter_jsonl(path):
        query_id = first(item, "query_id", "qid", "id")
        if not query_id:
            raise ValueError(f"query_id absent dans {path}: {item}")
        category = first(item, "category", "question_type") or "unknown"
        raw_answerable = item.get("answerable")
        if raw_answerable is None:
            answerable = category.lower() != "unanswerable"
        elif isinstance(raw_answerable, bool):
            answerable = raw_answerable
        else:
            answerable = str(raw_answerable).strip().lower() not in {
                "false", "0", "no", "non", "unanswerable"
            }
        output[query_id] = {
            "query_id": query_id,
            "split": (first(item, "split") or "unknown").lower(),
            "category": category,
            "language": first(item, "language", "lang") or "unknown",
            "answerable": answerable,
        }
    return output


def load_references(path: Path) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for item in iter_jsonl(path):
        query_id = first(item, "query_id", "qid", "id")
        question = first(item, "question", "query", "text")
        answer = first(item, "reference_answer", "gold_answer", "answer")
        if not query_id or not question or not answer:
            raise ValueError(f"Référence incomplète dans {path}: {item}")
        output[query_id] = {
            "question": question,
            "reference_answer": answer,
        }
    return output


def pair_key(record: dict[str, Any]) -> tuple[str, str]:
    return (
        first_any(record, "query_id", "qid"),
        first_any(record, "chunk_id", "passage_id", "document_id"),
    )


def load_judgments(path: Path) -> dict[tuple[str, str], int]:
    if not path.exists():
        return {}
    output: dict[tuple[str, str], int] = {}
    for item in iter_jsonl(path):
        key = pair_key(item)
        label = first_any(item, "relevance", "label", "score", "grade")
        if key[0] and key[1] and label:
            output[key] = int(float(label))
    return output


def load_pool(
    path: Path,
    judgments: dict[tuple[str, str], int],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for order, item in enumerate(iter_jsonl(path), start=1):
        query_id, chunk_id = pair_key(item)
        if not query_id or not chunk_id:
            raise ValueError(
                "Impossible de lire query_id/chunk_id dans le pool. "
                f"Clés: {sorted(item.keys())}"
            )
        candidate = dict(item)
        candidate["_query_id"] = query_id
        candidate["_chunk_id"] = chunk_id
        candidate["_order"] = order
        candidate["_human"] = judgments.get((query_id, chunk_id))
        grouped.setdefault(query_id, []).append(candidate)

    for candidates in grouped.values():
        candidates.sort(
            key=lambda item: (
                -(item["_human"] if item["_human"] is not None else -1),
                item["_order"],
            )
        )
    return grouped


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {
        first(item, "query_id", "qid"): item
        for item in iter_jsonl(path)
        if first(item, "query_id", "qid")
    }


def write_records(path: Path, records: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for query_id in sorted(records, key=natural_key):
            handle.write(json.dumps(records[query_id], ensure_ascii=False) + "\n")


def candidate_text(item: dict[str, Any]) -> str:
    return first_any(item, "chunk_text", "text", "content", "passage", "body")


def metadata(item: dict[str, Any]) -> tuple[str, str, str]:
    document = first_any(
        item, "document", "document_name", "source", "filename", "file_name"
    ) or "document inconnu"
    pages = first_any(item, "pages", "page", "page_range") or "?"
    section = first_any(item, "section", "heading", "title") or "section inconnue"
    return document, pages, section


def compact(text: str, width: int = 260) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


def show_candidates(candidates: list[dict[str, Any]], count: int) -> None:
    print("-" * 100)
    for index, candidate in enumerate(candidates[:count], start=1):
        document, pages, section = metadata(candidate)
        human = candidate["_human"]
        marker = f" | ancien jugement={human}" if human is not None else ""
        print(f"[{index:02d}] {candidate['_chunk_id']}{marker}")
        print(f"     Document : {document}")
        print(f"     Pages    : {pages}")
        print(f"     Section  : {compact(section, 120)}")
        print(f"     Extrait  : {compact(candidate_text(candidate))}")
        print()
    if count < len(candidates):
        print(f"{len(candidates) - count} candidat(s) masqué(s). Tape `m` pour tout afficher.")


def show_full(candidates: list[dict[str, Any]], index: int) -> None:
    candidate = candidates[index - 1]
    document, pages, section = metadata(candidate)
    print("\n" + "=" * 100)
    print(f"Chunk ID : {candidate['_chunk_id']}")
    print(f"Document : {document}")
    print(f"Pages    : {pages}")
    print(f"Section  : {section}")
    print("-" * 100)
    print(candidate_text(candidate))
    print("=" * 100 + "\n")


def suggestions(candidates: list[dict[str, Any]]) -> list[int]:
    grade3 = [i for i, item in enumerate(candidates, 1) if item["_human"] == 3]
    if grade3:
        return grade3[:3]
    return [i for i, item in enumerate(candidates, 1) if item["_human"] == 2][:3]


def parse_query_ids(values: list[str] | None) -> set[str]:
    if not values:
        return set()
    output: set[str] = set()
    for raw in values:
        output.update(value.strip().upper() for value in raw.split(",") if value.strip())
    return output


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    directory = root / "data" / "evaluation" / "retrieval" / "v0.1"
    parser = argparse.ArgumentParser(
        description="Sélection interactive de 1 à 3 chunks gold par question."
    )
    parser.add_argument("--queries", type=Path, default=directory / "queries.jsonl")
    parser.add_argument("--references", type=Path, default=directory / "reference_answers.jsonl")
    parser.add_argument("--pool", type=Path, default=directory / "annotation_pool.jsonl")
    parser.add_argument("--judgments", type=Path, default=directory / "judgments.jsonl")
    parser.add_argument("--output", type=Path, default=directory / "gold_evidence.jsonl")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--query-id", action="append")
    parser.add_argument("--assessor-id", default="sami")
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--redo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for required in (args.queries, args.references, args.pool):
        if not required.exists():
            raise FileNotFoundError(f"Fichier introuvable : {required}")

    queries = load_queries(args.queries)
    references = load_references(args.references)
    judgments = load_judgments(args.judgments)
    pool = load_pool(args.pool, judgments)
    records = load_existing(args.output)
    wanted = parse_query_ids(args.query_id)

    selected = [
        query for query in queries.values()
        if (args.split == "all" or query["split"] == args.split)
        and (not wanted or query["query_id"] in wanted)
        and (args.redo or query["query_id"] not in records)
    ]
    selected.sort(key=lambda item: natural_key(item["query_id"]))

    print("=== Construction des gold evidence ===")
    print(f"Split                : {args.split}")
    print(f"Questions à traiter  : {len(selected)}")
    print(f"Déjà sauvegardées    : {len(records)}")
    print(f"Jugements disponibles: {len(judgments)}")
    print(f"Sortie               : {args.output}")

    if not selected:
        print("Aucune question restante.")
        return

    newly_saved = 0

    for position, query in enumerate(selected, start=1):
        query_id = query["query_id"]
        reference = references.get(query_id)
        if reference is None:
            raise ValueError(f"Réponse de référence absente pour {query_id}")

        print("\n" + "=" * 100)
        print(f"Question {position}/{len(selected)} | {query_id} | {query['category']} | {query['split']}")
        print("=" * 100)
        print(f"Question : {reference['question']}")
        print(f"Réponse  : {reference['reference_answer']}")

        if not query["answerable"]:
            records[query_id] = {
                "query_id": query_id,
                "split": query["split"],
                "category": query["category"],
                "answerable": False,
                "gold_chunk_ids": [],
                "status": "verified",
                "assessor_id": args.assessor_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            write_records(args.output, records)
            newly_saved += 1
            print("[AUTO] Question non répondable sauvegardée avec gold_chunk_ids=[].")
            continue

        candidates = pool.get(query_id, [])
        if not candidates:
            print("[ATTENTION] Aucun candidat dans le pool. Question passée.")
            continue

        display_count = min(max(1, args.max_candidates), len(candidates))

        while True:
            show_candidates(candidates, display_count)
            suggested = suggestions(candidates)
            if suggested:
                print("Suggestion issue des anciens jugements : " + ",".join(map(str, suggested)))

            raw = input(
                "\nSélection [ex: 2 ou 2,5] | a=suggestion | f 3=texte complet "
                "| m=tout | s=passer | q=quitter | ?=aide : "
            ).strip().lower()

            if raw == "q":
                print("\n=== Session interrompue proprement ===")
                print(f"Nouvelles questions sauvegardées : {newly_saved}")
                print(f"Total gold evidence              : {len(records)}")
                print(f"Fichier                          : {args.output}")
                return
            if raw == "s":
                print("[PASSÉ] Aucun changement sauvegardé.")
                break
            if raw == "?":
                print(
                    "\nSélectionne seulement les chunks contenant réellement la preuve.\n"
                    "Un seul chunk suffit généralement. Deux ou trois seulement si la réponse "
                    "nécessite plusieurs preuves ou si plusieurs formulations sont valides.\n"
                )
                continue
            if raw == "m":
                display_count = len(candidates)
                continue
            if raw.startswith("f "):
                try:
                    index = int(raw.split(maxsplit=1)[1])
                except (IndexError, ValueError):
                    print("Commande invalide. Exemple : f 3")
                    continue
                if not 1 <= index <= len(candidates):
                    print("Indice hors plage.")
                    continue
                show_full(candidates, index)
                continue

            if raw == "a":
                indices = suggested
                if not indices:
                    print("Aucune suggestion humaine disponible.")
                    continue
            else:
                try:
                    indices = [int(value.strip()) for value in raw.split(",") if value.strip()]
                except ValueError:
                    print("Entrée invalide. Exemple : 2 ou 2,5")
                    continue

            indices = list(dict.fromkeys(indices))
            if not 1 <= len(indices) <= 3:
                print("Sélectionne entre 1 et 3 candidats.")
                continue
            if any(index < 1 or index > display_count for index in indices):
                print("Indice hors de la liste affichée. Tape `m` pour tout afficher.")
                continue

            chosen = [candidates[index - 1] for index in indices]
            print("\nPreuves sélectionnées :")
            for candidate in chosen:
                print(f"- {candidate['_chunk_id']}")

            confirmation = input("Confirmer [o/n] : ").strip().lower()
            if confirmation not in {"o", "oui", "y", "yes"}:
                print("Sélection annulée.")
                continue

            records[query_id] = {
                "query_id": query_id,
                "split": query["split"],
                "category": query["category"],
                "answerable": True,
                "gold_chunk_ids": [candidate["_chunk_id"] for candidate in chosen],
                "status": "verified",
                "assessor_id": args.assessor_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            write_records(args.output, records)
            newly_saved += 1
            print(f"[SAUVEGARDÉ] {query_id}")
            break

    print("\n=== Construction terminée ===")
    print(f"Nouvelles questions sauvegardées : {newly_saved}")
    print(f"Total gold evidence              : {len(records)}")
    print(f"Fichier                          : {args.output}")


if __name__ == "__main__":
    main()
