from __future__ import annotations

import argparse
import json
import re
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


SYSTEM_PROMPT = """Tu es un ?valuateur strict de passages pour un benchmark de recherche documentaire industrielle.

Ta t?che consiste uniquement ? d?terminer si le PASSAGE soutient la r?ponse ? la QUESTION.
N'utilise jamais tes connaissances externes et ne compl?te jamais une information absente.

Avant de noter, v?rifie mentalement quatre ?l?ments :
1. L'entit? demand?e est-elle la bonne ?
2. La propri?t? ou le m?canisme demand? est-il r?ellement trait? ?
3. La valeur ou la cha?ne causale appara?t-elle dans le passage ?
4. Le passage fournit-il toute la r?ponse ou seulement une partie ?

?chelle obligatoire :

3 = R?PONSE DIRECTE ET COMPL?TE
Le passage contient explicitement la bonne information pour la bonne entit?.
Pour une question causale, il soutient la cha?ne causale essentielle demand?e.

2 = FORTEMENT PERTINENT MAIS PARTIEL
Le passage soutient une partie importante de la r?ponse, mais il manque une ?tape,
une cons?quence ou un ?l?ment n?cessaire.
Cette note doit r?ellement ?tre utilis?e quand une moiti? importante de la r?ponse
est pr?sente.

1 = CONTEXTE UTILE UNIQUEMENT
Le passage parle du m?me proc?d?, ?quipement, ph?nom?ne ou variable,
mais ne r?pond pas r?ellement ? la question.
Une valeur concernant un ?quipement voisin ou un proc?d? diff?rent vaut au maximum 1.

0 = NON PERTINENT
Le passage ne permet pas de r?pondre, concerne une autre question,
un autre proc?d? sans lien utile, ou contient seulement des mots-cl?s superficiels.

R?GLES POUR exact_numeric :
- 3 seulement si la valeur demand?e est explicitement pr?sente et associ?e ? la bonne entit?.
- 2 seulement si la valeur exacte peut ?tre calcul?e directement ? partir du passage,
  sans connaissance externe.
- 1 si le passage parle de la bonne variable mais ne donne pas la valeur,
  ou donne une valeur pour un syst?me diff?rent.
- 0 si le passage ne traite pas r?ellement de la variable demand?e.
- Ne d?duis jamais arbitrairement une valeur ? partir de nombres sans relation explicite.

R?GLES POUR causal_mechanism :
- 3 si le passage donne la cause, le m?canisme principal et la cons?quence demand?e.
- 2 s'il donne au moins une liaison causale majeure ou la cons?quence technique directe.
- 1 s'il parle seulement du ph?nom?ne g?n?ral, sans expliquer le m?canisme demand?.
- 0 s'il explique une autre cause ou un autre probl?me.

EXEMPLES DE CALIBRATION :

Exemple num?rique A :
Question : rapport de recirculation Jacobs ?
R?ponse attendue : 40:1
Passage : ? an overall recirculation ratio of about 40:1 ?
Note : 3

Exemple num?rique B :
Question : rapport de recirculation Jacobs ?
Passage : ? the single-tank agitator induces a recirculation ratio of 330:1 ?
Note : 1
Justification : m?me variable, mais mauvaise technologie et mauvaise valeur.

Exemple num?rique C :
Question : rapport de recirculation Jacobs ?
Passage : bilan thermique sans rapport de recirculation
Note : 0

Exemple causal A :
Question : pourquoi la forte supersaturation donne-t-elle de petits cristaux difficiles ? filtrer ?
Passage : au-dessus de la ligne de supersaturation, une nucl?ation spontan?e forme
des millions de petits cristaux difficiles ? filtrer.
Note : 3

Exemple causal B :
Passage : les cristaux inf?rieurs ? 40 microm?tres r?duisent fortement la filtration.
Note : 2
Justification : la cons?quence sur la filtration est donn?e, mais pas la nucl?ation
provoqu?e par la supersaturation.

Exemple causal C :
Passage : la supersaturation influence la vitesse de croissance des cristaux.
Note : 1
Justification : contexte utile, mais la cha?ne demand?e n'est pas expliqu?e.

Exemple causal D :
Passage : description d'un autre proc?d? sans lien avec supersaturation et filtration.
Note : 0

La confiance doit ?tre un nombre entre 0 et 1 :
- 0.90 ? 1.00 : preuve explicite et non ambigu? ;
- 0.70 ? 0.89 : d?cision solide ;
- 0.50 ? 0.69 : passage ambigu ;
- moins de 0.50 : grande incertitude.

Dans evidence_quote, copie une courte preuve exacte tir?e du passage.
Pour une note 0, evidence_quote peut ?tre vide.

R?ponds uniquement avec l'objet JSON demand?.
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



def normalize_numeric_token(value: str) -> str:
    return re.sub(r"\s+", "", value.replace(",", ".")).lower()


def extract_reference_numeric_targets(
    reference_answer: str,
) -> tuple[str, set[str]]:
    ratio_pattern = re.compile(
        r"(?<![A-Za-z0-9])"
        r"(\d+(?:[.,]\d+)?)\s*:\s*(\d+(?:[.,]\d+)?)"
        r"(?![A-Za-z0-9])"
    )

    ratios = {
        f"{normalize_numeric_token(left)}:"
        f"{normalize_numeric_token(right)}"
        for left, right in ratio_pattern.findall(reference_answer)
    }

    if ratios:
        return "ratio", ratios

    number_pattern = re.compile(
        r"(?<![A-Za-z0-9])"
        r"\d+(?:[.,]\d+)?"
        r"(?![A-Za-z0-9])"
    )

    numbers = {
        normalize_numeric_token(value)
        for value in number_pattern.findall(reference_answer)
    }

    return "number", numbers


def passage_contains_required_numbers(
    reference_answer: str,
    chunk_text: str,
) -> bool:
    mode, targets = extract_reference_numeric_targets(reference_answer)

    if not targets:
        # Par exemple : ? five operating and one spare ?.
        # Dans ce cas, on laisse le juge s?mantique d?cider.
        return True

    if mode == "ratio":
        ratio_pattern = re.compile(
            r"(?<![A-Za-z0-9])"
            r"(\d+(?:[.,]\d+)?)\s*:\s*(\d+(?:[.,]\d+)?)"
            r"(?![A-Za-z0-9])"
        )

        passage_ratios = {
            f"{normalize_numeric_token(left)}:"
            f"{normalize_numeric_token(right)}"
            for left, right in ratio_pattern.findall(chunk_text)
        }

        return targets.issubset(passage_ratios)

    number_pattern = re.compile(
        r"(?<![A-Za-z0-9])"
        r"\d+(?:[.,]\d+)?"
        r"(?![A-Za-z0-9])"
    )

    passage_numbers = {
        normalize_numeric_token(value)
        for value in number_pattern.findall(chunk_text)
    }

    return targets.issubset(passage_numbers)


def apply_rule_based_guardrails(
    *,
    judgment: dict[str, Any],
    category: str,
    reference_answer: str,
    chunk_text: str,
) -> dict[str, Any]:
    result = dict(judgment)

    relevance = int(result["relevance"])
    confidence = float(result["confidence"])
    reason = str(result["reason"]).strip()

    if category.strip().lower() == "exact_numeric":
        contains_targets = passage_contains_required_numbers(
            reference_answer=reference_answer,
            chunk_text=chunk_text,
        )

        if not contains_targets and relevance >= 2:
            original_relevance = relevance
            relevance = 1
            confidence = min(confidence, 0.60)

            reason = (
                f"[Protection num?rique : la valeur attendue n'appara?t pas "
                f"explicitement dans le passage ; note {original_relevance} "
                f"ramen?e ? 1.] {reason}"
            )

    evidence_quote = str(result.get("evidence_quote", "")).strip()

    if evidence_quote:
        normalized_quote = re.sub(
            r"\s+",
            " ",
            evidence_quote,
        ).strip().lower()

        normalized_chunk = re.sub(
            r"\s+",
            " ",
            chunk_text,
        ).strip().lower()

        if normalized_quote not in normalized_chunk:
            confidence = min(confidence, 0.50)
            reason = (
                "[La citation fournie n'a pas ?t? retrouv?e exactement "
                "dans le passage.] "
                + reason
            )

    result["relevance"] = relevance
    result["confidence"] = max(0.0, min(1.0, confidence))
    result["reason"] = reason

    return result


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

            judgment = apply_rule_based_guardrails(
                judgment=judgment,
                category=category,
                reference_answer=reference["reference_answer"],
                chunk_text=chunk_text,
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
