from __future__ import annotations

import sys
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "scripts"),
)

from evaluate_retrieval_dev import load_dev_cases

from phosprocess.reranking.reranker import (
    build_reranking_passage,
    clean_passage_text,
    load_reranking_config,
)
from phosprocess.retrieval.bm25 import (
    load_bm25_config,
)
from phosprocess.retrieval.hybrid import (
    HybridRetriever,
)


RETRIEVAL_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "retrieval_v2.yaml"
)

RERANKING_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "reranking.yaml"
)

GOLD_ID = (
    "01_becker_phosphates_and_phosphoric_acid_"
    "000664_ec7b1f7db5f9"
)


def resolve_path(value: str) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def main() -> None:
    cases = {
        case["query_id"]: case
        for case in load_dev_cases()
    }

    query = cases["Q015"]["question"]

    bm25_config = load_bm25_config(
        RETRIEVAL_CONFIG
    )

    retriever = HybridRetriever(
        dense_index_directory=(
            PROJECT_ROOT
            / "data"
            / "indexes"
            / "dense"
            / "bge_m3"
        ),
        bm25_index_directory=resolve_path(
            bm25_config.output_directory
        ),
        embedding_config_path=(
            PROJECT_ROOT
            / "configs"
            / "embeddings.yaml"
        ),
        retrieval_config_path=RETRIEVAL_CONFIG,
    )

    response = retriever.search(
        query,
        top_k=20,
        dense_candidate_k=20,
        bm25_candidate_k=20,
        use_query_expansion=True,
    )

    candidate = next(
        (
            result
            for result in response.results
            if result.chunk.chunk_id == GOLD_ID
        ),
        None,
    )

    if candidate is None:
        raise RuntimeError(
            "Gold Q015 absent des candidats."
        )

    config = load_reranking_config(
        RERANKING_CONFIG
    )

    raw_text = candidate.chunk.text

    cleaned_text = clean_passage_text(
        raw_text
    )

    reranking_passage = build_reranking_passage(
        candidate.chunk,
        config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name
    )

    passage_tokens = tokenizer.encode(
        reranking_passage,
        add_special_tokens=False,
    )

    pair_encoding = tokenizer(
        query,
        reranking_passage,
        truncation=True,
        max_length=config.max_length,
    )

    truncated_pair = tokenizer.decode(
        pair_encoding["input_ids"],
        skip_special_tokens=True,
    )

    important_phrases = [
        "co-crystallized",
        "lattice losses",
        "cake impregnation",
        "unattacked P2O5",
        "mechanical losses",
        "sludge removal",
    ]

    print("\n=== Q015 ===")
    print(query)

    print("\n=== Tailles ===")
    print("Texte brut, caractères       :", len(raw_text))
    print("Texte nettoyé, caractères    :", len(cleaned_text))
    print("Passage final, caractères     :", len(reranking_passage))
    print("Passage final, tokens         :", len(passage_tokens))
    print("Paire après troncature, tokens:", len(pair_encoding["input_ids"]))

    print("\n=== Marqueurs après nettoyage ===")
    markers = [
        "formula-not-decoded",
        "Start of picture text",
        "End of picture text",
    ]

    for marker in markers:
        print(
            f"{marker:<25}:",
            marker.casefold()
            in reranking_passage.casefold(),
        )

    print("\n=== Présence après troncature ===")

    for phrase in important_phrases:
        print(
            f"{phrase:<25}:",
            phrase.casefold()
            in truncated_pair.casefold(),
        )

    normalized = reranking_passage.casefold()

    position = normalized.find(
        "losses include"
    )

    print("\nPosition de 'Losses include' :", position)

    if position >= 0:
        start = max(0, position - 250)
        end = min(
            len(reranking_passage),
            position + 900,
        )

        print("\n=== Zone pertinente transmise ===")
        print(
            reranking_passage[start:end]
        )


if __name__ == "__main__":
    main()
