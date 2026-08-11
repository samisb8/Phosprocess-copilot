"""Run five non-benchmark business smoke questions through production RAG."""

from __future__ import annotations

import logging

from phosprocess.rag.pipeline import PhosProcessRAG, RAGError

BUSINESS_QUESTIONS = (
    (
        "Quel rôle le lavage du gâteau de gypse joue-t-il dans la "
        "récupération d'acide phosphorique ?"
    ),
    (
        "Comment la température de réaction influence-t-elle la "
        "cristallisation du sulfate de calcium dans une unité dihydrate ?"
    ),
    (
        "Quels signes opératoires peuvent indiquer un encrassement "
        "progressif du filtre à gypse ?"
    ),
    (
        "Pourquoi la teneur en solides de la bouillie doit-elle être "
        "maîtrisée avant la filtration ?"
    ),
    (
        "Quelles précautions opérationnelles sont utiles lors d'un "
        "changement de qualité du phosphate naturel ?"
    ),
)


def main() -> int:
    """Load models once and print concise smoke-test results."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    service = PhosProcessRAG()
    failures = 0

    for number, question in enumerate(BUSINESS_QUESTIONS, start=1):
        print(f"\n=== Question métier {number} ===", flush=True)
        print(question, flush=True)

        try:
            response = service.answer(question)
        except (RAGError, ValueError, TypeError) as error:
            failures += 1
            print(f"ERREUR: {error}", flush=True)
            continue

        print(response.answer, flush=True)
        print(
            "Chunks: "
            + ", ".join(
                source.chunk_id
                for source in response.sources
            ),
            flush=True,
        )

    print(
        f"\nBilan: {len(BUSINESS_QUESTIONS) - failures}/"
        f"{len(BUSINESS_QUESTIONS)} réponses valides.",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
