"""Test rapide du modèle d'embeddings BGE-M3."""

from pathlib import Path

import numpy as np

from phosprocess.embeddings.embedder import (
    BGEEmbedder,
    load_embedding_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "embeddings.yaml"


def main() -> None:
    """Tester une recherche bilingue simple."""

    config = load_embedding_config(CONFIG_PATH)
    embedder = BGEEmbedder(config)

    query = (
        "Pourquoi une supersaturation excessive "
        "réduit-elle la filtration du gypse ?"
    )

    passages = [
        (
            "Excessive supersaturation produces many nuclei "
            "and small gypsum crystals, resulting in poor "
            "filtration performance."
        ),
        (
            "The administrative offices are located near "
            "the main entrance of the industrial site."
        ),
    ]

    query_vector = embedder.embed_query(query)
    passage_vectors = embedder.embed_documents(passages)

    scores = passage_vectors @ query_vector

    print("\n=== Test BGE-M3 ===")
    print(f"Modèle      : {embedder.model_name}")
    print(f"Device      : {embedder.device}")
    print(f"Query shape : {query_vector.shape}")
    print(f"Docs shape  : {passage_vectors.shape}")
    print(
        "Norme query : "
        f"{np.linalg.norm(query_vector):.6f}"
    )
    print(f"Score pertinent : {scores[0]:.6f}")
    print(f"Score hors sujet: {scores[1]:.6f}")

    if scores[0] <= scores[1]:
        raise AssertionError(
            "Le passage pertinent devrait avoir le meilleur score."
        )

    print("Résultat     : OK")


if __name__ == "__main__":
    main()