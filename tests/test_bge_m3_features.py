"""BGE-M3 sparse and ColBERT feature API invariants."""

from __future__ import annotations

import numpy as np

from phosprocess.embeddings.embedder import BGEEmbedder, EmbeddingConfig


class _FeatureModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], dict[str, object]]] = []

    def encode_queries(self, texts: list[str], **kwargs: object) -> dict[str, object]:
        self.calls.append(("queries", texts, kwargs))
        return self._output(texts, kwargs)

    def encode_corpus(self, texts: list[str], **kwargs: object) -> dict[str, object]:
        self.calls.append(("corpus", texts, kwargs))
        return self._output(texts, kwargs)

    @staticmethod
    def _output(texts: list[str], kwargs: dict[str, object]) -> dict[str, object]:
        output: dict[str, object] = {}
        if kwargs.get("return_sparse"):
            output["lexical_weights"] = [
                {"7": 1.5, "9": 0.0}
                for _text in texts
            ]
        if kwargs.get("return_colbert_vecs"):
            output["colbert_vecs"] = [
                np.ones((2, 3), dtype=np.float32)
                for _text in texts
            ]
        return output

    @staticmethod
    def colbert_score(query: np.ndarray, passage: np.ndarray) -> float:
        return float(query.shape[0] + passage.shape[0])


def _embedder() -> BGEEmbedder:
    embedder = object.__new__(BGEEmbedder)
    embedder.config = EmbeddingConfig(
        model_name="BAAI/bge-m3",
        embedding_dimension=1024,
        device="cpu",
        use_fp16=False,
        batch_size=4,
    )
    embedder._device = "cpu"
    embedder._model = _FeatureModel()
    return embedder


def test_sparse_features_use_query_and_corpus_apis() -> None:
    embedder = _embedder()

    query = embedder.embed_sparse_query("circulation pump")
    documents = embedder.embed_sparse_documents(["pump", "falling film"])

    assert query == {7: 1.5}
    assert documents == [{7: 1.5}, {7: 1.5}]
    assert [call[0] for call in embedder._model.calls] == ["queries", "corpus"]
    assert embedder._model.calls[0][2]["return_dense"] is False
    assert embedder._model.calls[0][2]["return_sparse"] is True


def test_colbert_features_and_score_use_official_model_methods() -> None:
    embedder = _embedder()

    query = embedder.embed_colbert_query("circulation pump")
    documents = embedder.embed_colbert_documents(["pump", "heater"])
    score = embedder.colbert_score(query, documents[0])

    assert query.shape == (2, 3)
    assert [document.shape for document in documents] == [(2, 3), (2, 3)]
    assert score == 4.0
    assert embedder._model.calls[0][2]["return_colbert_vecs"] is True
