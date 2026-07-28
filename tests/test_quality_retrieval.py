"""Quality retrieval score-adjustment invariants."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

from phosprocess.ingestion.chunk_serialization import (
    TechnicalChildChunk,
    TechnicalChunkType,
    write_jsonl,
)
from phosprocess.preprocessing.chunk_schemas import DocumentChunk
from phosprocess.rag.quality_retrieval import QualityRetrievalEngine
from phosprocess.reranking.reranker import (
    RerankedSearchResult,
    RerankingResponse,
)
from phosprocess.retrieval.domain_router import DomainRoutingDecision


def _chunk(chunk_id: str, document_id: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source_file=f"{document_id}.pdf",
        chunk_index=0,
        source_pages=[1],
        page_start=1,
        page_end=1,
        text="Technical passage.",
        embedding_text="Document\nTechnical passage.",
        body_token_count=3,
        token_count=3,
        chunk_type="narrative",
    )


def _result(rank: int, score: float, chunk: DocumentChunk) -> RerankedSearchResult:
    return RerankedSearchResult(
        rank=rank,
        reranker_score=score,
        original_hybrid_rank=rank,
        original_rrf_score=0.01,
        matched_retrievers=("dense",),
        dense_rank=rank,
        dense_score=score,
        bm25_rank=None,
        bm25_score=None,
        chunk=chunk,
    )


def test_quality_boost_is_soft_and_nonpreferred_can_still_win() -> None:
    preferred = _chunk("preferred", "preferred_book")
    other = _chunk("other", "other_book")
    response = RerankingResponse(
        query="query",
        model_name="model",
        device="cpu",
        candidates_received=2,
        top_k_requested=2,
        reranking_duration_ms=1.0,
        results=[
            _result(1, 0.55, other),
            _result(2, 0.40, preferred),
        ],
    )
    routing = DomainRoutingDecision(
        detected_domains=(),
        confidence=0.8,
        preferred_documents=("preferred_book",),
        soft_boosts={"preferred_book": 0.06},
        explanation="test",
        hard_filter=None,
        source_mode="auto",
    )

    adjusted, boosts = QualityRetrievalEngine._adjust_reranking(
        response,
        routing=routing,
    )

    assert adjusted.results[0].chunk.chunk_id == "other"
    assert boosts["preferred"] == 0.06


def test_question_type_favors_matching_chunk_type() -> None:
    definition = _chunk("definition", "book").model_copy(
        update={"chunk_type": "definition"}
    )
    narrative = _chunk("narrative", "book")
    response = RerankingResponse(
        query="what is",
        model_name="model",
        device="cpu",
        candidates_received=2,
        top_k_requested=2,
        reranking_duration_ms=1.0,
        results=[
            _result(1, 0.51, narrative),
            _result(2, 0.50, definition),
        ],
    )
    routing = DomainRoutingDecision(
        detected_domains=(),
        confidence=0.4,
        preferred_documents=(),
        soft_boosts={},
        explanation="test",
        hard_filter=None,
        source_mode="auto",
    )

    adjusted, _boosts = QualityRetrievalEngine._adjust_reranking(
        response,
        routing=routing,
        question_type="definition",
    )

    assert adjusted.results[0].chunk.chunk_id == "definition"


class _Recorder:
    def __init__(self, results: list[object]) -> None:
        self.results = results

    def search(self, _query: str, **_kwargs: object) -> object:
        return SimpleNamespace(results=self.results, search_duration_ms=1.0)


class _Reranker:
    def rerank(
        self,
        query: str,
        candidates: list[object],
        *,
        top_k: int,
    ) -> RerankingResponse:
        return RerankingResponse(
            query=query,
            model_name="fake",
            device="cpu",
            candidates_received=len(candidates),
            top_k_requested=top_k,
            reranking_duration_ms=2.0,
            results=[
                _result(rank, 1.0 - rank / 100, candidate.chunk)
                for rank, candidate in enumerate(candidates, start=1)
            ],
        )


def test_quality_retrieval_returns_five_bounded_bundles(
    tmp_path: Path,
) -> None:
    technical: list[TechnicalChildChunk] = []
    runtime: list[DocumentChunk] = []

    for index in range(20):
        text = f"Evaporator definition and heat transfer passage {index}."
        technical.append(
            TechnicalChildChunk(
                chunk_id=f"chunk_{index:02d}",
                parent_id=f"parent_{index:02d}",
                document_id="becker_phosphates_and_phosphoric_acid",
                document_title="Phosphates and Phosphoric Acid",
                source_file="01_becker_phosphates_and_phosphoric_acid.pdf",
                domains=("phosphoric_acid_process",),
                section="Evaporation",
                chunk_type=TechnicalChunkType.DEFINITION,
                page_start=index + 1,
                page_end=index + 1,
                text=text,
                display_text=text,
                embedding_text=f"Document: Becker\n{text}",
                bm25_text=f"Evaporation\n{text}",
                token_count=100,
                sha256=f"{index + 1:064x}",
            )
        )
        runtime.append(
            DocumentChunk(
                chunk_id=f"chunk_{index:02d}",
                document_id="becker_phosphates_and_phosphoric_acid",
                source_file="01_becker_phosphates_and_phosphoric_acid.pdf",
                chunk_index=index,
                source_pages=[index + 1],
                page_start=index + 1,
                page_end=index + 1,
                text=text,
                embedding_text=f"Document: Becker\n{text}",
                body_token_count=100,
                token_count=100,
                chunk_type="definition",
            )
        )

    write_jsonl(tmp_path / "chunks.jsonl", technical)
    write_jsonl(tmp_path / "parents.jsonl", [])
    dense_results = [
        SimpleNamespace(rank=rank, score=0.9, chunk=chunk)
        for rank, chunk in enumerate(runtime, start=1)
    ]
    bm25_results = [
        SimpleNamespace(rank=rank, score=3.0, chunk=chunk)
        for rank, chunk in enumerate(runtime, start=1)
    ]
    retriever = SimpleNamespace(
        dense_retriever=_Recorder(dense_results),
        bm25_retriever=_Recorder(bm25_results),
        config=SimpleNamespace(
            dense_weight=1.0,
            bm25_weight=1.0,
            rrf_k=60,
        ),
    )
    engine = QualityRetrievalEngine(
        version_directory=tmp_path,
        retriever=retriever,
        reranker=_Reranker(),
    )

    result = engine.retrieve(
        "C’est quoi un évaporateur ?",
        standalone_query="C’est quoi un évaporateur ?",
        question_type="definition",
    )

    assert len(result.hybrid.results) == 20
    assert len(result.selected) == 5
    assert len(result.bundles) == 5
    assert len(
        {bundle.anchor_chunk_id for bundle in result.bundles}
    ) == 5
    assert sum(bundle.token_count for bundle in result.bundles) <= 2600


def test_quality_inference_has_no_gold_or_reference_answer_input() -> None:
    signature = inspect.signature(QualityRetrievalEngine.retrieve)
    source = inspect.getsource(QualityRetrievalEngine.retrieve).casefold()

    assert "gold" not in source
    assert "reference_answer" not in source
    assert "query_id" not in signature.parameters


class _CoverageAwareRecorder:
    def __init__(
        self,
        *,
        primary_results: list[object],
        product_results: list[object],
    ) -> None:
        self.primary_results = primary_results
        self.product_results = product_results

    def search(
        self,
        query: str,
        *,
        top_k: int,
        **_kwargs: object,
    ) -> object:
        normalized = query.casefold()
        is_product_recovery = (
            "product is withdrawn" in normalized
            or "product acid outlet" in normalized
        )
        results = (
            self.product_results
            if is_product_recovery
            else self.primary_results
        )
        return SimpleNamespace(
            results=results[:top_k],
            search_duration_ms=1.0,
        )


def test_process_flow_recovers_product_outlet_outside_primary_pool(
    tmp_path: Path,
) -> None:
    document_id = "becker_phosphates_and_phosphoric_acid"
    evidence_texts = [
        "Weak phosphoric acid is introduced into the evaporator.",
            "The cycling acid leaves the vapor body through a conical bottom.",
        "The circulation pump sends acid through the heat exchanger.",
        "The acid then enters the vapor body and flash chamber.",
        "Liquid returns through the recirculation line.",
    ]
    filler_texts = [
        f"General equipment note {index}."
        for index in range(16)
    ]
    product_text = (
        "Concentrated phosphoric acid product is withdrawn from the "
        "evaporator circulation loop."
    )
    all_texts = [*evidence_texts, *filler_texts, product_text]
    technical: list[TechnicalChildChunk] = []
    runtime: list[DocumentChunk] = []

    for index, text in enumerate(all_texts):
        chunk_id = f"flow_{index:02d}"
        technical.append(
            TechnicalChildChunk(
                chunk_id=chunk_id,
                parent_id=f"parent_{index:02d}",
                document_id=document_id,
                document_title="Phosphates and Phosphoric Acid",
                source_file="01_becker_phosphates_and_phosphoric_acid.pdf",
                domains=("phosphoric_acid_process", "equipment"),
                section="Acid concentration systems",
                chunk_type=TechnicalChunkType.PROCESS_DESCRIPTION,
                page_start=index + 1,
                page_end=index + 1,
                text=text,
                display_text=text,
                embedding_text=f"Document: Becker\n{text}",
                bm25_text=f"Acid concentration systems\n{text}",
                token_count=40,
                sha256=f"{index + 1:064x}",
            )
        )
        runtime.append(
            DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                source_file="01_becker_phosphates_and_phosphoric_acid.pdf",
                chunk_index=index,
                source_pages=[index + 1],
                page_start=index + 1,
                page_end=index + 1,
                text=text,
                embedding_text=f"Document: Becker\n{text}",
                body_token_count=40,
                token_count=40,
                chunk_type="process_description",
            )
        )

    write_jsonl(tmp_path / "chunks.jsonl", technical)
    write_jsonl(tmp_path / "parents.jsonl", [])
    primary_runtime = runtime[:20]
    product_runtime = [runtime[-1], *runtime[:19]]
    dense_primary = [
        SimpleNamespace(rank=rank, score=0.9, chunk=chunk)
        for rank, chunk in enumerate(primary_runtime, start=1)
    ]
    dense_product = [
        SimpleNamespace(rank=rank, score=0.9, chunk=chunk)
        for rank, chunk in enumerate(product_runtime, start=1)
    ]
    bm25_primary = [
        SimpleNamespace(rank=rank, score=3.0, chunk=chunk)
        for rank, chunk in enumerate(primary_runtime, start=1)
    ]
    bm25_product = [
        SimpleNamespace(rank=rank, score=3.0, chunk=chunk)
        for rank, chunk in enumerate(product_runtime, start=1)
    ]
    retriever = SimpleNamespace(
        dense_retriever=_CoverageAwareRecorder(
            primary_results=dense_primary,
            product_results=dense_product,
        ),
        bm25_retriever=_CoverageAwareRecorder(
            primary_results=bm25_primary,
            product_results=bm25_product,
        ),
        config=SimpleNamespace(
            dense_weight=1.0,
            bm25_weight=1.0,
            rrf_k=60,
        ),
    )
    engine = QualityRetrievalEngine(
        version_directory=tmp_path,
        retriever=retriever,
        reranker=_Reranker(),
    )

    result = engine.retrieve(
        "Describe the acid path from feed inlet to product outlet.",
        standalone_query=(
            "Describe the phosphoric acid path through a forced-circulation "
            "evaporator from feed inlet to concentrated product outlet."
        ),
        question_type="process_flow",
    )

    assert result.coverage.complete
    assert "flow_21" in {item.chunk_id for item in result.selected}
    assert any(
        "coverage_recovery" in candidate.matched_retrievers
        for candidate in result.hybrid.results
    )


def test_process_scope_removes_refrigeration_passage() -> None:
    from phosprocess.retrieval.retrieval_planner import build_retrieval_plan

    process = _chunk("process", "book").model_copy(
        update={
            "section": "EVAPORATOR TYPES AND APPLICATIONS",
            "text": (
                "A forced-circulation evaporator pump returns circulating "
                "liquor to the flash chamber."
            ),
        }
    )
    refrigeration = _chunk("refrigeration", "book").model_copy(
        update={
            "section": "MECHANICAL REFRIGERATION (VAPOR COMPRESSION SYSTEMS)",
            "text": (
                "Liquid refrigerant enters the air cooler and the refrigerant "
                "returns to the compressor crankcase."
            ),
        }
    )
    response = RerankingResponse(
        query="circulation pump",
        model_name="model",
        device="cpu",
        candidates_received=2,
        top_k_requested=2,
        reranking_duration_ms=1.0,
        results=[
            _result(1, 0.9, refrigeration),
            _result(2, 0.8, process),
        ],
    )
    question = "Quel est le rôle de la pompe de circulation ?"
    plan = build_retrieval_plan(
        question,
        standalone_query=(
            "Quel est le rôle de la pompe de circulation dans un évaporateur "
            "à circulation forcée d'acide phosphorique ?"
        ),
        question_type="explanation",
    )

    filtered = QualityRetrievalEngine._filter_process_scope_incompatibilities(
        response,
        plan=plan,
    )

    assert [item.chunk.chunk_id for item in filtered.results] == ["process"]
    assert filtered.results[0].rank == 1
