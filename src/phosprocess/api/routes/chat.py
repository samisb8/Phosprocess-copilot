"""Chat route exposing the production PhosProcess RAG."""

from __future__ import annotations

import logging
from asyncio import Lock
from functools import partial
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

from phosprocess.api.dependencies import (
    RAGService,
    get_rag_inference_lock,
    get_rag_service,
)
from phosprocess.api.schemas.chat import (
    ChatPolicyResponse,
    ChatRequest,
    ChatResponse,
    ChatSourceResponse,
    ChatTimingsResponse,
)
from phosprocess.rag.schemas import RAGResponse, RAGSource

LOGGER = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["chat"],
)


def _map_source(source: RAGSource) -> ChatSourceResponse:
    """Convert an internal retrieval source to a public citation."""

    return ChatSourceResponse(
        source_number=source.source_number,
        chunk_id=source.chunk_id,
        document_name=source.document_name,
        pages=list(source.pages),
        section=source.section,
        excerpt=source.excerpt,
        document_title=source.document_title,
        filename=source.filename,
        chapter=source.chapter,
        page_start=source.page_start,
        page_end=source.page_end,
        domain=source.domain,
        chunk_type=source.chunk_type,
    )


def _map_response(response: RAGResponse) -> ChatResponse:
    """Convert the internal RAG contract to the public API contract."""

    return ChatResponse(
        question=response.question,
        answer=response.answer,
        sources=[
            _map_source(source)
            for source in response.sources
        ],
        cited_source_numbers=list(response.cited_source_numbers),
        insufficient_context=response.insufficient_context,
        model_name=response.model_name,
        selected_variant=response.selected_variant,
        knowledge_base_snapshot_sha256=response.snapshot_sha256,
        candidate_count=response.candidate_count,
        selected_count=response.selected_count,
        source_policy=ChatPolicyResponse(
            route=response.source_policy_route,
            mode=response.source_policy_mode,
            primary=response.source_policy_primary,
            fallback_used=response.source_policy_fallback_used,
            forced=response.source_policy_forced,
        ),
        response_language=response.response_language,
        standalone_query=response.standalone_query,
        question_type=response.question_type,
        detected_domains=list(response.detected_domains),
        timings=ChatTimingsResponse(
            hybrid_ms=response.timings.hybrid_ms,
            reranking_ms=response.timings.reranking_ms,
            generation_ms=response.timings.generation_ms,
            total_ms=response.timings.total_ms,
            first_token_ms=response.timings.first_token_ms,
        ),
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Invalid RAG execution option.",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "The RAG service is not ready.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "The RAG request failed.",
        },
    },
    summary="Ask a grounded technical question",
)
async def chat(
    payload: ChatRequest,
    service: Annotated[
        RAGService,
        Depends(get_rag_service),
    ],
    inference_lock: Annotated[
        Lock,
        Depends(get_rag_inference_lock),
    ],
) -> ChatResponse:
    """Run one serialized RAG request without blocking FastAPI."""

    try:
        async with inference_lock:
            operation = partial(
                service.answer,
                payload.question,
                source_mode=payload.source_mode,
                language_mode=payload.language_mode,
            )
            rag_response = await run_in_threadpool(operation)
    except ValueError as exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exception) or "Invalid RAG request.",
        ) from exception
    except Exception as exception:
        LOGGER.exception("The RAG chat request failed.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The RAG request could not be completed.",
        ) from exception

    return _map_response(rag_response)
