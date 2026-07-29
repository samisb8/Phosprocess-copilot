"""Chat route exposing the production PhosProcess RAG."""

from __future__ import annotations

import logging
from asyncio import Lock
from functools import partial
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

from phosprocess.api.database_dependencies import (
    get_chat_persistence_service,
)
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
from phosprocess.database.repositories.chat_repository import (
    ChatSessionNotFoundError,
)
from phosprocess.database.services.chat_persistence import (
    ChatPersistenceService,
    PersistedChatExchange,
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


def _map_response(
    response: RAGResponse,
    persisted: PersistedChatExchange,
) -> ChatResponse:
    """Convert the internal RAG and persistence results to the API."""

    return ChatResponse(
        session_id=persisted.session_id,
        user_message_id=persisted.user_message_id,
        assistant_message_id=persisted.assistant_message_id,
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
        status.HTTP_404_NOT_FOUND: {
            "description": "The requested chat session does not exist.",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "The RAG or database service is not ready.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": (
                "The RAG execution or persistence transaction failed."
            ),
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
    persistence_service: Annotated[
        ChatPersistenceService,
        Depends(get_chat_persistence_service),
    ],
) -> ChatResponse:
    """Run the RAG and persist its complete exchange."""

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

    try:
        persistence_operation = partial(
            persistence_service.persist_exchange,
            response=rag_response,
            session_id=payload.session_id,
        )
        persisted = await run_in_threadpool(
            persistence_operation
        )
    except ChatSessionNotFoundError as exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exception),
        ) from exception
    except Exception as exception:
        LOGGER.exception(
            "The RAG exchange could not be persisted."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The chat exchange could not be persisted.",
        ) from exception

    return _map_response(rag_response, persisted)
