"""Chat route exposing the production PhosProcess RAG."""

from __future__ import annotations

import logging
from asyncio import Lock
from functools import partial
from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Response,
    status,
)
from starlette.concurrency import run_in_threadpool

from phosprocess.api.database_dependencies import (
    get_chat_history_service,
    get_chat_persistence_service,
    get_chat_session_listing_service,
    get_chat_session_management_service,
)
from phosprocess.api.dependencies import (
    RAGService,
    get_rag_inference_lock,
    get_rag_service,
)
from phosprocess.api.schemas.chat import (
    ChatHistoryCitationResponse,
    ChatHistoryMessageResponse,
    ChatPolicyResponse,
    ChatRequest,
    ChatResponse,
    ChatSessionHistoryResponse,
    ChatSessionListResponse,
    ChatSessionRenameRequest,
    ChatSessionRenameResponse,
    ChatSessionSummaryResponse,
    ChatSourceResponse,
    ChatTimingsResponse,
)
from phosprocess.database.repositories.chat_repository import (
    ChatSessionNotFoundError,
)
from phosprocess.database.services.chat_history import (
    ChatHistoryCitation,
    ChatHistoryMessage,
    ChatHistoryService,
    ChatSessionHistory,
)
from phosprocess.database.services.chat_persistence import (
    ChatPersistenceService,
    PersistedChatExchange,
)
from phosprocess.database.services.chat_session_listing import (
    ChatSessionListingService,
    ChatSessionPage,
    ChatSessionSummary,
)
from phosprocess.database.services.chat_session_management import (
    ChatSessionManagementService,
    RenamedChatSession,
)
from phosprocess.rag.schemas import (
    ChatMessage as RAGChatMessage,
)
from phosprocess.rag.schemas import (
    RAGResponse,
    RAGSource,
)

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


def _map_renamed_session(
    session: RenamedChatSession,
) -> ChatSessionRenameResponse:
    """Convert a renamed conversation to the public API."""

    return ChatSessionRenameResponse(
        session_id=session.session_id,
        title=session.title,
        updated_at=session.updated_at,
    )


def _map_session_summary(
    summary: ChatSessionSummary,
) -> ChatSessionSummaryResponse:
    """Convert one conversation summary to the public API."""

    return ChatSessionSummaryResponse(
        session_id=summary.session_id,
        title=summary.title,
        created_at=summary.created_at,
        updated_at=summary.updated_at,
        message_count=summary.message_count,
    )


def _map_session_page(
    page: ChatSessionPage,
) -> ChatSessionListResponse:
    """Convert one paginated result to the public API."""

    return ChatSessionListResponse(
        items=[_map_session_summary(summary) for summary in page.items],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
    )


def _map_history_citation(
    citation: ChatHistoryCitation,
) -> ChatHistoryCitationResponse:
    """Convert one persisted citation to its public contract."""

    return ChatHistoryCitationResponse(
        id=citation.id,
        source_number=citation.source_number,
        chunk_id=citation.chunk_id,
        document_name=citation.document_name,
        pages=list(citation.pages),
        section=citation.section,
        excerpt=citation.excerpt,
        document_title=citation.document_title,
        filename=citation.filename,
        chapter=citation.chapter,
        page_start=citation.page_start,
        page_end=citation.page_end,
        domain=citation.domain,
        chunk_type=citation.chunk_type,
        is_cited=citation.is_cited,
        created_at=citation.created_at,
    )


def _map_history_message(
    message: ChatHistoryMessage,
) -> ChatHistoryMessageResponse:
    """Convert one persisted message to its public contract."""

    return ChatHistoryMessageResponse(
        id=message.id,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
        insufficient_context=message.insufficient_context,
        model_name=message.model_name,
        response_language=message.response_language,
        question_type=message.question_type,
        total_ms=message.total_ms,
        citations=[_map_history_citation(citation) for citation in message.citations],
    )


def _map_history(
    history: ChatSessionHistory,
) -> ChatSessionHistoryResponse:
    """Convert a complete persisted conversation to the API."""

    return ChatSessionHistoryResponse(
        session_id=history.session_id,
        title=history.title,
        created_at=history.created_at,
        updated_at=history.updated_at,
        messages=[_map_history_message(message) for message in history.messages],
    )


def _rag_history_messages(
    history: ChatSessionHistory,
) -> list[RAGChatMessage]:
    """Convert persisted session messages to non-documentary RAG memory."""

    return [
        RAGChatMessage(
            role=message.role,
            content=message.content,
        )
        for message in history.messages
        if message.role in {"user", "assistant"}
    ]


def _answer_with_history(
    service: RAGService,
    *,
    question: str,
    history: list[RAGChatMessage],
    source_mode: str,
    language_mode: str,
) -> RAGResponse:
    """Consume the conversational RAG stream and return its final response."""

    error_message: str | None = None

    for event in service.stream_answer(
        question,
        history,
        source_mode=source_mode,
        language_mode=language_mode,
    ):
        if event.event_type == "completed" and event.response is not None:
            return event.response

        if event.event_type == "error":
            error_message = event.content or "Conversational RAG execution failed."

    raise RuntimeError(
        error_message or "Conversational RAG did not produce a final response."
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
        sources=[_map_source(source) for source in response.sources],
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


@router.get(
    "/chat/sessions",
    response_model=ChatSessionListResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "The database service is not ready.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "The conversations could not be listed.",
        },
    },
    summary="List persisted chat conversations",
)
async def list_chat_sessions(
    listing_service: Annotated[
        ChatSessionListingService,
        Depends(get_chat_session_listing_service),
    ],
    limit: Annotated[
        int,
        Query(ge=1, le=100),
    ] = 20,
    offset: Annotated[
        int,
        Query(ge=0),
    ] = 0,
) -> ChatSessionListResponse:
    """Return conversations ordered by latest activity."""

    try:
        operation = partial(
            listing_service.list_sessions,
            limit=limit,
            offset=offset,
        )
        page = await run_in_threadpool(operation)
    except Exception as exception:
        LOGGER.exception("The chat sessions could not be listed.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The chat sessions could not be listed.",
        ) from exception

    return _map_session_page(page)


@router.patch(
    "/chat/sessions/{session_id}",
    response_model=ChatSessionRenameResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "The new title is invalid.",
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "The requested chat session does not exist.",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "The database service is not ready.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "The conversation could not be renamed.",
        },
    },
    summary="Rename a persisted chat conversation",
)
async def rename_chat_session(
    session_id: UUID,
    payload: ChatSessionRenameRequest,
    management_service: Annotated[
        ChatSessionManagementService,
        Depends(get_chat_session_management_service),
    ],
) -> ChatSessionRenameResponse:
    """Rename one persisted conversation."""

    try:
        operation = partial(
            management_service.rename_session,
            session_id,
            title=payload.title,
        )
        renamed_session = await run_in_threadpool(operation)
    except ChatSessionNotFoundError as exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exception),
        ) from exception
    except ValueError as exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exception),
        ) from exception
    except Exception as exception:
        LOGGER.exception("The chat session could not be renamed.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The chat session could not be renamed.",
        ) from exception

    return _map_renamed_session(renamed_session)


@router.delete(
    "/chat/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "The requested chat session does not exist.",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "The database service is not ready.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "The conversation could not be deleted.",
        },
    },
    summary="Delete a persisted chat conversation",
)
async def delete_chat_session(
    session_id: UUID,
    management_service: Annotated[
        ChatSessionManagementService,
        Depends(get_chat_session_management_service),
    ],
) -> Response:
    """Delete one conversation and its dependent records."""

    try:
        operation = partial(
            management_service.delete_session,
            session_id,
        )
        await run_in_threadpool(operation)
    except ChatSessionNotFoundError as exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exception),
        ) from exception
    except Exception as exception:
        LOGGER.exception("The chat session could not be deleted.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The chat session could not be deleted.",
        ) from exception

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/chat/sessions/{session_id}",
    response_model=ChatSessionHistoryResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "The requested chat session does not exist.",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "The database service is not ready.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "The conversation history could not be loaded.",
        },
    },
    summary="Read a persisted chat conversation",
)
async def get_chat_session_history(
    session_id: UUID,
    history_service: Annotated[
        ChatHistoryService,
        Depends(get_chat_history_service),
    ],
) -> ChatSessionHistoryResponse:
    """Return one complete persisted conversation."""

    try:
        operation = partial(
            history_service.get_session_history,
            session_id,
        )
        history = await run_in_threadpool(operation)
    except ChatSessionNotFoundError as exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exception),
        ) from exception
    except Exception as exception:
        LOGGER.exception("The chat session history could not be loaded.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The chat session history could not be loaded.",
        ) from exception

    return _map_history(history)


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
            "description": ("The RAG execution or persistence transaction failed."),
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
    history_service: Annotated[
        ChatHistoryService,
        Depends(get_chat_history_service),
    ],
) -> ChatResponse:
    """Run the RAG and persist its complete exchange."""

    rag_history: list[RAGChatMessage] | None = None

    if payload.session_id is not None:
        try:
            history_operation = partial(
                history_service.get_session_history,
                payload.session_id,
            )
            persisted_history = await run_in_threadpool(history_operation)
            rag_history = _rag_history_messages(persisted_history)
        except ChatSessionNotFoundError as exception:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exception),
            ) from exception
        except Exception as exception:
            LOGGER.exception("The chat session history could not be loaded.")

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The chat session history could not be loaded.",
            ) from exception

    try:
        async with inference_lock:
            if rag_history is None:
                operation = partial(
                    service.answer,
                    payload.question,
                    source_mode=payload.source_mode,
                    language_mode=payload.language_mode,
                )
            else:
                operation = partial(
                    _answer_with_history,
                    service,
                    question=payload.question,
                    history=rag_history,
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
        persisted = await run_in_threadpool(persistence_operation)
    except ChatSessionNotFoundError as exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exception),
        ) from exception
    except Exception as exception:
        LOGGER.exception("The RAG exchange could not be persisted.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The chat exchange could not be persisted.",
        ) from exception

    return _map_response(rag_response, persisted)
