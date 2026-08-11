import type {
  ChatSessionHistory,
  ChatSessionPage,
  RenameSessionResponse,
  SendChatRequest,
  SendChatResponse,
} from '../types/chat'

const API_ROOT = '/api/v1'

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

function extractErrorMessage(rawBody: string, fallback: string): string {
  if (!rawBody) {
    return fallback
  }

  try {
    const parsed = JSON.parse(rawBody) as { detail?: unknown }

    if (typeof parsed.detail === 'string') {
      return parsed.detail
    }
  } catch {
    return rawBody
  }

  return fallback
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    ...options,
    headers: {
      Accept: 'application/json',
      ...options.headers,
    },
  })

  if (!response.ok) {
    const rawBody = await response.text()
    const message = extractErrorMessage(
      rawBody,
      `Erreur HTTP ${response.status}`,
    )

    throw new ApiError(response.status, message)
  }

  if (response.status === 204) {
    return undefined as T
  }

  return (await response.json()) as T
}

export function listSessions(
  limit = 50,
  offset = 0,
): Promise<ChatSessionPage> {
  const query = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  })

  return request<ChatSessionPage>(
    `/chat/sessions?${query.toString()}`,
  )
}

export function getSession(
  sessionId: string,
): Promise<ChatSessionHistory> {
  return request<ChatSessionHistory>(
    `/chat/sessions/${encodeURIComponent(sessionId)}`,
  )
}

export function sendChat(
  payload: SendChatRequest,
): Promise<SendChatResponse> {
  return request<SendChatResponse>('/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      question: payload.question,
      session_id: payload.session_id,
      source_mode: payload.source_mode ?? 'automatic',
      language_mode: payload.language_mode ?? 'auto',
    }),
  })
}

export function renameSession(
  sessionId: string,
  title: string,
): Promise<RenameSessionResponse> {
  return request<RenameSessionResponse>(
    `/chat/sessions/${encodeURIComponent(sessionId)}`,
    {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ title }),
    },
  )
}

export function deleteSession(sessionId: string): Promise<void> {
  return request<void>(
    `/chat/sessions/${encodeURIComponent(sessionId)}`,
    {
      method: 'DELETE',
    },
  )
}