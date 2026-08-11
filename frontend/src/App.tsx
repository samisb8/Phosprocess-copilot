import {
  useCallback,
  useEffect,
  useState,
} from 'react'

import {
  deleteSession,
  getSession,
  listSessions,
  renameSession,
  sendChat,
} from './api/chatApi'
import { ChatPanel } from './components/ChatPanel'
import { Sidebar } from './components/Sidebar'
import type {
  ChatSessionHistory,
  ChatSessionSummary,
} from './types/chat'

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message
  }

  return 'Une erreur inattendue est survenue.'
}

function App() {
  const [sessions, setSessions] = useState<
    ChatSessionSummary[]
  >([])
  const [activeSessionId, setActiveSessionId] = useState<
    string | null
  >(null)
  const [history, setHistory] =
    useState<ChatSessionHistory | null>(null)
  const [loadingSessions, setLoadingSessions] =
    useState(true)
  const [loadingHistory, setLoadingHistory] =
    useState(false)
  const [sending, setSending] = useState(false)
  const [pendingQuestion, setPendingQuestion] = useState<
    string | null
  >(null)
  const [error, setError] = useState<string | null>(null)

  const refreshSessions = useCallback(async () => {
    setLoadingSessions(true)

    try {
      const page = await listSessions(50, 0)
      setSessions(page.items)
    } catch (requestError) {
      setError(getErrorMessage(requestError))
    } finally {
      setLoadingSessions(false)
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    void listSessions(50, 0)
      .then((page) => {
        if (!cancelled) {
          setSessions(page.items)
        }
      })
      .catch((requestError: unknown) => {
        if (!cancelled) {
          setError(getErrorMessage(requestError))
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingSessions(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [])

  async function handleSelectSession(sessionId: string) {
    setActiveSessionId(sessionId)
    setLoadingHistory(true)
    setError(null)

    try {
      const loadedHistory = await getSession(sessionId)
      setHistory(loadedHistory)
    } catch (requestError) {
      setError(getErrorMessage(requestError))
    } finally {
      setLoadingHistory(false)
    }
  }

  function handleNewChat() {
    setActiveSessionId(null)
    setHistory(null)
    setError(null)
  }

  async function handleSend(question: string) {
    setSending(true)
    setPendingQuestion(question)
    setError(null)

    try {
      const response = await sendChat({
        question,
        session_id: activeSessionId ?? undefined,
      })

      const loadedHistory = await getSession(
        response.session_id,
      )

      setActiveSessionId(response.session_id)
      setHistory(loadedHistory)
      await refreshSessions()
    } catch (requestError) {
      setError(getErrorMessage(requestError))
      throw requestError
    } finally {
      setSending(false)
      setPendingQuestion(null)
    }
  }

  async function handleRename(
    session: ChatSessionSummary,
  ) {
    const nextTitle = window.prompt(
      'Nouveau titre de la conversation :',
      session.title ?? '',
    )

    if (nextTitle === null) {
      return
    }

    setError(null)

    try {
      const renamed = await renameSession(
        session.session_id,
        nextTitle,
      )

      if (history?.session_id === session.session_id) {
        setHistory({
          ...history,
          title: renamed.title,
          updated_at: renamed.updated_at,
        })
      }

      await refreshSessions()
    } catch (requestError) {
      setError(getErrorMessage(requestError))
    }
  }

  async function handleDelete(
    session: ChatSessionSummary,
  ) {
    const confirmed = window.confirm(
      `Supprimer définitivement « ${
        session.title ?? 'Conversation sans titre'
      } » ?`,
    )

    if (!confirmed) {
      return
    }

    setError(null)

    try {
      await deleteSession(session.session_id)

      if (activeSessionId === session.session_id) {
        setActiveSessionId(null)
        setHistory(null)
      }

      await refreshSessions()
    } catch (requestError) {
      setError(getErrorMessage(requestError))
    }
  }

  return (
    <div className="app-shell">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        loading={loadingSessions}
        onNewChat={handleNewChat}
        onSelect={(sessionId) => {
          void handleSelectSession(sessionId)
        }}
        onRename={(session) => {
          void handleRename(session)
        }}
        onDelete={(session) => {
          void handleDelete(session)
        }}
      />

      <div className="workspace">
        {error && (
          <div className="error-banner">
            <span>{error}</span>
            <button
              type="button"
              aria-label="Fermer le message d’erreur"
              onClick={() => setError(null)}
            >
              ×
            </button>
          </div>
        )}

        <ChatPanel
          history={history}
          loading={loadingHistory}
          sending={sending}
          pendingQuestion={pendingQuestion}
          onSend={handleSend}
        />
      </div>
    </div>
  )
}

export default App