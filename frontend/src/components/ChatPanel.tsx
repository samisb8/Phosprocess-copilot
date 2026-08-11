import {
  useState,
  type FormEvent,
} from 'react'

import type {
  ChatSessionHistory,
} from '../types/chat'

import { ChatComposer } from './chat/ChatComposer'
import { ChatHeader } from './chat/ChatHeader'
import { MessageList } from './chat/MessageList'
import { WelcomeState } from './chat/WelcomeState'

interface ChatPanelProps {
  history: ChatSessionHistory | null
  loading: boolean
  sending: boolean
  pendingQuestion: string | null
  onSend: (question: string) => Promise<void>
}

export function ChatPanel({
  history,
  loading,
  sending,
  pendingQuestion,
  onSend,
}: ChatPanelProps) {
  const [draft, setDraft] = useState('')

  async function submitQuestion(
    event?: FormEvent<HTMLFormElement>,
  ) {
    event?.preventDefault()

    const question = draft.trim()

    if (!question || sending) {
      return
    }

    setDraft('')

    try {
      await onSend(question)
    } catch {
      setDraft(question)
    }
  }

  const messages = history?.messages ?? []

  const showWelcome =
    !loading &&
    messages.length === 0 &&
    !pendingQuestion

  return (
    <main
      className={`chat-panel${
        showWelcome
          ? ' chat-panel-empty'
          : ''
      }`}
    >
      <ChatHeader
        title={
          history?.title ??
          'Nouvelle conversation'
        }
      />

      <section
        className={`message-area${
          showWelcome
            ? ' welcome-area'
            : ''
        }`}
      >
        {loading && (
          <div className="center-state">
            Chargement de l’historique…
          </div>
        )}

        {showWelcome && (
          <div className="welcome-stack">
            <WelcomeState
              onSuggestion={setDraft}
            />

            <ChatComposer
              draft={draft}
              sending={sending}
              onDraftChange={setDraft}
              onSubmit={submitQuestion}
            />
          </div>
        )}

        {!loading && !showWelcome && (
          <MessageList
            messages={messages}
            pendingQuestion={pendingQuestion}
          />
        )}
      </section>

      {!showWelcome && (
        <div className="composer-dock">
          <ChatComposer
            draft={draft}
            sending={sending}
            onDraftChange={setDraft}
            onSubmit={submitQuestion}
          />
        </div>
      )}
    </main>
  )
}