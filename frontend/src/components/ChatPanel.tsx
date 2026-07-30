import {
  useState,
  type FormEvent,
  type KeyboardEvent,
} from 'react'

import type {
  ChatMessage,
  ChatSessionHistory,
} from '../types/chat'

interface ChatPanelProps {
  history: ChatSessionHistory | null
  loading: boolean
  sending: boolean
  pendingQuestion: string | null
  onSend: (question: string) => Promise<void>
}

function MessageCard({ message }: { message: ChatMessage }) {
  const isAssistant = message.role === 'assistant'

  return (
    <article
      className={`message ${
        isAssistant ? 'assistant-message' : 'user-message'
      }`}
    >
      <div className="message-avatar">
        {isAssistant ? 'AI' : 'Vous'}
      </div>

      <div className="message-body">
        <div className="message-header">
          <strong>
            {isAssistant ? 'PhosProcess Copilot' : 'Vous'}
          </strong>

          {message.insufficient_context && (
            <span className="context-warning">
              Contexte insuffisant
            </span>
          )}
        </div>

        <p className="message-content">{message.content}</p>

        {message.citations.length > 0 && (
          <details className="citations">
            <summary>
              {message.citations.length}{' '}
              {message.citations.length === 1
                ? 'source documentaire'
                : 'sources documentaires'}
            </summary>

            <div className="citation-list">
              {message.citations.map((citation) => (
                <article
                  className="citation-card"
                  key={citation.id}
                >
                  <div className="citation-title">
                    <strong>
                      Source {citation.source_number}
                    </strong>
                    <span>
                      {citation.document_title ??
                        citation.document_name}
                    </span>
                  </div>

                  <p>{citation.excerpt}</p>

                  <div className="citation-meta">
                    {citation.section && (
                      <span>{citation.section}</span>
                    )}

                    {citation.pages.length > 0 && (
                      <span>
                        Page{citation.pages.length > 1 ? 's' : ''}{' '}
                        {citation.pages.join(', ')}
                      </span>
                    )}

                    {citation.is_cited && (
                      <span className="cited-badge">
                        Citée dans la réponse
                      </span>
                    )}
                  </div>
                </article>
              ))}
            </div>
          </details>
        )}
      </div>
    </article>
  )
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

  function handleKeyDown(
    event: KeyboardEvent<HTMLTextAreaElement>,
  ) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      void submitQuestion()
    }
  }

  const messages = history?.messages ?? []

  return (
    <main className="chat-panel">
      <header className="chat-header">
        <div>
          <span className="eyebrow">Assistant industriel RAG</span>
          <h1>
            {history?.title ?? 'Nouvelle conversation'}
          </h1>
        </div>

        <div className="status-pill">
          <span />
          API connectée
        </div>
      </header>

      <section className="message-area">
        {loading && (
          <div className="center-state">
            Chargement de l’historique…
          </div>
        )}

        {!loading && messages.length === 0 && !pendingQuestion && (
          <div className="welcome-state">
            <div className="welcome-icon">P</div>
            <h2>Interrogez vos documents industriels</h2>
            <p>
              Posez une question sur les procédés, les équipements,
              l’exploitation ou la production d’acide phosphorique.
            </p>

            <div className="suggestion-grid">
              <button
                type="button"
                onClick={() =>
                  setDraft(
                    'Quel est le rôle de la pompe de circulation dans un évaporateur à circulation forcée ?',
                  )
                }
              >
                Fonctionnement d’un équipement
              </button>

              <button
                type="button"
                onClick={() =>
                  setDraft(
                    'Quels paramètres influencent les performances d’un évaporateur industriel ?',
                  )
                }
              >
                Paramètres de procédé
              </button>

              <button
                type="button"
                onClick={() =>
                  setDraft(
                    'Explique les principaux risques associés à une circulation insuffisante.',
                  )
                }
              >
                Diagnostic industriel
              </button>
            </div>
          </div>
        )}

        {!loading && (
          <div className="message-list">
            {messages.map((message) => (
              <MessageCard
                message={message}
                key={message.id}
              />
            ))}

            {pendingQuestion && (
              <>
                <article className="message user-message">
                  <div className="message-avatar">Vous</div>
                  <div className="message-body">
                    <div className="message-header">
                      <strong>Vous</strong>
                    </div>
                    <p className="message-content">
                      {pendingQuestion}
                    </p>
                  </div>
                </article>

                <article className="message assistant-message">
                  <div className="message-avatar">AI</div>
                  <div className="message-body">
                    <div className="message-header">
                      <strong>PhosProcess Copilot</strong>
                    </div>
                    <div className="thinking">
                      <span />
                      <span />
                      <span />
                      Recherche et génération en cours
                    </div>
                  </div>
                </article>
              </>
            )}
          </div>
        )}
      </section>

      <form className="composer" onSubmit={submitQuestion}>
        <textarea
          value={draft}
          disabled={sending}
          maxLength={4000}
          placeholder="Posez une question technique…"
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={handleKeyDown}
        />

        <div className="composer-footer">
          <span>
            Entrée pour envoyer · Maj + Entrée pour une nouvelle ligne
          </span>

          <button
            type="submit"
            disabled={sending || !draft.trim()}
          >
            {sending ? 'Génération…' : 'Envoyer'}
          </button>
        </div>
      </form>
    </main>
  )
}