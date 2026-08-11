import type { ChatMessage } from '../../types/chat'
import { CitationList } from './CitationList'

interface MessageCardProps {
  message: ChatMessage
}

export function MessageCard({
  message,
}: MessageCardProps) {
  const isAssistant = message.role === 'assistant'

  return (
    <article
      className={`message ${
        isAssistant
          ? 'assistant-message'
          : 'user-message'
      }`}
    >
      <div className="message-avatar">
        {isAssistant ? 'P' : 'Vous'}
      </div>

      <div className="message-body">
        <div className="message-header">
          <strong>
            {isAssistant
              ? 'PhosProcess Copilot'
              : 'Vous'}
          </strong>

          {message.insufficient_context && (
            <span className="context-warning">
              Contexte insuffisant
            </span>
          )}
        </div>

        <p className="message-content">
          {message.content}
        </p>

        <CitationList
          citations={message.citations}
        />
      </div>
    </article>
  )
}