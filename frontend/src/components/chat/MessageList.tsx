import type { ChatMessage } from '../../types/chat'
import { MessageCard } from './MessageCard'
import { ThinkingMessage } from './ThinkingMessage'

interface MessageListProps {
  messages: ChatMessage[]
  pendingQuestion: string | null
}

export function MessageList({
  messages,
  pendingQuestion,
}: MessageListProps) {
  return (
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
            <div className="message-avatar">
              Vous
            </div>

            <div className="message-body">
              <div className="message-header">
                <strong>Vous</strong>
              </div>

              <p className="message-content">
                {pendingQuestion}
              </p>
            </div>
          </article>

          <ThinkingMessage />
        </>
      )}
    </div>
  )
}