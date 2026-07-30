import type { ChatSessionSummary } from '../types/chat'

interface SidebarProps {
  sessions: ChatSessionSummary[]
  activeSessionId: string | null
  loading: boolean
  onNewChat: () => void
  onSelect: (sessionId: string) => void
  onRename: (session: ChatSessionSummary) => void
  onDelete: (session: ChatSessionSummary) => void
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat('fr-FR', {
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value))
}

export function Sidebar({
  sessions,
  activeSessionId,
  loading,
  onNewChat,
  onSelect,
  onRename,
  onDelete,
}: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">P</div>
        <div>
          <strong>PhosProcess</strong>
          <span>Industrial Copilot</span>
        </div>
      </div>

      <button
        className="new-chat-button"
        type="button"
        onClick={onNewChat}
      >
        <span aria-hidden="true">＋</span>
        Nouveau chat
      </button>

      <div className="sidebar-section-header">
        <span>Conversations</span>
        <span>{sessions.length}</span>
      </div>

      <div className="session-list">
        {loading && (
          <p className="sidebar-placeholder">
            Chargement des conversations…
          </p>
        )}

        {!loading && sessions.length === 0 && (
          <p className="sidebar-placeholder">
            Aucune conversation enregistrée.
          </p>
        )}

        {sessions.map((session) => {
          const active = session.session_id === activeSessionId

          return (
            <div
              className={`session-card${active ? ' active' : ''}`}
              key={session.session_id}
            >
              <button
                className="session-select"
                type="button"
                onClick={() => onSelect(session.session_id)}
              >
                <strong>
                  {session.title ?? 'Conversation sans titre'}
                </strong>

                <span>
                  {session.message_count} messages ·{' '}
                  {formatDate(session.updated_at)}
                </span>
              </button>

              <div className="session-actions">
                <button
                  type="button"
                  title="Renommer"
                  aria-label="Renommer la conversation"
                  onClick={() => onRename(session)}
                >
                  ✎
                </button>

                <button
                  type="button"
                  title="Supprimer"
                  aria-label="Supprimer la conversation"
                  onClick={() => onDelete(session)}
                >
                  ×
                </button>
              </div>
            </div>
          )
        })}
      </div>
    </aside>
  )
}