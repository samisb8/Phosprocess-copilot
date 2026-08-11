interface ChatHeaderProps {
  title: string
}

export function ChatHeader({
  title,
}: ChatHeaderProps) {
  return (
    <header className="chat-header">
      <div className="chat-header-copy">
        <span className="eyebrow">
          Assistant industriel RAG
        </span>

        <h1>{title}</h1>

        <p>
          Intelligence documentaire pour les procédés phosphoriques
        </p>
      </div>

      <div className="status-pill">
        <span aria-hidden="true" />
        Système opérationnel
      </div>
    </header>
  )
}