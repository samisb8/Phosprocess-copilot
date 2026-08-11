export function ThinkingMessage() {
  return (
    <article className="message assistant-message thinking-message">
      <div className="message-avatar">P</div>

      <div className="message-body">
        <div className="message-header">
          <strong>PhosProcess Copilot</strong>
        </div>

        <div className="thinking">
          <span />
          <span />
          <span />

          <p>
            Recherche documentaire et génération de la réponse…
          </p>
        </div>
      </div>
    </article>
  )
}