import type {
  FormEvent,
  KeyboardEvent,
} from 'react'

interface ChatComposerProps {
  draft: string
  sending: boolean
  onDraftChange: (value: string) => void
  onSubmit: (
    event?: FormEvent<HTMLFormElement>,
  ) => void
}

export function ChatComposer({
  draft,
  sending,
  onDraftChange,
  onSubmit,
}: ChatComposerProps) {
  function handleKeyDown(
    event: KeyboardEvent<HTMLTextAreaElement>,
  ) {
    if (
      event.key === 'Enter' &&
      !event.shiftKey
    ) {
      event.preventDefault()
      onSubmit()
    }
  }

  return (
    <form
      className="composer"
      onSubmit={onSubmit}
    >
      <textarea
        value={draft}
        disabled={sending}
        maxLength={4000}
        placeholder="Posez une question sur votre procédé…"
        aria-label="Question pour PhosProcess Copilot"
        onChange={(event) =>
          onDraftChange(event.target.value)
        }
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
          {sending
            ? 'Génération…'
            : 'Envoyer'}
        </button>
      </div>
    </form>
  )
}