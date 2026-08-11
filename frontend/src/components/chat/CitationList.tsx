import type { ChatMessage } from '../../types/chat'

interface CitationListProps {
  citations: ChatMessage['citations']
}

export function CitationList({
  citations,
}: CitationListProps) {
  if (citations.length === 0) {
    return null
  }

  return (
    <details className="citations">
      <summary>
        <span>Sources documentaires</span>
        <strong>{citations.length}</strong>
      </summary>

      <div className="citation-list">
        {citations.map((citation) => (
          <article
            className="citation-card"
            key={citation.id}
          >
            <div className="citation-title">
              <span className="citation-number">
                {String(citation.source_number).padStart(2, '0')}
              </span>

              <div>
                <strong>
                  {citation.document_title ??
                    citation.document_name}
                </strong>

                <div className="citation-location">
                  {citation.section && (
                    <span>{citation.section}</span>
                  )}

                  {citation.pages.length > 0 && (
                    <span>
                      Page
                      {citation.pages.length > 1
                        ? 's'
                        : ''}{' '}
                      {citation.pages.join(', ')}
                    </span>
                  )}
                </div>
              </div>
            </div>

            <p>{citation.excerpt}</p>

            {citation.is_cited && (
              <span className="cited-badge">
                Citée dans la réponse
              </span>
            )}
          </article>
        ))}
      </div>
    </details>
  )
}