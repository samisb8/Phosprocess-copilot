interface WelcomeStateProps {
  onSuggestion: (question: string) => void
}

const suggestions = [
  {
    label: 'Fonctionnement équipement',
    description:
      'Comprendre le rôle et le fonctionnement des équipements du procédé.',
    question:
      'Quel est le rôle de la pompe de circulation dans un évaporateur à circulation forcée ?',
  },
  {
    label: 'Paramètres de procédé',
    description:
      'Identifier les variables qui influencent les performances du procédé.',
    question:
      'Quels paramètres influencent les performances d’un évaporateur industriel ?',
  },
  {
    label: 'Diagnostic industriel',
    description:
      'Analyser les causes et conséquences d’une situation d’exploitation.',
    question:
      'Explique les principaux risques associés à une circulation insuffisante.',
  },
]

export function WelcomeState({
  onSuggestion,
}: WelcomeStateProps) {
  return (
    <section className="welcome-state">
      <div className="welcome-icon">
        <img
          src="/assets/ocp_logo.png"
          alt="OCP Group"
        />
      </div>

      <span className="eyebrow">
        Assistant industriel RAG
      </span>

      <h2>
        L’intelligence documentaire
        <br />
        pour vos procédés phosphoriques.
      </h2>

      <p>
        Interrogez votre documentation technique et obtenez
        des réponses contextualisées, sourcées et directement
        exploitables.
      </p>

      <div className="suggestion-section">
        <span className="suggestion-label">
          Exemples de questions
        </span>

        <div className="suggestion-grid">
          {suggestions.map((suggestion) => (
            <button
              key={suggestion.label}
              type="button"
              onClick={() => onSuggestion(suggestion.question)}
            >
              <strong>{suggestion.label}</strong>

              <span>
                {suggestion.description}
              </span>

              <span className="suggestion-action">
                Utiliser cette question →
              </span>
            </button>
          ))}
        </div>
      </div>
    </section>
  )
}