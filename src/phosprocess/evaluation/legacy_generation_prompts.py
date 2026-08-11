"""Research-only repair prompt retained for Phase-11 reproducibility."""

REPAIR_SYSTEM_PROMPT = (
    "Corrige uniquement les affirmations ou citations qui ne sont pas "
    "soutenues par les passages documentaires fournis. "
    "Supprime une affirmation si aucune source ne la soutient. "
    "N'ajoute aucun fait absent des preuves. "
    "Conserve uniquement des citations [Source N] correspondant réellement "
    "aux sources disponibles. "
    "Si aucune réponse fiable ne reste, utilise exactement la formulation "
    "d'insuffisance demandée. "
    "Ne limite pas artificiellement le nombre de phrases, de faits ou de "
    "citations."
)


def build_repair_prompt(
    *,
    original_prompt: str,
    invalid_output: str,
    rejection_reason: str,
    json_output: bool,
) -> str:
    """Recreate the rejected Phase-11 repair request for research replay."""

    expected = (
        'Retourne uniquement {"answer":"..."} sans autre champ.'
        if json_output
        else "Retourne uniquement le texte corrigé."
    )
    return "\n\n".join(
        [
            original_prompt,
            f"SORTIE INVALIDE\n{invalid_output}",
            f"REJET\n{rejection_reason}",
            (
                "Corrige la réponse en utilisant uniquement les preuves déjà "
                "fournies. Si le rejet signale des aspects importants manquants, "
                "ajoute les informations nécessaires uniquement lorsqu'elles sont "
                "explicitement soutenues par ces mêmes preuves. Corrige ou supprime "
                "toute affirmation non soutenue. N'introduis aucun fait extérieur "
                "aux preuves."
            ),
            expected,
        ]
    )
