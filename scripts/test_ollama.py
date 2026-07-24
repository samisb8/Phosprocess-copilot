"""Test manuel de la connexion avec Qwen3."""

from phosprocess.llm.ollama_client import OllamaLLM

SYSTEM_PROMPT = """
Tu es un assistant spécialisé dans la production d'acide phosphorique
par voie humide dihydrate.

Réponds clairement et brièvement.
N'invente aucune information technique.
"""


def main() -> None:
    """Tester la communication Python vers Ollama."""

    llm = OllamaLLM()

    response = llm.chat(
        question=(
        "Contexte : dans le procédé humide dihydrate, le sulfate de calcium "
        "cristallise sous forme CaSO4·2H2O.\n\n"
        "Question : que signifie DH et quel composé cristallise ?"
    ),
        system_prompt=SYSTEM_PROMPT,
)

    print(response)


if __name__ == "__main__":
    main()