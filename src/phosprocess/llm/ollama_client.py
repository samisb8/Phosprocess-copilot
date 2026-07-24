"""Client Python pour communiquer avec le serveur local Ollama."""

from dataclasses import dataclass

from ollama import Client, ResponseError


@dataclass(frozen=True, slots=True)
class OllamaConfig:
    """Configuration du modèle local."""

    host: str = "http://localhost:11434"
    model: str = "qwen3:8b"
    temperature: float = 0.1
    context_size: int = 4096
    max_output_tokens: int = 600


class OllamaLLM:
    """Interface centralisée entre PhosProcess Copilot et Ollama."""

    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config = config or OllamaConfig()
        self.client = Client(host=self.config.host)

    def chat(self, question: str, system_prompt: str) -> str:
        """Envoyer une question au modèle et retourner sa réponse."""

        if not question.strip():
            raise ValueError("La question ne peut pas être vide.")

        try:
            response = self.client.chat(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                think=False,
                options={
                    "temperature": self.config.temperature,
                    "num_ctx": self.config.context_size,
                    "num_predict": self.config.max_output_tokens,
                },
            )
        except ResponseError as error:
            raise RuntimeError(
                f"Erreur Ollama {error.status_code}: {error.error}"
            ) from error

        content = response.message.content

        if not content:
            raise RuntimeError("Ollama a retourné une réponse vide.")

        return content.strip()