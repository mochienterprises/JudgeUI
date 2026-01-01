import os
from openai import OpenAI

from .base import LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    DEFAULT_MODEL = "gpt-4o"

    AVAILABLE_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "o1-preview",
        "o1-mini",
    ]

    def __init__(self, api_key: str | None = None):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY env var.")
        self.client = OpenAI(api_key=self.api_key)

    @property
    def name(self) -> str:
        return "openai"

    def call(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 2000,
        model: str | None = None,
    ) -> LLMResponse:
        model = model or self.DEFAULT_MODEL

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Newer models (o1, gpt-5) use max_completion_tokens instead of max_tokens
        uses_completion_tokens = model.startswith("o1") or model.startswith("gpt-5")

        kwargs = {
            "model": model,
            "messages": messages,
        }

        if not uses_completion_tokens:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
        else:
            kwargs["max_completion_tokens"] = max_tokens

        response = self.client.chat.completions.create(**kwargs)

        return LLMResponse(
            text=response.choices[0].message.content.strip(),
            model=model,
            provider=self.name,
        )

    def list_models(self) -> list[str]:
        return self.AVAILABLE_MODELS
