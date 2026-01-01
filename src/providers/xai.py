import os
from openai import OpenAI

from .base import LLMProvider, LLMResponse


class XAIProvider(LLMProvider):
    """xAI Grok provider (uses OpenAI-compatible API)."""

    DEFAULT_MODEL = "grok-2-latest"

    AVAILABLE_MODELS = [
        "grok-2-latest",
        "grok-2",
        "grok-beta",
    ]

    def __init__(self, api_key: str | None = None):
        """
        Initialize xAI provider.

        Args:
            api_key: xAI API key. If None, reads from GROK_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("GROK_API_KEY")
        if not self.api_key:
            raise ValueError("xAI API key not found. Set GROK_API_KEY env var.")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1",
        )

    @property
    def name(self) -> str:
        return "xai"

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

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return LLMResponse(
            text=response.choices[0].message.content.strip(),
            model=model,
            provider=self.name,
        )

    def list_models(self) -> list[str]:
        return self.AVAILABLE_MODELS
