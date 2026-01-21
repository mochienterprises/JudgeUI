import os
from anthropic import Anthropic

from .base import LLMProvider, LLMResponse


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

    AVAILABLE_MODELS = [
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
    ]

    def __init__(self, api_key: str | None = None):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY env var.")
        self.client = Anthropic(api_key=self.api_key)

    @property
    def name(self) -> str:
        return "anthropic"

    def call(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 2000,
        model: str | None = None,
    ) -> LLMResponse:
        model = model or self.DEFAULT_MODEL

        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Anthropic uses 'system' as a top-level parameter
        if system_prompt:
            kwargs["system"] = system_prompt

        # Only add temperature if not 1.0 (Anthropic's default)
        if temperature != 1.0:
            kwargs["temperature"] = temperature

        response = self.client.messages.create(**kwargs)

        return LLMResponse(
            text=response.content[0].text.strip(),
            model=model,
            provider=self.name,
        )

    def list_models(self) -> list[str]:
        return self.AVAILABLE_MODELS
