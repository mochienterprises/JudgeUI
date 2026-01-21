from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    text: str
    model: str
    provider: str


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'anthropic', 'openai')."""
        pass

    @abstractmethod
    def call(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 2000,
        model: str | None = None,
    ) -> LLMResponse:
        """
        Make a call to the LLM.

        Args:
            prompt: The user prompt to send
            system_prompt: Optional system prompt for context/instructions
            temperature: Sampling temperature (0.0 = deterministic, 1.0+ = creative)
            max_tokens: Maximum tokens in response
            model: Model ID to use (uses provider default if None)

        Returns:
            LLMResponse with the generated text and metadata
        """
        pass

    @abstractmethod
    def list_models(self) -> list[str]:
        """List available models for this provider."""
        pass
