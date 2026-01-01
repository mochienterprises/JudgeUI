import os
import google.generativeai as genai

from .base import LLMProvider, LLMResponse


class GoogleProvider(LLMProvider):
    """Google Gemini provider."""

    DEFAULT_MODEL = "gemini-2.0-flash-exp"

    AVAILABLE_MODELS = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash-exp",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
    ]

    def __init__(self, api_key: str | None = None):
        """
        Initialize Google provider.

        Args:
            api_key: Google API key. If None, reads from GEMINI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found. Set GEMINI_API_KEY env var.")
        genai.configure(api_key=self.api_key)

    @property
    def name(self) -> str:
        return "google"

    def call(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 2000,
        model: str | None = None,
    ) -> LLMResponse:
        model_name = model or self.DEFAULT_MODEL

        # Create model with generation config
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        model_instance = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            system_instruction=system_prompt if system_prompt else None,
        )

        response = model_instance.generate_content(prompt)

        # Handle cases where response is blocked by safety filters
        try:
            text = response.text.strip()
        except ValueError as e:
            # Check if blocked by safety filters
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    reason = str(candidate.finish_reason)
                    if 'SAFETY' in reason or candidate.finish_reason.value == 3:
                        raise ValueError(f"Response blocked by safety filter: {reason}")
            # Re-raise if we can't determine the issue
            raise ValueError(f"Could not extract response text: {e}")

        return LLMResponse(
            text=text,
            model=model_name,
            provider=self.name,
        )

    def list_models(self) -> list[str]:
        return self.AVAILABLE_MODELS
