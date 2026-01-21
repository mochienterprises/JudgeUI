import os
from google import genai
from google.genai import types

from .base import LLMProvider, LLMResponse


class GoogleProvider(LLMProvider):
    """Google Gemini provider using the new google-genai SDK."""

    DEFAULT_MODEL = "gemini-2.0-flash"

    AVAILABLE_MODELS = [
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-pro-preview-05-06",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
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
        self.client = genai.Client(api_key=self.api_key)

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

        # Build generation config
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="OFF",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="OFF",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="OFF",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="OFF",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_CIVIC_INTEGRITY",
                    threshold="OFF",
                ),
            ],
        )

        if system_prompt:
            config.system_instruction = system_prompt

        response = self.client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )

        # Handle response
        if not response.text:
            # Build error details
            error_details = []

            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                error_details.append(f"Prompt feedback: {response.prompt_feedback}")

            if response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                    error_details.append(f"Finish reason: {candidate.finish_reason}")

                if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                    ratings = [f"{r.category}: {r.probability}" for r in candidate.safety_ratings]
                    error_details.append(f"Safety ratings: {', '.join(ratings)}")

            error_msg = "; ".join(error_details) if error_details else "No content returned"
            raise ValueError(f"Gemini response failed - {error_msg}")

        return LLMResponse(
            text=response.text.strip(),
            model=model_name,
            provider=self.name,
        )

    def list_models(self) -> list[str]:
        return self.AVAILABLE_MODELS
