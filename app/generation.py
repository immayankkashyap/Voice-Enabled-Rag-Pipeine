"""Structured Groq/Llama answer-generation boundary."""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import GenerationRequest, GenerationResult


@dataclass(frozen=True, slots=True)
class GroqGenerationSettings:
    api_key: str
    model: str
    max_output_tokens: int

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("Groq API key cannot be empty")
        if not self.model:
            raise ValueError("Groq model cannot be empty")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")


class GroqAnswerGenerator:
    """Generate only from validated retrieved context and return typed output."""

    def __init__(self, settings: GroqGenerationSettings) -> None:
        self.settings = settings

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Call Groq with retry/timing and validate its structured response."""

        raise NotImplementedError(
            "The real Groq structured-output client is not implemented"
        )
