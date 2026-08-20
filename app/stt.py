"""Sarvam fast-streaming speech-to-text boundary."""

from __future__ import annotations

from collections.abc import AsyncIterable
from dataclasses import dataclass

from .schemas import TranscriptionResult


@dataclass(frozen=True, slots=True)
class SarvamSTTSettings:
    api_key: str
    language_code: str
    connect_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("Sarvam API key cannot be empty")
        if not self.language_code:
            raise ValueError("Sarvam language code cannot be empty")
        if self.connect_timeout_seconds <= 0:
            raise ValueError("Connect timeout must be positive")


class SarvamStreamingSTT:
    """Forward client audio to Sarvam fast mode over a retryable WebSocket."""

    def __init__(self, settings: SarvamSTTSettings) -> None:
        self.settings = settings

    async def transcribe(
        self,
        audio_chunks: AsyncIterable[bytes],
    ) -> TranscriptionResult:
        """Return only a real final transcript; never substitute mock text."""

        raise NotImplementedError(
            "The Sarvam fast-mode WebSocket client is not implemented yet"
        )
