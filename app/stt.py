"""Sarvam Realtime streaming speech-to-text boundary.

The newer Realtime API is used instead of Sarvam's legacy streaming endpoint:
only Realtime emits genuine ``transcript.partial`` events. Audio supplied to
this module must be mono raw PCM matching ``encoding`` and ``sample_rate``.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import time
from collections.abc import AsyncIterable, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlencode

import websockets
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from websockets.exceptions import ConnectionClosed

from .schemas import TranscriptionResult

SarvamMode = Literal["transcribe", "translate", "verbatim", "translit", "codemix"]
PartialCallback = Callable[[str], Awaitable[None] | None]


class SarvamSTTError(RuntimeError):
    """A non-retryable Sarvam protocol, authentication, or input error."""


class SarvamSTTTransientError(SarvamSTTError):
    """A retryable provider-side failure."""


class SarvamAudioSourceError(SarvamSTTError):
    """The caller's audio iterator failed or yielded an invalid value."""


@dataclass(frozen=True, slots=True)
class SarvamSTTSettings:
    api_key: str
    language_code: str = "en-IN"
    endpoint: str = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
    model: str = "saaras:v3-realtime"
    # ``verbatim`` remains available, but the measured reference utterance had
    # a repeatable word-boundary error in that mode. Normal transcription was
    # exact in 10/10 trials, so correctness takes precedence as the default.
    mode: SarvamMode = "transcribe"
    stream_type: Literal["fast", "balanced", "simulated"] = "fast"
    # ``transcribe`` receives one explicitly delimited turn, so manual mode can
    # send ``speech_end`` at iterator EOF instead of paying the default VAD wait.
    endpointing: Literal["vad", "manual"] = "manual"
    encoding: Literal["linear16", "linear32", "mulaw", "alaw"] = "linear16"
    sample_rate: int = 16_000
    vad_threshold: float = 0.3
    silence_duration_ms: int = 500
    min_speech_duration_ms: int = 250
    connect_timeout_seconds: float = 5.0
    final_timeout_seconds: float = 30.0
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("Sarvam API key cannot be empty")
        if not self.language_code:
            raise ValueError("Sarvam language code cannot be empty")
        if not self.endpoint.startswith("wss://"):
            raise ValueError("Sarvam endpoint must use wss://")
        if self.model != "saaras:v3-realtime":
            raise ValueError("The Realtime endpoint only supports saaras:v3-realtime")
        if self.sample_rate not in (8_000, 16_000):
            raise ValueError("Sarvam Realtime sample_rate must be 8000 or 16000")
        if not 0 <= self.vad_threshold <= 1:
            raise ValueError("VAD threshold must be between 0 and 1")
        if self.silence_duration_ms <= 0 or self.min_speech_duration_ms <= 0:
            raise ValueError("VAD durations must be positive")
        if self.connect_timeout_seconds <= 0 or self.final_timeout_seconds <= 0:
            raise ValueError("Timeouts must be positive")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")


@dataclass(slots=True)
class _TimingState:
    call_started: float
    connected_at: float | None = None
    first_audio_at: float | None = None
    audio_finished_at: float | None = None
    first_partial_at: float | None = None
    final_at: float | None = None


class _ReplayableAudio:
    """Cache consumed chunks so a failed WebSocket attempt can replay them."""

    def __init__(self, source: AsyncIterable[bytes]) -> None:
        self._source = source.__aiter__()
        self._cache: list[bytes] = []
        self._exhausted = False

    async def for_attempt(self) -> AsyncIterable[bytes]:
        for chunk in self._cache:
            yield chunk

        while not self._exhausted:
            try:
                chunk = await anext(self._source)
            except StopAsyncIteration:
                self._exhausted = True
                break
            except Exception as exc:
                raise SarvamAudioSourceError("The audio source failed") from exc

            if not isinstance(chunk, bytes):
                raise SarvamAudioSourceError(
                    f"Audio chunks must be bytes, received {type(chunk).__name__}"
                )
            if not chunk:
                continue
            self._cache.append(chunk)
            yield chunk


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, SarvamSTTTransientError):
        return True
    if isinstance(exc, (TimeoutError, OSError)):
        return True
    if isinstance(exc, ConnectionClosed):
        # 1003 is also used for invalid keys/quota and 4000 for bad parameters;
        # neither should be hammered with retries. 1008 inactivity and 1011
        # provider failures are safe to retry, as are abnormal closes.
        return exc.code in (1006, 1008, 1011)
    return False


def _event_payload(raw_message: str | bytes) -> dict[str, Any]:
    try:
        if isinstance(raw_message, bytes):
            raw_message = raw_message.decode("utf-8")
        payload = json.loads(raw_message)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SarvamSTTError("Sarvam returned a non-JSON WebSocket message") from exc
    if not isinstance(payload, dict):
        raise SarvamSTTError("Sarvam returned a non-object WebSocket message")
    return payload


class SarvamStreamingSTT:
    """Forward raw PCM to Sarvam and return a measured final transcript."""

    def __init__(self, settings: SarvamSTTSettings) -> None:
        self.settings = settings

    def _url(self) -> str:
        settings = self.settings
        query: dict[str, str | int | float] = {
            "language_code": settings.language_code,
            "model": settings.model,
            "mode": settings.mode,
            "stream_type": settings.stream_type,
            "endpointing": settings.endpointing,
            "encoding": settings.encoding,
            "sample_rate": settings.sample_rate,
        }
        if settings.endpointing == "vad":
            query.update(
                threshold=settings.vad_threshold,
                silence_duration_ms=settings.silence_duration_ms,
                min_speech_duration_ms=settings.min_speech_duration_ms,
            )
        return f"{settings.endpoint}?{urlencode(query)}"

    async def transcribe(
        self,
        audio_chunks: AsyncIterable[bytes],
        *,
        on_partial: PartialCallback | None = None,
    ) -> TranscriptionResult:
        """Return only a real final transcript; never substitute mock text.

        Network/provider failures are retried with exponential backoff. Consumed
        audio is cached and replayed on a new connection so a retry never returns
        a transcript with a silently missing prefix.
        """

        timing = _TimingState(call_started=time.perf_counter())
        replayable = _ReplayableAudio(audio_chunks)
        partials: list[str] = []
        language_code = self.settings.language_code

        retrying = AsyncRetrying(
            stop=stop_after_attempt(self.settings.max_attempts),
            wait=wait_exponential(multiplier=0.25, min=0.25, max=2),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                transcript, detected_language = await self._transcribe_once(
                    replayable=replayable,
                    timing=timing,
                    partials=partials,
                    on_partial=on_partial,
                )
                if detected_language:
                    language_code = detected_language

        if timing.connected_at is None or timing.final_at is None:
            raise SarvamSTTError("Sarvam closed without a final transcript")
        if timing.audio_finished_at is None:
            raise SarvamAudioSourceError("The audio stream contained no bytes")

        total_ms = (time.perf_counter() - timing.call_started) * 1000
        return TranscriptionResult(
            provider="sarvam",
            model=self.settings.model,
            transcript=transcript,
            language_code=language_code,
            is_final=True,
            partial_transcripts=partials,
            connection_ms=(timing.connected_at - timing.call_started) * 1000,
            time_to_first_partial_ms=(
                (timing.first_partial_at - timing.call_started) * 1000
                if timing.first_partial_at is not None
                else None
            ),
            time_to_final_transcript_ms=(timing.final_at - timing.call_started) * 1000,
            final_after_audio_end_ms=max(
                0.0, (timing.final_at - timing.audio_finished_at) * 1000
            ),
            first_audio_to_final_ms=(
                (timing.final_at - timing.first_audio_at) * 1000
                if timing.first_audio_at is not None
                else None
            ),
            audio_duration_ms=(
                (timing.audio_finished_at - timing.first_audio_at) * 1000
                if timing.first_audio_at is not None
                else None
            ),
            total_ms=total_ms,
        )

    async def _transcribe_once(
        self,
        *,
        replayable: _ReplayableAudio,
        timing: _TimingState,
        partials: list[str],
        on_partial: PartialCallback | None,
    ) -> tuple[str, str | None]:
        settings = self.settings
        async with websockets.connect(
            self._url(),
            additional_headers={"Api-Subscription-Key": settings.api_key},
            open_timeout=settings.connect_timeout_seconds,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
        ) as socket:
            timing.connected_at = time.perf_counter()

            async def send_audio() -> None:
                sent_any = False
                if settings.endpointing == "manual":
                    await socket.send('{"event":"speech_start"}')
                async for chunk in replayable.for_attempt():
                    sent_any = True
                    if timing.first_audio_at is None:
                        timing.first_audio_at = time.perf_counter()
                    message = {
                        "event": "audio_input",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                    await socket.send(json.dumps(message, separators=(",", ":")))
                if not sent_any:
                    raise SarvamAudioSourceError("The audio stream contained no bytes")
                timing.audio_finished_at = time.perf_counter()
                if settings.endpointing == "manual":
                    # Iterator EOF must mean a reliably detected turn boundary;
                    # callers must not emit ``speech_end`` at an arbitrary packet.
                    await socket.send('{"event":"speech_end"}')
                else:
                    await socket.send('{"event":"end"}')

            async def receive_events() -> tuple[str, str | None]:
                while True:
                    raw = await socket.recv()
                    payload = _event_payload(raw)
                    event = payload.get("event")
                    if event == "transcript.partial":
                        text = str(payload.get("text") or "").strip()
                        if text:
                            if timing.first_partial_at is None:
                                timing.first_partial_at = time.perf_counter()
                            if not partials or partials[-1] != text:
                                partials.append(text)
                            if on_partial is not None:
                                callback_result = on_partial(text)
                                if inspect.isawaitable(callback_result):
                                    await callback_result
                    elif event == "transcript.final":
                        text = str(payload.get("text") or "").strip()
                        if not text:
                            raise SarvamSTTError(
                                "Sarvam returned an empty final transcript"
                            )
                        timing.final_at = time.perf_counter()
                        language = payload.get("language")
                        return text, str(language) if language else None
                    elif event == "error":
                        code = payload.get("code", "unknown")
                        message = payload.get("message", "unspecified Sarvam error")
                        error_type = (
                            SarvamSTTError
                            if payload.get("is_fatal", False)
                            else SarvamSTTTransientError
                        )
                        raise error_type(f"Sarvam error {code}: {message}")

            try:
                async with asyncio.timeout(settings.final_timeout_seconds):
                    _, received = await asyncio.gather(send_audio(), receive_events())
            except TimeoutError as exc:
                raise TimeoutError(
                    f"No final Sarvam transcript within {settings.final_timeout_seconds}s"
                ) from exc
            if settings.endpointing == "manual":
                await socket.send('{"event":"end"}')
            return received
