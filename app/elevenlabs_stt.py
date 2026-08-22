"""Cost-guarded ElevenLabs Scribe v2 Realtime speech-to-text boundary.

This module intentionally has no provider fallback.  It issues short-lived
Scribe tokens only after checking the documented Free subscription fields and
streams mono 16-bit little-endian PCM at 16 kHz over ElevenLabs' realtime
WebSocket.  The PAYG/Auto Top Up state isn't exposed by the subscription API,
so callers must also make the explicit acknowledgement represented by
``FREE_TIER_ACKNOWLEDGEMENT`` after checking the ElevenLabs dashboard.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
import threading
import time
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Self
from urllib.parse import urlencode, urlsplit

import httpx
import websockets
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException

from .schemas import TranscriptionResult

FREE_TIER_ACKNOWLEDGEMENT = "I_CONFIRM_NO_PAYG_OR_AUTO_TOP_UP"
_REALTIME_ENDPOINT = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
_TOKEN_ENDPOINT = "https://api.elevenlabs.io/v1/single-use-token/realtime_scribe"
_SUBSCRIPTION_ENDPOINT = "https://api.elevenlabs.io/v1/user/subscription"
_PCM_BYTES_PER_SECOND = 16_000 * 2

PartialCallback = Callable[[str], Awaitable[None] | None]
SettledCallback = Callable[[str], Awaitable[None] | None]
OriginAuthorizer = Callable[[str | None], bool]


class ElevenLabsSTTError(RuntimeError):
    """Base class whose messages are safe to return across an API boundary."""


class ElevenLabsConfigurationError(ElevenLabsSTTError):
    """Invalid local configuration; never retry."""


class ElevenLabsBillingSafetyError(ElevenLabsSTTError):
    """Free-only billing assertions could not be established; never retry."""


class ElevenLabsAuthenticationError(ElevenLabsSTTError):
    """Provider authentication or terms failure; never retry."""


class ElevenLabsQuotaError(ElevenLabsSTTError):
    """Included quota is exhausted; never retry or switch to a paid provider."""


class ElevenLabsOriginError(ElevenLabsSTTError):
    """A token/session request did not pass the origin policy."""


class ElevenLabsLimitError(ElevenLabsSTTError):
    """A local cost or concurrency ceiling was reached."""


class ElevenLabsAudioError(ElevenLabsSTTError):
    """Invalid or failed caller-owned audio source; never retry."""


class ElevenLabsProtocolError(ElevenLabsSTTError):
    """Unexpected provider data which cannot safely be used."""


class ElevenLabsTransientError(ElevenLabsSTTError):
    """A sanitized provider/network failure which may be retried."""


@dataclass(frozen=True, slots=True)
class ElevenLabsSTTSettings:
    """Settings for a free-tier-only Scribe v2 Realtime session."""

    api_key: str = field(repr=False)
    payg_and_auto_top_up_disabled_acknowledged: bool = False
    language_code: str = "eng"
    secondary_languages: tuple[str, ...] = ()
    allowed_origins: tuple[str, ...] = ()
    require_origin: bool = False
    model: str = "scribe_v2_realtime"
    realtime_endpoint: str = _REALTIME_ENDPOINT
    token_endpoint: str = _TOKEN_ENDPOINT
    subscription_endpoint: str = _SUBSCRIPTION_ENDPOINT
    sample_rate: int = 16_000
    audio_format: str = "pcm_16000"
    max_audio_seconds: float = 20.0
    max_concurrent_sessions: int = 2
    daily_session_cap: int = 100
    daily_token_cap: int = 110
    connect_timeout_seconds: float = 5.0
    final_timeout_seconds: float = 10.0
    http_timeout_seconds: float = 5.0
    preflight_ttl_seconds: float = 60.0
    max_attempts: int = 3
    retry_min_seconds: float = 0.25
    retry_max_seconds: float = 2.0

    def __post_init__(self) -> None:
        if not self.api_key or not self.api_key.strip():
            raise ElevenLabsConfigurationError(
                "The ElevenLabs API key is not configured"
            )
        if self.model != "scribe_v2_realtime":
            raise ElevenLabsConfigurationError(
                "Only the realtime Scribe v2 model is permitted"
            )
        if self.realtime_endpoint != _REALTIME_ENDPOINT:
            raise ElevenLabsConfigurationError(
                "Only the official ElevenLabs realtime endpoint is permitted"
            )
        if self.token_endpoint != _TOKEN_ENDPOINT:
            raise ElevenLabsConfigurationError(
                "Only the official ElevenLabs token endpoint is permitted"
            )
        if self.subscription_endpoint != _SUBSCRIPTION_ENDPOINT:
            raise ElevenLabsConfigurationError(
                "Only the official ElevenLabs subscription endpoint is permitted"
            )
        if self.sample_rate != 16_000 or self.audio_format != "pcm_16000":
            raise ElevenLabsConfigurationError(
                "ElevenLabs demo audio must be mono PCM16 at 16000 Hz"
            )
        _validate_language_code(self.language_code)
        if len(self.secondary_languages) > 8:
            raise ElevenLabsConfigurationError(
                "At most eight secondary languages may be configured"
            )
        if self.language_code in self.secondary_languages:
            raise ElevenLabsConfigurationError(
                "The primary language cannot also be a secondary language"
            )
        if len(set(self.secondary_languages)) != len(self.secondary_languages):
            raise ElevenLabsConfigurationError(
                "Secondary language codes must be unique"
            )
        for language in self.secondary_languages:
            _validate_language_code(language)
        if not 0 < self.max_audio_seconds <= 120:
            raise ElevenLabsConfigurationError(
                "max_audio_seconds must be between 0 and 120"
            )
        # ElevenLabs currently documents six concurrent Realtime STT streams
        # for Free.  A local ceiling may be stricter, never looser.
        if not 1 <= self.max_concurrent_sessions <= 6:
            raise ElevenLabsConfigurationError(
                "max_concurrent_sessions must be between 1 and 6"
            )
        if self.daily_session_cap <= 0 or self.daily_token_cap <= 0:
            raise ElevenLabsConfigurationError("Daily caps must be positive")
        if self.daily_token_cap < self.daily_session_cap:
            raise ElevenLabsConfigurationError(
                "daily_token_cap must cover at least daily_session_cap"
            )
        if (
            min(
                self.connect_timeout_seconds,
                self.final_timeout_seconds,
                self.http_timeout_seconds,
            )
            <= 0
        ):
            raise ElevenLabsConfigurationError("Timeouts must be positive")
        if self.preflight_ttl_seconds < 0:
            raise ElevenLabsConfigurationError(
                "preflight_ttl_seconds cannot be negative"
            )
        if self.max_attempts <= 0:
            raise ElevenLabsConfigurationError("max_attempts must be positive")
        if self.retry_min_seconds < 0:
            raise ElevenLabsConfigurationError("retry_min_seconds cannot be negative")
        if self.retry_max_seconds < self.retry_min_seconds:
            raise ElevenLabsConfigurationError(
                "retry_max_seconds cannot be below retry_min_seconds"
            )
        for origin in self.allowed_origins:
            _validate_configured_origin(origin)
        if len(set(self.allowed_origins)) != len(self.allowed_origins):
            raise ElevenLabsConfigurationError("Allowed origins must be unique")

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        **overrides: Any,
    ) -> ElevenLabsSTTSettings:
        """Load secrets and the explicit free-tier operator acknowledgement."""

        source = os.environ if environ is None else environ
        acknowledgement = source.get("ELEVENLABS_FREE_TIER_ACKNOWLEDGEMENT", "")
        values: dict[str, Any] = {
            "api_key": source.get("ELEVENLABS_API_KEY", ""),
            "payg_and_auto_top_up_disabled_acknowledged": (
                acknowledgement == FREE_TIER_ACKNOWLEDGEMENT
            ),
        }
        values.update(overrides)
        return cls(**values)


def _validate_language_code(language_code: str) -> None:
    code = language_code.strip()
    if not 2 <= len(code) <= 16 or any(
        not (character.isalpha() or character == "-") for character in code
    ):
        raise ElevenLabsConfigurationError(
            "Language codes must be ISO-639-style alphabetic codes"
        )


def _validate_configured_origin(origin: str) -> None:
    parsed = urlsplit(origin)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or origin.endswith("/")
    ):
        raise ElevenLabsConfigurationError(
            "Allowed origins must be exact scheme/host/port origins"
        )


class _ProcessSessionBudget:
    """UTC-daily ceilings shared by every provider instance in this process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._day: date = datetime.now(UTC).date()
        self._tokens = 0
        self._sessions = 0
        self._active_sessions = 0

    def _roll_day(self) -> None:
        today = datetime.now(UTC).date()
        if today != self._day:
            self._day = today
            self._tokens = 0
            self._sessions = 0

    def reserve_token(self, daily_cap: int) -> None:
        with self._lock:
            self._roll_day()
            if self._tokens >= daily_cap:
                raise ElevenLabsLimitError(
                    "The local daily ElevenLabs token cap has been reached"
                )
            self._tokens += 1

    def start_session(self, daily_cap: int, concurrent_cap: int) -> None:
        with self._lock:
            self._roll_day()
            if self._sessions >= daily_cap:
                raise ElevenLabsLimitError(
                    "The local daily ElevenLabs session cap has been reached"
                )
            if self._active_sessions >= concurrent_cap:
                raise ElevenLabsLimitError(
                    "The local concurrent ElevenLabs session cap has been reached"
                )
            self._sessions += 1
            self._active_sessions += 1

    def finish_session(self) -> None:
        with self._lock:
            self._active_sessions = max(0, self._active_sessions - 1)

    def reset_for_tests(self) -> None:
        with self._lock:
            self._day = datetime.now(UTC).date()
            self._tokens = 0
            self._sessions = 0
            self._active_sessions = 0


_PROCESS_BUDGET = _ProcessSessionBudget()


def _reset_process_budget_for_tests() -> None:
    """Reset process counters for isolated unit tests only."""

    _PROCESS_BUDGET.reset_for_tests()


def _authorize_origin(
    settings: ElevenLabsSTTSettings,
    origin: str | None,
    authorizer: OriginAuthorizer | None,
) -> None:
    if origin is not None:
        try:
            _validate_configured_origin(origin)
        except ElevenLabsConfigurationError:
            raise ElevenLabsOriginError("The request origin is not permitted") from None

    if settings.require_origin and origin is None:
        raise ElevenLabsOriginError("An allowlisted request origin is required")
    if origin is not None and origin not in settings.allowed_origins:
        raise ElevenLabsOriginError("The request origin is not permitted")
    if authorizer is not None:
        try:
            accepted = authorizer(origin)
        # A policy hook is application-owned and may raise any regular error;
        # its details must not cross the public service boundary.
        except Exception:  # noqa: BLE001
            raise ElevenLabsOriginError(
                "The request origin could not be authorized"
            ) from None
        if not accepted:
            raise ElevenLabsOriginError("The request origin is not permitted")


def _safe_error_code(response: Any) -> str | None:
    try:
        payload = response.json()
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    detail = payload.get("detail")
    containers = [payload]
    if isinstance(detail, dict):
        containers.insert(0, detail)
    for container in containers:
        for key in ("status", "code", "type"):
            value = container.get(key)
            if isinstance(value, str):
                return value.strip().lower()
    return None


def _response_payload(response: Any) -> dict[str, Any]:
    try:
        payload = response.json()
    except (json.JSONDecodeError, TypeError, ValueError):
        raise ElevenLabsProtocolError(
            "ElevenLabs returned an invalid control-plane response"
        ) from None
    if not isinstance(payload, dict):
        raise ElevenLabsProtocolError(
            "ElevenLabs returned an invalid control-plane response"
        )
    return payload


def _raise_for_control_response(response: Any) -> None:
    status_code = int(getattr(response, "status_code", 0))
    if 200 <= status_code < 300:
        return
    code = _safe_error_code(response)
    if (
        code
        in {
            "quota_exceeded",
            "insufficient_quota",
            "max_character_limit_exceeded",
            "payment_required",
        }
        or status_code == 402
    ):
        raise ElevenLabsQuotaError(
            "The included ElevenLabs quota is unavailable or exhausted"
        )
    if code in {"invalid_api_key", "authentication_error", "auth_error"} or (
        status_code in {401, 403}
    ):
        raise ElevenLabsAuthenticationError(
            "ElevenLabs authentication or account authorization failed"
        )
    if status_code in {408, 425, 429} or 500 <= status_code <= 599:
        raise ElevenLabsTransientError(
            "The ElevenLabs control plane is temporarily unavailable"
        )
    raise ElevenLabsProtocolError("ElevenLabs rejected the control-plane request")


def _assert_free_subscription(payload: dict[str, Any]) -> None:
    """Require all documented billing-safety fields; absence fails closed."""

    try:
        amount = Decimal(str(payload["current_overage"]["amount"]))
    except (KeyError, TypeError, InvalidOperation):
        raise ElevenLabsBillingSafetyError(
            "The ElevenLabs subscription response could not prove zero overage"
        ) from None

    extension = payload.get("max_credit_limit_extension")
    extension_is_numeric_zero = type(extension) in {int, float} and extension == 0
    safe = (
        payload.get("tier") == "free"
        and payload.get("status") == "active"
        and extension_is_numeric_zero
        and payload.get("can_extend_character_limit") is False
        and payload.get("allowed_to_extend_character_limit") is False
        and amount == 0
        and payload.get("has_open_invoices") is False
        and payload.get("open_invoices") == []
        and payload.get("next_invoice") is None
    )
    if not safe:
        raise ElevenLabsBillingSafetyError(
            "ElevenLabs Free-only billing checks did not pass"
        )


class ElevenLabsTokenBroker:
    """Issue single-use Scribe tokens without exposing the API key to clients."""

    def __init__(
        self,
        settings: ElevenLabsSTTSettings,
        *,
        http_client: httpx.AsyncClient | None = None,
        origin_authorizer: OriginAuthorizer | None = None,
    ) -> None:
        self.settings = settings
        self._client = http_client or httpx.AsyncClient(
            timeout=settings.http_timeout_seconds
        )
        self._owns_client = http_client is None
        self._origin_authorizer = origin_authorizer
        self._preflight_lock = asyncio.Lock()
        self._preflight_checked_at: float | None = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def ensure_free_tier(self, *, force: bool = False) -> None:
        settings = self.settings
        if not settings.payg_and_auto_top_up_disabled_acknowledged:
            raise ElevenLabsBillingSafetyError(
                "PAYG and Auto Top Up must be manually disabled and acknowledged"
            )

        now = time.perf_counter()
        if (
            not force
            and self._preflight_checked_at is not None
            and now - self._preflight_checked_at <= settings.preflight_ttl_seconds
        ):
            return

        async with self._preflight_lock:
            now = time.perf_counter()
            if (
                not force
                and self._preflight_checked_at is not None
                and now - self._preflight_checked_at <= settings.preflight_ttl_seconds
            ):
                return
            response = await self._http_get(settings.subscription_endpoint)
            _raise_for_control_response(response)
            _assert_free_subscription(_response_payload(response))
            self._preflight_checked_at = time.perf_counter()

    async def issue_token(self, *, origin: str | None = None) -> str:
        """Return one short-lived token after origin and billing checks."""

        self.authorize_origin(origin)
        await self.ensure_free_tier()
        _PROCESS_BUDGET.reserve_token(self.settings.daily_token_cap)
        response = await self._http_post(self.settings.token_endpoint)
        _raise_for_control_response(response)
        payload = _response_payload(response)
        token = payload.get("token")
        if not isinstance(token, str) or not token.strip():
            raise ElevenLabsProtocolError(
                "ElevenLabs did not return a usable single-use token"
            )
        return token

    def authorize_origin(self, origin: str | None) -> None:
        """Apply the exact allowlist and optional application policy hook."""

        _authorize_origin(self.settings, origin, self._origin_authorizer)

    async def _http_get(self, url: str) -> Any:
        try:
            return await self._client.get(
                url,
                headers={"xi-api-key": self.settings.api_key},
                timeout=self.settings.http_timeout_seconds,
            )
        except (httpx.TimeoutException, httpx.TransportError):
            raise ElevenLabsTransientError(
                "The ElevenLabs control plane could not be reached"
            ) from None

    async def _http_post(self, url: str) -> Any:
        try:
            return await self._client.post(
                url,
                headers={"xi-api-key": self.settings.api_key},
                timeout=self.settings.http_timeout_seconds,
            )
        except (httpx.TimeoutException, httpx.TransportError):
            raise ElevenLabsTransientError(
                "The ElevenLabs control plane could not be reached"
            ) from None


@dataclass(slots=True)
class _TimingState:
    call_started: float
    connected_at: float | None = None
    first_audio_at: float | None = None
    audio_finished_at: float | None = None
    first_partial_at: float | None = None
    final_at: float | None = None
    committed_at: float | None = None


class _ReplayablePCM16:
    """Cache consumed source audio so a transient retry has the full prefix."""

    def __init__(self, source: AsyncIterable[bytes], max_bytes: int) -> None:
        self._source = source.__aiter__()
        self._max_bytes = max_bytes
        self._cache: list[bytes] = []
        self._total_bytes = 0
        self._exhausted = False

    @property
    def audio_duration_ms(self) -> float:
        """Duration of unique source audio, excluding bytes replayed on retry."""

        return self._total_bytes / _PCM_BYTES_PER_SECOND * 1000

    async def for_attempt(self) -> AsyncIterable[bytes]:
        for chunk in self._cache:
            yield chunk

        while not self._exhausted:
            try:
                chunk = await anext(self._source)
            except StopAsyncIteration:
                self._exhausted = True
                break
            # The application owns this iterator; sanitize arbitrary source
            # failures while allowing cancellation (a BaseException) through.
            except Exception:  # noqa: BLE001
                raise ElevenLabsAudioError("The audio source failed") from None
            if not isinstance(chunk, bytes):
                raise ElevenLabsAudioError("Audio chunks must be bytes")
            if not chunk:
                continue
            if len(chunk) % 2:
                raise ElevenLabsAudioError(
                    "PCM16 audio chunks must contain complete 16-bit samples"
                )
            self._total_bytes += len(chunk)
            if self._total_bytes > self._max_bytes:
                raise ElevenLabsAudioError(
                    "The audio stream exceeds the configured duration limit"
                )
            self._cache.append(chunk)
            yield chunk


def _is_retryable(exception: BaseException) -> bool:
    return isinstance(exception, ElevenLabsTransientError)


def _event_payload(raw_message: str | bytes) -> dict[str, Any]:
    try:
        if isinstance(raw_message, bytes):
            raw_message = raw_message.decode("utf-8")
        payload = json.loads(raw_message)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ElevenLabsProtocolError(
            "ElevenLabs returned an invalid realtime event"
        ) from None
    if not isinstance(payload, dict):
        raise ElevenLabsProtocolError("ElevenLabs returned an invalid realtime event")
    return payload


def _raise_for_realtime_error(message_type: str) -> None:
    normalized = message_type.removeprefix("scribe_")
    if normalized == "quota_exceeded":
        raise ElevenLabsQuotaError(
            "The included ElevenLabs quota is unavailable or exhausted"
        )
    if normalized in {"auth_error", "unaccepted_terms"}:
        raise ElevenLabsAuthenticationError(
            "ElevenLabs authentication or account authorization failed"
        )
    if normalized in {
        "input_error",
        "invalid_request",
        "chunk_size_exceeded",
        "insufficient_audio_activity",
        "session_time_limit_exceeded",
    }:
        raise ElevenLabsProtocolError("ElevenLabs rejected the realtime request")
    if normalized in {
        "rate_limited",
        "throttled",
        "commit_throttled",
        "queue_overflow",
        "resource_exhausted",
        "transcriber_error",
        "error",
    }:
        raise ElevenLabsTransientError(
            "The ElevenLabs realtime service is temporarily unavailable"
        )


@asynccontextmanager
async def _session_slot(settings: ElevenLabsSTTSettings):
    _PROCESS_BUDGET.start_session(
        settings.daily_session_cap,
        settings.max_concurrent_sessions,
    )
    try:
        yield
    finally:
        _PROCESS_BUDGET.finish_session()


class ElevenLabsStreamingSTT:
    """Stream one explicitly delimited voice turn through Scribe v2 Realtime."""

    def __init__(
        self,
        settings: ElevenLabsSTTSettings,
        *,
        http_client: httpx.AsyncClient | None = None,
        origin_authorizer: OriginAuthorizer | None = None,
    ) -> None:
        self.settings = settings
        self.token_broker = ElevenLabsTokenBroker(
            settings,
            http_client=http_client,
            origin_authorizer=origin_authorizer,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self.token_broker.aclose()

    def _url(self, token: str) -> str:
        settings = self.settings
        query: list[tuple[str, str]] = [
            ("model_id", settings.model),
            ("token", token),
            ("audio_format", settings.audio_format),
            ("language_code", settings.language_code),
            ("commit_strategy", "manual"),
            ("include_timestamps", "false"),
            ("include_language_detection", "false"),
            # Preserve fillers, false starts, and spoken forms for query fidelity.
            ("no_verbatim", "false"),
        ]
        query.extend(
            ("secondary_languages", language)
            for language in settings.secondary_languages
        )
        return f"{settings.realtime_endpoint}?{urlencode(query)}"

    async def transcribe(
        self,
        audio_chunks: AsyncIterable[bytes],
        *,
        on_partial: PartialCallback | None = None,
        on_settled: SettledCallback | None = None,
        origin: str | None = None,
    ) -> TranscriptionResult:
        """Return only a committed transcript from an approved Free session.

        ``on_settled`` is an optimization hook, not an authorization boundary.
        It may prepare local work after ElevenLabs emits ``final_transcript``,
        but callers must not publish or reuse that work unless the subsequently
        committed transcript matches exactly.  This class itself rejects a
        final/committed mismatch.
        """

        self.token_broker.authorize_origin(origin)
        timing = _TimingState(call_started=time.perf_counter())
        max_bytes = int(self.settings.max_audio_seconds * _PCM_BYTES_PER_SECOND)
        replayable = _ReplayablePCM16(audio_chunks, max_bytes=max_bytes)
        partials: list[str] = []
        language_code = self.settings.language_code

        async with _session_slot(self.settings):
            retrying = AsyncRetrying(
                stop=stop_after_attempt(self.settings.max_attempts),
                wait=wait_exponential(
                    multiplier=max(self.settings.retry_min_seconds, 0.001),
                    min=self.settings.retry_min_seconds,
                    max=self.settings.retry_max_seconds,
                ),
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
                        on_settled=on_settled,
                        origin=origin,
                    )
                    if detected_language:
                        language_code = detected_language

        if timing.connected_at is None or timing.committed_at is None:
            raise ElevenLabsProtocolError(
                "ElevenLabs closed without a committed transcript"
            )
        if timing.audio_finished_at is None:
            raise ElevenLabsAudioError("The audio stream contained no samples")

        finished_at = time.perf_counter()
        final_at = timing.final_at or timing.committed_at
        return TranscriptionResult(
            provider="elevenlabs",
            model=self.settings.model,
            transcript=transcript,
            language_code=language_code,
            is_final=True,
            partial_transcripts=partials,
            connection_ms=max(0.0, (timing.connected_at - timing.call_started) * 1000),
            time_to_first_partial_ms=(
                max(
                    0.0,
                    (timing.first_partial_at - timing.call_started) * 1000,
                )
                if timing.first_partial_at is not None
                else None
            ),
            time_to_final_transcript_ms=max(
                0.0, (final_at - timing.call_started) * 1000
            ),
            # A committed transcript is the correctness boundary exposed to RAG.
            final_after_audio_end_ms=max(
                0.0, (timing.committed_at - timing.audio_finished_at) * 1000
            ),
            first_audio_to_final_ms=(
                max(0.0, (timing.committed_at - timing.first_audio_at) * 1000)
                if timing.first_audio_at is not None
                else None
            ),
            audio_duration_ms=replayable.audio_duration_ms,
            total_ms=max(0.0, (finished_at - timing.call_started) * 1000),
        )

    async def _transcribe_once(
        self,
        *,
        replayable: _ReplayablePCM16,
        timing: _TimingState,
        partials: list[str],
        on_partial: PartialCallback | None,
        on_settled: SettledCallback | None,
        origin: str | None,
    ) -> tuple[str, str | None]:
        settings = self.settings
        token = await self.token_broker.issue_token(origin=origin)
        url = self._url(token)

        try:
            async with websockets.connect(
                url,
                open_timeout=settings.connect_timeout_seconds,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=2,
                max_size=1_048_576,
            ) as socket:
                timing.connected_at = time.perf_counter()

                async def send_audio() -> None:
                    sent_any = False
                    async for chunk in replayable.for_attempt():
                        sent_any = True
                        if timing.first_audio_at is None:
                            timing.first_audio_at = time.perf_counter()
                        await socket.send(
                            json.dumps(
                                {
                                    "message_type": "input_audio_chunk",
                                    "audio_base_64": base64.b64encode(chunk).decode(
                                        "ascii"
                                    ),
                                    "sample_rate": settings.sample_rate,
                                },
                                separators=(",", ":"),
                            )
                        )
                    if not sent_any:
                        raise ElevenLabsAudioError(
                            "The audio stream contained no samples"
                        )
                    timing.audio_finished_at = time.perf_counter()
                    # The official SDK's manual commit is represented by a final
                    # input_audio_chunk with commit=true.  No VAD silence is paid.
                    await socket.send(
                        json.dumps(
                            {
                                "message_type": "input_audio_chunk",
                                "audio_base_64": "",
                                "commit": True,
                                "sample_rate": settings.sample_rate,
                            },
                            separators=(",", ":"),
                        )
                    )

                async def receive_events() -> tuple[str, str | None]:
                    settled_text: str | None = None
                    detected_language: str | None = None
                    while True:
                        payload = _event_payload(await socket.recv())
                        message_type = str(
                            payload.get("message_type") or payload.get("event") or ""
                        ).strip()
                        _raise_for_realtime_error(message_type)

                        if message_type == "partial_transcript":
                            text = str(payload.get("text") or "").strip()
                            if not text:
                                continue
                            if timing.first_partial_at is None:
                                timing.first_partial_at = time.perf_counter()
                            if not partials or partials[-1] != text:
                                partials.append(text)
                                if on_partial is not None:
                                    callback_result = on_partial(text)
                                    if inspect.isawaitable(callback_result):
                                        await callback_result
                        elif message_type in {
                            "final_transcript",
                            "final_transcript_with_timestamps",
                        }:
                            text = str(payload.get("text") or "").strip()
                            if not text:
                                raise ElevenLabsProtocolError(
                                    "ElevenLabs returned an empty final transcript"
                                )
                            settled_text = text
                            timing.final_at = time.perf_counter()
                            if on_settled is not None:
                                callback_result = on_settled(text)
                                if inspect.isawaitable(callback_result):
                                    await callback_result
                            language = payload.get("language_code")
                            if isinstance(language, str) and language.strip():
                                detected_language = language.strip()
                        elif message_type in {
                            "committed_transcript",
                            "committed_transcript_with_timestamps",
                        }:
                            text = str(payload.get("text") or "").strip()
                            if not text:
                                raise ElevenLabsProtocolError(
                                    "ElevenLabs returned an empty committed transcript"
                                )
                            if settled_text is not None and text != settled_text:
                                raise ElevenLabsProtocolError(
                                    "ElevenLabs final and committed transcripts differ"
                                )
                            timing.committed_at = time.perf_counter()
                            language = payload.get("language_code")
                            if isinstance(language, str) and language.strip():
                                detected_language = language.strip()
                            return text, detected_language

                send_task = asyncio.create_task(send_audio())
                receive_task = asyncio.create_task(receive_events())
                try:
                    # The configured final timeout starts at true audio EOF. It
                    # must not cap the spoken turn itself. At the same time, a
                    # provider-side error should stop a still-open input stream
                    # promptly instead of waiting for the caller to finish.
                    done, _ = await asyncio.wait(
                        (send_task, receive_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if receive_task in done:
                        received = await receive_task
                        await send_task
                    else:
                        await send_task
                        async with asyncio.timeout(settings.final_timeout_seconds):
                            received = await receive_task
                    if (
                        timing.committed_at is not None
                        and timing.audio_finished_at is not None
                        and timing.committed_at < timing.audio_finished_at
                    ):
                        raise ElevenLabsProtocolError(
                            "ElevenLabs committed before the explicit audio boundary"
                        )
                finally:
                    for task in (send_task, receive_task):
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(
                        send_task,
                        receive_task,
                        return_exceptions=True,
                    )
                return received
        except (
            ElevenLabsAudioError,
            ElevenLabsAuthenticationError,
            ElevenLabsBillingSafetyError,
            ElevenLabsLimitError,
            ElevenLabsOriginError,
            ElevenLabsProtocolError,
            ElevenLabsQuotaError,
            ElevenLabsTransientError,
        ):
            raise
        except TimeoutError:
            raise ElevenLabsTransientError(
                "ElevenLabs did not commit a transcript before the timeout"
            ) from None
        except InvalidStatus as exception:
            status_code = int(getattr(exception.response, "status_code", 0))
            if status_code in {401, 403}:
                raise ElevenLabsAuthenticationError(
                    "ElevenLabs realtime authentication failed"
                ) from None
            if status_code in {408, 425, 429} or 500 <= status_code <= 599:
                raise ElevenLabsTransientError(
                    "The ElevenLabs realtime connection is temporarily unavailable"
                ) from None
            raise ElevenLabsProtocolError(
                "ElevenLabs rejected the realtime connection"
            ) from None
        except ConnectionClosed as exception:
            if exception.code in {1006, 1011, 1012, 1013}:
                raise ElevenLabsTransientError(
                    "The ElevenLabs realtime connection ended unexpectedly"
                ) from None
            raise ElevenLabsProtocolError(
                "ElevenLabs closed before committing a transcript"
            ) from None
        except (OSError, WebSocketException):
            raise ElevenLabsTransientError(
                "The ElevenLabs realtime connection failed"
            ) from None
