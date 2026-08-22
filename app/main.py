"""FastAPI harness for the warmed, free-tier-gated voice RAG demo."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import secrets
import time
from collections import defaultdict, deque
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .elevenlabs_stt import (
    ElevenLabsStreamingSTT,
    ElevenLabsSTTError,
    ElevenLabsSTTSettings,
)
from .runtime import RuntimeLoadError, RuntimeServices, RuntimeSettings, load_runtime
from .schemas import (
    ErrorResponse,
    HealthResponse,
    RAGRequest,
    RAGResponse,
    VoicePipelineLatencies,
    VoiceRAGResponse,
)

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEMO_STATIC_DIR = (_PROJECT_ROOT / "static").resolve()
_DEMO_SECURITY_HEADERS = {
    "Cache-Control": "no-store, max-age=0",
    "Pragma": "no-cache",
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; "
        "connect-src 'self'; media-src 'self'; img-src 'self'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'; "
        "object-src 'none'"
    ),
    "Permissions-Policy": "microphone=(self)",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}


class _DemoStaticFiles(StaticFiles):
    """Serve only bundled demo assets with cache and browser controls."""

    async def get_response(self, path: str, scope: dict[str, Any]) -> Response:
        response = await super().get_response(path, scope)
        response.headers.update(_DEMO_SECURITY_HEADERS)
        return response


_ISO2_TO_ELEVENLABS = {
    "as": "asm",
    "bn": "ben",
    "gu": "guj",
    "hi": "hin",
    "kn": "kan",
    "ml": "mal",
    "mr": "mar",
    "ne": "nep",
    "or": "ori",
    "pa": "pan",
    "sa": "san",
    "ta": "tam",
    "te": "tel",
    "ur": "urd",
    "en": "eng",
}
_ELEVENLABS_TO_ISO2 = {value: key for key, value in _ISO2_TO_ELEVENLABS.items()}


class _VoiceProtocolError(RuntimeError):
    """Sanitized invalid client input on the voice WebSocket."""


@dataclass(frozen=True, slots=True)
class VoiceAccessSettings:
    """Fail-closed controls applied before any provider-owned operation."""

    demo_token: str = field(default="", repr=False)
    allowed_origins: tuple[str, ...] = ()
    require_origin: bool = True
    max_frame_bytes: int = 16_000
    sessions_per_window: int = 6
    rate_limit_window_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.demo_token and not 16 <= len(self.demo_token) <= 512:
            raise ValueError("demo_token must contain between 16 and 512 characters")
        if not 2 <= self.max_frame_bytes <= 64_000 or self.max_frame_bytes % 2:
            raise ValueError(
                "max_frame_bytes must be an even PCM16 byte count from 2 to 64000"
            )
        if self.sessions_per_window <= 0:
            raise ValueError("sessions_per_window must be positive")
        if (
            not math.isfinite(self.rate_limit_window_seconds)
            or self.rate_limit_window_seconds <= 0
        ):
            raise ValueError("rate_limit_window_seconds must be positive")

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> VoiceAccessSettings:
        source = os.environ if environ is None else environ
        origins = tuple(
            value.strip()
            for value in source.get("ELEVENLABS_ALLOWED_ORIGINS", "").split(",")
            if value.strip()
        )
        return cls(
            demo_token=source.get("VOICE_DEMO_TOKEN", ""),
            allowed_origins=origins,
            require_origin=(
                source.get("ELEVENLABS_REQUIRE_ORIGIN", "true").strip().lower()
                not in {"0", "false", "no", "off"}
            ),
            max_frame_bytes=int(source.get("VOICE_DEMO_MAX_FRAME_BYTES", "16000")),
            sessions_per_window=int(source.get("VOICE_DEMO_SESSIONS_PER_MINUTE", "6")),
            rate_limit_window_seconds=float(
                source.get("VOICE_DEMO_RATE_LIMIT_WINDOW_SECONDS", "60")
            ),
        )


def _load_voice_access_settings() -> tuple[VoiceAccessSettings, str | None]:
    """Keep text RAG available while invalid voice config fails closed."""

    try:
        return VoiceAccessSettings.from_environment(), None
    except (OverflowError, ValueError):
        return VoiceAccessSettings(), "VoiceAccessConfigurationError"


class _VoiceRateLimiter:
    """Small process-local sliding-window limiter for demo abuse containment."""

    def __init__(self) -> None:
        self._events: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def admit(
        self,
        namespace: str,
        identity: str,
        settings: VoiceAccessSettings,
    ) -> None:
        now = time.monotonic()
        cutoff = now - settings.rate_limit_window_seconds
        key = (namespace, identity)
        async with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= settings.sessions_per_window:
                raise _VoiceProtocolError("The demo voice rate limit was reached")
            events.append(now)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float_env(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _origins_env() -> tuple[str, ...]:
    return tuple(
        value.strip()
        for value in os.getenv("ELEVENLABS_ALLOWED_ORIGINS", "").split(",")
        if value.strip()
    )


def _stt_settings(language_code: str = "eng") -> ElevenLabsSTTSettings:
    return ElevenLabsSTTSettings.from_env(
        language_code=language_code,
        allowed_origins=_origins_env(),
        require_origin=_bool_env("ELEVENLABS_REQUIRE_ORIGIN", True),
        max_audio_seconds=_float_env("ELEVENLABS_DEMO_MAX_AUDIO_SECONDS", 15.0),
        max_concurrent_sessions=_int_env("ELEVENLABS_DEMO_MAX_CONCURRENT_SESSIONS", 1),
        daily_session_cap=_int_env("ELEVENLABS_DEMO_DAILY_SESSION_CAP", 20),
        daily_token_cap=_int_env("ELEVENLABS_DEMO_DAILY_TOKEN_CAP", 25),
    )


async def _close_stt_clients(application: FastAPI) -> None:
    clients = list(getattr(application.state, "stt_clients", {}).values())
    await asyncio.gather(
        *(client.aclose() for client in clients), return_exceptions=True
    )


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.lifespan_started = True
    application.state.runtime_settings = RuntimeSettings.from_environment()
    application.state.runtime = None
    application.state.runtime_error = None
    application.state.runtime_lock = asyncio.Lock()
    application.state.stt_clients = {}
    application.state.stt_lock = asyncio.Lock()
    (
        application.state.voice_access_settings,
        application.state.voice_error,
    ) = _load_voice_access_settings()
    application.state.voice_rate_limiter = _VoiceRateLimiter()

    if application.state.runtime_settings.preload:
        try:
            application.state.runtime = await load_runtime(
                application.state.runtime_settings
            )
        except Exception as exc:  # noqa: BLE001 - health remains available
            application.state.runtime_error = type(exc).__name__

    try:
        yield
    finally:
        await _close_stt_clients(application)
        application.state.lifespan_started = False


app = FastAPI(
    title="Voice-Enabled RAG Pipeline",
    version="0.2.0",
    description=(
        "ElevenLabs Free allowance → local Jina/FAISS retrieval → "
        "zero-cost extractive answer → deterministic guardrails"
    ),
    lifespan=lifespan,
)


@app.get("/demo", include_in_schema=False)
async def browser_demo() -> FileResponse:
    """Return the bundled operator-driven voice demo without caching secrets."""

    return FileResponse(
        _DEMO_STATIC_DIR / "index.html",
        media_type="text/html",
        headers=_DEMO_SECURITY_HEADERS,
    )


app.mount(
    "/static",
    _DemoStaticFiles(directory=str(_DEMO_STATIC_DIR), html=False),
    name="demo-static",
)

# Safe defaults also make dependency injection possible in unit tests that do
# not enter the ASGI lifespan context.
app.state.runtime_settings = RuntimeSettings.from_environment()
app.state.lifespan_started = False
app.state.runtime = None
app.state.runtime_error = None
app.state.runtime_lock = asyncio.Lock()
app.state.stt_clients = {}
app.state.stt_lock = asyncio.Lock()
app.state.voice_access_settings, app.state.voice_error = _load_voice_access_settings()
app.state.voice_rate_limiter = _VoiceRateLimiter()


async def _runtime(application: FastAPI) -> RuntimeServices:
    existing = application.state.runtime
    if existing is not None:
        return existing
    if not application.state.lifespan_started:
        raise RuntimeLoadError("The application lifespan has not started")
    async with application.state.runtime_lock:
        if application.state.runtime is not None:
            return application.state.runtime
        try:
            application.state.runtime = await load_runtime(
                application.state.runtime_settings
            )
        except Exception as exc:  # noqa: BLE001 - convert to safe readiness error
            application.state.runtime_error = type(exc).__name__
            raise RuntimeLoadError("The local RAG runtime is not ready") from None
        return application.state.runtime


async def _stt_client(
    application: FastAPI,
    *,
    language_code: str,
) -> ElevenLabsStreamingSTT:
    existing = application.state.stt_clients.get(language_code)
    if existing is not None:
        return existing
    async with application.state.stt_lock:
        existing = application.state.stt_clients.get(language_code)
        if existing is not None:
            return existing
        client = ElevenLabsStreamingSTT(_stt_settings(language_code))
        try:
            await client.token_broker.ensure_free_tier()
        except BaseException:
            # This lazy path has the same ownership boundary as startup. Close
            # on terminal refusal and cancellation before propagating it.
            await asyncio.gather(client.aclose(), return_exceptions=True)
            raise
        application.state.stt_clients[language_code] = client
        application.state.voice_error = None
        return client


@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    runtime = request.app.state.runtime
    voice_ready = bool(request.app.state.stt_clients) and not bool(
        request.app.state.voice_error
    )
    return HealthResponse(
        status="ok" if runtime is not None else "degraded",
        implementation_phase="day3_free_demo_harness",
        rag_ready=runtime is not None,
        voice_ready=voice_ready,
        vector_count=runtime.vector_count if runtime is not None else 0,
        supported_languages=(
            list(getattr(runtime, "supported_languages", ()))
            if runtime is not None
            else []
        ),
        latency_target_ms=request.app.state.runtime_settings.latency_target_ms,
    )


@app.post(
    "/rag",
    response_model=RAGResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse}},
)
async def process_rag_query(request: RAGRequest, raw_request: Request) -> Response:
    """Run the warmed local retrieval/extractive/grounding path."""

    try:
        runtime = await _runtime(raw_request.app)
    except RuntimeLoadError:
        error = ErrorResponse(
            error_code="rag_runtime_not_ready",
            message="The local model/index bundle is not ready.",
            retryable=False,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error.model_dump(mode="json"),
        )
    return await runtime.pipeline.answer(request)


def _start_language(payload: dict[str, Any]) -> tuple[str, str]:
    raw = str(payload.get("language_code") or "hi").strip().lower()
    primary = raw.split("-", 1)[0]
    provider_code = _ISO2_TO_ELEVENLABS.get(primary, primary)
    if provider_code not in _ELEVENLABS_TO_ISO2:
        raise _VoiceProtocolError("Unsupported demo language_code")
    return provider_code, _ELEVENLABS_TO_ISO2[provider_code]


async def _authorize_voice_start(
    websocket: WebSocket,
    payload: dict[str, Any],
) -> VoiceAccessSettings:
    """Authorize one start message without retaining or reflecting its secret."""

    settings: VoiceAccessSettings = websocket.app.state.voice_access_settings
    limiter: _VoiceRateLimiter = websocket.app.state.voice_rate_limiter
    client_host = websocket.client.host if websocket.client is not None else "unknown"
    await limiter.admit("ip", client_host, settings)

    supplied = payload.get("demo_token")
    candidate = supplied if isinstance(supplied, str) and len(supplied) <= 512 else ""
    expected = settings.demo_token
    if not expected or not secrets.compare_digest(
        candidate.encode("utf-8"), expected.encode("utf-8")
    ):
        raise _VoiceProtocolError("Voice demo authentication failed")

    # Store only a one-way identity, never the configured bearer token itself.
    token_identity = hashlib.sha256(expected.encode("utf-8")).hexdigest()
    await limiter.admit("token", token_identity, settings)

    origin = websocket.headers.get("origin")
    if settings.require_origin and origin is None:
        raise _VoiceProtocolError("An allowlisted request origin is required")
    if origin is not None and origin not in settings.allowed_origins:
        raise _VoiceProtocolError("The request origin is not permitted")
    return settings


@app.websocket("/ws/voice-rag")
async def voice_rag_socket(websocket: WebSocket) -> None:
    """Proxy one explicit PCM16 turn through Scribe and the local RAG path.

    Client protocol: JSON ``{"event":"start","language_code":"hi",``
    ``"demo_token":"..."}``, then binary mono PCM16/16 kHz frames, then JSON
    ``{"event":"end"}``.
    """

    await websocket.accept()
    stt_task: asyncio.Task[Any] | None = None
    settled_rag_task: asyncio.Task[RAGResponse] | None = None
    settled_query: str | None = None
    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=8)
    first_binary_received_at: float | None = None
    audio_end_received_at: float | None = None
    try:
        try:
            start = await websocket.receive_json()
        except (json.JSONDecodeError, ValueError):
            raise _VoiceProtocolError("Invalid voice start control message") from None
        if not isinstance(start, dict):
            raise _VoiceProtocolError("The first message must be a start event")
        if start.get("event") != "start":
            raise _VoiceProtocolError("The first message must be a start event")
        access_settings = await _authorize_voice_start(websocket, start)
        runtime = await _runtime(websocket.app)
        provider_language, rag_language = _start_language(start)
        supported_languages = tuple(getattr(runtime, "supported_languages", ()))
        if supported_languages and rag_language not in supported_languages:
            # Reject before issuing an STT token: an index that cannot serve the
            # requested language must not consume even included provider quota.
            raise _VoiceProtocolError("The local index does not cover this language")
        origin = websocket.headers.get("origin")
        existing_stt = websocket.app.state.stt_clients.get(provider_language)
        session_settings = (
            existing_stt.settings
            if existing_stt is not None
            else _stt_settings(provider_language)
        )
        sample_rate = int(session_settings.sample_rate)
        max_audio_bytes = int(
            float(session_settings.max_audio_seconds) * sample_rate * 2
        )
        audio_bytes_received = 0

        async def audio_chunks():
            while True:
                chunk = await audio_queue.get()
                if chunk is None:
                    return
                yield chunk

        async def partial(text: str) -> None:
            await websocket.send_json({"event": "partial_transcript", "text": text})

        async def prepare_settled_rag(text: str) -> None:
            """Prepare one local answer, but never publish before exact commit."""

            nonlocal settled_query, settled_rag_task
            if settled_query == text and settled_rag_task is not None:
                return
            if settled_rag_task is not None and not settled_rag_task.done():
                settled_rag_task.cancel()
                await asyncio.gather(settled_rag_task, return_exceptions=True)
            try:
                request = RAGRequest(query=text, language_code=rag_language)
            except ValueError:
                settled_query = None
                settled_rag_task = None
                return
            settled_query = text
            settled_rag_task = asyncio.create_task(runtime.pipeline.answer(request))

        await websocket.send_json(
            {
                "event": "ready",
                "audio_format": "pcm_s16le",
                "sample_rate_hz": sample_rate,
                "commit_strategy": "manual",
            }
        )

        def validate_audio_frame(binary: bytes) -> None:
            """Reject malformed/billable input before it reaches the queue."""

            nonlocal audio_bytes_received
            if not binary:
                raise _VoiceProtocolError("Audio frames cannot be empty")
            if len(binary) > access_settings.max_frame_bytes:
                raise _VoiceProtocolError("An audio frame exceeded the byte limit")
            if len(binary) % 2:
                raise _VoiceProtocolError(
                    "PCM16 audio frames must contain full samples"
                )
            next_total = audio_bytes_received + len(binary)
            if next_total > max_audio_bytes:
                raise _VoiceProtocolError("The audio turn exceeded the byte limit")
            audio_bytes_received = next_total

        async def require_live_stt() -> None:
            """Propagate a typed STT failure instead of waiting on a dead queue."""

            assert stt_task is not None
            try:
                await stt_task
            except ElevenLabsSTTError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - sanitize the provider boundary
                raise ElevenLabsSTTError(
                    "The speech-to-text stage ended unexpectedly"
                ) from None
            raise _VoiceProtocolError(
                "The speech-to-text stage ended before the audio boundary"
            )

        async def receive_while_stt_live() -> dict[str, Any]:
            assert stt_task is not None
            receive_task = asyncio.create_task(websocket.receive())
            done, _ = await asyncio.wait(
                (receive_task, stt_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stt_task in done:
                receive_task.cancel()
                await asyncio.gather(receive_task, return_exceptions=True)
                await require_live_stt()
            return await receive_task

        async def enqueue_while_stt_live(chunk: bytes | None) -> None:
            assert stt_task is not None
            put_task = asyncio.create_task(audio_queue.put(chunk))
            done, _ = await asyncio.wait(
                (put_task, stt_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if put_task in done:
                await put_task
                return
            if stt_task in done:
                put_task.cancel()
                await asyncio.gather(put_task, return_exceptions=True)
                await require_live_stt()
            await put_task

        # A byte-duration limit alone does not stop a silent/stalled browser
        # from holding a billable provider session open. Bound the entire live
        # turn as well, with a small allowance for transport scheduling.
        max_turn_wall_seconds = float(session_settings.max_audio_seconds) + 2.0
        try:
            async with asyncio.timeout(max_turn_wall_seconds):
                # Do not create a provider session until one bounded PCM frame
                # has arrived. Invalid/oversized first frames consume no token.
                first_message = await websocket.receive()
                if first_message.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect(first_message.get("code", 1000))
                first_binary = first_message.get("bytes")
                if first_binary is None:
                    raise _VoiceProtocolError(
                        "The first message after ready must be binary audio"
                    )
                validate_audio_frame(first_binary)
                first_binary_received_at = time.perf_counter()
                stt = await _stt_client(
                    websocket.app,
                    language_code=provider_language,
                )
                stt_task = asyncio.create_task(
                    stt.transcribe(
                        audio_chunks(),
                        on_partial=partial,
                        on_settled=prepare_settled_rag,
                        origin=origin,
                    )
                )
                await enqueue_while_stt_live(first_binary)

                while True:
                    message = await receive_while_stt_live()
                    if message.get("type") == "websocket.disconnect":
                        raise WebSocketDisconnect(message.get("code", 1000))
                    binary = message.get("bytes")
                    if binary is not None:
                        validate_audio_frame(binary)
                        await enqueue_while_stt_live(binary)
                        continue
                    text = message.get("text")
                    if text is None:
                        continue
                    try:
                        control = json.loads(text)
                    except json.JSONDecodeError:
                        raise _VoiceProtocolError(
                            "Invalid voice control message"
                        ) from None
                    if not isinstance(control, dict):
                        raise _VoiceProtocolError("Invalid voice control message")
                    if control.get("event") == "end":
                        if first_binary_received_at is None:
                            raise _VoiceProtocolError(
                                "At least one audio frame is required"
                            )
                        audio_end_received_at = time.perf_counter()
                        await enqueue_while_stt_live(None)
                        break
                    raise _VoiceProtocolError(
                        "Only binary audio or an end event is accepted"
                    )
        except TimeoutError:
            raise _VoiceProtocolError(
                "The voice turn exceeded its wall-clock limit"
            ) from None

        transcription = await stt_task
        committed_at = time.perf_counter()
        if first_binary_received_at is None or audio_end_received_at is None:
            raise _VoiceProtocolError("The voice turn is missing its audio boundaries")
        await websocket.send_json(
            {
                "event": "committed_transcript",
                "payload": transcription.model_dump(mode="json"),
            }
        )
        try:
            committed_request = RAGRequest(
                query=transcription.transcript,
                language_code=rag_language,
            )
        except ValueError:
            raise _VoiceProtocolError(
                "The committed transcript is not a valid RAG query"
            ) from None
        if settled_rag_task is not None and settled_query == transcription.transcript:
            rag = await settled_rag_task
        else:
            if settled_rag_task is not None and not settled_rag_task.done():
                settled_rag_task.cancel()
                await asyncio.gather(settled_rag_task, return_exceptions=True)
            rag = await runtime.pipeline.answer(committed_request)
        answered_at = time.perf_counter()
        committed_to_answer_ms = (answered_at - committed_at) * 1000
        first_audio_to_committed_ms = (committed_at - first_binary_received_at) * 1000
        audio_eof_to_committed_ms = (committed_at - audio_end_received_at) * 1000
        audio_eof_to_answer_ms = (answered_at - audio_end_received_at) * 1000
        response = VoiceRAGResponse(
            transcription=transcription,
            rag=rag,
            latencies=VoicePipelineLatencies(
                metric_definition=(
                    "Server-observed anchors: first_audio begins when ASGI receives "
                    "the first non-empty PCM frame; audio_eof begins when ASGI receives "
                    "the end event; answer stops when the grounded local RAG result "
                    "returns. Final response construction, WebSocket serialization/send, "
                    "client network, and rendering are excluded; provider audio duration "
                    "is reported separately."
                ),
                first_audio_to_committed_ms=first_audio_to_committed_ms,
                audio_eof_to_committed_ms=audio_eof_to_committed_ms,
                committed_to_answer_ms=committed_to_answer_ms,
                audio_eof_to_answer_ms=audio_eof_to_answer_ms,
                first_audio_to_answer_ms=(answered_at - first_binary_received_at)
                * 1000,
                target_ms=runtime.pipeline.latency_target_ms,
                target_met=(
                    audio_eof_to_answer_ms <= runtime.pipeline.latency_target_ms
                ),
            ),
        )
        await websocket.send_json(
            {"event": "answer", "payload": response.model_dump(mode="json")}
        )
        await websocket.close(code=1000)
    except WebSocketDisconnect:
        if stt_task is not None:
            stt_task.cancel()
    except (ElevenLabsSTTError, RuntimeLoadError, _VoiceProtocolError) as exc:
        if stt_task is not None and not stt_task.done():
            stt_task.cancel()
        message = (
            "Invalid voice WebSocket protocol input."
            if isinstance(exc, _VoiceProtocolError)
            else str(exc)
        )
        error = ErrorResponse(
            error_code="voice_pipeline_refused",
            message=message,
            retryable=False,
        )
        try:
            await websocket.send_json(error.model_dump(mode="json"))
            await websocket.close(code=1013, reason="Voice pipeline refused")
        except RuntimeError:
            pass
    finally:
        if stt_task is not None:
            if not stt_task.done():
                stt_task.cancel()
            await asyncio.gather(stt_task, return_exceptions=True)
        if settled_rag_task is not None:
            if not settled_rag_task.done():
                settled_rag_task.cancel()
            await asyncio.gather(settled_rag_task, return_exceptions=True)
