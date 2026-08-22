from __future__ import annotations

import asyncio
import inspect
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.elevenlabs_stt import ElevenLabsQuotaError
from app.generation import GroqAnswerGenerator
from app.main import VoiceAccessSettings, _VoiceRateLimiter, app, lifespan
from app.runtime import RuntimeSettings
from app.schemas import (
    GroundednessAssessment,
    RAGRequest,
    RAGResponse,
    ResponseStatus,
    StageLatencies,
    TranscriptionResult,
)


class _FakeLocalPipeline:
    latency_target_ms = 200.0

    def __init__(self) -> None:
        self.requests: list[RAGRequest] = []

    async def answer(self, request: RAGRequest) -> RAGResponse:
        self.requests.append(request)
        return RAGResponse(
            request_id=f"fake-{len(self.requests)}",
            query=request.query,
            status=ResponseStatus.ANSWERED,
            answer="[chunk:test-1] RAG uses retrieved evidence.",
            groundedness=GroundednessAssessment(
                is_grounded=True,
                score=1.0,
                supporting_chunk_ids=["test-1"],
                reason="Injected harness evidence is exact.",
                latency_ms=0.01,
            ),
            latencies=StageLatencies(
                retrieval_ms=0.10,
                relevance_ms=0.01,
                generation_ms=0.01,
                groundedness_ms=0.01,
                total_ms=0.13,
                target_ms=self.latency_target_ms,
                target_met=True,
            ),
        )


class _FakeStreamingSTT:
    def __init__(
        self,
        *,
        settled_transcript: str = "What is RAG?",
        committed_transcript: str = "What is RAG?",
    ) -> None:
        self.received_audio: list[bytes] = []
        self.origins: list[str | None] = []
        self.transcribe_calls = 0
        self.settings = SimpleNamespace(max_audio_seconds=15.0, sample_rate=16_000)
        self.settled_transcript = settled_transcript
        self.committed_transcript = committed_transcript

    async def transcribe(
        self,
        audio_chunks,
        *,
        on_partial=None,
        on_settled=None,
        origin: str | None = None,
    ) -> TranscriptionResult:
        self.transcribe_calls += 1
        self.origins.append(origin)
        emitted_partial = False
        async for chunk in audio_chunks:
            self.received_audio.append(chunk)
            if not emitted_partial and on_partial is not None:
                callback_result = on_partial("What is")
                if inspect.isawaitable(callback_result):
                    await callback_result
                emitted_partial = True
        if on_settled is not None:
            callback_result = on_settled(self.settled_transcript)
            if inspect.isawaitable(callback_result):
                await callback_result
            # Give the prepared local task one scheduling turn before commit.
            await asyncio.sleep(0)
        audio_duration_ms = sum(map(len, self.received_audio)) / 32.0
        return TranscriptionResult(
            provider="elevenlabs",
            model="scribe_v2_realtime",
            transcript=self.committed_transcript,
            language_code="eng",
            is_final=True,
            partial_transcripts=["What is"],
            connection_ms=700.0,
            time_to_first_partial_ms=701.0,
            time_to_final_transcript_ms=777.0,
            final_after_audio_end_ms=555.0,
            first_audio_to_final_ms=999.0,
            audio_duration_ms=audio_duration_ms,
            total_ms=1_000.0,
        )


class _FailingStreamingSTT(_FakeStreamingSTT):
    async def transcribe(self, audio_chunks, **kwargs) -> TranscriptionResult:
        del audio_chunks, kwargs
        raise ElevenLabsQuotaError("The included quota is exhausted")


class LifespanResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_does_not_preflight_provider_before_demo_auth(self) -> None:
        application = FastAPI()

        with (
            patch.dict(
                "app.main.os.environ",
                {"ELEVENLABS_API_KEY": "test-key"},
                clear=False,
            ),
            patch(
                "app.main.RuntimeSettings.from_environment",
                return_value=RuntimeSettings(preload=False),
            ),
            patch("app.main.ElevenLabsStreamingSTT") as stt_constructor,
        ):
            async with lifespan(application):
                self.assertIsNone(application.state.voice_error)
                self.assertEqual(application.state.stt_clients, {})
        stt_constructor.assert_not_called()

    async def test_default_access_policy_requires_an_origin_and_rate_limits_keys(
        self,
    ) -> None:
        settings = VoiceAccessSettings.from_environment(
            {"VOICE_DEMO_TOKEN": "configured-token"}
        )
        self.assertTrue(settings.require_origin)
        with self.assertRaisesRegex(ValueError, "demo_token"):
            VoiceAccessSettings.from_environment({"VOICE_DEMO_TOKEN": "short"})

        limiter = _VoiceRateLimiter()
        strict = VoiceAccessSettings(
            demo_token="configured-token",
            require_origin=False,
            sessions_per_window=1,
        )
        await limiter.admit("ip", "192.0.2.1", strict)
        with self.assertRaisesRegex(RuntimeError, "rate limit"):
            await limiter.admit("ip", "192.0.2.1", strict)
        # IP and authenticated-token identities have independent ceilings.
        await limiter.admit("token", "one-way-token-id", strict)


class MainHarnessTests(unittest.TestCase):
    DEMO_TOKEN = "unit-test-demo-token"

    def setUp(self) -> None:
        self.saved_runtime = app.state.runtime
        self.saved_clients = app.state.stt_clients
        self.saved_voice_error = app.state.voice_error
        self.saved_access_settings = app.state.voice_access_settings
        self.saved_rate_limiter = app.state.voice_rate_limiter
        self.pipeline = _FakeLocalPipeline()
        self.stt = _FakeStreamingSTT()
        app.state.runtime = SimpleNamespace(
            pipeline=self.pipeline,
            vector_count=1,
        )
        app.state.stt_clients = {"eng": self.stt}
        app.state.voice_error = None
        app.state.voice_access_settings = VoiceAccessSettings(
            demo_token=self.DEMO_TOKEN,
            allowed_origins=("https://demo.example",),
            require_origin=False,
        )
        app.state.voice_rate_limiter = _VoiceRateLimiter()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()
        app.state.runtime = self.saved_runtime
        app.state.stt_clients = self.saved_clients
        app.state.voice_error = self.saved_voice_error
        app.state.voice_access_settings = self.saved_access_settings
        app.state.voice_rate_limiter = self.saved_rate_limiter

    def start_payload(self, **overrides: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "event": "start",
            "language_code": "en",
            "demo_token": self.DEMO_TOKEN,
        }
        payload.update(overrides)
        return payload

    def test_ready_rag_uses_injected_local_runtime(self) -> None:
        with (
            patch.object(
                GroqAnswerGenerator,
                "generate",
                new_callable=AsyncMock,
            ) as paid_generate,
            patch(
                "app.main.load_runtime",
                new_callable=AsyncMock,
            ) as runtime_loader,
        ):
            response = self.client.post(
                "/rag",
                json={"query": "What is RAG?", "language_code": "en"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "answered")
        self.assertEqual(
            payload["answer"],
            "[chunk:test-1] RAG uses retrieved evidence.",
        )
        self.assertEqual(self.pipeline.requests[0].query, "What is RAG?")
        paid_generate.assert_not_awaited()
        runtime_loader.assert_not_awaited()

    def test_voice_start_binary_end_emits_partial_committed_and_answer(self) -> None:
        frame = b"\x01\x00" * 320
        with (
            patch.object(
                GroqAnswerGenerator,
                "generate",
                new_callable=AsyncMock,
            ) as paid_generate,
            patch(
                "app.main.load_runtime",
                new_callable=AsyncMock,
            ) as runtime_loader,
            self.client.websocket_connect(
                "/ws/voice-rag",
                headers={"origin": "https://demo.example"},
            ) as websocket,
        ):
            websocket.send_json(self.start_payload())
            ready = websocket.receive_json()
            websocket.send_bytes(frame)
            websocket.send_json({"event": "end"})
            partial = websocket.receive_json()
            committed = websocket.receive_json()
            answer = websocket.receive_json()

        self.assertEqual(ready["event"], "ready")
        self.assertEqual(ready["sample_rate_hz"], 16_000)
        self.assertEqual(partial, {"event": "partial_transcript", "text": "What is"})
        self.assertEqual(committed["event"], "committed_transcript")
        self.assertTrue(committed["payload"]["is_final"])
        self.assertEqual(committed["payload"]["provider"], "elevenlabs")
        self.assertEqual(committed["payload"]["time_to_final_transcript_ms"], 777.0)
        self.assertEqual(committed["payload"]["final_after_audio_end_ms"], 555.0)
        self.assertEqual(answer["event"], "answer")
        self.assertEqual(
            answer["payload"]["transcription"]["transcript"], "What is RAG?"
        )
        self.assertEqual(answer["payload"]["rag"]["status"], "answered")
        self.assertEqual(self.stt.received_audio, [frame])
        self.assertEqual(self.stt.origins, ["https://demo.example"])
        self.assertEqual(self.pipeline.requests[0].language_code, "en")
        self.assertEqual(
            [request.query for request in self.pipeline.requests],
            ["What is RAG?"],
        )

        latencies = answer["payload"]["latencies"]
        self.assertIn("ASGI receives", latencies["metric_definition"])
        self.assertIn("WebSocket serialization/send", latencies["metric_definition"])
        self.assertGreaterEqual(latencies["first_audio_to_committed_ms"], 0)
        self.assertGreaterEqual(latencies["audio_eof_to_committed_ms"], 0)
        self.assertGreaterEqual(latencies["committed_to_answer_ms"], 0)
        self.assertGreaterEqual(
            latencies["first_audio_to_answer_ms"],
            latencies["audio_eof_to_answer_ms"],
        )
        self.assertAlmostEqual(
            latencies["audio_eof_to_answer_ms"],
            latencies["audio_eof_to_committed_ms"]
            + latencies["committed_to_answer_ms"],
            places=6,
        )
        # The server anchors must not copy the intentionally misleading fake
        # provider diagnostics (999 ms / 555 ms).
        self.assertNotEqual(latencies["first_audio_to_committed_ms"], 999.0)
        self.assertNotEqual(latencies["audio_eof_to_committed_ms"], 555.0)
        paid_generate.assert_not_awaited()
        runtime_loader.assert_not_awaited()

    def test_revised_commit_discards_settled_preparation(self) -> None:
        self.stt = _FakeStreamingSTT(
            settled_transcript="What is a graph?",
            committed_transcript="What is RAG?",
        )
        app.state.stt_clients = {"eng": self.stt}
        with (
            patch.object(
                GroqAnswerGenerator,
                "generate",
                new_callable=AsyncMock,
            ) as paid_generate,
            self.client.websocket_connect("/ws/voice-rag") as websocket,
        ):
            websocket.send_json(self.start_payload())
            self.assertEqual(websocket.receive_json()["event"], "ready")
            websocket.send_bytes(b"\x01\x00" * 160)
            websocket.send_json({"event": "end"})
            self.assertEqual(websocket.receive_json()["event"], "partial_transcript")
            committed = websocket.receive_json()
            answer = websocket.receive_json()

        self.assertEqual(committed["event"], "committed_transcript")
        self.assertEqual(committed["payload"]["transcript"], "What is RAG?")
        self.assertEqual(answer["event"], "answer")
        self.assertEqual(answer["payload"]["rag"]["query"], "What is RAG?")
        self.assertEqual(
            [request.query for request in self.pipeline.requests],
            ["What is a graph?", "What is RAG?"],
        )
        paid_generate.assert_not_awaited()

    def test_malformed_start_message_is_sanitized(self) -> None:
        secret_marker = "should-not-be-reflected"
        with self.client.websocket_connect("/ws/voice-rag") as websocket:
            websocket.send_text(f'{{"event":"start","value":"{secret_marker}"')
            error = websocket.receive_json()

        self.assertEqual(error["error_code"], "voice_pipeline_refused")
        self.assertEqual(
            error["message"],
            "Invalid voice WebSocket protocol input.",
        )
        self.assertNotIn(secret_marker, json.dumps(error))

    def test_unsupported_index_language_is_rejected_before_stt(self) -> None:
        app.state.runtime = SimpleNamespace(
            pipeline=self.pipeline,
            vector_count=1,
            supported_languages=("hi", "ta", "ur"),
        )
        with self.client.websocket_connect("/ws/voice-rag") as websocket:
            websocket.send_json(self.start_payload())
            error = websocket.receive_json()

        self.assertEqual(error["error_code"], "voice_pipeline_refused")
        self.assertEqual(self.stt.received_audio, [])
        self.assertEqual(self.stt.origins, [])
        self.assertEqual(self.pipeline.requests, [])

    def test_provider_failure_is_observed_without_filling_audio_queue(self) -> None:
        failing_stt = _FailingStreamingSTT()
        app.state.stt_clients = {"eng": failing_stt}
        with self.client.websocket_connect("/ws/voice-rag") as websocket:
            websocket.send_json(self.start_payload())
            self.assertEqual(websocket.receive_json()["event"], "ready")
            websocket.send_bytes(b"\x01\x00" * 160)
            error = websocket.receive_json()

        self.assertEqual(error["error_code"], "voice_pipeline_refused")
        self.assertIn("quota", error["message"].lower())

    def test_missing_and_wrong_demo_token_never_start_stt(self) -> None:
        for supplied in (None, "wrong-secret-should-not-be-reflected"):
            with self.subTest(supplied=supplied):
                payload: dict[str, object] = {
                    "event": "start",
                    "language_code": "en",
                }
                if supplied is not None:
                    payload["demo_token"] = supplied
                with (
                    patch(
                        "app.main._stt_client",
                        new_callable=AsyncMock,
                    ) as stt_factory,
                    self.client.websocket_connect("/ws/voice-rag") as websocket,
                ):
                    websocket.send_json(payload)
                    error = websocket.receive_json()

                self.assertEqual(error["error_code"], "voice_pipeline_refused")
                self.assertNotIn(str(supplied), json.dumps(error))
                stt_factory.assert_not_awaited()
                self.assertEqual(self.stt.transcribe_calls, 0)
                self.assertEqual(self.stt.received_audio, [])

    def test_missing_token_configuration_fails_closed_before_stt(self) -> None:
        app.state.voice_access_settings = VoiceAccessSettings(
            demo_token="",
            require_origin=False,
        )
        with (
            patch("app.main._stt_client", new_callable=AsyncMock) as stt_factory,
            self.client.websocket_connect("/ws/voice-rag") as websocket,
        ):
            websocket.send_json(self.start_payload())
            error = websocket.receive_json()

        self.assertEqual(error["error_code"], "voice_pipeline_refused")
        stt_factory.assert_not_awaited()
        self.assertEqual(self.stt.transcribe_calls, 0)

    def test_oversized_first_frame_never_starts_provider_or_enters_queue(self) -> None:
        app.state.voice_access_settings = VoiceAccessSettings(
            demo_token=self.DEMO_TOKEN,
            allowed_origins=("https://demo.example",),
            require_origin=False,
            max_frame_bytes=4,
        )
        with (
            patch("app.main._stt_client", new_callable=AsyncMock) as stt_factory,
            self.client.websocket_connect("/ws/voice-rag") as websocket,
        ):
            websocket.send_json(self.start_payload())
            self.assertEqual(websocket.receive_json()["event"], "ready")
            websocket.send_bytes(b"\x01\x00" * 3)
            error = websocket.receive_json()

        self.assertEqual(error["error_code"], "voice_pipeline_refused")
        stt_factory.assert_not_awaited()
        self.assertEqual(self.stt.transcribe_calls, 0)
        self.assertEqual(self.stt.received_audio, [])

    def test_cumulative_audio_limit_is_checked_before_queue_put(self) -> None:
        self.stt.settings.max_audio_seconds = 0.00025  # eight PCM16 bytes
        app.state.voice_access_settings = VoiceAccessSettings(
            demo_token=self.DEMO_TOKEN,
            require_origin=False,
            max_frame_bytes=8,
        )
        with self.client.websocket_connect("/ws/voice-rag") as websocket:
            websocket.send_json(self.start_payload())
            self.assertEqual(websocket.receive_json()["event"], "ready")
            websocket.send_bytes(b"\x01\x00" * 2)
            websocket.send_bytes(b"\x02\x00" * 3)
            error = websocket.receive_json()
            if error.get("event") == "partial_transcript":
                error = websocket.receive_json()

        self.assertEqual(error["error_code"], "voice_pipeline_refused")
        self.assertNotIn(b"\x02\x00" * 3, self.stt.received_audio)


if __name__ == "__main__":
    unittest.main()
