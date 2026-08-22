from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Self
from unittest.mock import patch

from app.schemas import (
    GroundednessAssessment,
    RAGResponse,
    ResponseStatus,
    StageLatencies,
    TranscriptionResult,
    VoicePipelineLatencies,
    VoiceRAGResponse,
)
from scripts import benchmark_live_websocket as benchmark

_DEMO_TOKEN = "offline-demo-token-123456789"


def _write_wav(path: Path, *, sample_value: int = 0, frames: int = 320) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        frame = sample_value.to_bytes(2, byteorder="little", signed=True)
        audio.writeframes(frame * frames)


def _transcription(text: str) -> TranscriptionResult:
    return TranscriptionResult(
        provider="elevenlabs",
        model="scribe_v2_realtime",
        transcript=text,
        language_code="hin",
        is_final=True,
        partial_transcripts=[text[: max(1, len(text) // 2)]],
        connection_ms=1.0,
        time_to_first_partial_ms=2.0,
        time_to_final_transcript_ms=3.0,
        final_after_audio_end_ms=1.0,
        first_audio_to_final_ms=3.0,
        audio_duration_ms=20.0,
        total_ms=4.0,
    )


def _answer_payload(
    transcript: str,
    *,
    answer: str = "The grounded fixture answer is 1912 [chunk:fixture].",
) -> dict[str, object]:
    transcription = _transcription(transcript)
    rag = RAGResponse(
        request_id=f"request-{transcript}",
        query=transcript,
        status=ResponseStatus.ANSWERED,
        answer=answer,
        groundedness=GroundednessAssessment(
            is_grounded=True,
            score=1.0,
            supporting_chunk_ids=["fixture"],
            reason="Offline fixture evidence supports the answer.",
            latency_ms=0.1,
        ),
        latencies=StageLatencies(
            retrieval_ms=0.2,
            relevance_ms=0.1,
            generation_ms=0.1,
            groundedness_ms=0.1,
            output_ms=0.1,
            total_ms=0.6,
            target_ms=200.0,
            target_met=True,
        ),
    )
    response = VoiceRAGResponse(
        transcription=transcription,
        rag=rag,
        latencies=VoicePipelineLatencies(
            metric_definition="Offline server fixture timing.",
            first_audio_to_committed_ms=3.0,
            audio_eof_to_committed_ms=1.0,
            committed_to_answer_ms=0.6,
            audio_eof_to_answer_ms=1.6,
            first_audio_to_answer_ms=4.0,
            target_ms=200.0,
            target_met=True,
        ),
    )
    return response.model_dump(mode="json")


class _FakeWebSocket:
    def __init__(self, transcript: str, *, bad_answer: bool = False) -> None:
        committed = _transcription(transcript).model_dump(mode="json")
        answer = _answer_payload(
            transcript,
            answer=(
                "A different grounded fixture answer [chunk:fixture]."
                if bad_answer
                else "The grounded fixture answer is 1912 [chunk:fixture]."
            ),
        )
        self.incoming = [
            {
                "event": "ready",
                "audio_format": "pcm_s16le",
                "sample_rate_hz": 16_000,
                "commit_strategy": "manual",
            },
            {"event": "partial_transcript", "text": transcript[:2]},
            {"event": "committed_transcript", "payload": committed},
            {"event": "answer", "payload": answer},
        ]
        self.sent: list[bytes | str] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def send(self, value: bytes | str) -> None:
        self.sent.append(value)

    async def recv(self) -> str:
        await asyncio.sleep(0)
        return json.dumps(self.incoming.pop(0), ensure_ascii=False)


class _FakeConnector:
    def __init__(
        self,
        transcripts: list[str],
        *,
        bad_answer_indices: set[int] | None = None,
    ) -> None:
        self.transcripts = transcripts
        self.bad_answer_indices = bad_answer_indices or set()
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.connections: list[_FakeWebSocket] = []

    def __call__(self, url: str, **kwargs: object) -> _FakeWebSocket:
        index = len(self.calls)
        self.calls.append((url, kwargs))
        websocket = _FakeWebSocket(
            self.transcripts[index],
            bad_answer=index in self.bad_answer_indices,
        )
        self.connections.append(websocket)
        return websocket


class LiveWebSocketBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.audio = self.directory / "question.wav"
        _write_wav(self.audio)

    def _args(self, *extra: str) -> object:
        return benchmark.build_parser().parse_args(
            [
                str(self.audio),
                "--ws-url",
                "ws://127.0.0.1:8000/ws/voice-rag",
                "--origin",
                "http://127.0.0.1:8000",
                *extra,
            ]
        )

    async def _run(self, args: object, connector: _FakeConnector) -> int:
        with (
            patch.object(benchmark, "load_dotenv"),
            patch("builtins.print"),
            patch.dict(os.environ, {"VOICE_DEMO_TOKEN": _DEMO_TOKEN}, clear=False),
        ):
            return await benchmark.run(args, connector=connector)  # type: ignore[arg-type]

    def test_defaults_and_security_surface(self) -> None:
        args = self._args()

        self.assertEqual(args.trials, 30)
        self.assertEqual(args.chunk_ms, 100)
        self.assertEqual(args.language_code, "hi")
        self.assertFalse(args.no_realtime_pacing)
        self.assertNotIn(
            "demo_token", {action.dest for action in benchmark.build_parser()._actions}
        )
        with self.assertRaisesRegex(ValueError, "exact http"):
            invalid = self._args("--origin", "http://127.0.0.1:8000/")
            benchmark._validate_args(invalid)

    async def test_pacing_uses_cumulative_pre_send_deadlines_and_exact_anchors(
        self,
    ) -> None:
        now = 10.0
        sleeps: list[float] = []
        sends: list[tuple[float, bytes | str]] = []
        clock = benchmark._ClientClock()

        class Socket:
            async def send(self, value: bytes | str) -> None:
                sends.append((now, value))

        def monotonic() -> float:
            return now

        async def sleep(delay: float) -> None:
            nonlocal now
            sleeps.append(delay)
            now += delay

        await benchmark._send_wav(
            Socket(),
            self.audio,
            chunk_ms=10,
            realtime_pacing=True,
            clock=clock,
            _monotonic=monotonic,
            _sleep=sleep,
        )

        self.assertEqual(len(sends), 3)  # two PCM frames and one end control
        self.assertAlmostEqual(clock.first_audio_at or 0.0, 10.0)
        self.assertAlmostEqual(sends[0][0], 10.0)
        self.assertAlmostEqual(sends[1][0], 10.01)
        self.assertAlmostEqual(clock.eof_at or 0.0, 10.01)
        self.assertAlmostEqual(clock.end_sent_at or 0.0, 10.01)
        self.assertAlmostEqual(sum(sleeps), 0.01)

    async def test_success_measures_client_receipt_and_never_persists_token(
        self,
    ) -> None:
        output = self.directory / "live-report.json"
        args = self._args(
            "--trials",
            "1",
            "--no-realtime-pacing",
            "--latency-only",
            "--target-ms",
            "100000",
            "--output",
            str(output),
        )
        connector = _FakeConnector(["एक सवाल"])

        exit_code = await self._run(args, connector)

        self.assertEqual(exit_code, 0)
        self.assertEqual(connector.calls[0][0], args.ws_url)
        self.assertEqual(connector.calls[0][1]["origin"], args.origin)
        start = json.loads(connector.connections[0].sent[0])
        self.assertEqual(start["demo_token"], _DEMO_TOKEN)
        self.assertEqual(start["language_code"], "hi")
        self.assertIsInstance(connector.connections[0].sent[1], bytes)
        self.assertEqual(
            json.loads(connector.connections[0].sent[-1]), {"event": "end"}
        )

        raw_report = output.read_text(encoding="utf-8")
        self.assertNotIn(_DEMO_TOKEN, raw_report)
        report = json.loads(raw_report)
        self.assertEqual(report["outcomes"]["completed"], 1)
        self.assertEqual(report["outcomes"]["failed"], 0)
        self.assertTrue(report["gates"]["latency_sla_met"])
        self.assertFalse(report["gates"]["live_submission_evidence_eligible"])
        distribution = report["latency_ms"]["full_voice"]
        self.assertTrue(
            distribution["end_sent_to_client_answer"]["p100_is_observed_max"]
        )
        self.assertGreaterEqual(
            distribution["first_audio_to_client_answer"]["p100"],
            distribution["end_sent_to_client_answer"]["p100"],
        )
        self.assertTrue(report["scope"]["client_to_asgi_transport_included"])
        self.assertTrue(report["scope"]["final_websocket_receipt_included"])
        self.assertFalse(report["scope"]["microphone_capture_included"])
        self.assertFalse(report["scope"]["browser_rendering_included"])

    async def test_wrong_answer_oracle_fails_quality_gate(self) -> None:
        manifest = self.directory / "manifest.json"
        manifest.write_text(
            json.dumps(
                [
                    {
                        "audio": self.audio.name,
                        "reference": "सही सवाल",
                        "expected_status": "answered",
                        "expected_answer_contains": "1912",
                    }
                ]
            ),
            encoding="utf-8",
        )
        output = self.directory / "wrong-answer.json"
        args = benchmark.build_parser().parse_args(
            [
                "--manifest",
                str(manifest),
                "--ws-url",
                "ws://127.0.0.1:8000/ws/voice-rag",
                "--origin",
                "http://127.0.0.1:8000",
                "--trials",
                "1",
                "--no-realtime-pacing",
                "--target-ms",
                "100000",
                "--output",
                str(output),
            ]
        )

        exit_code = await self._run(
            args,
            _FakeConnector(["सही सवाल"], bad_answer_indices={0}),
        )

        self.assertEqual(exit_code, 1)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(report["gates"]["latency_sla_met"])
        self.assertFalse(report["gates"]["quality_gate_met"])
        self.assertFalse(report["trials"][0]["quality"]["answer_oracle_match"])
        self.assertFalse(report["gates"]["benchmark_passed"])

    async def test_thirty_distinct_wavs_and_questions_are_submission_eligible(
        self,
    ) -> None:
        entries: list[dict[str, object]] = []
        references: list[str] = []
        for index in range(1, 31):
            audio = self.directory / f"question-{index:02d}.wav"
            _write_wav(audio, sample_value=index)
            reference = f"अलग सवाल {index}"
            references.append(reference)
            entries.append(
                {
                    "audio": audio.name,
                    "reference": reference,
                    "expected_status": "answered",
                    "expected_answer_contains": "1912",
                }
            )
        manifest = self.directory / "submission-manifest.json"
        manifest.write_text(json.dumps(entries), encoding="utf-8")
        output = self.directory / "submission-report.json"
        args = benchmark.build_parser().parse_args(
            [
                "--manifest",
                str(manifest),
                "--ws-url",
                "ws://127.0.0.1:8000/ws/voice-rag",
                "--origin",
                "http://127.0.0.1:8000",
                "--target-ms",
                "100000",
                "--output",
                str(output),
            ]
        )

        local_exit_code = await self._run(args, _FakeConnector(references))

        self.assertEqual(local_exit_code, 0)
        local_report = json.loads(output.read_text(encoding="utf-8"))
        self.assertFalse(local_report["gates"]["secure_transport_met"])
        self.assertFalse(local_report["gates"]["workload_submission_ready"])
        self.assertFalse(local_report["gates"]["live_submission_evidence_eligible"])

        args.ws_url = "wss://demo.example/ws/voice-rag"
        args.origin = "https://demo.example"
        exit_code = await self._run(args, _FakeConnector(references))

        self.assertEqual(exit_code, 0)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["outcomes"]["attempted"], 30)
        self.assertEqual(report["audio"]["distinct_recordings"], 30)
        self.assertEqual(report["quality"]["unique_normalized_reference_questions"], 30)
        self.assertTrue(report["gates"]["workload_submission_ready"])
        self.assertTrue(report["gates"]["secure_transport_met"])
        self.assertTrue(report["gates"]["live_submission_evidence_eligible"])
        self.assertTrue(report["gates"]["live_submission_p100_validated"])

    async def test_protocol_failure_is_retained_as_sla_violation(self) -> None:
        class FailingSocket(_FakeWebSocket):
            async def recv(self) -> str:
                if len(self.sent) == 0:
                    return await super().recv()
                return json.dumps(
                    {
                        "error_code": "voice_pipeline_refused",
                        "message": f"unsafe detail {_DEMO_TOKEN}",
                        "retryable": False,
                    }
                )

        class FailingConnector(_FakeConnector):
            def __call__(self, url: str, **kwargs: object) -> _FakeWebSocket:
                self.calls.append((url, kwargs))
                websocket = FailingSocket("एक सवाल")
                self.connections.append(websocket)
                return websocket

        output = self.directory / "failure-report.json"
        args = self._args(
            "--trials",
            "1",
            "--no-realtime-pacing",
            "--latency-only",
            "--target-ms",
            "100000",
            "--output",
            str(output),
        )

        exit_code = await self._run(args, FailingConnector(["एक सवाल"]))

        self.assertEqual(exit_code, 1)
        raw = output.read_text(encoding="utf-8")
        self.assertNotIn(_DEMO_TOKEN, raw)
        self.assertNotIn("unsafe detail", raw)
        report = json.loads(raw)
        self.assertEqual(report["outcomes"]["failed"], 1)
        self.assertEqual(report["outcomes"]["sla_violations"], 1)
        self.assertFalse(report["gates"]["latency_sla_met"])
        self.assertEqual(
            report["latency_ms"]["full_voice"]["end_sent_to_client_answer"][
                "failed_trials"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
