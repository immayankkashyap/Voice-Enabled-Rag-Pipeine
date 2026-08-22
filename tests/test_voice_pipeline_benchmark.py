from __future__ import annotations

import asyncio
import inspect
import json
import os
import shutil
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from app.schemas import (
    GroundednessAssessment,
    RAGResponse,
    RefusalReason,
    ResponseStatus,
    StageLatencies,
    TranscriptionResult,
)
from scripts import benchmark_voice_pipeline as benchmark

_API_KEY = "test-elevenlabs-key-must-not-enter-report"
_ACKNOWLEDGEMENT = "I_CONFIRM_NO_PAYG_OR_AUTO_TOP_UP"


def _write_wav(
    path: Path,
    *,
    sample_rate: int = 16_000,
    sample_value: int = 0,
) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        frame = sample_value.to_bytes(2, byteorder="little", signed=True)
        audio.writeframes(frame * 320)


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
        time_to_final_transcript_ms=4.0,
        final_after_audio_end_ms=0.5,
        first_audio_to_final_ms=4.0,
        audio_duration_ms=20.0,
        total_ms=4.5,
    )


def _rag_response(query: str, status: ResponseStatus) -> RAGResponse:
    latencies = StageLatencies(
        retrieval_ms=1.0,
        relevance_ms=0.2,
        generation_ms=0.3,
        groundedness_ms=0.2,
        output_ms=0.1,
        total_ms=1.8,
        target_ms=200.0,
        target_met=True,
    )
    if status is ResponseStatus.REFUSED:
        return RAGResponse(
            request_id=f"refused-{query}",
            query=query,
            status=status,
            refusal_reason=RefusalReason.NO_RELEVANT_CONTEXT,
            latencies=latencies,
        )
    return RAGResponse(
        request_id=f"answered-{query}",
        query=query,
        status=status,
        answer=f"Grounded fixture answer for {query}",
        groundedness=GroundednessAssessment(
            is_grounded=True,
            score=1.0,
            supporting_chunk_ids=["fixture-chunk"],
            reason="offline fixture is grounded",
            latency_ms=0.2,
        ),
        latencies=latencies,
    )


class _FakeTokenBroker:
    def __init__(self) -> None:
        self.preflight_calls = 0

    async def ensure_free_tier(self, *, force: bool = False) -> None:
        if not force:
            raise AssertionError("benchmark preflight must be forced once")
        self.preflight_calls += 1


class _FakeSTT:
    def __init__(
        self,
        results: list[TranscriptionResult | Exception],
        *,
        settled_sequences: list[list[str]] | None = None,
    ) -> None:
        self.results = results
        self.settled_sequences = settled_sequences
        self.token_broker = _FakeTokenBroker()
        self.calls = 0
        self.closed = False

    async def transcribe(
        self,
        audio_chunks: object,
        *,
        on_partial: object = None,
        on_settled: object = None,
    ) -> TranscriptionResult:
        call_index = self.calls
        self.calls += 1
        received = bytearray()
        async for chunk in audio_chunks:  # type: ignore[attr-defined]
            received.extend(chunk)
        if not received:
            raise AssertionError("benchmark did not stream PCM frames")

        result = self.results[call_index]
        if isinstance(result, Exception):
            raise result
        if on_partial is not None:
            callback_result = on_partial(result.partial_transcripts[-1])
            if inspect.isawaitable(callback_result):
                await callback_result
        settled = (
            self.settled_sequences[call_index]
            if self.settled_sequences is not None
            else [result.transcript]
        )
        if on_settled is not None:
            for text in settled:
                callback_result = on_settled(text)
                if inspect.isawaitable(callback_result):
                    await callback_result
                await asyncio.sleep(0)
        return result

    async def aclose(self) -> None:
        self.closed = True


class _FakePipeline:
    def __init__(
        self,
        outcomes: list[ResponseStatus | Exception],
        *,
        answers: list[str] | None = None,
    ) -> None:
        self.outcomes = outcomes
        self.answers = answers
        self.requests = []

    async def answer(self, request: object) -> RAGResponse:
        self.requests.append(request)
        await asyncio.sleep(0)
        index = min(len(self.requests) - 1, len(self.outcomes) - 1)
        outcome = self.outcomes[index]
        if isinstance(outcome, Exception):
            raise outcome
        response = _rag_response(request.query, outcome)  # type: ignore[attr-defined]
        if self.answers is not None and response.status is ResponseStatus.ANSWERED:
            response.answer = self.answers[min(index, len(self.answers) - 1)]
        return response


def _runtime(pipeline: _FakePipeline, *, languages: tuple[str, ...] = ("hi",)):
    return SimpleNamespace(
        pipeline=pipeline,
        vector_count=12,
        supported_languages=languages,
        embedding_model="offline-fixture-encoder",
        device="cpu",
        load_ms=10.0,
        warmup_ms=2.0,
    )


class VoicePipelineBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.audio_one = self.directory / "question-one.wav"
        self.audio_two = self.directory / "question-two.wav"
        _write_wav(self.audio_one)
        _write_wav(self.audio_two)

    def test_audio_diversity_hash_ignores_non_pcm_file_bytes(self) -> None:
        original = self.directory / "original.wav"
        metadata_variant = self.directory / "metadata-variant.wav"
        _write_wav(original, sample_value=17)
        shutil.copyfile(original, metadata_variant)
        with metadata_variant.open("ab") as audio_file:
            audio_file.write(b"header-only-variation")

        self.assertNotEqual(original.read_bytes(), metadata_variant.read_bytes())
        self.assertEqual(
            benchmark._wav_content_digest(original),
            benchmark._wav_content_digest(metadata_variant),
        )

    async def _run_offline(
        self,
        args: object,
        *,
        stt: _FakeSTT,
        pipeline: _FakePipeline,
        languages: tuple[str, ...] = ("hi",),
    ) -> tuple[int, Mock, AsyncMock]:
        runtime_loader = AsyncMock(return_value=_runtime(pipeline, languages=languages))
        stt_factory = Mock(return_value=stt)
        with (
            patch.object(benchmark, "load_runtime", runtime_loader),
            patch.object(benchmark, "ElevenLabsStreamingSTT", stt_factory),
            patch.object(benchmark, "load_dotenv"),
            patch("builtins.print"),
            patch.dict(
                os.environ,
                {
                    "ELEVENLABS_API_KEY": _API_KEY,
                    "ELEVENLABS_FREE_TIER_ACKNOWLEDGEMENT": _ACKNOWLEDGEMENT,
                },
            ),
        ):
            exit_code = await benchmark.run(args)  # type: ignore[arg-type]
        return exit_code, stt_factory, runtime_loader

    def test_defaults_are_live_hindi_overlap_and_thirty_trials(self) -> None:
        args = benchmark.build_parser().parse_args([str(self.audio_one)])

        self.assertEqual(args.trials, 30)
        self.assertEqual(args.stt_language_code, "hin")
        self.assertEqual(args.rag_language_code, "hi")
        self.assertTrue(args.overlap_enabled)
        self.assertFalse(args.no_realtime_pacing)

    def test_distribution_uses_observed_max_and_keeps_failures_visible(self) -> None:
        result = benchmark._distribution(
            [10.0, 20.0, 90.0],
            failed_trials=1,
            target_ms=100.0,
        )

        self.assertEqual(result["p50"], 20.0)
        self.assertAlmostEqual(result["p70"], 48.0)
        self.assertEqual(result["p100"], 90.0)
        self.assertTrue(result["p100_is_observed_max"])
        self.assertEqual(result["failed_trials"], 1)
        self.assertFalse(result["all_attempts_meet_target"])

    def test_invalid_wav_is_rejected_before_any_provider_work(self) -> None:
        wrong_rate = self.directory / "wrong-rate.wav"
        _write_wav(wrong_rate, sample_rate=8_000)

        with self.assertRaisesRegex(ValueError, "PCM16 WAV at 16000 Hz"):
            benchmark.wav_metadata(wrong_rate)

    def test_answered_manifest_sample_requires_answer_or_source_oracle(self) -> None:
        manifest = self.directory / "missing-oracle.json"
        manifest.write_text(
            json.dumps([{"audio": self.audio_one.name, "expected_status": "answered"}]),
            encoding="utf-8",
        )
        args = benchmark.build_parser().parse_args(["--manifest", str(manifest)])

        with self.assertRaisesRegex(ValueError, "requires an answer oracle"):
            benchmark._load_samples(args)

    async def test_realtime_pacing_sends_on_cumulative_capture_deadlines(self) -> None:
        now = 10.0
        sleeps: list[float] = []
        sent_at: list[float] = []
        clock = benchmark._AudioClock()

        def monotonic() -> float:
            return now

        async def sleep(delay: float) -> None:
            nonlocal now
            sleeps.append(delay)
            now += delay

        async for _chunk in benchmark.wav_chunks(
            self.audio_one,
            chunk_ms=10,
            realtime_pacing=True,
            clock=clock,
            _monotonic=monotonic,
            _sleep=sleep,
        ):
            # This loop is the deterministic stand-in for the WebSocket sender.
            sent_at.append(monotonic())

        self.assertEqual(len(sent_at), 2)
        self.assertAlmostEqual(clock.first_audio_at, 10.0)
        self.assertAlmostEqual(sent_at[0], 10.01)
        self.assertAlmostEqual(sent_at[1], 10.02)
        self.assertAlmostEqual(clock.eof_at, sent_at[-1])
        self.assertAlmostEqual(sum(sleeps), 0.02)

    async def test_manifest_run_reuses_runtime_and_client_and_is_secret_free(
        self,
    ) -> None:
        manifest = self.directory / "samples.json"
        manifest.write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "audio": self.audio_one.name,
                            "reference": "पहला सवाल",
                            "expected_answer_contains": "पहला सवाल",
                        },
                        {
                            "audio": self.audio_two.name,
                            "reference": "दूसरा सवाल",
                            "expected_status": "refused",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        output = self.directory / "report.json"
        args = benchmark.build_parser().parse_args(
            [
                "--manifest",
                str(manifest),
                "--trials",
                "3",
                "--no-realtime-pacing",
                "--target-ms",
                "1000000",
                "--output",
                str(output),
            ]
        )
        stt = _FakeSTT(
            [
                _transcription("पहला सवाल"),
                _transcription("दूसरा सवाल"),
                _transcription("पहला सवाल"),
            ]
        )
        pipeline = _FakePipeline(
            [ResponseStatus.ANSWERED, ResponseStatus.REFUSED, ResponseStatus.ANSWERED]
        )

        exit_code, stt_factory, runtime_loader = await self._run_offline(
            args,
            stt=stt,
            pipeline=pipeline,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(runtime_loader.await_count, 1)
        self.assertEqual(stt_factory.call_count, 1)
        self.assertEqual(stt.token_broker.preflight_calls, 1)
        self.assertEqual(stt.calls, 3)
        self.assertTrue(stt.closed)
        self.assertEqual(len(pipeline.requests), 3)
        self.assertTrue(
            all(request.language_code == "hi" for request in pipeline.requests)
        )

        raw_report = output.read_text(encoding="utf-8")
        self.assertNotIn(_API_KEY, raw_report)
        self.assertNotIn("api_key", raw_report.casefold())
        self.assertEqual(list(self.directory.glob(f".{output.name}.*.tmp")), [])
        report = json.loads(raw_report)
        self.assertEqual(report["audio"]["source_count"], 2)
        self.assertEqual(report["audio"]["distinct_recordings"], 1)
        self.assertIn("SHA-256", report["audio"]["distinct_recordings_basis"])
        self.assertFalse(report["audio"]["submission_quality"])
        self.assertEqual(
            [trial["source_file"] for trial in report["trials"]],
            [self.audio_one.name, self.audio_two.name, self.audio_one.name],
        )
        self.assertEqual(report["outcomes"]["answered"], 2)
        self.assertEqual(report["outcomes"]["refused"], 1)
        self.assertEqual(report["outcomes"]["failed"], 0)
        self.assertEqual(report["outcomes"]["overlap"]["used"], 3)
        self.assertEqual(report["outcomes"]["overlap"]["exact_commit_matches"], 3)
        self.assertTrue(report["quality"]["all_attempts_eligible"])
        self.assertTrue(report["quality"]["manifest_supplied"])
        self.assertEqual(report["quality"]["referenced_trials"], 3)
        self.assertTrue(report["gates"]["benchmark_passed"])
        self.assertFalse(report["gates"]["live_submission_p100_validated"])
        self.assertEqual(report["reference_quality"]["reference_trials"], 3)
        self.assertEqual(
            report["reference_quality"]["normalized_exact_match"]["matches"],
            3,
        )
        first_source_stability = report["transcript_stability"]["per_source"][
            "audio-001"
        ]
        self.assertTrue(first_source_stability["all_identical"])
        self.assertEqual(first_source_stability["transcripts_available"], 2)
        eof_values = [
            trial["latency_ms"]["audio_eof_to_answer"] for trial in report["trials"]
        ]
        first_audio_values = [
            trial["latency_ms"]["first_audio_to_answer"] for trial in report["trials"]
        ]
        full_voice = report["latency_ms"]["full_voice"]
        self.assertEqual(full_voice["audio_eof_to_answer"]["p100"], max(eof_values))
        self.assertEqual(
            full_voice["first_audio_to_answer"]["p100"],
            max(first_audio_values),
        )
        self.assertIsNotNone(full_voice["audio_eof_to_answer"]["p50"])
        self.assertIsNotNone(full_voice["audio_eof_to_answer"]["p70"])

    async def test_copied_wavs_fail_content_distinct_submission_gate(self) -> None:
        entries = []
        transcriptions = []
        for index in range(1, 31):
            copied_audio = self.directory / f"copied-{index:02d}.wav"
            shutil.copyfile(self.audio_one, copied_audio)
            reference = f"अलग सवाल {index}"
            entries.append(
                {
                    "audio": copied_audio.name,
                    "reference": reference,
                    "expected_status": "refused",
                }
            )
            transcriptions.append(_transcription(reference))

        manifest = self.directory / "copied-audio-samples.json"
        manifest.write_text(json.dumps(entries), encoding="utf-8")
        output = self.directory / "copied-audio-report.json"
        args = benchmark.build_parser().parse_args(
            [
                "--manifest",
                str(manifest),
                "--trials",
                "30",
                "--target-ms",
                "1000000",
                "--output",
                str(output),
            ]
        )
        stt = _FakeSTT(transcriptions)
        pipeline = _FakePipeline([ResponseStatus.REFUSED] * 30)

        exit_code, _, _ = await self._run_offline(args, stt=stt, pipeline=pipeline)

        self.assertEqual(exit_code, 0)
        raw_report = output.read_text(encoding="utf-8")
        self.assertNotIn(str(self.directory), raw_report)
        self.assertNotRegex(raw_report, r"\b[0-9a-f]{64}\b")
        report = json.loads(raw_report)
        self.assertEqual(report["audio"]["source_count"], 30)
        self.assertEqual(report["audio"]["distinct_recordings"], 1)
        self.assertEqual(
            report["quality"]["unique_normalized_reference_questions"],
            30,
        )
        self.assertFalse(report["gates"]["minimum_content_distinct_recordings_met"])
        self.assertTrue(
            report["gates"]["minimum_unique_normalized_reference_questions_met"]
        )
        self.assertFalse(report["gates"]["workload_submission_ready"])
        self.assertFalse(report["gates"]["component_evidence_eligible"])

    async def test_repeated_references_fail_question_diversity_gate(self) -> None:
        repeated_reference = "एक ही सवाल"
        entries = []
        for index in range(1, 31):
            distinct_audio = self.directory / f"distinct-{index:02d}.wav"
            _write_wav(distinct_audio, sample_value=index)
            reference_variant = (
                f" {repeated_reference}? " if index % 2 else f"{repeated_reference} !"
            )
            entries.append(
                {
                    "audio": distinct_audio.name,
                    "reference": reference_variant,
                    "expected_status": "refused",
                }
            )

        manifest = self.directory / "repeated-reference-samples.json"
        manifest.write_text(json.dumps(entries), encoding="utf-8")
        output = self.directory / "repeated-reference-report.json"
        args = benchmark.build_parser().parse_args(
            [
                "--manifest",
                str(manifest),
                "--trials",
                "30",
                "--target-ms",
                "1000000",
                "--output",
                str(output),
            ]
        )
        stt = _FakeSTT([_transcription(repeated_reference) for _ in range(30)])
        pipeline = _FakePipeline([ResponseStatus.REFUSED] * 30)

        exit_code, _, _ = await self._run_offline(args, stt=stt, pipeline=pipeline)

        self.assertEqual(exit_code, 0)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["audio"]["distinct_recordings"], 30)
        self.assertEqual(
            report["quality"]["unique_normalized_reference_questions"],
            1,
        )
        self.assertTrue(report["gates"]["minimum_content_distinct_recordings_met"])
        self.assertFalse(
            report["gates"]["minimum_unique_normalized_reference_questions_met"]
        )
        self.assertTrue(report["gates"]["quality_gate_met"])
        self.assertFalse(report["gates"]["workload_submission_ready"])
        self.assertFalse(report["gates"]["component_evidence_eligible"])

    async def test_settled_replacement_reuses_only_exact_committed_text(self) -> None:
        args = benchmark.build_parser().parse_args(
            [str(self.audio_one), "--no-realtime-pacing", "--target-ms", "1000000"]
        )
        sample = benchmark._load_samples(args)[0]
        stt = _FakeSTT(
            [_transcription("final query")],
            settled_sequences=[["superseded query", "final query"]],
        )
        pipeline = _FakePipeline([ResponseStatus.ANSWERED])

        trial = await benchmark._run_trial(
            trial_number=1,
            args=args,
            sample=sample,
            stt=stt,  # type: ignore[arg-type]
            pipeline=pipeline,
        )

        self.assertEqual(trial["outcome"], "completed")
        self.assertEqual(trial["overlap"]["replacement_count"], 1)
        self.assertTrue(trial["overlap"]["exact_commit_match"])
        self.assertTrue(trial["overlap"]["used"])
        self.assertEqual(pipeline.requests[-1].query, "final query")

    async def test_mismatched_settled_work_is_discarded_and_recomputed(self) -> None:
        args = benchmark.build_parser().parse_args(
            [str(self.audio_one), "--no-realtime-pacing", "--target-ms", "1000000"]
        )
        sample = benchmark._load_samples(args)[0]
        stt = _FakeSTT(
            [_transcription("committed query")],
            settled_sequences=[["different settled query"]],
        )
        pipeline = _FakePipeline([ResponseStatus.ANSWERED])

        trial = await benchmark._run_trial(
            trial_number=1,
            args=args,
            sample=sample,
            stt=stt,  # type: ignore[arg-type]
            pipeline=pipeline,
        )

        self.assertEqual(trial["outcome"], "completed")
        self.assertFalse(trial["overlap"]["exact_commit_match"])
        self.assertFalse(trial["overlap"]["used"])
        self.assertEqual(pipeline.requests[-1].query, "committed query")

    async def test_pipeline_failure_is_an_sla_violation_without_error_details(
        self,
    ) -> None:
        output = self.directory / "failure-report.json"
        args = benchmark.build_parser().parse_args(
            [
                str(self.audio_one),
                "--trials",
                "3",
                "--no-realtime-pacing",
                "--sequential",
                "--target-ms",
                "1000000",
                "--reference",
                "एक सवाल",
                "--output",
                str(output),
            ]
        )
        stt = _FakeSTT([_transcription("एक सवाल") for _ in range(3)])
        pipeline = _FakePipeline(
            [
                ResponseStatus.ANSWERED,
                RuntimeError(f"upstream leaked {_API_KEY}"),
                ResponseStatus.ANSWERED,
            ]
        )

        exit_code, _, _ = await self._run_offline(args, stt=stt, pipeline=pipeline)

        self.assertEqual(exit_code, 1)
        raw_report = output.read_text(encoding="utf-8")
        self.assertNotIn(_API_KEY, raw_report)
        self.assertNotIn("upstream leaked", raw_report)
        report = json.loads(raw_report)
        self.assertEqual(report["outcomes"]["attempted"], 3)
        self.assertEqual(report["outcomes"]["completed"], 2)
        self.assertEqual(report["outcomes"]["failed"], 1)
        self.assertEqual(report["outcomes"]["sla_violations"], 1)
        self.assertFalse(report["outcomes"]["all_attempts_meet_target"])
        self.assertEqual(report["trials"][1]["error_type"], "RuntimeError")
        distribution = report["latency_ms"]["full_voice"]["audio_eof_to_answer"]
        self.assertEqual(distribution["failed_trials"], 1)
        self.assertFalse(distribution["all_attempts_meet_target"])

    async def test_fast_unexpected_refusal_cannot_pass_default_gate(self) -> None:
        output = self.directory / "refusal-report.json"
        args = benchmark.build_parser().parse_args(
            [
                str(self.audio_one),
                "--trials",
                "1",
                "--no-realtime-pacing",
                "--target-ms",
                "1000000",
                "--output",
                str(output),
            ]
        )
        stt = _FakeSTT([_transcription("एक सवाल")])
        pipeline = _FakePipeline([ResponseStatus.REFUSED])

        exit_code, _, _ = await self._run_offline(args, stt=stt, pipeline=pipeline)

        self.assertEqual(exit_code, 1)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(report["gates"]["latency_sla_met"])
        self.assertFalse(report["gates"]["quality_gate_met"])
        self.assertFalse(report["gates"]["benchmark_passed"])
        self.assertFalse(report["trials"][0]["quality"]["status_match"])

    async def test_reference_mismatch_cannot_pass_default_gate(self) -> None:
        output = self.directory / "reference-mismatch-report.json"
        args = benchmark.build_parser().parse_args(
            [
                str(self.audio_one),
                "--trials",
                "1",
                "--no-realtime-pacing",
                "--target-ms",
                "1000000",
                "--reference",
                "अपेक्षित सवाल",
                "--output",
                str(output),
            ]
        )
        stt = _FakeSTT([_transcription("गलत सवाल")])
        pipeline = _FakePipeline([ResponseStatus.ANSWERED])

        exit_code, _, _ = await self._run_offline(args, stt=stt, pipeline=pipeline)

        self.assertEqual(exit_code, 1)
        report = json.loads(output.read_text(encoding="utf-8"))
        quality = report["trials"][0]["quality"]
        self.assertFalse(quality["reference_normalized_match"])
        self.assertFalse(quality["eligible"])
        self.assertFalse(report["gates"]["benchmark_passed"])

    async def test_wrong_grounded_extract_fails_answer_oracle(self) -> None:
        manifest = self.directory / "answer-oracle-samples.json"
        manifest.write_text(
            json.dumps(
                [
                    {
                        "audio": self.audio_one.name,
                        "reference": "सही सवाल",
                        "expected_status": "answered",
                        "expected_answer_contains": ["सही उत्तर", "1912"],
                    }
                ]
            ),
            encoding="utf-8",
        )
        output = self.directory / "wrong-grounded-report.json"
        args = benchmark.build_parser().parse_args(
            [
                "--manifest",
                str(manifest),
                "--trials",
                "1",
                "--no-realtime-pacing",
                "--target-ms",
                "1000000",
                "--output",
                str(output),
            ]
        )
        stt = _FakeSTT([_transcription("सही सवाल")])
        pipeline = _FakePipeline(
            [ResponseStatus.ANSWERED],
            answers=["गलत लेकिन grounded उत्तर 1913"],
        )

        exit_code, _, _ = await self._run_offline(args, stt=stt, pipeline=pipeline)

        self.assertEqual(exit_code, 1)
        report = json.loads(output.read_text(encoding="utf-8"))
        trial = report["trials"][0]
        self.assertTrue(trial["grounded"])
        self.assertFalse(trial["quality"]["answer_contains_match"])
        self.assertFalse(trial["quality"]["answer_oracle_match"])
        self.assertFalse(trial["quality"]["eligible"])
        self.assertFalse(report["gates"]["benchmark_passed"])

    async def test_manifest_without_transcript_reference_cannot_pass_gate(self) -> None:
        manifest = self.directory / "no-reference-samples.json"
        manifest.write_text(
            json.dumps(
                [
                    {
                        "audio": self.audio_one.name,
                        "expected_status": "answered",
                        "expected_answer_contains": "सही सवाल",
                    }
                ]
            ),
            encoding="utf-8",
        )
        output = self.directory / "no-reference-report.json"
        args = benchmark.build_parser().parse_args(
            [
                "--manifest",
                str(manifest),
                "--trials",
                "1",
                "--no-realtime-pacing",
                "--target-ms",
                "1000000",
                "--output",
                str(output),
            ]
        )
        stt = _FakeSTT([_transcription("सही सवाल")])
        pipeline = _FakePipeline([ResponseStatus.ANSWERED])

        exit_code, _, _ = await self._run_offline(args, stt=stt, pipeline=pipeline)

        self.assertEqual(exit_code, 1)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["quality"]["outcome_oracle_eligible_trials"], 1)
        self.assertEqual(report["quality"]["eligible_trials"], 0)
        self.assertEqual(report["quality"]["referenced_trials"], 0)
        self.assertFalse(report["quality"]["all_attempts_eligible"])
        self.assertFalse(report["gates"]["benchmark_passed"])

    async def test_latency_only_mode_is_explicitly_non_submission(self) -> None:
        output = self.directory / "latency-only-report.json"
        args = benchmark.build_parser().parse_args(
            [
                str(self.audio_one),
                "--trials",
                "1",
                "--no-realtime-pacing",
                "--latency-only",
                "--target-ms",
                "1000000",
                "--output",
                str(output),
            ]
        )
        stt = _FakeSTT([_transcription("एक सवाल")])
        pipeline = _FakePipeline([ResponseStatus.REFUSED])

        exit_code, _, _ = await self._run_offline(args, stt=stt, pipeline=pipeline)

        self.assertEqual(exit_code, 0)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(
            report["configuration"]["benchmark_mode"],
            "latency_only_non_submission_smoke",
        )
        self.assertFalse(report["quality"]["gate_enabled"])
        self.assertTrue(report["gates"]["benchmark_passed"])
        self.assertFalse(report["gates"]["component_evidence_eligible"])

    async def test_unsupported_rag_language_rejects_before_stt_preflight(self) -> None:
        args = benchmark.build_parser().parse_args(
            [str(self.audio_one), "--no-realtime-pacing"]
        )
        stt = _FakeSTT([_transcription("एक सवाल")])
        pipeline = _FakePipeline([ResponseStatus.ANSWERED])
        runtime_loader = AsyncMock(return_value=_runtime(pipeline, languages=("ta",)))
        stt_factory = Mock(return_value=stt)
        with (
            patch.object(benchmark, "load_runtime", runtime_loader),
            patch.object(benchmark, "ElevenLabsStreamingSTT", stt_factory),
            patch.object(benchmark, "load_dotenv"),
            self.assertRaisesRegex(ValueError, "does not cover"),
        ):
            await benchmark.run(args)

        self.assertEqual(runtime_loader.await_count, 1)
        stt_factory.assert_not_called()
        self.assertEqual(stt.token_broker.preflight_calls, 0)


if __name__ == "__main__":
    unittest.main()
