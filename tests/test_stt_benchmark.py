from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from app.schemas import TranscriptionResult
from scripts import test_stt


class _FakeSarvamStreamingSTT:
    transcripts = (
        "What is the capital of India?",
        "What is the capital of India?",
        "what is the capital of India",
    )

    def __init__(self, settings: object) -> None:
        self.settings = settings
        self.call_count = 0

    async def transcribe(self, audio_chunks, *, on_partial=None):
        async for _ in audio_chunks:
            pass
        transcript = self.transcripts[self.call_count]
        self.call_count += 1
        if on_partial is not None:
            on_partial("What is")
        multiplier = float(self.call_count)
        return TranscriptionResult(
            transcript=transcript,
            language_code="en-IN",
            is_final=True,
            partial_transcripts=["What is"],
            connection_ms=10.0 * multiplier,
            time_to_first_partial_ms=20.0 * multiplier,
            time_to_final_transcript_ms=30.0 * multiplier,
            final_after_audio_end_ms=5.0 * multiplier,
            total_ms=31.0 * multiplier,
        )


class _FailingOnceSarvamStreamingSTT:
    def __init__(self, settings: object) -> None:
        self.settings = settings
        self.call_count = 0

    async def transcribe(self, audio_chunks, *, on_partial=None):
        async for _ in audio_chunks:
            pass
        self.call_count += 1
        if self.call_count == 1:
            raise RuntimeError("provider rejected super-secret")
        return TranscriptionResult(
            transcript="Recovered transcript",
            language_code="en-IN",
            is_final=True,
            partial_transcripts=[],
            connection_ms=10.0,
            time_to_first_partial_ms=None,
            time_to_final_transcript_ms=20.0,
            final_after_audio_end_ms=5.0,
            total_ms=21.0,
        )


def _write_test_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 160)


class STTBenchmarkTests(unittest.TestCase):
    def test_distribution_reports_missing_and_interpolated_percentiles(self) -> None:
        distribution = test_stt._distribution([10.0, 20.0, 30.0, 40.0, None])

        self.assertEqual(distribution["samples"], 4)
        self.assertEqual(distribution["missing"], 1)
        self.assertEqual(distribution["p50"], 25.0)
        self.assertAlmostEqual(distribution["p70"], 31.0)
        self.assertEqual(distribution["p100"], 40.0)

    def test_reference_file_is_trimmed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.txt"
            reference.write_text("  नमस्ते दुनिया\n", encoding="utf-8")
            args = argparse.Namespace(reference=None, reference_file=reference)

            self.assertEqual(test_stt._reference_text(args), "नमस्ते दुनिया")

    def test_offline_fake_trials_write_secret_free_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "sample.wav"
            output = root / "report.json"
            _write_test_wav(audio)
            args = test_stt.build_parser().parse_args(
                [
                    str(audio),
                    "--trials",
                    "3",
                    "--mode",
                    "transcribe",
                    "--endpointing",
                    "manual",
                    "--chunk-ms",
                    "250",
                    "--no-realtime-pacing",
                    "--reference",
                    "What is the capital of India?",
                    "--output",
                    str(output),
                ]
            )

            with (
                patch.dict(os.environ, {"SARVAM_API_KEY": "super-secret"}),
                patch.object(test_stt, "load_dotenv"),
                patch.object(test_stt, "SarvamStreamingSTT", _FakeSarvamStreamingSTT),
                patch("builtins.print"),
            ):
                exit_code = asyncio.run(test_stt.run(args))

            self.assertEqual(exit_code, 0)
            raw_report = output.read_text(encoding="utf-8")
            self.assertNotIn("super-secret", raw_report)
            report = json.loads(raw_report)
            self.assertEqual(report["outcomes"]["attempted"], 3)
            self.assertEqual(report["outcomes"]["successful"], 3)
            self.assertEqual(report["configuration"]["chunk_ms"], 250)
            self.assertEqual(report["latency_ms"]["connection"]["p50"], 20.0)
            self.assertEqual(report["latency_ms"]["connection"]["p70"], 24.0)
            self.assertEqual(report["latency_ms"]["connection"]["p100"], 30.0)
            self.assertEqual(report["transcript_stability"]["unique_transcripts"], 2)
            self.assertAlmostEqual(report["transcript_stability"]["modal_rate"], 2 / 3)
            self.assertEqual(
                report["reference_quality"]["strict_exact_match"]["matches"], 2
            )
            self.assertEqual(
                report["reference_quality"]["normalized_exact_match"]["matches"], 3
            )
            self.assertEqual(report["audio"]["file"], "sample.wav")
            self.assertNotIn(str(root), raw_report)

    def test_failed_trial_is_counted_and_secret_is_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "sample.wav"
            output = root / "report.json"
            _write_test_wav(audio)
            args = test_stt.build_parser().parse_args(
                [
                    str(audio),
                    "--trials",
                    "2",
                    "--no-realtime-pacing",
                    "--output",
                    str(output),
                ]
            )

            with (
                patch.dict(os.environ, {"SARVAM_API_KEY": "super-secret"}),
                patch.object(test_stt, "load_dotenv"),
                patch.object(
                    test_stt,
                    "SarvamStreamingSTT",
                    _FailingOnceSarvamStreamingSTT,
                ),
                patch("builtins.print"),
            ):
                exit_code = asyncio.run(test_stt.run(args))

            self.assertEqual(exit_code, 1)
            raw_report = output.read_text(encoding="utf-8")
            self.assertNotIn("super-secret", raw_report)
            report = json.loads(raw_report)
            self.assertEqual(report["outcomes"]["attempted"], 2)
            self.assertEqual(report["outcomes"]["failed"], 1)
            self.assertEqual(report["latency_ms"]["connection"]["samples"], 1)
            self.assertEqual(
                report["latency_ms"]["first_partial_from_start"]["missing"], 1
            )
            self.assertEqual(
                report["trials"][0]["error"], "provider rejected [REDACTED]"
            )


if __name__ == "__main__":
    unittest.main()
