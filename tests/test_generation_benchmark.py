from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from app.schemas import GenerationResult
from scripts import test_generation

_ANSWER = (
    "Sundarpur was founded in 1912 [chunk:fixture-1] and its civic bird is "
    "the blue kite [chunk:fixture-2]."
)


class _FakeGroqAnswerGenerator:
    instances: ClassVar[list[_FakeGroqAnswerGenerator]] = []

    def __init__(self, settings: object) -> None:
        self.settings = settings
        self.call_count = 0
        self.closed = False
        self.instances.append(self)

    async def generate(self, request: object) -> GenerationResult:
        self.call_count += 1
        ttft = (10.0, 20.0, 40.0)[self.call_count - 1]
        total = (20.0, 30.0, 50.0)[self.call_count - 1]
        answer = _ANSWER if self.call_count < 3 else _ANSWER.lower()
        return GenerationResult(
            answer=answer,
            cited_chunk_ids=["fixture-1", "fixture-2"],
            model="qwen/qwen3.6-27b",
            finish_reason="stop",
            time_to_first_token_ms=ttft,
            total_ms=total,
        )

    async def aclose(self) -> None:
        self.closed = True


class _FailingOnceGroqAnswerGenerator:
    instances: ClassVar[list[_FailingOnceGroqAnswerGenerator]] = []

    def __init__(self, settings: object) -> None:
        self.settings = settings
        self.call_count = 0
        self.closed = False
        self.instances.append(self)

    async def generate(self, request: object) -> GenerationResult:
        self.call_count += 1
        if self.call_count == 1:
            raise RuntimeError("provider exposed super-secret")
        return GenerationResult(
            answer=_ANSWER,
            cited_chunk_ids=["fixture-1", "fixture-2"],
            model="qwen/qwen3.6-27b",
            finish_reason="stop",
            time_to_first_token_ms=25.0,
            total_ms=35.0,
        )

    async def aclose(self) -> None:
        self.closed = True


class GenerationBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeGroqAnswerGenerator.instances.clear()
        _FailingOnceGroqAnswerGenerator.instances.clear()

    def test_distribution_uses_interpolated_percentiles(self) -> None:
        distribution = test_generation._distribution([10.0, 20.0, 40.0, None])

        self.assertEqual(distribution["samples"], 3)
        self.assertEqual(distribution["missing"], 1)
        self.assertAlmostEqual(distribution["mean"], 70 / 3)
        self.assertEqual(distribution["p50"], 20.0)
        self.assertEqual(distribution["p70"], 28.0)
        self.assertEqual(distribution["p100"], 40.0)

    def test_reuses_one_client_and_writes_atomic_secret_free_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "generation.json"
            args = test_generation.build_parser().parse_args(
                ["--trials", "3", "--output", str(output)]
            )

            with (
                patch.dict(os.environ, {"GROQ_API_KEY": "super-secret"}),
                patch.object(test_generation, "load_dotenv"),
                patch.object(
                    test_generation,
                    "GroqAnswerGenerator",
                    _FakeGroqAnswerGenerator,
                ),
                patch("builtins.print"),
            ):
                exit_code = asyncio.run(test_generation.run(args))

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(_FakeGroqAnswerGenerator.instances), 1)
            instance = _FakeGroqAnswerGenerator.instances[0]
            self.assertEqual(instance.call_count, 3)
            self.assertTrue(instance.closed)

            raw_report = output.read_text(encoding="utf-8")
            self.assertNotIn("super-secret", raw_report)
            self.assertNotIn("api_key", raw_report)
            report = json.loads(raw_report)
            self.assertEqual(report["configuration"]["trials"], 3)
            self.assertIsNone(report["configuration"]["service_tier"])
            self.assertEqual(report["outcomes"]["successful"], 3)
            self.assertEqual(report["latency_ms"]["time_to_first_token"]["p50"], 20.0)
            self.assertEqual(report["latency_ms"]["time_to_first_token"]["p70"], 28.0)
            self.assertEqual(report["latency_ms"]["time_to_first_token"]["p100"], 40.0)
            self.assertEqual(report["latency_ms"]["total"]["mean"], 100 / 3)
            self.assertEqual(report["latency_ms"]["total"]["p70"], 38.0)
            self.assertEqual(report["latency_ms"]["total"]["p100"], 50.0)
            self.assertEqual(report["answer_stability"]["unique_answers"], 2)
            self.assertEqual(report["answer_stability"]["normalized_unique_answers"], 1)
            self.assertEqual(report["answer_stability"]["modal_count"], 2)
            self.assertAlmostEqual(report["answer_stability"]["modal_rate"], 2 / 3)
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_failed_trial_is_counted_without_persisting_exception_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generation.json"
            args = test_generation.build_parser().parse_args(
                ["--trials", "2", "--output", str(output)]
            )

            with (
                patch.dict(os.environ, {"GROQ_API_KEY": "super-secret"}),
                patch.object(test_generation, "load_dotenv"),
                patch.object(
                    test_generation,
                    "GroqAnswerGenerator",
                    _FailingOnceGroqAnswerGenerator,
                ),
                patch("builtins.print"),
            ):
                exit_code = asyncio.run(test_generation.run(args))

            self.assertEqual(exit_code, 1)
            raw_report = output.read_text(encoding="utf-8")
            self.assertNotIn("super-secret", raw_report)
            self.assertNotIn("provider exposed", raw_report)
            report = json.loads(raw_report)
            self.assertEqual(report["outcomes"]["failed"], 1)
            self.assertEqual(
                report["outcomes"]["failures_by_type"], {"RuntimeError": 1}
            )
            self.assertEqual(report["latency_ms"]["time_to_first_token"]["samples"], 1)
            self.assertEqual(report["latency_ms"]["time_to_first_token"]["missing"], 1)
            self.assertEqual(report["trials"][0]["error_type"], "RuntimeError")
            self.assertNotIn("error", report["trials"][0])
            self.assertTrue(_FailingOnceGroqAnswerGenerator.instances[0].closed)

    def test_rejects_non_positive_trial_count_before_loading_credentials(self) -> None:
        args = test_generation.build_parser().parse_args(["--trials", "0"])

        with (
            patch.object(test_generation, "load_dotenv") as load_dotenv,
            self.assertRaisesRegex(ValueError, "--trials must be positive"),
        ):
            asyncio.run(test_generation.run(args))

        load_dotenv.assert_not_called()


if __name__ == "__main__":
    unittest.main()
