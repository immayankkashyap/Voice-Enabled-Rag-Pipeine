from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Self

from app.generation import (
    REFUSAL_ANSWER,
    GroqAnswerGenerator,
    GroqGenerationError,
    GroqGenerationSettings,
)
from tests.test_generation import synthetic_fixture_request


class _FakeStream:
    def __init__(self, fragments: tuple[str, ...], finish_reason: str) -> None:
        events = []
        for index, fragment in enumerate(fragments):
            events.append(
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=fragment),
                            finish_reason=(
                                finish_reason if index == len(fragments) - 1 else None
                            ),
                        )
                    ]
                )
            )
        self._events = iter(events)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self) -> object:
        try:
            return next(self._events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _FakeCompletions:
    def __init__(self, fragments: tuple[str, ...], finish_reason: str) -> None:
        self.arguments: dict[str, object] | None = None
        self.fragments = fragments
        self.finish_reason = finish_reason

    async def create(self, **kwargs: object) -> _FakeStream:
        self.arguments = kwargs
        return _FakeStream(self.fragments, self.finish_reason)


class _FakeClient:
    def __init__(
        self,
        fragments: tuple[str, ...] = (
            "Sundarpur was founded ",
            "in 1912 [chunk:fixture-1].",
        ),
        finish_reason: str = "stop",
    ) -> None:
        self.completions = _FakeCompletions(fragments, finish_reason)
        self.chat = SimpleNamespace(completions=self.completions)

    async def close(self) -> None:
        return None


class GroqAnswerGeneratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_streams_answer_and_accepts_only_known_citations(self) -> None:
        client = _FakeClient()
        generator = GroqAnswerGenerator(
            GroqGenerationSettings(api_key="test-key", max_attempts=1),
            client=client,  # type: ignore[arg-type]
        )
        result = await generator.generate(synthetic_fixture_request())

        self.assertIn("founded in 1912", result.answer)
        self.assertEqual(result.cited_chunk_ids, ["fixture-1"])
        self.assertEqual(result.finish_reason, "stop")
        self.assertGreaterEqual(result.time_to_first_token_ms, 0)
        self.assertGreaterEqual(result.total_ms, result.time_to_first_token_ms)
        assert client.completions.arguments is not None
        self.assertTrue(client.completions.arguments["stream"])
        self.assertEqual(client.completions.arguments["model"], "qwen/qwen3.6-27b")
        self.assertEqual(client.completions.arguments["max_completion_tokens"], 96)
        self.assertEqual(client.completions.arguments["reasoning_effort"], "none")
        self.assertNotIn("service_tier", client.completions.arguments)
        messages = client.completions.arguments["messages"]
        self.assertIn("TEST FIXTURE ONLY", str(messages))

    async def test_allows_exact_grounded_refusal_without_a_citation(self) -> None:
        client = _FakeClient((REFUSAL_ANSWER,), "stop")
        generator = GroqAnswerGenerator(
            GroqGenerationSettings(api_key="test-key", max_attempts=1),
            client=client,  # type: ignore[arg-type]
        )

        result = await generator.generate(synthetic_fixture_request())

        self.assertEqual(result.answer, REFUSAL_ANSWER)
        self.assertEqual(result.cited_chunk_ids, [])

    async def test_rejects_non_refusal_without_a_citation(self) -> None:
        client = _FakeClient(("Sundarpur was founded in 1912.",), "stop")
        generator = GroqAnswerGenerator(
            GroqGenerationSettings(api_key="test-key", max_attempts=1),
            client=client,  # type: ignore[arg-type]
        )

        with self.assertRaisesRegex(GroqGenerationError, "without citations"):
            await generator.generate(synthetic_fixture_request())

    async def test_rejects_unknown_citation(self) -> None:
        client = _FakeClient(("It was founded in 1912 [chunk:invented].",), "stop")
        generator = GroqAnswerGenerator(
            GroqGenerationSettings(api_key="test-key", max_attempts=1),
            client=client,  # type: ignore[arg-type]
        )

        with self.assertRaisesRegex(GroqGenerationError, "invented"):
            await generator.generate(synthetic_fixture_request())

    async def test_rejects_truncated_answer(self) -> None:
        client = _FakeClient(("It was founded [chunk:fixture-1]",), "length")
        generator = GroqAnswerGenerator(
            GroqGenerationSettings(api_key="test-key", max_attempts=1),
            client=client,  # type: ignore[arg-type]
        )

        with self.assertRaisesRegex(GroqGenerationError, "finish_reason='length'"):
            await generator.generate(synthetic_fixture_request())

    def test_validates_model_specific_reasoning_effort(self) -> None:
        with self.assertRaisesRegex(ValueError, "Qwen 3"):
            GroqGenerationSettings(api_key="test-key", reasoning_effort="low")
        with self.assertRaisesRegex(ValueError, "GPT-OSS"):
            GroqGenerationSettings(
                api_key="test-key",
                model="openai/gpt-oss-20b",
                reasoning_effort="none",
            )
        with self.assertRaisesRegex(ValueError, "only supported"):
            GroqGenerationSettings(
                api_key="test-key",
                model="allam-2-7b",
                reasoning_effort="none",
            )


if __name__ == "__main__":
    unittest.main()
