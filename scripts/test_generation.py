#!/usr/bin/env python3
"""Repeatable live Groq check using explicitly synthetic fixtures."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
import unicodedata
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import AsyncGroq

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.generation import GroqAnswerGenerator, GroqGenerationSettings
from app.schemas import (
    Chunk,
    ChunkStrategy,
    GenerationRequest,
    RetrievedChunk,
)


def synthetic_fixture_request() -> GenerationRequest:
    """Return fake facts solely for exercising the generation boundary."""

    fixture_texts = (
        "TEST FIXTURE ONLY: The fictional city of Sundarpur was founded in 1912.",
        "TEST FIXTURE ONLY: Sundarpur's fictional civic bird is the blue kite.",
        "TEST FIXTURE ONLY: The fictional 2024 census counted 42,500 residents.",
    )
    context: list[RetrievedChunk] = []
    for index, text in enumerate(fixture_texts, start=1):
        context.append(
            RetrievedChunk(
                chunk=Chunk(
                    id=f"fixture-{index}",
                    document_id="synthetic-generation-fixture",
                    text=text,
                    strategy=ChunkStrategy.NAIVE_RECURSIVE,
                    metadata={"fixture": True},
                ),
                mrl_score=0.9 - index / 100,
                full_score=0.95 - index / 100,
                rank=index,
            )
        )
    return GenerationRequest(
        query="When was Sundarpur founded and what is its civic bird?",
        context=context,
        language_code="en-IN",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated grounded Groq generations and report TTFT/total "
            "P50, P70, P100, failures, and answer stability."
        )
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--model", default=None, help="Override the configured model")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Override the bounded completion-token budget",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "default", "low", "medium", "high"),
        default=None,
        help="Model-compatible reasoning setting (Qwen defaults to none)",
    )
    parser.add_argument(
        "--service-tier",
        choices=("auto", "on_demand", "flex", "performance"),
        default=None,
        help="Request an optional Groq service tier (default: provider plan default)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optionally atomically write the complete secret-free report as JSON.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List model IDs available to the configured API key and exit.",
    )
    return parser


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return a linearly interpolated percentile without NumPy."""

    if not values:
        raise ValueError("Cannot compute a percentile over an empty sequence")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float | None]) -> dict[str, float | int | None]:
    measured = [float(value) for value in values if value is not None]
    if not measured:
        return {
            "samples": 0,
            "missing": len(values),
            "mean": None,
            "p50": None,
            "p70": None,
            "p100": None,
        }
    return {
        "samples": len(measured),
        "missing": len(values) - len(measured),
        "mean": sum(measured) / len(measured),
        "p50": _percentile(measured, 50),
        "p70": _percentile(measured, 70),
        "p100": max(measured),
    }


def _normalized_answer(answer: str) -> str:
    normalized = unicodedata.normalize("NFKC", answer).casefold()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _success_trial(trial: int, result: Any) -> dict[str, Any]:
    return {
        "trial": trial,
        "status": "success",
        "answer": result.answer,
        "cited_chunk_ids": result.cited_chunk_ids,
        "finish_reason": result.finish_reason,
        "latency_ms": {
            "time_to_first_token": result.time_to_first_token_ms,
            "total": result.total_ms,
        },
    }


def _build_report(
    *,
    trials: list[dict[str, Any]],
    settings: GroqGenerationSettings,
    request: GenerationRequest,
) -> dict[str, Any]:
    successful = [trial for trial in trials if trial["status"] == "success"]
    answers = [str(trial["answer"]) for trial in successful]
    exact_counts = Counter(answers)
    normalized_counts = Counter(_normalized_answer(answer) for answer in answers)
    modal_answer, modal_count = (
        exact_counts.most_common(1)[0] if exact_counts else (None, 0)
    )
    failure_counts = Counter(
        str(trial["error_type"]) for trial in trials if trial["status"] == "failed"
    )

    def latencies(field: str) -> list[float | None]:
        return [
            float(trial["latency_ms"][field]) if trial["status"] == "success" else None
            for trial in trials
        ]

    attempted = len(trials)
    return {
        "schema_version": 1,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "groq",
        "fixture": {
            "name": "synthetic-generation-fixture",
            "query": request.query,
            "context_chunk_ids": [item.chunk.id for item in request.context],
        },
        # Deliberately enumerate safe configuration fields instead of serializing
        # the settings object, which also contains the API key.
        "configuration": {
            "trials": attempted,
            "model": settings.model,
            "max_output_tokens": settings.max_output_tokens,
            "temperature": settings.temperature,
            "timeout_seconds": settings.timeout_seconds,
            "max_attempts": settings.max_attempts,
            "reasoning_effort": settings.reasoning_effort,
            "service_tier": settings.service_tier,
        },
        "outcomes": {
            "attempted": attempted,
            "successful": len(successful),
            "failed": attempted - len(successful),
            "success_rate": len(successful) / attempted if attempted else 0.0,
            "failures_by_type": dict(sorted(failure_counts.items())),
        },
        "latency_ms": {
            "time_to_first_token": _distribution(latencies("time_to_first_token")),
            "total": _distribution(latencies("total")),
        },
        "answer_stability": {
            "successful_trials": len(successful),
            "unique_answers": len(exact_counts),
            "normalized_unique_answers": len(normalized_counts),
            "all_identical": len(exact_counts) == 1 if successful else False,
            "modal_answer": modal_answer,
            "modal_count": modal_count,
            "modal_rate": modal_count / len(successful) if successful else 0.0,
            "variants": [
                {
                    "answer": answer,
                    "sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
                    "count": count,
                }
                for answer, count in exact_counts.most_common()
            ],
        },
        "trials": trials,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    """Atomically replace a JSON report without ever persisting credentials."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(report, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


async def run(args: argparse.Namespace) -> int:
    if args.trials <= 0:
        raise ValueError("--trials must be positive")

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        print("GROQ_API_KEY is missing from the environment/.env", file=sys.stderr)
        return 2

    if args.list_models:
        client = AsyncGroq(api_key=api_key, max_retries=0)
        try:
            models = await client.models.list()
        finally:
            await client.close()
        for model in sorted(models.data, key=lambda item: item.id):
            print(model.id)
        return 0

    settings_kwargs: dict[str, Any] = {"api_key": api_key}
    if args.model:
        settings_kwargs["model"] = args.model
    if args.max_output_tokens is not None:
        settings_kwargs["max_output_tokens"] = args.max_output_tokens
    if args.reasoning_effort is not None:
        settings_kwargs["reasoning_effort"] = args.reasoning_effort
    if args.service_tier is not None:
        settings_kwargs["service_tier"] = args.service_tier
    settings = GroqGenerationSettings(**settings_kwargs)
    generator = GroqAnswerGenerator(settings)
    request = synthetic_fixture_request()
    measured: list[dict[str, Any]] = []
    try:
        for trial_number in range(1, args.trials + 1):
            trial_started = time.perf_counter()
            try:
                result = await generator.generate(request)
            # Provider errors are measured outcomes. Error messages are omitted
            # from both console and JSON so credentials can never leak through a
            # third-party exception string.
            except Exception as exc:  # noqa: BLE001
                failed = {
                    "trial": trial_number,
                    "status": "failed",
                    "elapsed_ms": (time.perf_counter() - trial_started) * 1000,
                    "error_type": type(exc).__name__,
                }
                measured.append(failed)
                print(
                    f"trial {trial_number}/{args.trials}: FAILED "
                    f"({failed['error_type']})",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            trial = _success_trial(trial_number, result)
            measured.append(trial)
            latency = trial["latency_ms"]
            print(
                f"trial {trial_number}/{args.trials}: "
                f"TTFT={latency['time_to_first_token']:.3f} ms; "
                f"total={latency['total']:.3f} ms; answer={result.answer!r}",
                flush=True,
            )
    finally:
        await generator.aclose()

    report = _build_report(trials=measured, settings=settings, request=request)
    if args.output is not None:
        _write_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["outcomes"]["failed"] == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
