#!/usr/bin/env python3
"""Benchmark the real warmed text-RAG runtime over versioned multilingual queries."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.runtime import RuntimeSettings, load_runtime
from app.schemas import ErrorResponse, RAGRequest, RAGResponse

_STAGES = (
    ("stt", "stt_ms"),
    ("input_safety", "input_safety_ms"),
    ("query_embedding", "query_embedding_ms"),
    ("retrieval_stage_1", "retrieval_stage_1_ms"),
    ("retrieval_stage_2", "retrieval_stage_2_ms"),
    ("retrieval_total", "retrieval_ms"),
    ("relevance_guardrail", "relevance_ms"),
    ("generation", "generation_ms"),
    ("groundedness_guardrail", "groundedness_ms"),
    ("output", "output_ms"),
    ("total", "total_ms"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/faiss_index/source_queries.json"),
    )
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--index-dir", type=Path, default=Path("data/faiss_index"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--target-ms", type=float, default=200.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmark_report.json"),
    )
    return parser


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile over no values")
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


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    measured = [float(value) for value in values]
    if not measured:
        return {
            "samples": 0,
            "mean": None,
            "p50": None,
            "p70": None,
            "p100": None,
        }
    return {
        "samples": len(measured),
        "mean": sum(measured) / len(measured),
        "p50": _percentile(measured, 50),
        "p70": _percentile(measured, 70),
        "p100": max(measured),
    }


def _load_queries(path: Path, limit: int) -> list[dict[str, str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read benchmark queries from {path}") from exc
    if not isinstance(payload, list):
        raise ValueError("Benchmark query file must contain a JSON list")
    queries: list[dict[str, str]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Query row {index} is not an object")
        try:
            query_id = str(item["id"]).strip()
            query = str(item["query"]).strip()
            language = str(item["language"]).strip().lower()
        except KeyError as exc:
            raise ValueError(f"Query row {index} is missing {exc.args[0]!r}") from exc
        if not query_id or not query or not language:
            raise ValueError(f"Query row {index} contains a blank required field")
        queries.append(
            {
                "id": query_id,
                "query": query,
                "language": language,
                "query_id": str(item.get("query_id") or query_id),
                "english_query": str(item.get("english_query") or ""),
            }
        )
    if limit <= 0:
        raise ValueError("--limit must be positive")
    selected = queries[:limit]
    if len(selected) < 30:
        raise ValueError(
            f"Full benchmark requires at least 30 queries; only {len(selected)} selected"
        )
    return selected


def _write_report(path: Path, report: dict[str, Any]) -> None:
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


def _latency_dict(response: RAGResponse | ErrorResponse) -> dict[str, float]:
    latencies = response.latencies
    if latencies is None:
        raise ValueError("Every benchmark response must include StageLatencies")
    payload = latencies.model_dump(mode="json")
    values: dict[str, float] = {}
    for stage, field in _STAGES:
        raw = payload.get(field)
        if raw is None:
            raise ValueError(f"Stage latency {field} is missing")
        values[stage] = float(raw)
    return values


def _query_result(
    sample: dict[str, str],
    response: RAGResponse | ErrorResponse,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": sample["id"],
        "query_id": sample["query_id"],
        "language": sample["language"],
        "query": sample["query"],
        "english_query": sample["english_query"],
        "latency_ms": _latency_dict(response),
    }
    if isinstance(response, ErrorResponse):
        return {
            **base,
            "outcome": "error",
            "error_code": response.error_code,
            "error_message": response.message,
            "error_stage": response.stage,
            "retryable": response.retryable,
        }
    return {
        **base,
        "outcome": "completed",
        "status": response.status.value,
        "refusal_reason": (
            response.refusal_reason.value if response.refusal_reason else None
        ),
        "answer": response.answer,
        "retrieved_chunk_ids": [item.chunk.id for item in response.retrieved_chunks],
        "relevance_classification": (
            response.relevance.classification.value if response.relevance else None
        ),
        "grounded": (
            response.groundedness.is_grounded if response.groundedness else None
        ),
        "target_met": response.latencies.target_met,
    }


def _summary_table(report: dict[str, Any]) -> str:
    header = f"{'Stage':<28} {'Samples':>7} {'P50 ms':>12} {'P70 ms':>12} {'P100 ms':>12}"
    lines = [header, "-" * len(header)]
    for stage, _ in _STAGES:
        values = report["latency_ms"][stage]
        lines.append(
            f"{stage:<28} {values['samples']:>7d} "
            f"{values['p50']:>12.3f} {values['p70']:>12.3f} "
            f"{values['p100']:>12.3f}"
        )
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> int:
    if args.candidate_k <= 0 or args.final_k <= 0:
        raise ValueError("candidate counts must be positive")
    if args.final_k > args.candidate_k:
        raise ValueError("--final-k cannot exceed --candidate-k")
    if args.target_ms <= 0:
        raise ValueError("--target-ms must be positive")
    queries = _load_queries(args.queries, args.limit)
    load_dotenv()
    environment_settings = RuntimeSettings.from_environment()
    settings = RuntimeSettings(
        index_dir=args.index_dir,
        device=(args.device if args.device is not None else environment_settings.device),
        cpu_threads=environment_settings.cpu_threads,
        preload=True,
        latency_target_ms=args.target_ms,
        groq_api_key=environment_settings.groq_api_key,
        generation_model=environment_settings.generation_model,
        generation_max_output_tokens=(
            environment_settings.generation_max_output_tokens
        ),
        generation_max_attempts=environment_settings.generation_max_attempts,
        generation_min_fallback_budget_ms=(
            environment_settings.generation_min_fallback_budget_ms
        ),
    )
    runtime = await load_runtime(settings)
    measured: list[dict[str, Any]] = []
    try:
        for index, sample in enumerate(queries, start=1):
            response = await runtime.pipeline.answer(
                RAGRequest(
                    query=sample["query"],
                    language_code=sample["language"],
                    candidate_k=args.candidate_k,
                    final_k=args.final_k,
                )
            )
            result = _query_result(sample, response)
            measured.append(result)
            print(
                f"{index:02d}/{len(queries)} {sample['id']}: "
                f"{result.get('status', result['outcome'])} "
                f"{result['latency_ms']['total']:.3f} ms",
                flush=True,
            )
    finally:
        await runtime.aclose()

    latency_values = {
        stage: [float(item["latency_ms"][stage]) for item in measured]
        for stage, _ in _STAGES
    }
    latency_report = {
        stage: _distribution(values) for stage, values in latency_values.items()
    }
    executed_latency_report = {
        stage: _distribution([value for value in values if value > 0])
        for stage, values in latency_values.items()
    }
    errors = [item for item in measured if item["outcome"] == "error"]
    completed = [item for item in measured if item["outcome"] == "completed"]
    status_counts = Counter(str(item.get("status")) for item in completed)
    refusal_counts = Counter(
        str(item["refusal_reason"])
        for item in completed
        if item.get("refusal_reason") is not None
    )
    semantic_ids = {item["query_id"] for item in measured}
    report: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "full_text_rag_pipeline",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_boundary": (
            "Direct RuntimeServices.pipeline.answer call, explicitly permitted by "
            "the task; includes response validation but excludes HTTP transport and STT."
        ),
        "workload": {
            "source": str(args.queries),
            "dataset": "ai4bharat/MSMARCO-XI validation subset",
            "queries": len(measured),
            "unique_query_ids": len(semantic_ids),
            "languages": dict(Counter(item["language"] for item in measured)),
            "limitation": (
                "The 30 real language-specific queries are translations of 10 "
                "semantic query IDs and were used to build the bounded demo index; "
                "this is latency evidence, not held-out retrieval-quality evidence."
            ),
        },
        "configuration": {
            "index_dir": str(args.index_dir),
            "vector_count": runtime.vector_count,
            "embedding_model": runtime.embedding_model,
            "device": runtime.device,
            "candidate_k": args.candidate_k,
            "final_k": args.final_k,
            "generation_model": settings.generation_model,
            "answer_mode": runtime.answer_mode,
            "generation_fast_path_model": str(
                getattr(runtime.pipeline.generator, "fast_path_model", "none")
            ),
            "generation_max_output_tokens": settings.generation_max_output_tokens,
            "generation_min_fallback_budget_ms": (
                settings.generation_min_fallback_budget_ms
            ),
            "target_ms": args.target_ms,
            "runtime_load_ms": runtime.load_ms,
            "runtime_warmup_ms": runtime.warmup_ms,
        },
        "outcomes": {
            "attempted": len(measured),
            "completed": len(completed),
            "errors": len(errors),
            "statuses": dict(sorted(status_counts.items())),
            "refusals_by_reason": dict(sorted(refusal_counts.items())),
            "target_met": sum(
                item.get("target_met") is True for item in completed
            ),
            "generation_routing": {
                "extractive_fast_path_answers": int(
                    getattr(runtime.pipeline.generator, "fast_path_answers", 0)
                ),
                "groq_fallback_calls": int(
                    getattr(runtime.pipeline.generator, "remote_fallbacks", 0)
                ),
                "budget_skipped_groq_fallbacks": int(
                    getattr(
                        runtime.pipeline.generator,
                        "budget_skipped_fallbacks",
                        0,
                    )
                ),
            },
        },
        "latency_ms": latency_report,
        "executed_stage_latency_ms": executed_latency_report,
        "slowest_queries": sorted(
            measured,
            key=lambda item: float(item["latency_ms"]["total"]),
            reverse=True,
        )[:5],
        "queries": measured,
    }
    _write_report(args.output, report)
    print()
    print(_summary_table(report))
    print()
    total = latency_report["total"]
    print(
        f"TOTAL P50={total['p50']:.3f} ms  P70={total['p70']:.3f} ms  "
        f"P100={total['p100']:.3f} ms"
    )
    print(
        f"Outcomes: completed={len(completed)}, errors={len(errors)}, "
        f"answered={status_counts.get('answered', 0)}, "
        f"refused={status_counts.get('refused', 0)}"
    )
    print(f"Saved {args.output}")
    return 0 if not errors else 1


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
