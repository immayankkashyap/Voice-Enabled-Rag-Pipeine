#!/usr/bin/env python3
"""Integrated real-WAV benchmark for ElevenLabs through serialized RAG output.

The measured path is the same ordered path used by the service: streamed PCM,
committed transcript, warmed local retrieval, extractive generation, grounding,
and response serialization.  Client-to-server transport and browser rendering
are outside this process and are therefore not included or implied.

Every attempted trial remains in the report.  Provider or pipeline failures are
explicit SLA violations; they are never dropped to make the percentiles look
better.  Numeric P100 is the literal maximum of completed observations.  A
single WAV is a smoke/jitter workload; submission-quality latency evidence must
use at least 30 content-distinct, representative recordings and 30 unique
normalized transcript reference questions (directly or by manifest).
This is a component-direct runner: provider preflight happens before its WAV
clock, and it does not measure client-to-ASGI transport or final WebSocket
receipt.  A separate external live-service/browser benchmark is still required.

Example quality-gated manifest entry::

    {"audio": "question-01.wav", "reference": "...",
     "expected_status": "answered", "expected_answer_contains": ["1912"]}
"""

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
import wave
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.elevenlabs_stt import ElevenLabsStreamingSTT, ElevenLabsSTTSettings
from app.runtime import RuntimeSettings, load_runtime
from app.schemas import (
    RAGRequest,
    RAGResponse,
    RefusalReason,
    TranscriptionResult,
    VoicePipelineLatencies,
    VoiceRAGResponse,
)

_PCM_SAMPLE_RATE = 16_000
_PCM_SAMPLE_WIDTH_BYTES = 2
_METRIC_DEFINITION = (
    "first_audio_to_answer_ms is paced WAV capture start through final "
    "VoiceRAGResponse JSON serialization; "
    "audio_eof_to_answer_ms starts after the final PCM frame is yielded "
    "(paced in realtime mode). "
    "The configured 200 ms serving SLA is anchored at audio EOF; the separate "
    "first-audio metric exposes utterance duration and must not be conflated "
    "with post-utterance latency. Provider preflight, client/ASGI transport, "
    "final WebSocket receipt, and rendering are excluded."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated integrated ElevenLabs → local RAG trials and report "
            "P50/P70/true-max P100 without excluding failures."
        )
    )
    parser.add_argument(
        "audio",
        type=Path,
        nargs="*",
        help=(
            "One or more mono PCM16 16 kHz WAV files, assigned to trials in "
            "deterministic round-robin order. This input mode is smoke-only; "
            "submission-quality gating requires --manifest."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "JSON list (or {'samples': [...]}) of objects with 'audio' and "
            "optional 'reference', 'expected_status', and "
            "'expected_refusal_reason'. Answered samples must also provide "
            "expected_answer_contains, expected_chunk_ids, or "
            "expected_document_ids. Relative audio paths use the manifest folder."
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Measured trials; defaults to the submission minimum of 30.",
    )
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument(
        "--no-realtime-pacing",
        action="store_true",
        help=(
            "Upload PCM without wall-clock pacing. The report labels this as a "
            "throughput run, not live voice latency."
        ),
    )
    parser.add_argument(
        "--stt-language-code",
        default="hin",
        help="ElevenLabs language code; default matches the bundled Hindi index.",
    )
    parser.add_argument(
        "--rag-language-code",
        default="hi",
        help="Index language code; rejected before STT if the index lacks it.",
    )
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--target-ms", type=float, default=200.0)
    parser.add_argument("--index-dir", type=Path, default=Path("data/faiss_index"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--show-partials", action="store_true")
    parser.add_argument(
        "--latency-only",
        action="store_true",
        help=(
            "Relax answer/reference quality gating for a non-submission smoke "
            "run. Failures and latency misses still fail the run."
        ),
    )
    parser.add_argument(
        "--sequential",
        action="store_false",
        dest="overlap_enabled",
        help=(
            "Disable correctness-safe work started from ElevenLabs settled text. "
            "Overlap is enabled by default and is reused only after exact commit."
        ),
    )
    parser.set_defaults(overlap_enabled=True)
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--reference")
    reference.add_argument("--reference-file", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/voice_pipeline_benchmark.json"),
    )
    return parser


def wav_metadata(path: Path) -> dict[str, int | float | str]:
    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        sample_rate = audio.getframerate()
        frame_count = audio.getnframes()
        compression = audio.getcomptype()
    if (
        channels != 1
        or sample_width != _PCM_SAMPLE_WIDTH_BYTES
        or sample_rate != _PCM_SAMPLE_RATE
        or compression != "NONE"
    ):
        raise ValueError(
            "Voice benchmark audio must be uncompressed mono PCM16 WAV at 16000 Hz; "
            f"received channels={channels}, width={sample_width * 8}-bit, "
            f"rate={sample_rate}, compression={compression}"
        )
    if frame_count <= 0:
        raise ValueError("Voice benchmark WAV contains no audio frames")
    return {
        "file": path.name,
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "sample_width_bits": sample_width * 8,
        "frame_count": frame_count,
        "duration_ms": frame_count / sample_rate * 1000,
    }


@dataclass(slots=True)
class _AudioClock:
    first_audio_at: float | None = None
    eof_at: float | None = None


@dataclass(frozen=True, slots=True)
class _BenchmarkSample:
    source_id: str
    audio: Path
    reference: str | None = None
    expected_status: str = "answered"
    expected_refusal_reason: str | None = None
    expected_answer_contains: tuple[str, ...] = ()
    expected_chunk_ids: tuple[str, ...] = ()
    expected_document_ids: tuple[str, ...] = ()


async def wav_chunks(
    path: Path,
    *,
    chunk_ms: int,
    realtime_pacing: bool,
    clock: _AudioClock | None = None,
    _monotonic: Callable[[], float] = time.perf_counter,
    _sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> AsyncIterator[bytes]:
    wav_metadata(path)
    frames_per_chunk = max(1, _PCM_SAMPLE_RATE * chunk_ms // 1000)
    active_clock = clock or _AudioClock()
    stream_started_at = _monotonic()
    active_clock.first_audio_at = stream_started_at
    cumulative_audio_seconds = 0.0
    with wave.open(str(path), "rb") as audio:
        while chunk := audio.readframes(frames_per_chunk):
            cumulative_audio_seconds += len(chunk) / (
                _PCM_SAMPLE_WIDTH_BYTES * _PCM_SAMPLE_RATE
            )
            if realtime_pacing:
                remaining = stream_started_at + cumulative_audio_seconds - _monotonic()
                if remaining > 0:
                    await _sleep(remaining)
            yield chunk
    active_clock.eof_at = _monotonic()


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile without observations")
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


def _distribution(
    values: Sequence[float | None],
    *,
    failed_trials: int,
    target_ms: float | None = None,
) -> dict[str, float | int | bool | None]:
    measured = [float(value) for value in values if value is not None]
    result: dict[str, float | int | bool | None] = {
        "samples": len(measured),
        "missing_completed_samples": len(values) - len(measured),
        "failed_trials": failed_trials,
        "mean": sum(measured) / len(measured) if measured else None,
        "p50": _percentile(measured, 50) if measured else None,
        "p70": _percentile(measured, 70) if measured else None,
        # P100 is deliberately max, never an interpolated or selected run.
        "p100": max(measured) if measured else None,
        "p100_is_observed_max": True,
    }
    if target_ms is not None:
        result["all_attempts_meet_target"] = bool(
            measured
            and failed_trials == 0
            and len(measured) == len(values)
            and max(measured) <= target_ms
        )
    return result


def _normalized_transcript(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def _wav_content_digest(path: Path) -> bytes:
    """Hash decoded PCM samples so header-only changes do not fake diversity."""

    digest = hashlib.sha256()
    with wave.open(str(path), "rb") as audio_file:
        while block := audio_file.readframes(512 * 1024):
            digest.update(block)
    return digest.digest()


def _reference_text(args: argparse.Namespace) -> str | None:
    if args.reference is not None:
        return str(args.reference).strip()
    if args.reference_file is not None:
        return args.reference_file.read_text(encoding="utf-8").strip()
    return None


def _manifest_strings(value: Any, *, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    values = [value] if isinstance(value, str) else value
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(item, str) or not item.strip() for item in values)
    ):
        raise ValueError(f"{field} must be a non-empty string or list of strings")
    return tuple(dict.fromkeys(item.strip() for item in values))


def _load_samples(args: argparse.Namespace) -> list[_BenchmarkSample]:
    audio_paths = list(args.audio)
    if bool(audio_paths) == bool(args.manifest):
        raise ValueError("Provide WAV paths or --manifest, but not both")

    global_reference = _reference_text(args)
    if args.manifest is None:
        return [
            _BenchmarkSample(
                source_id=f"audio-{index:03d}",
                audio=path,
                reference=global_reference,
            )
            for index, path in enumerate(audio_paths, start=1)
        ]

    manifest_path = args.manifest
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Could not read a valid benchmark manifest") from exc
    entries = manifest.get("samples") if isinstance(manifest, dict) else manifest
    if not isinstance(entries, list) or not entries:
        raise ValueError("Benchmark manifest must contain a non-empty samples list")

    samples: list[_BenchmarkSample] = []
    for index, entry in enumerate(entries, start=1):
        allowed_fields = {
            "audio",
            "reference",
            "expected_status",
            "expected_refusal_reason",
            "expected_answer_contains",
            "expected_chunk_ids",
            "expected_document_ids",
        }
        if not isinstance(entry, dict) or set(entry) - allowed_fields:
            raise ValueError(
                "Manifest samples may contain only audio, reference, "
                "expected_status, expected_refusal_reason, "
                "expected_answer_contains, expected_chunk_ids, and "
                "expected_document_ids"
            )
        audio_value = entry.get("audio")
        if not isinstance(audio_value, str) or not audio_value.strip():
            raise ValueError("Each manifest sample requires a non-empty audio path")
        reference_value = entry.get("reference", global_reference)
        if reference_value is not None and (
            not isinstance(reference_value, str) or not reference_value.strip()
        ):
            raise ValueError("Manifest references must be non-empty strings")
        expected_status = entry.get("expected_status", "answered")
        if not isinstance(expected_status, str) or expected_status not in {
            "answered",
            "refused",
        }:
            raise ValueError("expected_status must be 'answered' or 'refused'")
        expected_refusal_reason = entry.get("expected_refusal_reason")
        allowed_refusal_reasons = {reason.value for reason in RefusalReason}
        if expected_refusal_reason is not None and (
            not isinstance(expected_refusal_reason, str)
            or expected_status != "refused"
            or expected_refusal_reason not in allowed_refusal_reasons
        ):
            raise ValueError(
                "expected_refusal_reason requires expected_status='refused' and "
                "a known refusal reason"
            )
        expected_answer_contains = _manifest_strings(
            entry.get("expected_answer_contains"),
            field="expected_answer_contains",
        )
        if any(not _normalized_transcript(value) for value in expected_answer_contains):
            raise ValueError(
                "expected_answer_contains values must contain letters or numbers"
            )
        expected_chunk_ids = _manifest_strings(
            entry.get("expected_chunk_ids"),
            field="expected_chunk_ids",
        )
        expected_document_ids = _manifest_strings(
            entry.get("expected_document_ids"),
            field="expected_document_ids",
        )
        has_answer_oracle = bool(
            expected_answer_contains or expected_chunk_ids or expected_document_ids
        )
        if expected_status == "answered" and not has_answer_oracle:
            raise ValueError(
                "Every expected answered manifest sample requires an answer oracle"
            )
        if expected_status == "refused" and has_answer_oracle:
            raise ValueError("Expected refusal samples cannot define answer oracles")
        audio_path = Path(audio_value)
        if not audio_path.is_absolute():
            audio_path = manifest_path.parent / audio_path
        samples.append(
            _BenchmarkSample(
                source_id=f"audio-{index:03d}",
                audio=audio_path,
                reference=(
                    reference_value.strip()
                    if isinstance(reference_value, str)
                    else None
                ),
                expected_status=expected_status,
                expected_refusal_reason=expected_refusal_reason,
                expected_answer_contains=expected_answer_contains,
                expected_chunk_ids=expected_chunk_ids,
                expected_document_ids=expected_document_ids,
            )
        )
    return samples


def _redact(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, tuple):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact(item, secrets) for key, item in value.items()}
    return value


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(report, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _quality_record(
    sample: _BenchmarkSample,
    *,
    transcription: TranscriptionResult | None,
    rag: RAGResponse | None,
) -> dict[str, Any]:
    reference_strict_match = (
        transcription.transcript == sample.reference
        if transcription is not None and sample.reference is not None
        else None
    )
    reference_normalized_match = (
        _normalized_transcript(transcription.transcript)
        == _normalized_transcript(sample.reference)
        if transcription is not None and sample.reference is not None
        else None
    )
    actual_status = rag.status.value if rag is not None else None
    actual_refusal_reason = (
        rag.refusal_reason.value
        if rag is not None and rag.refusal_reason is not None
        else None
    )
    status_match = actual_status == sample.expected_status if rag is not None else False
    refusal_reason_match = (
        actual_refusal_reason == sample.expected_refusal_reason
        if sample.expected_refusal_reason is not None
        else True
    )
    grounded_answer = bool(
        rag is not None
        and rag.status.value == "answered"
        and rag.groundedness is not None
        and rag.groundedness.is_grounded
    )
    normalized_answer = _normalized_transcript(rag.answer or "") if rag else ""
    answer_contains_match = (
        all(
            f" {_normalized_transcript(expected)} " in f" {normalized_answer} "
            for expected in sample.expected_answer_contains
        )
        if sample.expected_answer_contains
        else None
    )
    supporting_chunk_ids = set(
        rag.groundedness.supporting_chunk_ids
        if rag is not None and rag.groundedness is not None
        else []
    )
    chunk_ids_match = (
        set(sample.expected_chunk_ids).issubset(supporting_chunk_ids)
        if sample.expected_chunk_ids
        else None
    )
    supporting_document_ids = {
        retrieved.chunk.document_id
        for retrieved in (rag.retrieved_chunks if rag is not None else [])
        if retrieved.chunk.id in supporting_chunk_ids
    }
    document_ids_match = (
        set(sample.expected_document_ids).issubset(supporting_document_ids)
        if sample.expected_document_ids
        else None
    )
    answer_oracle_checks = [
        check
        for check in (answer_contains_match, chunk_ids_match, document_ids_match)
        if check is not None
    ]
    answer_oracle_configured = bool(answer_oracle_checks)
    answer_oracle_match = bool(answer_oracle_configured and all(answer_oracle_checks))
    outcome_quality = bool(
        status_match
        and refusal_reason_match
        and (
            sample.expected_status == "refused"
            or (grounded_answer and answer_oracle_match)
        )
    )
    reference_quality = bool(
        sample.reference is None or reference_normalized_match is True
    )
    return {
        "expected_status": sample.expected_status,
        "expected_refusal_reason": sample.expected_refusal_reason,
        "actual_status": actual_status,
        "actual_refusal_reason": actual_refusal_reason,
        "status_match": status_match,
        "refusal_reason_match": refusal_reason_match,
        "answered_and_grounded": grounded_answer,
        "answer_oracle_configured": answer_oracle_configured,
        "expected_answer_contains": list(sample.expected_answer_contains),
        "answer_contains_match": answer_contains_match,
        "expected_chunk_ids": list(sample.expected_chunk_ids),
        "supporting_chunk_ids_match": chunk_ids_match,
        "expected_document_ids": list(sample.expected_document_ids),
        "supporting_document_ids_match": document_ids_match,
        "answer_oracle_match": answer_oracle_match,
        "reference_required": sample.reference is not None,
        "reference_strict_match": reference_strict_match,
        "reference_normalized_match": reference_normalized_match,
        "eligible": bool(rag is not None and outcome_quality and reference_quality),
    }


def _failed_trial(
    trial_number: int,
    *,
    sample: _BenchmarkSample,
    phase: str,
    started: float,
    exception: BaseException,
    transcription: TranscriptionResult | None = None,
    overlap: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trial: dict[str, Any] = {
        "trial": trial_number,
        "outcome": "failed",
        "failure_phase": phase,
        "error_type": type(exception).__name__,
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "sla_met": False,
        "post_utterance_target_met": False,
        "first_audio_target_met": False,
        "overlap": overlap,
        "source_id": sample.source_id,
        "source_file": sample.audio.name,
        "reference": sample.reference,
        "quality": _quality_record(
            sample,
            transcription=transcription,
            rag=None,
        ),
    }
    if transcription is not None:
        trial.update(
            transcript=transcription.transcript,
            transcription_available=True,
            stt_provider=transcription.provider,
            stt_model=transcription.model,
        )
    else:
        trial["transcription_available"] = False
    return trial


async def _run_trial(
    *,
    trial_number: int,
    args: argparse.Namespace,
    sample: _BenchmarkSample,
    stt: ElevenLabsStreamingSTT,
    pipeline: Any,
) -> dict[str, Any]:
    trial_started = time.perf_counter()
    clock = _AudioClock()
    prepared_text: str | None = None
    prepared_task: asyncio.Task[tuple[RAGResponse, float, float]] | None = None
    superseded_tasks: list[asyncio.Task[tuple[RAGResponse, float, float]]] = []
    overlap_replacements = 0

    def overlap_record(
        *,
        exact_commit_match: bool | None,
        used: bool,
    ) -> dict[str, Any]:
        return {
            "enabled": bool(args.overlap_enabled),
            "settled_callback_received": prepared_text is not None,
            "exact_commit_match": exact_commit_match,
            "used": used,
            "replacement_count": overlap_replacements,
        }

    def request_for(text: str) -> RAGRequest:
        return RAGRequest(
            query=text,
            language_code=args.rag_language_code,
            candidate_k=args.candidate_k,
            final_k=args.final_k,
        )

    async def prepare_rag(text: str) -> tuple[RAGResponse, float, float]:
        started_at = time.perf_counter()
        result: RAGResponse = await pipeline.answer(request_for(text))
        return result, started_at, time.perf_counter()

    def show_partial(text: str) -> None:
        if args.show_partials:
            print(f"trial {trial_number} partial: {text}", flush=True)

    def prepare_settled(text: str) -> None:
        nonlocal prepared_task, prepared_text, overlap_replacements
        if prepared_task is not None:
            prepared_task.cancel()
            superseded_tasks.append(prepared_task)
            overlap_replacements += 1
        prepared_text = text
        prepared_task = asyncio.create_task(prepare_rag(text))

    try:
        transcription = await stt.transcribe(
            wav_chunks(
                sample.audio,
                chunk_ms=args.chunk_ms,
                realtime_pacing=not args.no_realtime_pacing,
                clock=clock,
            ),
            on_partial=show_partial,
            on_settled=prepare_settled if args.overlap_enabled else None,
        )
    except Exception as exc:  # noqa: BLE001 - measured failure, details omitted
        if prepared_task is not None:
            prepared_task.cancel()
        await asyncio.gather(
            *superseded_tasks,
            *([prepared_task] if prepared_task is not None else []),
            return_exceptions=True,
        )
        return _failed_trial(
            trial_number,
            sample=sample,
            phase="stt",
            started=trial_started,
            exception=exc,
            overlap=overlap_record(exact_commit_match=None, used=False),
        )

    stt_returned_at = time.perf_counter()
    overlap_exact_match = bool(
        args.overlap_enabled
        and prepared_task is not None
        and prepared_text == transcription.transcript
    )
    overlap_used = False
    try:
        if overlap_exact_match and prepared_task is not None:
            rag, rag_started_at, rag_returned_at = await prepared_task
            overlap_used = True
        else:
            if prepared_task is not None:
                prepared_task.cancel()
                await asyncio.gather(prepared_task, return_exceptions=True)
            rag, rag_started_at, rag_returned_at = await prepare_rag(
                transcription.transcript
            )
    except Exception as exc:  # noqa: BLE001 - measured failure, details omitted
        await asyncio.gather(*superseded_tasks, return_exceptions=True)
        return _failed_trial(
            trial_number,
            sample=sample,
            phase="rag",
            started=trial_started,
            exception=exc,
            transcription=transcription,
            overlap=overlap_record(
                exact_commit_match=overlap_exact_match,
                used=False,
            ),
        )
    await asyncio.gather(*superseded_tasks, return_exceptions=True)

    if clock.first_audio_at is None or clock.eof_at is None:
        return _failed_trial(
            trial_number,
            sample=sample,
            phase="instrumentation",
            started=trial_started,
            exception=RuntimeError("audio timing anchor missing"),
            transcription=transcription,
            overlap=overlap_record(
                exact_commit_match=overlap_exact_match,
                used=overlap_used,
            ),
        )

    response_build_started = time.perf_counter()
    provisional_at = time.perf_counter()
    provisional_first_audio_ms = (provisional_at - clock.first_audio_at) * 1000
    provisional_eof_ms = (provisional_at - clock.eof_at) * 1000
    first_audio_to_committed_ms = (
        clock.eof_at - clock.first_audio_at
    ) * 1000 + transcription.final_after_audio_end_ms
    committed_to_provisional_answer_ms = max(
        0.0,
        provisional_eof_ms - transcription.final_after_audio_end_ms,
    )
    response = VoiceRAGResponse(
        transcription=transcription,
        rag=rag,
        latencies=VoicePipelineLatencies(
            metric_definition=_METRIC_DEFINITION,
            first_audio_to_committed_ms=first_audio_to_committed_ms,
            audio_eof_to_committed_ms=transcription.final_after_audio_end_ms,
            committed_to_answer_ms=committed_to_provisional_answer_ms,
            audio_eof_to_answer_ms=provisional_eof_ms,
            first_audio_to_answer_ms=provisional_first_audio_ms,
            target_ms=args.target_ms,
            target_met=provisional_eof_ms <= args.target_ms,
        ),
    )
    # Serialization is part of the measured output boundary. The exact outer
    # clocks below, not the provisional embedded values, drive every SLA claim.
    response.model_dump_json()
    answered_at = time.perf_counter()

    first_audio_to_answer_ms = (answered_at - clock.first_audio_at) * 1000
    audio_eof_to_answer_ms = (answered_at - clock.eof_at) * 1000
    committed_to_answer_ms = max(
        0.0,
        audio_eof_to_answer_ms - transcription.final_after_audio_end_ms,
    )
    return {
        "trial": trial_number,
        "outcome": "completed",
        "source_id": sample.source_id,
        "source_file": sample.audio.name,
        "reference": sample.reference,
        "rag_status": rag.status.value,
        "refusal_reason": (
            rag.refusal_reason.value if rag.refusal_reason is not None else None
        ),
        "grounded": bool(rag.groundedness and rag.groundedness.is_grounded),
        "quality": _quality_record(
            sample,
            transcription=transcription,
            rag=rag,
        ),
        "transcript": transcription.transcript,
        "transcription_available": True,
        "stt_provider": transcription.provider,
        "stt_model": transcription.model,
        "partial_count": len(transcription.partial_transcripts),
        "overlap": overlap_record(
            exact_commit_match=overlap_exact_match,
            used=overlap_used,
        ),
        "sla_met": audio_eof_to_answer_ms <= args.target_ms,
        "post_utterance_target_met": audio_eof_to_answer_ms <= args.target_ms,
        "first_audio_target_met": first_audio_to_answer_ms <= args.target_ms,
        "latency_ms": {
            "first_audio_to_answer": first_audio_to_answer_ms,
            "audio_eof_to_answer": audio_eof_to_answer_ms,
            "first_audio_to_committed": first_audio_to_committed_ms,
            "audio_eof_to_committed": transcription.final_after_audio_end_ms,
            "committed_to_answer": committed_to_answer_ms,
            "stt_connection": transcription.connection_ms,
            "stt_first_partial_from_call": transcription.time_to_first_partial_ms,
            "stt_final_from_call": transcription.time_to_final_transcript_ms,
            "stt_call_total": transcription.total_ms,
            "stt_return_to_rag_start": max(
                0.0, (rag_started_at - stt_returned_at) * 1000
            ),
            "rag_started_before_stt_return": max(
                0.0, (stt_returned_at - rag_started_at) * 1000
            ),
            "rag_wall": (rag_returned_at - rag_started_at) * 1000,
            "rag_retrieval": rag.latencies.retrieval_ms,
            "rag_relevance": rag.latencies.relevance_ms,
            "rag_generation": rag.latencies.generation_ms,
            "rag_groundedness": rag.latencies.groundedness_ms,
            "rag_output": rag.latencies.output_ms,
            "rag_reported_total": rag.latencies.total_ms,
            "response_build_and_serialization": (answered_at - response_build_started)
            * 1000,
        },
    }


def _reference_quality(trials: list[dict[str, Any]]) -> dict[str, Any] | None:
    referenced = [trial for trial in trials if isinstance(trial.get("reference"), str)]
    if not referenced:
        return None
    available = [trial for trial in referenced if trial.get("transcription_available")]
    strict_matches = sum(
        trial.get("transcript") == trial.get("reference") for trial in available
    )
    normalized_matches = sum(
        _normalized_transcript(str(trial.get("transcript") or ""))
        == _normalized_transcript(str(trial["reference"]))
        for trial in available
    )
    attempted = len(trials)
    return {
        "reference_trials": len(referenced),
        "transcripts_available": len(available),
        "strict_exact_match": {
            "matches": strict_matches,
            "available_rate": strict_matches / len(available) if available else 0.0,
            "referenced_attempts_rate": strict_matches / len(referenced),
            "all_attempts_rate": strict_matches / attempted if attempted else 0.0,
        },
        "normalized_exact_match": {
            "matches": normalized_matches,
            "available_rate": (
                normalized_matches / len(available) if available else 0.0
            ),
            "referenced_attempts_rate": normalized_matches / len(referenced),
            "all_attempts_rate": (normalized_matches / attempted if attempted else 0.0),
        },
    }


def _build_report(
    *,
    trials: list[dict[str, Any]],
    args: argparse.Namespace,
    audio: list[dict[str, int | float | str]],
    distinct_recordings: int,
    runtime: Any,
    runtime_initialization_ms: float,
    stt_preflight_ms: float,
) -> dict[str, Any]:
    completed = [trial for trial in trials if trial["outcome"] == "completed"]
    failures = [trial for trial in trials if trial["outcome"] == "failed"]
    status_counts = Counter(
        str(trial["rag_status"]) for trial in completed if trial.get("rag_status")
    )
    refusal_counts = Counter(
        str(trial["refusal_reason"])
        for trial in completed
        if trial.get("refusal_reason")
    )
    transcripts = [
        str(trial["transcript"])
        for trial in trials
        if trial.get("transcription_available")
    ]
    transcript_counts = Counter(transcripts)
    modal_transcript, modal_count = (
        transcript_counts.most_common(1)[0] if transcript_counts else (None, 0)
    )
    per_source_stability: dict[str, dict[str, Any]] = {}
    for source_id in sorted({str(trial["source_id"]) for trial in trials}):
        source_trials = [
            trial
            for trial in trials
            if trial["source_id"] == source_id and trial.get("transcription_available")
        ]
        source_counts = Counter(str(trial["transcript"]) for trial in source_trials)
        source_modal, source_modal_count = (
            source_counts.most_common(1)[0] if source_counts else (None, 0)
        )
        per_source_stability[source_id] = {
            "source_file": next(
                str(trial["source_file"])
                for trial in trials
                if trial["source_id"] == source_id
            ),
            "transcripts_available": len(source_trials),
            "unique_transcripts": len(source_counts),
            "all_identical": len(source_counts) == 1 if source_counts else False,
            "modal_transcript": source_modal,
            "modal_count": source_modal_count,
            "modal_rate": (
                source_modal_count / len(source_trials) if source_trials else 0.0
            ),
        }

    def values(field: str) -> list[float | None]:
        return [trial["latency_ms"].get(field) for trial in completed]

    failed_count = len(failures)
    full_voice = {
        "first_audio_to_answer": _distribution(
            values("first_audio_to_answer"),
            failed_trials=failed_count,
            target_ms=args.target_ms,
        ),
        "audio_eof_to_answer": _distribution(
            values("audio_eof_to_answer"),
            failed_trials=failed_count,
            target_ms=args.target_ms,
        ),
    }
    stage_fields = (
        "first_audio_to_committed",
        "audio_eof_to_committed",
        "committed_to_answer",
        "stt_connection",
        "stt_first_partial_from_call",
        "stt_final_from_call",
        "stt_call_total",
        "stt_return_to_rag_start",
        "rag_started_before_stt_return",
        "rag_wall",
        "rag_retrieval",
        "rag_relevance",
        "rag_generation",
        "rag_groundedness",
        "rag_output",
        "rag_reported_total",
        "response_build_and_serialization",
    )
    stage_breakdown = {
        field: _distribution(values(field), failed_trials=failed_count)
        for field in stage_fields
    }
    sla_met = sum(trial.get("sla_met") is True for trial in completed)
    first_audio_sla_met = sum(
        trial.get("first_audio_target_met") is True for trial in completed
    )
    attempted = len(trials)
    quality_eligible = sum(
        trial.get("quality", {}).get("eligible") is True for trial in trials
    )
    referenced_trials = sum(
        trial.get("quality", {}).get("reference_required") is True for trial in trials
    )
    normalized_reference_questions = {
        normalized
        for trial in trials
        if isinstance((reference := trial.get("reference")), str)
        and (normalized := _normalized_transcript(reference))
    }
    unique_normalized_reference_questions = len(normalized_reference_questions)
    manifest_supplied = args.manifest is not None
    claim_eligible_trials = sum(
        manifest_supplied
        and trial.get("quality", {}).get("eligible") is True
        and trial.get("quality", {}).get("reference_required") is True
        for trial in trials
    )
    latency_gate_met = sla_met == attempted and attempted > 0
    quality_gate_met = claim_eligible_trials == attempted and attempted > 0
    benchmark_passed = latency_gate_met and (args.latency_only or quality_gate_met)
    workload_submission_ready = (
        distinct_recordings >= 30
        and unique_normalized_reference_questions >= 30
        and args.trials >= 30
        and not args.no_realtime_pacing
        and manifest_supplied
        and referenced_trials == attempted
    )
    overlap_trials = [
        overlap
        for trial in trials
        if isinstance((overlap := trial.get("overlap")), dict)
    ]
    return {
        "schema_version": 1,
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "benchmark": "integrated_voice_rag",
        "metric_definition": _METRIC_DEFINITION,
        "scope": {
            "runner": "component_direct",
            "end_to_end_percentiles": (
                "Observed per-trial wrapper clocks; stage percentiles are never summed."
            ),
            "p100_policy": "Literal maximum completed observation.",
            "failure_policy": (
                "Failures remain explicit SLA and quality violations and are not "
                "silently removed from all-attempt gates."
            ),
            "provider_setup_before_wav_clock": True,
            "client_to_asgi_transport_included": False,
            "final_websocket_receipt_included": False,
            "browser_rendering_included": False,
            "live_service_p100_validated": False,
            "external_live_service_benchmark_required": True,
        },
        "audio": {
            "source_count": len(audio),
            "distinct_recordings": distinct_recordings,
            "distinct_recordings_basis": (
                "Full SHA-256 of decoded PCM sample bytes among attempted trials; "
                "header-only differences cannot create diversity and digests are "
                "not persisted."
            ),
            "sources": audio,
            "assignment": "deterministic_round_robin",
            "submission_quality": workload_submission_ready,
            "submission_guidance": (
                "Use a manifest with transcript references and answer/outcome "
                "oracles for at least 30 content-distinct, representative WAVs "
                "and 30 unique normalized reference questions; copied files or "
                "repeated questions measure jitter rather than workload breadth."
            ),
        },
        "configuration": {
            "trials": args.trials,
            "chunk_ms": args.chunk_ms,
            "realtime_pacing": not args.no_realtime_pacing,
            "stt_language_code": args.stt_language_code,
            "rag_language_code": args.rag_language_code,
            "candidate_k": args.candidate_k,
            "final_k": args.final_k,
            "target_ms": args.target_ms,
            "target_anchor": "audio_eof_to_answer_ms",
            "benchmark_mode": (
                "latency_only_non_submission_smoke"
                if args.latency_only
                else "quality_gated_component"
            ),
            "input_mode": "manifest" if manifest_supplied else "positional_smoke",
            "answer_mode": "local_extractive",
            "overlap_enabled": bool(args.overlap_enabled),
            "overlap_mode": (
                "settled_exact_commit_match"
                if args.overlap_enabled
                else "sequential_committed_transcript"
            ),
            "distinct_recordings": distinct_recordings,
            "unique_normalized_reference_questions": (
                unique_normalized_reference_questions
            ),
        },
        "runtime": {
            "reused_runtime_instances": 1,
            "reused_stt_client_instances": 1,
            "vector_count": int(runtime.vector_count),
            "supported_languages": list(getattr(runtime, "supported_languages", ())),
            "embedding_model": str(runtime.embedding_model),
            "device": str(runtime.device),
            "runtime_initialization_ms": runtime_initialization_ms,
            "runtime_reported_load_ms": float(runtime.load_ms),
            "runtime_reported_warmup_ms": float(runtime.warmup_ms),
            "stt_free_tier_preflight_ms": stt_preflight_ms,
        },
        "outcomes": {
            "attempted": attempted,
            "completed": len(completed),
            "failed": failed_count,
            "answered": status_counts.get("answered", 0),
            "refused": status_counts.get("refused", 0),
            "rag_statuses": dict(sorted(status_counts.items())),
            "refusal_reasons": dict(sorted(refusal_counts.items())),
            "sla_met": sla_met,
            # Failures and completed target misses are both violations.
            "sla_violations": attempted - sla_met,
            "sla_attainment_rate": sla_met / attempted if attempted else 0.0,
            "all_attempts_meet_target": sla_met == attempted and attempted > 0,
            "all_attempts_meet_post_utterance_target": (
                sla_met == attempted and attempted > 0
            ),
            "first_audio_target_met": first_audio_sla_met,
            "all_attempts_meet_first_audio_target": (
                first_audio_sla_met == attempted and attempted > 0
            ),
            "overlap": {
                "enabled_trials": sum(
                    overlap.get("enabled") is True for overlap in overlap_trials
                ),
                "settled_callbacks": sum(
                    overlap.get("settled_callback_received") is True
                    for overlap in overlap_trials
                ),
                "exact_commit_matches": sum(
                    overlap.get("exact_commit_match") is True
                    for overlap in overlap_trials
                ),
                "used": sum(overlap.get("used") is True for overlap in overlap_trials),
                "fallbacks": sum(
                    bool(trial.get("overlap", {}).get("enabled"))
                    and not bool(trial.get("overlap", {}).get("used"))
                    for trial in completed
                ),
                "unverified_callbacks": sum(
                    overlap.get("settled_callback_received") is True
                    and overlap.get("exact_commit_match") is None
                    for overlap in overlap_trials
                ),
                "replacement_count": sum(
                    int(overlap.get("replacement_count", 0))
                    for overlap in overlap_trials
                ),
            },
        },
        "quality": {
            "gate_enabled": not args.latency_only,
            "outcome_oracle_eligible_trials": quality_eligible,
            "eligible_trials": claim_eligible_trials,
            "quality_violations": attempted - claim_eligible_trials,
            "manifest_supplied": manifest_supplied,
            "referenced_trials": referenced_trials,
            "unique_normalized_reference_questions": (
                unique_normalized_reference_questions
            ),
            "all_trials_have_transcript_references": referenced_trials == attempted,
            "all_attempts_eligible": quality_gate_met,
            "rule": (
                "The default gate requires a manifest and transcript reference "
                "for every trial. Every completed trial must match expected_status; answered "
                "trials must be grounded and match every configured answer/source "
                "oracle, explicitly expected refusals may pass, and supplied "
                "transcript references must match after normalization."
            ),
            "expected_answer_contains_semantics": (
                "Each value is Unicode-normalized and case-folded, then matched "
                "as a contiguous whole-token phrase; every configured value must match."
            ),
        },
        "gates": {
            "latency_sla_met": latency_gate_met,
            "quality_gate_met": quality_gate_met,
            "quality_gate_enabled": not args.latency_only,
            "benchmark_passed": benchmark_passed,
            "workload_submission_ready": workload_submission_ready,
            "minimum_content_distinct_recordings_met": distinct_recordings >= 30,
            "minimum_unique_normalized_reference_questions_met": (
                unique_normalized_reference_questions >= 30
            ),
            "component_evidence_eligible": bool(
                benchmark_passed and workload_submission_ready and not args.latency_only
            ),
            "live_submission_p100_validated": False,
        },
        "latency_ms": {
            "full_voice": full_voice,
            "stage_breakdown": stage_breakdown,
        },
        "transcript_stability": {
            "scope_note": (
                "Overall uniqueness reflects different questions; per-source "
                "stability measures repeated-transcription consistency."
            ),
            "transcripts_available": len(transcripts),
            "unique_transcripts": len(transcript_counts),
            "all_identical": len(transcript_counts) == 1 if transcripts else False,
            "modal_transcript": modal_transcript,
            "modal_count": modal_count,
            "modal_rate": modal_count / len(transcripts) if transcripts else 0.0,
            "per_source": per_source_stability,
        },
        "reference_quality": _reference_quality(trials),
        "trials": trials,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.chunk_ms <= 0 or args.chunk_ms > 1_000:
        raise ValueError("--chunk-ms must be between 1 and 1000")
    if args.candidate_k <= 0 or args.final_k <= 0:
        raise ValueError("candidate counts must be positive")
    if args.final_k > args.candidate_k:
        raise ValueError("--final-k cannot exceed --candidate-k")
    if args.target_ms <= 0:
        raise ValueError("--target-ms must be positive")


async def run(args: argparse.Namespace) -> int:
    _validate_args(args)
    samples = _load_samples(args)
    audio_profiles: list[dict[str, int | float | str]] = []
    for sample in samples:
        profile = wav_metadata(sample.audio)
        profile["source_id"] = sample.source_id
        audio_profiles.append(profile)
    attempted_samples = [
        samples[trial_index % len(samples)] for trial_index in range(args.trials)
    ]
    digests_by_path: dict[Path, bytes] = {}
    for sample in attempted_samples:
        resolved_path = sample.audio.resolve()
        if resolved_path not in digests_by_path:
            digests_by_path[resolved_path] = _wav_content_digest(sample.audio)
    distinct_recordings = len(set(digests_by_path.values()))
    load_dotenv()

    runtime_started = time.perf_counter()
    runtime = await load_runtime(
        RuntimeSettings(
            index_dir=args.index_dir,
            device=args.device,
            preload=True,
            latency_target_ms=args.target_ms,
        )
    )
    runtime_initialization_ms = (time.perf_counter() - runtime_started) * 1000
    supported_languages = tuple(getattr(runtime, "supported_languages", ()))
    if args.rag_language_code not in supported_languages:
        raise ValueError(
            "The loaded index does not cover --rag-language-code="
            f"{args.rag_language_code!r}; supported={supported_languages!r}"
        )

    session_count = args.trials
    max_attempts = 3
    longest_audio_ms = max(float(profile["duration_ms"]) for profile in audio_profiles)
    settings = ElevenLabsSTTSettings.from_env(
        language_code=args.stt_language_code,
        max_audio_seconds=min(120.0, max(1.0, longest_audio_ms / 1000 + 1)),
        max_concurrent_sessions=1,
        daily_session_cap=session_count,
        daily_token_cap=session_count * max_attempts,
        max_attempts=max_attempts,
    )
    stt = ElevenLabsStreamingSTT(settings)
    try:
        preflight_started = time.perf_counter()
        await stt.token_broker.ensure_free_tier(force=True)
        stt_preflight_ms = (time.perf_counter() - preflight_started) * 1000

        trials: list[dict[str, Any]] = []
        for trial_number in range(1, args.trials + 1):
            sample = samples[(trial_number - 1) % len(samples)]
            trial = await _run_trial(
                trial_number=trial_number,
                args=args,
                sample=sample,
                stt=stt,
                pipeline=runtime.pipeline,
            )
            trials.append(trial)
            if trial["outcome"] == "completed":
                latency = trial["latency_ms"]
                print(
                    f"trial {trial_number}/{args.trials}: "
                    f"status={trial['rag_status']} "
                    f"EOF→answer={latency['audio_eof_to_answer']:.3f} ms "
                    f"first-audio→answer={latency['first_audio_to_answer']:.3f} ms",
                    flush=True,
                )
            else:
                print(
                    f"trial {trial_number}/{args.trials}: FAILED "
                    f"phase={trial['failure_phase']} type={trial['error_type']}",
                    file=sys.stderr,
                    flush=True,
                )

        report = _build_report(
            trials=trials,
            args=args,
            audio=audio_profiles,
            distinct_recordings=distinct_recordings,
            runtime=runtime,
            runtime_initialization_ms=runtime_initialization_ms,
            stt_preflight_ms=stt_preflight_ms,
        )
        report = _redact(report, (settings.api_key,))
        _write_report(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["gates"]["benchmark_passed"] else 1
    finally:
        await stt.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
