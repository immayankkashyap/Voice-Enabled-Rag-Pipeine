#!/usr/bin/env python3
"""Black-box benchmark for a running ``/ws/voice-rag`` endpoint.

This runner exercises the public WebSocket protocol instead of importing the
application runtime.  It includes client-to-ASGI transport and receipt,
decoding, schema validation, and oracle validation of the final answer.  It
deliberately excludes microphone capture and browser rendering.

The bearer value is read only from ``VOICE_DEMO_TOKEN``.  There is no CLI flag
for it, and it is redacted before an atomic report write.  A submission-quality
run requires a quality-gated manifest, realtime pacing, at least 30 measured
trials, 30 full-SHA256-content-distinct WAV files, and 30 distinct normalized
reference questions.  Failures remain SLA violations; P100 is the observed
maximum of completed trials, never a selected successful run.

Example::

    VOICE_DEMO_TOKEN='use-a-restricted-demo-token' \
      python scripts/benchmark_live_websocket.py \
      --ws-url ws://127.0.0.1:8000/ws/voice-rag \
      --origin http://127.0.0.1:8000 \
      --manifest tests/audio/submission-manifest.json \
      --output data/live_websocket_benchmark.json

Manifest entries use the same quality contract as the component benchmark::

    {"audio": "question-01.wav", "reference": "...",
     "expected_status": "answered", "expected_answer_contains": ["1912"]}

Answered cases require at least one answer/source oracle.  Refusal cases may
instead specify ``expected_refusal_reason``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import wave
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from dotenv import load_dotenv
from pydantic import ValidationError
from websockets.asyncio.client import connect

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas import TranscriptionResult, VoiceRAGResponse
from scripts.benchmark_support import (
    _BenchmarkSample,
    _distribution,
    _load_samples,
    _normalized_transcript,
    _quality_record,
    _redact,
    _wav_content_digest,
    _write_report,
    wav_metadata,
)

_PCM_SAMPLE_RATE = 16_000
_PCM_SAMPLE_WIDTH_BYTES = 2
_MAX_FRAME_BYTES = 16_000
_METRIC_DEFINITION = (
    "Client-observed first_audio_to_client_answer_ms starts immediately before "
    "the first binary WebSocket send. end_sent_to_client_answer_ms starts after "
    "the end control send completes. Both stop after the answer frame is received, "
    "JSON-decoded, schema-validated, checked against the committed transcript, and "
    "quality-oracle validated. Connection/start/ready setup, microphone capture, "
    "and browser rendering are excluded. Client-to-ASGI transport, the server and "
    "provider path, response serialization, final WebSocket receipt, and client "
    "parsing/validation are included."
)


class _LiveBenchmarkProtocolError(RuntimeError):
    """Sanitized protocol failure; exception details never enter reports."""


@dataclass(slots=True)
class _ClientClock:
    first_audio_at: float | None = None
    eof_at: float | None = None
    end_sent_at: float | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a running /ws/voice-rag endpoint using client-observed "
            "P50/P70/observed-max P100 clocks and quality-gated WAV fixtures. "
            "VOICE_DEMO_TOKEN is required in the environment and is never "
            "accepted on the command line or written to the report."
        ),
        epilog=(
            "Start the server separately. Use a manifest with >=30 distinct WAV "
            "contents and >=30 unique reference questions for evidence eligibility."
        ),
    )
    parser.add_argument(
        "audio",
        type=Path,
        nargs="*",
        help=(
            "Mono PCM16 16 kHz WAV files used round-robin. Positional mode is "
            "smoke-only; submission eligibility requires --manifest."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "JSON list (or {'samples': [...]}) with audio, reference, expected "
            "status/refusal, and answer/source oracle fields. Relative audio "
            "paths resolve from the manifest directory."
        ),
    )
    parser.add_argument(
        "--ws-url",
        required=True,
        help="Exact ws:// or wss:// URL ending in the voice endpoint path.",
    )
    parser.add_argument(
        "--origin",
        required=True,
        help="Exact allowlisted HTTP(S) Origin header, without a path or slash.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Measured WebSocket sessions; defaults to 30.",
    )
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument(
        "--no-realtime-pacing",
        action="store_true",
        help="Send without pacing; explicitly ineligible as live-voice evidence.",
    )
    parser.add_argument(
        "--language-code",
        default="hi",
        help="Two-letter language code sent in the start event; default: hi.",
    )
    parser.add_argument("--target-ms", type=float, default=200.0)
    parser.add_argument("--ready-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--answer-timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--latency-only",
        action="store_true",
        help=(
            "Skip quality gating for a smoke run. Such a run is always marked "
            "ineligible for submission evidence."
        ),
    )
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--reference")
    reference.add_argument("--reference-file", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/live_websocket_benchmark.json"),
    )
    return parser


def _validate_ws_url(value: str) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"ws", "wss"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or parsed.query
    ):
        raise ValueError(
            "--ws-url must be a ws:// or wss:// URL without credentials, query, "
            "or fragment"
        )


def _validate_origin(value: str) -> None:
    parsed = urlsplit(value)
    expected = f"{parsed.scheme}://{parsed.netloc}"
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
        or value != expected
    ):
        raise ValueError(
            "--origin must be an exact http(s) origin such as "
            "http://127.0.0.1:8000, without a path or trailing slash"
        )


def _validate_args(args: argparse.Namespace) -> None:
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.chunk_ms <= 0 or args.chunk_ms > 500:
        raise ValueError(
            "--chunk-ms must be between 1 and 500 so a PCM16/16 kHz frame "
            "does not exceed the server's 16000-byte limit"
        )
    if args.target_ms <= 0:
        raise ValueError("--target-ms must be positive")
    if args.ready_timeout_seconds <= 0 or args.answer_timeout_seconds <= 0:
        raise ValueError("WebSocket timeouts must be positive")
    if not isinstance(args.language_code, str) or not args.language_code.strip():
        raise ValueError("--language-code must be non-empty")
    _validate_ws_url(args.ws_url)
    _validate_origin(args.origin)


async def _send_wav(
    websocket: Any,
    path: Path,
    *,
    chunk_ms: int,
    realtime_pacing: bool,
    clock: _ClientClock,
    _monotonic: Callable[[], float] = time.perf_counter,
    _sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Send validated PCM on cumulative pre-send deadlines.

    Microphone capture is out of scope, so the first already-captured frame is
    sent immediately.  Every later frame waits until the cumulative duration of
    all preceding frames.  This avoids per-frame sleep drift and anchors the
    first-audio clock immediately before the first WebSocket send.
    """

    wav_metadata(path)
    frames_per_chunk = max(1, _PCM_SAMPLE_RATE * chunk_ms // 1000)
    prior_audio_seconds = 0.0
    sent_any = False
    with wave.open(str(path), "rb") as audio:
        while chunk := audio.readframes(frames_per_chunk):
            if len(chunk) > _MAX_FRAME_BYTES:
                raise ValueError("A generated PCM frame exceeds 16000 bytes")
            if not sent_any:
                clock.first_audio_at = _monotonic()
                sent_any = True
            elif realtime_pacing:
                assert clock.first_audio_at is not None
                remaining = clock.first_audio_at + prior_audio_seconds - _monotonic()
                if remaining > 0:
                    await _sleep(remaining)
            await websocket.send(chunk)
            prior_audio_seconds += len(chunk) / (
                _PCM_SAMPLE_WIDTH_BYTES * _PCM_SAMPLE_RATE
            )
    if not sent_any:
        raise ValueError("Voice benchmark WAV contains no audio frames")
    clock.eof_at = _monotonic()
    await websocket.send(json.dumps({"event": "end"}, separators=(",", ":")))
    clock.end_sent_at = _monotonic()


async def _receive_object(websocket: Any) -> dict[str, Any]:
    raw = await websocket.recv()
    if not isinstance(raw, str):
        raise _LiveBenchmarkProtocolError("The service returned a binary control frame")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        raise _LiveBenchmarkProtocolError("The service returned invalid JSON") from None
    if not isinstance(payload, dict):
        raise _LiveBenchmarkProtocolError("The service returned a non-object frame")
    return payload


def _validate_ready(payload: dict[str, Any]) -> None:
    if (
        payload.get("event") != "ready"
        or payload.get("audio_format") != "pcm_s16le"
        or payload.get("sample_rate_hz") != _PCM_SAMPLE_RATE
        or payload.get("commit_strategy") != "manual"
    ):
        raise _LiveBenchmarkProtocolError("The service returned an invalid ready event")


async def _collect_answer(
    websocket: Any,
    *,
    sample: _BenchmarkSample,
    clock: _ClientClock,
    _monotonic: Callable[[], float] = time.perf_counter,
) -> tuple[TranscriptionResult, VoiceRAGResponse, dict[str, Any], dict[str, Any]]:
    partials: list[str] = []
    committed: TranscriptionResult | None = None
    committed_received_at: float | None = None
    first_partial_received_at: float | None = None

    while True:
        payload = await _receive_object(websocket)
        event = payload.get("event")
        if event == "partial_transcript":
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                raise _LiveBenchmarkProtocolError("Invalid partial transcript event")
            partials.append(text)
            if first_partial_received_at is None:
                first_partial_received_at = _monotonic()
            continue
        if event == "committed_transcript":
            if committed is not None:
                raise _LiveBenchmarkProtocolError("Duplicate committed transcript")
            try:
                committed = TranscriptionResult.model_validate(payload.get("payload"))
            except ValidationError:
                raise _LiveBenchmarkProtocolError(
                    "Invalid committed transcript payload"
                ) from None
            committed_received_at = _monotonic()
            continue
        if event == "answer":
            if committed is None:
                raise _LiveBenchmarkProtocolError(
                    "Answer arrived before a committed transcript"
                )
            try:
                response = VoiceRAGResponse.model_validate(payload.get("payload"))
            except ValidationError:
                raise _LiveBenchmarkProtocolError("Invalid answer payload") from None
            if response.transcription != committed:
                raise _LiveBenchmarkProtocolError(
                    "Answer transcription differs from the committed transcript"
                )
            if response.rag.query != committed.transcript:
                raise _LiveBenchmarkProtocolError(
                    "Answer query differs from the committed transcript"
                )
            quality = _quality_record(
                sample,
                transcription=response.transcription,
                rag=response.rag,
            )
            validated_at = _monotonic()
            if clock.first_audio_at is None or clock.end_sent_at is None:
                raise _LiveBenchmarkProtocolError("Missing client timing anchors")
            event_timings = {
                "partial_count": len(partials),
                "first_audio_to_first_partial": (
                    (first_partial_received_at - clock.first_audio_at) * 1000
                    if first_partial_received_at is not None
                    else None
                ),
                "first_audio_to_committed_receipt": (
                    (committed_received_at - clock.first_audio_at) * 1000
                    if committed_received_at is not None
                    else None
                ),
                "end_sent_to_committed_receipt": (
                    (committed_received_at - clock.end_sent_at) * 1000
                    if committed_received_at is not None
                    else None
                ),
                "first_audio_to_client_answer": (validated_at - clock.first_audio_at)
                * 1000,
                "end_sent_to_client_answer": (validated_at - clock.end_sent_at) * 1000,
                "client_stream_send_span": (clock.end_sent_at - clock.first_audio_at)
                * 1000,
            }
            return committed, response, quality, event_timings
        # FastAPI emits a sanitized ErrorResponse without an event key.
        if "error_code" in payload:
            raise _LiveBenchmarkProtocolError("The service refused the voice turn")
        raise _LiveBenchmarkProtocolError("The service returned an unexpected event")


def _failed_trial(
    trial_number: int,
    *,
    sample: _BenchmarkSample,
    phase: str,
    started_at: float,
    exception: BaseException,
) -> dict[str, Any]:
    return {
        "trial": trial_number,
        "outcome": "failed",
        "failure_phase": phase,
        "error_type": type(exception).__name__,
        "elapsed_ms": (time.perf_counter() - started_at) * 1000,
        "source_id": sample.source_id,
        "source_file": sample.audio.name,
        "reference": sample.reference,
        "sla_met": False,
        "post_utterance_target_met": False,
        "first_audio_target_met": False,
        "quality": _quality_record(sample, transcription=None, rag=None),
    }


async def _run_trial(
    *,
    trial_number: int,
    args: argparse.Namespace,
    sample: _BenchmarkSample,
    demo_token: str,
    connector: Callable[..., Any],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    phase = "connect_and_ready"
    clock = _ClientClock()
    try:
        async with connector(
            args.ws_url,
            origin=args.origin,
            open_timeout=args.ready_timeout_seconds,
            close_timeout=2.0,
            max_size=8 * 1024 * 1024,
        ) as websocket:
            # Token exists only in this outbound frame and never in a trial/report.
            await websocket.send(
                json.dumps(
                    {
                        "event": "start",
                        "language_code": args.language_code,
                        "demo_token": demo_token,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            async with asyncio.timeout(args.ready_timeout_seconds):
                _validate_ready(await _receive_object(websocket))

            phase = "audio_send"
            await _send_wav(
                websocket,
                sample.audio,
                chunk_ms=args.chunk_ms,
                realtime_pacing=not args.no_realtime_pacing,
                clock=clock,
            )
            phase = "answer_receive_and_validation"
            async with asyncio.timeout(args.answer_timeout_seconds):
                committed, response, quality, latency = await _collect_answer(
                    websocket,
                    sample=sample,
                    clock=clock,
                )
    except Exception as exc:  # noqa: BLE001 - record type, never unsafe details
        return _failed_trial(
            trial_number,
            sample=sample,
            phase=phase,
            started_at=started_at,
            exception=exc,
        )

    post_utterance_ms = float(latency["end_sent_to_client_answer"])
    first_audio_ms = float(latency["first_audio_to_client_answer"])
    return {
        "trial": trial_number,
        "outcome": "completed",
        "source_id": sample.source_id,
        "source_file": sample.audio.name,
        "reference": sample.reference,
        "transcript": committed.transcript,
        "transcription_available": True,
        "stt_provider": committed.provider,
        "stt_model": committed.model,
        "rag_status": response.rag.status.value,
        "refusal_reason": (
            response.rag.refusal_reason.value
            if response.rag.refusal_reason is not None
            else None
        ),
        "grounded": bool(
            response.rag.groundedness and response.rag.groundedness.is_grounded
        ),
        "quality": quality,
        "partial_count": latency.pop("partial_count"),
        "sla_met": post_utterance_ms <= args.target_ms,
        "post_utterance_target_met": post_utterance_ms <= args.target_ms,
        "first_audio_target_met": first_audio_ms <= args.target_ms,
        "latency_ms": {
            **latency,
            "server_reported_first_audio_to_answer": (
                response.latencies.first_audio_to_answer_ms
            ),
            "server_reported_audio_eof_to_answer": (
                response.latencies.audio_eof_to_answer_ms
            ),
        },
    }


def _reference_quality(trials: list[dict[str, Any]]) -> dict[str, Any] | None:
    referenced = [trial for trial in trials if isinstance(trial.get("reference"), str)]
    if not referenced:
        return None
    available = [trial for trial in referenced if trial.get("transcription_available")]
    normalized_matches = sum(
        _normalized_transcript(str(trial.get("transcript") or ""))
        == _normalized_transcript(str(trial["reference"]))
        for trial in available
    )
    return {
        "reference_trials": len(referenced),
        "transcripts_available": len(available),
        "normalized_exact_matches": normalized_matches,
        "all_referenced_attempts_match": (
            normalized_matches == len(referenced) and bool(referenced)
        ),
    }


def _build_report(
    *,
    args: argparse.Namespace,
    trials: list[dict[str, Any]],
    audio_profiles: list[dict[str, int | float | str]],
    distinct_recordings: int,
) -> dict[str, Any]:
    completed = [trial for trial in trials if trial["outcome"] == "completed"]
    failed_count = len(trials) - len(completed)

    def values(field: str) -> list[float | None]:
        return [trial["latency_ms"].get(field) for trial in completed]

    first_audio_distribution = _distribution(
        values("first_audio_to_client_answer"),
        failed_trials=failed_count,
        target_ms=args.target_ms,
    )
    post_utterance_distribution = _distribution(
        values("end_sent_to_client_answer"),
        failed_trials=failed_count,
        target_ms=args.target_ms,
    )
    attempted = len(trials)
    sla_met_count = sum(trial.get("sla_met") is True for trial in trials)
    quality_eligible_count = sum(
        trial.get("quality", {}).get("eligible") is True for trial in trials
    )
    referenced_count = sum(
        trial.get("quality", {}).get("reference_required") is True for trial in trials
    )
    unique_references = len(
        {
            normalized
            for trial in trials
            if isinstance((reference := trial.get("reference")), str)
            and (normalized := _normalized_transcript(reference))
        }
    )
    manifest_supplied = args.manifest is not None
    claim_eligible_count = sum(
        manifest_supplied
        and trial.get("quality", {}).get("eligible") is True
        and trial.get("quality", {}).get("reference_required") is True
        for trial in trials
    )
    latency_gate_met = sla_met_count == attempted and attempted > 0
    quality_gate_met = claim_eligible_count == attempted and attempted > 0
    benchmark_passed = latency_gate_met and (args.latency_only or quality_gate_met)
    secure_transport = bool(
        urlsplit(args.ws_url).scheme == "wss"
        and urlsplit(args.origin).scheme == "https"
    )
    workload_submission_ready = bool(
        manifest_supplied
        and attempted >= 30
        and distinct_recordings >= 30
        and unique_references >= 30
        and referenced_count == attempted
        and not args.no_realtime_pacing
        and secure_transport
    )
    evidence_eligible = bool(
        benchmark_passed and workload_submission_ready and not args.latency_only
    )
    status_counts = Counter(
        str(trial["rag_status"])
        for trial in completed
        if trial.get("rag_status") is not None
    )
    stage_fields = (
        "first_audio_to_first_partial",
        "first_audio_to_committed_receipt",
        "end_sent_to_committed_receipt",
        "client_stream_send_span",
        "server_reported_first_audio_to_answer",
        "server_reported_audio_eof_to_answer",
    )
    return {
        "schema_version": 1,
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "benchmark": "live_websocket_voice_rag",
        "metric_definition": _METRIC_DEFINITION,
        "scope": {
            "runner": "external_websocket_client",
            "connection_start_and_ready_included": False,
            "client_to_asgi_transport_included": True,
            "final_websocket_receipt_included": True,
            "answer_json_and_schema_validation_included": True,
            "answer_oracle_validation_included": True,
            "microphone_capture_included": False,
            "browser_rendering_included": False,
            "p100_policy": "Literal maximum completed observation.",
            "failure_policy": (
                "Failures remain explicit SLA and quality violations; they are "
                "never dropped from all-attempt gates."
            ),
        },
        "endpoint": {
            "websocket_url": args.ws_url,
            "origin": args.origin,
            "authentication": "VOICE_DEMO_TOKEN environment only; value omitted",
        },
        "configuration": {
            "trials": args.trials,
            "chunk_ms": args.chunk_ms,
            "max_frame_bytes": _MAX_FRAME_BYTES,
            "realtime_pacing": not args.no_realtime_pacing,
            "language_code": args.language_code,
            "target_ms": args.target_ms,
            "target_anchor": "end_sent_to_client_answer_ms",
            "input_mode": "manifest" if manifest_supplied else "positional_smoke",
            "benchmark_mode": (
                "latency_only_non_submission_smoke"
                if args.latency_only
                else "quality_gated_live_websocket"
            ),
        },
        "audio": {
            "source_count": len(audio_profiles),
            "distinct_recordings": distinct_recordings,
            "distinct_recordings_basis": (
                "Full SHA-256 of decoded PCM sample bytes among attempted trials; "
                "header-only differences cannot create diversity and digests are "
                "used in memory and never persisted."
            ),
            "sources": audio_profiles,
            "assignment": "deterministic_round_robin",
        },
        "outcomes": {
            "attempted": attempted,
            "completed": len(completed),
            "failed": failed_count,
            "answered": status_counts.get("answered", 0),
            "refused": status_counts.get("refused", 0),
            "sla_met": sla_met_count,
            "sla_violations": attempted - sla_met_count,
            "all_attempts_meet_post_utterance_target": latency_gate_met,
            "first_audio_target_met": sum(
                trial.get("first_audio_target_met") is True for trial in trials
            ),
        },
        "quality": {
            "gate_enabled": not args.latency_only,
            "eligible_trials": claim_eligible_count,
            "outcome_oracle_eligible_trials": quality_eligible_count,
            "quality_violations": attempted - claim_eligible_count,
            "manifest_supplied": manifest_supplied,
            "referenced_trials": referenced_count,
            "unique_normalized_reference_questions": unique_references,
            "all_attempts_eligible": quality_gate_met,
            "rule": (
                "Every trial must match its normalized transcript reference and "
                "expected outcome. Answers must be grounded and match every "
                "configured answer/source oracle; explicitly expected refusals "
                "may pass."
            ),
        },
        "gates": {
            "latency_sla_met": latency_gate_met,
            "quality_gate_met": quality_gate_met,
            "quality_gate_enabled": not args.latency_only,
            "benchmark_passed": benchmark_passed,
            "workload_submission_ready": workload_submission_ready,
            "minimum_trial_count_met": attempted >= 30,
            "minimum_content_distinct_recordings_met": distinct_recordings >= 30,
            "minimum_unique_normalized_reference_questions_met": (
                unique_references >= 30
            ),
            "secure_transport_met": secure_transport,
            "live_submission_evidence_eligible": evidence_eligible,
            "live_submission_p100_validated": evidence_eligible,
        },
        "latency_ms": {
            "full_voice": {
                "first_audio_to_client_answer": first_audio_distribution,
                "end_sent_to_client_answer": post_utterance_distribution,
            },
            "stage_observations": {
                field: _distribution(values(field), failed_trials=failed_count)
                for field in stage_fields
            },
        },
        "reference_quality": _reference_quality(trials),
        "trials": trials,
    }


async def run(
    args: argparse.Namespace,
    *,
    connector: Callable[..., Any] | None = None,
) -> int:
    _validate_args(args)
    samples = _load_samples(args)
    audio_profiles: list[dict[str, int | float | str]] = []
    for sample in samples:
        profile = wav_metadata(sample.audio)
        profile["source_id"] = sample.source_id
        audio_profiles.append(profile)
    attempted_samples = [samples[index % len(samples)] for index in range(args.trials)]
    digests_by_path: dict[Path, bytes] = {}
    for sample in attempted_samples:
        resolved = sample.audio.resolve()
        if resolved not in digests_by_path:
            digests_by_path[resolved] = _wav_content_digest(sample.audio)
    distinct_recordings = len(set(digests_by_path.values()))

    load_dotenv()
    demo_token = os.environ.get("VOICE_DEMO_TOKEN", "")
    if not 16 <= len(demo_token) <= 512:
        raise ValueError(
            "VOICE_DEMO_TOKEN must be configured with 16 to 512 characters"
        )

    active_connector = connector or connect
    trials: list[dict[str, Any]] = []
    for trial_number, sample in enumerate(attempted_samples, start=1):
        trial = await _run_trial(
            trial_number=trial_number,
            args=args,
            sample=sample,
            demo_token=demo_token,
            connector=active_connector,
        )
        trials.append(trial)
        if trial["outcome"] == "completed":
            latency = trial["latency_ms"]
            print(
                f"trial {trial_number}/{args.trials}: "
                f"status={trial['rag_status']} "
                f"end-sent→client-answer="
                f"{latency['end_sent_to_client_answer']:.3f} ms "
                f"first-audio→client-answer="
                f"{latency['first_audio_to_client_answer']:.3f} ms",
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
        args=args,
        trials=trials,
        audio_profiles=audio_profiles,
        distinct_recordings=distinct_recordings,
    )
    secret_free_report = _redact(report, (demo_token,))
    _write_report(args.output, secret_free_report)
    print(json.dumps(secret_free_report, ensure_ascii=False, indent=2))
    return 0 if report["gates"]["benchmark_passed"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
