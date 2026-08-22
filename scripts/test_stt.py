#!/usr/bin/env python3
"""Repeatable real-audio latency benchmark for Sarvam Realtime STT.

The benchmark intentionally opens a fresh provider session for every measured
trial. This exposes connection jitter and reports failed trials instead of
silently excluding them from the latency distribution.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
import time
import unicodedata
import wave
from collections import Counter
from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas import TranscriptionResult
from app.stt import SarvamStreamingSTT, SarvamSTTSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run N real Sarvam STT trials and report P50/P70/P100 latency, "
            "transcript stability, and optional reference exact-match."
        )
    )
    parser.add_argument("audio", type=Path, help="Mono 16-bit PCM WAV at 8 or 16 kHz")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--language-code", default="en-IN")
    parser.add_argument(
        "--mode",
        choices=("transcribe", "translate", "verbatim", "translit", "codemix"),
        default="transcribe",
    )
    parser.add_argument(
        "--endpointing",
        choices=("manual", "vad"),
        default="manual",
        help="Use an explicit speech_end by default; VAD includes its silence wait.",
    )
    parser.add_argument("--silence-duration-ms", type=int, default=100)
    parser.add_argument("--chunk-ms", type=int, default=250)
    parser.add_argument(
        "--no-realtime-pacing",
        action="store_true",
        help=(
            "Upload audio as fast as possible. Results are then throughput tests, "
            "not live microphone end-to-end latency."
        ),
    )
    parser.add_argument(
        "--show-partials",
        action="store_true",
        help="Print partial transcripts during every trial.",
    )
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument(
        "--reference",
        help="Expected transcript used for strict and normalized exact-match.",
    )
    reference.add_argument(
        "--reference-file",
        type=Path,
        help="UTF-8 text file containing the expected transcript.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optionally write the complete, secret-free benchmark report as JSON.",
    )
    return parser


def wav_metadata(path: Path) -> dict[str, int | float | str]:
    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        width = audio.getsampwidth()
        sample_rate = audio.getframerate()
        frame_count = audio.getnframes()
    if channels != 1 or width != 2 or sample_rate not in (8_000, 16_000):
        raise ValueError(
            "Sarvam Realtime requires mono 16-bit PCM WAV data at 8000 or 16000 Hz; "
            f"received channels={channels}, width={width * 8}-bit, rate={sample_rate}"
        )
    return {
        # A basename makes the saved report portable and avoids leaking a user's
        # absolute home-directory path.
        "file": path.name,
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "sample_width_bits": width * 8,
        "frame_count": frame_count,
        "duration_ms": frame_count / sample_rate * 1000,
    }


async def wav_chunks(
    path: Path,
    *,
    chunk_ms: int,
    realtime_pacing: bool,
) -> AsyncIterator[bytes]:
    metadata = wav_metadata(path)
    sample_rate = int(metadata["sample_rate_hz"])
    frames_per_chunk = max(1, sample_rate * chunk_ms // 1000)
    with wave.open(str(path), "rb") as audio:
        while chunk := audio.readframes(frames_per_chunk):
            yield chunk
            if realtime_pacing:
                frame_count = len(chunk) // 2
                await asyncio.sleep(frame_count / sample_rate)


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return a linearly interpolated percentile without a NumPy dependency."""

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


def _normalized_transcript(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def _reference_text(args: argparse.Namespace) -> str | None:
    if args.reference is not None:
        return str(args.reference).strip()
    if args.reference_file is not None:
        return args.reference_file.read_text(encoding="utf-8").strip()
    return None


def _safe_error(exc: Exception, api_key: str) -> str:
    message = str(exc)
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    return message


def _success_trial(
    trial: int,
    result: TranscriptionResult,
    *,
    reference: str | None,
) -> dict[str, Any]:
    strict_match = result.transcript == reference if reference is not None else None
    normalized_match = (
        _normalized_transcript(result.transcript) == _normalized_transcript(reference)
        if reference is not None
        else None
    )
    return {
        "trial": trial,
        "status": "success",
        "transcript": result.transcript,
        "language_code": result.language_code,
        "partial_count": len(result.partial_transcripts),
        "latency_ms": {
            "connection": result.connection_ms,
            "first_partial_from_start": result.time_to_first_partial_ms,
            "final_from_start": result.time_to_final_transcript_ms,
            "final_after_audio_eof": result.final_after_audio_end_ms,
            "total": result.total_ms,
        },
        "strict_reference_match": strict_match,
        "normalized_reference_match": normalized_match,
    }


def _build_report(
    *,
    trials: list[dict[str, Any]],
    audio: dict[str, int | float | str],
    args: argparse.Namespace,
    reference: str | None,
) -> dict[str, Any]:
    successful = [trial for trial in trials if trial["status"] == "success"]
    transcripts = [str(trial["transcript"]) for trial in successful]
    counts = Counter(transcripts)
    modal_transcript, modal_count = counts.most_common(1)[0] if counts else (None, 0)

    def latencies(field: str) -> list[float | None]:
        return [trial["latency_ms"][field] for trial in successful]

    attempted = len(trials)
    strict_matches = sum(
        trial["strict_reference_match"] is True for trial in successful
    )
    normalized_matches = sum(
        trial["normalized_reference_match"] is True for trial in successful
    )
    exact_denominator = attempted if reference is not None else 0
    reference_quality: dict[str, Any] | None = None
    if reference is not None:
        reference_quality = {
            "reference": reference,
            # Failed attempts count as non-matches; excluding them would inflate
            # recognition reliability in a submission report.
            "strict_exact_match": {
                "matches": strict_matches,
                "attempted_trials": exact_denominator,
                "rate": strict_matches / exact_denominator
                if exact_denominator
                else 0.0,
            },
            "normalized_exact_match": {
                "matches": normalized_matches,
                "attempted_trials": exact_denominator,
                "rate": (
                    normalized_matches / exact_denominator if exact_denominator else 0.0
                ),
            },
        }

    return {
        "schema_version": 1,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "sarvam",
        "audio": audio,
        "configuration": {
            "trials": args.trials,
            "language_code": args.language_code,
            "mode": args.mode,
            "endpointing": args.endpointing,
            "silence_duration_ms": args.silence_duration_ms,
            "chunk_ms": args.chunk_ms,
            "realtime_pacing": not args.no_realtime_pacing,
        },
        "outcomes": {
            "attempted": attempted,
            "successful": len(successful),
            "failed": attempted - len(successful),
            "success_rate": len(successful) / attempted if attempted else 0.0,
        },
        "latency_ms": {
            "connection": _distribution(latencies("connection")),
            "first_partial_from_start": _distribution(
                latencies("first_partial_from_start")
            ),
            "final_from_start": _distribution(latencies("final_from_start")),
            "final_after_audio_eof": _distribution(latencies("final_after_audio_eof")),
            "total": _distribution(latencies("total")),
        },
        "transcript_stability": {
            "successful_trials": len(successful),
            "unique_transcripts": len(counts),
            "all_identical": len(counts) == 1 if successful else False,
            "modal_transcript": modal_transcript,
            "modal_count": modal_count,
            "modal_rate": modal_count / len(successful) if successful else 0.0,
        },
        "reference_quality": reference_quality,
        "trials": trials,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


async def run(args: argparse.Namespace) -> int:
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.chunk_ms <= 0:
        raise ValueError("--chunk-ms must be positive")
    if args.silence_duration_ms <= 0:
        raise ValueError("--silence-duration-ms must be positive")

    audio = wav_metadata(args.audio)
    reference = _reference_text(args)
    load_dotenv()
    api_key = os.getenv("SARVAM_API_KEY", "")
    if not api_key:
        print("SARVAM_API_KEY is missing from the environment/.env", file=sys.stderr)
        return 2

    client = SarvamStreamingSTT(
        SarvamSTTSettings(
            api_key=api_key,
            language_code=args.language_code,
            mode=args.mode,
            endpointing=args.endpointing,
            sample_rate=int(audio["sample_rate_hz"]),
            silence_duration_ms=args.silence_duration_ms,
        )
    )
    measured: list[dict[str, Any]] = []
    for trial_number in range(1, args.trials + 1):

        def print_partial(text: str, current: int = trial_number) -> None:
            if args.show_partials:
                print(f"trial {current} partial: {text}", flush=True)

        trial_started = time.perf_counter()
        try:
            result = await client.transcribe(
                wav_chunks(
                    args.audio,
                    chunk_ms=args.chunk_ms,
                    realtime_pacing=not args.no_realtime_pacing,
                ),
                on_partial=print_partial,
            )
        # A benchmark must preserve provider/protocol failures as measured
        # outcomes rather than aborting early or quietly shrinking N.
        except Exception as exc:  # noqa: BLE001
            failed = {
                "trial": trial_number,
                "status": "failed",
                "elapsed_ms": (time.perf_counter() - trial_started) * 1000,
                "error_type": type(exc).__name__,
                "error": _safe_error(exc, api_key),
            }
            measured.append(failed)
            print(
                f"trial {trial_number}/{args.trials}: FAILED "
                f"({failed['error_type']}: {failed['error']})",
                file=sys.stderr,
                flush=True,
            )
            continue

        trial = _success_trial(trial_number, result, reference=reference)
        measured.append(trial)
        latency = trial["latency_ms"]
        print(
            f"trial {trial_number}/{args.trials}: "
            f"final-after-EOF={latency['final_after_audio_eof']:.3f} ms; "
            f"final-from-start={latency['final_from_start']:.3f} ms; "
            f"transcript={result.transcript!r}",
            flush=True,
        )

    report = _build_report(
        trials=measured,
        audio=audio,
        args=args,
        reference=reference,
    )
    if args.output is not None:
        _write_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["outcomes"]["failed"] == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
