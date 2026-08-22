"""Shared, provider-neutral helpers for real-audio WebSocket benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.schemas import RAGResponse, RefusalReason, TranscriptionResult

_PCM_SAMPLE_RATE = 16_000
_PCM_SAMPLE_WIDTH_BYTES = 2


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
            Path(temporary_name).unlink(missing_ok=True)


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
    oracle_checks = [
        check
        for check in (answer_contains_match, chunk_ids_match, document_ids_match)
        if check is not None
    ]
    oracle_configured = bool(oracle_checks)
    oracle_match = bool(oracle_configured and all(oracle_checks))
    outcome_quality = bool(
        status_match
        and refusal_reason_match
        and (
            sample.expected_status == "refused"
            or (grounded_answer and oracle_match)
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
        "answer_oracle_configured": oracle_configured,
        "expected_answer_contains": list(sample.expected_answer_contains),
        "answer_contains_match": answer_contains_match,
        "expected_chunk_ids": list(sample.expected_chunk_ids),
        "supporting_chunk_ids_match": chunk_ids_match,
        "expected_document_ids": list(sample.expected_document_ids),
        "supporting_document_ids_match": document_ids_match,
        "answer_oracle_match": oracle_match,
        "reference_required": sample.reference is not None,
        "reference_strict_match": reference_strict_match,
        "reference_normalized_match": reference_normalized_match,
        "eligible": bool(rag is not None and outcome_quality and reference_quality),
    }

