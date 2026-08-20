#!/usr/bin/env python3
"""Profile bounded samples of ai4bharat/MSMARCO-XI without downloading it all.

The repository is tens of gigabytes and stores one Parquet file per language and
split.  This script therefore reads the Parquet files directly through ``hf://``
URIs, uses Hugging Face streaming, and caps the number of query records sampled
from each language independently.

The generated report is evidence for a later chunk-size decision.  It does not
choose or hard-code any chunking parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import platform
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

DATASET_ID = "ai4bharat/MSMARCO-XI"

# The current Hub files use ISO-639-3-like filename stems, while the public
# dataset card exposes the shorter language codes below.
LANGUAGES: dict[str, tuple[str, str]] = {
    "as": ("Assamese", "asm"),
    "bn": ("Bengali", "ben"),
    "gu": ("Gujarati", "guj"),
    "hi": ("Hindi", "hin"),
    "kn": ("Kannada", "kan"),
    "ml": ("Malayalam", "mal"),
    "mr": ("Marathi", "mar"),
    "ne": ("Nepali", "nep"),
    "or": ("Odia", "ori"),
    "pa": ("Punjabi", "pan"),
    "sa": ("Sanskrit", "san"),
    "ta": ("Tamil", "tam"),
    "te": ("Telugu", "tel"),
    "ur": ("Urdu", "urd"),
}

META_FIELDS = (
    "model_name",
    "temperature",
    "max_tokens",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
)
TOP_LEVEL_FIELDS = (
    "source_lang",
    "target_lang",
    "meta",
    "query",
    "Answer",
    "query_id",
    "query_type",
    "passages",
    "Eng_Query",
    "Eng_Answer",
)
PERCENTILES = (0, 25, 50, 70, 75, 90, 95, 99, 100)
LOGGER = logging.getLogger("inspect_dataset")


class InspectionError(RuntimeError):
    """Raised for an actionable dataset inspection failure."""


def percentile(values: Sequence[float], percent: float) -> float | None:
    """Return a linearly interpolated percentile (NumPy's default convention)."""

    if not values:
        return None
    if not 0 <= percent <= 100:
        raise ValueError("percent must be in [0, 100]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


@dataclass
class NumericSeries:
    values: list[float] = field(default_factory=list)

    def add(self, value: float | None) -> None:
        if value is None or isinstance(value, bool):
            return
        number = float(value)
        if math.isfinite(number):
            self.values.append(number)

    def extend(self, values: Iterable[float]) -> None:
        for value in values:
            self.add(value)

    def summary(self) -> dict[str, int | float | None]:
        if not self.values:
            return {"count": 0, "mean": None, **{f"p{p}": None for p in PERCENTILES}}
        result: dict[str, int | float | None] = {
            "count": len(self.values),
            "mean": statistics.fmean(self.values),
        }
        result.update({f"p{p}": percentile(self.values, p) for p in PERCENTILES})
        return result


def _counter_key(value: Any) -> str:
    if value is None or value == "":
        return "<missing>"
    if isinstance(value, float):
        return format(value, ".8g")
    return str(value)


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _text(value: Any) -> str:
    return value if isinstance(value, str) else "" if value is None else str(value)


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


def _passage_columns(
    record: Mapping[str, Any],
) -> tuple[list[Any], list[Any], list[Any]]:
    passages = record.get("passages")
    if isinstance(passages, Mapping):
        return (
            _list(passages.get("Translated_passages")),
            _list(passages.get("English_passages")),
            _list(passages.get("is_selected")),
        )

    # Tolerate a row-oriented representation if a future dataset conversion
    # changes the nested struct into a list of passage dictionaries.
    translated: list[Any] = []
    english: list[Any] = []
    selected: list[Any] = []
    for item in _list(passages):
        if isinstance(item, Mapping):
            translated.append(item.get("Translated_passages", item.get("translated")))
            english.append(item.get("English_passages", item.get("english")))
            selected.append(item.get("is_selected"))
    return translated, english, selected


def _observe_schema(value: Any, path: str, output: dict[str, set[str]]) -> None:
    if isinstance(value, Mapping):
        output[path or "$"].add("object")
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            _observe_schema(child, child_path, output)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        output[path or "$"].add("array")
        child = next((item for item in value if item is not None), None)
        if child is not None:
            _observe_schema(child, f"{path}[]", output)
    else:
        output[path or "$"].add(type(value).__name__)


@dataclass
class TokenCounter:
    name: str
    count_fn: Callable[[str], int]

    def count(self, text: str) -> int:
        return self.count_fn(text)


@dataclass
class ProfileState:
    token_counter: TokenCounter | None = None
    schema_sample_limit: int = 10
    record_count: int = 0
    metrics: defaultdict[str, NumericSeries] = field(
        default_factory=lambda: defaultdict(NumericSeries)
    )
    counters: defaultdict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    quality: Counter[str] = field(default_factory=Counter)
    coverage: Counter[str] = field(default_factory=Counter)
    schema: defaultdict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    _query_ids: set[str] = field(default_factory=set)

    def _measure_text(self, prefix: str, text: str) -> None:
        if not text:
            return
        self.metrics[f"{prefix}_characters"].add(len(text))
        self.metrics[f"{prefix}_whitespace_words"].add(len(text.split()))
        self.metrics[f"{prefix}_utf8_bytes"].add(len(text.encode("utf-8")))
        if self.token_counter is not None:
            self.metrics[f"{prefix}_tokens"].add(self.token_counter.count(text))

    def add_record(self, record: Mapping[str, Any]) -> None:
        if self.record_count < self.schema_sample_limit:
            _observe_schema(record, "", self.schema)

        self.record_count += 1
        self.quality["records"] += 1

        for field_name in TOP_LEVEL_FIELDS:
            value = record.get(field_name)
            if value is not None and value != "" and value != [] and value != {}:
                self.coverage[field_name] += 1

        meta = record.get("meta")
        meta = meta if isinstance(meta, Mapping) else {}
        for field_name in META_FIELDS:
            value = meta.get(field_name)
            if value is not None and value != "":
                self.coverage[f"meta.{field_name}"] += 1
            self.counters[f"meta.{field_name}"][_counter_key(value)] += 1

        for name in ("source_lang", "target_lang", "query_type"):
            self.counters[name][_counter_key(record.get(name))] += 1

        query_id = _counter_key(record.get("query_id"))
        if query_id in self._query_ids:
            self.quality["duplicate_query_ids_within_language_sample"] += 1
        self._query_ids.add(query_id)

        translated_query = _text(record.get("query"))
        english_query = _text(record.get("Eng_Query"))
        translated_answer = _text(record.get("Answer", record.get("answer")))
        english_answer = _text(record.get("Eng_Answer"))
        self._measure_text("translated_query", translated_query)
        self._measure_text("english_query", english_query)
        self._measure_text("translated_answer", translated_answer)
        self._measure_text("english_answer", english_answer)

        translated, english, selected = _passage_columns(record)
        lengths = (len(translated), len(english), len(selected))
        if len(set(lengths)) > 1:
            self.quality["records_with_mismatched_passage_arrays"] += 1
        slot_count = max(lengths, default=0)
        if slot_count == 0:
            self.quality["records_without_passages"] += 1
            return

        self.metrics["passages_per_record"].add(slot_count)
        self.quality["passage_slots"] += slot_count
        for index in range(slot_count):
            translated_text = (
                _text(translated[index]) if index < len(translated) else ""
            )
            english_text = _text(english[index]) if index < len(english) else ""
            label = selected[index] if index < len(selected) else None

            if translated_text:
                self.quality["nonempty_translated_passages"] += 1
                self._measure_text("translated_passage", translated_text)
            else:
                self.quality["empty_or_missing_translated_passages"] += 1

            if english_text:
                self.quality["nonempty_english_passages"] += 1
                self._measure_text("english_passage", english_text)
            else:
                self.quality["empty_or_missing_english_passages"] += 1

            if translated_text and english_text:
                self.quality["paired_nonempty_passages"] += 1
                self.metrics["passage_translation_character_ratio"].add(
                    len(translated_text) / len(english_text)
                )
                english_words = len(english_text.split())
                if english_words:
                    self.metrics["passage_translation_word_ratio"].add(
                        len(translated_text.split()) / english_words
                    )
                if translated_text.strip() == english_text.strip():
                    self.quality["translated_passages_equal_to_english"] += 1

            label_key = _counter_key(label)
            self.counters["is_selected"][label_key] += 1
            if label in (1, True, "1"):
                self.quality["selected_passages"] += 1
            elif label not in (0, False, "0", None, ""):
                self.quality["invalid_is_selected_labels"] += 1

    def merge(self, other: ProfileState) -> None:
        self.record_count += other.record_count
        for name, series in other.metrics.items():
            self.metrics[name].extend(series.values)
        for name, counter in other.counters.items():
            self.counters[name].update(counter)
        self.quality.update(other.quality)
        self.coverage.update(other.coverage)
        for path, types in other.schema.items():
            self.schema[path].update(types)

    def as_dict(self) -> dict[str, Any]:
        return {
            "records": self.record_count,
            "length_distributions": {
                name: series.summary() for name, series in sorted(self.metrics.items())
            },
            "categorical_distributions": {
                name: _counter_dict(counter)
                for name, counter in sorted(self.counters.items())
            },
            "field_coverage": {
                name: {
                    "present": count,
                    "records": self.record_count,
                    "rate": count / self.record_count if self.record_count else None,
                }
                for name, count in sorted(self.coverage.items())
            },
            "data_quality": dict(sorted(self.quality.items())),
            "observed_schema": {
                path: sorted(types) for path, types in sorted(self.schema.items())
            },
        }


@dataclass
class SampleResult:
    language: str
    file_path: str
    hf_uri: str
    records: list[Mapping[str, Any]]
    elapsed_ms: float


def expected_file_path(language: str, split: str) -> str:
    try:
        stem = LANGUAGES[language][1]
    except KeyError as exc:
        raise ValueError(f"Unsupported language code: {language}") from exc
    suffix = "train" if split == "train" else "val"
    return f"{split}/{stem}{suffix}.parquet"


def make_hf_uri(dataset_id: str, revision: str, file_path: str) -> str:
    return f"hf://datasets/{dataset_id}@{revision}/{file_path}"


def _retry_call(
    operation: str,
    function: Callable[[], Any],
    attempts: int,
) -> Any:
    try:
        from tenacity import (
            Retrying,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )
    except ImportError as exc:  # pragma: no cover - exercised by CLI dependency check
        raise InspectionError(
            "Missing dependency 'tenacity'. Install requirements.txt before profiling."
        ) from exc

    def before_sleep(retry_state: Any) -> None:
        LOGGER.warning(
            "%s failed (attempt %d/%d); retrying: %s",
            operation,
            retry_state.attempt_number,
            attempts,
            retry_state.outcome.exception(),
        )

    retrying = Retrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep,
        reraise=True,
    )
    for attempt in retrying:
        with attempt:
            return function()
    raise AssertionError("tenacity exhausted without returning or raising")


def fetch_repository_manifest(
    dataset_id: str,
    revision: str,
    attempts: int,
) -> tuple[dict[str, int | None], str | None, float]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - exercised by CLI dependency check
        raise InspectionError(
            "Missing dependency 'huggingface_hub'. Install requirements.txt before profiling."
        ) from exc

    token = os.getenv("HF_TOKEN") or None

    def request() -> Any:
        return HfApi(token=token).dataset_info(
            dataset_id,
            revision=revision,
            files_metadata=True,
        )

    started = time.perf_counter()
    info = _retry_call("Hugging Face dataset manifest", request, attempts)
    elapsed_ms = (time.perf_counter() - started) * 1000
    files = {
        sibling.rfilename: getattr(sibling, "size", None)
        for sibling in (getattr(info, "siblings", None) or [])
    }
    return files, getattr(info, "sha", None), elapsed_ms


def sample_language(
    *,
    dataset_id: str,
    revision: str,
    language: str,
    split: str,
    max_records: int,
    shuffle_buffer: int,
    seed: int,
    cache_dir: str | None,
    attempts: int,
) -> SampleResult:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised by CLI dependency check
        raise InspectionError(
            "Missing dependency 'datasets'. Install requirements.txt before profiling."
        ) from exc

    path = expected_file_path(language, split)
    uri = make_hf_uri(dataset_id, revision, path)

    def load_once() -> list[Mapping[str, Any]]:
        kwargs: dict[str, Any] = {
            "path": "parquet",
            "data_files": {split: uri},
            "split": split,
            "streaming": True,
        }
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        stream = load_dataset(**kwargs)
        if shuffle_buffer > 0:
            stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)

        records: list[Mapping[str, Any]] = []
        for record in stream:
            if not isinstance(record, Mapping):
                raise InspectionError(
                    f"Expected mapping records in {path}; received {type(record).__name__}"
                )
            records.append(record)
            if len(records) >= max_records:
                break
        if not records:
            raise InspectionError(f"No records were read from {path}")
        return records

    started = time.perf_counter()
    records = _retry_call(f"stream sample {language}/{split}", load_once, attempts)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return SampleResult(language, path, uri, records, elapsed_ms)


def load_token_counter(model_name: str, attempts: int) -> tuple[TokenCounter, float]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised by CLI dependency check
        raise InspectionError(
            "--tokenizer requires 'transformers'. Install requirements.txt first."
        ) from exc

    token = os.getenv("HF_TOKEN") or None

    def load() -> Any:
        return AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            use_fast=True,
            trust_remote_code=False,
        )

    started = time.perf_counter()
    tokenizer = _retry_call(f"tokenizer {model_name}", load, attempts)
    elapsed_ms = (time.perf_counter() - started) * 1000

    def count(text: str) -> int:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return len(encoded["input_ids"])

    return TokenCounter(model_name, count), elapsed_ms


def parse_languages(raw_values: Sequence[str]) -> list[str]:
    values: list[str] = []
    for raw in raw_values:
        values.extend(item.strip().lower() for item in raw.split(",") if item.strip())
    if not values or values == ["all"]:
        return list(LANGUAGES)
    if "all" in values:
        raise ValueError("Use 'all' by itself, or provide explicit language codes")
    unknown = sorted(set(values) - LANGUAGES.keys())
    if unknown:
        raise ValueError(
            f"Unknown language code(s): {', '.join(unknown)}. "
            f"Choose from: {', '.join(LANGUAGES)}"
        )
    return list(dict.fromkeys(values))


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in (
        "datasets",
        "huggingface-hub",
        "pyarrow",
        "transformers",
        "tenacity",
    ):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def _atomic_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def _length_rows(language_states: Mapping[str, ProfileState]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for language, state in language_states.items():
        for metric, series in sorted(state.metrics.items()):
            rows.append({"language": language, "metric": metric, **series.summary()})
    return rows


def _language_rows(
    languages: Sequence[str],
    results: Mapping[str, dict[str, Any]],
    states: Mapping[str, ProfileState],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for language in languages:
        result = results[language]
        state = states.get(language)
        passage_chars = (
            state.metrics["translated_passage_characters"].summary() if state else {}
        )
        passage_words = (
            state.metrics["translated_passage_whitespace_words"].summary()
            if state
            else {}
        )
        rows.append(
            {
                "language": language,
                "language_name": LANGUAGES[language][0],
                "status": result["status"],
                "expected_file": result["expected_file"],
                "repository_file_bytes": result.get("repository_file_bytes"),
                "sampled_records": state.record_count if state else 0,
                "passage_slots": state.quality.get("passage_slots", 0) if state else 0,
                "passage_char_p50": passage_chars.get("p50"),
                "passage_char_p95": passage_chars.get("p95"),
                "passage_char_p100": passage_chars.get("p100"),
                "passage_word_p50": passage_words.get("p50"),
                "passage_word_p95": passage_words.get("p95"),
                "error": result.get("error"),
            }
        )
    return rows


def _plot_passage_lengths(
    output_path: Path, states: Mapping[str, ProfileState]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    translated_chars = [
        value
        for state in states.values()
        for value in state.metrics["translated_passage_characters"].values
    ]
    translated_words = [
        value
        for state in states.values()
        for value in state.metrics["translated_passage_whitespace_words"].values
    ]
    english_chars = [
        value
        for state in states.values()
        for value in state.metrics["english_passage_characters"].values
    ]
    ratios = [
        value
        for state in states.values()
        for value in state.metrics["passage_translation_character_ratio"].values
    ]
    if not translated_chars:
        raise InspectionError("No translated passage lengths are available to plot")

    def clipped(values: list[float]) -> tuple[list[float], float]:
        cap = percentile(values, 99) or max(values)
        return [min(value, cap) for value in values], cap

    plot_chars, char_cap = clipped(translated_chars)
    plot_words, word_cap = clipped(translated_words)
    plot_ratios, ratio_cap = clipped(ratios) if ratios else ([], 0)

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(plot_chars, bins=50, color="#345995", alpha=0.85)
    axes[0, 0].set_title(
        f"Translated passage characters (values clipped at P99={char_cap:.0f})"
    )
    axes[0, 0].set_xlabel("Unicode code points")
    axes[0, 0].set_ylabel("Passages")

    axes[0, 1].hist(plot_words, bins=50, color="#03cea4", alpha=0.85)
    axes[0, 1].set_title(
        f"Translated passage words (values clipped at P99={word_cap:.0f})"
    )
    axes[0, 1].set_xlabel("Whitespace-delimited words")
    axes[0, 1].set_ylabel("Passages")

    _, english_cap = clipped(english_chars) if english_chars else ([], 0)
    common_cap = max(char_cap, english_cap)
    axes[1, 0].hist(
        [min(value, common_cap) for value in translated_chars],
        bins=50,
        density=True,
        alpha=0.55,
        label="Translated",
    )
    if english_chars:
        axes[1, 0].hist(
            [min(value, common_cap) for value in english_chars],
            bins=50,
            density=True,
            alpha=0.45,
            label="English",
        )
    axes[1, 0].set_title("English vs translated passage characters (P99-clipped)")
    axes[1, 0].set_xlabel("Unicode code points")
    axes[1, 0].legend()

    if plot_ratios:
        axes[1, 1].hist(plot_ratios, bins=50, color="#fb4d3d", alpha=0.8)
        axes[1, 1].set_title(
            f"Translated/English character ratio (values clipped at P99={ratio_cap:.2f})"
        )
        axes[1, 1].set_xlabel("Ratio")
        axes[1, 1].set_ylabel("Paired passages")
    else:
        axes[1, 1].text(0.5, 0.5, "No paired passages", ha="center", va="center")

    figure.suptitle("MSMARCO-XI bounded-sample passage profile", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _markdown_report(
    report: Mapping[str, Any],
    language_rows: Sequence[Mapping[str, Any]],
) -> str:
    sampling = report["sampling"]
    lines = [
        "# MSMARCO-XI dataset profile",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        (
            "This is a bounded streaming sample used to inform a later chunking decision. "
            "It is not a corpus-wide census, and **no chunk size is selected here**."
        ),
        "",
        "## Sampling",
        "",
        f"- Dataset: `{report['dataset']['id']}`",
        f"- Requested revision: `{report['dataset']['requested_revision']}`",
        f"- Resolved revision: `{report['dataset'].get('resolved_revision') or 'unknown'}`",
        f"- Split: `{sampling['split']}`",
        f"- Maximum records per language: `{sampling['max_records_per_language']}`",
        (
            f"- Shuffle buffer: `{sampling['shuffle_buffer']}` "
            "(`0` means a deterministic head sample)"
        ),
        f"- Tokenizer: `{sampling.get('tokenizer') or 'not requested'}`",
        "",
        (
            "> Language proportions below describe independently capped samples. They must not "
            "be interpreted as the full corpus's language proportions."
        ),
        "",
        "## Per-language passage lengths",
        "",
        "| Lang | Status | Records | Passages | Chars P50 | Chars P95 | Chars max | Words P50 | Words P95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def display(row: Mapping[str, Any], key: str) -> str:
        value = row.get(key)
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.1f}"
        return str(value)

    for row in language_rows:
        lines.append(
            "| {language} | {status} | {sampled_records} | {passage_slots} | "
            "{p50} | {p95} | {p100} | {w50} | {w95} |".format(
                language=row["language"],
                status=row["status"],
                sampled_records=display(row, "sampled_records"),
                passage_slots=display(row, "passage_slots"),
                p50=display(row, "passage_char_p50"),
                p95=display(row, "passage_char_p95"),
                p100=display(row, "passage_char_p100"),
                w50=display(row, "passage_word_p50"),
                w95=display(row, "passage_word_p95"),
            )
        )

    errors = {
        language: result.get("error")
        for language, result in report["languages"].items()
        if result["status"] != "ok"
    }
    if errors:
        lines.extend(["", "## Unavailable or failed samples", ""])
        for language, error in errors.items():
            lines.append(f"- `{language}`: {error}")

    query_types = report["overall_profile"]["categorical_distributions"].get(
        "query_type", {}
    )
    if query_types:
        lines.extend(["", "## Observed query types", ""])
        for query_type, count in list(query_types.items())[:20]:
            lines.append(f"- `{query_type}`: {count}")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `profile.json`: complete machine-readable summaries, schema, metadata, and quality checks",
            "- `language_summary.csv`: one row per requested language",
            "- `length_summary.csv`: percentiles for every text-length metric and language",
            "- `passage_length_distribution.png`: P99-clipped visualization (when plotting succeeds)",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream bounded samples of ai4bharat/MSMARCO-XI and write language, "
            "schema, metadata, passage-length, and data-quality profiles."
        )
    )
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--split", choices=("train", "validation"), default="validation"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["all"],
        metavar="CODE",
        help="Language codes separated by spaces or commas; default: all",
    )
    parser.add_argument(
        "--max-records-per-language",
        type=int,
        default=250,
        help="Independent sample cap for each language (default: 250)",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=0,
        help=(
            "Streaming shuffle buffer; 0 uses a faster deterministic head sample. "
            "A value such as 2500 reduces ordering bias but reads more data."
        ),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--schema-samples", type=int, default=10)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "Optional Hugging Face tokenizer id for exact token-length distributions. "
            "Leave unset until the embedding model candidate is chosen."
        ),
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/dataset_profile"),
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a PNG distribution plot (default: enabled)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first unavailable language or sampling failure",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved sampling plan without network access",
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List supported language codes and exit",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def inspect_dataset(args: argparse.Namespace) -> int:
    try:
        languages = parse_languages(args.languages)
    except ValueError as exc:
        raise InspectionError(str(exc)) from exc

    if args.max_records_per_language <= 0:
        raise InspectionError("--max-records-per-language must be greater than zero")
    if args.shuffle_buffer < 0:
        raise InspectionError("--shuffle-buffer cannot be negative")
    if args.schema_samples < 0:
        raise InspectionError("--schema-samples cannot be negative")
    if args.retries <= 0:
        raise InspectionError("--retries must be greater than zero")

    plan = {
        language: {
            "name": LANGUAGES[language][0],
            "file": expected_file_path(language, args.split),
            "uri": make_hf_uri(
                args.dataset_id,
                args.revision,
                expected_file_path(language, args.split),
            ),
        }
        for language in languages
    }
    if args.dry_run:
        print(
            json.dumps(
                {"dataset": args.dataset_id, "split": args.split, "plan": plan},
                indent=2,
            )
        )
        return 0

    tokenizer: TokenCounter | None = None
    tokenizer_load_ms: float | None = None
    if args.tokenizer:
        LOGGER.info("Loading tokenizer %s", args.tokenizer)
        tokenizer, tokenizer_load_ms = load_token_counter(args.tokenizer, args.retries)

    manifest_files: dict[str, int | None] = {}
    resolved_revision: str | None = None
    manifest_ms: float | None = None
    manifest_error: str | None = None
    try:
        LOGGER.info("Fetching repository manifest for %s", args.dataset_id)
        manifest_files, resolved_revision, manifest_ms = fetch_repository_manifest(
            args.dataset_id, args.revision, args.retries
        )
    except Exception as exc:  # noqa: BLE001 - manifest failure has a documented fallback
        manifest_error = f"{type(exc).__name__}: {exc}"
        LOGGER.warning(
            "Manifest lookup failed; sampling expected paths directly: %s", exc
        )

    overall = ProfileState(
        token_counter=tokenizer, schema_sample_limit=args.schema_samples
    )
    states: dict[str, ProfileState] = {}
    language_results: dict[str, dict[str, Any]] = {}

    for position, language in enumerate(languages, start=1):
        expected_path = expected_file_path(language, args.split)
        file_known_missing = (
            bool(manifest_files) and expected_path not in manifest_files
        )
        base_result: dict[str, Any] = {
            "name": LANGUAGES[language][0],
            "status": "pending",
            "expected_file": expected_path,
            "repository_file_bytes": manifest_files.get(expected_path),
            "sample_cap": args.max_records_per_language,
        }
        language_results[language] = base_result

        if file_known_missing:
            message = f"File is absent at requested revision: {expected_path}"
            base_result.update(status="unavailable", error=message)
            LOGGER.warning("[%d/%d] %s", position, len(languages), message)
            if args.fail_fast:
                raise InspectionError(message)
            continue

        LOGGER.info(
            "[%d/%d] Streaming at most %d %s records",
            position,
            len(languages),
            args.max_records_per_language,
            language,
        )
        try:
            sample = sample_language(
                dataset_id=args.dataset_id,
                revision=args.revision,
                language=language,
                split=args.split,
                max_records=args.max_records_per_language,
                shuffle_buffer=args.shuffle_buffer,
                seed=args.seed + position,
                cache_dir=args.cache_dir,
                attempts=args.retries,
            )
            state = ProfileState(
                token_counter=tokenizer,
                schema_sample_limit=args.schema_samples,
            )
            for record in sample.records:
                state.add_record(record)
            states[language] = state
            overall.merge(state)
            base_result.update(
                status="ok",
                sampled_records=state.record_count,
                sampled_passage_slots=state.quality.get("passage_slots", 0),
                load_and_sample_ms=sample.elapsed_ms,
                hf_uri=sample.hf_uri,
                profile=state.as_dict(),
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            base_result.update(status="error", error=message)
            LOGGER.error("Failed to profile %s: %s", language, message)
            if args.fail_fast:
                raise

    if not states:
        raise InspectionError("No language sample succeeded; no profile was written")

    observed_target_mix = overall.counters.get("target_lang", Counter())
    requested_sample_mix = {
        language: state.record_count for language, state in states.items()
    }
    repository_file_bytes = {
        language: size
        for language in languages
        if (size := manifest_files.get(expected_file_path(language, args.split)))
        is not None
    }
    total_repository_bytes = sum(repository_file_bytes.values())
    repository_byte_share = {
        language: size / total_repository_bytes
        for language, size in repository_file_bytes.items()
    }
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Dataset evidence for later Late Chunking and naive-baseline parameter selection; "
            "this report does not finalize chunk sizes."
        ),
        "dataset": {
            "id": args.dataset_id,
            "requested_revision": args.revision,
            "resolved_revision": resolved_revision,
            "manifest_file_count": len(manifest_files) if manifest_files else None,
            "manifest_error": manifest_error,
        },
        "sampling": {
            "split": args.split,
            "requested_languages": languages,
            "max_records_per_language": args.max_records_per_language,
            "method": "buffered_stream_shuffle"
            if args.shuffle_buffer
            else "stream_head",
            "shuffle_buffer": args.shuffle_buffer,
            "seed": args.seed,
            "tokenizer": tokenizer.name if tokenizer else None,
            "caveat": (
                "Each language is capped independently. Counts are the observed bounded sample, "
                "not corpus-wide language proportions. Head sampling may reflect source ordering."
            ),
        },
        "timings_ms": {
            "repository_manifest": manifest_ms,
            "tokenizer_load": tokenizer_load_ms,
            "per_language_load_and_sample": {
                language: result.get("load_and_sample_ms")
                for language, result in language_results.items()
                if result.get("load_and_sample_ms") is not None
            },
        },
        "language_mix": {
            "requested_language_sample_records": requested_sample_mix,
            "observed_target_lang_values": _counter_dict(observed_target_mix),
            "repository_file_bytes_for_requested_split": repository_file_bytes,
            "repository_byte_share_for_requested_split": repository_byte_share,
            "interpretation": (
                "This mix validates sampled language coverage only; it is not an estimate of "
                "full-dataset prevalence because per-language samples use equal caps. File-byte "
                "shares describe compressed storage, not row or content proportions."
            ),
        },
        "languages": language_results,
        "overall_profile": overall.as_dict(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_status: dict[str, Any] = {"requested": args.plots, "written": False}
    if args.plots:
        plot_path = args.output_dir / "passage_length_distribution.png"
        try:
            _plot_passage_lengths(plot_path, states)
            plot_status.update(written=True, path=plot_path.name)
        except Exception as exc:  # noqa: BLE001 - plotting is an optional artifact
            plot_status["error"] = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("Could not write plot: %s", exc)
    report["artifacts"] = {"plot": plot_status}

    length_fieldnames = [
        "language",
        "metric",
        "count",
        "mean",
        *[f"p{p}" for p in PERCENTILES],
    ]
    language_fieldnames = [
        "language",
        "language_name",
        "status",
        "expected_file",
        "repository_file_bytes",
        "sampled_records",
        "passage_slots",
        "passage_char_p50",
        "passage_char_p95",
        "passage_char_p100",
        "passage_word_p50",
        "passage_word_p95",
        "error",
    ]
    language_rows = _language_rows(languages, language_results, states)
    _atomic_csv(
        args.output_dir / "length_summary.csv",
        length_fieldnames,
        _length_rows(states),
    )
    _atomic_csv(
        args.output_dir / "language_summary.csv",
        language_fieldnames,
        language_rows,
    )
    _atomic_text(
        args.output_dir / "PROFILE.md",
        _markdown_report(report, language_rows),
    )
    _atomic_text(
        args.output_dir / "profile.json",
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )

    LOGGER.info(
        "Profiled %d records and %d passage slots across %d/%d requested languages",
        overall.record_count,
        overall.quality.get("passage_slots", 0),
        len(states),
        len(languages),
    )
    LOGGER.info("Wrote profile artifacts to %s", args.output_dir)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.list_languages:
        for code, (name, stem) in LANGUAGES.items():
            print(f"{code}\t{name}\t{stem}")
        return 0
    try:
        return inspect_dataset(args)
    except (InspectionError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
