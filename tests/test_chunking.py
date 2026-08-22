#!/usr/bin/env python3
"""Compare real late and naive chunking on sampled MSMARCO-XI passages."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chunking import (
    JinaEmbeddingModel,
    LateChunkingParameters,
    NaiveChunkingParameters,
    late_chunking,
    naive_recursive_chunking,
)
from scripts.inspect_dataset import DATASET_ID, sample_language


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare both chunking strategies on real MSMARCO-XI passages."
    )
    parser.add_argument("--language", default="hi")
    parser.add_argument("--sample-records", type=int, default=30)
    parser.add_argument("--documents", type=int, default=6)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("data/chunking_comparison.json")
    )
    return parser


def _translated_passages(record: Mapping[str, Any]) -> list[str]:
    passages = record.get("passages")
    if not isinstance(passages, Mapping):
        return []
    values = passages.get("Translated_passages")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _select_documents(
    records: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    count: int,
    language: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        query_id = str(record.get("query_id", "unknown"))
        for passage_index, text in enumerate(_translated_passages(record)):
            if text in seen:
                continue
            seen.add(text)
            token_count = len(
                tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]
            )
            candidates.append(
                {
                    "document_id": f"{language}:{query_id}:{passage_index}",
                    "text": text,
                    "token_count": token_count,
                    "language": language,
                    "query_id": query_id,
                }
            )
    if len(candidates) < count:
        raise RuntimeError(
            f"Only {len(candidates)} unique passages were sampled; need {count}"
        )
    candidates.sort(key=lambda item: item["token_count"])
    if count == 1:
        return [candidates[len(candidates) // 2]]
    positions = {
        round(index * (len(candidates) - 1) / (count - 1)) for index in range(count)
    }
    return [candidates[position] for position in sorted(positions)]


def _mean(values: Sequence[int]) -> float:
    return statistics.fmean(values) if values else 0.0


def run(args: argparse.Namespace) -> int:
    if args.sample_records <= 0 or args.documents <= 0:
        raise ValueError("Sample and document counts must be positive")
    load_dotenv()
    sampled = sample_language(
        dataset_id=DATASET_ID,
        revision="main",
        language=args.language,
        split="validation",
        max_records=args.sample_records,
        shuffle_buffer=0,
        seed=2026,
        cache_dir=None,
        attempts=3,
    )
    encoder = JinaEmbeddingModel(device=args.device)
    documents = _select_documents(
        sampled.records,
        tokenizer=encoder.tokenizer,
        count=args.documents,
        language=args.language,
    )
    late_parameters = LateChunkingParameters()
    naive_parameters = NaiveChunkingParameters()

    late_chunks = []
    late_token_counts: list[int] = []
    late_by_document: dict[str, list[str]] = {}
    late_started = time.perf_counter()
    for document in documents:
        result = late_chunking(
            text=document["text"],
            document_id=document["document_id"],
            encoder=encoder,
            parameters=late_parameters,
            metadata={"language": args.language, "query_id": document["query_id"]},
        )
        late_chunks.extend(result.chunks)
        late_token_counts.extend(result.token_counts)
        late_by_document[document["document_id"]] = [
            chunk.text for chunk in result.chunks
        ]
    late_ms = (time.perf_counter() - late_started) * 1000

    naive_chunks = []
    naive_by_document: dict[str, list[str]] = {}
    naive_started = time.perf_counter()
    for document in documents:
        chunks = naive_recursive_chunking(
            text=document["text"],
            document_id=document["document_id"],
            parameters=naive_parameters,
            metadata={"language": args.language, "query_id": document["query_id"]},
        )
        naive_chunks.extend(chunks)
        naive_by_document[document["document_id"]] = [chunk.text for chunk in chunks]
    naive_ms = (time.perf_counter() - naive_started) * 1000
    naive_token_counts = [
        len(
            encoder.tokenizer(chunk.text, add_special_tokens=False, truncation=False)[
                "input_ids"
            ]
        )
        for chunk in naive_chunks
    ]

    qualitative = []
    for document in documents[-3:]:
        document_id = document["document_id"]
        qualitative.append(
            {
                "document_id": document_id,
                "document_tokens": document["token_count"],
                "late_chunks": [text[:500] for text in late_by_document[document_id]][
                    :3
                ],
                "naive_chunks": [text[:500] for text in naive_by_document[document_id]][
                    :3
                ],
            }
        )

    report = {
        "dataset": DATASET_ID,
        "split": "validation",
        "language": args.language,
        "sampled_records": len(sampled.records),
        "selected_documents": len(documents),
        "selected_document_token_counts": [
            document["token_count"] for document in documents
        ],
        "model": encoder.model_name,
        "model_exposes_token_embeddings": encoder.exposes_token_embeddings,
        "model_max_tokens": encoder.max_tokens,
        "late_parameters": asdict(late_parameters),
        "naive_parameters": asdict(naive_parameters),
        "late": {
            "chunks": len(late_chunks),
            "average_characters": _mean([len(chunk.text) for chunk in late_chunks]),
            "average_tokens": _mean(late_token_counts),
            "processing_ms": late_ms,
        },
        "naive": {
            "chunks": len(naive_chunks),
            "average_characters": _mean([len(chunk.text) for chunk in naive_chunks]),
            "average_tokens": _mean(naive_token_counts),
            "processing_ms": naive_ms,
        },
        "qualitative_samples": qualitative,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
