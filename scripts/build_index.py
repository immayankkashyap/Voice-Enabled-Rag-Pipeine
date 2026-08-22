#!/usr/bin/env python3
"""Build the bounded multilingual MSMARCO-XI late-chunked FAISS bundle."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chunking import (
    JINA_FULL_DIMENSION,
    JinaEmbeddingModel,
    LateChunkingParameters,
    late_chunking,
)
from scripts.inspect_dataset import DATASET_ID, parse_languages, sample_language

DEFAULT_LANGUAGES = ("hi", "ta", "ur")
DEFAULT_RECORDS_PER_LANGUAGE = 10
DEFAULT_MAX_DOCUMENTS = 300


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build full and native-MRL FAISS indices from MSMARCO-XI."
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(DEFAULT_LANGUAGES),
        help="Dataset language codes (default: hi ta ur).",
    )
    parser.add_argument(
        "--records-per-language", type=int, default=DEFAULT_RECORDS_PER_LANGUAGE
    )
    parser.add_argument("--max-documents", type=int, default=DEFAULT_MAX_DOCUMENTS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mrl-dimension", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("data/faiss_index"))
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("data/index_build_work"),
        help="Native-runtime handoff directory used during the build.",
    )
    parser.add_argument(
        "--source-cache",
        type=Path,
        default=Path("data/index_source_subset.json"),
        help="Local handoff file between dataset acquisition and model inference.",
    )
    parser.add_argument(
        "--reuse-source-cache",
        action="store_true",
        help="Skip Hub access and reuse --source-cache after validating its selection.",
    )
    parser.add_argument("--fetch-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--faiss-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--report", type=Path, default=Path("data/index_build_report.json")
    )
    return parser


def _values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


def _collect_source_data(
    records: Sequence[Mapping[str, Any]], language: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for record_number, record in enumerate(records):
        query_id = str(record.get("query_id", record_number))
        translated_query = str(record.get("query") or "").strip()
        english_query = str(record.get("Eng_Query") or "").strip()
        if translated_query:
            queries.append(
                {
                    "id": f"{language}:{query_id}",
                    "query_id": query_id,
                    "language": language,
                    "query": translated_query,
                    "english_query": english_query,
                }
            )

        passages = record.get("passages")
        if not isinstance(passages, Mapping):
            continue
        texts = _values(passages.get("Translated_passages"))
        selections = _values(passages.get("is_selected"))
        for passage_index, value in enumerate(texts):
            text = str(value or "").strip()
            if not text:
                continue
            selected_value = (
                bool(selections[passage_index])
                if passage_index < len(selections)
                and selections[passage_index] is not None
                else False
            )
            documents.append(
                {
                    "id": f"{language}:{query_id}:{passage_index}",
                    "text": text,
                    "language": language,
                    "query_id": query_id,
                    "passage_index": passage_index,
                    "is_selected": selected_value,
                }
            )
    return documents, queries


def _artifact_sizes(output_dir: Path) -> dict[str, int]:
    names = ("mrl.faiss", "full.faiss", "full_embeddings.npy", "chunks.json")
    return {
        name: (output_dir / name).stat().st_size
        for name in names
        if (output_dir / name).is_file()
    }


def _fetch_source_subset(
    *, languages: list[str], records_per_language: int, output: Path
) -> dict[str, Any]:
    source_documents: list[dict[str, Any]] = []
    source_queries: list[dict[str, Any]] = []
    fetch_by_language_ms: dict[str, float] = {}
    for language in languages:
        print(
            f"Fetching {records_per_language} {language} validation records...",
            flush=True,
        )
        sample = sample_language(
            dataset_id=DATASET_ID,
            revision="main",
            language=language,
            split="validation",
            max_records=records_per_language,
            shuffle_buffer=0,
            seed=2026,
            cache_dir=None,
            attempts=3,
        )
        documents, queries = _collect_source_data(sample.records, language)
        source_documents.extend(documents)
        source_queries.extend(queries)
        fetch_by_language_ms[language] = sample.elapsed_ms
    payload = {
        "dataset": DATASET_ID,
        "revision": "main",
        "split": "validation",
        "languages": languages,
        "records_per_language": records_per_language,
        "documents": source_documents,
        "queries": source_queries,
        "fetch_by_language_ms": fetch_by_language_ms,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _load_source_subset(
    path: Path, *, languages: list[str], records_per_language: int
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "dataset": DATASET_ID,
        "revision": "main",
        "split": "validation",
        "languages": languages,
        "records_per_language": records_per_language,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Source cache selection does not match this build request")
    if not payload.get("documents") or not payload.get("queries"):
        raise ValueError("Source cache has no documents or queries")
    return payload


def _run_faiss_phase(args: argparse.Namespace) -> int:
    """Build FAISS artifacts without importing Torch in this process."""

    from app.indexing import FaissVectorStore, IndexDimensions
    from app.schemas import Chunk

    embeddings = np.load(args.work_dir / "embeddings.npy", allow_pickle=False)
    raw_chunks = json.loads((args.work_dir / "chunks.json").read_text(encoding="utf-8"))
    chunks = [Chunk.model_validate(value) for value in raw_chunks]
    provenance = json.loads(
        (args.work_dir / "provenance.json").read_text(encoding="utf-8")
    )
    dimensions = IndexDimensions(full=JINA_FULL_DIMENSION, mrl=args.mrl_dimension)
    store = FaissVectorStore(dimensions, native_mrl=True, provenance=provenance)
    started = time.perf_counter()
    store.add(chunks, embeddings)
    store.save(args.output_dir)
    elapsed_ms = (time.perf_counter() - started) * 1000
    (args.work_dir / "faiss_phase_report.json").write_text(
        json.dumps(
            {"faiss_add_and_save_ms": elapsed_ms, "chunks": len(chunks)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def run(args: argparse.Namespace) -> int:
    if args.records_per_language <= 0 or args.max_documents <= 0:
        raise ValueError("Record and document limits must be positive")
    load_dotenv()
    languages = parse_languages(args.languages)
    if args.faiss_only:
        return _run_faiss_phase(args)
    if args.fetch_only:
        _fetch_source_subset(
            languages=languages,
            records_per_language=args.records_per_language,
            output=args.source_cache,
        )
        return 0

    total_started = time.perf_counter()

    if not args.reuse_source_cache:
        # Arrow-backed streaming can retain large native row-group buffers.
        # Acquire data in a short-lived process so Jina/Torch starts with a
        # clean native heap; this avoids a reproducible macOS exit-139 crash.
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--languages",
            *languages,
            "--records-per-language",
            str(args.records_per_language),
            "--source-cache",
            str(args.source_cache),
            "--fetch-only",
        ]
        subprocess.run(command, check=True)
    source_payload = _load_source_subset(
        args.source_cache,
        languages=languages,
        records_per_language=args.records_per_language,
    )
    source_documents = source_payload["documents"]
    source_queries = source_payload["queries"]
    fetch_by_language_ms = source_payload["fetch_by_language_ms"]

    available_documents = len(source_documents)
    source_documents = source_documents[: args.max_documents]
    if not source_documents:
        raise RuntimeError("The selected dataset subset contained no passages")

    model_started = time.perf_counter()
    encoder = JinaEmbeddingModel(device=args.device)
    model_load_ms = (time.perf_counter() - model_started) * 1000
    dimensions = {"full": JINA_FULL_DIMENSION, "mrl": args.mrl_dimension}
    if dimensions["full"] != encoder.embedding_dimension:
        raise RuntimeError("Full index dimension does not match the embedding model")
    if dimensions["mrl"] not in encoder.matryoshka_dimensions:
        raise RuntimeError(
            f"{dimensions['mrl']} is not a native Matryoshka dimension for "
            f"{encoder.model_name}"
        )

    late_parameters = LateChunkingParameters()
    all_chunks = []
    all_embeddings: list[np.ndarray] = []
    all_token_counts: list[int] = []
    chunking_started = time.perf_counter()
    for document_number, document in enumerate(source_documents, start=1):
        result = late_chunking(
            text=document["text"],
            document_id=document["id"],
            encoder=encoder,
            parameters=late_parameters,
            metadata={
                "language": document["language"],
                "query_id": document["query_id"],
                "passage_index": document["passage_index"],
                "is_selected": document["is_selected"],
            },
        )
        all_chunks.extend(result.chunks)
        all_embeddings.append(result.embeddings)
        all_token_counts.extend(result.token_counts)
        if document_number % 25 == 0 or document_number == len(source_documents):
            print(
                f"Late-chunked {document_number}/{len(source_documents)} documents "
                f"into {len(all_chunks)} chunks...",
                flush=True,
            )
    chunking_embedding_ms = (time.perf_counter() - chunking_started) * 1000
    embedding_matrix = np.ascontiguousarray(
        np.concatenate(all_embeddings, axis=0), dtype=np.float32
    )

    subset_reason = (
        f"Bounded local build: first {args.records_per_language} validation query "
        f"records for {', '.join(languages)} (covering distinct Indic scripts), "
        f"capped at {args.max_documents} passages to keep CPU indexing reproducible."
    )
    provenance = {
        "dataset": DATASET_ID,
        "revision": "main",
        "split": "validation",
        "languages": languages,
        "records_per_language": args.records_per_language,
        "document_limit": args.max_documents,
        "subset_reason": subset_reason,
        "embedding_model": encoder.model_name,
        "embedding_model_revision": encoder.model_revision,
        "embedding_task": "retrieval.passage",
        "late_chunking_parameters": asdict(late_parameters),
        "mrl_note": (
            "Jina v3 is natively Matryoshka-trained at both stored dimensions; "
            "the low-dimensional index is not an untrained truncation proxy."
        ),
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.work_dir / "embeddings.npy", embedding_matrix, allow_pickle=False)
    (args.work_dir / "chunks.json").write_text(
        json.dumps(
            [chunk.model_dump(mode="json") for chunk in all_chunks],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.work_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # FAISS and Torch both ship native parallel runtimes that segfault when
    # real HNSW work follows Jina inference on this macOS environment. Run the
    # exact FAISS phase in a clean child rather than weakening either stage.
    faiss_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--faiss-only",
        "--mrl-dimension",
        str(args.mrl_dimension),
        "--work-dir",
        str(args.work_dir),
        "--output-dir",
        str(args.output_dir),
    ]
    subprocess.run(faiss_command, check=True)
    faiss_report = json.loads(
        (args.work_dir / "faiss_phase_report.json").read_text(encoding="utf-8")
    )
    if int(faiss_report["chunks"]) != len(all_chunks):
        raise RuntimeError("FAISS worker indexed an unexpected number of chunks")
    index_add_save_ms = float(faiss_report["faiss_add_and_save_ms"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "source_queries.json").write_text(
        json.dumps(source_queries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    sizes = _artifact_sizes(args.output_dir)
    total_ms = (time.perf_counter() - total_started) * 1000
    report = {
        "dataset": DATASET_ID,
        "revision": "main",
        "split": "validation",
        "languages": languages,
        "records_per_language": args.records_per_language,
        "queries_saved": len(source_queries),
        "available_documents_before_cap": available_documents,
        "documents_indexed": len(source_documents),
        "chunks_indexed": len(all_chunks),
        "chunk_tokens": {
            "mean": statistics.fmean(all_token_counts),
            "min": min(all_token_counts),
            "max": max(all_token_counts),
        },
        "subset_reason": subset_reason,
        "embedding_model": encoder.model_name,
        "embedding_model_revision": encoder.model_revision,
        "device": encoder.device,
        "model_exposes_token_embeddings": encoder.exposes_token_embeddings,
        "native_mrl": True,
        "dimensions": dimensions,
        "late_chunking_parameters": asdict(late_parameters),
        "timings_ms": {
            "dataset_fetch_by_language": fetch_by_language_ms,
            "dataset_fetch_total": sum(fetch_by_language_ms.values()),
            "model_load": model_load_ms,
            "late_chunking_and_embedding": chunking_embedding_ms,
            "faiss_add_and_save": index_add_save_ms,
            "total_wall": total_ms,
        },
        "artifact_bytes": sizes,
        "index_payload_bytes": sum(sizes.values()),
        "output_dir": str(args.output_dir),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
