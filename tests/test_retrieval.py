#!/usr/bin/env python3
"""Benchmark real two-stage retrieval on saved MSMARCO-XI queries."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chunking import JinaEmbeddingModel
from app.indexing import FaissVectorStore
from app.retrieval import TwoStageRetriever
from app.schemas import RetrievalRequest, RetrievalResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark native-MRL candidate search and full reranking."
    )
    parser.add_argument("--index-dir", type=Path, default=Path("data/faiss_index"))
    parser.add_argument("--queries", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--final-k", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("data/retrieval_benchmark.json")
    )
    return parser


def _balanced_queries(values: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for value in values:
        grouped[str(value["language"])].append(value)
    selected: list[dict[str, Any]] = []
    position = 0
    while len(selected) < count:
        added = False
        for language in sorted(grouped):
            if position < len(grouped[language]):
                selected.append(grouped[language][position])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        position += 1
    if len(selected) < count:
        raise ValueError(
            f"Only {len(selected)} queries are available; requested {count}"
        )
    return selected


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": statistics.fmean(values),
        "p50": float(np.percentile(array, 50)),
        "p70": float(np.percentile(array, 70)),
        "p100": float(np.max(array)),
    }


def _selected_hit(query: dict[str, Any], result: RetrievalResult) -> bool:
    return any(
        item.chunk.metadata.get("language") == query["language"]
        and str(item.chunk.metadata.get("query_id")) == str(query["query_id"])
        and item.chunk.metadata.get("is_selected") is True
        for item in result.chunks
    )


async def run(args: argparse.Namespace) -> int:
    if args.queries < 20 or args.queries > 30:
        raise ValueError("Use 20-30 measured queries for this benchmark")
    if args.warmup < 0:
        raise ValueError("warmup cannot be negative")
    load_dotenv()
    raw_queries = json.loads(
        (args.index_dir / "source_queries.json").read_text(encoding="utf-8")
    )
    queries = _balanced_queries(raw_queries, args.queries)

    # Initialize Torch before loading FAISS; the reverse native-library order
    # is unsafe in this macOS environment and was caught during the real build.
    encoder = JinaEmbeddingModel(device=args.device)
    store = FaissVectorStore.load(args.index_dir)
    retriever = TwoStageRetriever(store, encoder)

    for index in range(args.warmup):
        query = queries[index % len(queries)]
        await retriever.retrieve(
            RetrievalRequest(
                query=query["query"],
                candidate_k=args.candidate_k,
                final_k=args.final_k,
            )
        )

    results: list[tuple[dict[str, Any], RetrievalResult]] = []
    for index, query in enumerate(queries, start=1):
        result = await retriever.retrieve(
            RetrievalRequest(
                query=query["query"],
                candidate_k=args.candidate_k,
                final_k=args.final_k,
            )
        )
        results.append((query, result))
        if index % 5 == 0 or index == len(queries):
            print(f"Measured {index}/{len(queries)} queries...", flush=True)

    fields = {
        "total": [result.total_ms for _, result in results],
        "query_embedding": [result.query_embedding_ms for _, result in results],
        "stage_1_mrl_search": [result.mrl_search_ms for _, result in results],
        "stage_2_full_rerank": [result.full_rerank_ms for _, result in results],
    }
    examples = []
    for query, result in results[:5]:
        examples.append(
            {
                "query_id": query["id"],
                "language": query["language"],
                "query": query["query"],
                "english_query": query["english_query"],
                "top_3": [
                    {
                        "rank": item.rank,
                        "chunk_id": item.chunk.id,
                        "mrl_score": item.mrl_score,
                        "full_score": item.full_score,
                        "is_selected_fixture_passage": item.chunk.metadata.get(
                            "is_selected"
                        ),
                        "text": item.chunk.text[:400],
                    }
                    for item in result.chunks[:3]
                ],
            }
        )
    hit_count = sum(_selected_hit(query, result) for query, result in results)
    report = {
        "index_dir": str(args.index_dir),
        "embedding_model": encoder.model_name,
        "device": encoder.device,
        "queries": len(results),
        "languages": sorted({query["language"] for query, _ in results}),
        "warmup_queries_excluded": args.warmup,
        "candidate_k": args.candidate_k,
        "final_k": args.final_k,
        "latency_ms": {name: _distribution(values) for name, values in fields.items()},
        "selected_passage_hit_at_3": {
            "hits": hit_count,
            "queries": len(results),
            "rate": hit_count / len(results),
        },
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
