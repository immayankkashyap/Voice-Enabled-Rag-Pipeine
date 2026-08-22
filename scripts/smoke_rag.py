#!/usr/bin/env python3
"""Hit the real FastAPI ``/rag`` route and print its complete typed response."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from fastapi.testclient import TestClient

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        default="उच्च पोटेशियम वाले खाद्य पदार्थों में क्या शामिल हैं?",
    )
    parser.add_argument("--language-code", default="hi")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with TestClient(app) as client:
        response = client.post(
            "/rag",
            json={"query": args.query, "language_code": args.language_code},
        )
    payload = response.json()
    print(f"HTTP {response.status_code}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if response.status_code != 200 or payload.get("status") != "answered":
        return 1
    latencies = payload.get("latencies") or {}
    required = {
        "stt_ms",
        "input_safety_ms",
        "query_embedding_ms",
        "retrieval_stage_1_ms",
        "retrieval_stage_2_ms",
        "retrieval_ms",
        "relevance_ms",
        "generation_ms",
        "groundedness_ms",
        "output_ms",
        "total_ms",
    }
    return 0 if required.issubset(latencies) and all(
        latencies[name] is not None for name in required
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
