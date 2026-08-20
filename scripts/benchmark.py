#!/usr/bin/env python3
"""Real-query latency and retrieval-quality benchmark entry point.

Synthetic queries, random embeddings, and mocked LLM responses must never be
written as submission evidence, so this scaffold exits until the real pipeline
and frozen multilingual query set exist.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Report P50/P70/P100 stage and end-to-end latency without hiding outliers."
    )


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    print(
        "Benchmarking is intentionally unavailable until the real pipeline and "
        "versioned query set are ready.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
