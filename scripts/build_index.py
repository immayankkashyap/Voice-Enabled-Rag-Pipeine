#!/usr/bin/env python3
"""Offline FAISS index builder entry point.

This command intentionally refuses to create synthetic vectors.  It will be
implemented after the MSMARCO-XI profile justifies chunking/model parameters.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Build the full and MRL-truncated FAISS index offline."
    )


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    print(
        "Index building is intentionally unavailable until dataset profiling and "
        "chunking/model selection are complete.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
