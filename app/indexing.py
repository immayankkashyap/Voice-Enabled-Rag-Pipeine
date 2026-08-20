"""FAISS index build/load boundary for full and MRL-truncated vectors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .schemas import Chunk


@dataclass(frozen=True, slots=True)
class IndexDimensions:
    full: int
    mrl: int

    def __post_init__(self) -> None:
        if self.full <= 0 or self.mrl <= 0:
            raise ValueError("Index dimensions must be positive")
        if self.mrl >= self.full:
            raise ValueError("MRL dimension must be smaller than full dimension")


class FaissVectorStore:
    """Own paired in-memory FAISS indices and their aligned chunk metadata."""

    def __init__(self, dimensions: IndexDimensions) -> None:
        self.dimensions = dimensions

    def add(self, chunks: list[Chunk], full_embeddings: object) -> None:
        """Normalize and add aligned full/MRL vectors to both indices."""

        raise NotImplementedError("FAISS indexing is implemented after chunk profiling")

    def save(self, output_dir: Path) -> None:
        """Persist both indices, chunks, dimensions, and build provenance."""

        raise NotImplementedError("FAISS persistence is not implemented yet")

    @classmethod
    def load(cls, input_dir: Path) -> FaissVectorStore:
        """Load and validate an offline-built index bundle."""

        raise NotImplementedError("FAISS loading is not implemented yet")
