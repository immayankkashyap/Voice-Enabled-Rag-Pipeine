"""Chunking strategy boundaries.

Parameter defaults are deliberately absent.  They will be chosen only after the
MSMARCO-XI profile is reviewed with the selected embedding tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import Chunk


@dataclass(frozen=True, slots=True)
class LateChunkingParameters:
    document_window_tokens: int
    target_chunk_tokens: int
    boundary_overlap_tokens: int

    def __post_init__(self) -> None:
        if self.document_window_tokens <= 0 or self.target_chunk_tokens <= 0:
            raise ValueError("Token limits must be positive")
        if not 0 <= self.boundary_overlap_tokens < self.target_chunk_tokens:
            raise ValueError("Overlap must be non-negative and smaller than a chunk")
        if self.target_chunk_tokens > self.document_window_tokens:
            raise ValueError("A target chunk cannot exceed its document window")


@dataclass(frozen=True, slots=True)
class NaiveChunkingParameters:
    chunk_characters: int
    overlap_characters: int

    def __post_init__(self) -> None:
        if self.chunk_characters <= 0:
            raise ValueError("Chunk size must be positive")
        if not 0 <= self.overlap_characters < self.chunk_characters:
            raise ValueError("Overlap must be non-negative and smaller than a chunk")


def late_chunking(
    *,
    text: str,
    document_id: str,
    parameters: LateChunkingParameters,
) -> list[Chunk]:
    """Create token-span chunks after document-wide contextual encoding."""

    raise NotImplementedError(
        "Late Chunking awaits dataset/tokenizer profiling and encoder selection"
    )


def naive_recursive_chunking(
    *,
    text: str,
    document_id: str,
    parameters: NaiveChunkingParameters,
) -> list[Chunk]:
    """Create the explicitly benchmarked recursive-character baseline."""

    raise NotImplementedError(
        "The naive baseline awaits parameters justified by the dataset profile"
    )
