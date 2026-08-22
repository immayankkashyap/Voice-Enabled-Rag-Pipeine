"""Evidence-backed Late Chunking and recursive-character baseline.

The selected encoder is ``jinaai/jina-embeddings-v3-hf``. Its bare model
returns ``last_hidden_state`` for every token before pooling, supports 8,192
tokens through RoPE, and is natively trained for Matryoshka dimensions. Those
properties make real late pooling possible; this module never substitutes
independently embedded chunks for the late strategy.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import numpy as np

from .schemas import Chunk, ChunkStrategy, MetadataValue

JINA_MODEL_ID = "jinaai/jina-embeddings-v3-hf"
JINA_MODEL_REVISION = "d18862d9a48706220815554fac3ebb4dfa46fc28"
JINA_FULL_DIMENSION = 1024
JINA_MATRYOSHKA_DIMENSIONS = (32, 64, 128, 256, 512, 768, 1024)


@dataclass(frozen=True, slots=True)
class LateChunkingParameters:
    # The 3,500-record MSMARCO-XI profile has passage token P50/P95/P99 of
    # 96/194/267 with Jina v3. A 192-token target keeps the typical factoid
    # passage whole, splits the long tail, and uses a 32-token (~17%) overlap.
    # Every observed passage (max 3,379 tokens) fits inside Jina's 8,192-token
    # document window, preserving true document-wide context for this dataset.
    document_window_tokens: int = 8_192
    target_chunk_tokens: int = 192
    boundary_overlap_tokens: int = 32

    def __post_init__(self) -> None:
        if self.document_window_tokens <= 0 or self.target_chunk_tokens <= 0:
            raise ValueError("Token limits must be positive")
        if not 0 <= self.boundary_overlap_tokens < self.target_chunk_tokens:
            raise ValueError("Overlap must be non-negative and smaller than a chunk")
        if self.target_chunk_tokens > self.document_window_tokens - 2:
            raise ValueError("A target chunk cannot exceed its document content window")


@dataclass(frozen=True, slots=True)
class NaiveChunkingParameters:
    # 192 Jina tokens correspond to about 600 characters at the observed
    # median ratio; 100 characters approximates the same 32-token overlap.
    chunk_characters: int = 600
    overlap_characters: int = 100

    def __post_init__(self) -> None:
        if self.chunk_characters <= 0:
            raise ValueError("Chunk size must be positive")
        if not 0 <= self.overlap_characters < self.chunk_characters:
            raise ValueError("Overlap must be non-negative and smaller than a chunk")


@dataclass(frozen=True, slots=True)
class EmbeddedChunks:
    chunks: list[Chunk]
    embeddings: np.ndarray
    token_counts: list[int]

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must be a rank-2 matrix")
        if len(self.chunks) != len(self.embeddings):
            raise ValueError("chunks and embeddings must be aligned")
        if len(self.chunks) != len(self.token_counts):
            raise ValueError("chunks and token counts must be aligned")


class TokenContextualEncoder(Protocol):
    model_name: str
    embedding_dimension: int
    max_tokens: int
    tokenizer: Any

    def contextualize(
        self, token_ids: Sequence[int], *, task: str = "retrieval.passage"
    ) -> np.ndarray:
        """Return one final-layer contextual vector per supplied content token."""


class JinaEmbeddingModel:
    """Local Jina v3 encoder exposing both token states and pooled embeddings."""

    _TASK_ADAPTERS: ClassVar[dict[str, str]] = {
        "retrieval.query": "retrieval_query",
        "retrieval.passage": "retrieval_passage",
        "separation": "separation",
        "classification": "classification",
        "text-matching": "text_matching",
    }

    def __init__(
        self,
        *,
        model_name: str = JINA_MODEL_ID,
        model_revision: str = JINA_MODEL_REVISION,
        device: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency installation path
            raise RuntimeError(
                "JinaEmbeddingModel requires torch and transformers"
            ) from exc

        self.model_name = model_name
        self.model_revision = model_revision
        self.embedding_dimension = JINA_FULL_DIMENSION
        self.max_tokens = 8_192
        self.matryoshka_dimensions = JINA_MATRYOSHKA_DIMENSIONS
        self.exposes_token_embeddings = True
        self._torch = torch
        self._lock = threading.Lock()
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if self.device == "mps" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=model_revision,
            use_fast=True,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            revision=model_revision,
            dtype=dtype,
        )
        if not hasattr(self.model, "load_adapter") or not hasattr(
            self.model, "set_adapter"
        ):
            raise RuntimeError(
                f"{model_name} does not expose the required task-adapter interface"
            )
        for adapter_name in ("retrieval_passage", "retrieval_query"):
            self.model.load_adapter(
                model_name,
                adapter_name=adapter_name,
                adapter_kwargs={
                    "subfolder": adapter_name,
                    "revision": model_revision,
                },
            )
        self.model.eval()
        self.model.to(self.device)

        hidden_size = int(getattr(self.model.config, "hidden_size", 0))
        max_positions = int(getattr(self.model.config, "max_position_embeddings", 0))
        if hidden_size != self.embedding_dimension or max_positions < self.max_tokens:
            raise RuntimeError(
                f"Unexpected Jina model capabilities: hidden={hidden_size}, "
                f"max_positions={max_positions}"
            )

    def _activate(self, task: str) -> None:
        try:
            adapter_name = self._TASK_ADAPTERS[task]
        except KeyError as exc:
            raise ValueError(f"Unsupported Jina embedding task: {task}") from exc
        self.model.set_adapter(adapter_name)

    def contextualize(
        self, token_ids: Sequence[int], *, task: str = "retrieval.passage"
    ) -> np.ndarray:
        """Run the entire token window through the transformer before pooling."""

        if not token_ids:
            raise ValueError("token_ids cannot be empty")
        if len(token_ids) + 2 > self.max_tokens:
            raise ValueError(
                f"Content has {len(token_ids)} tokens; maximum is {self.max_tokens - 2}"
            )

        # Transformers 5's native XLM-R tokenizer deliberately exposes a
        # smaller public surface than PreTrainedTokenizerBase. This selected
        # checkpoint declares exactly two special tokens and tokenizes text as
        # ``<s> content </s>``; build that documented sequence explicitly.
        content_ids = list(token_ids)
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        if (
            bos_token_id is None
            or eos_token_id is None
            or self.tokenizer.num_special_tokens_to_add() != 2
        ):
            raise RuntimeError("Unexpected special-token layout for Jina v3")
        input_ids = [bos_token_id, *content_ids, eos_token_id]
        special_mask_values = [True, *([False] * len(content_ids)), True]
        prepared = {
            "input_ids": self._torch.tensor([input_ids], dtype=self._torch.long),
            "attention_mask": self._torch.ones(
                (1, len(input_ids)), dtype=self._torch.long
            ),
        }
        special_mask = self._torch.tensor(special_mask_values, dtype=self._torch.bool)
        inputs = {name: value.to(self.device) for name, value in prepared.items()}
        with self._lock, self._torch.inference_mode():
            self._activate(task)
            output = self.model(**inputs)
        states = output.last_hidden_state[0]
        content_states = states[~special_mask.to(states.device)]
        if content_states.shape != (len(token_ids), self.embedding_dimension):
            raise RuntimeError(
                "The selected model did not expose one pre-pooling vector per token"
            )
        return content_states.float().cpu().numpy()

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        task: str,
        batch_size: int = 16,
    ) -> np.ndarray:
        """Mean-pool complete texts for query encoding or the naive baseline."""

        if not texts:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        embeddings: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            if any(not text.strip() for text in batch):
                raise ValueError("Embedding text cannot be empty")
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="pt",
            )
            inputs = {name: value.to(self.device) for name, value in encoded.items()}
            with self._lock, self._torch.inference_mode():
                self._activate(task)
                output = self.model(**inputs)
            mask = (
                inputs["attention_mask"]
                .unsqueeze(-1)
                .to(output.last_hidden_state.dtype)
            )
            pooled = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(
                dim=1
            ).clamp(min=1)
            pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.float().cpu().numpy())
        return np.ascontiguousarray(np.concatenate(embeddings), dtype=np.float32)


_BOUNDARY_PATTERN = re.compile(r"(?:\n\s*\n|[.!?;:।॥]\s+|\n|,\s+)")


def _token_chunk_spans(
    text: str,
    offsets: Sequence[tuple[int, int]],
    *,
    first_token: int,
    final_token: int,
    target_tokens: int,
    overlap_tokens: int,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = first_token
    while start < final_token:
        hard_end = min(start + target_tokens, final_token)
        end = hard_end
        if hard_end < final_token:
            minimum = start + max(1, target_tokens // 2)
            char_start = offsets[minimum][0]
            char_end = offsets[hard_end - 1][1]
            boundary_chars = [
                match.end() + char_start
                for match in _BOUNDARY_PATTERN.finditer(text[char_start:char_end])
            ]
            if boundary_chars:
                boundary_char = boundary_chars[-1]
                candidates = [
                    index + 1
                    for index in range(minimum, hard_end)
                    if offsets[index][1] <= boundary_char
                ]
                if candidates:
                    end = candidates[-1]

        if end <= start:
            end = hard_end
        spans.append((start, end))
        if end == final_token:
            break
        start = max(start + 1, end - overlap_tokens)
    return spans


def _trimmed_offsets(text: str, start: int, end: int) -> tuple[int, int, str]:
    raw = text[start:end]
    left = len(raw) - len(raw.lstrip())
    right = len(raw.rstrip())
    trimmed_start = start + left
    trimmed_end = start + right
    return trimmed_start, trimmed_end, text[trimmed_start:trimmed_end]


def late_chunking(
    *,
    text: str,
    document_id: str,
    encoder: TokenContextualEncoder,
    parameters: LateChunkingParameters | None = None,
    metadata: Mapping[str, MetadataValue] | None = None,
) -> EmbeddedChunks:
    """Contextualize each full document window, then mean-pool chunk spans."""

    parameters = parameters or LateChunkingParameters()
    if not text.strip():
        raise ValueError("text cannot be empty")
    if not document_id:
        raise ValueError("document_id cannot be empty")
    if not getattr(encoder, "exposes_token_embeddings", True):
        raise ValueError("The encoder does not expose pre-pooling token embeddings")
    if parameters.document_window_tokens > encoder.max_tokens:
        raise ValueError(
            "document_window_tokens exceeds the selected encoder's context limit"
        )

    tokenized = encoder.tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    token_ids = [int(value) for value in tokenized["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in tokenized["offset_mapping"]]
    if not token_ids or len(token_ids) != len(offsets):
        raise ValueError("Tokenizer returned no aligned content tokens")

    content_window = parameters.document_window_tokens - 2
    chunks: list[Chunk] = []
    vectors: list[np.ndarray] = []
    counts: list[int] = []
    base_metadata = dict(metadata or {})

    for window_index, window_start in enumerate(
        range(0, len(token_ids), content_window)
    ):
        window_end = min(window_start + content_window, len(token_ids))
        window_states = encoder.contextualize(
            token_ids[window_start:window_end], task="retrieval.passage"
        )
        spans = _token_chunk_spans(
            text,
            offsets,
            first_token=window_start,
            final_token=window_end,
            target_tokens=parameters.target_chunk_tokens,
            overlap_tokens=parameters.boundary_overlap_tokens,
        )
        for span_start, span_end in spans:
            char_start, char_end, chunk_text = _trimmed_offsets(
                text, offsets[span_start][0], offsets[span_end - 1][1]
            )
            if not chunk_text:
                continue
            local_start = span_start - window_start
            local_end = span_end - window_start
            embedding = window_states[local_start:local_end].mean(axis=0)
            norm = float(np.linalg.norm(embedding))
            if not np.isfinite(norm) or norm == 0:
                raise RuntimeError("Late pooling produced an invalid embedding")
            embedding = np.asarray(embedding / norm, dtype=np.float32)
            chunk_index = len(chunks)
            chunk_metadata = {
                **base_metadata,
                "embedding_model": encoder.model_name,
                "embedding_model_revision": getattr(
                    encoder, "model_revision", "unversioned"
                ),
                "embedding_task": "retrieval.passage",
                "late_chunking": True,
                "token_count": span_end - span_start,
                "document_token_count": len(token_ids),
                "window_index": window_index,
            }
            chunks.append(
                Chunk(
                    id=f"{document_id}:late:{chunk_index}",
                    document_id=document_id,
                    text=chunk_text,
                    strategy=ChunkStrategy.LATE,
                    start_char=char_start,
                    end_char=char_end,
                    metadata=chunk_metadata,
                )
            )
            vectors.append(embedding)
            counts.append(span_end - span_start)

    if not chunks:
        raise RuntimeError("Late Chunking produced no non-empty chunks")
    matrix = np.ascontiguousarray(np.stack(vectors), dtype=np.float32)
    if matrix.shape[1] != encoder.embedding_dimension:
        raise RuntimeError(
            "Late Chunking embedding dimension does not match the encoder"
        )
    return EmbeddedChunks(chunks=chunks, embeddings=matrix, token_counts=counts)


def naive_recursive_chunking(
    *,
    text: str,
    document_id: str,
    parameters: NaiveChunkingParameters | None = None,
    metadata: Mapping[str, MetadataValue] | None = None,
) -> list[Chunk]:
    """Create the separately benchmarked recursive-character baseline."""

    parameters = parameters or NaiveChunkingParameters()
    if not text.strip():
        raise ValueError("text cannot be empty")
    if not document_id:
        raise ValueError("document_id cannot be empty")

    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        hard_end = min(start + parameters.chunk_characters, len(text))
        end = hard_end
        if hard_end < len(text):
            minimum = start + parameters.chunk_characters // 2
            matches = list(_BOUNDARY_PATTERN.finditer(text, minimum, hard_end))
            if matches:
                end = matches[-1].end()
        if end <= start:
            end = hard_end

        char_start, char_end, chunk_text = _trimmed_offsets(text, start, end)
        if chunk_text:
            index = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{document_id}:naive:{index}",
                    document_id=document_id,
                    text=chunk_text,
                    strategy=ChunkStrategy.NAIVE_RECURSIVE,
                    start_char=char_start,
                    end_char=char_end,
                    metadata={
                        **dict(metadata or {}),
                        "late_chunking": False,
                        "character_count": len(chunk_text),
                    },
                )
            )
        if end == len(text):
            break
        start = max(start + 1, end - parameters.overlap_characters)

    if not chunks:
        raise RuntimeError("Naive chunking produced no non-empty chunks")
    return chunks
