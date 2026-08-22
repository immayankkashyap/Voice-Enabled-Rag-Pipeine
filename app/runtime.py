"""Warm, reusable retrieval and budgeted-generation runtime services."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import JinaEmbeddingModel
from .generation import (
    GroqAnswerGenerator,
    GroqGenerationSettings,
    HybridAnswerGenerator,
)
from .indexing import FaissVectorStore
from .pipeline import FastRAGPipeline
from .retrieval import TwoStageRetriever
from .schemas import RetrievalRequest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RuntimeLoadError(RuntimeError):
    """The local index/model bundle could not be made ready."""


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    index_dir: Path = _PROJECT_ROOT / "data/faiss_index"
    device: str | None = None
    cpu_threads: int = 2
    preload: bool = True
    latency_target_ms: float = 200.0
    groq_api_key: str = field(default="", repr=False)
    generation_model: str = "qwen/qwen3.6-27b"
    generation_max_output_tokens: int = 96
    generation_max_attempts: int = 2
    generation_min_fallback_budget_ms: float = 350.0

    def __post_init__(self) -> None:
        if self.latency_target_ms <= 0:
            raise ValueError("latency_target_ms must be positive")
        if not 1 <= self.cpu_threads <= 64:
            raise ValueError("cpu_threads must be between 1 and 64")
        if not self.generation_model:
            raise ValueError("generation_model cannot be empty")
        if self.generation_max_output_tokens <= 0:
            raise ValueError("generation_max_output_tokens must be positive")
        if self.generation_max_attempts <= 0:
            raise ValueError("generation_max_attempts must be positive")
        if self.generation_min_fallback_budget_ms < 0:
            raise ValueError("generation_min_fallback_budget_ms must be non-negative")

    @classmethod
    def from_environment(cls) -> RuntimeSettings:
        preload = os.getenv("RAG_PRELOAD", "true").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        device = os.getenv("RAG_DEVICE", "").strip() or None
        configured_index = Path(os.getenv("RAG_INDEX_DIR", "data/faiss_index"))
        if not configured_index.is_absolute():
            configured_index = _PROJECT_ROOT / configured_index
        return cls(
            index_dir=configured_index,
            device=device,
            cpu_threads=int(os.getenv("RAG_CPU_THREADS", "2")),
            preload=preload,
            latency_target_ms=float(os.getenv("RAG_LATENCY_TARGET_MS", "200")),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            generation_model=os.getenv(
                "GROQ_MODEL",
                "qwen/qwen3.6-27b",
            ),
            generation_max_output_tokens=int(
                os.getenv("GROQ_MAX_OUTPUT_TOKENS", "96")
            ),
            generation_max_attempts=int(os.getenv("GROQ_MAX_ATTEMPTS", "2")),
            generation_min_fallback_budget_ms=float(
                os.getenv("GROQ_MIN_FALLBACK_BUDGET_MS", "350")
            ),
        )


@dataclass(slots=True)
class RuntimeServices:
    pipeline: FastRAGPipeline
    vector_count: int
    supported_languages: tuple[str, ...]
    embedding_model: str
    device: str
    load_ms: float
    warmup_ms: float
    answer_mode: str

    async def aclose(self) -> None:
        generator = self.pipeline.generator
        close = getattr(generator, "aclose", None)
        if close is not None:
            await close()


def _warmup_query(index_dir: Path) -> str:
    path = index_dir / "source_queries.json"
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
        query = str(values[0]["query"]).strip()
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        query = "What is retrieval augmented generation?"
    return query


def _load_encoder(device: str | None, cpu_threads: int) -> JinaEmbeddingModel:
    # On the demo CPU, two intra-op workers gave a lower warm-query P100 than
    # Torch's wider default.  This is configurable and is set before model load;
    # accelerator-backed deployments are unaffected by the CPU pool size.
    import torch

    torch.set_num_threads(cpu_threads)
    return JinaEmbeddingModel(device=device)


def _validate_encoder_provenance(
    store: FaissVectorStore, encoder: JinaEmbeddingModel
) -> None:
    expected_model = str(store.provenance.get("embedding_model") or "")
    expected_revision = str(store.provenance.get("embedding_model_revision") or "")
    if expected_model != encoder.model_name:
        raise RuntimeLoadError("The index embedding model does not match the runtime")
    if expected_revision != encoder.model_revision:
        raise RuntimeLoadError(
            "The index embedding revision does not match the pinned runtime"
        )


async def load_runtime(settings: RuntimeSettings) -> RuntimeServices:
    """Load Torch before FAISS, then warm the exact online retrieval path."""

    started = time.perf_counter()
    if not (settings.index_dir / "manifest.json").is_file():
        raise RuntimeLoadError(
            f"FAISS bundle is missing at {settings.index_dir}; run build_index.py"
        )

    # This ordering avoids the reproducible Torch/FAISS native-runtime collision
    # seen on macOS. Both objects remain process-wide and never reload per query.
    encoder = await asyncio.to_thread(
        _load_encoder,
        settings.device,
        settings.cpu_threads,
    )
    store = await asyncio.to_thread(FaissVectorStore.load, settings.index_dir)
    _validate_encoder_provenance(store, encoder)
    retriever = TwoStageRetriever(store, encoder)
    if not settings.groq_api_key:
        raise RuntimeLoadError("GROQ_API_KEY is required for the generation stage")
    groq_generator = GroqAnswerGenerator(
        GroqGenerationSettings(
            api_key=settings.groq_api_key,
            model=settings.generation_model,
            max_output_tokens=settings.generation_max_output_tokens,
            max_attempts=settings.generation_max_attempts,
            reasoning_effort=(
                "low" if settings.generation_model.startswith("openai/gpt-oss-") else None
            ),
        )
    )
    generator = HybridAnswerGenerator(
        groq_generator,
        min_fallback_budget_ms=settings.generation_min_fallback_budget_ms,
    )
    pipeline = FastRAGPipeline(
        retriever=retriever,
        generator=generator,
        latency_target_ms=settings.latency_target_ms,
    )
    load_ms = (time.perf_counter() - started) * 1000

    warm_started = time.perf_counter()
    await retriever.retrieve(
        RetrievalRequest(
            query=_warmup_query(settings.index_dir),
            candidate_k=min(50, max(1, len(store))),
            final_k=min(5, max(1, len(store))),
        )
    )
    warmup_ms = (time.perf_counter() - warm_started) * 1000
    return RuntimeServices(
        pipeline=pipeline,
        vector_count=len(store),
        supported_languages=tuple(
            sorted(
                {
                    str(chunk.metadata.get("language") or "").strip().lower()
                    for chunk in store.chunks
                    if str(chunk.metadata.get("language") or "").strip()
                }
            )
        ),
        embedding_model=encoder.model_name,
        device=encoder.device,
        load_ms=load_ms,
        warmup_ms=warmup_ms,
        answer_mode=generator.answer_mode,
    )
