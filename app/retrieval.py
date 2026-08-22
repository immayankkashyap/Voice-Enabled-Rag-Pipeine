"""Measured two-stage native-MRL retrieval."""

from __future__ import annotations

import asyncio
import time
from typing import Protocol

import numpy as np

from .indexing import FaissVectorStore
from .schemas import RetrievalRequest, RetrievalResult, RetrievedChunk


class QueryEmbeddingModel(Protocol):
    embedding_dimension: int

    def embed_texts(
        self, texts: list[str], *, task: str, batch_size: int = 16
    ) -> np.ndarray:
        """Return aligned normalized embeddings."""


class TwoStageRetriever:
    """128-d HNSW candidates followed by exact 1,024-d reranking."""

    def __init__(
        self, store: FaissVectorStore, embedding_model: QueryEmbeddingModel
    ) -> None:
        if embedding_model.embedding_dimension != store.dimensions.full:
            raise ValueError("Embedding model and full FAISS dimension do not match")
        self.store = store
        self.embedding_model = embedding_model

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Return scored chunks and separately measured stage timings."""

        total_started = time.perf_counter()
        embedding_started = time.perf_counter()
        matrix = await asyncio.to_thread(
            self.embedding_model.embed_texts,
            [request.query],
            task="retrieval.query",
            batch_size=1,
        )
        query_embedding_ms = (time.perf_counter() - embedding_started) * 1000
        query_embedding = np.asarray(matrix, dtype=np.float32)
        if query_embedding.shape != (1, self.store.dimensions.full):
            raise RuntimeError("Query encoder returned an unexpected shape")

        mrl_started = time.perf_counter()
        candidate_ids, candidate_scores = self.store.search_mrl(
            query_embedding[0], request.candidate_k
        )
        mrl_search_ms = (time.perf_counter() - mrl_started) * 1000

        rerank_started = time.perf_counter()
        final_ids, full_scores = self.store.rerank_full(
            query_embedding[0], candidate_ids, request.final_k
        )
        full_rerank_ms = (time.perf_counter() - rerank_started) * 1000
        mrl_by_id = {
            int(row_id): float(score)
            for row_id, score in zip(candidate_ids, candidate_scores, strict=True)
        }
        chunks = [
            RetrievedChunk(
                chunk=self.store.chunks[int(row_id)],
                mrl_score=mrl_by_id[int(row_id)],
                full_score=float(full_score),
                rank=rank,
            )
            for rank, (row_id, full_score) in enumerate(
                zip(final_ids, full_scores, strict=True), start=1
            )
        ]
        return RetrievalResult(
            chunks=chunks,
            query_embedding_ms=query_embedding_ms,
            mrl_search_ms=mrl_search_ms,
            full_rerank_ms=full_rerank_ms,
            total_ms=(time.perf_counter() - total_started) * 1000,
        )
