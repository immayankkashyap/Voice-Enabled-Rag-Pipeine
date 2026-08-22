from __future__ import annotations

import asyncio
import unittest

import numpy as np

from app.indexing import FaissVectorStore, IndexDimensions
from app.retrieval import TwoStageRetriever
from app.schemas import Chunk, ChunkStrategy, RetrievalRequest


class _EmbeddingModel:
    embedding_dimension = 4

    def embed_texts(
        self, texts: list[str], *, task: str, batch_size: int = 16
    ) -> np.ndarray:
        if texts != ["fixture query"] or task != "retrieval.query" or batch_size != 1:
            raise AssertionError("Retriever used an unexpected embedding call")
        return np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)


class TwoStageRetrieverTests(unittest.TestCase):
    def test_retrieves_and_reports_each_stage(self) -> None:
        store = FaissVectorStore(IndexDimensions(full=4, mrl=2), hnsw_m=4)
        chunks = [
            Chunk(
                id=f"chunk-{index}",
                document_id=f"doc-{index}",
                text=f"Fixture {index}",
                strategy=ChunkStrategy.LATE,
            )
            for index in range(3)
        ]
        store.add(
            chunks,
            np.asarray(
                [[1.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.4, 0.0], [0, 1, 0, 1]],
                dtype=np.float32,
            ),
        )
        result = asyncio.run(
            TwoStageRetriever(store, _EmbeddingModel()).retrieve(
                RetrievalRequest(query="fixture query", candidate_k=3, final_k=2)
            )
        )
        self.assertEqual(
            [item.chunk.id for item in result.chunks], ["chunk-0", "chunk-1"]
        )
        self.assertEqual([item.rank for item in result.chunks], [1, 2])
        self.assertGreaterEqual(result.query_embedding_ms, 0)
        self.assertGreaterEqual(result.mrl_search_ms, 0)
        self.assertGreaterEqual(result.full_rerank_ms, 0)
        self.assertGreaterEqual(result.total_ms, result.mrl_search_ms)


if __name__ == "__main__":
    unittest.main()
