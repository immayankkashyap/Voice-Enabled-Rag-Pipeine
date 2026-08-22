from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.indexing import FaissVectorStore, IndexDimensions
from app.schemas import Chunk, ChunkStrategy


def _chunk(index: int) -> Chunk:
    return Chunk(
        id=f"chunk-{index}",
        document_id=f"document-{index}",
        text=f"Fixture passage {index}",
        strategy=ChunkStrategy.LATE,
    )


class FaissVectorStoreTests(unittest.TestCase):
    def test_two_stage_search_and_persistence(self) -> None:
        dimensions = IndexDimensions(full=4, mrl=2)
        store = FaissVectorStore(dimensions, hnsw_m=4)
        vectors = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.8, 0.2, 0.5, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        store.add([_chunk(index) for index in range(3)], vectors)

        candidate_ids, candidate_scores = store.search_mrl(vectors[0], 3)
        reranked_ids, full_scores = store.rerank_full(
            vectors[0], candidate_ids, final_k=2
        )
        self.assertEqual(int(candidate_ids[0]), 0)
        self.assertEqual(int(reranked_ids[0]), 0)
        self.assertEqual(candidate_scores.shape, (3,))
        self.assertEqual(full_scores.shape, (2,))

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            store.save(output_dir)
            loaded = FaissVectorStore.load(output_dir)
            loaded_ids, _ = loaded.search_mrl(vectors[0], 3)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(int(loaded_ids[0]), 0)
            self.assertEqual(loaded.chunks[0].id, "chunk-0")

    def test_rejects_duplicate_ids(self) -> None:
        store = FaissVectorStore(IndexDimensions(full=4, mrl=2), hnsw_m=4)
        with self.assertRaisesRegex(ValueError, "unique"):
            store.add(
                [_chunk(0), _chunk(0)],
                np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
