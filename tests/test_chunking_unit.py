from __future__ import annotations

import re
import unittest
from collections.abc import Sequence

import numpy as np

from app.chunking import (
    LateChunkingParameters,
    NaiveChunkingParameters,
    late_chunking,
    naive_recursive_chunking,
)
from app.schemas import ChunkStrategy


class _WhitespaceTokenizer:
    def __call__(self, text: str, **_: object) -> dict[str, object]:
        matches = list(re.finditer(r"\S+", text))
        return {
            "input_ids": list(range(1, len(matches) + 1)),
            "offset_mapping": [(match.start(), match.end()) for match in matches],
        }


class _ContextualEncoder:
    model_name = "test-token-contextual-encoder"
    embedding_dimension = 4
    max_tokens = 64
    exposes_token_embeddings = True
    tokenizer = _WhitespaceTokenizer()

    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def contextualize(
        self, token_ids: Sequence[int], *, task: str = "retrieval.passage"
    ) -> np.ndarray:
        self.calls.append(list(token_ids))
        document_sum = float(sum(token_ids))
        return np.asarray(
            [
                (float(token), document_sum, float(index), 1.0)
                for index, token in enumerate(token_ids)
            ],
            dtype=np.float32,
        )


class ChunkingTests(unittest.TestCase):
    def test_profile_backed_defaults(self) -> None:
        self.assertEqual(LateChunkingParameters().target_chunk_tokens, 192)
        self.assertEqual(LateChunkingParameters().boundary_overlap_tokens, 32)
        self.assertEqual(NaiveChunkingParameters().chunk_characters, 600)
        self.assertEqual(NaiveChunkingParameters().overlap_characters, 100)

    def test_late_chunking_contextualizes_before_pooling(self) -> None:
        text = "one two three four five six seven eight nine ten eleven twelve"
        encoder = _ContextualEncoder()
        result = late_chunking(
            text=text,
            document_id="doc-1",
            encoder=encoder,
            parameters=LateChunkingParameters(
                document_window_tokens=14,
                target_chunk_tokens=5,
                boundary_overlap_tokens=1,
            ),
        )

        self.assertEqual(encoder.calls, [list(range(1, 13))])
        self.assertGreater(len(result.chunks), 1)
        self.assertTrue(
            all(chunk.strategy is ChunkStrategy.LATE for chunk in result.chunks)
        )
        np.testing.assert_allclose(
            np.linalg.norm(result.embeddings, axis=1),
            np.ones(len(result.chunks)),
            atol=1e-6,
        )
        self.assertEqual(result.chunks[0].text, text[: result.chunks[0].end_char])

    def test_naive_strategy_remains_separate(self) -> None:
        text = "First sentence. Second sentence is longer. Third sentence concludes."
        chunks = naive_recursive_chunking(
            text=text,
            document_id="doc-2",
            parameters=NaiveChunkingParameters(
                chunk_characters=32,
                overlap_characters=6,
            ),
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all(chunk.strategy is ChunkStrategy.NAIVE_RECURSIVE for chunk in chunks)
        )
        self.assertTrue(all(chunk.text in text for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
