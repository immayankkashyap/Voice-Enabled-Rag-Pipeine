"""Two-stage MRL retrieval service boundary."""

from __future__ import annotations

from .indexing import FaissVectorStore
from .schemas import RetrievalRequest, RetrievalResult


class TwoStageRetriever:
    """MRL-truncated candidate search followed by full-vector reranking."""

    def __init__(self, store: FaissVectorStore) -> None:
        self.store = store

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Return scored chunks plus measured embedding/search/rerank timings."""

        raise NotImplementedError(
            "Real embeddings and two-stage FAISS retrieval are not implemented yet"
        )
