import faiss
import numpy as np
from typing import List
from .schemas import Chunk

class VectorStore:
    def __init__(self, dim: int = 768, mrl_dim: int = 128):
        # MRL (Matryoshka Representation Learning) setup
        self.dim = dim
        self.mrl_dim = mrl_dim
        
        # Two indices: one for fast MRL search, one for full dimension reranking
        self.mrl_index = faiss.IndexFlatIP(mrl_dim)
        self.full_index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []

    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray):
        """
        embeddings should be of shape (N, dim)
        """
        assert len(chunks) == embeddings.shape[0]
        
        # MRL embeddings are typically the first `mrl_dim` dimensions
        mrl_embeddings = embeddings[:, :self.mrl_dim]
        
        # Normalize for Inner Product -> Cosine Similarity
        faiss.normalize_L2(mrl_embeddings)
        faiss.normalize_L2(embeddings)
        
        self.mrl_index.add(mrl_embeddings)
        self.full_index.add(embeddings)
        self.chunks.extend(chunks)

# Singleton vector store instance
vector_store = VectorStore()
