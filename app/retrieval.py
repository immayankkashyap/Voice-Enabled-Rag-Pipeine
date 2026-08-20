import faiss
import numpy as np
from typing import List
from .schemas import Chunk
from .indexing import vector_store

async def search(query: str, top_k_mrl: int = 50, top_k_final: int = 5) -> List[Chunk]:
    """
    Two-stage search:
    1. MRL-truncated low-dimensional first pass
    2. Full-dimension rerank on top candidates
    """
    if vector_store.mrl_index.ntotal == 0:
        return []

    # 1. Embed query (mocked here, replace with actual embedding call)
    query_embedding = np.random.randn(1, vector_store.dim).astype(np.float32)
    
    query_mrl = query_embedding[:, :vector_store.mrl_dim].copy()
    faiss.normalize_L2(query_mrl)
    
    # Stage 1: Fast MRL search
    mrl_distances, mrl_indices = vector_store.mrl_index.search(query_mrl, top_k_mrl)
    
    if mrl_indices[0][0] == -1:
        return []
        
    candidate_indices = mrl_indices[0]
    candidate_indices = candidate_indices[candidate_indices != -1]
    
    # Stage 2: Rerank top candidates with full dimensions
    # In FAISS, we could just extract the full embeddings for these candidates
    # Or keep a dictionary. We will use the full_index directly or re-compute dot products.
    # For a real implementation, we extract the stored full embeddings and do a dot product.
    
    faiss.normalize_L2(query_embedding)
    
    # Reconstructing full embeddings is supported in IndexFlat
    candidate_embeddings = np.vstack([vector_store.full_index.reconstruct(int(idx)) for idx in candidate_indices])
    
    # Compute full similarities
    similarities = np.dot(candidate_embeddings, query_embedding[0])
    
    # Sort by full similarity
    best_relative_indices = np.argsort(similarities)[::-1][:top_k_final]
    best_absolute_indices = [candidate_indices[i] for i in best_relative_indices]
    
    retrieved = [vector_store.chunks[i] for i in best_absolute_indices]
    return retrieved
