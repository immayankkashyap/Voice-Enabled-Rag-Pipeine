from typing import List
from .schemas import Chunk

async def check_relevance(query: str, chunks: List[Chunk]) -> List[Chunk]:
    """
    CRAG-style Correct/Incorrect/Ambiguous classification.
    Filters out or flags low-relevance retrievals before calling the LLM.
    Fast/deterministic method preferred (e.g., cross-encoder or similarity threshold).
    """
    relevant_chunks = []
    # Mock implementation: assume all are relevant for the scaffold
    for c in chunks:
        # Placeholder for relevance score calculation
        score = 0.8
        if score > 0.5:
            relevant_chunks.append(c)
            
    return relevant_chunks

async def check_groundedness(answer: str, context_chunks: List[Chunk]) -> bool:
    """
    Lightweight groundedness check.
    Ensures the answer's content traces back to retrieved context.
    """
    # Mock implementation
    # A real implementation might use an NLI model, or strict keyword overlap.
    if "cannot answer" in answer.lower() or not context_chunks:
        return False
        
    return True
