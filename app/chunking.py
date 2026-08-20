from typing import List
from .schemas import Chunk

def naive_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Chunk]:
    """
    Baseline fixed-size/recursive character chunking for comparison.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append(Chunk(id=f"naive_{start}", text=chunk_text))
        start += (chunk_size - overlap)
    return chunks

def late_chunking(text: str) -> List[Chunk]:
    """
    Late Chunking strategy: process larger document contexts and contextualize smaller chunks.
    This is a stub to be implemented.
    """
    # In practice, late chunking involves creating embeddings at a smaller granularity
    # but pooling/contextualizing them with the document-level embedding or attention.
    # We'll mock the extraction here.
    sentences = text.split('.')
    chunks = []
    for i, sentence in enumerate(sentences):
        if sentence.strip():
            chunks.append(Chunk(id=f"late_{i}", text=sentence.strip() + "."))
    return chunks
