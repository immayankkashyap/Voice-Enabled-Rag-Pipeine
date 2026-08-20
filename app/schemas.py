from pydantic import BaseModel, Field
from typing import List, Optional

class RAGRequest(BaseModel):
    query: str = Field(..., description="The user's query text (transcribed from audio or direct text)")
    
class RAGResponse(BaseModel):
    query: str = Field(..., description="The original transcribed query")
    answer: str = Field(..., description="The generated answer")
    retrieved_chunks: List[str] = Field(default_factory=list, description="Chunks retrieved from the vector store")
    is_grounded: bool = Field(..., description="Whether the answer passed the groundedness check")
    latencies: dict = Field(default_factory=dict, description="Latency (ms) for each pipeline stage")

class Chunk(BaseModel):
    id: str
    text: str
    metadata: dict = Field(default_factory=dict)
