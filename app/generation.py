import os
from groq import AsyncGroq
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
from .schemas import Chunk

# Initialize Groq client
# client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def generate_answer(query: str, context_chunks: List[Chunk]) -> str:
    """
    Route through Groq (LPU-hosted Llama 3) for near-instant time-to-first-token.
    """
    if not context_chunks:
        return "No relevant context found."
        
    context_text = "\n\n".join([f"Context:\n{c.text}" for c in context_chunks])
    
    prompt = f"""You are a helpful and concise assistant. Answer the user's question based strictly on the provided context. If the answer cannot be found in the context, explicitly state that you cannot answer it based on the available information. Do not use outside knowledge.

Context Information:
{context_text}

User Query: {query}
Answer:"""

    # Mock call for now
    # response = await client.chat.completions.create(
    #     messages=[{"role": "user", "content": prompt}],
    #     model="llama3-8b-8192",
    #     temperature=0.1,
    #     max_tokens=256
    # )
    # return response.choices[0].message.content
    
    return f"This is a generated answer based on {len(context_chunks)} retrieved chunks."
