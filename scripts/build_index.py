import os
import sys
# Add parent dir to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.chunking import late_chunking, naive_chunking
from app.indexing import vector_store
from datasets import load_dataset
import numpy as np
import time

def build_index():
    print("Building FAISS index offline...")
    start_time = time.time()
    
    # Load dataset sample
    print("Loading dataset sample...")
    # Mocking data for index build script
    texts = [
        "The capital of India is New Delhi.",
        "FAISS is an efficient similarity search library.",
        "Retrieval-Augmented Generation improves LLM accuracy."
    ]
    
    chunks = []
    for text in texts:
        # Using late chunking
        chunks.extend(late_chunking(text))
        
    print(f"Created {len(chunks)} chunks.")
    
    # Generate mock embeddings (768 dim)
    print("Generating embeddings...")
    embeddings = np.random.randn(len(chunks), vector_store.dim).astype(np.float32)
    
    # Add to vector store
    print("Indexing chunks...")
    vector_store.add_chunks(chunks, embeddings)
    
    # Optional: Save FAISS index to disk
    # faiss.write_index(vector_store.mrl_index, "data/mrl_index.faiss")
    # faiss.write_index(vector_store.full_index, "data/full_index.faiss")
    
    print(f"Index built in {time.time() - start_time:.2f}s")
    print(f"Total vectors in index: {vector_store.full_index.ntotal}")

if __name__ == "__main__":
    build_index()
