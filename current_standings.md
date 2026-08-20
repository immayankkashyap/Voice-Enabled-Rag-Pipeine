# Project Current Standings - Voice-Enabled RAG Pipeline

This document outlines the current state of implementation, what is completed, and what tasks remain. Use this to quickly resume development in the next session.

---

## 1. What is Completed

- **FastAPI Harness (`app/main.py` & `app/schemas.py`):** 
  - FastAPI scaffolding is fully set up.
  - Endpoints for RAG REST (`POST /rag`), Voice RAG WebSocket (`WS /ws/voice-rag`), and Health Check (`GET /health`) are ready.
  - Request/response schemas (`RAGRequest`, `RAGResponse`, `Chunk`) are defined using Pydantic.
- **Sarvam Streaming STT (`app/stt.py`):** 
  - Complete implementation of WebSocket audio streaming to Sarvam's realtime API using `saaras:v3-realtime` and streaming audio from the client.
  - Implemented tenacity-based exponential backoff retries.
- **Python Environment:**
  - `requirements.txt` has been updated to include `torch` and `sentence-transformers`.
  - All requirements are successfully installed in the virtual environment `./venv/bin/`.
- **Project Documentation (`README.md`):**
  - Updated with detailed project context, goals, setup instructions, and manual run commands.

---

## 2. What Still Needs to be Implemented

### A. Late Chunking Chunker (`app/chunking.py`)
- **Current Status:** Stub/Mock implementation.
- **Remaining Task:** Implement a real Late Chunking strategy. Load a local multilingual model (`intfloat/multilingual-e5-small`) using `sentence-transformers`/`transformers`. Pass the entire document to extract token-level contextualized embeddings, then pool (average) these embeddings within sentence boundaries.

### B. FAISS Indexing & Two-Stage MRL Retrieval (`app/indexing.py` & `app/retrieval.py`)
- **Current Status:** Uses mock random embeddings.
- **Remaining Task:**
  - Generate real embeddings using the embedding model.
  - Implement Matryoshka Representation Learning (MRL) truncation (e.g., slice first 128 dimensions from the 384-dimensional output).
  - Setup two FAISS indices: a 128-dimensional index (`mrl_index`) and a full 384-dimensional index (`full_index`).
  - Implement two-stage retrieval: 1st pass using `mrl_index` to find top 50 candidates; 2nd pass extracting full embeddings for candidates and reranking via full cosine similarity.

### C. Groq Answer Generation (`app/generation.py`)
- **Current Status:** Returns mock text.
- **Remaining Task:** Instantiate `AsyncGroq` client, route through LPU-hosted `llama3-8b-8192`, and enforce strict system prompt constraints (no outside knowledge, refuse to answer if unsupported).

### D. Guardrails (`app/guardrails.py`)
- **Current Status:** Placeholders returning hardcoded values.
- **Remaining Task:**
  - **Relevance Guardrail (CRAG):** Check similarity scores between query and retrieved chunks, filtering out low-relevance matches. Refuse if no chunks are relevant.
  - **Groundedness Guardrail:** Compare generated answer against retrieved context using lexical overlap (e.g. Rouge/N-grams) or fast embedding similarity to verify grounding before returning the answer.

### E. Offline & Benchmarking Scripts (`scripts/`)
- **`scripts/inspect_dataset.py`:** Update to download and profile Hindi validation (`validation/hinval.parquet`) or training data from HF `ai4bharat/MSMARCO-XI`.
- **`scripts/build_index.py`:** Update to process the inspected dataset, chunk it via Late Chunking, compute embeddings, and build/save the FAISS index to disk.
- **`scripts/benchmark.py`:** Run a suite of test queries to output the P50/P70/P100 latencies and verify that the end-to-end pipeline meets the <200ms latency target.
