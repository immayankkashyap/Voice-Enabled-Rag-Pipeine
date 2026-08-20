also 

# Voice-Enabled RAG Pipeline

This is a voice-enabled Retrieval-Augmented Generation (RAG) pipeline built for the HH Goa 2026 Hackathon (Task 2). It consumes user spoken questions, transcribes them using streaming Sarvam STT, performs low-latency two-stage vector retrieval (with Matryoshka Representation Learning and Late Chunking), runs relevance/groundedness guardrails, and generates grounded answers using Groq-hosted Llama 3.

## Goals & Targets

- **End-to-End Latency:** < 200ms (P50/P70/P100 metrics profiled via benchmark).
- **Retrieval Latency:** < 50ms (achieved via Matryoshka first pass and FAISS in-memory index).
- **Speech-to-Text (STT):** Sarvam real-time WebSocket connection.
- **Generation:** Groq LPU (Llama 3 8B) for high-speed generation.
- **Guardrails:** Pre-generation CRAG relevance filter + post-generation groundedness check. Refuses off-topic/ungrounded queries.
- **Chunking Strategy:** Late Chunking (context-pooled embeddings) compared against a naive fixed-size character-splitting baseline.

## Project Structure

- `app/main.py`: FastAPI application, HTTP/WebSocket endpoints.
- `app/stt.py`: Sarvam streaming STT WebSocket wrapper.
- `app/chunking.py`: Late chunking implementation and naive baseline chunker.
- `app/indexing.py`: In-memory FAISS indices (MRL-dim + Full-dim) and chunk management.
- `app/retrieval.py`: Two-stage search implementation (MRL truncated candidates -> full dimension rerank).
- `app/generation.py`: Groq/Llama-3 generation wrapper.
- `app/guardrails.py`: Relevance filters and groundedness check algorithms.
- `app/schemas.py`: Pydantic request/response models.
- `scripts/inspect_dataset.py`: Profile passage lengths and language mix of `ai4bharat/MSMARCO-XI`.
- `scripts/build_index.py`: Offline script to chunk, embed, and index dataset texts.
- `scripts/benchmark.py`: Runs query latency benchmark profiling P50/P70/P100 values.

## Setup Instructions

1. **Clone & Virtualenv Setup**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment Variables**
   Create a `.env` file using `.env.example` as a template:

   ```bash
   cp .env.example .env
   ```
   Add your `SARVAM_API_KEY` and `GROQ_API_KEY`.
4. **Verify Components**

   - Profile the dataset:
     ```bash
     python scripts/inspect_dataset.py
     ```
   - Build index and verify retrieval + late chunking:
     ```bash
     python scripts/build_index.py
     ```
    - Run latency and quality benchmark:
      ```bash
      python scripts/benchmark.py
      ```

## Running the Application

### 1. Start the FastAPI Server
To start the server locally with auto-reload:
```bash
./venv/bin/python -m uvicorn app.main:app --reload --port 8000
```
The server will start at `http://localhost:8000`. You can access the auto-generated API docs at `http://localhost:8000/docs`.

### 2. Test the RAG REST Endpoint
You can send a POST request to the `/rag` endpoint with a JSON query:
```bash
curl -X POST "http://localhost:8000/rag" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the capital of India?"}'
```

### 3. Test the Voice-RAG WebSocket Endpoint
To stream audio and receive the transcribed query and RAG answer, connect to:
`ws://localhost:8000/ws/voice-rag`

You can use a Python script or a tool like `wscat` to interact with it. The client streams raw binary audio data, and the server returns a structured response after transcription and retrieval:
```json
{
  "query": "transcribed query",
  "answer": "RAG-generated answer",
  "retrieved_chunks": ["chunk 1", "chunk 2"],
  "is_grounded": true,
  "latencies": {
    "retrieval": 12.5,
    "relevance_check": 5.2,
    "generation": 120.4,
    "groundedness_check": 8.1
  }
}
```
