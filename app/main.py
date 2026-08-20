from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
import time
import os
import json
from dotenv import load_dotenv
from .schemas import RAGRequest, RAGResponse
from .stt import process_audio_stream
from .retrieval import search
from .generation import generate_answer
from .guardrails import check_relevance, check_groundedness

load_dotenv()

app = FastAPI(title="Voice-Enabled RAG Pipeline")

@app.post("/rag", response_model=RAGResponse)
async def process_rag_query(request: RAGRequest):
    latencies = {}
    
    # 1. Retrieval
    start_time = time.time()
    top_chunks = await search(request.query)
    latencies['retrieval'] = round((time.time() - start_time) * 1000, 2)
    
    # 2. Pre-generation Guardrail (Relevance)
    start_time = time.time()
    relevant_chunks = await check_relevance(request.query, top_chunks)
    latencies['relevance_check'] = round((time.time() - start_time) * 1000, 2)
    
    if not relevant_chunks:
        return RAGResponse(
            query=request.query,
            answer="I could not find relevant information to answer your query.",
            retrieved_chunks=[c.text for c in top_chunks],
            is_grounded=False,
            latencies=latencies
        )
    
    # 3. Generation (Groq)
    start_time = time.time()
    answer = await generate_answer(request.query, relevant_chunks)
    latencies['generation'] = round((time.time() - start_time) * 1000, 2)
    
    # 4. Post-generation Guardrail (Groundedness)
    start_time = time.time()
    is_grounded = await check_groundedness(answer, relevant_chunks)
    latencies['groundedness_check'] = round((time.time() - start_time) * 1000, 2)
    
    if not is_grounded:
        answer = "I found some information, but I cannot confidently ground my answer in the retrieved context."
        
    return RAGResponse(
        query=request.query,
        answer=answer,
        retrieved_chunks=[c.text for c in relevant_chunks],
        is_grounded=is_grounded,
        latencies=latencies
    )

@app.websocket("/ws/voice-rag")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # 1. STT Stream Processing (Sarvam)
        query = await process_audio_stream(websocket)
        
        # 2. Process query via standard RAG pipeline
        request = RAGRequest(query=query)
        response = await process_rag_query(request)
        
        await websocket.send_text(response.model_dump_json())
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_text(json.dumps({"error": str(e)}))

@app.get("/health")
def health_check():
    return {"status": "ok"}
