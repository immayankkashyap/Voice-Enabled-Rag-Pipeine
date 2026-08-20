"""FastAPI application scaffold.

Pipeline endpoints fail explicitly until the real services are wired.  Returning
fabricated transcripts, answers, retrievals, or latency measurements is forbidden.
"""

from __future__ import annotations

from fastapi import FastAPI, WebSocket, status
from fastapi.responses import JSONResponse, Response

from .schemas import ErrorResponse, HealthResponse, RAGRequest, RAGResponse

app = FastAPI(
    title="Voice-Enabled RAG Pipeline",
    version="0.1.0",
    description="Sarvam → Late Chunking/FAISS → Groq with groundedness guardrails",
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok", implementation_phase="dataset_profiling")


@app.post(
    "/rag",
    response_model=RAGResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse}},
)
async def process_rag_query(request: RAGRequest) -> Response:
    """Run the typed text RAG pipeline after its real services are configured."""

    error = ErrorResponse(
        error_code="pipeline_not_implemented",
        message=(
            "The scaffold is healthy, but retrieval, guardrails, and generation "
            "have not been implemented yet."
        ),
        retryable=False,
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=error.model_dump(mode="json"),
    )


@app.websocket("/ws/voice-rag")
async def voice_rag_socket(websocket: WebSocket) -> None:
    """Accept voice frames once the real Sarvam and RAG services are wired."""

    await websocket.accept()
    error = ErrorResponse(
        error_code="voice_pipeline_not_implemented",
        message="The Sarvam streaming voice pipeline is not implemented yet.",
        retryable=False,
    )
    await websocket.send_json(error.model_dump(mode="json"))
    await websocket.close(code=1013, reason="Voice pipeline is not ready")
