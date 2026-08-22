"""Validated API and service-boundary models.

External services must consume and return these structured models.  Plain
prompt-in/text-out interfaces are intentionally not part of the scaffold.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

MetadataValue = str | int | float | bool | None


class StrictModel(BaseModel):
    """Base model that rejects misspelled or unexpected fields."""

    model_config = ConfigDict(extra="forbid")


class ChunkStrategy(StrEnum):
    LATE = "late"
    NAIVE_RECURSIVE = "naive_recursive"


class RelevanceClassification(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    AMBIGUOUS = "ambiguous"


class ResponseStatus(StrEnum):
    ANSWERED = "answered"
    REFUSED = "refused"


class RefusalReason(StrEnum):
    OFF_TOPIC = "off_topic"
    NO_RELEVANT_CONTEXT = "no_relevant_context"
    AMBIGUOUS_RETRIEVAL = "ambiguous_retrieval"
    UNGROUNDED_ANSWER = "ungrounded_answer"
    UPSTREAM_FAILURE = "upstream_failure"


class Chunk(StrictModel):
    id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    strategy: ChunkStrategy
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_offsets(self) -> Self:
        if (
            self.start_char is not None
            and self.end_char is not None
            and self.end_char < self.start_char
        ):
            raise ValueError("end_char cannot precede start_char")
        return self


class RetrievedChunk(StrictModel):
    chunk: Chunk
    mrl_score: float
    full_score: float
    rank: int = Field(ge=1)


class RetrievalRequest(StrictModel):
    query: str = Field(min_length=1, max_length=2_000)
    candidate_k: int = Field(default=50, ge=1, le=1_000)
    final_k: int = Field(default=5, ge=1, le=1_000)

    @model_validator(mode="after")
    def validate_candidate_counts(self) -> Self:
        if self.final_k > self.candidate_k:
            raise ValueError("final_k cannot exceed candidate_k")
        return self


class RetrievalResult(StrictModel):
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    query_embedding_ms: float = Field(ge=0)
    mrl_search_ms: float = Field(ge=0)
    full_rerank_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


class RelevanceAssessment(StrictModel):
    classification: RelevanceClassification
    confidence: float = Field(ge=0, le=1)
    accepted_chunk_ids: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)
    latency_ms: float = Field(ge=0)


class GroundednessAssessment(StrictModel):
    is_grounded: bool
    score: float = Field(ge=0, le=1)
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)
    latency_ms: float = Field(ge=0)


class GenerationRequest(StrictModel):
    query: str = Field(min_length=1, max_length=2_000)
    context: list[RetrievedChunk] = Field(min_length=1)
    language_code: str | None = Field(default=None, min_length=2, max_length=32)


class GenerationResult(StrictModel):
    answer: str = Field(min_length=1)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    model: str = Field(min_length=1)
    finish_reason: str | None = None
    time_to_first_token_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


class TranscriptionResult(StrictModel):
    provider: str = Field(default="unknown", min_length=1)
    model: str | None = Field(default=None, min_length=1)
    transcript: str = Field(min_length=1)
    language_code: str = Field(min_length=2, max_length=32)
    is_final: bool
    partial_transcripts: list[str] = Field(default_factory=list)
    connection_ms: float = Field(ge=0)
    time_to_first_partial_ms: float | None = Field(default=None, ge=0)
    time_to_final_transcript_ms: float = Field(ge=0)
    final_after_audio_end_ms: float = Field(ge=0)
    first_audio_to_final_ms: float | None = Field(default=None, ge=0)
    audio_duration_ms: float | None = Field(default=None, ge=0)
    total_ms: float = Field(ge=0)


class StageLatencies(StrictModel):
    stt_ms: float | None = Field(default=None, ge=0)
    retrieval_ms: float | None = Field(default=None, ge=0)
    relevance_ms: float | None = Field(default=None, ge=0)
    generation_ms: float | None = Field(default=None, ge=0)
    groundedness_ms: float | None = Field(default=None, ge=0)
    output_ms: float | None = Field(default=None, ge=0)
    total_ms: float = Field(ge=0)
    target_ms: float | None = Field(default=None, gt=0)
    target_met: bool | None = None


class RAGRequest(StrictModel):
    query: str = Field(min_length=1, max_length=2_000)
    language_code: str | None = Field(default=None, min_length=2, max_length=32)
    candidate_k: int = Field(default=50, ge=1, le=1_000)
    final_k: int = Field(default=5, ge=1, le=100)

    @model_validator(mode="after")
    def validate_candidate_counts(self) -> Self:
        if self.final_k > self.candidate_k:
            raise ValueError("final_k cannot exceed candidate_k")
        return self


class RAGResponse(StrictModel):
    request_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    status: ResponseStatus
    answer: str | None = None
    refusal_reason: RefusalReason | None = None
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    relevance: RelevanceAssessment | None = None
    groundedness: GroundednessAssessment | None = None
    latencies: StageLatencies

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.status is ResponseStatus.ANSWERED:
            if not self.answer:
                raise ValueError("An answered response requires an answer")
            if self.refusal_reason is not None:
                raise ValueError("An answered response cannot include a refusal reason")
            if self.groundedness is None or not self.groundedness.is_grounded:
                raise ValueError("An answered response must pass groundedness")
        elif self.refusal_reason is None:
            raise ValueError("A refused response requires a refusal reason")
        return self


class VoicePipelineLatencies(StrictModel):
    """Voice latency anchors that cannot hide the duration of the utterance."""

    metric_definition: str = Field(min_length=1)
    first_audio_to_committed_ms: float = Field(ge=0)
    audio_eof_to_committed_ms: float = Field(ge=0)
    committed_to_answer_ms: float = Field(ge=0)
    audio_eof_to_answer_ms: float = Field(ge=0)
    first_audio_to_answer_ms: float = Field(ge=0)
    target_ms: float = Field(default=200.0, gt=0)
    target_met: bool


class VoiceRAGResponse(StrictModel):
    transcription: TranscriptionResult
    rag: RAGResponse
    latencies: VoicePipelineLatencies


class HealthResponse(StrictModel):
    status: str
    implementation_phase: str
    rag_ready: bool = False
    voice_ready: bool = False
    stt_provider: str = "elevenlabs"
    answer_mode: str = "local_extractive"
    vector_count: int = Field(default=0, ge=0)
    supported_languages: list[str] = Field(default_factory=list)
    latency_target_ms: float = Field(default=200.0, gt=0)


class ErrorResponse(StrictModel):
    error_code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool
