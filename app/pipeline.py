"""Fail-closed, latency-budgeted RAG orchestration."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Protocol

from .generation import GroqGenerationError
from .guardrails import (
    GroundednessGuardrail,
    InputSafetyGuardrail,
    RelevanceGuardrail,
)
from .retrieval import TwoStageRetriever
from .schemas import (
    ErrorResponse,
    GenerationRequest,
    GenerationResult,
    GroundednessAssessment,
    InputSafetyAssessment,
    RAGRequest,
    RAGResponse,
    RefusalReason,
    RelevanceAssessment,
    ResponseStatus,
    RetrievalRequest,
    RetrievedChunk,
    StageLatencies,
)

_LANGUAGE_ALIASES = {
    "asm": "as",
    "ben": "bn",
    "guj": "gu",
    "hin": "hi",
    "kan": "kn",
    "mal": "ml",
    "mar": "mr",
    "nep": "ne",
    "ori": "or",
    "pan": "pa",
    "pun": "pa",
    "san": "sa",
    "tam": "ta",
    "tel": "te",
    "urd": "ur",
}

LOGGER = logging.getLogger(__name__)


class AnswerGenerator(Protocol):
    """Typed answer-generator contract for local and remote implementations."""

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Return a cited answer or a canonical refusal."""


@dataclass(slots=True)
class _StageState:
    """Mutable per-request timings, initialized for every possible stage."""

    stt_ms: float = 0.0
    input_safety_ms: float = 0.0
    query_embedding_ms: float = 0.0
    retrieval_stage_1_ms: float = 0.0
    retrieval_stage_2_ms: float = 0.0
    retrieval_ms: float = 0.0
    relevance_ms: float = 0.0
    generation_ms: float = 0.0
    groundedness_ms: float = 0.0


class FastRAGPipeline:
    """Retrieve, reject weak evidence, generate within budget, then ground."""

    def __init__(
        self,
        *,
        retriever: TwoStageRetriever,
        generator: AnswerGenerator,
        input_safety: InputSafetyGuardrail | None = None,
        relevance: RelevanceGuardrail | None = None,
        groundedness: GroundednessGuardrail | None = None,
        latency_target_ms: float = 200.0,
    ) -> None:
        if latency_target_ms <= 0:
            raise ValueError("latency_target_ms must be positive")
        self.retriever = retriever
        self.generator = generator
        self.input_safety = input_safety or InputSafetyGuardrail()
        self.relevance = relevance or RelevanceGuardrail()
        self.groundedness = groundedness or GroundednessGuardrail()
        self.latency_target_ms = latency_target_ms

    @staticmethod
    def _language_code(value: str | None) -> str | None:
        if not value:
            return None
        primary = value.strip().lower().split("-", 1)[0]
        return _LANGUAGE_ALIASES.get(primary, primary)

    def _same_language(
        self, chunks: list[RetrievedChunk], language_code: str | None
    ) -> list[RetrievedChunk]:
        requested = self._language_code(language_code)
        if requested is None:
            return chunks
        return [
            item
            for item in chunks
            if self._language_code(str(item.chunk.metadata.get("language") or ""))
            == requested
        ]

    def _response(
        self,
        *,
        request_id: str,
        request: RAGRequest,
        started: float,
        status: ResponseStatus,
        refusal_reason: RefusalReason | None,
        chunks: list[RetrievedChunk],
        input_safety: InputSafetyAssessment | None,
        relevance: RelevanceAssessment | None,
        stages: _StageState,
        answer: str | None = None,
        groundedness: GroundednessAssessment | None = None,
    ) -> RAGResponse:
        """Build and validate the response before closing the latency window.

        ``total_ms`` and ``output_ms`` are written after the measurement boundary
        because they are fields of the object being measured.  The boundary is
        immediately after the complete ``RAGResponse`` has been constructed and
        validated; only the unavoidable constant-time insertion of those two
        measurements and ``target_met`` falls outside it.
        """

        output_started = time.perf_counter()
        latencies = StageLatencies(
            stt_ms=stages.stt_ms,
            input_safety_ms=stages.input_safety_ms,
            query_embedding_ms=stages.query_embedding_ms,
            retrieval_stage_1_ms=stages.retrieval_stage_1_ms,
            retrieval_stage_2_ms=stages.retrieval_stage_2_ms,
            retrieval_ms=stages.retrieval_ms,
            relevance_ms=stages.relevance_ms,
            generation_ms=stages.generation_ms,
            groundedness_ms=stages.groundedness_ms,
            output_ms=0.0,
            total_ms=0.0,
            target_ms=self.latency_target_ms,
            target_met=False,
        )
        response = RAGResponse(
            request_id=request_id,
            query=request.query,
            status=status,
            answer=answer,
            refusal_reason=refusal_reason,
            input_safety=input_safety,
            retrieved_chunks=chunks,
            relevance=relevance,
            groundedness=groundedness,
            latencies=latencies,
        )
        completed = time.perf_counter()
        output_ms = (completed - output_started) * 1000
        total_ms = (completed - started) * 1000
        response.latencies.output_ms = output_ms
        response.latencies.total_ms = total_ms
        response.latencies.target_met = total_ms <= self.latency_target_ms
        return response

    def _stage_failure(
        self,
        *,
        request_id: str,
        started: float,
        stages: _StageState,
        stage: str,
        error_code: str,
        message: str,
        retryable: bool,
    ) -> ErrorResponse:
        """Return a sanitized typed stage error with a complete timing envelope."""

        total_ms = (time.perf_counter() - started) * 1000
        return ErrorResponse(
            error_code=error_code,
            message=message,
            retryable=retryable,
            request_id=request_id,
            stage=stage,
            latencies=StageLatencies(
                stt_ms=stages.stt_ms,
                input_safety_ms=stages.input_safety_ms,
                query_embedding_ms=stages.query_embedding_ms,
                retrieval_stage_1_ms=stages.retrieval_stage_1_ms,
                retrieval_stage_2_ms=stages.retrieval_stage_2_ms,
                retrieval_ms=stages.retrieval_ms,
                relevance_ms=stages.relevance_ms,
                generation_ms=stages.generation_ms,
                groundedness_ms=stages.groundedness_ms,
                output_ms=0.0,
                total_ms=total_ms,
                target_ms=self.latency_target_ms,
                target_met=total_ms <= self.latency_target_ms,
            ),
        )

    async def answer(self, request: RAGRequest) -> RAGResponse | ErrorResponse:
        """Return a real answer or an explicit refusal; never synthesize a fallback."""

        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        chunks: list[RetrievedChunk] = []
        input_safety: InputSafetyAssessment | None = None
        relevance: RelevanceAssessment | None = None
        stages = _StageState()

        stage_started = time.perf_counter()
        try:
            input_safety = await self.input_safety.assess(text=request.query)
            stages.input_safety_ms = input_safety.latency_ms
        except Exception:  # noqa: BLE001 - guardrail stage must fail closed
            stages.input_safety_ms = (time.perf_counter() - stage_started) * 1000
            return self._stage_failure(
                request_id=request_id,
                started=started,
                stages=stages,
                stage="input_safety",
                error_code="rag_input_guardrail_failed",
                message="The input-safety stage failed.",
                retryable=False,
            )
        if not input_safety.is_safe:
            return self._response(
                request_id=request_id,
                request=request,
                started=started,
                status=ResponseStatus.REFUSED,
                refusal_reason=RefusalReason.UNSAFE_INPUT,
                chunks=chunks,
                input_safety=input_safety,
                relevance=None,
                stages=stages,
            )

        stage_started = time.perf_counter()
        try:
            retrieval = await self.retriever.retrieve(
                RetrievalRequest(
                    query=request.query,
                    candidate_k=request.candidate_k,
                    # When a language is requested, rerank the whole MRL
                    # candidate pool before filtering. Slicing globally first
                    # lets cross-language translations crowd all same-language
                    # evidence out of a small final_k.
                    final_k=(
                        request.candidate_k
                        if self._language_code(request.language_code) is not None
                        else request.final_k
                    ),
                )
            )
        except Exception:  # noqa: BLE001 - external stage must fail closed
            stages.retrieval_ms = (time.perf_counter() - stage_started) * 1000
            return self._stage_failure(
                request_id=request_id,
                started=started,
                stages=stages,
                stage="retrieval",
                error_code="rag_retrieval_failed",
                message="The retrieval stage failed.",
                retryable=True,
            )
        stages.query_embedding_ms = retrieval.query_embedding_ms
        stages.retrieval_stage_1_ms = retrieval.mrl_search_ms
        stages.retrieval_stage_2_ms = retrieval.full_rerank_ms
        stages.retrieval_ms = retrieval.total_ms
        if not retrieval.chunks:
            return self._stage_failure(
                request_id=request_id,
                started=started,
                stages=stages,
                stage="retrieval",
                error_code="rag_retrieval_empty",
                message="The retrieval stage returned no candidates.",
                retryable=False,
            )
        chunks = self._same_language(retrieval.chunks, request.language_code)[
            : request.final_k
        ]
        stage_started = time.perf_counter()
        try:
            relevance = await self.relevance.assess(query=request.query, chunks=chunks)
            stages.relevance_ms = relevance.latency_ms
            refusal_reason = self.relevance.refusal_reason(relevance)
        except Exception:  # noqa: BLE001 - guardrail stage must fail closed
            stages.relevance_ms = (time.perf_counter() - stage_started) * 1000
            return self._stage_failure(
                request_id=request_id,
                started=started,
                stages=stages,
                stage="relevance",
                error_code="rag_relevance_guardrail_failed",
                message="The relevance guardrail failed.",
                retryable=False,
            )
        if refusal_reason is not None:
            return self._response(
                request_id=request_id,
                request=request,
                started=started,
                status=ResponseStatus.REFUSED,
                refusal_reason=refusal_reason,
                chunks=chunks,
                input_safety=input_safety,
                relevance=relevance,
                stages=stages,
            )

        accepted = set(relevance.accepted_chunk_ids)
        evidence = [item for item in chunks if item.chunk.id in accepted]
        if not evidence:
            return self._response(
                request_id=request_id,
                request=request,
                started=started,
                status=ResponseStatus.REFUSED,
                refusal_reason=RefusalReason.NO_RELEVANT_CONTEXT,
                chunks=chunks,
                input_safety=input_safety,
                relevance=relevance,
                stages=stages,
            )

        stage_started = time.perf_counter()
        try:
            generation_budget_ms = max(
                0.0,
                self.latency_target_ms - (time.perf_counter() - started) * 1000,
            )
            generation = await self.generator.generate(
                GenerationRequest(
                    query=request.query,
                    context=evidence,
                    language_code=request.language_code,
                    latency_budget_ms=generation_budget_ms,
                )
            )
            stages.generation_ms = generation.total_ms
        except GroqGenerationError as exc:
            # A syntactically complete provider response that violates the
            # citation/finish contract is not an upstream outage. Preserve the
            # measured call, refuse explicitly, and never expose the invalid
            # text to the groundedness stage or client.
            stages.generation_ms = (time.perf_counter() - stage_started) * 1000
            LOGGER.warning(
                "Answer generation returned invalid output (%s): %s",
                type(exc).__name__,
                exc,
            )
            return self._response(
                request_id=request_id,
                request=request,
                started=started,
                status=ResponseStatus.REFUSED,
                refusal_reason=RefusalReason.GENERATION_INVALID_OUTPUT,
                chunks=chunks,
                input_safety=input_safety,
                relevance=relevance,
                stages=stages,
            )
        except Exception as exc:  # noqa: BLE001 - generation stage must fail closed
            stages.generation_ms = (time.perf_counter() - stage_started) * 1000
            LOGGER.warning(
                "Answer generation failed (%s): %s",
                type(exc).__name__,
                exc,
            )
            return self._stage_failure(
                request_id=request_id,
                started=started,
                stages=stages,
                stage="generation",
                error_code="rag_generation_failed",
                message="The answer-generation provider failed.",
                retryable=True,
            )
        stage_started = time.perf_counter()
        try:
            remaining_budget_ms = max(
                0.0,
                self.latency_target_ms - (time.perf_counter() - started) * 1000,
            )
            groundedness = await self.groundedness.assess(
                answer=generation.answer,
                chunks=evidence,
                remaining_budget_ms=remaining_budget_ms,
            )
            stages.groundedness_ms = groundedness.latency_ms
            safe_refusal = self.groundedness.is_safe_refusal(generation.answer)
        except Exception:  # noqa: BLE001 - grounding stage must fail closed
            stages.groundedness_ms = (time.perf_counter() - stage_started) * 1000
            return self._stage_failure(
                request_id=request_id,
                started=started,
                stages=stages,
                stage="groundedness",
                error_code="rag_groundedness_guardrail_failed",
                message="The groundedness guardrail failed.",
                retryable=False,
            )
        if safe_refusal:
            status = ResponseStatus.REFUSED
            answer = None
            refusal_reason = (
                RefusalReason.LATENCY_BUDGET_EXHAUSTED
                if generation.finish_reason == "budget_skipped"
                else RefusalReason.NO_RELEVANT_CONTEXT
            )
        elif not groundedness.is_grounded:
            status = ResponseStatus.REFUSED
            answer = None
            refusal_reason = RefusalReason.UNGROUNDED_ANSWER
        else:
            status = ResponseStatus.ANSWERED
            answer = generation.answer
            refusal_reason = None

        return self._response(
            request_id=request_id,
            request=request,
            started=started,
            status=status,
            answer=answer,
            refusal_reason=refusal_reason,
            chunks=chunks,
            input_safety=input_safety,
            relevance=relevance,
            groundedness=groundedness,
            stages=stages,
        )
