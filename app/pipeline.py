"""Fail-closed, zero-paid-generation RAG orchestration."""

from __future__ import annotations

import time
import uuid
from typing import Protocol

from .guardrails import GroundednessGuardrail, RelevanceGuardrail
from .retrieval import TwoStageRetriever
from .schemas import (
    GenerationRequest,
    GenerationResult,
    GroundednessAssessment,
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


class LocalAnswerGenerator(Protocol):
    """The demo generator contract; implementations must not call paid APIs."""

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Return a locally produced, cited answer or a canonical refusal."""


class FastRAGPipeline:
    """Retrieve, reject weak evidence, extract locally, then verify grounding."""

    def __init__(
        self,
        *,
        retriever: TwoStageRetriever,
        generator: LocalAnswerGenerator,
        relevance: RelevanceGuardrail | None = None,
        groundedness: GroundednessGuardrail | None = None,
        latency_target_ms: float = 200.0,
    ) -> None:
        if latency_target_ms <= 0:
            raise ValueError("latency_target_ms must be positive")
        self.retriever = retriever
        self.generator = generator
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
        relevance: RelevanceAssessment | None,
        retrieval_ms: float | None,
        relevance_ms: float | None,
        generation_ms: float | None,
        groundedness_ms: float | None,
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
            retrieval_ms=retrieval_ms,
            relevance_ms=relevance_ms,
            generation_ms=generation_ms,
            groundedness_ms=groundedness_ms,
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

    def _upstream_failure(
        self,
        *,
        request_id: str,
        request: RAGRequest,
        started: float,
        chunks: list[RetrievedChunk],
        relevance: RelevanceAssessment | None,
        retrieval_ms: float | None,
        relevance_ms: float | None,
        generation_ms: float | None,
        groundedness_ms: float | None,
    ) -> RAGResponse:
        """Return a typed refusal without exposing an exception or partial answer."""

        return self._response(
            request_id=request_id,
            request=request,
            started=started,
            status=ResponseStatus.REFUSED,
            refusal_reason=RefusalReason.UPSTREAM_FAILURE,
            chunks=chunks,
            relevance=relevance,
            retrieval_ms=retrieval_ms,
            relevance_ms=relevance_ms,
            generation_ms=generation_ms,
            groundedness_ms=groundedness_ms,
        )

    async def answer(self, request: RAGRequest) -> RAGResponse:
        """Return a real answer or an explicit refusal; never synthesize a fallback."""

        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        chunks: list[RetrievedChunk] = []
        relevance = None
        retrieval_ms = None
        relevance_ms = None
        generation_ms = None
        groundedness_ms = None

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
            return self._upstream_failure(
                request_id=request_id,
                request=request,
                started=started,
                chunks=chunks,
                relevance=relevance,
                retrieval_ms=retrieval_ms,
                relevance_ms=relevance_ms,
                generation_ms=generation_ms,
                groundedness_ms=groundedness_ms,
            )
        retrieval_ms = retrieval.total_ms
        chunks = self._same_language(retrieval.chunks, request.language_code)[
            : request.final_k
        ]
        try:
            relevance = await self.relevance.assess(query=request.query, chunks=chunks)
            relevance_ms = relevance.latency_ms
            refusal_reason = self.relevance.refusal_reason(relevance)
        except Exception:  # noqa: BLE001 - guardrail stage must fail closed
            return self._upstream_failure(
                request_id=request_id,
                request=request,
                started=started,
                chunks=chunks,
                relevance=relevance,
                retrieval_ms=retrieval_ms,
                relevance_ms=relevance_ms,
                generation_ms=generation_ms,
                groundedness_ms=groundedness_ms,
            )
        if refusal_reason is not None:
            return self._response(
                request_id=request_id,
                request=request,
                started=started,
                status=ResponseStatus.REFUSED,
                refusal_reason=refusal_reason,
                chunks=chunks,
                relevance=relevance,
                retrieval_ms=retrieval_ms,
                relevance_ms=relevance_ms,
                generation_ms=None,
                groundedness_ms=None,
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
                relevance=relevance,
                retrieval_ms=retrieval_ms,
                relevance_ms=relevance_ms,
                generation_ms=None,
                groundedness_ms=None,
            )

        try:
            generation = await self.generator.generate(
                GenerationRequest(
                    query=request.query,
                    context=evidence,
                    language_code=request.language_code,
                )
            )
            generation_ms = generation.total_ms
        except Exception:  # noqa: BLE001 - generation stage must fail closed
            return self._upstream_failure(
                request_id=request_id,
                request=request,
                started=started,
                chunks=chunks,
                relevance=relevance,
                retrieval_ms=retrieval_ms,
                relevance_ms=relevance_ms,
                generation_ms=generation_ms,
                groundedness_ms=groundedness_ms,
            )
        try:
            groundedness = await self.groundedness.assess(
                answer=generation.answer,
                chunks=evidence,
            )
            groundedness_ms = groundedness.latency_ms
            safe_refusal = self.groundedness.is_safe_refusal(generation.answer)
        except Exception:  # noqa: BLE001 - grounding stage must fail closed
            return self._upstream_failure(
                request_id=request_id,
                request=request,
                started=started,
                chunks=chunks,
                relevance=relevance,
                retrieval_ms=retrieval_ms,
                relevance_ms=relevance_ms,
                generation_ms=generation_ms,
                groundedness_ms=groundedness_ms,
            )
        if safe_refusal:
            status = ResponseStatus.REFUSED
            answer = None
            refusal_reason = RefusalReason.NO_RELEVANT_CONTEXT
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
            relevance=relevance,
            groundedness=groundedness,
            retrieval_ms=retrieval_ms,
            relevance_ms=relevance_ms,
            generation_ms=generation_ms,
            groundedness_ms=groundedness_ms,
        )
