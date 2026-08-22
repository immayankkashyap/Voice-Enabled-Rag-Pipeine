from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from app.extractive import ExtractiveAnswerGenerator
from app.guardrails import SAFE_REFUSAL_TEXT
from app.pipeline import FastRAGPipeline
from app.schemas import (
    Chunk,
    ChunkStrategy,
    GenerationRequest,
    GenerationResult,
    RAGRequest,
    RefusalReason,
    RelevanceClassification,
    ResponseStatus,
    RetrievalRequest,
    RetrievalResult,
    RetrievedChunk,
)
from app.schemas import (
    RAGResponse as RAGResponseModel,
)


def _retrieved(
    chunk_id: str,
    text: str,
    *,
    rank: int = 1,
    language: str = "en",
    full_score: float = 0.82,
    mrl_score: float = 0.80,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id=chunk_id,
            document_id=f"doc-{chunk_id}",
            text=text,
            strategy=ChunkStrategy.LATE,
            metadata={"language": language},
        ),
        mrl_score=mrl_score,
        full_score=full_score,
        rank=rank,
    )


def _retrieval_result(
    chunks: list[RetrievedChunk], *, total_ms: float = 7.5
) -> RetrievalResult:
    return RetrievalResult(
        chunks=chunks,
        query_embedding_ms=6.0,
        mrl_search_ms=0.5,
        full_rerank_ms=0.25,
        total_ms=total_ms,
    )


class _StaticRetriever:
    def __init__(
        self,
        result: RetrievalResult | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.requests: list[RetrievalRequest] = []

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        if self.result is None:
            raise AssertionError("Static retriever has no configured result")
        return self.result


class _TrackingGenerator:
    def __init__(
        self,
        answer: str,
        *,
        cited_chunk_ids: list[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.answer = answer
        self.cited_chunk_ids = cited_chunk_ids or []
        self.error = error
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return GenerationResult(
            answer=self.answer,
            cited_chunk_ids=self.cited_chunk_ids,
            model="local/test-fixture",
            finish_reason="stop",
            time_to_first_token_ms=0.2,
            total_ms=0.3,
        )


class _ExplodingRelevance:
    async def assess(self, **_: object) -> None:
        raise RuntimeError("provider-secret relevance failure")


class _ExplodingGroundedness:
    async def assess(self, **_: object) -> None:
        raise RuntimeError("provider-secret grounding failure")

    def is_safe_refusal(self, answer: str) -> bool:
        raise AssertionError(f"is_safe_refusal must not run for {answer!r}")


class FastRAGPipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.query = "When was Sundarpur founded?"
        self.city = _retrieved(
            "city",
            "The fictional city of Sundarpur was founded in 1912.",
        )

    def _pipeline(
        self,
        *,
        retriever: object | None = None,
        generator: object | None = None,
        relevance: object | None = None,
        groundedness: object | None = None,
        latency_target_ms: float = 1_000_000.0,
    ) -> FastRAGPipeline:
        return FastRAGPipeline(
            retriever=retriever  # type: ignore[arg-type]
            or _StaticRetriever(_retrieval_result([self.city])),
            generator=generator  # type: ignore[arg-type]
            or ExtractiveAnswerGenerator(),
            relevance=relevance,  # type: ignore[arg-type]
            groundedness=groundedness,  # type: ignore[arg-type]
            latency_target_ms=latency_target_ms,
        )

    async def test_answered_path_returns_only_grounded_extractive_evidence(
        self,
    ) -> None:
        retriever = _StaticRetriever(_retrieval_result([self.city]))
        response = await self._pipeline(retriever=retriever).answer(
            RAGRequest(
                query=self.query,
                candidate_k=17,
                final_k=3,
            )
        )

        self.assertEqual(response.status, ResponseStatus.ANSWERED)
        self.assertEqual(
            response.answer,
            "[chunk:city] The fictional city of Sundarpur was founded in 1912.",
        )
        self.assertIsNone(response.refusal_reason)
        self.assertEqual(response.retrieved_chunks, [self.city])
        self.assertIsNotNone(response.relevance)
        self.assertEqual(
            response.relevance.classification, RelevanceClassification.CORRECT
        )
        self.assertIsNotNone(response.groundedness)
        self.assertTrue(response.groundedness.is_grounded)
        self.assertEqual(response.groundedness.supporting_chunk_ids, ["city"])
        self.assertEqual(retriever.requests[0].candidate_k, 17)
        self.assertEqual(retriever.requests[0].final_k, 3)
        self.assertEqual(response.latencies.retrieval_ms, 7.5)
        self.assertIsNotNone(response.latencies.relevance_ms)
        self.assertIsNotNone(response.latencies.generation_ms)
        self.assertIsNotNone(response.latencies.groundedness_ms)
        self.assertIsNotNone(response.latencies.output_ms)
        self.assertTrue(response.latencies.target_met)
        self.assertLessEqual(response.latencies.total_ms, response.latencies.target_ms)

    async def test_relevance_refusal_skips_generation(self) -> None:
        unrelated = _retrieved(
            "unrelated",
            "Bananas ripen faster inside a paper bag.",
            full_score=0.99,
        )
        generator = _TrackingGenerator("must not be returned")
        response = await self._pipeline(
            retriever=_StaticRetriever(_retrieval_result([unrelated])),
            generator=generator,
        ).answer(RAGRequest(query="How deep is the Pacific Ocean?"))

        self.assertEqual(response.status, ResponseStatus.REFUSED)
        self.assertEqual(response.refusal_reason, RefusalReason.NO_RELEVANT_CONTEXT)
        self.assertIsNone(response.answer)
        self.assertEqual(generator.requests, [])
        self.assertEqual(
            response.relevance.classification, RelevanceClassification.INCORRECT
        )
        self.assertIsNone(response.groundedness)
        self.assertIsNone(response.latencies.generation_ms)

    async def test_requested_language_with_no_matching_evidence_fails_closed(
        self,
    ) -> None:
        generator = _TrackingGenerator("must not be returned")
        response = await self._pipeline(generator=generator).answer(
            RAGRequest(query=self.query, language_code="hi-IN")
        )

        self.assertEqual(response.status, ResponseStatus.REFUSED)
        self.assertEqual(response.refusal_reason, RefusalReason.NO_RELEVANT_CONTEXT)
        self.assertEqual(response.retrieved_chunks, [])
        self.assertEqual(response.relevance.accepted_chunk_ids, [])
        self.assertEqual(generator.requests, [])
        self.assertIsNone(response.answer)

    async def test_language_filter_happens_before_final_result_slice(self) -> None:
        hindi = _retrieved(
            "hindi-city",
            "सुंदरपुर शहर की स्थापना 1912 में हुई थी।",
            rank=6,
            language="hi",
        )
        cross_language = [
            _retrieved(
                f"english-{rank}",
                "Sundarpur has an unrelated civic record.",
                rank=rank,
                language="en",
            )
            for rank in range(1, 6)
        ]
        retriever = _StaticRetriever(_retrieval_result([*cross_language, hindi]))

        response = await self._pipeline(retriever=retriever).answer(
            RAGRequest(
                query="सुंदरपुर शहर की स्थापना कब हुई",
                language_code="hi",
                candidate_k=10,
                final_k=1,
            )
        )

        self.assertEqual(retriever.requests[0].final_k, 10)
        self.assertEqual(response.status, ResponseStatus.ANSWERED)
        self.assertEqual(response.retrieved_chunks, [hindi])

    async def test_extractive_ambiguity_is_an_explicit_refusal(self) -> None:
        conflicting = _retrieved(
            "conflict",
            "Sundarpur was founded in 1913.",
            rank=2,
            full_score=0.81,
        )
        response = await self._pipeline(
            retriever=_StaticRetriever(_retrieval_result([self.city, conflicting]))
        ).answer(RAGRequest(query=self.query))

        self.assertEqual(response.status, ResponseStatus.REFUSED)
        self.assertEqual(response.refusal_reason, RefusalReason.NO_RELEVANT_CONTEXT)
        self.assertIsNone(response.answer)
        self.assertEqual(
            response.relevance.classification, RelevanceClassification.CORRECT
        )
        self.assertIsNotNone(response.groundedness)
        self.assertTrue(response.groundedness.reason.startswith("safe_refusal:"))
        self.assertFalse(response.groundedness.is_grounded)
        self.assertIsNotNone(response.latencies.generation_ms)

    async def test_ungrounded_generated_claim_is_never_returned(self) -> None:
        unsupported = "Sundarpur was founded in 1913 [chunk:city]."
        generator = _TrackingGenerator(unsupported, cited_chunk_ids=["city"])
        response = await self._pipeline(generator=generator).answer(
            RAGRequest(query=self.query)
        )

        self.assertEqual(response.status, ResponseStatus.REFUSED)
        self.assertEqual(response.refusal_reason, RefusalReason.UNGROUNDED_ANSWER)
        self.assertIsNone(response.answer)
        self.assertFalse(response.groundedness.is_grounded)
        self.assertIn("1913", response.groundedness.unsupported_claims[0])
        self.assertNotIn(unsupported, response.model_dump_json())

    async def test_latency_target_flag_is_derived_from_measured_total(self) -> None:
        response = await self._pipeline(latency_target_ms=0.000001).answer(
            RAGRequest(query=self.query)
        )

        self.assertEqual(response.status, ResponseStatus.ANSWERED)
        self.assertFalse(response.latencies.target_met)
        self.assertGreater(response.latencies.total_ms, response.latencies.target_ms)
        with self.assertRaisesRegex(ValueError, "latency_target_ms"):
            self._pipeline(latency_target_ms=0)

    async def test_total_and_output_include_final_response_construction(self) -> None:
        class _Clock:
            now = 100.0

            def read(self) -> float:
                return self.now

        clock = _Clock()

        def build_response(**kwargs: object) -> RAGResponseModel:
            clock.now += 0.050
            return RAGResponseModel(**kwargs)

        pipeline = self._pipeline(latency_target_ms=25.0)
        with (
            patch("app.pipeline.time.perf_counter", side_effect=clock.read),
            patch("app.pipeline.RAGResponse", side_effect=build_response),
        ):
            response = await pipeline.answer(RAGRequest(query=self.query))

        self.assertAlmostEqual(response.latencies.output_ms, 50.0)
        self.assertAlmostEqual(response.latencies.total_ms, 50.0)
        self.assertFalse(response.latencies.target_met)

    async def test_each_upstream_stage_exception_becomes_typed_refusal(
        self,
    ) -> None:
        cases = {
            "retrieval": self._pipeline(
                retriever=_StaticRetriever(
                    error=RuntimeError("provider-secret retrieval failure")
                )
            ),
            "relevance": self._pipeline(relevance=_ExplodingRelevance()),
            "generation": self._pipeline(
                generator=_TrackingGenerator(
                    "unused",
                    error=RuntimeError("provider-secret generation failure"),
                )
            ),
            "groundedness": self._pipeline(
                generator=_TrackingGenerator(
                    "partial-generated-answer [chunk:city].",
                    cited_chunk_ids=["city"],
                ),
                groundedness=_ExplodingGroundedness(),
            ),
        }

        for stage, pipeline in cases.items():
            with self.subTest(stage=stage):
                response = await pipeline.answer(RAGRequest(query=self.query))
                serialized = response.model_dump_json()

                self.assertEqual(response.status, ResponseStatus.REFUSED)
                self.assertEqual(
                    response.refusal_reason, RefusalReason.UPSTREAM_FAILURE
                )
                self.assertIsNone(response.answer)
                self.assertIsNone(response.groundedness)
                self.assertNotIn("provider-secret", serialized)
                self.assertNotIn("partial-generated-answer", serialized)
                self.assertEqual(
                    response.latencies.target_met,
                    response.latencies.total_ms <= response.latencies.target_ms,
                )
                self.assertIsNotNone(response.latencies.output_ms)
                if stage == "retrieval":
                    self.assertEqual(response.retrieved_chunks, [])
                    self.assertIsNone(response.relevance)
                    self.assertIsNone(response.latencies.retrieval_ms)
                else:
                    self.assertEqual(response.retrieved_chunks, [self.city])
                    self.assertEqual(response.latencies.retrieval_ms, 7.5)
                if stage in {"generation", "groundedness"}:
                    self.assertIsNotNone(response.relevance)
                    self.assertIsNotNone(response.latencies.relevance_ms)

    async def test_task_cancellation_is_not_converted_to_a_refusal(self) -> None:
        retriever = _StaticRetriever(error=asyncio.CancelledError())

        with self.assertRaises(asyncio.CancelledError):
            await self._pipeline(retriever=retriever).answer(
                RAGRequest(query=self.query)
            )

    async def test_generator_canonical_refusal_constant_matches_guardrail(
        self,
    ) -> None:
        generator = _TrackingGenerator(SAFE_REFUSAL_TEXT)
        response = await self._pipeline(generator=generator).answer(
            RAGRequest(query=self.query)
        )

        self.assertEqual(response.status, ResponseStatus.REFUSED)
        self.assertEqual(response.refusal_reason, RefusalReason.NO_RELEVANT_CONTEXT)


if __name__ == "__main__":
    unittest.main()
