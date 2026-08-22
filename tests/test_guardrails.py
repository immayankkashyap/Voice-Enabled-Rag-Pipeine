from __future__ import annotations

import math
import unittest

from app.guardrails import (
    SAFE_REFUSAL_TEXT,
    GroundednessGuardrail,
    RelevanceGuardrail,
    RelevanceGuardrailSettings,
)
from app.schemas import (
    Chunk,
    ChunkStrategy,
    RefusalReason,
    RelevanceClassification,
    RetrievedChunk,
)


def _retrieved(
    chunk_id: str,
    text: str,
    *,
    rank: int = 1,
    mrl_score: float = 0.80,
    full_score: float = 0.80,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id=chunk_id,
            document_id=f"doc-{chunk_id}",
            text=text,
            strategy=ChunkStrategy.LATE,
        ),
        mrl_score=mrl_score,
        full_score=full_score,
        rank=rank,
    )


class RelevanceGuardrailTests(unittest.IsolatedAsyncioTestCase):
    async def test_accepts_only_evidence_with_strong_lexical_and_score_signals(
        self,
    ) -> None:
        assessment = await RelevanceGuardrail().assess(
            query="When was Sundarpur founded?",
            chunks=[
                _retrieved(
                    "city",
                    "The fictional city of Sundarpur was founded in 1912.",
                ),
                _retrieved(
                    "bird",
                    "The blue kite is the city's civic bird.",
                    rank=2,
                    full_score=0.90,
                ),
            ],
        )

        self.assertEqual(assessment.classification, RelevanceClassification.CORRECT)
        self.assertEqual(assessment.accepted_chunk_ids, ["city"])
        self.assertLess(assessment.confidence, 1.0)
        self.assertIn("does not certify semantic truth", assessment.reason)
        self.assertGreaterEqual(assessment.latency_ms, 0)

    async def test_high_similarity_without_lexical_evidence_fails_closed(self) -> None:
        assessment = await RelevanceGuardrail().assess(
            query="How deep is the Pacific Ocean?",
            chunks=[
                _retrieved(
                    "unrelated",
                    "Bananas ripen faster inside a paper bag.",
                    full_score=0.99,
                )
            ],
        )

        self.assertEqual(assessment.classification, RelevanceClassification.INCORRECT)
        self.assertEqual(assessment.accepted_chunk_ids, [])
        self.assertEqual(
            RelevanceGuardrail.refusal_reason(assessment),
            RefusalReason.NO_RELEVANT_CONTEXT,
        )

    async def test_minimum_only_evidence_is_ambiguous_and_refused(self) -> None:
        assessment = await RelevanceGuardrail().assess(
            query="Sundarpur founding date and population",
            chunks=[
                _retrieved(
                    "partial",
                    "Sundarpur's founding date was 1912.",
                    full_score=0.60,
                )
            ],
        )

        self.assertEqual(assessment.classification, RelevanceClassification.AMBIGUOUS)
        self.assertEqual(assessment.accepted_chunk_ids, ["partial"])
        self.assertEqual(
            RelevanceGuardrail.refusal_reason(assessment),
            RefusalReason.AMBIGUOUS_RETRIEVAL,
        )

    async def test_configured_scope_produces_explicit_off_topic_refusal(self) -> None:
        guardrail = RelevanceGuardrail(
            RelevanceGuardrailSettings(scope_terms=("spaceflight", "orbital"))
        )
        assessment = await guardrail.assess(
            query="How should I bake sourdough?",
            chunks=[_retrieved("mission", "An orbital mission launched in 2024.")],
        )

        self.assertTrue(assessment.reason.startswith("off_topic:"))
        self.assertEqual(
            RelevanceGuardrail.refusal_reason(assessment), RefusalReason.OFF_TOPIC
        )

    async def test_empty_duplicate_and_non_finite_evidence_are_rejected(self) -> None:
        empty = await RelevanceGuardrail().assess(query="query", chunks=[])
        self.assertTrue(empty.reason.startswith("invalid_evidence:"))

        duplicate = await RelevanceGuardrail().assess(
            query="Sundarpur",
            chunks=[
                _retrieved("same", "Sundarpur", rank=1),
                _retrieved("same", "Sundarpur", rank=2),
            ],
        )
        self.assertIn("duplicated", duplicate.reason)

        non_finite = await RelevanceGuardrail().assess(
            query="Sundarpur",
            chunks=[_retrieved("bad", "Sundarpur", full_score=math.nan)],
        )
        self.assertIn("non-finite", non_finite.reason)
        self.assertEqual(non_finite.classification, RelevanceClassification.INCORRECT)


class GroundednessGuardrailTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.guardrail = GroundednessGuardrail()
        self.chunks = [
            _retrieved(
                "city",
                "The fictional city of Sundarpur was founded in 1912.",
            ),
            _retrieved(
                "bird",
                "Sundarpur's fictional civic bird is the blue kite.",
                rank=2,
            ),
        ]

    async def test_every_sentence_with_known_supported_citation_passes(self) -> None:
        assessment = await self.guardrail.assess(
            answer=(
                "Sundarpur was founded in 1912 [chunk:city]. "
                "Its civic bird is the blue kite [chunk:bird]."
            ),
            chunks=self.chunks,
        )

        self.assertTrue(assessment.is_grounded)
        self.assertEqual(assessment.supporting_chunk_ids, ["city", "bird"])
        self.assertEqual(assessment.unsupported_claims, [])
        self.assertGreaterEqual(assessment.score, 0.80)

    async def test_missing_sentence_citation_fails_closed(self) -> None:
        assessment = await self.guardrail.assess(
            answer=(
                "Sundarpur was founded in 1912 [chunk:city]. "
                "Its population is one million."
            ),
            chunks=self.chunks,
        )

        self.assertFalse(assessment.is_grounded)
        self.assertEqual(len(assessment.unsupported_claims), 1)
        self.assertIn("population", assessment.unsupported_claims[0])

    async def test_unknown_citation_is_never_accepted(self) -> None:
        assessment = await self.guardrail.assess(
            answer="Sundarpur was founded in 1912 [chunk:invented].",
            chunks=self.chunks,
        )

        self.assertFalse(assessment.is_grounded)
        self.assertEqual(assessment.supporting_chunk_ids, [])
        self.assertEqual(len(assessment.unsupported_claims), 1)

    async def test_known_citation_cannot_support_a_different_fact(self) -> None:
        assessment = await self.guardrail.assess(
            answer="Sundarpur was founded in 1913 [chunk:city].",
            chunks=self.chunks,
        )

        self.assertFalse(assessment.is_grounded)
        self.assertLess(assessment.score, 0.80)

    async def test_changed_number_or_negation_fails_even_with_high_overlap(
        self,
    ) -> None:
        evidence = [
            _retrieved(
                "mission",
                "The successful red orbital science mission launched from Goa in 2020.",
            )
        ]
        changed_number = await self.guardrail.assess(
            answer=(
                "The successful red orbital science mission launched from Goa "
                "in 2021 [chunk:mission]."
            ),
            chunks=evidence,
        )
        added_negation = await self.guardrail.assess(
            answer=(
                "The successful red orbital science mission did not launch from "
                "Goa in 2020 [chunk:mission]."
            ),
            chunks=evidence,
        )

        self.assertFalse(changed_number.is_grounded)
        self.assertFalse(added_negation.is_grounded)

    async def test_safe_refusal_is_routed_as_refused_not_answered(self) -> None:
        assessment = await self.guardrail.assess(
            answer=SAFE_REFUSAL_TEXT,
            chunks=[],
        )

        self.assertTrue(self.guardrail.is_safe_refusal(SAFE_REFUSAL_TEXT))
        self.assertFalse(assessment.is_grounded)
        self.assertEqual(assessment.unsupported_claims, [])
        self.assertTrue(assessment.reason.startswith("safe_refusal:"))
        self.assertFalse(
            self.guardrail.is_safe_refusal(
                SAFE_REFUSAL_TEXT + " Also, here is an unsupported answer."
            )
        )

    async def test_non_finite_evidence_cannot_ground_an_answer(self) -> None:
        assessment = await self.guardrail.assess(
            answer="Sundarpur was founded in 1912 [chunk:city].",
            chunks=[
                _retrieved(
                    "city",
                    "Sundarpur was founded in 1912.",
                    mrl_score=math.inf,
                )
            ],
        )

        self.assertFalse(assessment.is_grounded)
        self.assertTrue(assessment.reason.startswith("invalid_evidence:"))


if __name__ == "__main__":
    unittest.main()
