from __future__ import annotations

import math
import unittest

from app.extractive import (
    MODEL_NAME,
    ExtractiveAnswerGenerator,
    ExtractiveGenerationSettings,
)
from app.guardrails import SAFE_REFUSAL_TEXT, GroundednessGuardrail
from app.schemas import (
    Chunk,
    ChunkStrategy,
    GenerationRequest,
    RetrievedChunk,
)


def _retrieved(
    chunk_id: str,
    text: str,
    *,
    rank: int,
    full_score: float = 0.82,
    mrl_score: float = 0.80,
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


class ExtractiveAnswerGeneratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_selects_query_aware_exact_span_across_all_evidence(self) -> None:
        target = "The fictional city of Sundarpur was founded in 1912."
        request = GenerationRequest(
            query="When was Sundarpur founded?",
            context=[
                _retrieved(
                    "irrelevant",
                    "Sundarpur has a blue civic flag. The flag was adopted in 1984.",
                    rank=1,
                    full_score=0.91,
                ),
                _retrieved("founding", target, rank=2),
            ],
        )

        result = await ExtractiveAnswerGenerator().generate(request)

        self.assertEqual(result.answer, f"[chunk:founding] {target}")
        self.assertIn(target, result.answer)
        self.assertEqual(result.cited_chunk_ids, ["founding"])
        self.assertEqual(result.model, MODEL_NAME)
        self.assertEqual(result.finish_reason, "stop")
        self.assertGreaterEqual(result.time_to_first_token_ms, 0)
        self.assertEqual(result.total_ms, result.time_to_first_token_ms)

        grounded = await GroundednessGuardrail().assess(
            answer=result.answer,
            chunks=request.context,
        )
        self.assertTrue(grounded.is_grounded)
        self.assertEqual(grounded.supporting_chunk_ids, ["founding"])

    async def test_unicode_numbers_and_negation_are_copied_without_change(self) -> None:
        exact = "चंद्रयान 3 वर्ष 2021 में नहीं उतरा; वह 2023 में उतरा।"
        request = GenerationRequest(
            query="चंद्रयान 3 कब उतरा",
            context=[_retrieved("चंद्रयान-3", exact, rank=1)],
            language_code="hi-IN",
        )

        result = await ExtractiveAnswerGenerator().generate(request)

        self.assertEqual(result.answer, f"[chunk:चंद्रयान-3] {exact}")
        self.assertIn("नहीं", result.answer)
        self.assertIn("2021", result.answer)
        self.assertIn("2023", result.answer)

    async def test_refuses_low_score_or_insufficient_lexical_support(self) -> None:
        low_score = GenerationRequest(
            query="When was Sundarpur founded?",
            context=[
                _retrieved(
                    "low",
                    "Sundarpur was founded in 1912.",
                    rank=1,
                    full_score=0.50,
                )
            ],
        )
        no_overlap = GenerationRequest(
            query="When was Sundarpur founded?",
            context=[
                _retrieved(
                    "other",
                    "Bananas ripen faster inside a paper bag.",
                    rank=1,
                    full_score=0.99,
                )
            ],
        )

        generator = ExtractiveAnswerGenerator()
        for request in (low_score, no_overlap):
            with self.subTest(chunk=request.context[0].chunk.id):
                result = await generator.generate(request)
                self.assertEqual(result.answer, SAFE_REFUSAL_TEXT)
                self.assertEqual(result.cited_chunk_ids, [])
                self.assertEqual(result.model, MODEL_NAME)

    async def test_refuses_ambiguous_evidence_without_a_safe_margin(self) -> None:
        request = GenerationRequest(
            query="When was Sundarpur founded?",
            context=[
                _retrieved(
                    "claim-a",
                    "Sundarpur was founded in 1912.",
                    rank=1,
                    full_score=0.82,
                ),
                _retrieved(
                    "claim-b",
                    "Sundarpur was founded in 1913.",
                    rank=2,
                    full_score=0.81,
                ),
            ],
        )

        result = await ExtractiveAnswerGenerator().generate(request)

        self.assertEqual(result.answer, SAFE_REFUSAL_TEXT)
        self.assertEqual(result.cited_chunk_ids, [])

    async def test_refuses_opposite_query_polarity_across_demo_languages(self) -> None:
        cases = (
            (
                "chart of foods low in potassium",
                "Foods high in potassium include bananas and potatoes.",
            ),
            (
                "पोटेशियम में कम खाद्य पदार्थ",
                "उच्च पोटेशियम वाले खाद्य पदार्थों में केले शामिल हैं।",
            ),
            (
                "பொட்டாசியம் குறைவுள்ள உணவுகள்",
                "அதிக பொட்டாசியம் உள்ள உணவுகளில் வாழைப்பழம் அடங்கும்.",
            ),
            (
                "پوٹاشیم میں کم خوراک",
                "زیادہ پوٹاشیم والی خوراک میں کیلے شامل ہیں۔",
            ),
        )
        generator = ExtractiveAnswerGenerator()
        for index, (query, evidence) in enumerate(cases):
            with self.subTest(query=query):
                result = await generator.generate(
                    GenerationRequest(
                        query=query,
                        context=[_retrieved(f"polarity-{index}", evidence, rank=1)],
                    )
                )
                self.assertEqual(result.answer, SAFE_REFUSAL_TEXT)
                self.assertEqual(result.cited_chunk_ids, [])

    async def test_quantity_query_requires_an_explicit_quantity(self) -> None:
        missing = await ExtractiveAnswerGenerator().generate(
            GenerationRequest(
                query="ஸ்ட்ரதர்ஸ் நகரப் பள்ளி மாவட்டம் மாநில எண்",
                context=[
                    _retrieved(
                        "topical-no-number",
                        "ஸ்ட்ரதர்ஸ் நகரப் பள்ளி மாவட்டம் ஓஹியோவில் அமைந்துள்ளது.",
                        rank=1,
                    )
                ],
            )
        )
        present = await ExtractiveAnswerGenerator().generate(
            GenerationRequest(
                query="कितनी जांच चल रही हैं",
                context=[
                    _retrieved(
                        "explicit-number",
                        "वर्तमान में तीन जांच चल रही हैं।",
                        rank=1,
                    )
                ],
            )
        )

        self.assertEqual(missing.answer, SAFE_REFUSAL_TEXT)
        self.assertEqual(
            present.answer,
            "[chunk:explicit-number] वर्तमान में तीन जांच चल रही हैं।",
        )

    async def test_identical_overlapping_chunks_do_not_create_false_ambiguity(
        self,
    ) -> None:
        exact = "Sundarpur was founded in 1912."
        request = GenerationRequest(
            query="When was Sundarpur founded?",
            context=[
                _retrieved("weaker-copy", exact, rank=1, full_score=0.80),
                _retrieved("stronger-copy", exact, rank=2, full_score=0.90),
            ],
        )

        result = await ExtractiveAnswerGenerator().generate(request)

        self.assertEqual(result.answer, f"[chunk:stronger-copy] {exact}")
        self.assertEqual(result.cited_chunk_ids, ["stronger-copy"])

    async def test_refuses_non_content_and_unsafe_citation_ids(self) -> None:
        punctuation_only = GenerationRequest(
            query="Sundarpur founding date",
            context=[_retrieved("punctuation", "...", rank=1)],
        )
        unsafe_id = GenerationRequest(
            query="Sundarpur founding date",
            context=[
                _retrieved(
                    "bad id]",
                    "Sundarpur's founding date was 1912.",
                    rank=1,
                )
            ],
        )

        generator = ExtractiveAnswerGenerator()
        for request in (punctuation_only, unsafe_id):
            result = await generator.generate(request)
            self.assertEqual(result.answer, SAFE_REFUSAL_TEXT)
            self.assertEqual(result.cited_chunk_ids, [])

    async def test_invalid_scores_fail_closed(self) -> None:
        request = GenerationRequest(
            query="Sundarpur founding date",
            context=[
                _retrieved(
                    "bad-score",
                    "Sundarpur's founding date was 1912.",
                    rank=1,
                    full_score=math.nan,
                )
            ],
        )

        result = await ExtractiveAnswerGenerator().generate(request)

        self.assertEqual(result.answer, SAFE_REFUSAL_TEXT)

    def test_settings_validate_conservative_gates(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_full_score"):
            ExtractiveGenerationSettings(min_full_score=math.nan)
        with self.assertRaisesRegex(ValueError, "min_query_coverage"):
            ExtractiveGenerationSettings(min_query_coverage=1.1)
        with self.assertRaisesRegex(ValueError, "min_selection_margin"):
            ExtractiveGenerationSettings(min_selection_margin=-0.1)
        with self.assertRaisesRegex(ValueError, "max_sentence_chars"):
            ExtractiveGenerationSettings(max_sentence_chars=0)


if __name__ == "__main__":
    unittest.main()
