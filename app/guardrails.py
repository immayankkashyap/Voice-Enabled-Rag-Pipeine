"""Fast deterministic relevance and groundedness guardrail boundaries."""

from __future__ import annotations

from .schemas import (
    GroundednessAssessment,
    RelevanceAssessment,
    RetrievedChunk,
)


class RelevanceGuardrail:
    """Classify retrieved evidence as correct, incorrect, or ambiguous."""

    async def assess(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> RelevanceAssessment:
        raise NotImplementedError(
            "CRAG-style relevance classification is not implemented"
        )


class GroundednessGuardrail:
    """Trace answer content to accepted chunks before it can be returned."""

    async def assess(
        self,
        *,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> GroundednessAssessment:
        raise NotImplementedError(
            "Deterministic groundedness checking is not implemented"
        )
