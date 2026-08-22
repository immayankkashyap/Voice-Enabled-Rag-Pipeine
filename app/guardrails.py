"""Fast, deterministic relevance and groundedness guardrails.

These gates deliberately fail closed.  Their lexical and cosine checks are
useful rejection signals, not proof that two pieces of text mean the same
thing.  A caller must therefore treat ``AMBIGUOUS`` exactly like a refusal and
must never describe ``CORRECT`` as semantic or factual certainty.
"""

from __future__ import annotations

import math
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from .schemas import (
    ChunkRelevanceAssessment,
    GroundednessAssessment,
    InputSafetyAssessment,
    InputSafetyCategory,
    RefusalReason,
    RelevanceAssessment,
    RelevanceClassification,
    RetrievedChunk,
)

SAFE_REFUSAL_TEXT = "I cannot answer from the provided context."

_CITATION_PATTERN = re.compile(r"\[chunk:([^\]\s]+)\]")
_SENTENCE_BOUNDARY_PATTERN = re.compile(
    r"(?<=[.!?\u3002\uff01\uff1f\u0964\u0965])\s+|\n+"
)
_MARKDOWN_PREFIX_PATTERN = re.compile(r"^\s*(?:[-*+]\s+|#{1,6}\s+|\d+[.)]\s+)")
_ENGLISH_FUNCTION_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)
_CONTRADICTION_SENSITIVE_TOKENS = frozenset({"no", "not", "never", "without"})

_UNSAFE_RULES: tuple[tuple[InputSafetyCategory, str, re.Pattern[str]], ...] = (
    (
        InputSafetyCategory.PROMPT_INJECTION,
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|override)\b.{0,48}"
            r"\b(?:previous|prior|system|developer|instructions?|prompt)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        InputSafetyCategory.PROMPT_INJECTION,
        "secret_extraction",
        re.compile(
            r"\b(?:reveal|show|print|leak|expose)\b.{0,48}"
            r"\b(?:system prompt|developer message|api[- ]?key|secret|credentials?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        InputSafetyCategory.PROMPT_INJECTION,
        "jailbreak",
        re.compile(r"\b(?:jailbreak|do anything now|developer mode)\b", re.IGNORECASE),
    ),
    (
        InputSafetyCategory.HARMFUL,
        "weapon_construction",
        re.compile(
            r"\b(?:build|make|create|assemble)\b.{0,48}"
            r"\b(?:bomb|explosive|weapon)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        InputSafetyCategory.HARMFUL,
        "credential_theft",
        re.compile(
            r"\b(?:steal|phish|exfiltrate|harvest)\b.{0,48}"
            r"\b(?:passwords?|credentials?|api[- ]?keys?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        InputSafetyCategory.HARMFUL,
        "physical_harm",
        re.compile(
            r"\b(?:kill|murder|poison)\b.{0,32}"
            r"\b(?:a person|people|someone|somebody)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        InputSafetyCategory.INAPPROPRIATE,
        "sexual_content_involving_minors",
        re.compile(
            r"\b(?:sexual|explicit|pornographic)\b.{0,40}"
            r"\b(?:child|children|minor|minors)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
)
_UNSAFE_CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u202a-\u202e\u2066-\u2069]")


@dataclass(frozen=True, slots=True)
class InputSafetyGuardrailSettings:
    """Auditable deterministic input rules evaluated before retrieval."""

    enabled_rules: tuple[str, ...] = tuple(rule[1] for rule in _UNSAFE_RULES)

    def __post_init__(self) -> None:
        known = {rule[1] for rule in _UNSAFE_RULES}
        unknown = set(self.enabled_rules) - known
        if unknown:
            raise ValueError("Unknown input-safety rules: " + ", ".join(sorted(unknown)))


@dataclass(frozen=True, slots=True)
class RelevanceGuardrailSettings:
    """Thresholds for conservative candidate acceptance.

    The score limits assume this repository's normalized inner-product index,
    whose values are cosine similarities in ``[-1, 1]``.  Both a lexical signal
    and a similarity signal are required so a high vector score alone cannot
    authorize generation.
    """

    min_full_score: float = 0.58
    strong_full_score: float = 0.68
    min_query_token_coverage: float = 0.34
    strong_query_token_coverage: float = 0.60
    score_epsilon: float = 1e-4
    scope_terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not -1 <= self.min_full_score <= self.strong_full_score <= 1:
            raise ValueError(
                "full-score thresholds must satisfy -1 <= min <= strong <= 1"
            )
        if not (
            0 <= self.min_query_token_coverage <= self.strong_query_token_coverage <= 1
        ):
            raise ValueError("coverage thresholds must satisfy 0 <= min <= strong <= 1")
        if not math.isfinite(self.score_epsilon) or not 0 <= self.score_epsilon <= 0.01:
            raise ValueError("score_epsilon must be finite and between 0 and 0.01")
        if any(not term.strip() for term in self.scope_terms):
            raise ValueError("scope_terms cannot contain blank values")


@dataclass(frozen=True, slots=True)
class GroundednessGuardrailSettings:
    """Thresholds for citation and sentence-evidence validation."""

    min_sentence_token_support: float = 0.80
    ambiguous_sentence_token_support: float = 0.60
    min_judge_budget_ms: float = 75.0
    safe_refusal_texts: tuple[str, ...] = (SAFE_REFUSAL_TEXT,)

    def __post_init__(self) -> None:
        if not (
            0
            <= self.ambiguous_sentence_token_support
            <= self.min_sentence_token_support
            <= 1
        ):
            raise ValueError(
                "grounding thresholds must satisfy 0 <= ambiguous <= minimum <= 1"
            )
        if not math.isfinite(self.min_judge_budget_ms) or self.min_judge_budget_ms < 0:
            raise ValueError("min_judge_budget_ms must be finite and non-negative")
        if not self.safe_refusal_texts or any(
            not refusal.strip() for refusal in self.safe_refusal_texts
        ):
            raise ValueError("safe_refusal_texts must contain non-blank text")


class GroundednessJudge(Protocol):
    """Optional slow fallback used only for lexically ambiguous answers."""

    async def assess(
        self,
        *,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> bool:
        """Return whether the answer is grounded in the supplied chunks."""


def _lexical_tokens(text: str) -> list[str]:
    """Tokenize Unicode letters, marks, and numbers without language guessing."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []
    current: list[str] = []
    for character in normalized:
        if unicodedata.category(character)[0] in {"L", "M", "N"}:
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current.clear()
    if current:
        tokens.append("".join(current))
    return tokens


def _content_tokens(text: str) -> list[str]:
    tokens = _lexical_tokens(text)
    filtered = [token for token in tokens if token not in _ENGLISH_FUNCTION_WORDS]
    return filtered or tokens


def _query_coverage(query: str, evidence: str) -> float:
    query_tokens = set(_content_tokens(query))
    if not query_tokens:
        return 0.0
    evidence_tokens = set(_lexical_tokens(evidence))
    matched = query_tokens.intersection(evidence_tokens)

    # Whitespace-free scripts may produce an entire phrase as one token.  In
    # that narrow case, exact substring presence is still a deterministic
    # lexical signal and avoids assuming an English-style word boundary.
    if len(query_tokens) == 1 and not matched:
        only_token = next(iter(query_tokens))
        normalized_evidence = "".join(_lexical_tokens(evidence))
        if len(only_token) >= 4 and only_token in normalized_evidence:
            matched.add(only_token)
    return len(matched) / len(query_tokens)


def _sentence_support(sentence: str, evidence: str) -> float:
    claim_tokens = Counter(_content_tokens(sentence))
    if not claim_tokens:
        return 0.0
    evidence_tokens = Counter(_lexical_tokens(evidence))
    sensitive_claim_tokens = {
        token
        for token in _lexical_tokens(sentence)
        if token in _CONTRADICTION_SENSITIVE_TOKENS
        or any(unicodedata.category(character).startswith("N") for character in token)
    }
    if not sensitive_claim_tokens.issubset(evidence_tokens):
        return 0.0
    supported = sum((claim_tokens & evidence_tokens).values())
    return supported / sum(claim_tokens.values())


def _evidence_problem(chunks: list[RetrievedChunk]) -> str | None:
    if not chunks:
        return "retrieval returned no evidence"
    chunk_ids: set[str] = set()
    ranks: set[int] = set()
    for item in chunks:
        if not item.chunk.text.strip():
            return f"chunk {item.chunk.id!r} has blank text"
        if item.chunk.id in chunk_ids:
            return f"chunk ID {item.chunk.id!r} is duplicated"
        if item.rank in ranks:
            return f"retrieval rank {item.rank} is duplicated"
        chunk_ids.add(item.chunk.id)
        ranks.add(item.rank)
        for score_name, score in (
            ("mrl_score", item.mrl_score),
            ("full_score", item.full_score),
        ):
            if not math.isfinite(score):
                return f"chunk {item.chunk.id!r} has non-finite {score_name}"
            if not -1.0001 <= score <= 1.0001:
                return (
                    f"chunk {item.chunk.id!r} has out-of-range {score_name}; "
                    "normalized cosine evidence was expected"
                )
    return None


def _bounded_signal_confidence(full_score: float, coverage: float) -> float:
    # This is confidence that the deterministic gate fired, not confidence in
    # factual truth.  Keep it below 1 because lexical/similarity heuristics
    # cannot certify meaning.
    normalized_score = min(1.0, max(0.0, (full_score + 1) / 2))
    return min(0.95, max(0.0, (normalized_score + coverage) / 2))


class InputSafetyGuardrail:
    """Reject a small, explicit set of unsafe requests before retrieval."""

    def __init__(self, settings: InputSafetyGuardrailSettings | None = None) -> None:
        self.settings = settings or InputSafetyGuardrailSettings()

    async def assess(self, *, text: str) -> InputSafetyAssessment:
        started = time.perf_counter()

        def result(
            is_safe: bool,
            category: InputSafetyCategory,
            reason: str,
            matched_rule: str | None = None,
        ) -> InputSafetyAssessment:
            return InputSafetyAssessment(
                is_safe=is_safe,
                category=category,
                matched_rule=matched_rule,
                reason=reason,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        normalized = unicodedata.normalize("NFKC", text).strip()
        if not normalized or not _content_tokens(normalized):
            return result(
                False,
                InputSafetyCategory.INVALID,
                "invalid_input: the query is blank or has no lexical content",
                "non_lexical_input",
            )
        if _UNSAFE_CONTROL_PATTERN.search(normalized):
            return result(
                False,
                InputSafetyCategory.INVALID,
                "invalid_input: disallowed control characters were detected",
                "control_characters",
            )

        enabled = set(self.settings.enabled_rules)
        for category, rule_id, pattern in _UNSAFE_RULES:
            if rule_id in enabled and pattern.search(normalized):
                return result(
                    False,
                    category,
                    "unsafe_input: the query matched a deterministic safety rule",
                    rule_id,
                )
        return result(
            True,
            InputSafetyCategory.SAFE,
            "safe_input: no deterministic unsafe-input rule matched",
        )


class RelevanceGuardrail:
    """Conservatively classify retrieved evidence and expose refusal routing."""

    def __init__(self, settings: RelevanceGuardrailSettings | None = None) -> None:
        self.settings = settings or RelevanceGuardrailSettings()

    async def assess(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> RelevanceAssessment:
        started = time.perf_counter()

        def result(
            classification: RelevanceClassification,
            confidence: float,
            accepted_chunk_ids: list[str],
            reason: str,
            chunk_assessments: list[ChunkRelevanceAssessment] | None = None,
        ) -> RelevanceAssessment:
            return RelevanceAssessment(
                classification=classification,
                confidence=confidence,
                accepted_chunk_ids=accepted_chunk_ids,
                chunk_assessments=chunk_assessments or [],
                reason=reason,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        if not query.strip() or not _content_tokens(query):
            return result(
                RelevanceClassification.INCORRECT,
                1.0,
                [],
                "invalid_query: the query is blank or has no lexical content",
            )

        problem = _evidence_problem(chunks)
        if problem is not None:
            return result(
                RelevanceClassification.INCORRECT,
                1.0,
                [],
                f"invalid_evidence: {problem}",
            )

        if self.settings.scope_terms:
            query_terms = set(_content_tokens(query))
            scope_tokens = {
                token
                for scope_term in self.settings.scope_terms
                for token in _content_tokens(scope_term)
            }
            if not query_terms.intersection(scope_tokens):
                assessments = [
                    ChunkRelevanceAssessment(
                        chunk_id=item.chunk.id,
                        classification=RelevanceClassification.INCORRECT,
                        full_score=item.full_score,
                        query_token_coverage=_query_coverage(query, item.chunk.text),
                        reason="scope_mismatch",
                    )
                    for item in sorted(chunks, key=lambda chunk: chunk.rank)
                ]
                return result(
                    RelevanceClassification.INCORRECT,
                    0.95,
                    [],
                    "off_topic: query contains none of the configured scope terms",
                    assessments,
                )

        assessments: list[ChunkRelevanceAssessment] = []
        correct: list[tuple[RetrievedChunk, float]] = []
        ambiguous: list[tuple[RetrievedChunk, float]] = []
        best_coverage = 0.0
        for item in sorted(chunks, key=lambda chunk: chunk.rank):
            coverage = _query_coverage(query, item.chunk.text)
            best_coverage = max(best_coverage, coverage)
            if (
                item.full_score + self.settings.score_epsilon
                >= self.settings.strong_full_score
                and coverage >= self.settings.strong_query_token_coverage
            ):
                classification = RelevanceClassification.CORRECT
                reason = "strong_similarity_and_lexical_coverage"
                correct.append((item, coverage))
            elif (
                item.full_score + self.settings.score_epsilon
                >= self.settings.min_full_score
                and coverage >= self.settings.min_query_token_coverage
            ):
                classification = RelevanceClassification.AMBIGUOUS
                reason = "minimum_similarity_and_lexical_coverage_only"
                ambiguous.append((item, coverage))
            else:
                classification = RelevanceClassification.INCORRECT
                reason = "below_similarity_or_lexical_coverage_gate"
            assessments.append(
                ChunkRelevanceAssessment(
                    chunk_id=item.chunk.id,
                    classification=classification,
                    full_score=item.full_score,
                    query_token_coverage=coverage,
                    reason=reason,
                )
            )

        if not correct and not ambiguous:
            confidence = 0.90 if best_coverage == 0 else 0.60
            prefix = "off_topic" if best_coverage == 0 else "no_relevant_context"
            return result(
                RelevanceClassification.INCORRECT,
                confidence,
                [],
                f"{prefix}: no chunk passed both the lexical and "
                "normalized-cosine gates; similarity alone is not semantic proof",
                assessments,
            )

        candidates = correct or ambiguous
        accepted_ids = [item.chunk.id for item, _ in candidates]
        strongest_item, strongest_coverage = max(
            candidates, key=lambda candidate: (candidate[1], candidate[0].full_score)
        )
        confidence = _bounded_signal_confidence(
            strongest_item.full_score, strongest_coverage
        )
        if not correct:
            return result(
                RelevanceClassification.AMBIGUOUS,
                confidence,
                accepted_ids,
                "ambiguous_retrieval: candidate evidence passed minimum gates but "
                "not both strong gates; refuse rather than infer relevance",
                assessments,
            )

        return result(
            RelevanceClassification.CORRECT,
            confidence,
            accepted_ids,
            "relevant_evidence: candidate evidence passed conservative lexical "
            "and normalized-cosine gates; this does not certify semantic truth",
            assessments,
        )

    @staticmethod
    def refusal_reason(
        assessment: RelevanceAssessment,
    ) -> RefusalReason | None:
        """Map every non-passing assessment to an explicit API refusal reason."""

        if assessment.classification is RelevanceClassification.CORRECT:
            return None
        if assessment.classification is RelevanceClassification.AMBIGUOUS:
            return RefusalReason.AMBIGUOUS_RETRIEVAL
        if assessment.reason.startswith("off_topic:"):
            return RefusalReason.OFF_TOPIC
        return RefusalReason.NO_RELEVANT_CONTEXT


class GroundednessGuardrail:
    """Require every answer sentence to cite and overlap known evidence."""

    def __init__(
        self,
        settings: GroundednessGuardrailSettings | None = None,
        *,
        judge: GroundednessJudge | None = None,
    ) -> None:
        self.settings = settings or GroundednessGuardrailSettings()
        self.judge = judge

    def is_safe_refusal(self, answer: str) -> bool:
        """Return true only for a complete, configured refusal and no extra text."""

        normalized_answer = " ".join(answer.split()).casefold()
        return any(
            normalized_answer == " ".join(refusal.split()).casefold()
            for refusal in self.settings.safe_refusal_texts
        )

    async def assess(
        self,
        *,
        answer: str,
        chunks: list[RetrievedChunk],
        remaining_budget_ms: float | None = None,
    ) -> GroundednessAssessment:
        started = time.perf_counter()

        def result(
            is_grounded: bool,
            score: float,
            supporting_chunk_ids: list[str],
            unsupported_claims: list[str],
            reason: str,
            *,
            method: str = "lexical_citation",
            judge_used: bool = False,
        ) -> GroundednessAssessment:
            return GroundednessAssessment(
                is_grounded=is_grounded,
                score=score,
                supporting_chunk_ids=supporting_chunk_ids,
                unsupported_claims=unsupported_claims,
                method=method,
                judge_used=judge_used,
                reason=reason,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        if self.is_safe_refusal(answer):
            # A refusal is safe to return with ResponseStatus.REFUSED, but it is
            # deliberately not allowed to satisfy RAGResponse's ANSWERED gate.
            return result(
                False,
                0.0,
                [],
                [],
                "safe_refusal: no factual answer was emitted; route as refused",
            )
        if not answer.strip() or not _content_tokens(answer):
            return result(
                False,
                0.0,
                [],
                ["<empty answer>"],
                "invalid_answer: answer is blank or has no lexical content",
            )

        problem = _evidence_problem(chunks)
        if problem is not None:
            return result(
                False,
                0.0,
                [],
                [answer.strip()],
                f"invalid_evidence: {problem}",
            )

        evidence_by_id = {item.chunk.id: item.chunk.text for item in chunks}
        claims = [
            _MARKDOWN_PREFIX_PATTERN.sub("", sentence.strip())
            for sentence in _SENTENCE_BOUNDARY_PATTERN.split(answer.strip())
            if sentence.strip()
        ]
        claims = [claim for claim in claims if _content_tokens(claim)]
        if not claims:
            return result(
                False,
                0.0,
                [],
                ["<no factual sentence>"],
                "invalid_answer: no factual sentence could be validated",
            )

        supporting_ids: list[str] = []
        unsupported_claims: list[str] = []
        ambiguous_claims: list[str] = []
        support_scores: list[float] = []
        for claim in claims:
            citation_ids = _CITATION_PATTERN.findall(claim)
            plain_claim = _CITATION_PATTERN.sub("", claim).strip()
            unknown_ids = [
                chunk_id for chunk_id in citation_ids if chunk_id not in evidence_by_id
            ]
            if not citation_ids or unknown_ids or not _content_tokens(plain_claim):
                support_scores.append(0.0)
                unsupported_claims.append(plain_claim or claim)
                continue

            combined_evidence = "\n".join(
                evidence_by_id[chunk_id] for chunk_id in dict.fromkeys(citation_ids)
            )
            support = _sentence_support(plain_claim, combined_evidence)
            support_scores.append(support)
            if support < self.settings.min_sentence_token_support:
                if support >= self.settings.ambiguous_sentence_token_support:
                    ambiguous_claims.append(plain_claim)
                else:
                    unsupported_claims.append(plain_claim)
                continue
            for chunk_id in citation_ids:
                if chunk_id not in supporting_ids:
                    supporting_ids.append(chunk_id)

        score = sum(support_scores) / len(support_scores)
        judge_budget_available = (
            remaining_budget_ms is not None
            and remaining_budget_ms >= self.settings.min_judge_budget_ms
        )
        if ambiguous_claims and not unsupported_claims:
            if self.judge is not None and judge_budget_available:
                try:
                    judge_grounded = await self.judge.assess(
                        answer=answer,
                        chunks=chunks,
                    )
                except Exception:  # noqa: BLE001 - optional judge must fail closed
                    judge_grounded = False
                if judge_grounded:
                    cited_ids = list(
                        dict.fromkeys(_CITATION_PATTERN.findall(answer))
                    )
                    return result(
                        True,
                        score,
                        cited_ids,
                        [],
                        "grounded_answer: the fast lexical check was ambiguous and "
                        "the budget-gated judge accepted the cited claims",
                        method="lexical_citation+judge",
                        judge_used=True,
                    )
                unsupported_claims.extend(ambiguous_claims)
                ambiguous_claims.clear()
            else:
                unsupported_claims.extend(ambiguous_claims)
                ambiguous_claims.clear()
        if unsupported_claims:
            fallback_reason = (
                "judge rejected or failed closed"
                if self.judge is not None and judge_budget_available
                else "judge skipped because it was unavailable or outside the latency budget"
            )
            return result(
                False,
                score,
                supporting_ids,
                unsupported_claims,
                "ungrounded_answer: every factual sentence must cite known chunks "
                f"and reach {self.settings.min_sentence_token_support:.0%} lexical "
                f"token support; {fallback_reason}",
                judge_used=self.judge is not None and judge_budget_available,
            )
        return result(
            True,
            score,
            supporting_ids,
            [],
            "grounded_answer: every factual sentence cites known chunks and "
            "passes the configured lexical-support threshold",
        )
