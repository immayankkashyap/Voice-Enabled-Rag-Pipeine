"""Zero-cost, deterministic extractive answer generation.

This module never calls a model or a network service.  It considers every
accepted retrieval result, chooses at most one unmodified evidence sentence,
and prefixes that exact span with its source citation.  Weak or ambiguous
evidence produces the same fail-closed refusal used by the grounding guardrail.

The selector is intentionally conservative: lexical overlap is only a routing
signal, not proof that a sentence answers a question.  Callers must still run
the normal groundedness guardrail before returning an answered response.
"""

from __future__ import annotations

import math
import re
import time
import unicodedata
from dataclasses import dataclass

from .guardrails import SAFE_REFUSAL_TEXT
from .schemas import GenerationRequest, GenerationResult, RetrievedChunk

MODEL_NAME = "local/extractive-v1"

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?\u3002\uff01\uff1f\u0964\u0965])\s+|\n+")
_SAFE_CITATION_ID = re.compile(r"^[^\]\s]+$")

# Removing grammatical question scaffolding makes selection depend on subject
# and relation terms.  Negations are deliberately absent from this set.
_FUNCTION_WORDS = frozenset(
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
        # Common MSMARCO-XI question words/copulas.  These are grammatical
        # scaffolding, not answer-bearing entities.  Keeping this list small
        # is safer than applying stemming or fuzzy cross-script matching.
        "कब",
        "कहाँ",
        "कहा",
        "क्या",
        "कौन",
        "कैसे",
        "क्यों",
        "है",
        "हैं",
        "का",
        "की",
        "के",
        "में",
        "कुत्र",
        "अस्ति",
        "आहे",
        "कुठे",
        "छ",
        "কোথায়",
        "কি",
        "কী",
        "আছে",
        "எங்கே",
        "என்ன",
        "உள்ளது",
        "ఎక్కడ",
        "ఏమి",
        "ఉంది",
        "ಎಲ್ಲಿ",
        "ಎಲ್ಲಿದೆ",
        "ಏನು",
        "ಇದೆ",
        "എവിടെ",
        "എവിടെയാണ്",
        "എന്ത്",
        "ആണ്",
        "ક્યારે",
        "ક્યાં",
        "કયાં",
        "શું",
        "છે",
        "کہاں",
        "کیا",
        "ہے",
        "କେଉଁଠି",
        "କଣ",
        "ଅଛି",
        "ਕਿੱਥੇ",
        "ਕੀ",
        "ਹੈ",
    }
)

# Exact extraction proves that wording came from evidence; it does not prove
# that the wording answers the requested polarity.  These deliberately small,
# auditable antonym sets cover the current Hindi/Tamil/Urdu demo index plus
# English.  If opposite poles occur in query and candidate, refusing is safer
# than returning a fluent but inverted fact (for example, low- vs high-potassium
# foods).  This is not advertised as complete semantic contradiction detection.
_CONTRADICTORY_TOKEN_GROUPS = (
    (
        frozenset(
            {
                "low",
                "lower",
                "lowest",
                "minimum",
                "कम",
                "निम्न",
                "न्यूनतम",
                "குறைவு",
                "குறைந்த",
                "குறைவுள்ள",
                "குறைவான",
                "کم",
                "کمترین",
            }
        ),
        frozenset(
            {
                "high",
                "higher",
                "highest",
                "maximum",
                "अधिक",
                "उच्च",
                "अधिकतम",
                "அதிக",
                "அதிகம்",
                "உயர்ந்த",
                "زیادہ",
                "اعلی",
                "اعلیٰ",
                "بلند",
            }
        ),
    ),
    (
        frozenset({"before", "earlier", "पहले", "पूर्व", "முன்", "پہلے"}),
        frozenset({"after", "later", "बाद", "पश्चात", "பின்", "بعد"}),
    ),
    (
        frozenset(
            {
                "increase",
                "increased",
                "rising",
                "बढ़ा",
                "बढ़ी",
                "அதிகரித்தது",
                "اضافہ",
                "بڑھا",
            }
        ),
        frozenset(
            {
                "decrease",
                "decreased",
                "falling",
                "घटा",
                "घटी",
                "குறைந்தது",
                "کمی",
                "گھٹا",
            }
        ),
    ),
)

_QUANTITY_QUERY_TOKENS = frozenset(
    {
        "howmany",
        "many",
        "number",
        "count",
        "कितना",
        "कितनी",
        "कितने",
        "संख्या",
        "नंबर",
        "எத்தனை",
        "எண்",
        "எண்ணிக்கை",
        "کتنا",
        "کتنی",
        "کتنے",
        "تعداد",
        "نمبر",
    }
)
_QUANTITY_ANSWER_WORDS = frozenset(
    {
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "शून्य",
        "एक",
        "दो",
        "तीन",
        "चार",
        "पांच",
        "पाँच",
        "छह",
        "सात",
        "आठ",
        "नौ",
        "दस",
        "பூஜ்ஜியம்",
        "ஒன்று",
        "இரண்டு",
        "மூன்று",
        "நான்கு",
        "ஐந்து",
        "ஆறு",
        "ஏழு",
        "எட்டு",
        "ஒன்பது",
        "பத்து",
        "صفر",
        "ایک",
        "دو",
        "تین",
        "چار",
        "پانچ",
        "چھ",
        "سات",
        "آٹھ",
        "نو",
        "دس",
    }
)


@dataclass(frozen=True, slots=True)
class ExtractiveGenerationSettings:
    """Conservative gates for deterministic evidence selection.

    ``full_score`` is expected to be normalized cosine similarity in [-1, 1].
    The margin is measured on the selector's bounded composite score.  Raising
    any threshold increases refusals; lowering one must be justified by a
    retrieval-quality evaluation rather than latency alone.
    """

    min_full_score: float = 0.68
    min_query_coverage: float = 0.60
    min_selection_margin: float = 0.08
    min_query_content_tokens: int = 2
    min_sentence_content_tokens: int = 2
    max_sentence_chars: int = 600
    full_score_weight: float = 0.20
    specificity_weight: float = 0.10
    score_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_full_score) or not -1 <= self.min_full_score <= 1:
            raise ValueError("min_full_score must be finite and between -1 and 1")
        if not 0 <= self.min_query_coverage <= 1:
            raise ValueError("min_query_coverage must be between 0 and 1")
        if not math.isfinite(self.min_selection_margin) or not (
            0 <= self.min_selection_margin <= 2
        ):
            raise ValueError("min_selection_margin must be finite and between 0 and 2")
        if self.min_query_content_tokens < 1:
            raise ValueError("min_query_content_tokens must be positive")
        if self.min_sentence_content_tokens < 1:
            raise ValueError("min_sentence_content_tokens must be positive")
        if self.max_sentence_chars < 1:
            raise ValueError("max_sentence_chars must be positive")
        if not math.isfinite(self.full_score_weight) or not (
            0 <= self.full_score_weight <= 1
        ):
            raise ValueError("full_score_weight must be finite and between 0 and 1")
        if not math.isfinite(self.specificity_weight) or not (
            0 <= self.specificity_weight <= 1
        ):
            raise ValueError("specificity_weight must be finite and between 0 and 1")
        if not math.isfinite(self.score_epsilon) or not 0 <= self.score_epsilon <= 0.01:
            raise ValueError("score_epsilon must be finite and between 0 and 0.01")


@dataclass(frozen=True, slots=True)
class _Candidate:
    span: str
    chunk_id: str
    rank: int
    sentence_index: int
    full_score: float
    query_coverage: float
    selection_score: float


def _lexical_tokens(text: str) -> list[str]:
    """Return Unicode letter/mark/number tokens without lossy transliteration."""

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
    filtered = [token for token in tokens if token not in _FUNCTION_WORDS]
    return filtered or tokens


def _sentences(text: str) -> list[str]:
    """Split on explicit Unicode sentence boundaries and retain exact spans."""

    return [span for part in _SENTENCE_BOUNDARY.split(text) if (span := part.strip())]


def _valid_retrieval_score(item: RetrievedChunk) -> bool:
    return all(
        math.isfinite(score) and -1.0001 <= score <= 1.0001
        for score in (item.mrl_score, item.full_score)
    )


def _contradicts_query(query_tokens: set[str], sentence_tokens: set[str]) -> bool:
    return any(
        (query_tokens.intersection(left) and sentence_tokens.intersection(right))
        or (query_tokens.intersection(right) and sentence_tokens.intersection(left))
        for left, right in _CONTRADICTORY_TOKEN_GROUPS
    )


def _misses_required_quantity(
    query_tokens: set[str], sentence_tokens: set[str]
) -> bool:
    # English "how many" is tokenized as two words; the multilingual terms are
    # normally single tokens. A quantity-seeking query must not be answered by
    # a merely topical sentence with no digit or auditable number word.
    asks_quantity = bool(query_tokens.intersection(_QUANTITY_QUERY_TOKENS)) or {
        "how",
        "many",
    }.issubset(query_tokens)
    if not asks_quantity:
        return False
    has_digit = any(
        any(character.isdigit() for character in token) for token in sentence_tokens
    )
    return not has_digit and not bool(
        sentence_tokens.intersection(_QUANTITY_ANSWER_WORDS)
    )


def _candidate_key(candidate: _Candidate) -> tuple[float, int, str, int]:
    """Stable descending-score order with retrieval rank as the first tie-break."""

    return (
        -candidate.selection_score,
        candidate.rank,
        candidate.chunk_id,
        candidate.sentence_index,
    )


class ExtractiveAnswerGenerator:
    """Select one exact sentence from accepted evidence or refuse."""

    def __init__(self, settings: ExtractiveGenerationSettings | None = None) -> None:
        self.settings = settings or ExtractiveGenerationSettings()

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        started = time.perf_counter()
        answer, cited_ids = self._select(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return GenerationResult(
            answer=answer,
            cited_chunk_ids=cited_ids,
            model=MODEL_NAME,
            finish_reason="stop",
            time_to_first_token_ms=elapsed_ms,
            total_ms=elapsed_ms,
        )

    def _select(self, request: GenerationRequest) -> tuple[str, list[str]]:
        settings = self.settings
        query_tokens = set(_content_tokens(request.query))
        if len(query_tokens) < settings.min_query_content_tokens:
            return SAFE_REFUSAL_TEXT, []

        candidates: list[_Candidate] = []
        # Identical overlapping chunks are one piece of evidence, not a margin.
        # Keep only the strongest deterministic source for each exact sentence.
        by_span: dict[str, _Candidate] = {}
        for item in sorted(
            request.context,
            key=lambda chunk: (chunk.rank, chunk.chunk.id),
        ):
            if not _valid_retrieval_score(item):
                continue
            if item.full_score + settings.score_epsilon < settings.min_full_score:
                continue
            if not _SAFE_CITATION_ID.fullmatch(item.chunk.id):
                continue

            for sentence_index, span in enumerate(_sentences(item.chunk.text)):
                if len(span) > settings.max_sentence_chars:
                    continue
                sentence_tokens = set(_content_tokens(span))
                if len(sentence_tokens) < settings.min_sentence_content_tokens:
                    continue
                if _contradicts_query(query_tokens, sentence_tokens):
                    continue
                if _misses_required_quantity(query_tokens, sentence_tokens):
                    continue
                matched = query_tokens.intersection(sentence_tokens)
                coverage = len(matched) / len(query_tokens)
                specificity = len(matched) / len(sentence_tokens)
                normalized_full_score = min(1.0, max(0.0, (item.full_score + 1) / 2))
                selection_score = (
                    coverage
                    + settings.full_score_weight * normalized_full_score
                    + settings.specificity_weight * specificity
                )
                candidate = _Candidate(
                    span=span,
                    chunk_id=item.chunk.id,
                    rank=item.rank,
                    sentence_index=sentence_index,
                    full_score=item.full_score,
                    query_coverage=coverage,
                    selection_score=selection_score,
                )
                previous = by_span.get(span)
                if previous is None or _candidate_key(candidate) < _candidate_key(
                    previous
                ):
                    by_span[span] = candidate

        candidates.extend(by_span.values())
        if not candidates:
            return SAFE_REFUSAL_TEXT, []
        candidates.sort(key=_candidate_key)
        best = candidates[0]
        if best.query_coverage + settings.score_epsilon < settings.min_query_coverage:
            return SAFE_REFUSAL_TEXT, []

        if len(candidates) > 1:
            margin = best.selection_score - candidates[1].selection_score
            if margin + settings.score_epsilon < settings.min_selection_margin:
                return SAFE_REFUSAL_TEXT, []

        # The citation is a prefix so ``best.span`` remains an exact contiguous
        # substring of the evidence and of the answer.  This preserves every
        # number and negation by construction and remains compatible with the
        # existing citation-aware groundedness validator.
        answer = f"[chunk:{best.chunk_id}] {best.span}"
        if (
            not best.span
            or best.span
            not in request.context[
                next(
                    index
                    for index, item in enumerate(request.context)
                    if item.chunk.id == best.chunk_id
                )
            ].chunk.text
        ):
            return SAFE_REFUSAL_TEXT, []
        return answer, [best.chunk_id]
