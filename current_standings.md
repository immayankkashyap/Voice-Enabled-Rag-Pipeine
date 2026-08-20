# Current standings

## Completed in phase 1

- Requested directory and Python-package scaffold is present.
- `.env.example`, `.gitignore`, package markers, `data/`, and `tests/` are present.
- API/service boundaries use strict Pydantic models with explicit answered/refused,
  relevance, groundedness, citation, retrieval-score, and per-stage timing fields.
- Inherited mocks that fabricated transcripts, embeddings, answers, or benchmark
  results were replaced with explicit not-implemented failures.
- `scripts/inspect_dataset.py` can stream bounded per-language MSMARCO-XI Parquet
  samples and write JSON, Markdown, CSV, and optional PNG profile artifacts.
- The profiler covers translated/original text lengths, observed target-language
  values, query types, translation settings, relevance labels, schema, field
  coverage, mismatched arrays, and empty passages.
- Hub manifest lookup, dataset streaming, and optional tokenizer loading use
  tenacity retries and record elapsed time.
- Dataset helper tests run without third-party dependencies; the complete suite
  also exercises the FastAPI health/not-ready contract after dependencies install.

## Deliberately not finalized

- Chunk/window/token limits: wait for the generated dataset profile and the
  selected embedding tokenizer's length distribution.
- Embedding model and MRL dimensions.
- Relevance and groundedness thresholds.
- Any latency claim.

## Existing placeholders to replace next

- `app/chunking.py`: not real Late Chunking yet.
- `app/indexing.py` and `app/retrieval.py`: mock/random embeddings.
- `app/generation.py`: mock response instead of a validated Groq response model.
- `app/guardrails.py`: hard-coded relevance and groundedness behavior.
- `scripts/build_index.py` and `scripts/benchmark.py`: synthetic data only.

These placeholders must not be used as submission evidence or described as a
working end-to-end pipeline.
