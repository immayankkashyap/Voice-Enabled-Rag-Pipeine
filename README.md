# Voice-Enabled RAG Pipeline

HH Goa 2026 Hackathon, Task 2: spoken question → Sarvam streaming STT →
Late Chunking/FAISS retrieval → CRAG-style relevance gate → Groq-hosted Llama 3
generation → deterministic groundedness gate → structured response.

## Status

Phase 1 is the project scaffold and dataset-inspection tooling. The application
modules exist, but several still contain explicit placeholders and are **not yet a
production pipeline**. No latency target is claimed and no chunk size has been
selected yet.

The first evidence-gathering step is profiling
[`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI):
passage/query/answer lengths, sampled language coverage, query types, translation
metadata, schema consistency, selected-passage labels, and English-to-translation
length ratios.

## Project layout

```text
.
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI REST and WebSocket harness
│   ├── stt.py              # Sarvam streaming STT wrapper
│   ├── chunking.py         # Late Chunking and naive comparison baseline
│   ├── indexing.py         # Embeddings and FAISS build/load
│   ├── retrieval.py        # MRL first pass and full-dimension rerank
│   ├── generation.py       # Typed Groq/Llama generation boundary
│   ├── guardrails.py       # Relevance and groundedness gates
│   └── schemas.py          # Pydantic API/domain schemas
├── scripts/
│   ├── inspect_dataset.py  # Bounded streaming MSMARCO-XI profiler
│   ├── build_index.py      # Offline index builder (placeholder)
│   └── benchmark.py        # P50/P70/P100 benchmark (placeholder)
├── data/                   # Generated profiles, indices, and benchmark reports
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

## Local setup

Python 3.11 or 3.12 is recommended for broad PyTorch and FAISS wheel support.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Populate `SARVAM_API_KEY` and `GROQ_API_KEY` only in `.env`. `HF_TOKEN` is
optional for the public dataset, but can improve Hub rate limits. Never commit
the populated file.

## Profile MSMARCO-XI

The Hub repository is tens of gigabytes. The profiler does not download it in
full: it opens each language's Parquet file directly with an `hf://` URI, enables
Hugging Face streaming, and stops at an independent per-language record cap.
Validation is the default split because it includes all advertised language
files and is smaller than train.

Preview the exact files without installing dependencies or using the network:

```bash
python scripts/inspect_dataset.py --dry-run
```

Run the default bounded profile (250 query records per language):

```bash
python scripts/inspect_dataset.py
```

Run a faster two-language smoke profile without plots:

```bash
python scripts/inspect_dataset.py \
  --languages hi bn \
  --max-records-per-language 25 \
  --no-plots \
  --output-dir data/dataset_profile_smoke
```

The default is a deterministic head sample, which minimizes remote reads but can
reflect source ordering. For a less order-sensitive sample, use a streaming
shuffle buffer; this reads more records:

```bash
python scripts/inspect_dataset.py \
  --shuffle-buffer 2500 \
  --max-records-per-language 250
```

Once an embedding-model candidate is chosen, exact tokenizer-length evidence can
be added without changing any chunking code:

```bash
python scripts/inspect_dataset.py --tokenizer YOUR_EMBEDDING_MODEL_ID
```

The command writes:

- `data/dataset_profile/profile.json`: full machine-readable report, including
  observed schema, metadata distributions, quality checks, and external-call timing.
- `data/dataset_profile/PROFILE.md`: compact human-readable summary.
- `data/dataset_profile/language_summary.csv`: per-language sample and passage stats.
- `data/dataset_profile/length_summary.csv`: P0/P25/P50/P70/P75/P90/P95/P99/P100
  for every recorded length metric.
- `data/dataset_profile/passage_length_distribution.png`: P99-clipped plots that
  preserve outliers in the JSON/CSV while keeping the visualization readable.

Language samples are capped independently. Their counts verify coverage; they do
not estimate full-corpus language prevalence. Failed or unavailable language files
are retained in the report instead of silently disappearing.

## Tests

The inspection helpers and dry-run path use only the standard library, so that
subset can be checked before installing the heavier ML stack:

```bash
python -m unittest tests.test_inspect_dataset -v
```

After installing `requirements.txt`, run the complete scaffold suite:

```bash
python -m unittest discover -s tests -v
python -m compileall -q app scripts tests
```

## Required implementation sequence

1. Generate and review the dataset profile; document the evidence used for chunk
   boundary/window candidates.
2. Implement real token-contextual Late Chunking plus a fixed/recursive baseline,
   then benchmark retrieval quality and preprocessing cost for both.
3. Build persistent full-dimensional and MRL-truncated FAISS artifacts; keep all
   embedding and vector-search work off the async event loop.
4. Replace scaffold mocks with typed, timed, retried Sarvam/Groq/embedding clients
   and explicit recovery behavior.
5. Implement deterministic CRAG relevance and groundedness gates with explicit
   refusal states.
6. Benchmark a real multilingual query set and report every P50/P70/P100 stage and
   end-to-end result, including outliers.
7. Add container/deployment configuration and publish a live Railway, Render,
   Fly.io, or VPS endpoint only after the same artifact set passes locally.

The `<50 ms` retrieval and `<200 ms` end-to-end figures are targets to measure,
not values to assume. Network STT plus generated output may make the latter
physically unattainable; the benchmark must report observed results honestly.
