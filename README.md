# Voice-Enabled RAG Pipeline

HH Goa 2026, Task 2. A spoken or text question is routed through real
multilingual retrieval, deterministic guardrails, latency-budgeted answer
generation, and typed FastAPI responses.

Repository: <https://github.com/immayankkashyap/Voice-Enabled-Rag-Pipeine>

## Live status

Temporary verified demo: <https://photography-permanent-practices-components.trycloudflare.com>

The URL above is a Cloudflare Quick Tunnel to the verified local process. It
supports HTTPS and WSS, but it has no uptime guarantee and ends when that process
stops. A durable Hugging Face deployment was attempted on 2026-08-22 and the
platform returned HTTP 402 because this account is not eligible for Gradio or
Docker compute Spaces. No Railway, Render, Fly.io, or VPS credential is available
in this workspace. The reproducible Hugging Face custom-Python configuration is
kept in `.space/` for an account with compute access.

Live verification through the URL above:

- `GET /health`: HTTP 200, `rag_ready=true`, 300 vectors, Sarvam selected.
- `POST /rag`: grounded Hindi answer with a real chunk citation, HTTP 200,
  95.283 ms server-measured total.
- `WSS /ws/voice-rag`: one real 4.536-second WAV completed through Sarvam with
  25 partials, a committed transcript, and a structured RAG refusal. Client
  end-send-to-answer was 523.287 ms; first-audio-to-answer was 5,024.795 ms.

The voice result is deliberately not described as sub-200 ms. Its complete
report is `data/live_websocket_smoke.json`.

## Architecture

```text
Text /rag ───────────────┐
                        v
Voice PCM16 ─> Sarvam STT ─> input-safety gate
                             ─> Jina v3 query embedding
                             ─> 128-d FAISS candidate search
                             ─> exact 1,024-d rerank
                             ─> CRAG relevance classification
                                  ├─ Incorrect/Ambiguous: structured refusal
                                  └─ Correct: generation
                                       ├─ exact cited evidence fast path
                                       └─ Groq/Qwen fallback only with ≥350 ms budget
                             ─> citation-aware groundedness gate
                             ─> RAGResponse or ErrorResponse
```

The active STT provider is intentionally **Sarvam** (`app/stt.py`). ElevenLabs
was tested during development but its configured account rejected live
authentication, so that unused path was removed rather than left as an
accidental fallback.

The bundled index contains 300 real Hindi, Tamil, and Urdu MSMARCO-XI passages.
It was built with `jinaai/jina-embeddings-v3-hf` pinned to the revision stored in
`data/faiss_index/manifest.json`. Runtime startup rejects mismatched model
provenance.

### Guardrails

- `InputSafetyGuardrail` runs before retrieval and deterministically rejects a
  small audited set of prompt-injection, secret-extraction, credential-theft,
  weapon, physical-harm, unsafe-control-character, and sexual-minor patterns.
- `RelevanceGuardrail` classifies every retrieved chunk as `correct`,
  `ambiguous`, or `incorrect` using normalized similarity plus lexical coverage.
  All-incorrect and ambiguous sets skip generation.
- `GroundednessGuardrail` requires every factual sentence to cite a known chunk,
  checks lexical support, numbers, and English negation, and fails closed. An
  optional judge can run only for ambiguous fast checks and only when sufficient
  latency budget remains; no judge is configured by default.

Unsafe input, off-topic input, ambiguous retrieval, invalid generated output,
insufficient generation budget, and ungrounded output have distinct refusal
reasons. STT, retrieval, and provider failures become sanitized typed
`ErrorResponse` objects; raw exceptions and generated-but-rejected text are not
returned.

## Measured performance

`data/benchmark_report.json` is the final 30-query text-RAG report. It ran the
actual warmed runtime over 10 semantic MSMARCO-XI questions translated into
Hindi, Tamil, and Urdu. These queries are real but also contributed to the
bounded demo index, so this is latency evidence—not held-out retrieval-quality
evidence.

| Stage | P50 | P70 | P100 |
|---|---:|---:|---:|
| Input safety | 0.016 ms | 0.018 ms | 0.028 ms |
| Query embedding | 22.302 ms | 29.406 ms | 32.067 ms |
| Retrieval stage 1 | 0.063 ms | 0.066 ms | 0.075 ms |
| Retrieval stage 2 | 0.035 ms | 0.039 ms | 0.047 ms |
| Retrieval total | 22.462 ms | 29.577 ms | 32.235 ms |
| Relevance guardrail | 0.157 ms | 0.192 ms | 0.244 ms |
| Generation | 0.000 ms | 0.000 ms | 0.064 ms |
| Groundedness guardrail | 0.000 ms | 0.000 ms | 0.098 ms |
| **Full text pipeline** | **22.681 ms** | **29.809 ms** | **32.470 ms** |

All 30 attempts completed without an error and met the 200 ms text target. The
outcomes were 2 exact grounded answers and 28 explicit refusals: 14
`no_relevant_context`, 12 `ambiguous_retrieval`, and 2
`latency_budget_exhausted`.

This low P100 has a real quality/coverage tradeoff. Before optimization, the
same workload had P50/P70/P100 of 37.830/48.522/361.004 ms; Groq generation
alone reached 329.133 ms P100 and produced three invalid outputs. The single
optimization added a conservative exact-evidence fast path and prevented a Groq
fallback when less than its measured 350 ms budget remained. The final run used
the exact path twice, skipped two Groq calls with visible budget refusals, and
made zero Groq calls. The pre-optimization evidence is preserved in
`data/benchmark_report_baseline.json`.

Historical component artifacts remain honest but have narrower scopes:

- Fresh Sarvam verification transcribed the 4.536-second fixture correctly in
  3/3 trials; post-EOF completion was 291.389, 427.076, and 299.026 ms.
- Fresh Groq/Qwen verification completed 3/3 synthetic-fixture generations;
  TTFT was 178.332, 116.470, and 116.140 ms, and total generation was 245.427,
  182.846, and 182.126 ms.
- `data/retrieval_benchmark.json` aggregates 25 real queries but retains only
  five example records; it is not the final full-pipeline benchmark.
- `data/generation_benchmark.json` contains ten non-zero trials of one synthetic
  fixture; it proves the provider boundary, not multi-query answer quality.

## Local setup

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Populate at least `SARVAM_API_KEY`, `GROQ_API_KEY`, and a random
`VOICE_DEMO_TOKEN`. Keep `.env` untracked.

Start one worker so the in-process voice rate limiter remains coherent:

```bash
uvicorn app.main:app \
  --host 127.0.0.1 --port 8000 --workers 1 --ws-max-size 16000
```

Then open <http://127.0.0.1:8000/>. The browser performs same-origin push-to-talk
capture, resamples to mono PCM16/16 kHz, and commits only after the operator
presses Stop.

## API use

Text RAG skips STT:

```bash
curl -sS http://127.0.0.1:8000/rag \
  -H 'Content-Type: application/json' \
  --data '{"query":"कितनी ट्रम्प प्रशासन की जांच चल रही हैं","language_code":"hi"}'
```

Every text response includes populated timings for input safety, query
embedding, both retrieval stages, total retrieval, relevance, generation,
groundedness, output validation, and total.

The one-turn voice protocol is:

1. Connect to `/ws/voice-rag` with an allowlisted `Origin`.
2. Send `{"event":"start","language_code":"hi","demo_token":"..."}`.
3. Wait for `ready`, then send bounded binary PCM16/16 kHz frames.
4. Send `{"event":"end"}` after the final frame.
5. Receive partial transcripts, exactly one committed transcript, then `answer`
   or a structured error.

Only the committed Sarvam transcript reaches retrieval. Partials are display
events and are never used speculatively as final questions.

## Verification and benchmarks

The final offline suite contains 106 passing tests.

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q app scripts tests

python scripts/test_stt.py data/stt_sample.wav --trials 3
python scripts/test_generation.py --trials 3
python scripts/test_guardrails.py
python scripts/smoke_rag.py
python scripts/smoke_voice_ws.py
python scripts/benchmark.py
```

Black-box WSS smoke test against a running deployment:

```bash
python scripts/benchmark_live_websocket.py data/stt_sample.wav \
  --ws-url wss://HOST/ws/voice-rag \
  --origin https://HOST \
  --trials 1 --language-code hi --latency-only \
  --output data/live_websocket_smoke.json
```

The live voice runner also supports a quality-gated manifest. Submission-grade
voice evidence requires at least 30 content-distinct recordings, 30 distinct
reference questions, explicit answer/refusal oracles, secure transport, zero
failures, and literal observed-max P100. The committed one-record smoke report
does not meet that workload gate.

## Project layout

```text
app/                    FastAPI, STT, retrieval, generation, guardrails
data/faiss_index/       Versioned 300-vector demo index
data/dataset_profile/   Real MSMARCO-XI profile artifacts
scripts/                Build, live checks, smoke tests, benchmarks, deployment
static/                 Same-origin push-to-talk UI
tests/                  Offline deterministic test suite
.space/                 Minimal Hugging Face custom-Python deployment bundle
```

Key evidence files:

- `data/dataset_profile/profile.json`
- `data/chunking_comparison.json`
- `data/index_build_report.json`
- `data/retrieval_benchmark.json`
- `data/generation_benchmark.json`
- `data/benchmark_report.json`
- `data/benchmark_report_baseline.json`
- `data/live_websocket_smoke.json`

## Deployment

For a Hugging Face account allowed to create compute Spaces:

```bash
python scripts/deploy_hf_space.py
```

The script creates only `OWNER/voice-rag-goa-2026`, uploads the minimal 25-file
runtime bundle, stores Sarvam/Groq/demo credentials as write-only Space secrets,
and configures exact-origin and latency variables. It never uploads `.env` or
the deployment `HF_TOKEN`.

For Railway, Render, Fly.io, or a VPS, run the same single-worker Uvicorn command
on the port supplied by the platform, persist `data/faiss_index/`, set the
variables from `.env.example`, and permit HTTPS WebSocket upgrades.

## Security and known limitations

- The original Git history contained `.env` with Sarvam and Groq credentials.
  This submission branch was rewritten on 2026-08-22 to remove every `.env`
  revision, and the final tracked tree contains none of the active credential
  values. History removal does not revoke leaked credentials; rotate both
  provider keys before treating the deployment as secure.
- The input-safety patterns are intentionally lightweight and auditable, not a
  complete moderation system.
- Relevance and groundedness are conservative lexical/similarity rejection
  signals, not proofs of truth. Answer coverage is low on the bounded index.
- The 30-query latency set is not held out, and the live voice artifact contains
  only one recording.
- The final text P100 is below 200 ms; real voice post-utterance latency is not.
- The current live URL is temporary until a durable deployment account is
  provided.
