# Voice-Enabled RAG Pipeline

HH Goa 2026 Hackathon, Task 2: spoken question → ElevenLabs Scribe v2
Realtime STT → Late Chunking/FAISS retrieval → conservative relevance gate →
local extractive answer → deterministic groundedness gate → structured response.

## Status

Day 1 and Day 2 components are independently implemented and measured. A Day 3
FastAPI harness now wires the saved local index, two-stage retriever, extractive
generator, guardrails, and a one-turn ElevenLabs WebSocket. The ElevenLabs path
is protocol/unit tested but has not run against a real account in this workspace,
so this is **not yet a benchmark-verified production pipeline**.

Historical Sarvam and Groq measurements are preserved: Sarvam post-audio
finalization reached 220.37 ms P100, and a complete Groq answer reached 229.02 ms
P100. The candidate path replaces them with ElevenLabs Free allowance and
`local/extractive-v1`, but **complete P100 below 200 ms remains unverified** until
the integrated real-audio benchmark passes. See `current_standings.md`; no lower
latency is claimed from mocked provider events, partial transcripts, or cached
answers.

> **Do not publish the current Git history yet.** The pre-existing repository
> tracked `.env` with Sarvam/Groq credentials. The working tree now deletes that
> file and retains an ignored local backup, but the exposed keys must be revoked
> and `.env` must be purged from all prior commits before submission. See
> `current_standings.md`.

## Strict zero-cost demo mode

The live harness makes no Groq call and has no paid-provider fallback. Answer
generation is deterministic and local: it selects one exact evidence sentence,
preserves numbers and negations verbatim, adds a known chunk citation, and
refuses weak or ambiguous evidence. Jina embedding inference, FAISS search,
extraction, and both guardrails run locally.

ElevenLabs is still a metered cloud API. “Free” means staying inside the
account's included allowance, not that the endpoint is intrinsically unmetered.
Before every demo:

1. Confirm the workspace is on the **Free** tier, remove any payment method where
   possible, and disable usage-based billing/PAYG and Auto Top Up in the
   dashboard.
2. Create a restricted, expiring demo API key with an account-side hard credit
   quota below the remaining included allowance. It needs permission to create
   Realtime Scribe single-use tokens and read the user subscription for the
   safety preflight; give it no TTS or generation permissions. The provider-side
   quota is authoritative because process counters reset on restart.
3. Only then set
   `ELEVENLABS_FREE_TIER_ACKNOWLEDGEMENT=I_CONFIRM_NO_PAYG_OR_AUTO_TOP_UP`.
4. Generate a long random `VOICE_DEMO_TOKEN`; the WebSocket rejects missing or
   incorrect tokens before it loads a request runtime or contacts ElevenLabs.
   Keep it out of source, posts, and browser logs.
5. Keep the `.env.example` exact-origin policy, 16 KB frame limit, 15-second
   turn, one-concurrent-session, per-IP/token rate limit, and daily token/session
   ceilings. These counters are process-local and reset on restart; they are a
   second guard, not a billing guarantee. Use one worker unless the limiter is
   moved to shared storage.
6. Stop if subscription preflight cannot prove an active Free tier with zero
   extension/overage/invoices, or if ElevenLabs reports authentication, quota,
   billing, or rate-limit failure. The harness fails closed and never switches
   to Sarvam, Groq, or another paid service.

The checks use ElevenLabs' official [subscription
endpoint](https://elevenlabs.io/docs/api-reference/user/subscription/get?explorer=true),
where `max_credit_limit_extension=0` means usage-based billing is disabled. The
[current API pricing page](https://elevenlabs.io/pricing/api?price.section=speech_to_text)
lists a Free Scribe v2 Realtime allowance, but limits and terms can change; verify
them on demo day.

Free-plan licensing is also restrictive. ElevenLabs states that Free output is
non-commercial and that published output requires attribution using
`elevenlabs.io` or `11.ai`; see its official [publishing and license
guidance](https://help.elevenlabs.io/hc/en-us/articles/13313564601361-Can-I-publish-the-content-I-generate-on-the-platform).
Treat the hackathon demo as non-commercial, include the attribution in every
published demo/video title or caption, and obtain separate permission or use a
suitable license if the submission or later deployment is commercial.

## What was adapted from `voice-HHGoa`

The low-latency path borrows the useful architectural ideas from the sibling
project: keep the provider master key on the server, warm and reuse the local
model/index, use manual end-of-turn signalling, and use a local extractive
answer path for the demo. It does not copy that project's 1.6-second VAD wait,
650 ms post-commit delay, source-query index boost, or its RAG-only latency claim.
Partials may start buffered local work, but an answer is reused only when the
committed transcript is an exact match; a revision cancels and recomputes it.

## Voice WebSocket protocol

Connect to `/ws/voice-rag` and send exactly one explicitly delimited turn:

1. Client JSON:
   `{"event":"start","language_code":"hi","demo_token":"..."}` over TLS.
   The current bounded
   index covers `hi`, `ta`, and `ur`; unsupported languages are rejected before
   an STT token can consume included quota. The server accepts
   the transport first, but the turn proceeds only when the local runtime and
   demo-token, rate-limit, and origin checks pass.
2. Server JSON: `{"event":"ready", ...}` specifying mono signed PCM16 little
   endian at 16 kHz and manual commit.
3. Client binary messages: consecutive PCM16 frames. The server may emit
   `{"event":"partial_transcript","text":"..."}` for display only.
4. Client JSON: `{"event":"end"}` only after the final audio frame. The provider
   proxy sends an official `input_audio_chunk` with `commit: true`, then waits
   for a non-empty `committed_transcript`. A partial is never accepted as final.
5. Server JSON: `{"event":"answer","payload":...}` after local retrieval,
   extraction, and groundedness validation, followed by close code 1000.

The implementation follows the official ElevenLabs [Realtime WebSocket event
flow](https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime),
[manual commit guidance](https://elevenlabs.io/docs/eleven-api/guides/how-to/speech-to-text/realtime/transcripts-and-commit-strategies),
and [15-minute single-use token
flow](https://elevenlabs.io/docs/api-reference/tokens/create). The API key stays
server-side.

## Latency definitions

Two clocks must be reported; they are not interchangeable:

- `audio_eof_to_answer_ms` is the post-speech processing metric. Its start is
  when ASGI receives the client's `end` event; its end is when the local RAG
  coroutine returns. It includes queue drain, ElevenLabs manual commit/
  finalization, the intermediate committed-transcript WebSocket send, query
  embedding, both FAISS stages, extraction, and guardrails. It excludes final
  answer-envelope construction, final JSON/WebSocket send, subsequent network
  transit, and UI rendering.
- `first_audio_to_answer_ms` starts when ASGI receives the first non-empty PCM
  frame and ends at the same local-RAG point. It includes the server-observed
  streamed utterance duration plus the lazy subscription preflight, Scribe-token
  issuance, and provider connection started after that first validated frame.
  Microphone capture/uplink before ASGI receives the frame and final answer
  delivery/rendering are excluded. A true user-perceived speech-start metric
  must be anchored in the browser and measured separately.

An utterance longer than 200 ms cannot satisfy a literal speech-start-to-final
answer target under 200 ms. The code's `target_met` currently evaluates the
post-EOF metric. Both post-EOF and first-audio distributions must be published,
with P100 as the maximum over the full trial set. Neither is yet measured for the
integrated ElevenLabs path, so **under 200 ms is a target, not a result**.

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
│   ├── elevenlabs_stt.py   # Free-tier-gated Scribe realtime wrapper
│   ├── stt.py              # Historical Sarvam benchmark wrapper
│   ├── chunking.py         # Late Chunking and naive comparison baseline
│   ├── indexing.py         # Embeddings and FAISS build/load
│   ├── retrieval.py        # MRL first pass and full-dimension rerank
│   ├── extractive.py       # Local exact-span zero-cost answer path
│   ├── generation.py       # Historical Groq/Qwen benchmark boundary
│   ├── guardrails.py       # Relevance and groundedness gates
│   ├── pipeline.py         # Typed local retrieval-to-answer orchestration
│   ├── runtime.py          # Reusable warmed model/index services
│   └── schemas.py          # Pydantic API/domain schemas
├── scripts/
│   ├── inspect_dataset.py  # Bounded streaming MSMARCO-XI profiler
│   ├── build_index.py      # Reproducible offline index builder
│   ├── benchmark_voice_pipeline.py # Provider + component benchmark
│   └── benchmark_live_websocket.py # Client-receipt live-service benchmark
├── data/                   # Generated profiles, indices, and benchmark reports
├── static/                 # Same-origin push-to-talk browser demo
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

For the strict Free demo, populate only `ELEVENLABS_API_KEY`, the explicit
acknowledgement, and a strong `VOICE_DEMO_TOKEN`; verify the account controls
above and keep `SARVAM_API_KEY`/`GROQ_API_KEY` blank. Those two variables exist
only to reproduce historical standalone measurements. `HF_TOKEN` is optional
for the public dataset but can improve Hub rate limits. Never commit the
populated file.

## Run the harness

The bundled 300-passage Hindi/Tamil/Urdu FAISS index makes a clean local start
possible without rebuilding the dataset. After the strict Free-account checks
above and environment setup, start one warmed process:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 \
  --workers 1 --ws-max-size 16000
```

`GET /health` reports whether the local runtime is ready; `voice_ready` becomes
true only after the first authenticated, valid-audio request completes the lazy
Free-tier preflight. `POST /rag` exercises only the local text path.
`/ws/voice-rag` implements the one-turn PCM protocol above. There is no paid
provider fallback.

Open `http://127.0.0.1:8000/demo` for the bundled same-origin push-to-talk UI.
It performs browser-side mono PCM16/16 kHz resampling, sends 20 ms frames, uses
explicit Stop/manual commit, and displays both server and client-receipt clocks.
The operator types `VOICE_DEMO_TOKEN` for each turn; the page clears it after the
start message and never stores it in a URL or browser storage. No CDN, analytics,
or third-party browser script is loaded.

## Integrated voice benchmark

A positional WAV run is intentionally only a latency smoke test:

```bash
python scripts/benchmark_voice_pipeline.py data/stt_sample.wav \
  --trials 3 --latency-only \
  --output data/voice_pipeline_smoke.json
```

The submission-quality runner requires a manifest with at least 30 distinct
representative recordings, transcript references, and outcome/answer oracles.
For an answered sample, provide `expected_answer_contains`,
`expected_chunk_ids`, or `expected_document_ids`; refusals must be explicit.

```json
{
  "samples": [
    {
      "audio": "audio/question-01.wav",
      "reference": "reference transcript",
      "expected_status": "answered",
      "expected_answer_contains": ["expected fact"]
    }
  ]
}
```

```bash
python scripts/benchmark_voice_pipeline.py \
  --manifest benchmark_manifest.json \
  --trials 30 \
  --output data/voice_pipeline_benchmark.json
```

The report retains failures, uses the literal observed maximum for P100, and
separates transcript, answer-quality, latency, and workload gates. This runner
directly exercises ElevenLabs plus local RAG, but excludes client-to-ASGI
transport and final WebSocket receipt. A browser/external WebSocket benchmark is
therefore also provided:

```bash
python scripts/benchmark_live_websocket.py \
  --ws-url wss://YOUR_DEMO_HOST/ws/voice-rag \
  --origin https://YOUR_DEMO_HOST \
  --manifest benchmark_manifest.json \
  --trials 30 \
  --output data/live_websocket_benchmark.json
```

That black-box runner includes client-to-ASGI transport, server/provider work,
final WebSocket delivery, JSON/schema parsing, transcript validation, and answer
oracles. It still excludes microphone capture and browser rendering. Only WSS +
HTTPS, 30 content-distinct WAVs, 30 distinct normalized reference questions,
zero failures, and passing latency/quality gates can set its live-evidence flag.
Every real run consumes included ElevenLabs allowance, so do not run either
benchmark until the zero-charge controls in this README have been verified. The
public defaults intentionally permit fewer than 30 rapid sessions; use a
dedicated localhost/deployment benchmark process with explicitly bounded caps
that cover the 30 trials, then restore the public limits. The provider-side hard
credit quota remains authoritative.

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
4. Keep external boundaries typed, timed, retried, cost-gated, and fail-closed;
   never add an automatic paid-provider fallback.
5. Implement deterministic CRAG relevance and groundedness gates with explicit
   refusal states.
6. Benchmark a real multilingual query set and report every P50/P70/P100 stage and
   end-to-end result, including outliers.
7. Add container/deployment configuration and publish a live Railway, Render,
   Fly.io, or VPS endpoint only after the same artifact set passes locally.

The `<50 ms` retrieval and `<200 ms` end-to-end figures are targets to measure,
not values to assume. Network STT plus generated output may make the latter
physically unattainable; the benchmark must report observed results honestly.

## Submission checklist

- Revoke the exposed historical Sarvam/Groq keys, purge `.env` from every Git
  revision, and scan the rewritten history before publishing the repository.
- Run the 30-recording quality-gated component benchmark and an external live
  WebSocket/browser benchmark; do not claim sub-200 ms unless both artifacts
  pass with P100 equal to the observed maximum.
- Provide the final GitHub repository and a live working link.
- Record the 90-second team/process video and the end-to-end demo video.
- Every team member must post both videos to Instagram and X; at least one
  Instagram account must be public. Every post must include `#RAGInGoa` and the
  required `elevenlabs.io` or `11.ai` Free-plan attribution.
- Submit the [official form](https://forms.gle/MNvCjcv23Hn2Eeu58) only once,
  before **August 22, 2026 at 11:59 PM**.
