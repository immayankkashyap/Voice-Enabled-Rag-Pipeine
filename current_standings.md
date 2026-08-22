# Current standings — measured Day 1/2 + candidate Day 3

Generated from real runs on 2026-08-21/22. The Day 3 local/ElevenLabs harness is
now implemented. Historical Sarvam/Groq numbers below remain real; the new
ElevenLabs complete path is protocol/unit tested but has not been measured
against a real Free account in this workspace.

## Status

| Area | Result |
|---|---|
| Dataset profile | 3,500 validation records and 34,944 passage slots across all 14 MSMARCO-XI target languages |
| STT | Historical: 10 real Sarvam trials. Candidate: cost-gated ElevenLabs Scribe v2 Realtime wrapper and one-turn WebSocket implemented; real run pending |
| Generation | Historical: 10 real Groq/Qwen trials. Candidate: local exact-span `local/extractive-v1` with no API call or paid fallback |
| Chunking | Real token-level Late Chunking plus separate recursive-character baseline completed |
| Indexing | Fresh end-to-end 300-passage multilingual build completed |
| Retrieval | Real 25-query benchmark completed against the saved paired FAISS index |
| Guardrails | Conservative relevance/refusal and sentence-level groundedness gates implemented |
| Browser demo | Same-origin `/demo` push-to-talk UI with local PCM16/16 kHz resampling, explicit manual stop, no CDN, and client-receipt clocks |
| Verification | 121 offline tests pass across ElevenLabs cost/protocol gates, browser harness, extractive selection, orchestration, benchmark quality gates, and existing components; real provider and external live-service latency verification remain pending |
| Submission security | **Blocked:** the pre-existing Git history contains a tracked `.env` with Sarvam/Groq credentials; the working copy now records it as deleted, but both keys must be revoked and the history purged before any push |

## Submission security blocker

The repository's pre-existing `HEAD` tracked `.env` with non-empty Sarvam and
Groq credentials. The working file has been moved to the ignored local backup
`.env.local.backup`, so the next commit can remove `.env` from the current tree.
That does not erase earlier commits. Before publishing, revoke/rotate both keys,
purge `.env` from every Git revision with a history-rewrite tool such as
`git filter-repo`, coordinate the forced update with every collaborator, and run
a secret scanner over the rewritten history. Do not submit this repository until
those checks pass.

## Dataset and finalized chunking

The bounded profile sampled 250 records per language. Translated passages are
mostly short answer/factoid material: median 96 Jina tokens, P70 117, P90 164,
P95 194, P99 266.57, and maximum 3,379. Median character length is 298 and P95
is 586. The long-prose tail is real but rare.

Observed target scripts cover Bengali, Gujarati, Devanagari, Kannada,
Malayalam, Odia, Gurmukhi, Tamil, Telugu, and Arabic, across Assamese, Bengali,
Gujarati, Hindi, Kannada, Malayalam, Marathi, Nepali, Odia, Punjabi, Sanskrit,
Tamil, Telugu, and Urdu. Required top-level fields were present in all 3,500
records; `Eng_Answer` was present in 99.6%.

Final parameters:

- Late: 8,192-token document window, 192-token chunks, 32-token overlap.
- Naive: 600-character chunks, 100-character overlap.

The 192-token target preserves nearly all typical passages while splitting the
tail, and the observed maximum fits Jina's document window. The selected native
Jina v3 model exposes final-layer token states before pooling and supports both
128- and 1,024-dimensional Matryoshka embeddings.

The embedding model is pinned to commit
`d18862d9a48706220815554fac3ebb4dfa46fc28`; the build stores that revision in
the index manifest and runtime startup fails if model or revision provenance
differs. PyTorch, Transformers, Sentence Transformers, and PEFT are also pinned
to the versions used by the bundle.

## Measured service latency

All percentiles below retain every successful trial; none is a hand-picked best
run. Sarvam and Groq each had 10/10 successful trials.

| Stage | Mean | P50 | P70 | P100 |
|---|---:|---:|---:|---:|
| Sarvam connection | 142.13 ms | 133.75 ms | 140.75 ms | 214.34 ms |
| Sarvam first partial, from call start | 986.88 ms | 978.30 ms | 983.08 ms | 1,067.07 ms |
| Sarvam final, from call start | 4,938.20 ms | 4,925.64 ms | 4,928.89 ms | 5,012.81 ms |
| Sarvam final, after audio EOF | 189.44 ms | 187.17 ms | 190.50 ms | 220.37 ms |
| Sarvam call total | 4,975.99 ms | 4,962.32 ms | 4,964.17 ms | 5,059.68 ms |
| Groq time to first token | 111.14 ms | 107.74 ms | 109.63 ms | 147.41 ms |
| Groq complete answer | 179.95 ms | 174.72 ms | 176.55 ms | 229.02 ms |
| Retrieval total (25 queries) | 26.63 ms | 23.07 ms | 32.06 ms | 35.93 ms |

The STT sample was a 4.536-second mono 16 kHz WAV. The measured configuration
used the provider's normal `speech_start`/`speech_end` manual turn protocol,
fast stream type, transcribe mode, realtime pacing, and 250 ms packets. Its
reference transcript was reproduced exactly in 10/10 trials. Verbatim remains
configurable, but it was not selected as the production default because the
same sample repeatedly contained a word-transition error in that mode.

The requested Llama 3.x model IDs are unavailable to this Groq account. The
current served default is `qwen/qwen3.6-27b` with reasoning disabled, a 96-token
cap, a reused async client, and the account's default service tier. It returned
the same complete, correctly cited synthetic-fixture answer in 10/10 trials.
Truncated, uncited, or unknown-citation output now fails closed. This fixture
validates the generation boundary and latency; it is not retrieval-quality
evidence.

## Candidate Free demo path — implemented, not yet measured

The runtime now uses ElevenLabs only for STT and local computation for every
subsequent stage:

```text
PCM16/16 kHz → Scribe v2 Realtime manual commit → Jina query embedding
→ 128-d FAISS candidates → exact 1,024-d rerank → extractive exact span
→ citation-aware groundedness gate
```

`local/extractive-v1` considers all accepted retrieval chunks, requires
conservative lexical, normalized-score, and ambiguity-margin gates, and emits
one verbatim sentence with a deterministic chunk citation. It therefore
preserves numbers and negations by construction. Weak, malformed, or competing
evidence produces the canonical refusal. It makes no Groq call and has no cloud
generation fallback.

Additional fail-closed checks reject high/low, before/after, and increase/decrease
polarity conflicts across the bundled Hindi/Tamil/Urdu path, and require an
auditable number for quantity questions. These rules prevent known failure
classes; they do not prove general semantic correctness.

The ElevenLabs boundary checks the subscription endpoint before issuing a
single-use Scribe token. It requires an active Free tier, disabled extension/
overage indicators, no invoice, an explicit operator acknowledgement that PAYG
and Auto Top Up are off, exact origin policy, a 15-second audio cap, one active
session, and process-local daily session/token caps. Quota/billing uncertainty
fails closed. Process counters reset on restart and cannot replace account-side
billing controls. The demo key must therefore be restricted and expiring, with
an account-side hard credit quota; it needs Realtime Scribe token creation plus
read-only subscription-preflight permission, but no TTS/generation permission.

The WebSocket accepts `start` JSON, binary mono PCM16 frames, and `end` JSON. It
uses manual commit rather than VAD silence, streams partials for display only,
waits for the provider's committed transcript, then runs local RAG. A partial
transcript is never released as a final query or answer. The saved bounded index
currently covers Hindi, Tamil, and Urdu, so another requested language is
rejected before the service issues an STT token.

Before any provider work, a voice turn must pass a constant-time demo-token
check, an exact Origin allowlist, and process-local per-IP and token-identity
rate limits. Binary PCM is rejected before queueing if it is empty, odd-byte,
larger than 16 KB, or would exceed the configured cumulative duration. The
production command must also set the ASGI WebSocket frame limit to 16 KB. The
Free-tier preflight and STT client are created lazily only after the first valid
frame, so malformed or unauthenticated traffic cannot consume included quota.

Official references: [Realtime event
flow](https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime),
[manual commit](https://elevenlabs.io/docs/eleven-api/guides/how-to/speech-to-text/realtime/transcripts-and-commit-strategies),
[single-use tokens](https://elevenlabs.io/docs/api-reference/tokens/create), and
[subscription fields](https://elevenlabs.io/docs/api-reference/user/subscription/get?explorer=true).

## Guardrails

The relevance gate rejects invalid evidence and requires both query-token
coverage and a normalized full-vector score. Minimum-only evidence is treated
as ambiguous, and configurable scope violations map to an off-topic refusal.
The groundedness gate requires every factual sentence to cite known chunks and
meet an 80% lexical-support threshold, with exact checks for numbers and English
negations. Safe refusals cannot be mislabeled as answered responses.

These deterministic gates are intentionally conservative rejection signals,
not proof of semantic truth. Their thresholds still need calibration on a held-
out multilingual set before a production answer endpoint can be enabled.

## Chunking comparison

Six real Hindi passages spanning 12 to 2,449 tokens produced 24 chunks with
each strategy. Late chunks averaged 140.71 tokens / 438.21 characters and took
3,316.26 ms including transformer inference. Naive chunks averaged 139.96
tokens / 435.96 characters and splitting took 0.135 ms.

For ordinary sub-192-token passages, the visible chunk text is identical. On
the 2,449-token repetitive outlier, boundaries are very similar; the meaningful
difference is that Late Chunking pooled each span only after whole-document
contextualization. This comparison does not by itself establish a retrieval
quality gain.

## Index build

The reproducible subset uses the first 10 validation queries each from Hindi,
Tamil, and Urdu: 30 queries, 300 passages, and three distinct scripts. It is
bounded because a full multi-gigabyte, all-language corpus build is not suitable
for this local iteration.

- Documents/chunks indexed: 300 / 300.
- Chunk tokens: mean 84.69, range 20–168.
- Fresh end-to-end wall time: 194.45 s.
- Dataset streaming: 174.83 s; model load: 8.33 s; late chunk/embed: 10.33 s;
  FAISS add/save: 9.29 ms.
- Paired FAISS files: 1,463,711 bytes; full saved payload: 3,053,864 bytes
  (2.91 MiB), including exact vectors and chunk metadata.
- Stage 1 is a 128-d HNSW inner-product index. Stage 2 reranks its top 50
  candidates exactly with aligned normalized 1,024-d vectors.

Torch/Jina and FAISS have a reproducible native-runtime collision on this macOS
environment during index construction. `build_index.py` isolates acquisition,
embedding, and FAISS build phases in clean processes; the final one-command
fresh build above validates that path.

## Retrieval quality check

The benchmark measured 25 balanced Hindi/Tamil/Urdu MSMARCO-XI queries after
two excluded warmups. Strict same-language dataset-selected passage hit@3 was
7/25 (28%). Examples are semantically sensible but often rank another language's
translation above the query language.

The online pipeline now overfetches, performs full-dimensional reranking, then
filters the requested language before taking final-k so cross-language
translations cannot crowd out same-language rows. A fresh diagnostic over all
30 source queries measured P50/P70/P100 local pipeline latency of
43.57/43.89/46.07 ms and selected-passage hit@1/3/5 of 5/11/12. Conservative
gates answered only 2/30 and refused 28/30. This paired source-query diagnostic
is not held-out answer-quality evidence; the current 300-passage subset remains
a latency smoke test, not a quality claim. A larger unseen multilingual,
answer-oracle-labelled evaluation is still required.

## 200 ms budget

The hard final-output target is infeasible with the measured providers, even
after replacing VAD silence waiting with correct manual turn endpointing:

- Sarvam finalization alone reaches **220.37 ms P100** after audio EOF, already
  20.37 ms beyond the entire SLA before retrieval or generation.
- At the stage medians, Sarvam finalization plus retrieval consumes
  187.17 + 23.07 = **210.24 ms**, leaving **-10.24 ms** for generation.
- Adding the measured complete-answer median gives a stage-budget sum of
  187.17 + 23.07 + 174.72 = **384.96 ms**.
- Adding the three measured maxima gives **485.32 ms**. This is a conservative
  budget sum, not a claimed end-to-end P100, because the stages were benchmarked
  independently as required before Day 3 wiring.
- From voice-stream start, Sarvam's final transcript alone is 4,925.64 ms at
  P50 for this 4.536-second utterance. A sub-200 ms metric must therefore be
  defined from detected end-of-speech, not from the instant the user starts
  speaking, even with different providers.

Speculative retrieval/generation can hide work during speech, but correctness
requires buffering it until the provider's final transcript exactly validates
the speculative query and index version. Any revision must be cancelled and
recomputed; emitting before that check would turn a latency optimization into a
wrong-answer path. A cached/extractive acknowledgment is likewise not counted
as the requested final answer.

The permitted ElevenLabs alternative is now wired, but no real
`ELEVENLABS_API_KEY`/Free-account benchmark has been run here. Its two reported
voice clocks have deliberately different meanings:

- `audio_eof_to_answer_ms`: ASGI receipt of the client's `end` event through
  provider manual commit, the intermediate committed-transcript send, and local
  RAG return. It excludes final answer-envelope construction, the final
  JSON/WebSocket send, subsequent network transit, and UI rendering.
- `first_audio_to_answer_ms`: ASGI receipt of the first non-empty PCM frame
  through local RAG return. It includes server-observed utterance streaming, but
  also includes the lazy Free-tier preflight, Scribe-token issuance, and provider
  setup begun after the first validated frame. Earlier microphone capture/uplink,
  final answer delivery, and rendering are excluded.

The current `target_met` field applies only to the post-EOF clock. A literal
speech-start-to-answer clock necessarily exceeds 200 ms whenever the spoken
question itself exceeds 200 ms. The real benchmark must report both clocks,
plus a browser-anchored user-perceived clock if available, across the complete
trial set with P100 as the maximum. **Complete-path P100 under 200 ms is
unverified and must not be claimed until that benchmark artifact exists.**

ElevenLabs' [current pricing](https://elevenlabs.io/pricing/api?price.section=speech_to_text)
lists a Free Scribe v2 Realtime allowance, but it remains a metered service and
terms can change. Its official [Free-plan publishing
guidance](https://help.elevenlabs.io/hc/en-us/articles/13313564601361-Can-I-publish-the-content-I-generate-on-the-platform)
says Free output is non-commercial and published output requires attribution
using `elevenlabs.io` or `11.ai`. The hackathon demo/posts must include that
attribution and must not be reused commercially without an appropriate license.

## Evidence artifacts

- `data/dataset_profile/profile.json` and `data/dataset_profile/PROFILE.md`
- `data/dataset_profile/passage_length_distribution.png`
- `data/chunking_comparison.json`
- `data/index_build_report.json`
- `data/faiss_index/manifest.json`
- `data/retrieval_benchmark.json`
- `data/stt_benchmark.json`
- `data/generation_benchmark.json`
- `data/stt_sample.wav`
- `scripts/benchmark_voice_pipeline.py` (writes the real integrated artifact only
  after a Free-account run; no mocked result is committed as evidence)
- `scripts/benchmark_live_websocket.py` (black-box client-receipt benchmark;
  requires WSS/HTTPS, 30 content-distinct WAVs, 30 distinct transcript
  references, and answer/refusal oracles before marking live evidence eligible)
