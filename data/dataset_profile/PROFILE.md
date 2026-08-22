# MSMARCO-XI dataset profile

Generated: `2026-08-21T16:47:01.252143+00:00`

This is a bounded streaming sample used to inform a later chunking decision. It is not a corpus-wide census, and **no chunk size is selected here**.

## Sampling

- Dataset: `ai4bharat/MSMARCO-XI`
- Requested revision: `main`
- Resolved revision: `bf5cdc1f26e581e519018e434db14edd1b77602b`
- Split: `validation`
- Maximum records per language: `250`
- Shuffle buffer: `0` (`0` means a deterministic head sample)
- Tokenizer: `jinaai/jina-embeddings-v3`

> Language proportions below describe independently capped samples. They must not be interpreted as the full corpus's language proportions.

## Per-language passage lengths

| Lang | Status | Records | Passages | Chars P50 | Chars P95 | Chars max | Words P50 | Words P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| as | ok | 250 | 2496 | 285.0 | 550.2 | 3497.0 | 44.0 | 88.0 |
| bn | ok | 250 | 2496 | 292.0 | 573.0 | 3718.0 | 45.0 | 90.0 |
| gu | ok | 250 | 2496 | 285.0 | 546.0 | 3350.0 | 47.0 | 93.2 |
| hi | ok | 250 | 2496 | 295.0 | 570.0 | 7956.0 | 57.0 | 111.0 |
| kn | ok | 250 | 2496 | 312.0 | 608.2 | 2540.0 | 37.0 | 73.0 |
| ml | ok | 250 | 2496 | 327.0 | 647.0 | 3412.0 | 35.0 | 70.2 |
| mr | ok | 250 | 2496 | 290.0 | 557.2 | 7385.0 | 42.0 | 83.0 |
| ne | ok | 250 | 2496 | 286.0 | 561.0 | 7538.0 | 43.0 | 84.0 |
| or | ok | 250 | 2496 | 292.0 | 574.0 | 1873.0 | 45.0 | 95.0 |
| pa | ok | 250 | 2496 | 294.0 | 578.2 | 2735.0 | 57.0 | 114.2 |
| sa | ok | 250 | 2496 | 307.0 | 601.2 | 12809.0 | 36.0 | 74.0 |
| ta | ok | 250 | 2496 | 337.0 | 666.5 | 3516.0 | 38.0 | 78.0 |
| te | ok | 250 | 2496 | 299.0 | 593.0 | 2558.0 | 38.0 | 77.2 |
| ur | ok | 250 | 2496 | 290.0 | 563.2 | 6847.0 | 62.0 | 123.2 |

## Observed query types

- `DESCRIPTION`: 2226
- `NUMERIC`: 1078
- `ENTITY`: 126
- `PERSON`: 56
- `LOCATION`: 14

## Artifacts

- `profile.json`: complete machine-readable summaries, schema, metadata, and quality checks
- `language_summary.csv`: one row per requested language
- `length_summary.csv`: percentiles for every text-length metric and language
- `passage_length_distribution.png`: P99-clipped visualization (when plotting succeeds)
