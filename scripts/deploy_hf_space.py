#!/usr/bin/env python3
"""Deploy the minimal FastAPI runtime to a Hugging Face custom-Python Space."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import CommitOperationAdd, HfApi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPACE_NAME = "voice-rag-goa-2026"

_APP_FILES = (
    "__init__.py",
    "chunking.py",
    "extractive.py",
    "generation.py",
    "guardrails.py",
    "indexing.py",
    "main.py",
    "pipeline.py",
    "retrieval.py",
    "runtime.py",
    "schemas.py",
    "stt.py",
)
_STATIC_FILES = ("app.js", "index.html", "pcm-worklet.js", "styles.css")
_INDEX_FILES = (
    "chunks.json",
    "full.faiss",
    "full_embeddings.npy",
    "manifest.json",
    "mrl.faiss",
    "source_queries.json",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--space-name", default=DEFAULT_SPACE_NAME)
    parser.add_argument("--repo-id", default=None)
    return parser


def _secret(name: str, values: dict[str, str | None]) -> str:
    value = os.getenv(name) or values.get(name) or ""
    if not value:
        raise ValueError(f"{name} is required in the environment or .env")
    return value


def _operations() -> list[CommitOperationAdd]:
    mappings: list[tuple[Path, str]] = [
        (PROJECT_ROOT / ".space/README.md", "README.md"),
        (PROJECT_ROOT / ".space/requirements.txt", "requirements.txt"),
        (
            PROJECT_ROOT / ".space/space_entrypoint.py",
            "space_entrypoint.py",
        ),
    ]
    mappings.extend(
        (PROJECT_ROOT / "app" / name, f"app/{name}") for name in _APP_FILES
    )
    mappings.extend(
        (PROJECT_ROOT / "static" / name, f"static/{name}")
        for name in _STATIC_FILES
    )
    mappings.extend(
        (PROJECT_ROOT / "data/faiss_index" / name, f"data/faiss_index/{name}")
        for name in _INDEX_FILES
    )
    missing = [str(source) for source, _ in mappings if not source.is_file()]
    if missing:
        raise FileNotFoundError("Deployment files are missing: " + ", ".join(missing))
    return [
        CommitOperationAdd(path_in_repo=target, path_or_fileobj=source)
        for source, target in mappings
    ]


def run(args: argparse.Namespace) -> int:
    values = dict(dotenv_values(args.env_file))
    hf_token = _secret("HF_TOKEN", values)
    sarvam_key = _secret("SARVAM_API_KEY", values)
    groq_key = _secret("GROQ_API_KEY", values)
    demo_token = _secret("VOICE_DEMO_TOKEN", values)

    api = HfApi(token=hf_token)
    account = api.whoami()
    owner = str(account.get("name") or "").strip()
    if not owner:
        raise RuntimeError("Hugging Face did not return an authenticated owner")
    repo_id = args.repo_id or f"{owner}/{args.space_name}"
    if not repo_id.startswith(f"{owner}/"):
        raise ValueError("--repo-id must belong to the authenticated user")

    app_url = "https://" + repo_id.replace("/", "-").lower() + ".hf.space"
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        exist_ok=True,
    )
    for key, value, description in (
        ("SARVAM_API_KEY", sarvam_key, "Sarvam streaming STT credential"),
        ("GROQ_API_KEY", groq_key, "Budget-gated Groq fallback credential"),
        ("VOICE_DEMO_TOKEN", demo_token, "Operator WebSocket demo token"),
    ):
        api.add_space_secret(
            repo_id,
            key,
            value,
            description=description,
        )

    variables = {
        "VOICE_ALLOWED_ORIGINS": app_url,
        "VOICE_REQUIRE_ORIGIN": "true",
        "VOICE_DEMO_SESSIONS_PER_MINUTE": "6",
        "VOICE_MAX_AUDIO_SECONDS": "30",
        "RAG_PRELOAD": "true",
        "RAG_DEVICE": "cpu",
        "RAG_CPU_THREADS": "2",
        "RAG_LATENCY_TARGET_MS": "200",
        "GROQ_MODEL": "qwen/qwen3.6-27b",
        "GROQ_MAX_OUTPUT_TOKENS": "96",
        "GROQ_MAX_ATTEMPTS": "2",
        "GROQ_MIN_FALLBACK_BUDGET_MS": "350",
    }
    for key, value in variables.items():
        api.add_space_variable(repo_id, key, value)

    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="space",
        operations=_operations(),
        commit_message="Deploy verified Voice RAG FastAPI runtime",
    )
    runtime = api.get_space_runtime(repo_id)
    print(
        json.dumps(
            {
                "repo_id": repo_id,
                "repo_url": f"https://huggingface.co/spaces/{repo_id}",
                "app_url": app_url,
                "commit_oid": commit.oid,
                "runtime_stage": runtime.stage,
                "hardware": runtime.hardware,
                "secret_keys": [
                    "SARVAM_API_KEY",
                    "GROQ_API_KEY",
                    "VOICE_DEMO_TOKEN",
                ],
                "variables": variables,
                "uploaded_files": len(_operations()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

