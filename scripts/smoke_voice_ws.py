#!/usr/bin/env python3
"""Exercise the real one-turn Sarvam ``/ws/voice-rag`` endpoint locally."""

from __future__ import annotations

import json
import os
import sys
import wave
from pathlib import Path

from dotenv import load_dotenv
from fastapi.testclient import TestClient

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

load_dotenv()
os.environ.setdefault("VOICE_ALLOWED_ORIGINS", "http://testserver")
os.environ.setdefault("VOICE_REQUIRE_ORIGIN", "true")

from app.main import app  # noqa: E402 - environment must be ready before app import


def _audio_frames(path: Path, frames_per_message: int = 4_000) -> list[bytes]:
    with wave.open(str(path), "rb") as audio:
        if (
            audio.getnchannels() != 1
            or audio.getsampwidth() != 2
            or audio.getframerate() != 16_000
        ):
            raise ValueError("Voice smoke audio must be mono PCM16 at 16 kHz")
        frames: list[bytes] = []
        while chunk := audio.readframes(frames_per_message):
            frames.append(chunk)
        return frames


def main() -> int:
    token = os.getenv("VOICE_DEMO_TOKEN", "")
    if not token:
        print("VOICE_DEMO_TOKEN is missing", file=sys.stderr)
        return 2
    frames = _audio_frames(Path("data/stt_sample.wav"))
    events: list[dict[str, object]] = []
    with TestClient(app) as client:
        with client.websocket_connect(
            "/ws/voice-rag",
            headers={"origin": "http://testserver"},
        ) as websocket:
            websocket.send_json(
                {
                    "event": "start",
                    "language_code": "hi",
                    "demo_token": token,
                }
            )
            ready = websocket.receive_json()
            events.append(ready)
            if ready.get("event") != "ready":
                print(json.dumps(events, ensure_ascii=False, indent=2))
                return 1
            for frame in frames:
                websocket.send_bytes(frame)
            websocket.send_json({"event": "end"})
            while True:
                event = websocket.receive_json()
                events.append(event)
                if event.get("event") == "answer" or "error_code" in event:
                    break

    print(json.dumps(events, ensure_ascii=False, indent=2))
    final = events[-1]
    if final.get("event") != "answer":
        return 1
    payload = final.get("payload")
    if not isinstance(payload, dict):
        return 1
    transcription = payload.get("transcription")
    rag = payload.get("rag")
    return 0 if (
        isinstance(transcription, dict)
        and transcription.get("provider") == "sarvam"
        and isinstance(rag, dict)
        and rag.get("status") in {"answered", "refused"}
        and isinstance(rag.get("latencies"), dict)
        and rag["latencies"].get("stt_ms") is not None
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
