from __future__ import annotations

import base64
import json
import unittest
from unittest.mock import patch

from app.stt import SarvamStreamingSTT, SarvamSTTSettings


class _FakeSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []
        self.received = iter(
            (
                {"event": "session.begin", "request_id": "test-session"},
                {"event": "transcript.partial", "text": "What is"},
                {"event": "transcript.final", "text": "What is RAG?"},
            )
        )

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    async def recv(self) -> str:
        return json.dumps(next(self.received))


class _FakeConnection:
    def __init__(self, socket: _FakeSocket) -> None:
        self.socket = socket

    async def __aenter__(self) -> _FakeSocket:
        return self.socket

    async def __aexit__(self, *args: object) -> None:
        return None


class SarvamStreamingSTTTests(unittest.IsolatedAsyncioTestCase):
    async def test_handles_partial_and_final_realtime_events(self) -> None:
        socket = _FakeSocket()
        partials: list[str] = []

        async def audio_chunks():
            yield b"\x01\x02"
            yield b"\x03\x04"

        with patch(
            "app.stt.websockets.connect", return_value=_FakeConnection(socket)
        ) as connect:
            result = await SarvamStreamingSTT(
                SarvamSTTSettings(api_key="test-key", max_attempts=1)
            ).transcribe(audio_chunks(), on_partial=partials.append)

        self.assertEqual(result.transcript, "What is RAG?")
        self.assertEqual(result.partial_transcripts, ["What is"])
        self.assertEqual(partials, ["What is"])
        self.assertIsNotNone(result.time_to_first_partial_ms)
        self.assertGreaterEqual(result.time_to_final_transcript_ms, 0)

        url = connect.call_args.args[0]
        self.assertIn("speech-to-text-realtime/ws", url)
        self.assertIn("mode=transcribe", url)
        self.assertEqual(
            connect.call_args.kwargs["additional_headers"],
            {"Api-Subscription-Key": "test-key"},
        )
        audio_messages = [
            item for item in socket.sent if item["event"] == "audio_input"
        ]
        self.assertEqual(
            [base64.b64decode(str(item["audio"])) for item in audio_messages],
            [b"\x01\x02", b"\x03\x04"],
        )
        self.assertEqual(socket.sent[0], {"event": "speech_start"})
        self.assertEqual(socket.sent[-2], {"event": "speech_end"})
        self.assertEqual(socket.sent[-1], {"event": "end"})

    def test_realtime_model_and_sample_rate_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            SarvamSTTSettings(api_key="test-key", model="saaras:v3")
        with self.assertRaises(ValueError):
            SarvamSTTSettings(api_key="test-key", sample_rate=44_100)


if __name__ == "__main__":
    unittest.main()
