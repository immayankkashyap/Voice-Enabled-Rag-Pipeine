from __future__ import annotations

import base64
import json
import unittest
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from app.elevenlabs_stt import (
    FREE_TIER_ACKNOWLEDGEMENT,
    ElevenLabsAudioError,
    ElevenLabsBillingSafetyError,
    ElevenLabsLimitError,
    ElevenLabsOriginError,
    ElevenLabsQuotaError,
    ElevenLabsStreamingSTT,
    ElevenLabsSTTSettings,
    ElevenLabsTokenBroker,
    _reset_process_budget_for_tests,
)

FREE_SUBSCRIPTION: dict[str, Any] = {
    "tier": "free",
    "status": "active",
    "max_credit_limit_extension": 0,
    "can_extend_character_limit": False,
    "allowed_to_extend_character_limit": False,
    "current_overage": {"amount": "0", "currency": "usd"},
    "has_open_invoices": False,
    "open_invoices": [],
    "next_invoice": None,
}


class _FakeResponse:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return deepcopy(self._payload)


class _FakeHTTPClient:
    def __init__(
        self,
        *,
        get_responses: list[_FakeResponse] | None = None,
        post_responses: list[_FakeResponse] | None = None,
    ) -> None:
        self.get_responses = list(get_responses or [])
        self.post_responses = list(post_responses or [])
        self.get_calls: list[tuple[str, dict[str, Any]]] = []
        self.post_calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    async def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.get_calls.append((url, kwargs))
        return self.get_responses.pop(0)

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.post_calls.append((url, kwargs))
        return self.post_responses.pop(0)

    async def aclose(self) -> None:
        self.closed = True


class _FakeSocket:
    def __init__(self, received: list[dict[str, Any] | BaseException]) -> None:
        self.received = iter(received)
        self.sent: list[dict[str, Any]] = []

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    async def recv(self) -> str:
        item = next(self.received)
        if isinstance(item, BaseException):
            raise item
        return json.dumps(item)


class _FakeConnection:
    def __init__(self, socket: _FakeSocket) -> None:
        self.socket = socket

    async def __aenter__(self) -> _FakeSocket:
        return self.socket

    async def __aexit__(self, *args: object) -> None:
        return None


def _settings(**overrides: Any) -> ElevenLabsSTTSettings:
    values: dict[str, Any] = {
        "api_key": "xi-secret-demo-key",
        "payg_and_auto_top_up_disabled_acknowledged": True,
        "max_attempts": 1,
        "retry_min_seconds": 0,
        "retry_max_seconds": 0,
    }
    values.update(overrides)
    return ElevenLabsSTTSettings(**values)


def _http_for_sessions(token_count: int = 1) -> _FakeHTTPClient:
    return _FakeHTTPClient(
        get_responses=[_FakeResponse(200, FREE_SUBSCRIPTION)],
        post_responses=[
            _FakeResponse(200, {"token": f"sutkn-test-{index}"})
            for index in range(token_count)
        ],
    )


async def _audio(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


def _successful_events() -> list[dict[str, Any]]:
    return [
        {
            "message_type": "session_started",
            "session_id": "safe-test-session",
        },
        {"message_type": "partial_transcript", "text": "What is"},
        {"message_type": "final_transcript", "text": "What is RAG?"},
        {"message_type": "committed_transcript", "text": "What is RAG?"},
    ]


class ElevenLabsSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_process_budget_for_tests()

    def test_from_env_requires_exact_acknowledgement_and_hides_secret(self) -> None:
        settings = ElevenLabsSTTSettings.from_env(
            {
                "ELEVENLABS_API_KEY": "do-not-print-this",
                "ELEVENLABS_FREE_TIER_ACKNOWLEDGEMENT": (FREE_TIER_ACKNOWLEDGEMENT),
            }
        )

        self.assertTrue(settings.payg_and_auto_top_up_disabled_acknowledged)
        self.assertNotIn("do-not-print-this", repr(settings))

        unacknowledged = ElevenLabsSTTSettings.from_env(
            {
                "ELEVENLABS_API_KEY": "secret",
                "ELEVENLABS_FREE_TIER_ACKNOWLEDGEMENT": "true",
            }
        )
        self.assertFalse(unacknowledged.payg_and_auto_top_up_disabled_acknowledged)


class ElevenLabsTokenBrokerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _reset_process_budget_for_tests()

    async def test_preflights_free_account_and_issues_origin_scoped_token(
        self,
    ) -> None:
        http = _http_for_sessions()
        settings = _settings(
            allowed_origins=("https://demo.example",),
            require_origin=True,
        )
        broker = ElevenLabsTokenBroker(settings, http_client=http)

        token = await broker.issue_token(origin="https://demo.example")

        self.assertEqual(token, "sutkn-test-0")
        self.assertEqual(len(http.get_calls), 1)
        self.assertEqual(len(http.post_calls), 1)
        self.assertEqual(
            http.get_calls[0][1]["headers"],
            {"xi-api-key": "xi-secret-demo-key"},
        )
        self.assertNotIn("xi-secret-demo-key", token)

    async def test_missing_operator_acknowledgement_fails_before_network(self) -> None:
        http = _http_for_sessions()
        settings = _settings(payg_and_auto_top_up_disabled_acknowledged=False)
        broker = ElevenLabsTokenBroker(settings, http_client=http)

        with self.assertRaises(ElevenLabsBillingSafetyError):
            await broker.issue_token()

        self.assertEqual(http.get_calls, [])
        self.assertEqual(http.post_calls, [])

    async def test_paid_or_extendable_account_fails_closed(self) -> None:
        for tier, extension in (("creator", 10), ("free", False)):
            with self.subTest(tier=tier, extension=extension):
                unsafe = deepcopy(FREE_SUBSCRIPTION)
                unsafe["tier"] = tier
                unsafe["max_credit_limit_extension"] = extension
                http = _FakeHTTPClient(
                    get_responses=[_FakeResponse(200, unsafe)],
                    post_responses=[_FakeResponse(200, {"token": "must-not-issue"})],
                )
                broker = ElevenLabsTokenBroker(_settings(), http_client=http)

                with self.assertRaises(ElevenLabsBillingSafetyError):
                    await broker.issue_token()

                self.assertEqual(http.post_calls, [])

    async def test_origin_and_process_token_caps_fail_closed(self) -> None:
        http = _http_for_sessions(token_count=2)
        settings = _settings(
            allowed_origins=("https://demo.example",),
            require_origin=True,
            daily_token_cap=1,
            daily_session_cap=1,
        )
        broker = ElevenLabsTokenBroker(settings, http_client=http)

        with self.assertRaises(ElevenLabsOriginError):
            await broker.issue_token(origin="https://evil.example")
        await broker.issue_token(origin="https://demo.example")
        with self.assertRaises(ElevenLabsLimitError):
            await broker.issue_token(origin="https://demo.example")

        self.assertEqual(len(http.post_calls), 1)


class ElevenLabsStreamingSTTTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _reset_process_budget_for_tests()

    async def test_streams_pcm_and_returns_only_committed_transcript(self) -> None:
        http = _http_for_sessions()
        socket = _FakeSocket(_successful_events())
        partials: list[str] = []
        settled: list[str] = []
        provider = ElevenLabsStreamingSTT(_settings(), http_client=http)
        first = b"\x01\x00" * 160
        second = b"\x02\x00" * 160

        with patch(
            "app.elevenlabs_stt.websockets.connect",
            return_value=_FakeConnection(socket),
        ) as connect:
            result = await provider.transcribe(
                _audio(first, second),
                on_partial=partials.append,
                on_settled=settled.append,
            )

        self.assertEqual(result.provider, "elevenlabs")
        self.assertEqual(result.model, "scribe_v2_realtime")
        self.assertEqual(result.transcript, "What is RAG?")
        self.assertEqual(result.language_code, "eng")
        self.assertEqual(result.partial_transcripts, ["What is"])
        self.assertEqual(partials, ["What is"])
        self.assertEqual(settled, ["What is RAG?"])
        self.assertAlmostEqual(result.audio_duration_ms or -1, 20.0)
        self.assertIsNotNone(result.first_audio_to_final_ms)
        self.assertGreaterEqual(result.final_after_audio_end_ms, 0)

        url = connect.call_args.args[0]
        query = parse_qs(urlsplit(url).query)
        self.assertEqual(query["model_id"], ["scribe_v2_realtime"])
        self.assertEqual(query["token"], ["sutkn-test-0"])
        self.assertEqual(query["commit_strategy"], ["manual"])
        self.assertEqual(query["include_timestamps"], ["false"])
        self.assertEqual(query["no_verbatim"], ["false"])
        self.assertNotIn("xi-secret-demo-key", url)
        self.assertNotIn("additional_headers", connect.call_args.kwargs)

        audio_messages = socket.sent[:-1]
        self.assertEqual(
            [base64.b64decode(message["audio_base_64"]) for message in audio_messages],
            [first, second],
        )
        self.assertEqual(
            socket.sent[-1],
            {
                "message_type": "input_audio_chunk",
                "audio_base_64": "",
                "commit": True,
                "sample_rate": 16_000,
            },
        )

    async def test_quota_event_is_terminal_and_is_not_retried(self) -> None:
        http = _http_for_sessions(token_count=3)
        socket = _FakeSocket([{"message_type": "quota_exceeded"}])
        provider = ElevenLabsStreamingSTT(
            _settings(max_attempts=3),
            http_client=http,
        )

        with (
            patch(
                "app.elevenlabs_stt.websockets.connect",
                return_value=_FakeConnection(socket),
            ) as connect,
            self.assertRaises(ElevenLabsQuotaError),
        ):
            await provider.transcribe(_audio(b"\x00\x00" * 160))

        self.assertEqual(connect.call_count, 1)
        self.assertEqual(len(http.post_calls), 1)

    async def test_transient_event_retries_with_replayed_audio(self) -> None:
        http = _http_for_sessions(token_count=2)
        failed_socket = _FakeSocket([{"message_type": "rate_limited"}])
        successful_socket = _FakeSocket(_successful_events())
        provider = ElevenLabsStreamingSTT(
            _settings(max_attempts=2),
            http_client=http,
        )
        chunk = b"\x03\x00" * 160

        with patch(
            "app.elevenlabs_stt.websockets.connect",
            side_effect=[
                _FakeConnection(failed_socket),
                _FakeConnection(successful_socket),
            ],
        ) as connect:
            result = await provider.transcribe(_audio(chunk))

        self.assertEqual(result.transcript, "What is RAG?")
        self.assertEqual(connect.call_count, 2)
        self.assertEqual(len(http.get_calls), 1)
        self.assertEqual(len(http.post_calls), 2)
        self.assertEqual(
            base64.b64decode(successful_socket.sent[0]["audio_base_64"]),
            chunk,
        )
        self.assertAlmostEqual(result.audio_duration_ms or -1, 10.0)

    async def test_invalid_or_overlong_pcm_is_terminal(self) -> None:
        odd_http = _http_for_sessions()
        odd_provider = ElevenLabsStreamingSTT(
            _settings(),
            http_client=odd_http,
        )
        odd_socket = _FakeSocket(_successful_events())
        with (
            patch(
                "app.elevenlabs_stt.websockets.connect",
                return_value=_FakeConnection(odd_socket),
            ),
            self.assertRaises(ElevenLabsAudioError),
        ):
            await odd_provider.transcribe(_audio(b"\x00"))

        _reset_process_budget_for_tests()
        long_http = _http_for_sessions()
        long_provider = ElevenLabsStreamingSTT(
            _settings(max_audio_seconds=0.001),
            http_client=long_http,
        )
        long_socket = _FakeSocket(_successful_events())
        with (
            patch(
                "app.elevenlabs_stt.websockets.connect",
                return_value=_FakeConnection(long_socket),
            ),
            self.assertRaises(ElevenLabsAudioError),
        ):
            await long_provider.transcribe(_audio(b"\x00\x00" * 17))


if __name__ == "__main__":
    unittest.main()
