from __future__ import annotations

import re
import unittest

from fastapi.testclient import TestClient

from app.main import _DEMO_STATIC_DIR, app


class BrowserDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls.index = (_DEMO_STATIC_DIR / "index.html").read_text(encoding="utf-8")
        cls.javascript = (_DEMO_STATIC_DIR / "app.js").read_text(encoding="utf-8")
        cls.worklet = (_DEMO_STATIC_DIR / "pcm-worklet.js").read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def assert_security_headers(self, response) -> None:
        self.assertEqual(response.headers["cache-control"], "no-store, max-age=0")
        self.assertEqual(response.headers["pragma"], "no-cache")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["x-frame-options"], "DENY")
        self.assertEqual(response.headers["referrer-policy"], "no-referrer")
        self.assertEqual(response.headers["permissions-policy"], "microphone=(self)")
        csp = response.headers["content-security-policy"]
        self.assertIn("default-src 'none'", csp)
        self.assertIn("script-src 'self'", csp)
        self.assertIn("connect-src 'self'", csp)
        self.assertIn("frame-ancestors 'none'", csp)
        self.assertNotIn("'unsafe-inline'", csp)
        self.assertNotIn("'unsafe-eval'", csp)

    def test_demo_and_assets_are_bundled_with_no_store_headers(self) -> None:
        self.assertTrue(_DEMO_STATIC_DIR.is_absolute())
        response = self.client.get("/demo")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/html"))
        self.assert_security_headers(response)
        self.assertIn('src="/static/app.js"', response.text)
        self.assertIn('href="/static/styles.css"', response.text)

        for asset, content_type in (
            ("app.js", "text/javascript"),
            ("pcm-worklet.js", "text/javascript"),
            ("styles.css", "text/css"),
        ):
            with self.subTest(asset=asset):
                asset_response = self.client.get(f"/static/{asset}")
                self.assertEqual(asset_response.status_code, 200)
                self.assertTrue(
                    asset_response.headers["content-type"].startswith(content_type)
                )
                self.assert_security_headers(asset_response)

    def test_page_loads_no_remote_code_or_analytics(self) -> None:
        script_sources = re.findall(r'<script[^>]+src="([^"]+)"', self.index)
        stylesheet_links = re.findall(
            r'<link[^>]+href="([^"]+)"',
            self.index,
        )
        self.assertEqual(script_sources, ["/static/app.js"])
        self.assertEqual(stylesheet_links, ["/static/styles.css"])
        self.assertNotIn("<script>", self.index)
        for marker in ("googletag", "analytics", "segment", "mixpanel", "cdn."):
            self.assertNotIn(marker, (self.index + self.javascript).lower())

    def test_operator_secret_is_not_persisted_logged_or_placed_in_url(self) -> None:
        self.assertRegex(
            self.index,
            r'id="demo-token"[\s\S]*?type="password"[\s\S]*?autocomplete="off"',
        )
        self.assertIn("demo_token: token", self.javascript)
        self.assertIn('tokenInput.value = ""', self.javascript)
        self.assertIn('token = ""', self.javascript)
        self.assertIn("/ws/voice-rag", self.javascript)
        for forbidden in (
            "localStorage",
            "sessionStorage",
            "indexedDB",
            "document.cookie",
            "console.",
            "URLSearchParams",
            "searchParams",
            "innerHTML",
            "insertAdjacentHTML",
            "ELEVENLABS_API_KEY",
            "xi-api-key",
        ):
            self.assertNotIn(forbidden, self.index + self.javascript)
        self.assertNotRegex(self.javascript, r"(?:ws|http)s?[^\n]*demo_token")

    def test_audio_is_manual_16khz_pcm16_and_frames_are_bounded(self) -> None:
        self.assertIn("getUserMedia", self.javascript)
        self.assertIn("AudioWorkletNode", self.javascript)
        self.assertIn("const TARGET_SAMPLE_RATE = 16_000", self.javascript)
        self.assertIn("const MAX_FRAME_BYTES = 3_200", self.javascript)
        self.assertIn("const BATCH_SAMPLES = 320", self.worklet)
        self.assertIn("const MAX_BATCH_SAMPLES = 1_600", self.worklet)
        self.assertIn(
            'turn.socket.send(JSON.stringify({ event: "end" }))', self.javascript
        )
        self.assertNotIn("voiceActivity", self.javascript)
        self.assertNotIn("silenceTimer", self.javascript)

        # The authentication/format handshake must complete before microphone
        # capture begins; provider work starts only after the first PCM frame.
        connect_at = self.javascript.index("await connectForTurn")
        microphone_at = self.javascript.index("await beginMicrophone")
        self.assertLess(connect_at, microphone_at)

    def test_static_mount_does_not_expose_project_files(self) -> None:
        response = self.client.get("/static/%2e%2e/.env.example")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
