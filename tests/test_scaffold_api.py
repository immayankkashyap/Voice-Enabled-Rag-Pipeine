from __future__ import annotations

import unittest

from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.schemas import RAGRequest


class ScaffoldAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health_reports_current_phase(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["implementation_phase"], "day3_free_demo_harness"
        )
        self.assertFalse(response.json()["rag_ready"])
        self.assertEqual(response.json()["stt_provider"], "elevenlabs")
        self.assertEqual(response.json()["answer_mode"], "local_extractive")

    def test_rag_refuses_to_fabricate_an_answer(self) -> None:
        response = self.client.post("/rag", json={"query": "What is RAG?"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error_code"], "rag_runtime_not_ready")
        self.assertFalse(response.json()["retryable"])

    def test_request_rejects_impossible_candidate_counts(self) -> None:
        with self.assertRaises(ValidationError):
            RAGRequest(query="test", candidate_k=4, final_k=5)

    def test_request_rejects_unknown_fields(self) -> None:
        response = self.client.post(
            "/rag",
            json={"query": "What is RAG?", "unvalidated_option": True},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
