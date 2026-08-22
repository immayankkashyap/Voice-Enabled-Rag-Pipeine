from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.runtime import RuntimeLoadError, _validate_encoder_provenance


class RuntimeProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.encoder = SimpleNamespace(
            model_name="example/embedding-model",
            model_revision="0123456789abcdef",
        )

    def test_accepts_exact_model_and_revision(self) -> None:
        store = SimpleNamespace(
            provenance={
                "embedding_model": self.encoder.model_name,
                "embedding_model_revision": self.encoder.model_revision,
            }
        )

        _validate_encoder_provenance(store, self.encoder)

    def test_rejects_model_mismatch(self) -> None:
        store = SimpleNamespace(
            provenance={
                "embedding_model": "different/model",
                "embedding_model_revision": self.encoder.model_revision,
            }
        )

        with self.assertRaisesRegex(RuntimeLoadError, "model does not match"):
            _validate_encoder_provenance(store, self.encoder)

    def test_rejects_missing_or_changed_revision(self) -> None:
        for revision in (None, "different-revision"):
            with self.subTest(revision=revision):
                store = SimpleNamespace(
                    provenance={
                        "embedding_model": self.encoder.model_name,
                        "embedding_model_revision": revision,
                    }
                )

                with self.assertRaisesRegex(RuntimeLoadError, "revision"):
                    _validate_encoder_provenance(store, self.encoder)


if __name__ == "__main__":
    unittest.main()
