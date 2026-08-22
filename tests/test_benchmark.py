from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import benchmark


class FullPipelineBenchmarkTests(unittest.TestCase):
    def test_distribution_uses_interpolation_and_observed_max(self) -> None:
        result = benchmark._distribution([10.0, 20.0, 40.0])

        self.assertEqual(result["samples"], 3)
        self.assertEqual(result["p50"], 20.0)
        self.assertEqual(result["p70"], 28.0)
        self.assertEqual(result["p100"], 40.0)

    def test_query_loader_requires_thirty_real_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.json"
            rows = [
                {
                    "id": f"hi:{index}",
                    "query_id": str(index),
                    "language": "hi",
                    "query": f"वास्तविक प्रश्न {index}",
                }
                for index in range(30)
            ]
            path.write_text(json.dumps(rows), encoding="utf-8")

            loaded = benchmark._load_queries(path, 30)

            self.assertEqual(len(loaded), 30)
            self.assertEqual(loaded[0]["language"], "hi")
            with self.assertRaisesRegex(ValueError, "at least 30"):
                benchmark._load_queries(path, 29)


if __name__ == "__main__":
    unittest.main()
