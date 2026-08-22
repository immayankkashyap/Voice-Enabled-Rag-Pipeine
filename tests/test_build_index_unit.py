from __future__ import annotations

import unittest

from scripts.build_index import _collect_source_data


class BuildIndexExtractionTests(unittest.TestCase):
    def test_collects_aligned_passages_and_queries(self) -> None:
        records = [
            {
                "query_id": 17,
                "query": "परीक्षण प्रश्न",
                "Eng_Query": "test question",
                "passages": {
                    "Translated_passages": ["पहला", "", "दूसरा"],
                    "is_selected": [1, 0, 0],
                },
            }
        ]
        documents, queries = _collect_source_data(records, "hi")
        self.assertEqual([item["id"] for item in documents], ["hi:17:0", "hi:17:2"])
        self.assertTrue(documents[0]["is_selected"])
        self.assertFalse(documents[1]["is_selected"])
        self.assertEqual(queries[0]["query"], "परीक्षण प्रश्न")


if __name__ == "__main__":
    unittest.main()
