from __future__ import annotations

import contextlib
import io
import unittest

from scripts.inspect_dataset import (
    ProfileState,
    TokenCounter,
    expected_file_path,
    main,
    parse_languages,
    percentile,
)


class PercentileTests(unittest.TestCase):
    def test_empty_series(self) -> None:
        self.assertIsNone(percentile([], 50))

    def test_linear_interpolation(self) -> None:
        self.assertEqual(percentile([0, 10], 50), 5.0)
        self.assertEqual(percentile([1, 2, 3, 4], 100), 4.0)

    def test_invalid_percentile(self) -> None:
        with self.assertRaises(ValueError):
            percentile([1], 101)


class DatasetShapeTests(unittest.TestCase):
    def test_current_file_mapping(self) -> None:
        self.assertEqual(
            expected_file_path("hi", "validation"), "validation/hinval.parquet"
        )
        self.assertEqual(expected_file_path("te", "train"), "train/teltrain.parquet")

    def test_language_parser_accepts_spaces_and_commas(self) -> None:
        self.assertEqual(parse_languages(["hi,bn", "ta"]), ["hi", "bn", "ta"])
        self.assertGreater(len(parse_languages(["all"])), 1)

    def test_language_parser_rejects_unknown_codes(self) -> None:
        with self.assertRaises(ValueError):
            parse_languages(["xx"])

    def test_profile_handles_nested_columns_and_mismatched_arrays(self) -> None:
        record = {
            "source_lang": "eng_Latn",
            "target_lang": "hin_Deva",
            "meta": {
                "model_name": "translation-model",
                "temperature": 0.0,
                "max_tokens": 4096,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            },
            "query": "भारत की राजधानी क्या है?",
            "Answer": "नई दिल्ली",
            "query_id": 42,
            "query_type": "LOCATION",
            "passages": {
                "Translated_passages": ["भारत की राजधानी नई दिल्ली है।", ""],
                "English_passages": ["The capital of India is New Delhi."],
                "is_selected": [1, 0, "unexpected"],
            },
            "Eng_Query": "What is the capital of India?",
            "Eng_Answer": "New Delhi",
        }
        state = ProfileState(
            token_counter=TokenCounter("test", lambda text: len(text.split())),
            schema_sample_limit=1,
        )
        state.add_record(record)
        report = state.as_dict()

        self.assertEqual(report["records"], 1)
        self.assertEqual(report["data_quality"]["passage_slots"], 3)
        self.assertEqual(
            report["data_quality"]["records_with_mismatched_passage_arrays"], 1
        )
        self.assertEqual(report["data_quality"]["selected_passages"], 1)
        self.assertEqual(report["data_quality"]["invalid_is_selected_labels"], 1)
        self.assertEqual(
            report["length_distributions"]["translated_passage_tokens"]["count"], 1
        )
        self.assertIn("passages.Translated_passages", report["observed_schema"])
        self.assertEqual(
            report["categorical_distributions"]["target_lang"], {"hin_Deva": 1}
        )

    def test_dry_run_needs_no_third_party_dependencies(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(
                [
                    "--languages",
                    "hi",
                    "--split",
                    "validation",
                    "--dry-run",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertIn("validation/hinval.parquet", output.getvalue())


if __name__ == "__main__":
    unittest.main()
