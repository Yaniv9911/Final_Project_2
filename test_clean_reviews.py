"""Unit tests for the pure pipeline functions in 1_clean_reviews.py.

The module under test starts with a digit, so it can't be imported with a
normal `import` statement — load it from its file path instead.

Run with: python test_clean_reviews.py
"""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "clean_reviews", Path(__file__).parent / "1_clean_reviews.py"
)
clean_reviews = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clean_reviews)


class TestStandardizeEntities(unittest.TestCase):
    def test_replaces_cs_any_case(self):
        self.assertEqual(
            clean_reviews.standardize_entities("CS never got back to me."),
            "customer service never got back to me.",
        )
        self.assertEqual(
            clean_reviews.standardize_entities("Contacted cs twice."),
            "Contacted customer service twice.",
        )

    def test_replaces_support_any_case(self):
        self.assertEqual(
            clean_reviews.standardize_entities("Support said they would call back."),
            "customer service said they would call back.",
        )
        self.assertEqual(
            clean_reviews.standardize_entities("support never followed up."),
            "customer service never followed up.",
        )

    def test_does_not_touch_supported_or_supports(self):
        self.assertEqual(
            clean_reviews.standardize_entities("The device supports fast charging."),
            "The device supports fast charging.",
        )
        self.assertEqual(
            clean_reviews.standardize_entities("It supported my old cable too."),
            "It supported my old cable too.",
        )

    def test_does_not_touch_cs_as_substring(self):
        # "cscript" contains "cs" but not as a whole word — must survive untouched.
        self.assertEqual(
            clean_reviews.standardize_entities("Run cscript from the shell."),
            "Run cscript from the shell.",
        )


class TestNormalizeText(unittest.TestCase):
    def test_lowercases_and_collapses_whitespace(self):
        self.assertEqual(
            clean_reviews.normalize_text("  Nobody   responded  to my emails.  "),
            "nobody responded to my emails.",
        )

    def test_preserves_ids_and_symbols(self):
        for text in ["order #48213", "$12.99 fee", "sku ax-7710", "9921-b", "save20"]:
            self.assertEqual(clean_reviews.normalize_text(text), text)


if __name__ == "__main__":
    unittest.main()
