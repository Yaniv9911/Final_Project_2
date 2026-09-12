"""Unit tests for pure logic in 4_conversational_rag.py — no API calls.

The module under test starts with a digit, so it can't be imported with a
normal `import` statement — load it from its file path instead.

Run with: python test_conversational_rag.py
"""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "conversational_rag", Path(__file__).parent / "4_conversational_rag.py"
)
rag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag)


class TestCheckCitations(unittest.TestCase):
    def test_passes_when_all_cited_ids_were_retrieved(self):
        retrieved = [("r011", 0.9), ("r024", 0.5)]
        answer = "Your order was cancelled [r011] and never arrived [r024]."
        result = rag.check_citations(answer, retrieved)
        self.assertTrue(result["passed"])
        self.assertEqual(result["invented_ids"], [])
        self.assertEqual(result["cited_ids"], ["r011", "r024"])

    def test_fails_and_names_invented_ids(self):
        retrieved = [("r011", 0.9)]
        answer = "This is documented in [r011] and also [r999]."
        result = rag.check_citations(answer, retrieved)
        self.assertFalse(result["passed"])
        self.assertEqual(result["invented_ids"], ["r999"])

    def test_no_citations_at_all_passes(self):
        result = rag.check_citations("No citations here.", [("r011", 0.9)])
        self.assertTrue(result["passed"])
        self.assertEqual(result["cited_ids"], [])


class TestBuildContext(unittest.TestCase):
    def test_formats_ids_and_text(self):
        retrieved = [("r001", 0.8), ("r002", 0.6)]
        corpus_by_id = {"r001": {"id": "r001", "text": "first review"},
                        "r002": {"id": "r002", "text": "second review"}}
        context = rag.build_context(retrieved, corpus_by_id)
        self.assertEqual(context, "[r001] first review\n[r002] second review")


class TestGroundingGate(unittest.TestCase):
    def test_empty_retrieval_never_calls_generation(self):
        # A query with no lexical/semantic overlap against a tiny corpus
        # should retrieve nothing, and process_turn must return the exact
        # refusal message with citation_check=None -- proving the early
        # return, not just checking the visible text.
        corpus = [
            {"id": "r001", "text": "order #48213 was cancelled without notice"},
            {"id": "r002", "text": "great value for the price would buy again"},
        ]
        result = rag.process_turn("do you offer gift wrapping", history=[], use_memory=False, corpus=corpus)
        self.assertEqual(result["answer"], rag.REFUSAL_MESSAGE)
        self.assertEqual(result["retrieved"], [])
        self.assertEqual(result["context"], "")
        self.assertIsNone(result["citation_check"])


if __name__ == "__main__":
    unittest.main()
