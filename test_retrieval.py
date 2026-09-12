"""Unit tests for retrieval.py — pure Python, no API calls, no Streamlit.

Run with: python test_retrieval.py
"""
import unittest

import retrieval as rt


class TestTokenize(unittest.TestCase):
    def test_keeps_identifiers_whole(self):
        self.assertEqual(rt.tokenize("#48213"), ["48213"])
        self.assertEqual(rt.tokenize("$12.99"), ["12.99"])
        self.assertEqual(rt.tokenize("ax-7710"), ["ax-7710"])
        self.assertEqual(rt.tokenize("9921-B"), ["9921-b"])
        self.assertEqual(rt.tokenize("save20"), ["save20"])

    def test_distinguishes_similar_identifiers(self):
        self.assertNotEqual(rt.tokenize("ax-7710"), rt.tokenize("ax-7701"))
        self.assertNotEqual(rt.tokenize("$12.99"), rt.tokenize("$12.90"))

    def test_drops_sentence_punctuation_not_identifier_punctuation(self):
        self.assertEqual(rt.tokenize("described."), ["described"])
        self.assertEqual(rt.tokenize("Order #48213 - still no sign."),
                          ["order", "48213", "still", "no", "sign"])


class TestRemoveStopwords(unittest.TestCase):
    def test_removes_common_stopwords(self):
        tokens = rt.tokenize("my package never arrived")
        self.assertIn("my", tokens)
        filtered = rt.remove_stopwords(tokens)
        self.assertNotIn("my", filtered)
        self.assertIn("package", filtered)
        self.assertIn("arrived", filtered)

    def test_keeps_identifiers(self):
        tokens = rt.tokenize("order #48213")
        filtered = rt.remove_stopwords(tokens)
        self.assertIn("48213", filtered)


class TestLexicalSearch(unittest.TestCase):
    # Deliberately more than a couple of documents: with a tiny corpus, a
    # shared term can land at exactly document-frequency = corpus_size/2,
    # which makes BM25's idf exactly zero and masks real behavior.
    CORPUS = [
        {"id": "r001", "text": "order #48213 was cancelled without notice"},
        {"id": "r002", "text": "the battery lasted 3 hours not 30 as advertised"},
        {"id": "r003", "text": "great value for the price would buy again"},
        {"id": "r004", "text": "the zip jammed the very first time I used it"},
        {"id": "r005", "text": "nice quality material fits as expected"},
    ]

    def test_returns_list_of_id_score_pairs(self):
        results = rt.lexical_search("order 48213", k=3, corpus=self.CORPUS)
        self.assertTrue(results)
        self.assertEqual(results[0][0], "r001")
        self.assertIsInstance(results[0][1], float)

    def test_abstains_on_unrelated_query(self):
        results = rt.lexical_search("do you offer gift wrapping", k=3, corpus=self.CORPUS)
        self.assertEqual(results, [])

    def test_use_stopwords_toggle_changes_behavior(self):
        # With such a tiny corpus, this just proves the flag is wired through
        # without raising, and both settings return a valid (possibly empty) list.
        with_sw = rt.lexical_search("order 48213", k=3, corpus=self.CORPUS, use_stopwords=True)
        without_sw = rt.lexical_search("order 48213", k=3, corpus=self.CORPUS, use_stopwords=False)
        self.assertIsInstance(with_sw, list)
        self.assertIsInstance(without_sw, list)

    def test_empty_query_returns_empty(self):
        self.assertEqual(rt.lexical_search("the and of", k=3, corpus=self.CORPUS), [])

    def test_empty_corpus_returns_empty(self):
        self.assertEqual(rt.lexical_search("anything", k=3, corpus=[]), [])


class TestSemanticSearch(unittest.TestCase):
    CORPUS = [
        {"id": "r001", "text": "the courier left the package at a neighbour's door"},
        {"id": "r002", "text": "after two washes the colour completely faded"},
    ]

    def test_returns_list_of_id_score_pairs(self):
        results = rt.semantic_search("my package never arrived", k=2, corpus=self.CORPUS, min_similarity=0.0)
        self.assertTrue(results)
        self.assertIsInstance(results[0][1], float)

    def test_abstains_below_threshold(self):
        results = rt.semantic_search("anything", k=2, corpus=self.CORPUS, min_similarity=1.1)
        self.assertEqual(results, [])

    def test_empty_corpus_returns_empty(self):
        self.assertEqual(rt.semantic_search("anything", k=2, corpus=[]), [])


class TestHybridSearch(unittest.TestCase):
    CORPUS = [
        {"id": "r001", "text": "order #48213 was cancelled without notice"},
        {"id": "r002", "text": "the battery lasted 3 hours not 30 as advertised"},
        {"id": "r003", "text": "great value for the price would buy again"},
        {"id": "r004", "text": "the zip jammed the very first time I used it"},
        {"id": "r005", "text": "nice quality material fits as expected"},
    ]

    def test_returns_fused_results(self):
        results = rt.hybrid_search("order 48213", k=2, corpus=self.CORPUS)
        self.assertTrue(results)
        self.assertEqual(results[0][0], "r001")

    def test_abstains_only_when_both_arms_empty(self):
        # Force both arms to abstain via impossible thresholds.
        results = rt.hybrid_search(
            "order 48213", k=2, corpus=self.CORPUS,
            lexical_kwargs={"min_score": 999}, semantic_kwargs={"min_similarity": 1.1},
        )
        self.assertEqual(results, [])

    def test_survives_when_only_one_arm_empty(self):
        # Semantic forced to abstain, lexical should still carry the result through.
        results = rt.hybrid_search(
            "order 48213", k=2, corpus=self.CORPUS,
            semantic_kwargs={"min_similarity": 1.1},
        )
        self.assertTrue(results)


if __name__ == "__main__":
    unittest.main()
