"""Shared retrieval configuration — the single source of truth for the
winning method and thresholds from step 3's tuning (see
results/tuning_log.md). Imported by both 3_measure_search.py (as its
slider defaults) and 4_conversational_rag.py (as the fixed, shipped
config), so the measured system and the shipped system cannot drift apart.
"""

TOP_K = 3

LEXICAL_KWARGS = {
    "use_stopwords": True,
    "use_stemming": True,
    "min_score": 0.30,
    "min_coverage": 0.50,
}

SEMANTIC_KWARGS = {
    "min_similarity": 0.30,
}

HYBRID_KWARGS = {
    "lexical_kwargs": LEXICAL_KWARGS,
    "semantic_kwargs": SEMANTIC_KWARGS,
    "weight_lexical": 1.0,
    "weight_semantic": 1.0,
    "gate_lexical": False,
    "identifier_override": True,
}
