"""Measured evidence for four written questions about the retrieval system.

Every number here is computed from the real corpus via retrieval.py's own
functions and internals (tokenize, the BM25 phantom-document mechanism
lexical_search already uses, the same embedding path semantic_search
uses) — nothing hardcoded, nothing invented.

Run with: python analysis/evidence.py > results/evidence.txt
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from rank_bm25 import BM25Okapi

import retrieval as rt
import score as sc

GOLD_PATH = "data/eval_queries.json"
K = 3


def load():
    with open(GOLD_PATH, encoding="utf-8") as f:
        gold = json.load(f)
    corpus = rt.load_reviews()
    by_id = {r["id"]: r for r in corpus}
    return gold, corpus, by_id


def bm25_score_for_doc(query, corpus, doc_id, use_stopwords=True, use_stemming=True):
    """Same phantom-document mechanism lexical_search uses internally:
    score every document (with the query appended as one extra document,
    so idf/avgdl stay well-defined) under a single BM25 index, then read
    off one document's score."""
    doc_tokens = [rt._prepare_tokens(r["text"], use_stopwords, use_stemming) for r in corpus]
    query_tokens = rt._prepare_tokens(query, use_stopwords, use_stemming)
    bm25 = BM25Okapi(doc_tokens + [query_tokens])
    scores = bm25.get_scores(query_tokens)
    idx = next(i for i, r in enumerate(corpus) if r["id"] == doc_id)
    return float(scores[idx])


def embed(texts):
    """Same embedding path semantic_search uses (EMBEDDING_PROVIDER=local)."""
    return rt._embed_texts(texts, "local")


def cosine(vec_a, vec_b):
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def q1_token_overlap(gold, corpus, by_id):
    print("=" * 78)
    print("Q1 -- token overlap: q09 'faulty zipper' vs r010, and semantic-type average")
    print("=" * 78)

    q09 = next(q for q in gold if q["id"] == "q09")
    r010 = by_id["r010"]
    query_tokens = rt.tokenize(q09["query"])
    doc_tokens = rt.tokenize(r010["text"])
    intersection = set(query_tokens) & set(doc_tokens)

    print(f"query: {q09['query']!r}")
    print(f"r010 text: {r010['text']!r}")
    print(f"tokenize(query) = {query_tokens}")
    print(f"tokenize(r010)  = {doc_tokens}")
    print(f"intersection    = {sorted(intersection)}")
    print(f"|intersection| = {len(intersection)}, |query tokens| = {len(query_tokens)}, "
          f"ratio = {len(intersection) / len(query_tokens):.3f}")

    bm25_r010 = bm25_score_for_doc(q09["query"], corpus, "r010")
    print(f"BM25 score for r010 against this query (phantom-doc method): {bm25_r010:.4f}")

    print()
    print("-- overlap ratio for every (query, relevant_id) pair among the 10 semantic-type queries --")
    semantic_queries = [q for q in gold if q["type"] == "semantic"]
    ratios = []
    for q in semantic_queries:
        q_tokens = set(rt.tokenize(q["query"]))
        for rid in q["relevant_ids"]:
            d_tokens = set(rt.tokenize(by_id[rid]["text"]))
            inter = q_tokens & d_tokens
            ratio = len(inter) / len(q_tokens) if q_tokens else 0.0
            ratios.append(ratio)
            print(f"  {q['id']:4} vs {rid}: intersection={sorted(inter)} "
                  f"|{len(inter)}|/|{len(q_tokens)}| = {ratio:.3f}")
    avg = sum(ratios) / len(ratios) if ratios else 0.0
    print(f"\naverage overlap ratio across {len(ratios)} (query, relevant_id) pairs: {avg:.3f}")
    print()


def q2_near_duplicate_identifiers(gold, corpus, by_id):
    print("=" * 78)
    print("Q2 -- near-duplicate identifiers: does the system distinguish them?")
    print("=" * 78)

    q13 = next(q for q in gold if q["id"] == "q13")
    print(f"query: {q13['query']!r}  (gold relevant_ids={q13['relevant_ids']})")
    top5 = rt.semantic_search(q13["query"], k=5, min_similarity=0.0, corpus=corpus)
    print("top-5 semantic results (min_similarity forced to 0.0 to show the full ranking, "
          "not what the tuned default returns):")
    for doc_id, sim in top5:
        print(f"  {doc_id}: cosine={sim:.4f}  text={by_id[doc_id]['text']!r}")
    print()

    pairs = [
        ("order #48213", "r011", "r068"),        # gold query for q11
        ("reference 9921-B", "r018", "r069"),    # gold query for q15
        ("SKU AX-7710", "r014", "r066"),         # gold query for q13
    ]
    for query_text, id_a, id_b in pairs:
        print(f"-- {query_text!r}: {id_a} vs {id_b} --")
        print(f"  {id_a}: {by_id[id_a]['text']!r}")
        print(f"  {id_b}: {by_id[id_b]['text']!r}")
        q_vec = embed([query_text])[0]
        vec_a, vec_b = embed([by_id[id_a]["text"], by_id[id_b]["text"]])
        sim_a, sim_b = cosine(q_vec, vec_a), cosine(q_vec, vec_b)
        bm25_a = bm25_score_for_doc(query_text, corpus, id_a)
        bm25_b = bm25_score_for_doc(query_text, corpus, id_b)
        print(f"  cosine(query, {id_a}) = {sim_a:.4f}   cosine(query, {id_b}) = {sim_b:.4f}   "
              f"gap = {abs(sim_a - sim_b):.4f}")
        print(f"  BM25(query, {id_a})   = {bm25_a:.4f}   BM25(query, {id_b})   = {bm25_b:.4f}   "
              f"gap = {abs(bm25_a - bm25_b):.4f}")
        print()


def q3_battery_gap(gold, corpus, by_id):
    print("=" * 78)
    print("Q3 -- r016 vs r019 gap, and each against q17/q18")
    print("=" * 78)

    r016_text = by_id["r016"]["text"]
    r019_text = by_id["r019"]["text"]
    q17 = next(q for q in gold if q["id"] == "q17")["query"]
    q18 = next(q for q in gold if q["id"] == "q18")["query"]

    print(f"r016: {r016_text!r}")
    print(f"r019: {r019_text!r}")
    print(f"q17 query: {q17!r}  (gold relevant_ids={next(q for q in gold if q['id']=='q17')['relevant_ids']})")
    print(f"q18 query: {q18!r}  (gold relevant_ids={next(q for q in gold if q['id']=='q18')['relevant_ids']})")

    vec_r016, vec_r019 = embed([r016_text, r019_text])
    vec_q17, vec_q18 = embed([q17, q18])

    sim_r016_r019 = cosine(vec_r016, vec_r019)
    sim_r016_q17 = cosine(vec_r016, vec_q17)
    sim_r019_q17 = cosine(vec_r019, vec_q17)
    sim_r016_q18 = cosine(vec_r016, vec_q18)
    sim_r019_q18 = cosine(vec_r019, vec_q18)

    print(f"\ncosine(r016, r019) = {sim_r016_r019:.4f}")
    print(f"cosine(r016, q17)  = {sim_r016_q17:.4f}")
    print(f"cosine(r019, q17)  = {sim_r019_q17:.4f}   gap = {abs(sim_r016_q17 - sim_r019_q17):.4f}")
    print(f"cosine(r016, q18)  = {sim_r016_q18:.4f}")
    print(f"cosine(r019, q18)  = {sim_r019_q18:.4f}   gap = {abs(sim_r016_q18 - sim_r019_q18):.4f}")
    print()


def q4_method_comparison(gold, corpus, by_id):
    print("=" * 78)
    print("Q4 -- per-type Recall@3 and MRR: lexical, semantic, plain RRF, gated hybrid")
    print("=" * 78)

    methods = {
        "lexical": lambda q: rt.lexical_search(q, k=K, corpus=corpus),
        "semantic": lambda q: rt.semantic_search(q, k=K, corpus=corpus),
        "plain_rrf": lambda q: rt.hybrid_search(
            q, k=K, corpus=corpus, identifier_override=False, gate_lexical=False),
        "gated_hybrid": lambda q: rt.hybrid_search(
            q, k=K, corpus=corpus, identifier_override=False, gate_lexical=True,
            gate_min_coverage=0.6, gate_max_doc_freq=5),
    }
    print("plain_rrf = hybrid_search(identifier_override=False, gate_lexical=False)")
    print("gated_hybrid = hybrid_search(gate_lexical=True, gate_min_coverage=0.6, "
          "gate_max_doc_freq=5) -- the swept-best gate setting from results/tuning_log.md")
    print()

    types = ["semantic", "lexical", "hybrid"]
    table_recall = {m: {} for m in methods}
    table_mrr = {m: {} for m in methods}

    for method_name, fn in methods.items():
        for t in types:
            qs = [q for q in gold if q["type"] == t]
            recalls, rrs = [], []
            for q in qs:
                retrieved = [doc_id for doc_id, _ in fn(q["query"])]
                r = sc.recall_at_k(retrieved, q["relevant_ids"], K)
                rr = sc.rr_at_k(retrieved, q["relevant_ids"], K)
                if r is not None:
                    recalls.append(r)
                if rr is not None:
                    rrs.append(rr)
            table_recall[method_name][t] = sum(recalls) / len(recalls) if recalls else 0.0
            table_mrr[method_name][t] = sum(rrs) / len(rrs) if rrs else 0.0

    method_names = list(methods.keys())
    header = "type".ljust(10) + "".join(m.rjust(15) for m in method_names)

    print("Recall@3 by query type:")
    print(header)
    for t in types:
        print(t.ljust(10) + "".join(f"{table_recall[m][t] * 100:14.1f}%" for m in method_names))

    print("\nMRR@3 by query type:")
    print(header)
    for t in types:
        print(t.ljust(10) + "".join(f"{table_mrr[m][t] * 100:14.1f}%" for m in method_names))
    print()


def main():
    gold, corpus, by_id = load()
    q1_token_overlap(gold, corpus, by_id)
    q2_near_duplicate_identifiers(gold, corpus, by_id)
    q3_battery_gap(gold, corpus, by_id)
    q4_method_comparison(gold, corpus, by_id)


if __name__ == "__main__":
    main()
