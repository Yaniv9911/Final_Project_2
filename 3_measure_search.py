"""Streamlit app — Step 3: measure lexical/semantic/hybrid retrieval.

Imports retrieval.py (unchanged) and evaluates it against data/eval_queries.json
using the same Recall@k / MRR@k / abstain definitions as score.py (copied
here, not imported — score.py is a CLI script, not a library — but never
modified). Run files written here are the honest output of retrieval.py at
whatever the sidebar sliders currently say.

Run with: streamlit run "3_measure_search.py"
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import config
import retrieval as rt

EVAL_PATH = Path("data/eval_queries.json")
RUNS_DIR = Path("runs")

# Same weights and per-query formulas as score.py — copied, not imported.
W_RECALL, W_MRR, W_ABSTAIN = 0.50, 0.30, 0.20
METHODS = ["lexical", "semantic", "hybrid"]
TYPES = ["semantic", "lexical", "hybrid"]


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return None
    hits = len(set(retrieved[:k]) & set(relevant))
    return hits / len(relevant)


def rr_at_k(retrieved, relevant, k):
    if not relevant:
        return None
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


@st.cache_data
def load_gold():
    with open(EVAL_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_corpus():
    return rt.load_reviews()


@st.cache_data
def compute_all_results(k, lex_kwargs, sem_kwargs, identifier_override):
    """Run all three methods over every gold query at the given settings."""
    corpus = load_corpus()
    gold = load_gold()
    results = {}
    for q in gold:
        query = q["query"]
        lexical = rt.lexical_search(query, k=k, corpus=corpus, **lex_kwargs)
        semantic = rt.semantic_search(query, k=k, corpus=corpus, **sem_kwargs)
        hybrid = rt.hybrid_search(
            query, k=k, corpus=corpus, lexical_kwargs=lex_kwargs, semantic_kwargs=sem_kwargs,
            identifier_override=identifier_override,
        )
        results[q["id"]] = {"lexical": lexical, "semantic": semantic, "hybrid": hybrid}
    return results


def score_method(method, results, gold, k):
    by_type = {t: [] for t in TYPES}
    all_recall, all_rr = [], []
    abstain_hits, abstain_total = 0, 0

    for q in gold:
        retrieved_ids = [doc_id for doc_id, _ in results[q["id"]][method]]
        if q["type"] == "unanswerable":
            abstain_total += 1
            abstain_hits += int(len(retrieved_ids) == 0)
            continue
        recall = recall_at_k(retrieved_ids, q["relevant_ids"], k)
        rr = rr_at_k(retrieved_ids, q["relevant_ids"], k)
        all_recall.append(recall)
        all_rr.append(rr)
        by_type[q["type"]].append(recall)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    recall = mean(all_recall)
    mrr = mean(all_rr)
    abstain = abstain_hits / abstain_total if abstain_total else 0.0
    return {
        "recall": recall,
        "mrr": mrr,
        "abstain": abstain,
        "abstain_hits": abstain_hits,
        "abstain_total": abstain_total,
        "by_type_recall": {t: mean(v) for t, v in by_type.items()},
        "final": W_RECALL * recall + W_MRR * mrr + W_ABSTAIN * abstain,
    }


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    st.set_page_config(page_title="Measure Search", layout="wide")
    st.title("Step 3 — Measure Search")

    gold = load_gold()
    st.caption(f"{len(gold)} gold queries loaded from data/eval_queries.json")

    st.sidebar.header("Settings")
    st.sidebar.caption("Defaults below come from config.py — the same winning "
                        "configuration 4_conversational_rag.py ships with.")
    k = st.sidebar.slider("k", min_value=1, max_value=10, value=config.TOP_K)

    st.sidebar.subheader("Lexical (BM25)")
    use_stopwords = st.sidebar.checkbox("Remove stopwords", value=config.LEXICAL_KWARGS["use_stopwords"])
    use_stemming = st.sidebar.checkbox("Use light stemming", value=config.LEXICAL_KWARGS["use_stemming"])
    lex_min_score = st.sidebar.slider(
        "Normalized score threshold", 0.0, 1.5, config.LEXICAL_KWARGS["min_score"], 0.01)
    lex_min_coverage = st.sidebar.slider(
        "Query-term coverage threshold", 0.0, 1.0, config.LEXICAL_KWARGS["min_coverage"], 0.05)

    st.sidebar.subheader("Semantic (embeddings)")
    sem_min_similarity = st.sidebar.slider(
        "Cosine similarity threshold", 0.0, 1.0, config.SEMANTIC_KWARGS["min_similarity"], 0.01)

    st.sidebar.subheader("Hybrid fusion")
    identifier_override = st.sidebar.checkbox(
        "Exact-identifier override", value=config.HYBRID_KWARGS["identifier_override"],
        help="Promotes a document to rank 1 when the query contains an identifier-shaped "
             "token (order #, SKU, reference code) that appears in it exactly. Measured to "
             "beat plain RRF fusion — see results/tuning_log.md.",
    )

    lex_kwargs = {
        "use_stopwords": use_stopwords, "use_stemming": use_stemming,
        "min_score": lex_min_score, "min_coverage": lex_min_coverage,
    }
    sem_kwargs = {"min_similarity": sem_min_similarity}

    results = compute_all_results(k, lex_kwargs, sem_kwargs, identifier_override)
    scores = {method: score_method(method, results, gold, k) for method in METHODS}

    st.subheader(f"Recall@{k} and MRR@{k} by method")
    cols = st.columns(3)
    for col, method in zip(cols, METHODS):
        s = scores[method]
        col.markdown(f"**{method}**")
        col.metric("Recall@k", pct(s["recall"]))
        col.metric("MRR@k", pct(s["mrr"]))
        col.metric("Final (score.py weights)", pct(s["final"]))

    st.subheader(f"Recall@{k} breakdown by query type")
    breakdown = pd.DataFrame(
        {method: scores[method]["by_type_recall"] for method in METHODS}
    ).T
    st.dataframe(breakdown.style.format(pct), use_container_width=True)

    st.subheader("Abstention — 6 unanswerable queries")
    abstain_row = pd.DataFrame(
        [[scores[m]["abstain_hits"] for m in METHODS]],
        columns=METHODS,
        index=["correctly returned nothing"],
    )
    st.dataframe(abstain_row, use_container_width=True)

    st.subheader("Per-query detail")
    for q in gold:
        relevant = set(q["relevant_ids"])
        with st.expander(f"{q['id']} [{q['type']}] — {q['query']}"):
            for method in METHODS:
                items = results[q["id"]][method]
                if not items:
                    st.markdown(f"**{method}**: _(abstained — no results)_")
                    continue
                lines = []
                for doc_id, score in items:
                    if q["type"] == "unanswerable":
                        mark = " ⚠️ false positive (query is unanswerable)"
                    elif doc_id in relevant:
                        mark = " ✅ relevant"
                    else:
                        mark = ""
                    lines.append(f"- `{doc_id}` ({score:.3f}){mark}")
                st.markdown(f"**{method}**")
                st.markdown("\n".join(lines))

    st.subheader("Write run files")
    st.caption("Writes runs/lexical.json, runs/semantic.json, runs/hybrid.json from the results above.")
    if st.button("Write run files"):
        RUNS_DIR.mkdir(exist_ok=True)
        for method in METHODS:
            run = {q["id"]: [doc_id for doc_id, _ in results[q["id"]][method]] for q in gold}
            with open(RUNS_DIR / f"{method}.json", "w", encoding="utf-8") as f:
                json.dump(run, f, indent=2, ensure_ascii=False)
        st.success("Wrote runs/lexical.json, runs/semantic.json, runs/hybrid.json")


if __name__ == "__main__":
    main()
