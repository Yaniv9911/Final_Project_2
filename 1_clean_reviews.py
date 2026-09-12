"""Streamlit app — Step 1: clean raw_reviews.json.

Pipeline: entity standardization -> normalization -> exact-duplicate
removal -> near-duplicate removal (TF-IDF cosine similarity).

Run with: streamlit run "1_clean_reviews.py"
"""
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RAW_PATH = Path("data/raw_reviews.json")
GOLD_PATH = Path("data/eval_queries.json")
CLEAN_PATH = Path("data/clean_reviews.json")

ENTITY_PATTERN = re.compile(r"\b(?:CS|support)\b", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")


# --------------------------------------------------------------------------
# Pipeline steps (pure functions — no Streamlit calls, so they're unit-testable
# and safe to import from test_clean_reviews.py).
# --------------------------------------------------------------------------

def standardize_entities(text: str) -> str:
    """Step 1: replace whole-word CS/support (any case) with 'customer service'."""
    return ENTITY_PATTERN.sub("customer service", text)


def normalize_text(text: str) -> str:
    """Step 2: lowercase, collapse whitespace, strip ends. Keeps punctuation/digits."""
    text = text.lower()
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def remove_exact_duplicates(records):
    """Step 3: keep the lowest-id record per unique normalized text.

    `records` must already be sorted ascending by id, so the first record
    seen for a given text is always the one with the lowest id.
    Returns (survivors, redirect) where redirect maps a dropped id to the
    id of the survivor that replaced it.
    """
    survivors = []
    redirect = {}
    kept_id_for_text = {}
    for record in records:
        text = record["text"]
        if text in kept_id_for_text:
            redirect[record["id"]] = kept_id_for_text[text]
        else:
            kept_id_for_text[text] = record["id"]
            survivors.append(record)
    return survivors, redirect


def remove_near_duplicates(records, threshold):
    """Step 4: greedy, transitive-safe near-duplicate removal.

    `records` must already be sorted ascending by id. Each candidate is
    compared only against the survivors kept so far (not pairwise against
    everything), so a chain A~B~C collapses onto the single lowest id.
    Returns (survivors, dropped_pairs).
    """
    if len(records) < 2:
        return records, []

    texts = [r["text"] for r in records]
    matrix = TfidfVectorizer().fit_transform(texts)
    similarity = cosine_similarity(matrix)

    survivor_indices = []
    dropped_pairs = []
    for i, record in enumerate(records):
        best_j, best_score = None, -1.0
        for j in survivor_indices:
            score = similarity[i, j]
            if score > best_score:
                best_score, best_j = score, j
        if survivor_indices and best_score >= threshold:
            dropped_pairs.append({
                "dropped_id": record["id"],
                "dropped_text": record["text"],
                "kept_id": records[best_j]["id"],
                "kept_text": records[best_j]["text"],
                "similarity": float(best_score),
            })
        else:
            survivor_indices.append(i)

    survivors = [records[i] for i in survivor_indices]
    return survivors, dropped_pairs


def resolve_redirect(original_id, redirect):
    """Follow (possibly chained) redirects to find the surviving id."""
    seen = set()
    current = original_id
    while current in redirect and current not in seen:
        seen.add(current)
        current = redirect[current]
    return current


# --------------------------------------------------------------------------
# Data loading (cached — cheap local JSON reads, but cached per rule 6 anyway)
# --------------------------------------------------------------------------

@st.cache_data
def load_raw_reviews():
    with open(RAW_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_gold_relevant_ids():
    with open(GOLD_PATH, encoding="utf-8") as f:
        gold = json.load(f)
    relevant_ids = set()
    for query in gold:
        relevant_ids.update(query["relevant_ids"])
    return relevant_ids


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Clean Reviews", layout="wide")
    st.title("Step 1 — Clean Reviews")

    raw_reviews = sorted(load_raw_reviews(), key=lambda r: r["id"])
    n_raw = len(raw_reviews)

    step1 = [{"id": r["id"], "text": standardize_entities(r["text"])} for r in raw_reviews]
    step2 = [{"id": r["id"], "text": normalize_text(r["text"])} for r in step1]
    step3, exact_redirect = remove_exact_duplicates(step2)

    threshold = st.slider(
        "Near-duplicate cosine similarity threshold",
        min_value=0.50, max_value=1.00, value=0.85, step=0.01,
    )
    step4, near_dup_pairs = remove_near_duplicates(step3, threshold)

    st.subheader("Pipeline counts")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Raw reviews", n_raw)
    c2.metric("1. Standardized", len(step1), delta=len(step1) - n_raw)
    c3.metric("2. Normalized", len(step2), delta=len(step2) - len(step1))
    c4.metric("3. Exact dedup", len(step3), delta=len(step3) - len(step2))
    c5.metric("4. Near dedup", len(step4), delta=len(step4) - len(step3))
    st.metric("Final review count", len(step4))

    with st.expander(f"Dropped near-duplicate pairs ({len(near_dup_pairs)})"):
        if near_dup_pairs:
            df = pd.DataFrame(near_dup_pairs)[
                ["dropped_id", "kept_id", "similarity", "dropped_text", "kept_text"]
            ]
            st.dataframe(df, use_container_width=True)
        else:
            st.write("No near-duplicates dropped at this threshold.")

    st.subheader("Gold set safety check")
    relevant_ids = load_gold_relevant_ids()
    final_ids = {r["id"] for r in step4}

    redirect = dict(exact_redirect)
    for pair in near_dup_pairs:
        redirect[pair["dropped_id"]] = pair["kept_id"]

    missing = [
        (rid, resolve_redirect(rid, redirect))
        for rid in sorted(relevant_ids)
        if rid not in final_ids
    ]

    if not missing:
        st.success(
            f"All {len(relevant_ids)} review ids referenced by the gold set "
            f"survive cleaning."
        )
    else:
        lines = "\n".join(
            f"- `{original}` is missing — replaced by survivor `{survivor}`"
            for original, survivor in missing
        )
        st.error(
            f"{len(missing)} gold-referenced id(s) were dropped during cleaning:\n\n{lines}"
        )

    st.subheader("Save")
    if st.button("Save data/clean_reviews.json"):
        ordered = sorted(step4, key=lambda r: r["id"])
        with open(CLEAN_PATH, "w", encoding="utf-8") as f:
            json.dump(ordered, f, indent=2, ensure_ascii=False)
        st.success(f"Wrote {len(ordered)} records to {CLEAN_PATH}")


if __name__ == "__main__":
    main()
