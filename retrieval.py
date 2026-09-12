"""Pure-Python retrieval library — no Streamlit import.

Exposes three ranked search functions over data/clean_reviews.json:
lexical_search (BM25), semantic_search (embeddings), hybrid_search (RRF
fusion of the two). 3_measure_search.py measures these; 4_conversational_rag.py
imports the exact same code, so what's measured is what ships.
"""
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

load_dotenv()

CLEAN_REVIEWS_PATH = Path("data/clean_reviews.json")
CACHE_DIR = Path("cache")

LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


# --------------------------------------------------------------------------
# Tokenization — used identically for documents and queries everywhere.
# --------------------------------------------------------------------------

def tokenize(text):
    """Lowercase and extract tokens, keeping identifiers whole: internal
    digits/hyphens/dots stay inside one token. "#48213" -> "48213",
    "$12.99" -> "12.99", "ax-7710" -> "ax-7710", "9921-B" -> "9921-b",
    "save20" -> "save20". A leading "#"/"$" is simply not matched, so it's
    dropped while the identifier itself survives intact."""
    return TOKEN_PATTERN.findall(text.lower())


assert tokenize("#48213") == ["48213"]
assert tokenize("$12.99") == ["12.99"]
assert tokenize("ax-7710") == ["ax-7710"]
assert tokenize("9921-B") == ["9921-b"]
assert tokenize("save20") == ["save20"]
assert tokenize("ax-7710") != tokenize("ax-7701")
assert tokenize("$12.99") != tokenize("$12.90")


def remove_stopwords(tokens):
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS]


def stem(token):
    """Minimal, dependency-free suffix stripper (no nltk installed in
    rag_env). Tokens containing a digit are returned unchanged — stemming
    a product code or price is meaningless and would only risk merging
    distinct identifiers."""
    if any(c.isdigit() for c in token):
        return token
    if len(token) <= 3:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith(("ses", "xes", "zes", "ches", "shes")) and len(token) > 4:
        return token[:-2]
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def _prepare_tokens(text, use_stopwords, use_stemming=False):
    tokens = tokenize(text)
    if use_stopwords:
        tokens = remove_stopwords(tokens)
    if use_stemming:
        tokens = [stem(t) for t in tokens]
    return tokens


# --------------------------------------------------------------------------
# Corpus loading
# --------------------------------------------------------------------------

def load_reviews():
    with open(CLEAN_REVIEWS_PATH, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Lexical search — BM25Okapi, with a normalized-score + coverage abstention
# gate applied to the top candidate only.
# --------------------------------------------------------------------------

def lexical_search(query, k=3, use_stopwords=True, use_stemming=True, min_score=0.3, min_coverage=0.5, corpus=None):
    """Returns [(review_id, bm25_score), ...], best first, len <= k.

    Abstains (returns []) when the top candidate fails either:
      (a) normalized score: top_score / reference_score is below `min_score`,
          where reference_score is the BM25 score a "phantom" document made
          of exactly the query's own tokens would get against itself, scored
          under the SAME BM25 index as the real corpus (query tokens
          appended as one extra document, so idf/avgdl stay well-defined —
          a true single-document index degenerates: with corpus_size=1 every
          term's document-frequency trivially equals corpus_size, making
          BM25's idf negative).
      (b) coverage: the fraction of the query's content tokens that the top
          document contains is below `min_coverage`. Coverage is always
          computed on plain stopword-stripped, unstemmed tokens regardless
          of `use_stopwords`/`use_stemming` — it means "does this document
          literally contain the words the user typed", independent of
          whatever normalization BM25 itself is configured to use.
    """
    reviews = corpus if corpus is not None else load_reviews()
    if not reviews:
        return []

    doc_tokens = [_prepare_tokens(r["text"], use_stopwords, use_stemming) for r in reviews]
    query_tokens = _prepare_tokens(query, use_stopwords, use_stemming)

    if not query_tokens:
        return []

    bm25 = BM25Okapi(doc_tokens + [query_tokens])
    all_scores = bm25.get_scores(query_tokens)
    doc_scores, reference_score = all_scores[:-1], float(all_scores[-1])

    order = np.argsort(doc_scores)[::-1][:k]
    top_idx = order[0]
    top_score = float(doc_scores[top_idx])

    query_content_tokens = remove_stopwords(tokenize(query))
    if not query_content_tokens:
        return []
    doc_content_tokens = remove_stopwords(tokenize(reviews[top_idx]["text"]))
    coverage = len(set(query_content_tokens) & set(doc_content_tokens)) / len(query_content_tokens)

    normalized_score = top_score / reference_score if reference_score > 0 else 0.0

    if normalized_score < min_score or coverage < min_coverage:
        return []

    return [(reviews[i]["id"], float(doc_scores[i])) for i in order if doc_scores[i] > 0]


# --------------------------------------------------------------------------
# Semantic search — embeddings, cached to disk keyed by a hash of the corpus.
# --------------------------------------------------------------------------

_local_model = None
_embedding_cache = {}  # in-memory: (provider, model, corpus_hash) -> (ids, vectors)


def _corpus_hash(reviews):
    payload = json.dumps(sorted((r["id"], r["text"]) for r in reviews), ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _embed_local(texts):
    global _local_model
    from sentence_transformers import SentenceTransformer

    if _local_model is None:
        _local_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
    return _local_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)


def _embed_openai(texts):
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=texts)
    vectors = np.array([d.embedding for d in response.data], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def _embed_gemini(texts):
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.embed_content(model=GEMINI_EMBEDDING_MODEL, contents=texts)
    vectors = np.array([e.values for e in response.embeddings], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def _embedding_model_name(provider):
    return {
        "local": LOCAL_EMBEDDING_MODEL,
        "openai": OPENAI_EMBEDDING_MODEL,
        "gemini": GEMINI_EMBEDDING_MODEL,
    }[provider]


def _embed_texts(texts, provider):
    if provider == "local":
        return _embed_local(texts)
    elif provider == "openai":
        return _embed_openai(texts)
    elif provider == "gemini":
        return _embed_gemini(texts)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r} (expected local, openai, or gemini)")


def _embed_corpus(reviews, provider):
    model_name = _embedding_model_name(provider)
    corpus_hash = _corpus_hash(reviews)
    cache_key = (provider, model_name, corpus_hash)
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    safe_model = model_name.replace("/", "_")
    cache_path = CACHE_DIR / f"embeddings_{provider}_{safe_model}_{corpus_hash}.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        ids, vectors = data["ids"], data["vectors"]
    else:
        ids = np.array([r["id"] for r in reviews])
        vectors = _embed_texts([r["text"] for r in reviews], provider)
        CACHE_DIR.mkdir(exist_ok=True)
        np.savez(cache_path, ids=ids, vectors=vectors)

    _embedding_cache[cache_key] = (ids, vectors)
    return ids, vectors


def semantic_search(query, k=3, min_similarity=0.30, corpus=None):
    """Returns [(review_id, cosine_similarity), ...], best first, len <= k.
    Abstains (returns []) when the top similarity is below `min_similarity`."""
    reviews = corpus if corpus is not None else load_reviews()
    if not reviews:
        return []

    provider = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
    ids, vectors = _embed_corpus(reviews, provider)
    query_vector = _embed_texts([query], provider)[0]

    similarities = vectors @ query_vector
    order = np.argsort(similarities)[::-1][:k]

    if float(similarities[order[0]]) < min_similarity:
        return []

    return [(str(ids[i]), float(similarities[i])) for i in order]


# --------------------------------------------------------------------------
# Hybrid search — Reciprocal Rank Fusion over the two ranked lists, with
# optional gating, per-arm weighting, and an exact-identifier override.
# All three are generic, corpus-wide rules — none branch on a specific
# query id or query text.
# --------------------------------------------------------------------------

def _content_tokens(text):
    return remove_stopwords(tokenize(text))


def _corpus_content_doc_frequencies(corpus):
    from collections import Counter

    df = Counter()
    for r in corpus:
        df.update(set(_content_tokens(r["text"])))
    return df


def _is_genuine_exact_hit(query, corpus, top_doc_id, min_coverage, max_doc_freq):
    """Gate for variant (b): does the lexical top candidate share a high
    fraction of the query's content tokens, including at least one token
    that is rare/distinctive across the corpus (low document frequency,
    not a common word)? Uses the same plain stopword-stripped tokens as
    lexical_search's own coverage check — a property of token overlap and
    corpus-wide statistics only, never of specific query text."""
    top_doc = next(r for r in corpus if r["id"] == top_doc_id)
    query_content = set(_content_tokens(query))
    if not query_content:
        return False
    doc_content = set(_content_tokens(top_doc["text"]))
    shared = query_content & doc_content
    if len(shared) / len(query_content) < min_coverage:
        return False
    df = _corpus_content_doc_frequencies(corpus)
    return any(df[t] <= max_doc_freq for t in shared)


def _looks_like_identifier(token):
    """A token "looks like an identifier" if it's purely digits/decimal
    with at least 4 digits (e.g. "48213", "12.99") or contains a hyphen
    together with a digit (e.g. "ax-7710", "9921-b") — a property of token
    shape only. The 4-digit floor on the pure-numeric branch matters: a
    bare short number like "30" or "900" is a common quantity (hours,
    product model numbers used as everyday nouns), not a distinctive
    identifier, and treating it as one caused a measured misfire — see
    tuning_log.md step 3, where "30" (from "battery lasted 30 hours")
    matched a document that happened to also contain "30" but wasn't the
    relevant one, demoting the correct answer from rank 1 to rank 2."""
    digits = "".join(c for c in token if c.isdigit())
    if not digits:
        return False
    if token.replace(".", "").isdigit():
        return len(digits) >= 4
    return "-" in token


def _identifier_override_target(query, corpus):
    """Variant (d): if the query contains an identifier-shaped token that
    appears as an exact token in some document, return that document's id
    (the one matching the most such tokens; ties broken by lowest id for
    determinism) — else None. Generic rule based on token shape and exact
    match, never on specific query text."""
    identifier_tokens = [t for t in tokenize(query) if _looks_like_identifier(t)]
    if not identifier_tokens:
        return None

    best_id, best_matches = None, 0
    for r in sorted(corpus, key=lambda r: r["id"]):
        doc_tokens = set(tokenize(r["text"]))
        matches = sum(1 for t in identifier_tokens if t in doc_tokens)
        if matches > best_matches:
            best_matches, best_id = matches, r["id"]
    return best_id


def hybrid_search(query, k=3, rrf_k=60, candidate_pool=20,
                   lexical_kwargs=None, semantic_kwargs=None,
                   weight_lexical=1.0, weight_semantic=1.0,
                   gate_lexical=False, gate_min_coverage=0.8, gate_max_doc_freq=3,
                   identifier_override=True, corpus=None):
    """Returns [(review_id, rrf_score), ...], best first, len <= k.
    Abstains (returns []) only when both arms return [].

    Variants, all independently controllable so each can be measured on
    its own or in combination — see analysis/hybrid_variants.py and
    results/tuning_log.md for the measured comparison:
      (a) plain RRF — weight_lexical=weight_semantic=1.0, gate_lexical=False,
          identifier_override=False. Pass identifier_override=False to get
          exactly this (it's no longer the default — see (d)).
      (b) gated RRF — gate_lexical=True: the lexical arm is dropped from
          the fusion (falling back to semantic-only) unless its own top
          candidate is a "genuine exact-token hit" (see
          _is_genuine_exact_hit). If semantic is also empty, still abstains.
          MEASURED: never beat plain RRF on this gold set at any swept
          (gate_min_coverage, gate_max_doc_freq) — ties at best, because
          every lexical top candidate that survives lexical_search's own
          abstention threshold already happens to be a genuine hit here.
          Kept available, not adopted as default.
      (c) weighted RRF — weight_lexical / weight_semantic multiply each
          arm's 1/(rrf_k+rank) term before summing. MEASURED: any
          weight_lexical < 1.0 strictly hurt recall/MRR on this gold set —
          down-weighting lexical only threw away correct hits. Kept
          available, not adopted as default.
      (d) exact-identifier override — identifier_override=True (now the
          DEFAULT): if the query contains an identifier-shaped token that
          exactly matches a document, that document is promoted to rank 1
          (added if it wasn't already in the fused list). Applied only
          when the query isn't already abstaining, so it can never
          manufacture a result out of thin air when both arms are
          genuinely empty. MEASURED: the only variant that beat plain RRF
          (FINAL 90.3% -> 91.5%, MRR 86.8% -> 91.0%, recall/abstain
          unchanged) — fixed two order-number queries (q11, q21) where RRF
          fusion ranked the right document 2nd instead of 1st. Adopted as
          the new default. (An earlier, looser identifier definition — any
          pure-digit token — caused a real regression on q17/q19 via
          common small numbers like "30"/"900"; fixed by requiring >=4
          digits for the pure-numeric branch. See tuning_log.md.)
    """
    reviews = corpus if corpus is not None else load_reviews()
    lexical_kwargs = lexical_kwargs or {}
    semantic_kwargs = semantic_kwargs or {}

    lexical_results = lexical_search(query, k=candidate_pool, corpus=reviews, **lexical_kwargs)
    semantic_results = semantic_search(query, k=candidate_pool, corpus=reviews, **semantic_kwargs)

    if gate_lexical and lexical_results:
        top_id = lexical_results[0][0]
        if not _is_genuine_exact_hit(query, reviews, top_id, gate_min_coverage, gate_max_doc_freq):
            lexical_results = []

    if not lexical_results and not semantic_results:
        return []

    rrf_scores = {}
    for rank, (doc_id, _) in enumerate(lexical_results, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + weight_lexical / (rrf_k + rank)
    for rank, (doc_id, _) in enumerate(semantic_results, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + weight_semantic / (rrf_k + rank)

    ranked = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    if identifier_override:
        target_id = _identifier_override_target(query, reviews)
        if target_id is not None:
            ranked = [item for item in ranked if item[0] != target_id]
            promoted_score = ranked[0][1] + 1.0 if ranked else 1.0
            ranked.insert(0, (target_id, promoted_score))

    return ranked[:k]
