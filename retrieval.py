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


def _prepare_tokens(text, use_stopwords):
    tokens = tokenize(text)
    return remove_stopwords(tokens) if use_stopwords else tokens


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

def lexical_search(query, k=3, use_stopwords=True, min_score=0.3, min_coverage=0.5, corpus=None):
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
      (b) coverage: the fraction of the query's content tokens (stopwords
          always removed for this check) that the top document contains is
          below `min_coverage`.
    """
    reviews = corpus if corpus is not None else load_reviews()
    if not reviews:
        return []

    doc_tokens = [_prepare_tokens(r["text"], use_stopwords) for r in reviews]
    query_tokens = _prepare_tokens(query, use_stopwords)
    query_content_tokens = remove_stopwords(tokenize(query))

    if not query_tokens or not query_content_tokens:
        return []

    bm25 = BM25Okapi(doc_tokens + [query_tokens])
    all_scores = bm25.get_scores(query_tokens)
    doc_scores, reference_score = all_scores[:-1], float(all_scores[-1])

    order = np.argsort(doc_scores)[::-1][:k]
    top_idx = order[0]
    top_score = float(doc_scores[top_idx])

    coverage = len(set(query_content_tokens) & set(doc_tokens[top_idx])) / len(query_content_tokens)
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


def semantic_search(query, k=3, min_similarity=0.36, corpus=None):
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
# Hybrid search — Reciprocal Rank Fusion over the two ranked lists.
# --------------------------------------------------------------------------

def hybrid_search(query, k=3, rrf_k=60, candidate_pool=20, lexical_kwargs=None, semantic_kwargs=None, corpus=None):
    """Returns [(review_id, rrf_score), ...], best first, len <= k.
    Abstains (returns []) only when both arms return []."""
    lexical_kwargs = lexical_kwargs or {}
    semantic_kwargs = semantic_kwargs or {}

    lexical_results = lexical_search(query, k=candidate_pool, corpus=corpus, **lexical_kwargs)
    semantic_results = semantic_search(query, k=candidate_pool, corpus=corpus, **semantic_kwargs)

    if not lexical_results and not semantic_results:
        return []

    rrf_scores = {}
    for rank, (doc_id, _) in enumerate(lexical_results, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    for rank, (doc_id, _) in enumerate(semantic_results, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

    ranked = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return ranked[:k]
