# Customer Review RAG Pipeline

A four-step pipeline over a small customer-reviews corpus: clean the raw
reviews, extract structured tickets from them via an LLM, measure and tune
three retrieval methods against a gold query set, and ship a conversational
RAG chat app that uses the winning configuration.

## Apps

Run any of these from the project root with `streamlit run <file>` (each
is self-contained and can be run independently, provided its input file
already exists from the previous step):

### `1_clean_reviews.py`
Reads `data/raw_reviews.json` (70 reviews) and runs a 4-step cleaning
pipeline: entity standardization (CS/support → "customer service"),
normalization (lowercase, whitespace collapse — identifiers like `#48213`
survive intact), exact-duplicate removal, and near-duplicate removal
(TF-IDF cosine similarity, threshold adjustable via a sidebar slider,
default 0.85). Shows per-step counts, a dropped-near-duplicates expander,
and a gold-set safety check (every review id referenced by
`data/eval_queries.json` must survive). A "Save" button writes
`data/clean_reviews.json` (55 reviews at the default threshold).

### `2_extract_tickets.py`
Reads `data/clean_reviews.json` and extracts a structured ticket
(`category`, `severity`, `refund_requested`, `summary`) from each review
via the configured LLM provider's native structured-output feature, then
re-validates every response in pure Python (`validate_ticket`) — never
trusting the provider's own schema enforcement — with one bounded retry on
validation failure. A "Run extraction" button is the only place the model
is called; results are cached in `st.session_state` and written to
`data/tickets.json`. A "Test validator" button proves the validator
actually rejects bad input, with no API call.

### `3_measure_search.py`
Reads `data/clean_reviews.json` and `data/eval_queries.json` (30 gold
queries) and measures `retrieval.py`'s three search methods —
`lexical_search` (BM25), `semantic_search` (embeddings), `hybrid_search`
(Reciprocal Rank Fusion) — using the exact same Recall@k / MRR@k / abstain
definitions as `score.py`. Sidebar sliders (defaulting to the values in
`config.py`, see below) let you move away from the tuned configuration
live. A "Write run files" button produces `runs/lexical.json`,
`runs/semantic.json`, `runs/hybrid.json` as the honest output of the
retrieval functions at whatever the sliders currently say.

### `4_conversational_rag.py`
A chat UI over `data/clean_reviews.json` using `retrieval.py`'s
`hybrid_search` at the **fixed, winning configuration from `config.py`**
(no sliders — this is the shipped system, not a measurement tool). Per
turn: an optional query rewrite step (toggleable "Conversation memory")
resolves pronouns using chat history; retrieval returns the top 3; if
retrieval finds nothing, the app prints a fixed refusal message and the
generation model is never called (the code path is structurally
unreachable, not just conditionally skipped); otherwise the model
generates an answer grounded only in the retrieved reviews, citing review
ids as `[rNNN]` and stating any contradiction between reviews rather than
picking a side. Every citation is regex-extracted and checked against what
was actually retrieved; a failure shows a red warning, logs to
`logs/citation_failures.jsonl`, and increments a sidebar counter. Every
assistant message has an expander showing the standalone query, retrieved
ids/scores, the raw context sent to the model, and the citation check
result.

## Setup

Requires the `rag_env` conda environment (Python 3.11, already provisioned
— see `CLAUDE.md` and `requirements.txt`, which documents the environment
rather than installing it). Copy `.env.example` to `.env` and fill in:

| Variable | Used by | Notes |
|---|---|---|
| `PROVIDER` | `2_extract_tickets.py`, `4_conversational_rag.py` | `gemini`, `openai`, or `anthropic` — which SDK generates text/tickets. This project runs on `gemini`; the other two SDKs aren't installed, but their code paths exist (each import lazily, so a missing SDK never breaks the other providers). |
| `OPENAI_API_KEY` | — | Only needed if `PROVIDER=openai`. |
| `ANTHROPIC_API_KEY` | — | Only needed if `PROVIDER=anthropic`. |
| `GEMINI_API_KEY` | — | Only needed if `PROVIDER=gemini` (the default). |
| `EMBEDDING_PROVIDER` | `retrieval.py` | `local` (sentence-transformers/all-MiniLM-L6-v2, no key needed — the default), `openai`, or `gemini`. |

`.env` is gitignored and never committed.

## The winning retrieval configuration

`config.py` is the single shared source of truth for the tuned
configuration, imported by both `3_measure_search.py` (as its slider
defaults) and `4_conversational_rag.py` (as the fixed, shipped settings)
so the measured system and the shipped system cannot silently drift apart.
Full tuning history, including every configuration that was tried and
rejected, is in `results/tuning_log.md`.

```python
TOP_K = 3

LEXICAL_KWARGS = {
    "use_stopwords": True,
    "use_stemming": True,
    "min_score": 0.30,       # normalized BM25 score threshold
    "min_coverage": 0.50,    # query-term coverage threshold
}

SEMANTIC_KWARGS = {
    "min_similarity": 0.30,  # cosine similarity threshold
}

HYBRID_KWARGS = {
    "lexical_kwargs": LEXICAL_KWARGS,
    "semantic_kwargs": SEMANTIC_KWARGS,
    "weight_lexical": 1.0,
    "weight_semantic": 1.0,
    "gate_lexical": False,           # measured: gating never beat plain RRF here
    "identifier_override": True,     # measured: the one variant that beat plain RRF
}
```

Hybrid (plain RRF + the exact-identifier override) is the winning method.
Gating and down-weighting the lexical arm were both measured and
**rejected** — full evidence in `results/tuning_log.md`.

## Final score

Reproduced with `python score.py runs/*.json --k 3` (verified from a fresh
`git clone` of this repo — see verification section below):

```
  GOLD SET: 30 queries   k = 3

  METHOD         FINAL   RECALL      MRR   ABSTAIN
  ------------------------------------------------
  hybrid         91.5%    95.1%    91.0%     83.3%
  semantic       83.6%    86.8%    78.5%     83.3%
  lexical        59.0%    47.9%    50.0%    100.0%

  BREAKDOWN BY QUERY TYPE  (recall@3)

  METHOD          SEMANTIC     LEXICAL      HYBRID
  ------------------------------------------------
  hybrid             88.3%      100.0%      100.0%
  semantic           88.3%       87.5%       83.3%
  lexical             5.0%      100.0%       50.0%
```

Full per-query breakdown: `results/score_output.txt`.

## Provided files are unmodified

`score.py`, `data/raw_reviews.json`, and `data/eval_queries.json` are
byte-identical to what was originally provided — verified against the
first commit of this repository (`360d758`, "provided inputs"):

```
$ git diff --stat 360d758 HEAD -- score.py data/raw_reviews.json data/eval_queries.json
(no output — zero differences)

$ git log --oneline -- score.py data/raw_reviews.json data/eval_queries.json
360d758 provided inputs
```

The second command confirms these three files have never been touched by
any commit after the first one — not just that the current diff happens
to be empty.

## Deliverables

- Four apps: `1_clean_reviews.py`, `2_extract_tickets.py`,
  `3_measure_search.py`, `4_conversational_rag.py`.
- `runs/lexical.json`, `runs/semantic.json`, `runs/hybrid.json` — the
  honest output of `retrieval.py`'s functions at the winning configuration,
  never hand-edited.
- `results/score_output.txt` — `score.py`'s full output (summary + per-query).
- `results/tuning_log.md` — every configuration tried during tuning, its
  score, and why it moved (including the ones that made things worse).
- `results/evidence.txt` / `results/ANSWERS.md` — measured evidence and
  factual draft answers for four follow-up questions about retrieval
  behavior (`analysis/evidence.py`).

## Where this breaks at 41,000 reviews instead of 70

- **O(n²) — the near-duplicate step in `1_clean_reviews.py`.**
  `remove_near_duplicates` calls `sklearn`'s `cosine_similarity(X)` on the
  full TF-IDF matrix, producing a dense N×N similarity matrix. At
  N=41,000 that's 1,681,000,000 cells — **13.4 GB** as float64 (6.7 GB even
  at float32), before even accounting for the O(N²) time to compute it. At N=55 this is
  free; at N=41,000 it doesn't fit in memory on a laptop and needs
  replacing with blocking/LSH or an approximate-nearest-neighbor index
  that never materializes the full pairwise matrix.
- **Retrieval rebuilds its index from scratch on every single call — no
  caching, unlike embeddings.** `lexical_search` constructs a fresh
  `BM25Okapi` index every call, tokenizing the *entire* corpus (the query
  is appended as one extra "phantom" document so both the real scores and
  the normalization reference come from the same well-defined index — a
  single build, but still a full-corpus one, every call). On top of that,
  the shipped default (`identifier_override=True`) does a **second**,
  completely separate full-corpus tokenization pass on every
  `hybrid_search` call, just to look for an exact identifier match. At 55
  documents both are instant; at 41,000 this turns every single chat turn
  (and every cell of a threshold sweep — 230+ grid cells × 30 queries in
  `analysis/sweep.py`) into two full-corpus re-tokenizations, multiplying
  what should be a one-time cost by every query ever issued. This needs to
  become a persistent, built-once index (BM25 index + a document-frequency
  table) reused across calls, the same way embeddings already are.
- **Embedding cost — scales, but no longer instant.** Embeddings are
  cached to disk keyed by a corpus hash, so this is a one-time cost, not a
  per-query one — but at 41,000 reviews, embedding all of them with
  `sentence-transformers/all-MiniLM-L6-v2` on CPU would take real minutes
  (vs. sub-second for 55), and the resulting vectors
  (41,000 × 384 float32 ≈ 63 MB) are a non-trivial but manageable
  in-memory footprint, held as a single dense array via `np.load`.
- **`2_extract_tickets.py`'s per-review LLM calls — linear cost, but the
  constant makes it impractical.** One (or two, with retry) API call per
  review means 41,000–82,000 calls. The free-tier Gemini key used for
  testing this project exhausted its 20-request/day quota after roughly a
  dozen reviews — at 41,000 reviews this step would require a paid tier,
  real batching/backoff, and would still take hours of wall-clock time,
  not the couple of minutes it took for 55.
- **What does scale fine as-is**: exact-duplicate removal in
  `1_clean_reviews.py` (a single dict pass, O(n)); the corpus and gold
  query set held fully in memory (a few tens of MB of JSON at 41,000
  reviews); the RRF fusion and cosine-similarity-against-a-cached-matrix
  steps inside `semantic_search` (both O(n) per query, not O(n²)).
