# 🏗️ Building a Retail AI Pipeline: Zero to RAG

Welcome! In this project, you will build a complete AI-powered data pipeline for an
E-Commerce company. You'll take messy, real-world product reviews and process them
through four stages — using an AI coding assistant (like **Claude Code** or **Cursor**)
to build each application.

| Step | Stage                  | What you'll build                                       |
| ---- | ---------------------- | ------------------------------------------------------- |
| 1    | **Data Cleansing**     | Normalize and deduplicate raw text                      |
| 2    | **Structured Outputs** | Force an LLM to extract clean JSON from text            |
| 3    | **Search Evaluation**  | Prove which retrieval method actually works, and why    |
| 4    | **Conversational RAG** | Build a chat interface with memory and semantic search  |

You are not being graded on your code. You are being graded on **the number your
system scores in Step 3**, and on whether you can explain *why* it scores that.

---

## 📦 What is provided

Do **not** generate these — they are engineered, and regenerating them will quietly
destroy the exercise.

| File | What it is |
| ---- | ---------- |
| `data/raw_reviews.json` | 70 messy customer reviews, with duplicates, near-duplicates and look-alike identifiers |
| `data/eval_queries.json` | The Gold Set: 30 queries with known correct answers |
| `score.py` | The grader. Computes Recall@k and MRR from a run file |

The corpus is deliberately built so that **keyword search and semantic search fail in
different places**. That is the entire point of Step 3. If you swap in your own data,
every method will score roughly the same and you will learn nothing.

---

## ⚙️ Prerequisites & Setup

### 1. Install dependencies

```bash
pip install streamlit scikit-learn rank-bm25 python-dotenv \
            google-genai openai anthropic sentence-transformers
```

### 2. Set up your API keys

Create a file named `.env` in the root of your project folder. This project is
**provider-agnostic** — choose whichever AI model you want to use.

```dotenv
# Choose your provider for TEXT GENERATION: openai, anthropic, or gemini
PROVIDER=openai

OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GEMINI_API_KEY=your_gemini_key_here

# Embeddings are configured SEPARATELY from generation — see the note below.
# Options: openai | gemini | local
EMBEDDING_PROVIDER=local
```

> ⚠️ **Note on embeddings.** Text generation and embeddings are two different
> services and you do not have to use the same vendor for both. Anthropic does not
> offer an embeddings API at all, so if you set `PROVIDER=anthropic` you must still
> pick `openai`, `gemini`, or `local` for embeddings. `local` uses
> `sentence-transformers/all-MiniLM-L6-v2`, runs on your laptop's CPU, needs no key,
> and is fast enough for 70 documents.

---

## Step 1 — Ingestion & Cleansing

Real-world data is messy. Before the AI ever sees it, it has to be normalized and
deduplicated.

> **Feed this prompt to your AI assistant:**

```text
Create a Python project for an E-commerce data pipeline demonstration.

Read the provided file data/raw_reviews.json. It contains 70 messy customer
reviews, each an object with an "id" and a "text" field. It includes exact
duplicates, near-duplicates (e.g., "Great value for the price, would buy again."
vs "Great value for the price. Would buy again."), varied casings, trailing
spaces, and inconsistent entities (a mix of "customer service", "CS", and
"support").

Build a local Streamlit app called 1_clean_reviews.py.
Implement a 4-step cleansing pipeline:

1. Standardize entities using a dictionary that maps "CS" and "support" to
   "customer service". Match whole words only and case-insensitively, so that
   "CS", "cs" and "support" are all caught but "supported" is not.
2. Normalize text (lowercase, collapse whitespace, strip trailing spaces).
3. Drop exact duplicates.
4. Drop near-duplicates using a TF-IDF vectorizer and cosine similarity with a
   threshold controlled by a Streamlit slider (default 0.85).

CRITICAL: the original "id" of every surviving review must be preserved
untouched. Steps 3 and 4 of this project match retrieved documents against these
ids, so if you renumber or drop them, nothing downstream will work. When you drop
a duplicate, always keep the record with the LOWEST id and discard the others.

Show Streamlit metrics for how many records each step removed. Add an expander to
view the dropped near-duplicate pairs with their similarity scores. Include a
button to save the cleaned data to data/clean_reviews.json, preserving the same
{"id", "text"} structure. No API keys are needed for this script.
```

**Output:** `1_clean_reviews.py` → `data/clean_reviews.json` · **Requires an API key:** No

**Checkpoint:** you should end up with roughly 55 reviews from the original 70. If you
have fewer than 50, your near-duplicate threshold is too aggressive and you are
deleting real content — check the expander to see what it removed.

---

## Step 2 — Structured Extraction

Business automation requires strict data structures (JSON), not chatty paragraphs.

> **Feed this prompt to your AI assistant:**

```text
Build a local Streamlit app called 2_extract_tickets.py that reads the
data/clean_reviews.json file.

The app must support multiple LLM providers based on the PROVIDER variable in the
.env file ("openai", "anthropic", or "gemini"). Initialize the appropriate
official SDK wrapper based on this setting.

For each review, send the text to the chosen provider's latest flagship/flash
model and request ONLY a JSON object in return, using the provider's native
structured output / JSON mode. The JSON schema must strictly include:

- id (carry through the review's original id unchanged)
- category (must be one of: "shipping", "product_quality", "customer_service",
  "billing", "other")
- severity ("low", "medium", "high")
- refund_requested (boolean)
- summary (a one-line summary)

Implement a validation function in Python to ensure the returned JSON matches
this schema and only contains allowed values. If validation fails, automatically
send the errors back to the model for ONE retry attempt. Do not retry forever.

In the Streamlit UI, display a table of the successfully extracted tickets, a
validation log showing which rows passed, retried, or failed, and a button to
download the final table as a CSV. Also save the result to
data/tickets.json.
```

**Output:** `2_extract_tickets.py` → `data/tickets.json` · **Requires an API key:** Yes

**Checkpoint:** with a provider's native structured output, expect almost every row
to pass first try — the provider enforces the schema for you. That is exactly why
you must prove your *own* validator works: inject one hand-written ticket with
`category: "delivery"` and `severity: "critical"` and confirm it is rejected, logged,
and sent for retry. A validator that has never rejected anything is untested code.

---

## Step 3 — Search Evaluation

This is the most important step in the project. Everything downstream is built on
retrieval, and until you have measured retrieval you are guessing.

You will build **three** search methods and prove which one wins on which kind of
question.

> **Feed this prompt to your AI assistant:**

```text
Build a local Streamlit app called 3_measure_search.py to evaluate retrieval
quality over data/clean_reviews.json.

Load the provided Gold Set from data/eval_queries.json. It contains 30 queries,
each with an "id", a "type" (semantic, lexical, hybrid, or unanswerable), the
"query" text, and "relevant_ids" — the review ids that genuinely answer it. The
six unanswerable queries have an EMPTY relevant_ids list on purpose: nothing in
the corpus answers them, and the correct behaviour is to return no results at all.

Implement three retrieval methods over the cleaned reviews:

1. LEXICAL: BM25 keyword search using the rank_bm25 library.
2. SEMANTIC: embed every review once with the embedding provider configured in
   .env (EMBEDDING_PROVIDER = openai, gemini, or local via
   sentence-transformers/all-MiniLM-L6-v2). Cache the vectors to disk so you only
   pay for this once. Rank by cosine similarity.
3. HYBRID: fuse the ranked lists from methods 1 and 2 using Reciprocal Rank
   Fusion, score = sum over lists of 1/(60 + rank).

Every method must be able to return an EMPTY result. Add a sidebar slider for a
minimum score threshold per method; anything below it is discarded. A method that
always returns exactly k documents will score zero on the unanswerable queries.

For every method, write a run file to runs/<method>.json in this exact format:
  { "q01": ["r003", "r001"], "q02": [], ... }
containing every query id in the Gold Set, mapped to its ranked list of review
ids, best first.

In the Streamlit UI:
- A slider for k (default 3).
- The three methods side by side in Streamlit columns, each showing Recall@k
  and MRR.
- A breakdown table of Recall@k grouped by query TYPE (semantic / lexical /
  hybrid), so the differences between the methods are visible.
- An expander per query showing the top-k results of each method, highlighting
  which ones were truly relevant.

No API key is needed if you use EMBEDDING_PROVIDER=local.
```

**Output:** `3_measure_search.py` → `runs/lexical.json`, `runs/semantic.json`, `runs/hybrid.json` · **Requires an API key:** No (with `local`)

### Grading your run

```bash
python score.py runs/*.json --k 3
python score.py runs/hybrid.json --k 3 --per-query
```

`score.py` is independent of your implementation — it only reads run files. Whatever
language, library or vector store you used, everyone lands on the same leaderboard.

### What you must be able to explain

A plain BM25 baseline scores about **44%** overall (43.6% when zero-score results are
filtered out; `get_top_n` with no filter gives 40.3%, because it never abstains).
Before you move on, answer these in writing:

1. BM25 scores near-perfect on the `lexical` queries and close to **5%** on the
   `semantic` ones. Why? Look at query `q09` ("faulty zipper") and the review it is
   supposed to find. What word do they share?
2. Your embedding method will do the reverse — strong on `semantic`, weak on
   `lexical`. Look at `q13` ("SKU AX-7710") and at the review about SKU AX-7701.
   What does an embedding model understand about a part number, and why does a
   look-alike identifier hurt it but not BM25?
3. The corpus contains both *"The battery lasted 3 hours, not the 30 the listing
   promised"* and *"The battery lasted 30 hours exactly as advertised"*. Run both as
   queries against your semantic method. How far apart are their scores? What does
   that tell you about using embeddings for anything numeric?
4. Does hybrid beat both? Check the per-type table before you answer — naive RRF
   often **loses** to pure semantic here, because RRF treats both lists as equally
   trustworthy, and BM25 ranks confidently even when all it matched is one stray
   word. That is not a bug in your code; it is a real production phenomenon:
   fusing in a weak arm can drag down a strong one. Now try *gating* the fusion —
   only include the BM25 list when its top score indicates a real exact-token hit.
   How close can you get your hybrid to the best single method, and which query
   types pay for it? Defend your final choice of method using the per-type table.

---

## Step 4 — Conversational RAG

Now build the chat interface — on top of whichever retrieval method won in Step 3.

> **Feed this prompt to your AI assistant:**

```text
Build a local Streamlit chat app called 4_conversational_rag.py that answers
questions over data/clean_reviews.json using RAG.

Reuse the retrieval method that scored highest in Step 3, including its score
threshold and its ability to return nothing.

Use the PROVIDER variable from .env for text generation, and EMBEDDING_PROVIDER
for the vectors. Load the cached embeddings from Step 3 rather than recomputing.

Implement a multi-turn chat interface. Before running retrieval, send the chat
history plus the user's newest message to the model and prompt it to rewrite them
into a single, standalone search query.

Retrieve the top 3 reviews, pass them as context, and instruct the model to answer
strictly from them, citing the review ids it used.

Three required behaviours:

1. GROUNDING. If retrieval returns nothing, or everything falls below the score
   threshold, do NOT call the generation model at all. Reply "I don't have
   information about that in the reviews." Test this with "do you have a loyalty
   program" and "what is your phone number" — neither is anywhere in the corpus,
   and neither was in the Gold Set you tuned the threshold on.

2. CITATION CHECK. After the model answers, verify in Python that every review id
   it cited was actually in the retrieved context. If it cites an id that was not
   passed to it, flag the answer in the UI with a visible warning. Log how often
   this happens.

3. MEMORY TOGGLE. A sidebar switch that disables the query-rewrite step, so the
   user can watch basic RAG fail on follow-up questions.

Below each chat bubble, add an expander showing the exact standalone query that
was generated, the retrieved documents with their scores, and whether the citation
check passed.
```

**Output:** `4_conversational_rag.py` · **Requires an API key:** Yes

### Try these, in this order

| # | Ask | What should happen |
| - | --- | ------------------ |
| 1 | "What do people say about the ProBlend 900?" | Answers, citing several review ids |
| 2 | "Was it ever damaged?" | **With memory on:** understands "it" = ProBlend 900. **With memory off:** retrieves random damage reviews. Toggle it mid-conversation and watch. |
| 3 | "Do you have a loyalty program?" | Refuses. Does not call the model. If it invents a policy, your threshold is too low. |
| 4 | "How long does the battery last?" | Watch closely — the corpus says both 3 hours and 30 hours. A good answer surfaces the contradiction instead of picking one. |

---

## 🏁 Deliverables

1. The four Streamlit apps.
2. Your three run files from Step 3.
3. Your `score.py` output at `k=3`.
4. Written answers to the four questions in Step 3.

The last item is the one that matters. Anyone can get an AI assistant to produce four
working apps. Knowing *why* your retrieval scores what it scores is the actual skill.
