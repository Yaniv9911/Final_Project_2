# Answers — draft (factual only, numbers from results/evidence.txt)

All numbers below are copied from `results/evidence.txt`, produced by
`python analysis/evidence.py`. Nothing here is retyped from memory or
rounded to fit a narrative. This is a draft for rewriting, not final prose.

## Q1 — q09 "faulty zipper" vs r010, and the semantic-type average

- `tokenize("faulty zipper")` = `['faulty', 'zipper']`
- `tokenize(r010)` = `['the', 'zip', 'jammed', 'the', 'very', 'first', 'time', 'i', 'used', 'it']`
- Intersection = `{}` (empty set). 0 of 2 query tokens appear in r010's token list.
- BM25 score for r010 against this query (phantom-document method,
  stopwords removed, stemming on): **0.0000**.
- Mechanism: "zipper" and "zip" are different tokens under `tokenize` (no
  stemming rule in `stem()` maps "zipper"→"zip"; the stemmer only strips
  suffixes like -s/-es/-ed/-ing/-ies, and "zipper" doesn't end in any of
  those). BM25 requires literal token overlap; with zero shared tokens the
  score is exactly zero, and `lexical_search` abstains on this query.
- Average token-overlap ratio (`|intersection| / |query tokens|`) across
  all 13 (query, relevant_id) pairs among the 10 semantic-type queries:
  **0.124** (13 pairs because q01 has 3 relevant ids and q03 has 2). Full
  per-pair breakdown is in `results/evidence.txt`; 6 of the 13 pairs have
  zero overlap, the highest is 0.333 (q04, `{'the', 'was'}` — both
  stopwords, not content words).
- Mechanism: semantic-type queries are paraphrases by construction (e.g.
  "faulty zipper" for a review saying "the zip jammed"), so low lexical
  overlap is expected and is exactly why `semantic_search` exists as a
  separate method — this part of the data supports a clean story.

## Q2 — near-duplicate identifiers: does the system distinguish them?

**q13 "SKU AX-7710" top-5 semantic results (`min_similarity=0.0`, full
ranking, not what the tuned default returns):**

| rank | id | cosine | text |
|---|---|---|---|
| 1 | r066 | 0.6750 | "sku ax-7701 was perfect, exactly as pictured." |
| 2 | r014 | 0.6129 | "sku ax-7710 arrived in the wrong colour." |
| 3 | r018 | 0.2681 | (unrelated) |
| 4 | r026 | 0.2465 | (unrelated) |
| 5 | r061 | 0.2283 | (unrelated) |

**r014 is the gold answer, but r066 (the wrong SKU, AX-7701) ranks above
it on cosine similarity alone.**

**All three pairwise comparisons (query vs. correct id vs. near-duplicate
id):**

| query | correct id | cosine(correct) | wrong id | cosine(wrong) | cosine gap | BM25(correct) | BM25(wrong) | BM25 gap |
|---|---|---|---|---|---|---|---|---|
| "SKU AX-7710" | r014 | 0.6129 | r066 | 0.6750 | 0.0621 | 5.9004 | 2.7699 | 3.1305 |
| "order #48213" | r011 | 0.6397 | r068 | 0.7293 | 0.0896 | 4.2851 | 1.5022 | 2.7829 |
| "reference 9921-B" | r018 | 0.4891 | r069 | 0.4955 | 0.0063 | 5.4218 | 2.5452 | 2.8766 |

**This does not support a clean "embeddings distinguish near-duplicates"
story — the opposite is measured.** In all three pairs, cosine similarity
ranks the *wrong* near-duplicate document higher than the correct one (by
margins of 0.006–0.09). BM25, in the same three pairs, ranks the correct
document higher by a wide margin every time (2.8–3.1 points). Mechanism:
`tokenize` produces different exact tokens for "ax-7710" vs "ax-7701",
"48213" vs "48231", "9921-b" vs "9912-b" (confirmed distinct via
`retrieval.py`'s own asserts), so BM25's exact-token matching separates
them cleanly; the sentence-transformers embedding model appears to encode
these near-identical sentences (same structure, one digit/character
different) as more similar to each other than either is to the query's
literal digit sequence — the model isn't sensitive to the specific digits
at that resolution. The reason the full pipeline still answers these
queries correctly (100% recall on lexical-type queries, confirmed
separately) is BM25's contribution to the fusion and, for the SKU/order
queries specifically, the hyphenated-token branch of the exact-identifier
override — not the embedding model's own discrimination.

## Q3 — r016 vs r019 gap

- r016: "the battery lasted 3 hours, not the 30 the listing promised."
- r019: "the battery lasted 30 hours exactly as advertised, very happy with it."
- cosine(r016, r019) = **0.8444** (the two reviews are highly similar to
  each other — same topic, overlapping vocabulary, opposite sentiment).
- q17 "battery lasted 30 hours as advertised" (relevant=r019): cosine to
  r016 = 0.7916, cosine to r019 = 0.8685. **Gap = 0.0769**, correct
  direction (favors r019).
- q18 "battery only lasted 3 hours" (relevant=r016): cosine to r016 =
  0.5156, cosine to r019 = 0.4628. **Gap = 0.0529**, correct direction
  (favors r016).
- Mechanism / caveat: both gaps are directionally correct (the model does
  prefer the actually-relevant review each time), but the margins (0.05–0.08)
  are small relative to how similar the two reviews are to each other
  (0.8444). This is a real but modest separation, not a wide margin —
  stated as measured, not amplified.

## Q4 — per-type Recall@3 and MRR

`plain_rrf` = `hybrid_search(identifier_override=False, gate_lexical=False)`.
`gated_hybrid` = `hybrid_search(gate_lexical=True, gate_min_coverage=0.6,
gate_max_doc_freq=5, identifier_override=False)` — the swept-best gate
setting recorded in `results/tuning_log.md`. Both exclude the identifier
override so this table isolates fusion behavior from that mechanism (the
project's actual default hybrid, with the override on, scores higher —
see `results/score_output.txt`).

**Recall@3 by query type:**

| type | lexical | semantic | plain_rrf | gated_hybrid |
|---|---|---|---|---|
| semantic | 5.0% | 88.3% | 88.3% | 88.3% |
| lexical | 100.0% | 87.5% | 100.0% | 100.0% |
| hybrid | 50.0% | 83.3% | 100.0% | 100.0% |

**MRR@3 by query type:**

| type | lexical | semantic | plain_rrf | gated_hybrid |
|---|---|---|---|---|
| semantic | 10.0% | 78.3% | 78.3% | 78.3% |
| lexical | 100.0% | 75.0% | 93.8% | 93.8% |
| hybrid | 50.0% | 83.3% | 91.7% | 91.7% |

**Observation, stated plainly**: `plain_rrf` and `gated_hybrid` are
byte-identical in every cell of both tables. Mechanism, consistent with
`results/tuning_log.md`'s step 3 finding: the gate's condition (high
coverage + a low-document-frequency shared token) is satisfied by every
lexical top candidate that already passes `lexical_search`'s own
abstention threshold on this gold set, so gating removes nothing beyond
what the threshold already removes here. This table does not show gating
adding value — consistent with, not contradicting, the earlier tuning
result.
