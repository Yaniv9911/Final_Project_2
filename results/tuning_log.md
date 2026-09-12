# Tuning log

Every configuration actually tried, in order, scored with `score.py`'s own
`evaluate()` (imported in `analysis/sweep.py`, never re-implemented). FINAL
uses `score.py`'s weights: `0.50*recall + 0.30*mrr + 0.20*abstain`, `k=3`.

## Step 1 — threshold sweep

**Starting point** (from the original build): lexical `min_score=0.30,
min_coverage=0.50` → FINAL 56.2%. semantic `min_similarity=0.36` → FINAL
81.3%. hybrid (plain RRF, using those inputs) → FINAL 87.8%.

### Semantic `min_similarity` — 1D sweep, step 0.01, range 0.20–0.50

| threshold | FINAL | RECALL | MRR | ABSTAIN |
|---|---|---|---|---|
| 0.20–0.27 | 66.9–70.3% | 86.8% | 78.5% | 0–16.7% |
| 0.28 | 73.6% | 86.8% | 78.5% | 33.3% |
| 0.29–0.31 | **83.6%** | 86.8% | 78.5% | 83.3% |
| 0.32–0.35 | 78.0–80.3% | 80.6–82.6% | 70.1–74.3% | 83.3% |
| 0.36–0.42 | 81.3% | 80.6% | 70.1% | 100.0% |
| 0.43–0.50 | 60.5–78.0% | 51.4–76.4% | 49.3–66.0% | 100.0% |

**Peak: 0.29–0.31, FINAL 83.6%.** Confirmed with a fine 0.01-step scan
around the coarse peak — this is a genuine 3-point contiguous plateau, not
a 1-sample fluke (my first pass, at step 0.02, only sampled the single
point 0.30 in this region and initially flagged it as an isolated spike;
the finer scan showed it's flat across [0.29, 0.31]). Recall/MRR are
*identical* (86.8%/78.5%) from 0.20 through 0.31 — raising the threshold in
this range costs **zero** real hits, it only turns off false-positive
returns on unanswerable queries one at a time (abstain climbs
0%→16.7%→33.3%→83.3%). The move from 83.3%→100% abstain (threshold 0.32+)
is NOT free: it costs 2 genuine semantic-type hits (q03, q07), dropping
FINAL from 83.6% to ~78–81%. **Chosen: 0.30** (center of the plateau) —
a real, mechanism-explained improvement over the original 0.36 guess, not
overfitting: the win at 0.30 comes from recall-neutral abstention, while
the win at 0.36+ trades 2 real hits for 1 more abstain and is strictly
worse on FINAL. Applied to `retrieval.py`'s `semantic_search` default and
`3_measure_search.py`'s slider default.

### Lexical `(min_score, min_coverage)` — 2D grid, step 0.05 x 0.1

Full grid in `analysis/sweep.py`'s output. Peak: `min_score=0.30,
min_coverage=0.0` (and identically at any coverage in [0.0, 0.5]), FINAL
56.2%.

- **min_coverage axis**: FINAL is **byte-identical** (56.2%) across
  min_coverage 0.0 through 0.5, dropping only at 0.6+. This is a wide, flat
  ridge — coverage is currently *non-binding* at `min_score=0.30` on this
  corpus (the score threshold alone already screens out what coverage
  would have caught). Kept at **0.50** anyway, as a conceptual safety net
  for corpora where it would matter, since it costs nothing here.
- **min_score axis**: the coarse grid (step 0.05) made 0.30 look like an
  isolated 1-point spike (neighbors at 0.25 and 0.35 both score lower). A
  fine 0.01-step re-scan around it showed the true shape: [0.30, 0.31] are
  both 56.2%, and — like the semantic case — the jump from 0.29 (52.9%) to
  0.30 (56.2%) is a pure abstain gain (83.3%→100%) with **zero** recall/MRR
  cost (both stay at 43.8%/47.9%). The drop after 0.31 (to 54.0% at 0.32+)
  is where it starts costing real hits. So this is narrow in absolute
  threshold units (width 0.01) but mechanistically the same "abstain
  becomes free, then stops being free" shape as semantic's wider plateau —
  not an accident of which 6 queries happen to be unanswerable, just a
  smaller numeric window because BM25's score distribution is more
  discrete than cosine similarity's. **Chosen: keep 0.30** (already the
  original default — confirmed optimal, not changed).

**Net effect of step 1**: semantic FINAL 81.3% → **83.6%** (lexical
unchanged, already optimal). Regenerated `runs/*.json` and ran
`python score.py runs/*.json --k 3` for real:

| METHOD | FINAL | RECALL | MRR | ABSTAIN |
|---|---|---|---|---|
| hybrid | 87.8% | 91.0% | 85.4% | 83.3% |
| semantic | 83.6% | 86.8% | 78.5% | 83.3% |
| lexical | 56.2% | 43.8% | 47.9% | 100.0% |

Hybrid's FINAL is unchanged (87.8% → 87.8%) but its RECALL/MRR both rose
noticeably (86.8→91.0%, 81.2→85.4%) while its ABSTAIN *fell* (100%→83.3%).
**Cause, confirmed not assumed**: hybrid's abstention rule is "abstain only
if both arms are empty" (OR logic). With the old semantic threshold
(0.36), semantic abstained on all 6 unanswerable queries, same as lexical
— so hybrid abstained on all 6 too. With the new threshold (0.30), semantic
now returns a (false-positive) result on 1 of the 6 unanswerable queries
that lexical still correctly abstains on — and because hybrid only needs
ONE arm to return something, hybrid inherits that false positive even
though lexical alone got it right. The recall/MRR gain and the abstain
loss happen to roughly cancel in FINAL here, but this exposes a real
design weakness in plain OR-based fusion that step 3's gating is meant to
address directly.

## Step 2 — stopwords x stemming ablation

Added a minimal, dependency-free `stem()` (no nltk in `rag_env`) that
strips common English suffixes (-ies→y, -es/-ed/-ing/-s) with length
guards, and — critically — leaves any token containing a digit completely
untouched (a product code or price must never be stemmed). Ran all 4
combinations of `use_stopwords x use_stemming` via `analysis/ablations.py`,
scored with `score.py`'s `evaluate()`:

| stopwords | stemming | LEX FINAL | LEX ABSTAIN | LEX hybrid-type recall | HYB FINAL | HYB ABSTAIN | HYB hybrid-type recall |
|---|---|---|---|---|---|---|---|
| True | **True** | **59.0%** | 100.0% | **50.0%** | **90.3%** | 83.3% | **100.0%** |
| True | False | 56.2% | 100.0% | 41.7% | 87.8% | 83.3% | 91.7% |
| False | True | 55.6% | 100.0% | 50.0% | 88.0% | 83.3% | 91.7% |
| False | False | 52.3% | 100.0% | 33.3% | 87.2% | 83.3% | 91.7% |

**Chosen: stopwords=True, stemming=True** (the best row on every column
except the abstain rate, which is identical across all 4 — thresholds
govern abstention, not tokenization, so stopwords/stemming don't move it
here).

**The expected trade-off, checked for and not found here**: a suffix
stripper this simple can in general merge two *unrelated* words that
happen to share an ending (e.g. "universe"/"university" both stemming
toward "univers") — that would show up as a false positive dragging recall
down somewhere even as it helps elsewhere. I checked this directly: I
computed `stem()` over every token actually appearing in the 55-review
corpus and grouped by stem. There are exactly 8 stems with more than one
surface form (`box`/`boxes`, `day`/`days`, `leak`ed/`leaks`, `order`/
`ordered`, `packag`ed/`packaging`, `refund`/`refunded`, `week`/`weeks`,
`work`ed/`working`) — every single one is a legitimate inflection of the
same word, zero accidental collisions. That's *why* the ablation shows a
clean win with no measured downside: the risk is real in general (and
would need re-checking on a larger/more varied corpus), but empirically
absent on this specific 55-review vocabulary. Per-query diff (comparing
stemming on vs. off, stopwords on) confirms it: only 3 queries changed at
all (q11, q19, q21), and all 3 moved a relevant document up in rank or
into the top-k — never down or out. This is named honestly as a
corpus-specific absence of harm, not a proof the trade-off doesn't exist
in general.

**Net effect of step 2**: hybrid FINAL 87.8% → **90.3%** (lexical
56.2% → 59.0%). Applied `use_stemming=True` as the new `lexical_search`
default and added a "Use light stemming" checkbox to `3_measure_search.py`.

## Step 3 — gated hybrid variants

Extended `hybrid_search` with `weight_lexical`/`weight_semantic`,
`gate_lexical` (+ `gate_min_coverage`/`gate_max_doc_freq`), and
`identifier_override` — all generic, corpus-wide rules, none branching on
query id or text. Measured via `analysis/hybrid_variants.py` against the
step-1/2 baseline (lexical: stopwords+stemming on, score=0.30, cov=0.50;
semantic: similarity=0.30):

| variant | FINAL | RECALL | MRR | ABSTAIN |
|---|---|---|---|---|
| (a) plain RRF | 90.3% | 95.1% | 86.8% | 83.3% |
| (b) gated RRF, best swept (min_cov=0.6, max_df=5) | 90.3% | 95.1% | 86.8% | 83.3% |
| (c) weighted RRF, best swept (weight_lexical=1.0, i.e. plain) | 90.3% | 95.1% | 86.8% | 83.3% |
| (d) identifier override | **91.5%** | 95.1% | **91.0%** | 83.3% |
| (b)+(d) combined | 91.5% | 95.1% | 91.0% | 83.3% |
| (c)+(d) combined | 91.5% | 95.1% | 91.0% | 83.3% |

**(b) gated RRF — did not help, name the reason why.** Swept
`gate_min_coverage ∈ {0.6,...,1.0}` x `gate_max_doc_freq ∈ {1,2,3,5}` (20
combos). Every setting either tied plain RRF or scored lower — never beat
it. Reason, checked directly: every lexical top candidate that survives
`lexical_search`'s OWN abstention threshold (score>=0.30, coverage>=0.50)
already happens to also be a "genuine exact hit" by the gate's stricter
definition on this corpus, so the gate never actually filters anything out
at its best setting. The naive-RRF failure mode the task described ("BM25
ranks confidently on one stray token") is a real risk in general, but
isn't present in this particular 24-answerable-query gold set once step
1's thresholds are already in place — gating had nothing left to fix.
**Not adopted** (`gate_lexical` defaults to `False`), kept fully runnable.

**(c) weighted RRF — actively worse, name the reason why.** Swept
`weight_lexical` 0.0 → 1.0 in steps of 0.1 (`weight_semantic` fixed at
1.0). FINAL is monotonically non-decreasing in `weight_lexical`, peaking
at exactly 1.0 (today's default) — every reduction strictly discarded
correct hits that the lexical arm was contributing (recall drops from
95.1% to 91.0% or 86.8% as weight decreases). There is no evidence in this
gold set that down-weighting lexical relative to semantic helps; if
anything lexical is pulling more than its numeric share of the weight
would suggest. **Not adopted**, kept fully runnable.

**(d) exact-identifier override — the winner, but only after fixing a bug
in it.** First version: "looks like an identifier" = any token containing
a digit (pure-numeric OR hyphenated). This actually **regressed** two
queries: q17 ("battery lasted 30 hours", relevant=r019) and q19
("ProBlend 900 leaking", relevant=r023) both got a WRONG document promoted
to rank 1 (via the bare numbers "30" and "900", which also happen to
appear in other, irrelevant documents), demoting the correct answer from
rank 1 to rank 2. Meanwhile it correctly fixed q11/q21 (order "#48213",
5 digits) — plain RRF ranked the right document 2nd, the override
correctly promoted it to 1st. These two regressions and two fixes
*exactly canceled* in the aggregate MRR (86.8% identical with and without
the override) — a textbook case of an aggregate metric hiding a real bug.
Caught by diffing every query's top-3 between plain and override, not by
trusting the aggregate number. **Fix**: require >=4 digits for the
pure-numeric branch of "looks like an identifier" (hyphenated tokens like
"ax-7710"/"9921-b" are unrestricted — their structure is already
distinctive). This is still a fully generic, query-independent rule (a
property of digit-count, not of which query it is). After the fix: the two
regressions disappear (q17/q19 no longer trigger the override, since "30"
and "900" have only 2-3 digits) and the two genuine fixes remain (q11/q21,
"48213" has 5 digits) — net FINAL 90.3% → **91.5%**, MRR 86.8% → 91.0%,
recall/abstain unchanged. **Adopted**: `identifier_override` now defaults
to `True` in `hybrid_search`, and a matching checkbox (default checked)
was added to `3_measure_search.py`'s sidebar.

**Chosen final hybrid configuration**: plain RRF weights
(`weight_lexical=weight_semantic=1.0`), `gate_lexical=False`,
`identifier_override=True`. Every rejected variant remains fully callable
via its parameters for comparison (`analysis/hybrid_variants.py` reproduces
this whole table on demand).

## Step 4 — final numbers

`runs/*.json` regenerated for real (`retrieval.py`'s functions called
directly, never hand-edited) at: lexical `use_stopwords=True,
use_stemming=True, min_score=0.30, min_coverage=0.50`; semantic
`min_similarity=0.30`; hybrid = plain RRF + `identifier_override=True`.
`python score.py runs/*.json --k 3` (verbatim in `results/score_output.txt`):

| METHOD | FINAL | RECALL | MRR | ABSTAIN |
|---|---|---|---|---|
| hybrid | **91.5%** | 95.1% | 91.0% | 83.3% |
| semantic | 83.6% | 86.8% | 78.5% | 83.3% |
| lexical | 59.0% | 47.9% | 50.0% | 100.0% |

By type (recall@3): hybrid scores **100.0%** on both lexical-type and
hybrid-type queries, 88.3% on semantic-type.

## Summary: FINAL score progression

| stage | hybrid FINAL | what changed |
|---|---|---|
| starting point | 87.8% | original thresholds (semantic 0.36) |
| step 1 (semantic threshold 0.36→0.30) | 87.8% (flat — recall/MRR rose, abstain fell, canceling out) | ridge-backed, mechanism confirmed |
| step 2 (+ stemming) | 90.3% | measured win, zero measured downside on this corpus |
| step 3 (+ identifier override, bug-fixed) | **91.5%** | fixed 2 order-number queries' rank |

Net improvement: **87.8% → 91.5%** (+3.7 points), achieved through three
measured, ridge-or-mechanism-justified changes and one caught-and-fixed
bug — not through any threshold chosen to fit the 6 unanswerable queries
in isolation (every change was checked for its effect on recall/MRR, not
just abstain).

## Remaining misses at the winning configuration

From `--per-query` (hybrid): every lexical-type and hybrid-type query now
hits 100% recall@3. The only misses are:

- **q01** ("my package never arrived", semantic, relevant={r001,r018,r024})
  — recall 33.3% (only 1 of 3 found). **Structurally hard at k=3**: with 3
  relevant ids and k=3, ALL three must land in the top 3 slots
  simultaneously — there is zero margin for any wrong guess. This is a
  ceiling imposed by k, not a tuning failure.
- **q03** ("I could not get hold of a human being", semantic,
  relevant={r003,r006}) — recall 50% (1 of 2 found, but ranked 1st, so
  MRR=1.0). One slot of margin at k=3; the miss here is a genuine semantic
  gap (r006 "Nobody responded to my three emails" is a real synonym match
  the embedding model doesn't rank in the top 3), not a margin issue.
- **q26** ("what is the warranty length", unanswerable) — the one
  unanswerable query semantic's threshold (0.30) deliberately still lets
  through (cosine 0.354, just above 0.30). Named explicitly in step 1: raising
  the threshold to catch this one too costs 2 genuine semantic-type hits
  elsewhere and is a worse trade on FINAL — this is a deliberate, measured
  choice, not an oversight.

**q24** (3 relevant ids, same zero-margin structure as q01) is *not*
currently missed — hybrid gets 100% recall on it — but it remains
structurally fragile: any future change that perturbs ranking even
slightly has zero room for error on this query, unlike every 1- or
2-relevant-id query.
