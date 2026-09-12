"""Stopwords x stemming ablation — measured, not assumed.

Runs all 4 combinations of use_stopwords x use_stemming through
lexical_search and through hybrid_search (plain RRF), scored with
score.py's own evaluate(). Reports FINAL, abstain rate, and specifically
recall@k on the 6 "hybrid"-type gold queries, since that's where a
stopword/stemming trade-off is expected to show up most clearly.

Run with: python analysis/ablations.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import retrieval as rt
import score as sc

K = 3
GOLD_PATH = "data/eval_queries.json"


def build_run(fn, gold_list, corpus, **kwargs):
    return {q["id"]: [doc_id for doc_id, _ in fn(q["query"], k=K, corpus=corpus, **kwargs)] for q in gold_list}


def score_run(run, gold_dict, tmp_dir, name):
    path = Path(tmp_dir) / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run, f)
    loaded = sc.load_run(path, gold_dict)
    return sc.evaluate(loaded, gold_dict, K)


def hybrid_type_recall(run, gold_list):
    values = [
        sc.recall_at_k(run[q["id"]], q["relevant_ids"], K)
        for q in gold_list if q["type"] == "hybrid"
    ]
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def main():
    gold_dict = sc.load_gold(GOLD_PATH)
    gold_list = list(gold_dict.values())
    corpus = rt.load_reviews()

    print(f"{'stopwords':>10} {'stemming':>9} | {'LEX FINAL':>10} {'LEX ABST':>9} {'LEX HYB-TYPE':>13} "
          f"| {'HYB FINAL':>10} {'HYB ABST':>9} {'HYB HYB-TYPE':>13}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        for use_stopwords in (True, False):
            for use_stemming in (True, False):
                lex_kwargs = {
                    "use_stopwords": use_stopwords, "use_stemming": use_stemming,
                    "min_score": 0.30, "min_coverage": 0.50,
                }
                sem_kwargs = {"min_similarity": 0.30}

                lex_run = build_run(rt.lexical_search, gold_list, corpus, **lex_kwargs)
                lex_res = score_run(lex_run, gold_dict, tmp_dir, "lex")
                lex_hyb_recall = hybrid_type_recall(lex_run, gold_list)

                hyb_run = {
                    q["id"]: [
                        doc_id for doc_id, _ in rt.hybrid_search(
                            q["query"], k=K, corpus=corpus,
                            lexical_kwargs=lex_kwargs, semantic_kwargs=sem_kwargs,
                        )
                    ] for q in gold_list
                }
                hyb_res = score_run(hyb_run, gold_dict, tmp_dir, "hyb")
                hyb_hyb_recall = hybrid_type_recall(hyb_run, gold_list)

                print(f"{str(use_stopwords):>10} {str(use_stemming):>9} | "
                      f"{lex_res['final']*100:>9.1f}% {lex_res['abstain']*100:>8.1f}% {lex_hyb_recall*100:>12.1f}% "
                      f"| {hyb_res['final']*100:>9.1f}% {hyb_res['abstain']*100:>8.1f}% {hyb_hyb_recall*100:>12.1f}%")


if __name__ == "__main__":
    main()
