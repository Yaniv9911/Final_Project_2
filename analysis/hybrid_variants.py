"""Measure each hybrid fusion variant (plain, gated, weighted, identifier
override, and promising combinations) against the gold set, scored with
score.py's own evaluate().

Run with: python analysis/hybrid_variants.py
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

LEX_KWARGS = {"use_stopwords": True, "use_stemming": True, "min_score": 0.30, "min_coverage": 0.50}
SEM_KWARGS = {"min_similarity": 0.30}


def build_run(gold_list, corpus, **hybrid_kwargs):
    return {
        q["id"]: [
            doc_id for doc_id, _ in rt.hybrid_search(
                q["query"], k=K, corpus=corpus,
                lexical_kwargs=LEX_KWARGS, semantic_kwargs=SEM_KWARGS,
                **hybrid_kwargs,
            )
        ] for q in gold_list
    }


def score_run(run, gold_dict, tmp_dir, name):
    path = Path(tmp_dir) / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run, f)
    loaded = sc.load_run(path, gold_dict)
    return sc.evaluate(loaded, gold_dict, K)


def report(label, run, gold_dict, tmp_dir):
    res = score_run(run, gold_dict, tmp_dir, "variant")
    print(f"{label:45} FINAL={res['final']*100:5.1f}%  RECALL={res['recall']*100:5.1f}%  "
          f"MRR={res['mrr']*100:5.1f}%  ABSTAIN={res['abstain']*100:5.1f}%")
    return res


def main():
    gold_dict = sc.load_gold(GOLD_PATH)
    gold_list = list(gold_dict.values())
    corpus = rt.load_reviews()

    with tempfile.TemporaryDirectory() as tmp_dir:
        print("=== (a) plain RRF (baseline) ===")
        report("plain", build_run(gold_list, corpus), gold_dict, tmp_dir)

        print("\n=== (b) gated RRF — sweep gate_min_coverage x gate_max_doc_freq ===")
        best_gate, best_gate_final = None, -1
        for min_cov in (0.6, 0.7, 0.8, 0.9, 1.0):
            for max_df in (1, 2, 3, 5):
                run = build_run(gold_list, corpus, gate_lexical=True,
                                 gate_min_coverage=min_cov, gate_max_doc_freq=max_df)
                res = score_run(run, gold_dict, tmp_dir, "gate")
                if res["final"] > best_gate_final:
                    best_gate_final = res["final"]
                    best_gate = (min_cov, max_df)
        print(f"best gate: min_coverage={best_gate[0]} max_doc_freq={best_gate[1]} FINAL={best_gate_final*100:.1f}%")
        report(f"gated (min_cov={best_gate[0]}, max_df={best_gate[1]})",
               build_run(gold_list, corpus, gate_lexical=True,
                         gate_min_coverage=best_gate[0], gate_max_doc_freq=best_gate[1]),
               gold_dict, tmp_dir)

        print("\n=== (c) weighted RRF — sweep weight_lexical (weight_semantic=1.0) ===")
        best_w, best_w_final = None, -1
        for w in [i / 10 for i in range(0, 11)]:
            run = build_run(gold_list, corpus, weight_lexical=w, weight_semantic=1.0)
            res = score_run(run, gold_dict, tmp_dir, "weighted")
            print(f"  weight_lexical={w:.1f} FINAL={res['final']*100:5.1f}% "
                  f"RECALL={res['recall']*100:5.1f}% MRR={res['mrr']*100:5.1f}% ABSTAIN={res['abstain']*100:5.1f}%")
            if res["final"] > best_w_final:
                best_w_final, best_w = res["final"], w
        print(f"best weight_lexical={best_w} FINAL={best_w_final*100:.1f}%")

        print("\n=== (d) exact-identifier override ===")
        report("identifier override (plain RRF base)",
               build_run(gold_list, corpus, identifier_override=True), gold_dict, tmp_dir)

        print("\n=== combinations ===")
        report("gated + identifier override",
               build_run(gold_list, corpus, gate_lexical=True,
                         gate_min_coverage=best_gate[0], gate_max_doc_freq=best_gate[1],
                         identifier_override=True),
               gold_dict, tmp_dir)
        report(f"weighted(w_lex={best_w}) + identifier override",
               build_run(gold_list, corpus, weight_lexical=best_w, weight_semantic=1.0,
                         identifier_override=True),
               gold_dict, tmp_dir)
        report(f"weighted(w_lex={best_w}) + gated",
               build_run(gold_list, corpus, weight_lexical=best_w, weight_semantic=1.0,
                         gate_lexical=True, gate_min_coverage=best_gate[0], gate_max_doc_freq=best_gate[1]),
               gold_dict, tmp_dir)
        report(f"weighted(w_lex={best_w}) + gated + identifier override",
               build_run(gold_list, corpus, weight_lexical=best_w, weight_semantic=1.0,
                         gate_lexical=True, gate_min_coverage=best_gate[0], gate_max_doc_freq=best_gate[1],
                         identifier_override=True),
               gold_dict, tmp_dir)


if __name__ == "__main__":
    main()
