#!/usr/bin/env python3
"""
score.py — grades any retrieval method against data/eval_queries.json.

It is deliberately decoupled from your implementation: your app only has to
write a "run file", and this script does the rest.  That means every student
can use a different language, library or vector store and still land on the
same leaderboard.

RUN FILE FORMAT  (runs/<method-name>.json)
------------------------------------------
    {
      "q01": ["r003", "r001", "r018"],     <- ranked list of review ids, best first
      "q02": ["r002"],
      ...
      "q25": []                            <- empty list = "I found nothing relevant"
    }

Every query id in the gold set must appear as a key.

USAGE
-----
    python score.py runs/bm25.json
    python score.py runs/*.json --k 5
    python score.py runs/*.json --k 5 --per-query

METRICS
-------
  Recall@k   Of the truly relevant reviews, how many showed up in the top k.
             This is THE number to look at first.  If it is low, no amount of
             prompt engineering downstream will save you.

  MRR@k      Mean Reciprocal Rank: 1/(rank of the first correct hit).
             Rank-sensitive, unlike recall.  This is the metric that punishes
             you for putting the "3 hours" review above the "30 hours" one.

  Abstain    Only for the 6 unanswerable queries.  The share of them where the
             method correctly returned nothing.  A method that always returns
             k results scores 0 here — as it should.

  FINAL      0.50*Recall@k + 0.30*MRR@k + 0.20*Abstain
"""
import argparse, glob, json, os, sys

W_RECALL, W_MRR, W_ABSTAIN = 0.50, 0.30, 0.20
TYPES = ["semantic", "lexical", "hybrid", "unanswerable"]


def load_gold(path="data/eval_queries.json"):
    with open(path, encoding="utf-8") as f:
        gold = json.load(f)
    return {g["id"]: g for g in gold}


def load_run(path, gold):
    with open(path, encoding="utf-8") as f:
        run = json.load(f)
    missing = [q for q in gold if q not in run]
    if missing:
        sys.exit(f"[{os.path.basename(path)}] missing {len(missing)} queries, "
                 f"first few: {missing[:5]}")
    return run


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return None
    hits = len(set(retrieved[:k]) & set(relevant))
    return hits / len(relevant)


def rr_at_k(retrieved, relevant, k):
    if not relevant:
        return None
    for i, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def evaluate(run, gold, k):
    per_query, by_type = {}, {t: {"recall": [], "rr": []} for t in TYPES}
    abstain_hits, abstain_total = 0, 0

    for qid, g in gold.items():
        retrieved = run.get(qid, [])
        relevant = g["relevant_ids"]
        qtype = g["type"]

        if qtype == "unanswerable":
            abstain_total += 1
            ok = len(retrieved) == 0
            abstain_hits += int(ok)
            per_query[qid] = {"type": qtype, "recall": None, "rr": None,
                              "abstained": ok, "returned": len(retrieved)}
            continue

        r = recall_at_k(retrieved, relevant, k)
        rr = rr_at_k(retrieved, relevant, k)
        by_type[qtype]["recall"].append(r)
        by_type[qtype]["rr"].append(rr)
        per_query[qid] = {"type": qtype, "recall": r, "rr": rr,
                          "abstained": None, "returned": len(retrieved)}

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    all_r = [v for t in TYPES for v in by_type[t]["recall"]]
    all_rr = [v for t in TYPES for v in by_type[t]["rr"]]

    abstain = abstain_hits / abstain_total if abstain_total else 0.0
    overall_recall, overall_mrr = mean(all_r), mean(all_rr)
    final = W_RECALL * overall_recall + W_MRR * overall_mrr + W_ABSTAIN * abstain

    return {
        "recall": overall_recall,
        "mrr": overall_mrr,
        "abstain": abstain,
        "final": final,
        "by_type": {t: {"recall": mean(by_type[t]["recall"]),
                        "mrr": mean(by_type[t]["rr"]),
                        "n": len(by_type[t]["recall"])}
                    for t in TYPES if t != "unanswerable"},
        "per_query": per_query,
    }


def pct(x):
    return f"{100 * x:5.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run file(s), e.g. runs/bm25.json")
    ap.add_argument("--k", type=int, default=3,
                    help="default 3 — the corpus is small, k=3 discriminates "
                         "between methods far better than k=5")
    ap.add_argument("--gold", default="data/eval_queries.json")
    ap.add_argument("--per-query", action="store_true",
                    help="print a per-query breakdown for the first run file")
    args = ap.parse_args()

    paths = []
    for p in args.runs:
        paths.extend(sorted(glob.glob(p)) or [p])

    gold = load_gold(args.gold)
    results = []
    for p in paths:
        res = evaluate(load_run(p, gold), gold, args.k)
        results.append((os.path.splitext(os.path.basename(p))[0], res))

    results.sort(key=lambda x: -x[1]["final"])
    name_w = max(12, max(len(n) for n, _ in results) + 2)

    print(f"\n  GOLD SET: {len(gold)} queries   k = {args.k}\n")
    print("  " + "METHOD".ljust(name_w)
          + "FINAL".rjust(8) + "RECALL".rjust(9) + "MRR".rjust(9) + "ABSTAIN".rjust(10))
    print("  " + "-" * (name_w + 36))
    for name, r in results:
        print("  " + name.ljust(name_w)
              + pct(r["final"]).rjust(8) + pct(r["recall"]).rjust(9)
              + pct(r["mrr"]).rjust(9) + pct(r["abstain"]).rjust(10))

    print(f"\n  BREAKDOWN BY QUERY TYPE  (recall@{args.k})\n")
    header = "  " + "METHOD".ljust(name_w)
    for t in ["semantic", "lexical", "hybrid"]:
        header += t.upper().rjust(12)
    print(header)
    print("  " + "-" * (name_w + 36))
    for name, r in results:
        line = "  " + name.ljust(name_w)
        for t in ["semantic", "lexical", "hybrid"]:
            line += pct(r["by_type"][t]["recall"]).rjust(12)
        print(line)

    if args.per_query and results:
        name, r = results[0]
        print(f"\n  PER-QUERY  ({name})\n")
        print("  " + "ID".ljust(6) + "TYPE".ljust(15)
              + "RECALL".rjust(8) + "RR".rjust(8) + "  QUERY")
        print("  " + "-" * 78)
        for qid in sorted(gold):
            pq, g = r["per_query"][qid], gold[qid]
            if pq["type"] == "unanswerable":
                mark = "  ok  " if pq["abstained"] else " MISS "
                print("  " + qid.ljust(6) + pq["type"].ljust(15)
                      + mark.rjust(8) + f'{pq["returned"]}'.rjust(8)
                      + "  " + g["query"])
            else:
                print("  " + qid.ljust(6) + pq["type"].ljust(15)
                      + pct(pq["recall"]).rjust(8) + f'{pq["rr"]:.2f}'.rjust(8)
                      + "  " + g["query"])
    print()


if __name__ == "__main__":
    main()
