"""
scripts/summarize_seqlen_eval.py — Tabulate the seq_len reload-eval results.

Reads the per-run JSONs written by reload_eval.py (seq_len 1 & 4) and
export_tier2_final_eval.py (seq_len 128) and prints mean +/- std over seeds of
the in-distribution mean focal return (the apples-to-apples Setting-2 metric,
same 9 in-dist scenarios for every seq_len) and the all-18-scenario mean
(available for seq_len 1 & 4 only).

Usage: python scripts/summarize_seqlen_eval.py [--results-dir results/seqlen_eval]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict


def _fmt(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "        —"
    m = sum(xs) / len(xs)
    s = st.pstdev(xs) if len(xs) > 1 else 0.0
    return f"{m:7.1f} ± {s:5.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/seqlen_eval")
    args = ap.parse_args()

    cells = defaultdict(lambda: {"in": [], "all": [], "seeds": 0})
    for f in glob.glob(os.path.join(args.results_dir, "*.json")):
        if os.path.basename(f) == "all_runs.json":
            continue
        d = json.load(open(f))
        key = (d["algorithm"], d["seq_len"])
        agg = d.get("aggregates", {})
        cells[key]["in"].append(agg.get("in_dist_mean"))
        cells[key]["all"].append(agg.get("all_mean"))
        cells[key]["seeds"] += 1

    seq_lens = sorted({sl for _, sl in cells})
    print("\nmean focal-agent episode return on clean_up  "
          "(mean +/- std over seeds)\n")
    print(f"{'algo':4} {'seq_len':>7} {'seeds':>5}   {'in-dist mean (9 scen)':>21}   {'all-scenario mean (18)':>22}")
    print("-" * 70)
    for algo in ["bc", "bcq", "iql", "cql"]:
        for sl in seq_lens:
            c = cells.get((algo, sl))
            if not c:
                continue
            print(f"{algo:4} {sl:>7} {c['seeds']:>5}   {_fmt(c['in']):>21}   {_fmt(c['all']):>22}")
        print()
    print("in-dist = same 9 in-distribution scenarios for every seq_len (the fair "
          "Setting-2 comparison).\nall-scenario (18) only computed for seq_len 1 & 4 "
          "(the baseline final-eval scored in-dist only).\nseq_len 1 & 4: 64 episodes/scenario; "
          "seq_len 128 baseline: 32 episodes/scenario.")


if __name__ == "__main__":
    main()
