#!/usr/bin/env python
"""Compare exact / MCTS / path-search baselines and write CSV."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagonal_synth.exact import synthesize_exact
from diagonal_synth.mcts import MCTSConfig, synthesize_mcts
from diagonal_synth.metrics import (
    cnot_budget_from_reduction,
    cnot_saved_ratio,
    utility_ratio,
)
from diagonal_synth.path_search import synthesize_path_search
from experiments.gen_random import random_lambda, seed_list


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--qubits", type=str, default="5,8")
    p.add_argument("--reductions", type=str, default="0.1,0.2")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--out", type=str, default="results/baselines.csv")
    p.add_argument("--skip-mcts-above", type=int, default=10)
    p.add_argument("--methods", type=str, default="exact,mcts,path_search")
    args = p.parse_args()

    qubits = [int(x) for x in args.qubits.split(",") if x.strip()]
    reductions = [float(x) for x in args.reductions.split(",") if x.strip()]
    methods = {x.strip() for x in args.methods.split(",") if x.strip()}
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "method",
        "n",
        "reduction",
        "trial",
        "seed",
        "cnot_budget",
        "cnot_used",
        "saved_ratio",
        "error",
        "utility",
        "time_s",
    ]
    rows: list[dict] = []

    for n in qubits:
        for red in reductions:
            c = cnot_budget_from_reduction(n, red)
            for t_idx, seed in enumerate(seed_list(args.seed_base + 1000 * n, args.trials)):
                lam = random_lambda(n, seed=seed)
                if "exact" in methods and red == reductions[0]:
                    # exact once per (n, trial); independent of reduction
                    r = synthesize_exact(lam, n)
                    rows.append(
                        {
                            "method": "exact",
                            "n": n,
                            "reduction": 0.0,
                            "trial": t_idx,
                            "seed": seed,
                            "cnot_budget": r.cnot_baseline,
                            "cnot_used": r.cnot_used,
                            "saved_ratio": 0.0,
                            "error": r.error,
                            "utility": 0.0,
                            "time_s": r.time_s,
                        }
                    )
                if "mcts" in methods and n <= args.skip_mcts_above:
                    r = synthesize_mcts(
                        lam, c, n=n, config=MCTSConfig(refine=True, max_expand=min(5000, c * 20))
                    )
                    saved = cnot_saved_ratio(n, r.cnot_used)
                    rows.append(
                        {
                            "method": "mcts",
                            "n": n,
                            "reduction": red,
                            "trial": t_idx,
                            "seed": seed,
                            "cnot_budget": c,
                            "cnot_used": r.cnot_used,
                            "saved_ratio": saved,
                            "error": r.error,
                            "utility": utility_ratio(saved, r.error),
                            "time_s": r.time_s,
                        }
                    )
                if "path_search" in methods:
                    r = synthesize_path_search(lam, c, n=n, refine=True)
                    saved = cnot_saved_ratio(n, r.cnot_used)
                    rows.append(
                        {
                            "method": "path_search",
                            "n": n,
                            "reduction": red,
                            "trial": t_idx,
                            "seed": seed,
                            "cnot_budget": c,
                            "cnot_used": r.cnot_used,
                            "saved_ratio": saved,
                            "error": r.error,
                            "utility": utility_ratio(saved, r.error),
                            "time_s": r.time_s,
                        }
                    )
                print(f"done n={n} red={red} trial={t_idx}", flush=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
