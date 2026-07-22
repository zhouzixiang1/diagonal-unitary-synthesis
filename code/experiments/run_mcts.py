#!/usr/bin/env python
"""Run MCTS approximate synthesis baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagonal_synth.mcts import MCTSConfig, synthesize_mcts
from diagonal_synth.metrics import cnot_budget_from_reduction, cnot_saved_ratio, utility_ratio
from experiments.gen_random import random_lambda


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--qubits", type=int, default=5)
    p.add_argument("--reduction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-refine", action="store_true")
    args = p.parse_args()
    n = args.qubits
    lam = random_lambda(n, seed=args.seed)
    c = cnot_budget_from_reduction(n, args.reduction)
    r = synthesize_mcts(lam, c, n=n, config=MCTSConfig(refine=not args.no_refine))
    saved = cnot_saved_ratio(n, r.cnot_used)
    util = utility_ratio(saved, r.error)
    print(
        f"n={n} method={r.method} C={c} cnot_used={r.cnot_used} "
        f"saved={saved:.4f} error={r.error:.6f} utility={util:.3f} time={r.time_s:.4f}s"
    )


if __name__ == "__main__":
    main()
