#!/usr/bin/env python
"""Run exact synthesis baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagonal_synth.exact import synthesize_exact
from experiments.gen_random import random_lambda


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--qubits", type=str, default="5,8")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    qubits = [int(x) for x in args.qubits.split(",") if x.strip()]
    for n in qubits:
        lam = random_lambda(n, seed=args.seed + n)
        r = synthesize_exact(lam, n)
        print(
            f"n={n} method={r.method} cnot={r.cnot_used}/{r.cnot_baseline} "
            f"error={r.error:.6e} time={r.time_s:.4f}s"
        )


if __name__ == "__main__":
    main()
