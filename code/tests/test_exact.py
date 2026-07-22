"""Exact synthesis should nearly recover the target (up to numerical error)."""

import numpy as np

from diagonal_synth.exact import synthesize_exact
from experiments.gen_random import random_lambda


def test_exact_near_zero_error():
    for n in (3, 5):
        lam = random_lambda(n, seed=42 + n)
        r = synthesize_exact(lam, n)
        assert r.error < 1e-8
        assert r.cnot_used == (1 << n)
