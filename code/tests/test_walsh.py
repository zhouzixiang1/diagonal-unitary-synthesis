"""Unit tests for Walsh–Hadamard transforms."""

import numpy as np

from diagonal_synth.walsh import alpha_to_lambda, hadamard_matrix, lambda_to_alpha


def test_roundtrip_random():
    rng = np.random.default_rng(0)
    for n in range(1, 7):
        lam = rng.normal(size=1 << n)
        alpha = lambda_to_alpha(lam, n)
        lam2 = alpha_to_lambda(alpha, n)
        assert np.allclose(lam, lam2, atol=1e-10)


def test_matches_dense_matrix():
    rng = np.random.default_rng(1)
    for n in range(1, 6):
        alpha = rng.normal(size=1 << n)
        h = hadamard_matrix(n)
        lam_dense = h @ alpha / (1 << n)
        lam_fast = alpha_to_lambda(alpha, n)
        assert np.allclose(lam_dense, lam_fast, atol=1e-10)
        alpha_dense = h @ lam_fast
        assert np.allclose(alpha_dense, lambda_to_alpha(lam_fast, n), atol=1e-10)
