"""Walsh–Hadamard transforms linking phase vector λ and Rz angles α.

Empirical convention aligned with arXiv:2412.01869 Table 1/3 scales
(α = O(1) Rz angles; random instances sample α then map to λ):

    λ = (1 / 2^n) * H^{⊗n} α
    α = H^{⊗n} λ

where H is the unnormalized Hadamard with entries (±1).
(The paper text writes the inverse pair in places; we lock the pair that
reproduces reported error magnitudes under α ~ Uniform(-π, π).)
"""

from __future__ import annotations

import numpy as np


def hadamard_matrix(n: int) -> np.ndarray:
    """Return unnormalized H^{⊗n} of shape (2^n, 2^n) with entries ±1."""
    if n < 0:
        raise ValueError("n must be non-negative")
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64)
    m = np.array([[1.0]], dtype=np.float64)
    for _ in range(n):
        m = np.kron(m, h)
    return m


def lambda_to_alpha(lambda_vec: np.ndarray, n: int | None = None) -> np.ndarray:
    """α = H λ."""
    lam = np.asarray(lambda_vec, dtype=np.float64).reshape(-1)
    if n is None:
        size = lam.size
        if size & (size - 1):
            raise ValueError("lambda length must be a power of 2")
        n = int(np.log2(size))
    expect = 1 << n
    if lam.size != expect:
        raise ValueError(f"expected length {expect}, got {lam.size}")
    return _fwht(lam)


def alpha_to_lambda(alpha_vec: np.ndarray, n: int | None = None) -> np.ndarray:
    """λ = H α / 2^n."""
    alpha = np.asarray(alpha_vec, dtype=np.float64).reshape(-1)
    if n is None:
        size = alpha.size
        if size & (size - 1):
            raise ValueError("alpha length must be a power of 2")
        n = int(np.log2(size))
    expect = 1 << n
    if alpha.size != expect:
        raise ValueError(f"expected length {expect}, got {alpha.size}")
    return _fwht(alpha) / float(expect)


def _fwht(a: np.ndarray) -> np.ndarray:
    """Fast Walsh–Hadamard transform (natural binary order)."""
    x = a.copy()
    h = 1
    n = x.size
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                u = x[j]
                v = x[j + h]
                x[j] = u + v
                x[j + h] = u - v
        h *= 2
    return x
