"""Random diagonal unitary instance generation."""

from __future__ import annotations

import numpy as np

from diagonal_synth.walsh import alpha_to_lambda


def random_alpha(n: int, seed: int | None = None, scale: float = np.pi) -> np.ndarray:
    """Sample Rz angles α ~ Uniform(-scale, scale)^{2^n}."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=(1 << n,))


def random_lambda(n: int, seed: int | None = None, scale: float = np.pi) -> np.ndarray:
    """Sample a random diagonal unitary in the paper-aligned protocol.

    Generate α first, then λ = H α. This matches Table 3 error magnitudes;
    sampling λ directly makes |α| too small after Walsh normalization.
    """
    alpha = random_alpha(n, seed=seed, scale=scale)
    return alpha_to_lambda(alpha, n)


def seed_list(base: int, count: int) -> list[int]:
    return [base + i for i in range(count)]
