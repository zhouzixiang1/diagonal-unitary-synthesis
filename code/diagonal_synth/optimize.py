"""Post-search gradient refinement of α on the selected support."""

from __future__ import annotations

import numpy as np

from .metrics import error_unitary
from .walsh import alpha_to_lambda, lambda_to_alpha


def refine_alpha(
    alpha_init: np.ndarray,
    lambda_t: np.ndarray,
    support: np.ndarray | list[int] | None = None,
    *,
    lr: float = 0.05,
    max_iters: int = 200,
    tol: float = 1e-10,
) -> np.ndarray:
    """Gradient descent on D(U_C, U_T) w.r.t. α on a fixed support.

    Uses a finite-difference / analytic style update matching the thesis idea:
    optimize α' with zeros outside the selected phase gadgets.
    """
    alpha = np.asarray(alpha_init, dtype=np.float64).copy()
    n = int(np.log2(alpha.size))
    if support is None:
        mask = np.ones_like(alpha, dtype=bool)
    else:
        mask = np.zeros_like(alpha, dtype=bool)
        for idx in support:
            if 0 <= int(idx) < mask.size:
                mask[int(idx)] = True
    alpha[~mask] = 0.0

    # Precompute Walsh matrix columns via FWHT Jacobian: ∂λ/∂α = H
    # For efficiency use coordinate-wise FD on small supports when n is large,
    # and exact FD-free update using H for moderate n.
    prev = error_unitary(alpha_to_lambda(alpha, n), lambda_t)
    from .walsh import _fwht

    for _ in range(max_iters):
        lam_c = alpha_to_lambda(alpha, n)
        diff = (np.exp(1j * lam_c) - np.exp(1j * lambda_t)) / 2.0
        denom = max(np.linalg.norm(diff), 1e-16)
        dD_dlam = np.real(np.conj(diff) * (1j * np.exp(1j * lam_c) / 2.0)) / denom
        # λ = H α / 2^n  =>  ∇_α D = (H / 2^n) ∇_λ D
        grad = _fwht(dD_dlam) / float(1 << n)
        grad[~mask] = 0.0
        # Adaptive step: avoid overshoot on large n
        step = lr / max(1.0, float(np.linalg.norm(grad)))
        alpha = alpha - step * grad
        alpha[~mask] = 0.0
        cur = error_unitary(alpha_to_lambda(alpha, n), lambda_t)
        if cur > prev:
            alpha = alpha + step * grad
            alpha[~mask] = 0.0
            lr *= 0.5
            if lr < 1e-8:
                break
            continue
        if abs(prev - cur) < tol:
            break
        prev = cur
    return alpha


def sparse_alpha_from_path(alpha_t: np.ndarray, path: list[int]) -> tuple[np.ndarray, list[int]]:
    support = sorted({int(p) for p in path})
    alpha = np.zeros_like(alpha_t)
    for i in support:
        alpha[i] = alpha_t[i]
    return alpha, support
