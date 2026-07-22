"""Exact diagonal unitary synthesis via Gray-code phase-gadget ordering."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .gadgets import gray_code_path, path_cnot_cost, path_to_alpha
from .metrics import baseline_cnot_count, error_unitary
from .walsh import alpha_to_lambda, lambda_to_alpha


@dataclass
class SynthesisResult:
    method: str
    n: int
    path: list[int]
    alpha_c: np.ndarray
    lambda_c: np.ndarray
    cnot_used: int
    cnot_baseline: int
    error: float
    time_s: float


def synthesize_exact(lambda_t: np.ndarray, n: int | None = None) -> SynthesisResult:
    """Synthesize with all phase gadgets ordered by Gray code."""
    t0 = time.perf_counter()
    lam = np.asarray(lambda_t, dtype=np.float64).reshape(-1)
    if n is None:
        n = int(np.log2(lam.size))
    alpha_t = lambda_to_alpha(lam, n)
    path = gray_code_path(n)
    alpha_c = path_to_alpha(path, alpha_t, include_zero=True)
    lambda_c = alpha_to_lambda(alpha_c, n)
    # Path edge cost is 2^n - 1 for Gray code; papers report baseline 2^n.
    edge_cost = path_cnot_cost(path, n)
    cnot_baseline = baseline_cnot_count(n)
    # Use path edge cost as "used" for exact; baseline for ReCNOT is 2^n.
    # Exact method uses full gadget set; report used = baseline for protocol match.
    cnot_used = cnot_baseline
    err = error_unitary(lambda_c, lam)
    return SynthesisResult(
        method="exact",
        n=n,
        path=path,
        alpha_c=alpha_c,
        lambda_c=lambda_c,
        cnot_used=cnot_used,
        cnot_baseline=cnot_baseline,
        error=err,
        time_s=time.perf_counter() - t0,
    )


def synthesize_from_path(
    lambda_t: np.ndarray,
    path: list[int],
    *,
    n: int | None = None,
    cnot_used: int | None = None,
    method: str = "path",
    refine: bool = False,
) -> SynthesisResult:
    """Build approximate synthesis from a selected gadget path."""
    t0 = time.perf_counter()
    lam = np.asarray(lambda_t, dtype=np.float64).reshape(-1)
    if n is None:
        n = int(np.log2(lam.size))
    alpha_t = lambda_to_alpha(lam, n)
    alpha_c = path_to_alpha(path, alpha_t, include_zero=True)
    support = sorted(set(path))
    if refine:
        from .optimize import refine_alpha

        alpha_c = refine_alpha(alpha_c, lam, support)
    lambda_c = alpha_to_lambda(alpha_c, n)
    baseline = baseline_cnot_count(n)
    if cnot_used is None:
        # Approximate: number of CNOT merges ≈ len(path) - 1 for Hamming-1 path
        cnot_used = max(0, len(path) - 1)
    err = error_unitary(lambda_c, lam)
    return SynthesisResult(
        method=method,
        n=n,
        path=list(path),
        alpha_c=alpha_c,
        lambda_c=lambda_c,
        cnot_used=int(cnot_used),
        cnot_baseline=baseline,
        error=err,
        time_s=time.perf_counter() - t0,
    )
