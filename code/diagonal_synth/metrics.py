"""Error metrics, utility ratio, and CNOT accounting helpers."""

from __future__ import annotations

import numpy as np


def error_unitary(lambda_c: np.ndarray, lambda_t: np.ndarray) -> float:
    """D(U_C, U_T) = ||(e^{iλc}-e^{iλt})/2||_2 (paper formula)."""
    lc = np.asarray(lambda_c, dtype=np.float64).reshape(-1)
    lt = np.asarray(lambda_t, dtype=np.float64).reshape(-1)
    if lc.shape != lt.shape:
        raise ValueError("lambda vectors must have the same shape")
    diff = (np.exp(1j * lc) - np.exp(1j * lt)) / 2.0
    return float(np.linalg.norm(diff))


def error_lambda(lambda_c: np.ndarray, lambda_t: np.ndarray) -> float:
    """Proxy error D_λ during search: Euclidean distance on λ."""
    lc = np.asarray(lambda_c, dtype=np.float64).reshape(-1)
    lt = np.asarray(lambda_t, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(lc - lt))


def utility_ratio(cnot_saved_ratio: float, error: float) -> float:
    """Algorithm utility ratio = CNOT saved ratio / error."""
    if error <= 0:
        return float("inf") if cnot_saved_ratio > 0 else 0.0
    return float(cnot_saved_ratio) / float(error)


def baseline_cnot_count(n: int) -> int:
    """Exact-synthesis CNOT baseline used in the papers: 2^n."""
    return 1 << n


def cnot_budget_from_reduction(n: int, reduction_ratio: float) -> int:
    """C = floor(2^n * (1 - ReCNOT))."""
    if not 0.0 <= reduction_ratio < 1.0:
        raise ValueError("reduction_ratio must be in [0, 1)")
    full = baseline_cnot_count(n)
    return max(0, int(np.floor(full * (1.0 - reduction_ratio))))


def cnot_saved_ratio(n: int, cnot_used: int) -> float:
    full = baseline_cnot_count(n)
    if full == 0:
        return 0.0
    return max(0.0, (full - cnot_used) / float(full))
