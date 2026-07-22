"""Phase-gadget graph utilities (Hamming-distance neighborhood)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


def hamming_distance(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def neighbors(node: int, n: int) -> list[int]:
    """Nodes at Hamming distance 1 (single CNOT merge between gadgets)."""
    return [node ^ (1 << i) for i in range(n)]


@lru_cache(maxsize=16)
def adjacency_lists(n: int) -> tuple[tuple[int, ...], ...]:
    size = 1 << n
    return tuple(tuple(neighbors(i, n)) for i in range(size))


def gray_code_path(n: int) -> list[int]:
    """Hamilton path on the n-cube via binary-reflected Gray code (0..2^n-1)."""
    return [i ^ (i >> 1) for i in range(1 << n)]


def path_cnot_cost(path: list[int], n: int) -> int:
    """Sum of Hamming distances between consecutive nodes.

    For a Gray-code full path this equals 2^n - 1 edges of cost 1, but papers
    use 2^n as the exact baseline; callers should prefer metrics.baseline_cnot_count
    for reporting ReCNOT against the published protocol.
    """
    if len(path) <= 1:
        return 0
    return int(sum(hamming_distance(path[i], path[i + 1]) for i in range(len(path) - 1)))


def path_to_alpha(path: list[int], alpha_t: np.ndarray, include_zero: bool = True) -> np.ndarray:
    """Build sparse α_C from selected phase indices along a path."""
    alpha_c = np.zeros_like(alpha_t, dtype=np.float64)
    start = 0 if include_zero else 1
    for node in path[start:]:
        if 0 <= node < alpha_c.size:
            alpha_c[node] = alpha_t[node]
    # Keep α[0] if present in target (global phase / all-zero monomial)
    if include_zero and path and path[0] == 0:
        alpha_c[0] = alpha_t[0]
    return alpha_c
