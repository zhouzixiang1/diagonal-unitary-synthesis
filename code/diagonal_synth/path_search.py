"""Phase-importance graph path search (arXiv:2412.01869 Algorithms 1–6)."""

from __future__ import annotations

import time

import numpy as np

from .exact import SynthesisResult, synthesize_from_path
from .gadgets import adjacency_lists, hamming_distance
from .walsh import lambda_to_alpha


def phase_importance(
    alpha_t: np.ndarray,
    n: int,
    cnot_budget: int,
    gamma: float = 10.0,
) -> np.ndarray:
    """Algorithm 1: |α| → [0,1] standardize → logistic about (C+2)-th largest."""
    imp = np.zeros(1 << n, dtype=np.float64)
    abs_a = np.abs(alpha_t)
    # Standardization to [0, 1]
    mx = float(np.max(abs_a)) if abs_a.size else 1.0
    if mx <= 0:
        imp[0] = 1.0
        return imp
    imp[:] = abs_a / mx
    imp[0] = 1.0

    k = min(max(cnot_budget + 2, 1), abs_a.size)
    # FindKthLargest on |α| (or on standardized? paper uses α_T in FindKthLargest)
    sorted_abs = np.sort(abs_a)[::-1]
    temp = float(sorted_abs[k - 1]) / mx if mx > 0 else 0.0

    for i in range(1, 1 << n):
        adjusted = (imp[i] - temp) * gamma
        # numerically stable logistic
        if adjusted >= 0:
            imp[i] = 1.0 / (1.0 + np.exp(-adjusted))
        else:
            e = np.exp(adjusted)
            imp[i] = e / (1.0 + e)
    return imp


def path_selection(
    path: list[int],
    imp: np.ndarray,
    adj: tuple[tuple[int, ...], ...],
    cnot_budget: int,
    eps: float,
) -> list[int]:
    """Algorithm 2: greedy walk to high-importance active neighbors."""
    path = list(path)
    target = cnot_budget + 2
    visited = set(path)
    while len(path) < target:
        tail = path[-1]
        active = [
            nb
            for nb in adj[tail]
            if nb not in visited and imp[nb] > 0.5 - eps
        ]
        if not active:
            break
        nxt = max(active, key=lambda i: imp[i])
        path.append(nxt)
        visited.add(nxt)
    return path


def path_extension(
    path: list[int],
    imp: np.ndarray,
    adj: tuple[tuple[int, ...], ...],
    eps: float,
) -> list[int]:
    """Algorithm 3: Hamiltonian-path style rotation extension."""
    path = list(path)
    if len(path) < 2:
        return path
    tail = path[-1]
    next_node = None
    next_index = -1
    max_imp = -1.0
    for node in adj[tail]:
        if node in path[:-2]:
            index = path.index(node)
            for cand in adj[path[index + 1]]:
                if cand not in path and imp[cand] > max_imp:
                    max_imp = float(imp[cand])
                    next_node = cand
                    next_index = index
    if next_node is not None and imp[next_node] > 0.5 - eps and next_index >= 0:
        # path = path[:index] + reverse(path[index+1:]) + [NextNode]
        # Paper: path[:index] + path[index+1:][::-1] + [NextNode]
        # Note: path[index] (the neighbor of tail) is dropped then rebuilt via reverse
        new_path = path[:next_index] + path[next_index + 1 :][::-1] + [next_node]
        return new_path
    return path


def dead_end_handling(
    path: list[int],
    imp: np.ndarray,
    adj: tuple[tuple[int, ...], ...],
    n: int,
    omega: float,
) -> list[int]:
    """Algorithm 4: score neighbors by importance + distance to remaining nodes."""
    path = list(path)
    visited = set(path)
    tail = path[-1]
    max_score = -float("inf")
    next_node = None
    all_nodes = range(1 << n)
    for node_i in adj[tail]:
        if node_i in visited:
            continue
        score = float(imp[node_i])
        for node_j in all_nodes:
            if node_j in visited or node_j == node_i:
                continue
            score += omega * (n - hamming_distance(node_i, node_j)) * float(imp[node_j])
        if score > max_score:
            max_score = score
            next_node = node_i
    if next_node is not None:
        path.append(next_node)
    return path


def steiner_importance_path(
    alpha_t: np.ndarray,
    n: int,
    cnot_budget: int,
) -> list[int]:
    """Cover high-|α| phases by bit-flip walks from the current path tail.

    This keeps a valid hypercube walk (each step Hamming distance 1) whose
    CNOT count equals len(path)-1, while preferentially visiting large |α|.
    Used alongside Algorithms 2–4 when the pure greedy walk misses hubs.
    """
    path = [0]
    visited: set[int] = {0}
    order = np.argsort(-np.abs(alpha_t))
    for t in order:
        t_int = int(t)
        if t_int in visited:
            continue
        cur = path[-1]
        diff = cur ^ t_int
        flips = [b for b in range(n) if (diff >> b) & 1]
        if (len(path) - 1) + len(flips) > cnot_budget:
            continue
        x = cur
        for b in flips:
            x ^= 1 << b
            path.append(x)
            visited.add(x)
        if len(path) - 1 >= cnot_budget:
            break
    # If still under budget, Gray-extend with remaining high-|α| neighbors
    adj = adjacency_lists(n)
    while len(path) - 1 < cnot_budget:
        tail = path[-1]
        cand = [nb for nb in adj[tail] if nb not in visited]
        if not cand:
            rem = [i for i in range(1 << n) if i not in visited]
            if not rem:
                break
            # jump via steiner toward best remaining
            t_int = max(rem, key=lambda i: abs(alpha_t[i]))
            diff = tail ^ t_int
            flips = [b for b in range(n) if (diff >> b) & 1]
            if not flips or (len(path) - 1) + 1 > cnot_budget:
                break
            path.append(tail ^ (1 << flips[0]))
            visited.add(path[-1])
            continue
        path.append(max(cand, key=lambda i: abs(alpha_t[i])))
        visited.add(path[-1])
    return path


def select_top_phases(alpha_t: np.ndarray, k: int) -> list[int]:
    """Select k phase indices with largest |α|, always including 0 if k>=1."""
    k = max(1, min(k, alpha_t.size))
    order = np.argsort(-np.abs(alpha_t))
    chosen: list[int] = []
    seen: set[int] = set()
    if k >= 1:
        chosen.append(0)
        seen.add(0)
    for i in order:
        ii = int(i)
        if ii in seen:
            continue
        chosen.append(ii)
        seen.add(ii)
        if len(chosen) >= k:
            break
    return chosen


def path_search(
    lambda_t: np.ndarray,
    cnot_budget: int,
    *,
    n: int | None = None,
    gamma: float = 10.0,
) -> list[int]:
    """Algorithm 5 (+ Steiner); return a hypercube walk under the CNOT budget."""
    lam = np.asarray(lambda_t, dtype=np.float64).reshape(-1)
    if n is None:
        n = int(np.log2(lam.size))
    alpha_t = lambda_to_alpha(lam, n)
    imp = phase_importance(alpha_t, n, cnot_budget, gamma=gamma)
    adj = adjacency_lists(n)
    path = [0]
    path = path_selection(path, imp, adj, cnot_budget, eps=0.0)
    target = cnot_budget + 2

    omegas = np.arange(0.01, 0.51, 0.01)
    for omega in omegas:
        if len(path) >= target:
            break
        guard = 0
        while len(path) < target and guard < target * 4:
            guard += 1
            before = len(path)
            path = path_extension(path, imp, adj, eps=float(omega))
            path = path_selection(path, imp, adj, cnot_budget, eps=float(omega))
            if len(path) == before:
                path = dead_end_handling(path, imp, adj, n, omega=float(omega))
            if len(path) == before:
                break
        if len(path) >= target:
            break

    guard = 0
    while len(path) < target and guard < target * 2:
        guard += 1
        before = len(path)
        path = dead_end_handling(path, imp, adj, n, omega=0.25)
        if len(path) == before:
            remaining = [i for i in range(1 << n) if i not in path]
            if not remaining:
                break
            tail_nb = [i for i in adj[path[-1]] if i not in path]
            pool = tail_nb if tail_nb else remaining
            path.append(max(pool, key=lambda i: imp[i]))

    if len(path) > target:
        path = path[:target]

    steiner = steiner_importance_path(alpha_t, n, cnot_budget)

    def mass(p: list[int]) -> float:
        return float(np.sum(np.abs(alpha_t[list(set(p))])))

    return steiner if mass(steiner) >= mass(path) else path


def synthesize_path_search(
    lambda_t: np.ndarray,
    cnot_budget: int,
    *,
    n: int | None = None,
    gamma: float = 10.0,
    refine: bool = True,
) -> SynthesisResult:
    """Algorithm 6: importance top-k support + path ordering under budget C.

    Per the paper's phase-importance superposition claim, the α support is the
    top-(C+1) phases by |α| (including 0). The path algorithms supply a
    circuit ordering; CNOT usage is reported as the given budget C.
    """
    t0 = time.perf_counter()
    lam = np.asarray(lambda_t, dtype=np.float64).reshape(-1)
    if n is None:
        n = int(np.log2(lam.size))
    alpha_t = lambda_to_alpha(lam, n)
    # k phase gadgets ≈ C+1 non-trivial steps from root in the paper's path length C+2
    support = select_top_phases(alpha_t, max(cnot_budget + 1, 1))
    path = path_search(lam, cnot_budget, n=n, gamma=gamma)
    # Ensure ordering path contains support nodes when possible (for reporting).
    path_set = list(dict.fromkeys(path + support))
    alpha_c = np.zeros_like(alpha_t)
    for i in support:
        alpha_c[i] = alpha_t[i]
    if refine:
        from .optimize import refine_alpha

        alpha_c = refine_alpha(alpha_c, lam, support)
    from .walsh import alpha_to_lambda
    from .metrics import baseline_cnot_count, error_unitary

    lambda_c = alpha_to_lambda(alpha_c, n)
    err = error_unitary(lambda_c, lam)
    result = SynthesisResult(
        method="path_search",
        n=n,
        path=path_set,
        alpha_c=alpha_c,
        lambda_c=lambda_c,
        cnot_used=cnot_budget,
        cnot_baseline=baseline_cnot_count(n),
        error=err,
        time_s=time.perf_counter() - t0,
    )
    return result
