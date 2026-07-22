"""Monte Carlo Tree Search approximate synthesis (thesis Algorithm 1)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .exact import SynthesisResult, synthesize_from_path
from .gadgets import adjacency_lists, hamming_distance
from .metrics import baseline_cnot_count, error_lambda
from .walsh import alpha_to_lambda, lambda_to_alpha


@dataclass
class MCTSNode:
    phase: int
    father: MCTSNode | None = None
    visit: int = 0
    quality: float = 0.0
    children: dict[int, MCTSNode] = field(default_factory=dict)

    @property
    def depth(self) -> int:
        d = 0
        cur = self
        while cur.father is not None:
            d += 1
            cur = cur.father
        return d

    def path(self) -> list[int]:
        nodes: list[int] = []
        cur: MCTSNode | None = self
        while cur is not None:
            nodes.append(cur.phase)
            cur = cur.father
        return list(reversed(nodes))


@dataclass
class MCTSConfig:
    beta1: float = 1.0
    beta2: float = 1.0
    beta3: float = 1.0
    refine: bool = True
    max_expand: int = 10_000


def synthesize_mcts(
    lambda_t: np.ndarray,
    cnot_budget: int,
    *,
    n: int | None = None,
    config: MCTSConfig | None = None,
) -> SynthesisResult:
    """Search a Hamming-1 path of length roughly cnot_budget + 1 starting at 0."""
    cfg = config or MCTSConfig()
    t0 = time.perf_counter()
    lam = np.asarray(lambda_t, dtype=np.float64).reshape(-1)
    if n is None:
        n = int(np.log2(lam.size))
    alpha_t = lambda_to_alpha(lam, n)
    adj = adjacency_lists(n)
    target_len = max(1, cnot_budget + 1)  # edges ≈ cnot_budget

    root = MCTSNode(phase=0, visit=0, quality=0.0)
    # Seed: expand once from root if needed
    best_path = [0]
    best_proxy = float("inf")

    expansions = 0
    while expansions < cfg.max_expand:
        # Current frontier leaves at depth < target_len
        leaves = _collect_leaves(root, target_len)
        if not leaves:
            break

        # Choose leaf to expand / extend
        chosen = None
        best_w = -float("inf")
        for node in leaves:
            if node.depth >= target_len:
                continue
            if node.visit == 0 and node is not root:
                w = float("inf")
            else:
                w = _score(node, alpha_t, adj, n, cfg, visited=_ancestors_set(node))
            if w > best_w:
                best_w = w
                chosen = node
        if chosen is None:
            break

        # If chosen already at budget depth, stop
        if chosen.depth >= target_len:
            break

        visited = _ancestors_set(chosen)
        # Prefer unvisited Hamming-1 neighbors; else any unvisited high-|α|
        candidates = [nb for nb in adj[chosen.phase] if nb not in visited]
        if not candidates:
            # Jump: pick globally best remaining by |α|
            remaining = [i for i in range(1 << n) if i not in visited]
            if not remaining:
                break
            candidates = [max(remaining, key=lambda i: abs(alpha_t[i]))]

        # Select child among candidates
        child_phase = None
        best_cw = -float("inf")
        for nb in candidates:
            # Create temporary scoring as if expanding nb
            if nb in chosen.children:
                child = chosen.children[nb]
                if child.visit == 0:
                    cw = float("inf")
                else:
                    cw = _score(child, alpha_t, adj, n, cfg, visited=visited | {nb})
            else:
                # Unvisited expansion priority: potential term
                pot = 0.0
                for j in range(1 << n):
                    if j not in visited and j != nb:
                        pot += (n - hamming_distance(nb, j)) * abs(alpha_t[j])
                cw = cfg.beta2 * pot + cfg.beta3 * 1.0 + abs(alpha_t[nb])
            if cw > best_cw:
                best_cw = cw
                child_phase = nb
        assert child_phase is not None

        if child_phase not in chosen.children:
            child = MCTSNode(phase=child_phase, father=chosen)
            chosen.children[child_phase] = child
        else:
            child = chosen.children[child_phase]

        # Evaluate quality with proxy D_λ on sparse α along path
        path = child.path()
        alpha_c = np.zeros_like(alpha_t)
        for p in path:
            alpha_c[p] = alpha_t[p]
        proxy = error_lambda(alpha_to_lambda(alpha_c, n), lam)
        # quality: lower error is better; store as negative for average reward style
        reward = -proxy
        # Backup
        cur: MCTSNode | None = child
        while cur is not None:
            cur.visit += 1
            cur.quality += reward
            cur = cur.father

        if proxy < best_proxy:
            best_proxy = proxy
            best_path = path

        expansions += 1
        if len(best_path) - 1 >= cnot_budget and expansions >= cnot_budget:
            # Allow a few more refinements then stop
            if expansions >= cnot_budget + n:
                break

    # Ensure path length matches budget when possible by truncating/padding
    if len(best_path) - 1 > cnot_budget:
        best_path = best_path[: cnot_budget + 1]

    # Post-select: keep the highest-|α| phases discovered on the search path
    # (and fill from global top-|α| if the path is short), reflecting the
    # thesis goal of retaining important phases under a CNOT budget.
    support_k = max(cnot_budget + 1, len(set(best_path)))
    support_k = min(support_k, 1 << n)
    order = np.argsort(-np.abs(alpha_t))
    support: list[int] = []
    seen: set[int] = set()
    for p in best_path:
        if p not in seen:
            support.append(p)
            seen.add(p)
    for i in order:
        ii = int(i)
        if ii in seen:
            continue
        support.append(ii)
        seen.add(ii)
        if len(support) >= support_k:
            break

    result = synthesize_from_path(
        lam,
        support,
        n=n,
        cnot_used=cnot_budget,
        method="mcts",
        refine=cfg.refine,
    )
    result.time_s = time.perf_counter() - t0
    return result


def _ancestors_set(node: MCTSNode) -> set[int]:
    s: set[int] = set()
    cur: MCTSNode | None = node
    while cur is not None:
        s.add(cur.phase)
        cur = cur.father
    return s


def _collect_leaves(root: MCTSNode, max_depth: int) -> list[MCTSNode]:
    out: list[MCTSNode] = []

    def dfs(node: MCTSNode) -> None:
        if not node.children or node.depth >= max_depth:
            out.append(node)
            return
        for ch in node.children.values():
            dfs(ch)

    dfs(root)
    return out


def _score(
    node: MCTSNode,
    alpha_t: np.ndarray,
    adj: tuple[tuple[int, ...], ...],
    n: int,
    cfg: MCTSConfig,
    visited: set[int],
) -> float:
    if node.visit <= 0:
        return float("inf")
    q = node.quality / node.visit
    pot = 0.0
    for j in range(alpha_t.size):
        if j not in visited:
            pot += (n - hamming_distance(node.phase, j)) * abs(alpha_t[j])
    father_visit = node.father.visit if node.father is not None else 1
    explore = math.sqrt(max(father_visit, 1) / node.visit)
    return cfg.beta1 * q + cfg.beta2 * pot + cfg.beta3 * explore
