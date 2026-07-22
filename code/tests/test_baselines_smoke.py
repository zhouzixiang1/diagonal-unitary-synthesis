"""Smoke tests for approximate baselines on tiny instances."""

from diagonal_synth.mcts import MCTSConfig, synthesize_mcts
from diagonal_synth.metrics import cnot_budget_from_reduction
from diagonal_synth.path_search import path_search, synthesize_path_search
from experiments.gen_random import random_lambda


def test_mcts_smoke_n5():
    n = 5
    lam = random_lambda(n, seed=1)
    c = cnot_budget_from_reduction(n, 0.2)
    r = synthesize_mcts(lam, c, n=n, config=MCTSConfig(refine=False, max_expand=200))
    assert r.method == "mcts"
    assert 0.0 <= r.error <= 5.0
    assert len(r.path) >= 1


def test_path_search_smoke_n5():
    n = 5
    lam = random_lambda(n, seed=2)
    c = cnot_budget_from_reduction(n, 0.25)
    path = path_search(lam, c, n=n)
    assert path[0] == 0
    assert len(path) <= c + 2
    r = synthesize_path_search(lam, c, n=n, refine=False)
    assert r.method == "path_search"
    assert 0.0 <= r.error <= 5.0
