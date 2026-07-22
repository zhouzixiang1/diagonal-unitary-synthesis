"""Unit tests for gadget graph helpers."""

from diagonal_synth.gadgets import gray_code_path, hamming_distance, neighbors, path_cnot_cost


def test_neighbors_count():
    n = 4
    for node in range(1 << n):
        nb = neighbors(node, n)
        assert len(nb) == n
        assert all(hamming_distance(node, x) == 1 for x in nb)


def test_gray_path_hamilton():
    n = 5
    path = gray_code_path(n)
    assert len(path) == 1 << n
    assert len(set(path)) == 1 << n
    assert path[0] == 0
    assert path_cnot_cost(path, n) == (1 << n) - 1
    for i in range(len(path) - 1):
        assert hamming_distance(path[i], path[i + 1]) == 1
