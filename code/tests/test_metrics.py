"""Unit tests for metrics."""

import numpy as np

from diagonal_synth.metrics import (
    baseline_cnot_count,
    cnot_budget_from_reduction,
    cnot_saved_ratio,
    error_lambda,
    error_unitary,
    utility_ratio,
)


def test_identical_error_zero():
    lam = np.linspace(-1, 1, 16)
    assert error_unitary(lam, lam) < 1e-15
    assert error_lambda(lam, lam) < 1e-15


def test_utility_and_budget():
    assert baseline_cnot_count(8) == 256
    assert cnot_budget_from_reduction(8, 0.1) == int(np.floor(256 * 0.9))
    assert abs(cnot_saved_ratio(8, 230) - (256 - 230) / 256) < 1e-12
    assert utility_ratio(0.1, 0.05) == 2.0
