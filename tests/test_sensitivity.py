"""Tests for Contribution B — layer-wise non-uniform K allocation.

Covers axmac.sensitivity (MAC counts, sensitivity probe, budget allocators)
and the per-layer-K extension of tiny_mlp_forward.

Contracts:
  1. layer_mac_counts reports in*out per linear layer.
  2. uniform_K spreads a budget evenly and sums to it exactly.
  3. allocate_K respects the budget and the K_max cap, reduces to uniform_K
     when layers are interchangeable, and steers K toward low-sensitivity /
     high-MAC layers.
  4. tiny_mlp_forward accepts a per-layer K list; an all-equal list matches
     the scalar form; a wrong-length list is rejected.
"""

from __future__ import annotations

import numpy as np
import pytest

from axmac.dnn_inference import tiny_mlp_forward
from axmac.exact_mac import INT8
from axmac.sensitivity import (
    allocate_K,
    allocate_K_for_mlp,
    layer_mac_counts,
    layer_sensitivity,
    output_divergence,
    uniform_K,
)


# ============================================================
# Helpers
# ============================================================

def _demo_mlp(seed: int = 0xB):
    """A 3-layer INT8 MLP (32 -> 16 -> 8 -> 4) on synthetic data."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 32, size=(24, 32))
    w1 = rng.integers(-16, 16, size=(32, 16))
    b1 = rng.integers(-64, 64, size=(16,))
    w2 = rng.integers(-16, 16, size=(16, 8))
    b2 = rng.integers(-64, 64, size=(8,))
    w3 = rng.integers(-16, 16, size=(8, 4))
    b3 = rng.integers(-64, 64, size=(4,))
    return x, [(w1, b1), (w2, b2), (w3, b3)]


# ============================================================
# layer_mac_counts
# ============================================================

def test_layer_mac_counts():
    _x, layers = _demo_mlp()
    assert layer_mac_counts(layers) == [32 * 16, 16 * 8, 8 * 4]


def test_layer_mac_counts_rejects_non_2d():
    bad = [(np.zeros((3, 3, 3)), None)]
    with pytest.raises(ValueError):
        layer_mac_counts(bad)


# ============================================================
# output_divergence
# ============================================================

def test_output_divergence_zero_when_identical():
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert output_divergence(y, y, metric="logit_nrmse") == pytest.approx(0.0)
    assert output_divergence(y, y, metric="argmax") == pytest.approx(0.0)


def test_output_divergence_argmax_counts_flips():
    y_ref = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([[0.0, 1.0], [0.0, 1.0]])  # first row's argmax flipped
    assert output_divergence(y, y_ref, metric="argmax") == pytest.approx(0.5)


def test_output_divergence_rejects_unknown_metric():
    y = np.zeros((2, 2))
    with pytest.raises(ValueError):
        output_divergence(y, y, metric="cosine")


# ============================================================
# uniform_K
# ============================================================

def test_uniform_K_spreads_evenly():
    assert uniform_K(4, 8) == [2, 2, 2, 2]
    assert uniform_K(4, 10) == [3, 3, 2, 2]   # remainder to the first layers
    assert uniform_K(3, 0) == [0, 0, 0]
    assert sum(uniform_K(5, 17)) == 17


def test_uniform_K_rejects_bad_input():
    with pytest.raises(ValueError):
        uniform_K(0, 4)
    with pytest.raises(ValueError):
        uniform_K(3, -1)


# ============================================================
# allocate_K
# ============================================================

def test_allocate_K_respects_budget_and_cap():
    K = allocate_K([1.0, 2.0, 3.0], [50, 50, 50], total_budget=9, K_max=6)
    assert sum(K) == 9
    assert all(0 <= k <= 6 for k in K)


def test_allocate_K_saturates_at_K_max():
    """Budget beyond n*K_max ⇒ every layer pinned at K_max."""
    K = allocate_K([1.0, 1.0, 1.0], [10, 10, 10], total_budget=100, K_max=4)
    assert K == [4, 4, 4]


def test_allocate_K_reduces_to_uniform_for_interchangeable_layers():
    """Equal sensitivity + equal MAC count ⇒ same result as uniform_K."""
    n, budget = 4, 10
    K = allocate_K([1.0] * n, [100] * n, total_budget=budget, K_max=20)
    assert K == uniform_K(n, budget)


def test_allocate_K_favours_low_sensitivity_layer():
    """At equal MAC counts, the tolerant layer absorbs the budget."""
    K = allocate_K([10.0, 1.0], [100, 100], total_budget=6, K_max=20)
    assert K[1] > K[0]


def test_allocate_K_favours_high_mac_layer():
    """At equal sensitivity, the MAC-heavy layer absorbs the budget."""
    K = allocate_K([1.0, 1.0], [1000, 10], total_budget=8, K_max=20)
    assert K[0] > K[1]


def test_allocate_K_rejects_bad_input():
    with pytest.raises(ValueError):
        allocate_K([1.0], [1, 2], total_budget=4, K_max=4)  # length mismatch
    with pytest.raises(ValueError):
        allocate_K([1.0], [1], total_budget=-1, K_max=4)
    with pytest.raises(ValueError):
        allocate_K([1.0], [1], total_budget=4, K_max=-1)


# ============================================================
# layer_sensitivity
# ============================================================

def test_layer_sensitivity_shape_and_sign():
    x, layers = _demo_mlp()
    sens = layer_sensitivity(x, layers, fmt=INT8, K_probe=4)
    assert len(sens) == len(layers)
    assert all(s >= 0.0 for s in sens)
    # Truncating *some* layer must perturb the output (not all layers inert).
    assert any(s > 0.0 for s in sens)


def test_layer_sensitivity_rejects_bad_probe():
    x, layers = _demo_mlp()
    with pytest.raises(ValueError):
        layer_sensitivity(x, layers, fmt=INT8, K_probe=0)


def test_allocate_K_for_mlp_end_to_end():
    x, layers = _demo_mlp()
    K, sens, macs = allocate_K_for_mlp(
        x, layers, fmt=INT8, total_budget=8, K_max=6
    )
    assert len(K) == len(sens) == len(macs) == len(layers)
    assert sum(K) == 8
    assert macs == layer_mac_counts(layers)


# ============================================================
# Per-layer K in tiny_mlp_forward
# ============================================================

def test_tiny_mlp_per_layer_K_list_matches_scalar():
    x, layers = _demo_mlp()
    scalar = tiny_mlp_forward(x, layers, fmt=INT8, K=2)
    as_list = tiny_mlp_forward(x, layers, fmt=INT8, K=[2, 2, 2])
    np.testing.assert_array_equal(scalar, as_list)


def test_tiny_mlp_per_layer_K_zero_list_matches_exact():
    x, layers = _demo_mlp()
    exact = tiny_mlp_forward(x, layers, fmt=INT8, K=0)
    zeros = tiny_mlp_forward(x, layers, fmt=INT8, K=[0, 0, 0])
    np.testing.assert_array_equal(exact, zeros)


def test_tiny_mlp_per_layer_K_can_differ_per_layer():
    """A non-uniform K vector is a distinct configuration from uniform K."""
    x, layers = _demo_mlp()
    uniform = tiny_mlp_forward(x, layers, fmt=INT8, K=3)
    nonuniform = tiny_mlp_forward(x, layers, fmt=INT8, K=[0, 3, 6])
    assert not np.array_equal(uniform, nonuniform)


def test_tiny_mlp_per_layer_K_wrong_length_rejected():
    x, layers = _demo_mlp()
    with pytest.raises(ValueError):
        tiny_mlp_forward(x, layers, fmt=INT8, K=[1, 2])  # 2 entries, 3 layers
