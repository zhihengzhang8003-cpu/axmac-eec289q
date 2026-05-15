"""W4 tests for axmac.accuracy_eval.

Contracts:
  1. K=0 + aca_window=None ⇒ every error metric is exactly zero (regression
     against exact_mac).
  2. Error metrics are monotone non-decreasing in K.
  3. INT truncation has positive bias on uniform inputs (always rounds toward -∞).
  4. Sample generators respect format ranges.
  5. Histogram returns matched edges/counts lengths and sums to n.
"""

from __future__ import annotations

import pytest

from axmac.accuracy_eval import (
    ErrorStats,
    error_histogram,
    error_stats_fp,
    error_stats_int,
    fp_samples,
    int_samples,
    sweep_fp,
    sweep_int,
)
from axmac.exact_mac import BF16, FP16, FP32, INT4, INT8, INT16


# ============================================================
# K=0 regression: zero error
# ============================================================

@pytest.mark.parametrize("fmt", [INT4, INT8, INT16])
def test_int_k0_zero_error(fmt):
    a = int_samples(fmt, 500, seed=1)
    b = int_samples(fmt, 500, seed=2)
    acc = [0] * 500
    stats = error_stats_int(fmt, a, b, acc, K=0, aca_window=None)
    assert stats.med == 0.0
    assert stats.rmse == 0.0
    assert stats.max_abs_err == 0.0
    assert stats.bias == 0.0


@pytest.mark.parametrize("fmt", [FP16, BF16, FP32])
def test_fp_k0_zero_error(fmt):
    a = fp_samples(fmt, 300, scale=2.0, seed=10)
    b = fp_samples(fmt, 300, scale=2.0, seed=11)
    acc = fp_samples(fmt, 300, scale=2.0, seed=12)
    stats = error_stats_fp(fmt, a, b, acc, K=0)
    assert stats.med == 0.0
    assert stats.rmse == 0.0
    assert stats.max_abs_err == 0.0


# ============================================================
# Monotonicity in K
# ============================================================

@pytest.mark.parametrize("fmt", [INT8, INT16])
def test_int_med_monotone_in_K(fmt):
    a = int_samples(fmt, 500, seed=20)
    b = int_samples(fmt, 500, seed=21)
    acc = [0] * 500
    prev = -1.0
    for K in range(0, fmt.bits + 1):
        med = error_stats_int(fmt, a, b, acc, K=K).med
        assert med >= prev - 1e-12, (K, med, prev)
        prev = med


@pytest.mark.parametrize("fmt", [FP16, BF16])
def test_fp_med_monotone_in_K(fmt):
    a = fp_samples(fmt, 300, scale=2.0, seed=30)
    b = fp_samples(fmt, 300, scale=2.0, seed=31)
    acc = fp_samples(fmt, 300, scale=2.0, seed=32)
    prev = -1.0
    for K in range(0, fmt.mant_bits + 2):
        med = error_stats_fp(fmt, a, b, acc, K=K).med
        assert med >= prev - 1e-12, (K, med, prev)
        prev = med


# ============================================================
# Truncation bias sign
# ============================================================

def test_int_truncation_bias_positive():
    """Zero-acc + K>0 ⇒ approx = (a*b) & ~mask ≤ a*b ⇒ exact - approx ≥ 0 ⇒ bias ≥ 0."""
    fmt = INT8
    a = int_samples(fmt, 1000, seed=40)
    b = int_samples(fmt, 1000, seed=41)
    acc = [0] * 1000
    for K in [1, 2, 4]:
        stats = error_stats_int(fmt, a, b, acc, K=K)
        assert stats.bias >= 0.0, (K, stats.bias)
        assert stats.max_abs_err < (1 << K), (K, stats.max_abs_err)


# ============================================================
# NMED scaling: comparable across formats
# ============================================================

def test_nmed_in_unit_range():
    """NMED for sane K should sit in [0, 1]."""
    fmt = INT8
    a = int_samples(fmt, 500, seed=50)
    b = int_samples(fmt, 500, seed=51)
    acc = [0] * 500
    for K in [0, 1, 4, 7]:
        stats = error_stats_int(fmt, a, b, acc, K=K)
        assert 0.0 <= stats.nmed <= 1.0


# ============================================================
# Sample generators
# ============================================================

@pytest.mark.parametrize("fmt", [INT4, INT8, INT16])
@pytest.mark.parametrize("dist", ["uniform", "normal", "relu"])
def test_int_samples_in_range(fmt, dist):
    samples = int_samples(fmt, 1000, distribution=dist, seed=60)
    assert len(samples) == 1000
    for x in samples:
        assert fmt.min_val <= x <= fmt.max_val


def test_int_samples_relu_non_negative():
    samples = int_samples(INT8, 500, distribution="relu", seed=70)
    assert all(x >= 0 for x in samples)


def test_int_samples_rejects_unknown_distribution():
    with pytest.raises(ValueError):
        int_samples(INT8, 10, distribution="cauchy")  # type: ignore[arg-type]


def test_int_samples_rejects_negative_n():
    with pytest.raises(ValueError):
        int_samples(INT8, -1)


@pytest.mark.parametrize("fmt", [FP16, FP32])
def test_fp_samples_returns_bits(fmt):
    samples = fp_samples(fmt, 200, scale=1.5, seed=80)
    bits_mask = (1 << fmt.total_bits) - 1
    assert all(0 <= s <= bits_mask for s in samples)


def test_fp_samples_relu_non_negative():
    samples = fp_samples(FP32, 200, distribution="relu", seed=81)
    from axmac.exact_mac import decode_fp
    for s in samples:
        x = decode_fp(s, FP32)
        assert x >= 0.0


# ============================================================
# Histogram
# ============================================================

def test_histogram_shape_and_total():
    errors = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] * 10
    edges, counts = error_histogram(errors, n_bins=6)
    assert len(edges) == 7
    assert len(counts) == 6
    assert sum(counts) == len(errors)


def test_histogram_degenerate_all_equal():
    edges, counts = error_histogram([3.0] * 20, n_bins=4)
    assert len(edges) == 5
    assert len(counts) == 4
    assert sum(counts) == 20


def test_histogram_rejects_empty():
    with pytest.raises(ValueError):
        error_histogram([], n_bins=10)


def test_histogram_rejects_bad_bins():
    with pytest.raises(ValueError):
        error_histogram([1.0, 2.0], n_bins=0)


# ============================================================
# Sweeps
# ============================================================

def test_sweep_int_returns_all_configs():
    Ks = [0, 2, 4]
    Ws = [None, 16, 4]
    result = sweep_int(INT8, Ks, Ws, n_samples=200)
    assert set(result.keys()) == {(k, w) for k in Ks for w in Ws}
    # K=0 + W=None must be the zero-error baseline.
    baseline = result[(0, None)]
    assert baseline.med == 0.0
    assert baseline.rmse == 0.0


def test_sweep_int_med_increases_with_K():
    result = sweep_int(INT8, [0, 2, 4, 6], [None], n_samples=400)
    meds = [result[(K, None)].med for K in [0, 2, 4, 6]]
    assert meds == sorted(meds)


def test_sweep_fp_returns_all_K():
    Ks = [0, 2, 4]
    result = sweep_fp(FP16, Ks, n_samples=200, scale=1.0)
    assert set(result.keys()) == set(Ks)
    assert result[0].med == 0.0


def test_sweep_fp_med_increases_with_K():
    result = sweep_fp(BF16, [0, 1, 2, 4], n_samples=400)
    meds = [result[K].med for K in [0, 1, 2, 4]]
    assert meds == sorted(meds)


# ============================================================
# Validation
# ============================================================

def test_error_stats_int_mismatched_lengths():
    with pytest.raises(ValueError):
        error_stats_int(INT8, [1, 2], [1], [0, 0])


def test_error_stats_fp_mismatched_lengths():
    with pytest.raises(ValueError):
        error_stats_fp(FP16, [1, 2], [1], [0, 0])


def test_error_stats_int_empty():
    with pytest.raises(ValueError):
        error_stats_int(INT8, [], [], [])


def test_error_stats_repr_contains_metrics():
    fmt = INT8
    a = int_samples(fmt, 50, seed=90)
    b = int_samples(fmt, 50, seed=91)
    acc = [0] * 50
    stats = error_stats_int(fmt, a, b, acc, K=2)
    r = repr(stats)
    for tag in ["n=", "med=", "rmse=", "max=", "nmed=", "bias="]:
        assert tag in r


def test_error_stats_is_frozen():
    stats = ErrorStats(1, 0.0, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises((AttributeError, TypeError)):
        stats.med = 0.5  # type: ignore[misc]
