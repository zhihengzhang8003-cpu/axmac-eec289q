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

import random

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
from axmac.exact_mac import BF16, FP8_E4M3, FP16, FP32, INT4, INT8, INT16


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


# ============================================================
# Rounding modes (Contribution A): bias compensation
# ============================================================
#
# error_stats_* / sweep_* thread the `rounding` knob through to approx_mac.
# The headline metric is `bias`: plain truncation is one-signed (a coherent
# error that accumulates ∝ N across an inference), while `round` and
# `stochastic` collapse it toward zero.

@pytest.mark.parametrize("fmt", [INT4, INT8, INT16])
@pytest.mark.parametrize("rounding", ["trunc", "round", "stochastic"])
def test_int_k0_zero_error_every_rounding(fmt, rounding):
    """K=0 ⇒ zero error for every rounding mode (regression vs exact_mac)."""
    a = int_samples(fmt, 400, seed=1)
    b = int_samples(fmt, 400, seed=2)
    acc = [0] * 400
    stats = error_stats_int(fmt, a, b, acc, K=0, rounding=rounding,
                            rng=random.Random(0))
    assert stats.med == 0.0
    assert stats.rmse == 0.0
    assert stats.bias == 0.0


def test_int_rounding_compensates_truncation_bias():
    """Headline of Contribution A: plain truncation has a large one-signed
    bias (bias == MED, because every error ≥ 0); deterministic `round` and
    `stochastic` both collapse that bias toward zero — `stochastic` most so."""
    fmt = INT8
    a = int_samples(fmt, 4000, seed=40)
    b = int_samples(fmt, 4000, seed=41)
    acc = [0] * 4000
    K = 6
    trunc = error_stats_int(fmt, a, b, acc, K=K, rounding="trunc")
    rnd = error_stats_int(fmt, a, b, acc, K=K, rounding="round")
    sto = error_stats_int(fmt, a, b, acc, K=K, rounding="stochastic",
                          rng=random.Random(99))
    # trunc: one-signed error, so the bias equals the mean abs error.
    assert trunc.bias > 0.0
    assert trunc.bias == pytest.approx(trunc.med)
    # round / stochastic cut the coherent bias several-fold.
    assert abs(rnd.bias) < trunc.bias / 3.0
    assert abs(sto.bias) < trunc.bias / 5.0
    # stochastic is genuinely zero-mean per-MAC: |bias| ≪ one LSB-field.
    assert abs(sto.bias) < 3.0


@pytest.mark.parametrize("rounding", ["round", "stochastic"])
def test_int_rounding_error_still_bounded(rounding):
    """round / stochastic trade bias for a wider error window, but the
    per-MAC error stays bounded: round by 2^(K-1), stochastic by 2^K."""
    fmt = INT8
    a = int_samples(fmt, 2000, seed=50)
    b = int_samples(fmt, 2000, seed=51)
    acc = [0] * 2000
    for K in [2, 4, 6]:
        stats = error_stats_int(fmt, a, b, acc, K=K, rounding=rounding,
                                rng=random.Random(7))
        bound = (1 << (K - 1)) if rounding == "round" else (1 << K)
        assert stats.max_abs_err <= bound, (K, rounding, stats.max_abs_err)


def test_sweep_int_accepts_rounding_and_keeps_baseline():
    """sweep_int threads `rounding` through every config; K=0 stays the
    zero-error baseline whatever the mode."""
    for mode in ("trunc", "round", "stochastic"):
        res = sweep_int(INT8, [0, 4], [None], n_samples=300, rounding=mode)
        assert res[(0, None)].med == 0.0
        assert res[(0, None)].bias == 0.0


def test_sweep_int_stochastic_is_reproducible():
    """The stochastic sweep fixes its RNG seed internally, so two runs of the
    same sweep are bit-identical (keeps the cross-config comparison paired)."""
    r1 = sweep_int(INT8, [3, 5], [None], n_samples=300, rounding="stochastic")
    r2 = sweep_int(INT8, [3, 5], [None], n_samples=300, rounding="stochastic")
    for key in r1:
        assert r1[key].bias == r2[key].bias
        assert r1[key].rmse == r2[key].rmse


@pytest.mark.parametrize("fmt", [FP16, FP8_E4M3])
@pytest.mark.parametrize("rounding", ["trunc", "round", "stochastic"])
def test_fp_k0_zero_error_every_rounding(fmt, rounding):
    """FP K=0 ⇒ zero error for every rounding mode."""
    a = fp_samples(fmt, 300, scale=2.0, seed=10)
    b = fp_samples(fmt, 300, scale=2.0, seed=11)
    acc = fp_samples(fmt, 300, scale=2.0, seed=12)
    stats = error_stats_fp(fmt, a, b, acc, K=0, rounding=rounding,
                           rng=random.Random(3))
    assert stats.med == 0.0
    assert stats.rmse == 0.0


def test_fp_rounding_modes_produce_valid_stats():
    """On the FP path the renormalisation RNE re-rounds the result, so the
    strong integer-style bias does not appear; this just checks all three
    modes run and the K>0 error is real (med > 0, within max_abs_err)."""
    fmt = FP8_E4M3
    a = fp_samples(fmt, 1000, scale=2.0, seed=20)
    b = fp_samples(fmt, 1000, scale=2.0, seed=21)
    acc = fp_samples(fmt, 1000, scale=2.0, seed=22)
    for mode in ("trunc", "round", "stochastic"):
        stats = error_stats_fp(fmt, a, b, acc, K=3, rounding=mode,
                               rng=random.Random(4))
        assert stats.med > 0.0
        assert stats.max_abs_err >= stats.med


def test_error_stats_rejects_unknown_rounding():
    fmt = INT8
    a = int_samples(fmt, 20, seed=1)
    b = int_samples(fmt, 20, seed=2)
    with pytest.raises(ValueError, match="rounding"):
        error_stats_int(fmt, a, b, [0] * 20, K=2,
                        rounding="banker")  # type: ignore[arg-type]
