"""W5 tests for axmac.dnn_inference.

Contracts:
  1. K=0 matches the exact integer matmul/conv2d reference (numpy @).
  2. K>0 matches the scalar approx_mac_int loop on tiny tensors.
  3. Conv2d shape correctness for various stride/padding.
  4. FP scalar linear matches scalar mac_fp at K=0.
  5. Torch wrapper smoke test if torch is importable; otherwise skipped.
"""

from __future__ import annotations

import numpy as np
import pytest

from axmac.approx_mac import approx_mac_int
from axmac.dnn_inference import (
    fp_linear_approx_scalar,
    int_conv2d_approx,
    int_linear_approx,
    int_matmul_approx,
    make_approx_linear_torch,
    quantize_to_int,
    tiny_mlp_forward,
    truncate_products,
)
from axmac.exact_mac import FP16, FP32, INT4, INT8, INT16, encode_fp, mac_fp


# ============================================================
# K=0 equivalence to exact reference
# ============================================================

@pytest.mark.parametrize("fmt", [INT4, INT8, INT16])
def test_int_matmul_k0_matches_numpy(fmt):
    rng = np.random.default_rng(0xA0)
    x = rng.integers(fmt.min_val, fmt.max_val + 1, size=(8, 5))
    w = rng.integers(fmt.min_val, fmt.max_val + 1, size=(5, 7))
    got = int_matmul_approx(x, w, fmt=fmt, K=0)
    ref = x.astype(np.int64) @ w.astype(np.int64)
    # Both should fit in int32 for these small sizes.
    assert got.shape == ref.shape
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("fmt", [INT4, INT8])
def test_int_linear_k0_with_bias(fmt):
    rng = np.random.default_rng(0xA1)
    x = rng.integers(fmt.min_val, fmt.max_val + 1, size=(4, 6))
    w = rng.integers(fmt.min_val, fmt.max_val + 1, size=(6, 3))
    bias = rng.integers(-1000, 1000, size=(3,))
    got = int_linear_approx(x, w, fmt=fmt, K=0, bias=bias)
    ref = x.astype(np.int64) @ w.astype(np.int64) + bias.astype(np.int64)
    np.testing.assert_array_equal(got, ref)


# ============================================================
# K>0: vectorized matches scalar per-element loop
# ============================================================

@pytest.mark.parametrize("fmt", [INT8])
@pytest.mark.parametrize("K", [1, 2, 4])
def test_int_matmul_matches_scalar_loop(fmt, K):
    rng = np.random.default_rng(0xB0 + K)
    x = rng.integers(fmt.min_val, fmt.max_val + 1, size=(3, 5))
    w = rng.integers(fmt.min_val, fmt.max_val + 1, size=(5, 4))
    vec = int_matmul_approx(x, w, fmt=fmt, K=K)
    # Build the scalar reference: for each output entry, accumulate via
    # approx_mac_int over the reduction axis.
    ref = np.zeros((3, 4), dtype=np.int64)
    for i in range(3):
        for j in range(4):
            acc = 0
            for k in range(5):
                acc = approx_mac_int(
                    int(x[i, k]), int(w[k, j]), acc, fmt, K=K, aca_window=None
                )
            ref[i, j] = acc
    np.testing.assert_array_equal(vec, ref)


# ============================================================
# Conv2d shape + K=0 equivalence
# ============================================================

@pytest.mark.parametrize("stride,padding,expected_hw",
                         [(1, 0, (6, 6)), (1, 1, (8, 8)), (2, 1, (4, 4))])
def test_conv2d_shapes(stride, padding, expected_hw):
    rng = np.random.default_rng(0xC0)
    x = rng.integers(-8, 8, size=(2, 3, 8, 8))
    w = rng.integers(-4, 4, size=(5, 3, 3, 3))
    out = int_conv2d_approx(x, w, fmt=INT8, K=0, stride=stride, padding=padding)
    assert out.shape == (2, 5, *expected_hw)


def test_conv2d_k0_matches_naive():
    rng = np.random.default_rng(0xC1)
    x = rng.integers(-8, 8, size=(1, 2, 4, 4))
    w = rng.integers(-4, 4, size=(3, 2, 2, 2))
    got = int_conv2d_approx(x, w, fmt=INT8, K=0, stride=1, padding=0)
    # Naive nested-loop reference.
    ref = np.zeros((1, 3, 3, 3), dtype=np.int64)
    for n in range(1):
        for co in range(3):
            for oh in range(3):
                for ow in range(3):
                    s = 0
                    for ci in range(2):
                        for kh in range(2):
                            for kw in range(2):
                                s += int(x[n, ci, oh + kh, ow + kw]) * int(w[co, ci, kh, kw])
                    ref[n, co, oh, ow] = s
    np.testing.assert_array_equal(got, ref)


def test_conv2d_with_bias_and_padding():
    rng = np.random.default_rng(0xC2)
    x = rng.integers(-4, 4, size=(2, 2, 5, 5))
    w = rng.integers(-2, 2, size=(3, 2, 3, 3))
    bias = rng.integers(-50, 50, size=(3,))
    got = int_conv2d_approx(x, w, fmt=INT8, K=0, bias=bias, stride=1, padding=1)
    # Verify center pixel matches a manual computation.
    center = got[0, 0, 2, 2]
    manual = 0
    for ci in range(2):
        for kh in range(3):
            for kw in range(3):
                manual += int(x[0, ci, 1 + kh, 1 + kw]) * int(w[0, ci, kh, kw])
    manual += int(bias[0])
    assert center == manual


# ============================================================
# Validation
# ============================================================

def test_matmul_rejects_negative_K():
    x = np.zeros((2, 3), dtype=np.int64)
    w = np.zeros((3, 4), dtype=np.int64)
    with pytest.raises(ValueError):
        int_matmul_approx(x, w, fmt=INT8, K=-1)


def test_matmul_rejects_shape_mismatch():
    x = np.zeros((2, 3), dtype=np.int64)
    w = np.zeros((4, 5), dtype=np.int64)
    with pytest.raises(ValueError):
        int_matmul_approx(x, w, fmt=INT8, K=0)


def test_matmul_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        int_matmul_approx(np.zeros(3), np.zeros((3, 4)), fmt=INT8)


def test_conv2d_rejects_wrong_channels():
    x = np.zeros((1, 3, 4, 4), dtype=np.int64)
    w = np.zeros((2, 4, 3, 3), dtype=np.int64)  # C_in mismatch
    with pytest.raises(ValueError):
        int_conv2d_approx(x, w, fmt=INT8)


# ============================================================
# Quantize helper
# ============================================================

def test_quantize_to_int_clips_and_rounds():
    fmt = INT8
    x = np.array([-200.0, -127.7, -0.3, 0.0, 0.6, 127.5, 200.0])
    got = quantize_to_int(x, fmt)
    expected = np.array([-128, -128, 0, 0, 1, 127, 127])
    np.testing.assert_array_equal(got, expected)


# ============================================================
# FP scalar path
# ============================================================

def test_fp_linear_k0_matches_scalar_mac():
    fmt = FP32
    rng = np.random.default_rng(0xD0)
    xs = rng.uniform(-1.0, 1.0, size=(2, 3)).astype(np.float32)
    ws = rng.uniform(-1.0, 1.0, size=(3, 4)).astype(np.float32)
    x_bits = np.vectorize(lambda v: encode_fp(float(v), fmt))(xs).astype(np.int64)
    w_bits = np.vectorize(lambda v: encode_fp(float(v), fmt))(ws).astype(np.int64)
    got = fp_linear_approx_scalar(x_bits, w_bits, fmt=fmt, K=0)
    # Reference: accumulate mac_fp without truncation.
    zero = encode_fp(0.0, fmt)
    for i in range(2):
        for j in range(4):
            acc = zero
            for k in range(3):
                acc = mac_fp(int(x_bits[i, k]), int(w_bits[k, j]), acc, fmt)
            assert got[i, j] == acc


# ============================================================
# Tiny MLP demo
# ============================================================

def test_tiny_mlp_forward_runs():
    fmt = INT8
    rng = np.random.default_rng(0xE0)
    x = rng.integers(0, 64, size=(4, 16))  # batch=4, 16 features
    w1 = rng.integers(-8, 8, size=(16, 8))
    b1 = rng.integers(-32, 32, size=(8,))
    w2 = rng.integers(-8, 8, size=(8, 4))
    b2 = rng.integers(-32, 32, size=(4,))
    out_exact = tiny_mlp_forward(x, [(w1, b1), (w2, b2)], fmt=fmt, K=0)
    out_approx = tiny_mlp_forward(x, [(w1, b1), (w2, b2)], fmt=fmt, K=2)
    assert out_exact.shape == (4, 4)
    assert out_approx.shape == (4, 4)
    # Approximation shouldn't blow up the magnitudes — values stay in a sane range.
    assert np.max(np.abs(out_exact - out_approx)) < 1e6


# ============================================================
# PyTorch backend (skip if torch unavailable)
# ============================================================

def test_torch_backend_smoke():
    torch = pytest.importorskip("torch")
    fn = make_approx_linear_torch(INT8, K=0)
    x = torch.randn(2, 5) * 10
    w = torch.randn(5, 3) * 10
    bias = torch.randn(3) * 10
    out = fn.apply(x, w, bias)
    assert out.shape == (2, 3)


def test_torch_factory_raises_without_torch(monkeypatch):
    """If torch import fails, factory must raise ImportError with a helpful message."""
    import axmac.dnn_inference as mod
    monkeypatch.setattr(mod, "_try_import_torch", lambda: None)
    with pytest.raises(ImportError, match="PyTorch"):
        make_approx_linear_torch(INT8, K=0)


# ============================================================
# Rounding modes in the vectorized backend (Contribution A)
# ============================================================
#
# truncate_products is the vectorized counterpart of approx_mac._apply_rounding;
# int_matmul_approx / tiny_mlp_forward thread `rounding` down to it. These tests
# pin the new modes at tensor scale.

@pytest.mark.parametrize("rounding", ["trunc", "round", "stochastic"])
def test_truncate_products_k0_is_identity(rounding):
    """K=0 returns the products untouched for every rounding mode."""
    prod = np.arange(-50, 50, dtype=np.int64)
    out = truncate_products(prod, 0, rounding=rounding,
                            rng=np.random.default_rng(0))
    np.testing.assert_array_equal(out, prod)


@pytest.mark.parametrize("rounding", ["trunc", "round", "stochastic"])
@pytest.mark.parametrize("K", [1, 3, 6])
def test_truncate_products_clears_low_bits(rounding, K):
    """Every mode zeroes the low K bits of the product."""
    rng = np.random.default_rng(0x100 + K)
    prod = rng.integers(-(1 << 20), 1 << 20, size=5000, dtype=np.int64)
    out = truncate_products(prod, K, rounding=rounding, rng=rng)
    assert np.all((out & ((1 << K) - 1)) == 0)


def test_truncate_products_round_matches_formula():
    """`round` == add the 2^(K-1) correction constant, then mask — the
    vectorized form of the scalar Schulte & Swartzlander compensation."""
    rng = np.random.default_rng(0x222)
    prod = rng.integers(-(1 << 22), 1 << 22, size=5000, dtype=np.int64)
    for K in [1, 2, 4, 6]:
        out = truncate_products(prod, K, rounding="round")
        expected = (prod + (1 << (K - 1))) & ~((1 << K) - 1)
        np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("K", [2, 4, 6])
def test_truncate_products_error_bounds(K):
    """round error ≤ 2^(K-1); stochastic error < 2^K — vectorized form of the
    scalar bounds in test_approx_mac."""
    rng = np.random.default_rng(0x333 + K)
    prod = rng.integers(-(1 << 22), 1 << 22, size=20000, dtype=np.int64)
    rnd = truncate_products(prod, K, rounding="round")
    assert np.max(np.abs(prod - rnd)) <= (1 << (K - 1))
    sto = truncate_products(prod, K, rounding="stochastic",
                            rng=np.random.default_rng(1))
    assert np.max(np.abs(prod - sto)) < (1 << K)


def test_truncate_products_stochastic_is_near_zero_mean():
    """Stochastic rounding has a genuinely zero-mean error, while plain
    truncation's mean error is a large positive bias (deficiency 1)."""
    rng = np.random.default_rng(0x444)
    prod = rng.integers(-(1 << 16), 1 << 16, size=200000, dtype=np.int64)
    K = 6
    trunc_err = float((prod - truncate_products(prod, K, rounding="trunc")).mean())
    sto_err = float((prod - truncate_products(
        prod, K, rounding="stochastic", rng=np.random.default_rng(2))).mean())
    assert trunc_err > (1 << (K - 2))      # large positive bias (~2^(K-1))
    assert abs(sto_err) < 1.0              # essentially unbiased


def test_truncate_products_stochastic_rng_reproducible():
    """A seeded NumPy Generator makes stochastic rounding reproducible."""
    prod = np.arange(-9999, 10000, dtype=np.int64)
    a = truncate_products(prod, 5, rounding="stochastic",
                          rng=np.random.default_rng(2026))
    b = truncate_products(prod, 5, rounding="stochastic",
                          rng=np.random.default_rng(2026))
    np.testing.assert_array_equal(a, b)


def test_truncate_products_rejects_unknown_rounding():
    with pytest.raises(ValueError, match="rounding"):
        truncate_products(np.arange(8, dtype=np.int64), 2,
                          rounding="ceil")  # type: ignore[arg-type]


@pytest.mark.parametrize("K", [1, 2, 4])
def test_int_matmul_round_matches_scalar_loop(K):
    """With rounding="round" the vectorized matmul still matches a scalar
    approx_mac_int loop config-for-config (round is deterministic)."""
    fmt = INT8
    rng = np.random.default_rng(0xB50 + K)
    x = rng.integers(fmt.min_val, fmt.max_val + 1, size=(3, 5))
    w = rng.integers(fmt.min_val, fmt.max_val + 1, size=(5, 4))
    vec = int_matmul_approx(x, w, fmt=fmt, K=K, rounding="round")
    ref = np.zeros((3, 4), dtype=np.int64)
    for i in range(3):
        for j in range(4):
            acc = 0
            for k in range(5):
                acc = approx_mac_int(int(x[i, k]), int(w[k, j]), acc, fmt,
                                     K=K, aca_window=None, rounding="round")
            ref[i, j] = acc
    np.testing.assert_array_equal(vec, ref)


def test_int_matmul_round_reduces_output_bias():
    """Across a whole matmul the `round` mode's mean output error is far
    smaller than plain truncation's — Contribution A at layer scale."""
    fmt = INT8
    rng = np.random.default_rng(0xB60)
    x = rng.integers(0, fmt.max_val + 1, size=(40, 64))
    w = rng.integers(fmt.min_val, fmt.max_val + 1, size=(64, 32))
    exact = x.astype(np.int64) @ w.astype(np.int64)
    K = 6
    trunc = int_matmul_approx(x, w, fmt=fmt, K=K, rounding="trunc")
    rnd = int_matmul_approx(x, w, fmt=fmt, K=K, rounding="round")
    trunc_bias = float((exact - trunc).mean())
    round_bias = float((exact - rnd).mean())
    assert trunc_bias > 0.0
    assert abs(round_bias) < abs(trunc_bias) / 3.0


@pytest.mark.parametrize("rounding", ["trunc", "round", "stochastic"])
def test_tiny_mlp_forward_k0_matches_exact_every_rounding(rounding):
    """K=0 reproduces the exact MLP output regardless of rounding mode."""
    fmt = INT8
    rng = np.random.default_rng(0xE5)
    x = rng.integers(0, 48, size=(6, 20))
    layers = [
        (rng.integers(-8, 8, size=(20, 12)), rng.integers(-32, 32, size=(12,))),
        (rng.integers(-8, 8, size=(12, 5)), rng.integers(-32, 32, size=(5,))),
    ]
    exact = tiny_mlp_forward(x, layers, fmt=fmt, K=0)
    got = tiny_mlp_forward(x, layers, fmt=fmt, K=0, rounding=rounding,
                           rng=np.random.default_rng(0))
    np.testing.assert_array_equal(exact, got)


@pytest.mark.parametrize("rounding", ["round", "stochastic"])
def test_tiny_mlp_forward_runs_with_rounding(rounding):
    """round / stochastic propagate through every layer and keep the shape."""
    fmt = INT8
    rng = np.random.default_rng(0xE6)
    x = rng.integers(0, 48, size=(6, 20))
    layers = [
        (rng.integers(-8, 8, size=(20, 12)), rng.integers(-32, 32, size=(12,))),
        (rng.integers(-8, 8, size=(12, 5)), rng.integers(-32, 32, size=(5,))),
    ]
    out = tiny_mlp_forward(x, layers, fmt=fmt, K=4, rounding=rounding,
                           rng=np.random.default_rng(1))
    assert out.shape == (6, 5)
